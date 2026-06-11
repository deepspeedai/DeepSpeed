# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Expert weight repacking for AutoEP.

Converts HuggingFace expert weight formats into TorchTitan-compatible
grouped tensors [E_local, hidden_dim, dim] for grouped GEMM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepspeed.module_inject.auto_ep_config import MoELayerSpec
from deepspeed.moe.fused_expert_layout import classify_fused_gate_up_layout


def repack_expert_weights(
    experts_source: nn.Module,
    spec: MoELayerSpec,
    ep_rank: int,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack expert weights from HF format to TorchTitan grouped format.

    Returns (w1, w2, w3) where:
        w1: [E_local, ffn_hidden_size, hidden_size]
        w2: [E_local, hidden_size, ffn_hidden_size]
        w3: [E_local, ffn_hidden_size, hidden_size]

    For fused_3d storage where expert_w3 is None (gate+up fused):
        Standard HF layout:
            Source gate_up_proj: [E, 2*ffn_hidden, hidden]
            Source down_proj: [E, hidden, ffn_hidden]

        Alternate Transformers fused layout:
            Source gate_up_proj: [E, hidden, 2*ffn_hidden]
            Source down_proj: [E, ffn_hidden, hidden]

        In both cases, the returned grouped-expert tensors are normalized to:
            w1 = gate_proj: [E_local, ffn_hidden, hidden]
            w3 = up_proj:   [E_local, ffn_hidden, hidden]
            w2 = down_proj: [E_local, hidden, ffn_hidden]
    """
    num_local_experts = spec.num_experts // ep_size
    expert_start = ep_rank * num_local_experts
    expert_end = expert_start + num_local_experts

    if spec.expert_storage == "fused_3d":
        return _repack_fused_3d(experts_source, spec, expert_start, expert_end)
    elif spec.expert_storage == "module_list":
        return _repack_module_list(experts_source, spec, expert_start, expert_end)
    else:
        raise ValueError(f"Unknown expert_storage type: {spec.expert_storage}")


def repack_expert_requires_grad_flags(
    experts_source: nn.Module,
    spec: MoELayerSpec,
    ep_rank: int,
    ep_size: int,
) -> tuple[bool, bool, bool]:
    """Return the requires_grad flags for repacked (w1, w2, w3) tensors."""
    num_local_experts = spec.num_experts // ep_size
    expert_start = ep_rank * num_local_experts
    expert_end = expert_start + num_local_experts

    if spec.expert_storage == "fused_3d":
        return _repack_fused_3d_requires_grad_flags(experts_source, spec)
    elif spec.expert_storage == "module_list":
        return _repack_module_list_requires_grad_flags(experts_source, spec, expert_start, expert_end)
    else:
        raise ValueError(f"Unknown expert_storage type: {spec.expert_storage}")


def _repack_fused_3d(
    experts_source: nn.Module,
    spec: MoELayerSpec,
    expert_start: int,
    expert_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack from fused 3D parameter tensors (transformers 5.0.0+)."""
    w1_full = getattr(experts_source, spec.expert_w1_name)
    w2_full = getattr(experts_source, spec.expert_w2_name)

    if isinstance(w1_full, nn.Parameter):
        w1_full = w1_full.data
    if isinstance(w2_full, nn.Parameter):
        w2_full = w2_full.data

    # Slice to local experts
    w1_local = w1_full[expert_start:expert_end].clone()
    w2_local = w2_full[expert_start:expert_end].clone()

    if spec.expert_w3_name is None:
        layout = classify_fused_gate_up_layout(tuple(w1_local.shape), tuple(w2_local.shape))
        if layout is None:
            raise ValueError("Unsupported fused expert weight layout for AutoEP repacking: "
                             f"{spec.expert_w1_name}={tuple(w1_local.shape)}, "
                             f"{spec.expert_w2_name}={tuple(w2_local.shape)}")

        ffn_hidden = layout.ffn_hidden_size
        if layout.layout == "gate_up_first":
            w1 = w1_local[:, :ffn_hidden, :].contiguous()  # [E_local, ffn, hidden]
            w3 = w1_local[:, ffn_hidden:, :].contiguous()  # [E_local, ffn, hidden]
            w2 = w2_local.contiguous()  # [E_local, hidden, ffn]
        else:
            w1 = w1_local[:, :, :ffn_hidden].transpose(1, 2).contiguous()  # [E_local, ffn, hidden]
            w3 = w1_local[:, :, ffn_hidden:].transpose(1, 2).contiguous()  # [E_local, ffn, hidden]
            w2 = w2_local.transpose(1, 2).contiguous()  # [E_local, hidden, ffn]
    else:
        # Separate w1 (gate), w3 (up)
        w3_full = getattr(experts_source, spec.expert_w3_name)
        if isinstance(w3_full, nn.Parameter):
            w3_full = w3_full.data
        w3_local = w3_full[expert_start:expert_end].clone()

        w1 = w1_local.contiguous()  # [E_local, ffn, hidden]
        w2 = w2_local.contiguous()  # [E_local, hidden, ffn]
        w3 = w3_local.contiguous()  # [E_local, ffn, hidden]

    return w1, w2, w3


def _repack_fused_3d_requires_grad_flags(
    experts_source: nn.Module,
    spec: MoELayerSpec,
) -> tuple[bool, bool, bool]:
    w1_param = getattr(experts_source, spec.expert_w1_name)
    w2_param = getattr(experts_source, spec.expert_w2_name)
    w1_requires_grad = _requires_grad(w1_param)
    w2_requires_grad = _requires_grad(w2_param)

    if spec.expert_w3_name is None:
        w3_requires_grad = w1_requires_grad
    else:
        w3_requires_grad = _requires_grad(getattr(experts_source, spec.expert_w3_name))

    return w1_requires_grad, w2_requires_grad, w3_requires_grad


def _repack_module_list(
    experts_source: nn.Module,
    spec: MoELayerSpec,
    expert_start: int,
    expert_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack from nn.ModuleList of individual expert modules (legacy transformers)."""
    assert isinstance(experts_source, nn.ModuleList), \
        f"Expected nn.ModuleList for module_list storage, got {type(experts_source)}"

    w1_list = []
    w2_list = []
    w3_list = []

    for expert_idx in range(expert_start, expert_end):
        expert = experts_source[expert_idx]

        # Get weight tensors - handle both nn.Linear children and direct attributes
        w1_param = _get_expert_weight(expert, spec.expert_w1_name)
        w2_param = _get_expert_weight(expert, spec.expert_w2_name)

        # nn.Linear stores weight as [out_features, in_features]
        # TorchTitan expects [ffn_hidden, hidden] for w1/w3 and [hidden, ffn_hidden] for w2
        # nn.Linear.weight is already [out, in] which matches TorchTitan's [ffn, hidden] for w1
        # No transpose needed - store as-is
        w1_list.append(w1_param.data.clone())
        w2_list.append(w2_param.data.clone())

        if spec.expert_w3_name is not None:
            w3_param = _get_expert_weight(expert, spec.expert_w3_name)
            w3_list.append(w3_param.data.clone())

    _require_consistent_dtype_device(w1_list, spec.expert_w1_name, expert_start, expert_end)
    _require_consistent_dtype_device(w2_list, spec.expert_w2_name, expert_start, expert_end)
    if spec.expert_w3_name is not None:
        _require_consistent_dtype_device(w3_list, spec.expert_w3_name, expert_start, expert_end)

    w1 = torch.stack(w1_list)  # [E_local, ffn_hidden, hidden]
    w2 = torch.stack(w2_list)  # [E_local, hidden, ffn_hidden]

    if spec.expert_w3_name is not None:
        w3 = torch.stack(w3_list)  # [E_local, ffn_hidden, hidden]
    else:
        # If no w3, this is fused gate+up - split w1
        ffn_hidden = w1.shape[1] // 2
        w3 = w1[:, ffn_hidden:, :].contiguous()
        w1 = w1[:, :ffn_hidden, :].contiguous()

    return w1, w2, w3


def _repack_module_list_requires_grad_flags(
    experts_source: nn.Module,
    spec: MoELayerSpec,
    expert_start: int,
    expert_end: int,
) -> tuple[bool, bool, bool]:
    assert isinstance(experts_source, nn.ModuleList), \
        f"Expected nn.ModuleList for module_list storage, got {type(experts_source)}"

    w1_flags = []
    w2_flags = []
    w3_flags = []
    for expert_idx in range(expert_start, expert_end):
        expert = experts_source[expert_idx]
        w1_flags.append(_get_expert_weight(expert, spec.expert_w1_name).requires_grad)
        w2_flags.append(_get_expert_weight(expert, spec.expert_w2_name).requires_grad)
        if spec.expert_w3_name is not None:
            w3_flags.append(_get_expert_weight(expert, spec.expert_w3_name).requires_grad)

    w1_requires_grad = _require_consistent_requires_grad(
        w1_flags,
        spec.expert_w1_name,
        expert_start,
        expert_end,
    )
    w2_requires_grad = _require_consistent_requires_grad(
        w2_flags,
        spec.expert_w2_name,
        expert_start,
        expert_end,
    )
    if spec.expert_w3_name is None:
        w3_requires_grad = w1_requires_grad
    else:
        w3_requires_grad = _require_consistent_requires_grad(
            w3_flags,
            spec.expert_w3_name,
            expert_start,
            expert_end,
        )

    return w1_requires_grad, w2_requires_grad, w3_requires_grad


def _get_expert_weight(expert_module: nn.Module, weight_name: str) -> torch.Tensor:
    """Get expert weight tensor by name, handling both attribute and child module patterns."""
    # Direct attribute
    param = getattr(expert_module, weight_name, None)
    if param is not None:
        if isinstance(param, nn.Linear):
            return param.weight
        if isinstance(param, (nn.Parameter, torch.Tensor)):
            return param

    # Try as child module name
    for name, child in expert_module.named_children():
        if name == weight_name:
            if isinstance(child, nn.Linear):
                return child.weight
            if hasattr(child, 'weight'):
                return child.weight

    raise ValueError(f"Could not find weight '{weight_name}' in expert module "
                     f"{type(expert_module).__name__}. Available attributes: "
                     f"{[n for n, _ in expert_module.named_parameters(recurse=False)]}")


def _requires_grad(weight: torch.Tensor | nn.Parameter) -> bool:
    return bool(getattr(weight, "requires_grad", False))


def _require_consistent_dtype_device(
    tensors: list[torch.Tensor],
    weight_name: str,
    expert_start: int,
    expert_end: int,
) -> None:
    expected_dtype = tensors[0].dtype
    expected_device = tensors[0].device
    for expert_idx, tensor in enumerate(tensors, start=expert_start):
        if tensor.dtype != expected_dtype or tensor.device != expected_device:
            raise ValueError("AutoEP cannot preserve mixed dtype/device for "
                             f"module_list experts {expert_start}:{expert_end} weight '{weight_name}' "
                             "because they are packed into one grouped expert tensor. "
                             f"Expected {expected_dtype} on {expected_device}, "
                             f"found {tensor.dtype} on {tensor.device} at expert {expert_idx}.")


def _require_consistent_requires_grad(
    flags: list[bool],
    weight_name: str,
    expert_start: int,
    expert_end: int,
) -> bool:
    if len(set(flags)) != 1:
        raise ValueError("AutoEP cannot preserve mixed requires_grad flags for "
                         f"module_list experts {expert_start}:{expert_end} weight '{weight_name}' "
                         "because they are packed into one grouped expert tensor.")
    return flags[0]
