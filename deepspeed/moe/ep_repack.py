# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Expert weight repacking for AutoEP.

Converts HuggingFace expert weight formats into TorchTitan-compatible
grouped tensors [E_local, hidden_dim, dim] for grouped GEMM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepspeed.module_inject.auto_ep_config import MoEModelPreset


def repack_expert_weights(
    moe_layer: nn.Module,
    preset: MoEModelPreset,
    ep_rank: int,
    ep_size: int,
) -> dict | None:
    """Repack expert weights from a HuggingFace MoE layer into grouped format.

    Args:
        moe_layer:  The original MoE sub-layer (the one being replaced).
                    The expert collection is accessed via ``preset.experts_attr``.
        preset:     Model preset that describes the weight layout.
                    If None, returns None (caller skips weight copy).
        ep_rank:    This rank's index in the EP group.
        ep_size:    Expert-parallel world size.

    Returns:
        dict with keys ``"w1"``, ``"w2"``, ``"w3"`` where each tensor has
        shape ``[E_local, ffn_hidden, hidden]`` / ``[E_local, hidden, ffn_hidden]``,
        or None when preset is None (no-op, experts keep their random init).

    Weight conventions (TorchTitan / GroupedExperts):
        w1: gate projection  [E_local, ffn_hidden, hidden]
        w2: down projection  [E_local, hidden, ffn_hidden]
        w3: up   projection  [E_local, ffn_hidden, hidden]
    """
    if preset is None:
        # No structural information — caller must handle weight init separately.
        return None

    experts_module = getattr(moe_layer, preset.experts_attr)
    num_experts = getattr(moe_layer, preset.num_experts_attr)
    num_local_experts = num_experts // ep_size
    expert_start = ep_rank * num_local_experts
    expert_end = expert_start + num_local_experts

    if preset.expert_storage == "fused_3d":
        w1, w2, w3 = _repack_fused_3d(experts_module, expert_start, expert_end)
    elif preset.expert_storage == "module_list":
        w1, w2, w3 = _repack_module_list(experts_module, expert_start, expert_end)
    else:
        raise ValueError(f"Unknown expert_storage format: {preset.expert_storage!r}")

    return {"w1": w1, "w2": w2, "w3": w3}


def _repack_fused_3d(
    experts_module: nn.Module,
    expert_start: int,
    expert_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack from fused 3D parameter tensors (transformers 5.0.0+).

    Expected layout on ``experts_module``:
        gate_up_proj: [E, 2*ffn_hidden, hidden]   (gate + up fused)
        down_proj:    [E, hidden, ffn_hidden]
    """
    gate_up_full = getattr(experts_module, "gate_up_proj")
    down_full = getattr(experts_module, "down_proj")

    if isinstance(gate_up_full, nn.Parameter):
        gate_up_full = gate_up_full.data
    if isinstance(down_full, nn.Parameter):
        down_full = down_full.data

    gate_up_local = gate_up_full[expert_start:expert_end].clone()  # [E_local, 2*ffn, hidden]
    down_local = down_full[expert_start:expert_end].clone()  # [E_local, hidden, ffn]

    ffn_hidden = gate_up_local.shape[1] // 2
    w1 = gate_up_local[:, :ffn_hidden, :].contiguous()  # gate_proj [E_local, ffn, hidden]
    w3 = gate_up_local[:, ffn_hidden:, :].contiguous()  # up_proj   [E_local, ffn, hidden]
    w2 = down_local.contiguous()  # down_proj [E_local, hidden, ffn]

    return w1, w2, w3


def _repack_module_list(
    experts_module: nn.ModuleList,
    expert_start: int,
    expert_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack from nn.ModuleList of individual expert modules (legacy transformers).

    Probes common attribute names for each weight:
        gate projection (w1): gate_proj, w1, fc1
        down projection (w2): down_proj, w2, fc2
        up   projection (w3): up_proj,   w3  (optional — fused in some models)
    """
    assert isinstance(experts_module, nn.ModuleList), \
        f"Expected nn.ModuleList for module_list storage, got {type(experts_module)}"

    _W1_NAMES = ("gate_proj", "w1", "fc1")
    _W2_NAMES = ("down_proj", "w2", "fc2")
    _W3_NAMES = ("up_proj", "w3")

    w1_list, w2_list, w3_list = [], [], []

    for expert_idx in range(expert_start, expert_end):
        expert = experts_module[expert_idx]

        w1_param = _get_expert_weight(expert, _W1_NAMES)
        w2_param = _get_expert_weight(expert, _W2_NAMES)
        w3_param = _get_expert_weight(expert, _W3_NAMES, required=False)

        # nn.Linear.weight is [out_features, in_features] = [ffn_hidden, hidden] for w1/w3
        # which already matches the [E, ffn_hidden, hidden] convention — no transpose needed.
        w1_list.append(w1_param.data.clone())
        w2_list.append(w2_param.data.clone())
        if w3_param is not None:
            w3_list.append(w3_param.data.clone())

    w1 = torch.stack(w1_list)  # [E_local, ffn_hidden, hidden]
    w2 = torch.stack(w2_list)  # [E_local, hidden, ffn_hidden]

    if w3_list:
        w3 = torch.stack(w3_list)  # [E_local, ffn_hidden, hidden]
    else:
        # gate+up fused into w1: split evenly
        ffn_hidden = w1.shape[1] // 2
        w3 = w1[:, ffn_hidden:, :].contiguous()
        w1 = w1[:, :ffn_hidden, :].contiguous()

    return w1, w2, w3


def _get_expert_weight(
    expert_module: nn.Module,
    weight_names: tuple,
    required: bool = True,
) -> torch.Tensor | None:
    """Get an expert weight tensor by probing a list of candidate attribute names.

    Args:
        expert_module: The individual expert sub-module.
        weight_names:  Candidate attribute names to try in order.
        required:      If True, raise ValueError when none found.
                       If False, return None when none found.
    """
    for name in weight_names:
        # Direct attribute (nn.Parameter or Tensor)
        param = getattr(expert_module, name, None)
        if param is not None:
            if isinstance(param, nn.Linear):
                return param.weight
            if isinstance(param, (nn.Parameter, torch.Tensor)):
                return param

        # Child module with that name
        child = dict(expert_module.named_children()).get(name)
        if child is not None:
            if isinstance(child, nn.Linear):
                return child.weight
            if hasattr(child, "weight"):
                return child.weight

    if required:
        available = [n for n, _ in expert_module.named_parameters(recurse=False)]
        raise ValueError(f"Could not find any of {weight_names} in expert module "
                         f"{type(expert_module).__name__}. Available parameters: {available}")
    return None
