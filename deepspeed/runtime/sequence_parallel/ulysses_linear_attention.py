# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib import metadata as importlib_metadata
import inspect
import types
from typing import Any
import weakref

from packaging import version
import torch

import deepspeed.comm as dist
from deepspeed.utils.logging import logger

MIN_FLA_CP_VERSION = "0.4.2"
FORWARD_METADATA_KWARG = "_deepspeed_ulysses_sp_metadata"
CURRENT_FORWARD_METADATA = ContextVar("deepspeed_ulysses_sp_forward_metadata", default=None)
_SUPPORTED_MODEL_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
_GATHER_HEADER_SIZE = 8
_GATHER_PROTOCOL_VERSION = 1


@dataclass
class SPForwardMetadata:
    full_position_ids: torch.LongTensor
    document_ids: torch.LongTensor
    global_cu_seqlens: torch.LongTensor
    global_cu_seqlens_cpu: torch.LongTensor
    max_seqlen: int
    cp_contexts: dict[int, Any]

    @property
    def is_packed(self) -> bool:
        if self.document_ids.shape[-1] < 2:
            return False
        return bool(torch.any(self.document_ids[:, 1:] != self.document_ids[:, :-1]).item())


@dataclass
class _FLACPOps:
    build_cp_context: Any
    causal_conv1d: Any
    chunk_gated_delta_rule: Any


@dataclass
class _ForwardPatch:
    module_ref: weakref.ReferenceType[torch.nn.Module]
    attribute: str
    had_instance_value: bool
    previous_value: Any
    previous_value_is_bound_method: bool

    def restore(self) -> None:
        module = self.module_ref()
        if module is None:
            return
        if self.had_instance_value:
            previous_value = self.previous_value
            if self.previous_value_is_bound_method:
                previous_value = types.MethodType(previous_value, module)
            setattr(module, self.attribute, previous_value)
        else:
            module.__dict__.pop(self.attribute, None)


def _callable_accepts_keyword(fn, keyword: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if keyword in parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _get_installed_fla_versions() -> dict[str, str]:
    versions = {}
    for distribution_name in ("flash-linear-attention", "fla-core"):
        try:
            versions[distribution_name] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def load_fla_cp_ops() -> _FLACPOps:
    installed_versions = _get_installed_fla_versions()
    parsed_versions = [version.parse(installed) for installed in installed_versions.values()]
    if parsed_versions and all(installed < version.parse(MIN_FLA_CP_VERSION) for installed in parsed_versions):
        found = ", ".join(f"{name}={installed}" for name, installed in installed_versions.items())
        raise ImportError(
            f"DeepSpeed linear attention CP requires flash-linear-attention/fla-core >= {MIN_FLA_CP_VERSION}; "
            f"found {found}.")

    try:
        cp_module = import_module("fla.ops.cp")
        conv_module = import_module("fla.modules.conv")
        gated_delta_module = import_module("fla.ops.gated_delta_rule")
    except ImportError as exc:
        raise ImportError("DeepSpeed Qwen3.5 linear attention CP requires FLA with fla.ops.cp, fla.modules.conv, and "
                          f"fla.ops.gated_delta_rule support (>= {MIN_FLA_CP_VERSION}).") from exc

    build_cp_context = getattr(cp_module, "build_cp_context", None)
    causal_conv1d = getattr(conv_module, "causal_conv1d", None)
    chunk_gated_delta_rule = getattr(gated_delta_module, "chunk_gated_delta_rule", None)
    missing = [
        name for name, symbol in (
            ("fla.ops.cp.build_cp_context", build_cp_context),
            ("fla.modules.conv.causal_conv1d", causal_conv1d),
            ("fla.ops.gated_delta_rule.chunk_gated_delta_rule", chunk_gated_delta_rule),
        ) if symbol is None
    ]
    if missing:
        raise ImportError(f"Installed FLA is missing required context-parallel symbols: {missing}.")
    for name, fn in (
        ("fla.modules.conv.causal_conv1d", causal_conv1d),
        ("fla.ops.gated_delta_rule.chunk_gated_delta_rule", chunk_gated_delta_rule),
    ):
        if not _callable_accepts_keyword(fn, "cp_context"):
            raise ImportError(f"Installed {name} does not accept cp_context.")
    return _FLACPOps(build_cp_context, causal_conv1d, chunk_gated_delta_rule)


def _normalize_sequence_ids(sequence_ids: torch.LongTensor, name: str) -> torch.LongTensor:
    if not isinstance(sequence_ids, torch.Tensor):
        raise RuntimeError(f"{name} must be a torch.Tensor.")
    if sequence_ids.ndim == 3 and sequence_ids.shape[0] in (3, 4):
        sequence_ids = sequence_ids[0]
    if sequence_ids.ndim != 2:
        raise RuntimeError(f"{name} must have shape [batch_size, seq_len].")
    if sequence_ids.shape[0] == 0 or sequence_ids.shape[1] == 0:
        raise RuntimeError(f"{name} must contain at least one sequence token.")
    if sequence_ids.dtype == torch.bool or torch.is_floating_point(sequence_ids) or torch.is_complex(sequence_ids):
        raise RuntimeError(f"{name} must use an integer dtype.")
    return sequence_ids.to(dtype=torch.long)


def _normalize_position_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    return _normalize_sequence_ids(position_ids, "position_ids")


def position_ids_to_document_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    """Convert batched positions to per-row document ids using every non-unit discontinuity as a boundary."""
    position_ids = _normalize_position_ids(position_ids)
    boundaries = torch.ones_like(position_ids, dtype=torch.bool)
    if position_ids.shape[1] > 1:
        boundaries[:, 1:] = position_ids[:, 1:] - position_ids[:, :-1] != 1
    return boundaries.to(dtype=torch.long).cumsum(dim=-1) - 1


def _sequence_ids_to_document_ids(sequence_ids: torch.LongTensor, name: str) -> torch.LongTensor:
    sequence_ids = _normalize_sequence_ids(sequence_ids, name)
    boundaries = torch.ones_like(sequence_ids, dtype=torch.bool)
    if sequence_ids.shape[1] > 1:
        boundaries[:, 1:] = sequence_ids[:, 1:] != sequence_ids[:, :-1]
    return boundaries.to(dtype=torch.long).cumsum(dim=-1) - 1


def document_ids_to_cu_seqlens(document_ids: torch.LongTensor) -> torch.LongTensor:
    document_ids = _normalize_sequence_ids(document_ids, "document_ids")
    if document_ids.shape[0] != 1:
        raise RuntimeError("document_ids_to_cu_seqlens requires document_ids with shape [1, seq_len].")
    flat_document_ids = document_ids[0]
    starts = torch.ones_like(flat_document_ids, dtype=torch.bool)
    if flat_document_ids.numel() > 1:
        starts[1:] = flat_document_ids[1:] != flat_document_ids[:-1]
    sequence_starts = starts.nonzero(as_tuple=False).flatten()
    sequence_end = sequence_starts.new_tensor([flat_document_ids.numel()])
    return torch.cat((sequence_starts, sequence_end)).to(dtype=torch.long)


def position_ids_to_packed_cu_seqlens(position_ids: torch.LongTensor) -> torch.LongTensor:
    position_ids = _normalize_position_ids(position_ids)
    document_ids = position_ids_to_document_ids(position_ids)
    batch_size, sequence_length = document_ids.shape
    cu_seqlens = [document_ids.new_zeros(1)]
    for batch_idx in range(batch_size):
        row_cu_seqlens = document_ids_to_cu_seqlens(document_ids[batch_idx:batch_idx + 1])
        cu_seqlens.append(row_cu_seqlens[1:] + batch_idx * sequence_length)
    return torch.cat(cu_seqlens).to(dtype=torch.long)


def _normalize_global_cu_seqlens(cu_seqlens, total_sequence_length: int, device: torch.device) -> torch.LongTensor:
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.as_tensor(cu_seqlens, device=device)
    if cu_seqlens.ndim == 2 and cu_seqlens.shape[0] == 1:
        cu_seqlens = cu_seqlens[0]
    if cu_seqlens.ndim != 1:
        raise RuntimeError("cu_seq_lens_q must be a one-dimensional global cumulative-length tensor.")
    if cu_seqlens.dtype == torch.bool or torch.is_floating_point(cu_seqlens) or torch.is_complex(cu_seqlens):
        raise RuntimeError("cu_seq_lens_q must use an integer dtype.")
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long).contiguous()
    if cu_seqlens.numel() < 2 or cu_seqlens[0].item() != 0:
        raise RuntimeError("cu_seq_lens_q must begin with 0 and contain at least one sequence.")
    if cu_seqlens[-1].item() != total_sequence_length:
        raise RuntimeError("cu_seq_lens_q must be global and end at the total sequence length "
                           f"{total_sequence_length}; got {cu_seqlens[-1].item()}.")
    if torch.any(cu_seqlens[1:] <= cu_seqlens[:-1]).item():
        raise RuntimeError("cu_seq_lens_q values must be strictly increasing.")
    return cu_seqlens


def _cu_seqlens_to_document_ids(cu_seqlens: torch.LongTensor, total_sequence_length: int) -> torch.LongTensor:
    sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    document_ids = torch.repeat_interleave(
        torch.arange(sequence_lengths.numel(), device=cu_seqlens.device, dtype=torch.long),
        sequence_lengths,
    )
    if document_ids.numel() != total_sequence_length:
        raise RuntimeError("cu_seq_lens_q does not describe the complete global sequence.")
    return document_ids.unsqueeze(0)


def _cu_seqlens_fingerprint(cu_seqlens: torch.LongTensor) -> tuple[int, int, int]:
    values = ",".join(str(value) for value in cu_seqlens.detach().cpu().tolist()).encode("ascii")
    digest = hashlib.sha256(values).digest()
    return (
        cu_seqlens.numel(),
        int.from_bytes(digest[:8], byteorder="little", signed=True),
        int.from_bytes(digest[8:16], byteorder="little", signed=True),
    )


def _require_linear_micro_batch(position_ids: torch.LongTensor) -> torch.LongTensor:
    position_ids = _normalize_position_ids(position_ids)
    if position_ids.ndim == 3 and position_ids.shape[0] in (3, 4):
        position_ids = position_ids[0]
    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise RuntimeError(
            "Qwen3.5 linear attention CP requires padding-free micro batches with position_ids shaped [1, seq_len].")
    return position_ids


def _argument_value(fn, args, kwargs, name: str):
    if name in kwargs:
        return kwargs[name]
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    return bound.arguments.get(name)


def _gated_delta_state_layout_kwargs(fn) -> dict[str, bool]:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "state_v_first" in parameters:
        return {"state_v_first": True}
    if "transpose_state_layout" in parameters:
        return {"transpose_state_layout": True}
    return {}


def _apply_attention_mask(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None or attention_mask.ndim > hidden_states.ndim:
        return hidden_states
    mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask > 0
    while mask.ndim < hidden_states.ndim:
        mask = mask.unsqueeze(-1)
    return hidden_states * mask.to(device=hidden_states.device, dtype=hidden_states.dtype)


def _call_with_accelerate_child_hook(module: torch.nn.Module, child_name: str, forward_fn):
    child_module = getattr(module, child_name)
    hook = getattr(child_module, "_hf_hook", None)
    if hook is None:
        return forward_fn()

    hook.pre_forward(child_module)
    try:
        return forward_fn()
    finally:
        hook.post_forward(child_module, ())


def _gated_delta_cp_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    metadata: SPForwardMetadata,
    fla_ops: _FLACPOps,
    cache_params=None,
    attention_mask=None,
    **kwargs,
):
    if cache_params is not None:
        raise RuntimeError("Qwen3.5 linear attention CP is training/prefill-only and does not support cache_params.")
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeError(
            f"FLA linear attention CP expects hidden_states [1, local_seq_len, hidden_size], got {hidden_states.shape}."
        )

    hidden_states = _apply_attention_mask(hidden_states, attention_mask)
    batch_size, local_seq_len, _ = hidden_states.shape
    cp_context = metadata.cp_contexts[module.conv_kernel_size]

    mixed_qkv = module.in_proj_qkv(hidden_states)
    z_gate = module.in_proj_z(hidden_states)
    beta_logits = module.in_proj_b(hidden_states)
    gate_logits = module.in_proj_a(hidden_states)

    conv_result = fla_ops.causal_conv1d(
        x=mixed_qkv,
        weight=module.conv1d.weight.squeeze(1).contiguous(),
        bias=module.conv1d.bias,
        activation=module.activation,
        cp_context=cp_context,
    )
    mixed_qkv = conv_result[0] if isinstance(conv_result, tuple) else conv_result

    key_dim = module.num_k_heads * module.head_k_dim
    value_dim = module.num_v_heads * module.head_v_dim
    expected_qkv_dim = 2 * key_dim + value_dim
    if mixed_qkv.shape[-1] != expected_qkv_dim:
        raise RuntimeError(
            f"Unexpected Qwen3.5 gated-delta projection dimension {mixed_qkv.shape[-1]}; expected {expected_qkv_dim}.")

    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch_size, local_seq_len, module.num_k_heads, module.head_k_dim)
    key = key.reshape(batch_size, local_seq_len, module.num_k_heads, module.head_k_dim)
    value = value.reshape(batch_size, local_seq_len, module.num_v_heads, module.head_v_dim)
    if module.num_v_heads % module.num_k_heads != 0:
        raise RuntimeError("Qwen3.5 num_v_heads must be divisible by num_k_heads.")
    heads_per_key = module.num_v_heads // module.num_k_heads
    if heads_per_key > 1:
        query = query.repeat_interleave(heads_per_key, dim=2)
        key = key.repeat_interleave(heads_per_key, dim=2)

    beta = beta_logits.sigmoid()
    gate = -module.A_log.float().exp() * torch.nn.functional.softplus(gate_logits.float() + module.dt_bias)
    core_attn_out, _ = fla_ops.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=gate,
        beta=beta,
        cp_context=cp_context,
        use_qk_l2norm_in_kernel=True,
        **_gated_delta_state_layout_kwargs(fla_ops.chunk_gated_delta_rule),
    )

    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z_gate = z_gate.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z_gate)
    core_attn_out = core_attn_out.reshape(batch_size, local_seq_len, value_dim)
    return module.out_proj(core_attn_out)


class Qwen35LinearAttentionCPRegistration:

    def __init__(
            self,
            model: torch.nn.Module,
            text_model: torch.nn.Module,
            decoder_layers: list[torch.nn.Module],
            linear_layers: list[torch.nn.Module],
            fla_ops: _FLACPOps,
            sp_group,
            sp_world_size: int,
            core_attn_implementation: str,
            disable_in_eval: bool,
            dense_attention_module_ids=(),
    ):
        self._model_ref = self._as_module_ref(model)
        self._text_model_ref = self._as_module_ref(text_model)
        self._decoder_layer_refs = tuple(self._as_module_ref(layer) for layer in decoder_layers)
        self._linear_layer_refs = tuple(self._as_module_ref(layer) for layer in linear_layers)
        self.dense_attention_module_ids = frozenset(dense_attention_module_ids)
        self.fla_ops = fla_ops
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.core_attn_implementation = core_attn_implementation
        self.disable_in_eval = disable_in_eval
        self._patches: list[_ForwardPatch] = []
        self._installed = False

    @staticmethod
    def _as_module_ref(module) -> weakref.ReferenceType[torch.nn.Module]:
        if isinstance(module, weakref.ReferenceType):
            return module
        return weakref.ref(module)

    @staticmethod
    def _resolve_module(module_ref, description: str) -> torch.nn.Module:
        module = module_ref()
        if module is None:
            raise RuntimeError(f"Cannot use Qwen3.5 linear-attention CP because {description} was released.")
        return module

    @property
    def model(self):
        return self._model_ref()

    @property
    def text_model(self):
        return self._text_model_ref()

    @property
    def decoder_layers(self):
        return [layer for layer_ref in self._decoder_layer_refs if (layer := layer_ref()) is not None]

    @property
    def linear_layers(self):
        return [layer for layer_ref in self._linear_layer_refs if (layer := layer_ref()) is not None]

    def owns_dense_attention_module(self, module: torch.nn.Module) -> bool:
        return id(module) in self.dense_attention_module_ids

    @staticmethod
    def _forward_patch_target(module: torch.nn.Module):
        if getattr(module, "_hf_hook", None) is not None and "_old_forward" in module.__dict__:
            return "_old_forward", module._old_forward
        return "forward", module.forward

    def _patch_forward(self, module: torch.nn.Module, attribute: str, forward_fn) -> None:
        had_instance_value = attribute in module.__dict__
        previous_value = module.__dict__.get(attribute)
        previous_value_is_bound_method = (isinstance(previous_value, types.MethodType)
                                          and previous_value.__self__ is module)
        if previous_value_is_bound_method:
            previous_value = previous_value.__func__
        self._patches.append(
            _ForwardPatch(
                module_ref=weakref.ref(module),
                attribute=attribute,
                had_instance_value=had_instance_value,
                previous_value=previous_value,
                previous_value_is_bound_method=previous_value_is_bound_method,
            ))
        setattr(module, attribute, types.MethodType(forward_fn, module))

    def _has_active_distributed_group(self) -> bool:
        if self.sp_world_size <= 1:
            return False
        try:
            return dist.is_initialized() and dist.get_world_size(group=self.sp_group) == self.sp_world_size
        except (AttributeError, RuntimeError):
            return False

    def _gather_sequence_metadata(self, position_ids, seq_idx, explicit_cu_seqlens, explicit_cu_seqlens_k):
        local_position_ids = _require_linear_micro_batch(position_ids).contiguous()
        batch_size, local_sequence_length = local_position_ids.shape
        total_sequence_length = local_sequence_length * self.sp_world_size

        local_seq_idx = None
        if seq_idx is not None:
            local_seq_idx = _normalize_sequence_ids(seq_idx, "seq_idx").to(device=local_position_ids.device)
            if local_seq_idx.shape != local_position_ids.shape:
                raise RuntimeError(
                    f"seq_idx shape {local_seq_idx.shape} must match position_ids shape {local_position_ids.shape}.")
            local_seq_idx = local_seq_idx.contiguous()

        normalized_cu_seqlens = None
        if explicit_cu_seqlens is not None:
            normalized_cu_seqlens = _normalize_global_cu_seqlens(
                explicit_cu_seqlens,
                total_sequence_length=total_sequence_length,
                device=local_position_ids.device,
            )
        normalized_cu_seqlens_k = None
        if explicit_cu_seqlens_k is not None:
            normalized_cu_seqlens_k = _normalize_global_cu_seqlens(
                explicit_cu_seqlens_k,
                total_sequence_length=total_sequence_length,
                device=local_position_ids.device,
            )
        if normalized_cu_seqlens is None:
            normalized_cu_seqlens = normalized_cu_seqlens_k
        elif normalized_cu_seqlens_k is not None and not torch.equal(normalized_cu_seqlens, normalized_cu_seqlens_k):
            raise RuntimeError("cu_seq_lens_q and cu_seq_lens_k must describe identical self-attention boundaries.")

        if self.sp_world_size == 1:
            return local_position_ids, local_seq_idx, normalized_cu_seqlens

        if not self._has_active_distributed_group():
            position_shards = [torch.empty_like(local_position_ids) for _ in range(self.sp_world_size)]
            dist.all_gather(position_shards, local_position_ids, group=self.sp_group)
            full_position_ids = torch.cat(position_shards, dim=-1)
            full_seq_idx = None
            if local_seq_idx is not None:
                seq_idx_shards = [torch.empty_like(local_seq_idx) for _ in range(self.sp_world_size)]
                dist.all_gather(seq_idx_shards, local_seq_idx, group=self.sp_group)
                full_seq_idx = torch.cat(seq_idx_shards, dim=-1)
            return full_position_ids, full_seq_idx, normalized_cu_seqlens

        cu_length, cu_digest_0, cu_digest_1 = ((0, 0, 0) if normalized_cu_seqlens is None else
                                               _cu_seqlens_fingerprint(normalized_cu_seqlens))
        header = local_position_ids.new_tensor([
            _GATHER_PROTOCOL_VERSION,
            batch_size,
            local_sequence_length,
            int(local_seq_idx is not None),
            int(normalized_cu_seqlens is not None),
            cu_length,
            cu_digest_0,
            cu_digest_1,
        ])
        seq_idx_payload = torch.zeros_like(local_position_ids) if local_seq_idx is None else local_seq_idx
        local_payload = torch.cat((header, local_position_ids.reshape(-1), seq_idx_payload.reshape(-1)))
        payload_shards = [torch.empty_like(local_payload) for _ in range(self.sp_world_size)]
        dist.all_gather(payload_shards, local_payload, group=self.sp_group)

        gathered_headers = [payload[:_GATHER_HEADER_SIZE].detach().cpu().tolist() for payload in payload_shards]
        expected_header = gathered_headers[0]
        for rank, gathered_header in enumerate(gathered_headers):
            if gathered_header[:5] != expected_header[:5]:
                raise RuntimeError(
                    "All SP ranks must provide matching position_ids shapes and the same packed metadata sources; "
                    f"rank 0 header={expected_header[:5]}, rank {rank} header={gathered_header[:5]}.")
            if gathered_header[5:] != expected_header[5:]:
                raise RuntimeError("All SP ranks must provide identical global cu_seq_lens_q metadata.")

        position_offset = _GATHER_HEADER_SIZE
        sequence_values = batch_size * local_sequence_length
        seq_idx_offset = position_offset + sequence_values
        position_shards = [
            payload[position_offset:seq_idx_offset].view(batch_size, local_sequence_length)
            for payload in payload_shards
        ]
        full_position_ids = torch.cat(position_shards, dim=-1)
        full_seq_idx = None
        if expected_header[3]:
            seq_idx_shards = [
                payload[seq_idx_offset:].view(batch_size, local_sequence_length) for payload in payload_shards
            ]
            full_seq_idx = torch.cat(seq_idx_shards, dim=-1)
        return full_position_ids, full_seq_idx, normalized_cu_seqlens

    def _build_metadata(
        self,
        position_ids,
        explicit_cu_seqlens=None,
        explicit_cu_seqlens_k=None,
        seq_idx=None,
    ) -> SPForwardMetadata:
        full_position_ids, full_seq_idx, normalized_cu_seqlens = self._gather_sequence_metadata(
            position_ids,
            seq_idx,
            explicit_cu_seqlens,
            explicit_cu_seqlens_k,
        )
        total_sequence_length = full_position_ids.numel()
        position_document_ids = position_ids_to_document_ids(full_position_ids)
        position_cu_seqlens = document_ids_to_cu_seqlens(position_document_ids)

        seq_idx_cu_seqlens = None
        if full_seq_idx is not None:
            seq_idx_document_ids = _sequence_ids_to_document_ids(full_seq_idx, "seq_idx")
            seq_idx_cu_seqlens = document_ids_to_cu_seqlens(seq_idx_document_ids)

        global_cu_seqlens = normalized_cu_seqlens
        if global_cu_seqlens is None:
            global_cu_seqlens = seq_idx_cu_seqlens
        if global_cu_seqlens is None:
            global_cu_seqlens = position_cu_seqlens

        for source_name, source_cu_seqlens in (
            ("position_ids", position_cu_seqlens),
            ("seq_idx", seq_idx_cu_seqlens),
        ):
            if source_cu_seqlens is not None and not torch.equal(source_cu_seqlens, global_cu_seqlens):
                raise RuntimeError(f"{source_name} document boundaries do not match the canonical packed metadata "
                                   f"{global_cu_seqlens.detach().cpu().tolist()}.")

        document_ids = _cu_seqlens_to_document_ids(global_cu_seqlens, total_sequence_length)
        if self.core_attn_implementation == "sdpa" and global_cu_seqlens.numel() > 2:
            raise RuntimeError(
                "Packed Ulysses SP is not supported with SDPA. Use flash_attention_2/3 or flex_attention.")

        global_cu_seqlens_cpu = global_cu_seqlens.detach().cpu()
        max_seqlen = int((global_cu_seqlens_cpu[1:] - global_cu_seqlens_cpu[:-1]).max().item())
        linear_layers = [
            self._resolve_module(layer_ref, "a registered linear-attention layer")
            for layer_ref in self._linear_layer_refs
        ]
        cp_contexts = {
            kernel_size:
            self.fla_ops.build_cp_context(
                cu_seqlens=global_cu_seqlens,
                cu_seqlens_cpu=global_cu_seqlens_cpu,
                group=self.sp_group,
                conv1d_kernel_size=kernel_size,
            )
            for kernel_size in {layer.conv_kernel_size
                                for layer in linear_layers}
        }
        return SPForwardMetadata(
            full_position_ids=full_position_ids,
            document_ids=document_ids,
            global_cu_seqlens=global_cu_seqlens,
            global_cu_seqlens_cpu=global_cu_seqlens_cpu,
            max_seqlen=max_seqlen,
            cp_contexts=cp_contexts,
        )

    def install(self) -> None:
        if self._installed:
            return

        text_model = self._resolve_module(self._text_model_ref, "the registered text model")
        decoder_layers = [
            self._resolve_module(layer_ref, "a registered decoder layer") for layer_ref in self._decoder_layer_refs
        ]
        linear_layers = [
            self._resolve_module(layer_ref, "a registered linear-attention layer")
            for layer_ref in self._linear_layer_refs
        ]
        text_forward_attribute, original_text_forward = self._forward_patch_target(text_model)
        registration = self

        def text_forward(module, *args, **kwargs):
            if registration.disable_in_eval and not module.training:
                return original_text_forward(*args, **kwargs)
            position_ids = _argument_value(original_text_forward, args, kwargs, "position_ids")
            if position_ids is None:
                raise RuntimeError(
                    "Qwen3.5 Ulysses SP requires position_ids in the input batch before sequence sharding.")
            metadata = registration._build_metadata(
                position_ids,
                explicit_cu_seqlens=kwargs.get("cu_seq_lens_q"),
                explicit_cu_seqlens_k=kwargs.get("cu_seq_lens_k"),
                seq_idx=kwargs.get("seq_idx"),
            )
            for max_length_key in ("max_length_q", "max_length_k", "max_seqlen_q", "max_seqlen_k"):
                max_length = kwargs.get(max_length_key)
                if max_length is None:
                    continue
                if torch.is_tensor(max_length):
                    if max_length.numel() != 1:
                        raise RuntimeError(f"{max_length_key} must be a scalar.")
                    max_length = int(max_length.item())
                if int(max_length) != metadata.max_seqlen:
                    raise RuntimeError(f"{max_length_key}={max_length} does not match canonical max sequence length "
                                       f"{metadata.max_seqlen}.")
            call_kwargs = dict(kwargs)
            call_kwargs[FORWARD_METADATA_KWARG] = metadata
            token = CURRENT_FORWARD_METADATA.set(metadata)
            try:
                return original_text_forward(*args, **call_kwargs)
            finally:
                CURRENT_FORWARD_METADATA.reset(token)

        self._patch_forward(text_model, text_forward_attribute, text_forward)

        for decoder_layer in decoder_layers:
            decoder_forward_attribute, original_decoder_forward = self._forward_patch_target(decoder_layer)
            is_linear = getattr(decoder_layer, "block_type", None) == "linear_attention"

            def make_decoder_forward(original_forward, linear_layer):

                def decoder_forward(module, *args, **kwargs):
                    metadata = kwargs.pop(FORWARD_METADATA_KWARG, None)
                    if metadata is not None and linear_layer:
                        kwargs[FORWARD_METADATA_KWARG] = metadata
                    token = CURRENT_FORWARD_METADATA.set(metadata) if metadata is not None else None
                    try:
                        return original_forward(*args, **kwargs)
                    finally:
                        if token is not None:
                            CURRENT_FORWARD_METADATA.reset(token)

                return decoder_forward

            self._patch_forward(
                decoder_layer,
                decoder_forward_attribute,
                make_decoder_forward(original_decoder_forward, is_linear),
            )

        for linear_layer in linear_layers:
            linear_forward_attribute, original_linear_forward = self._forward_patch_target(linear_layer)

            def make_linear_forward(original_forward):

                def linear_forward(module, hidden_states, cache_params=None, attention_mask=None, *args, **kwargs):
                    metadata = kwargs.pop(FORWARD_METADATA_KWARG, None) or CURRENT_FORWARD_METADATA.get()
                    if metadata is None or (registration.disable_in_eval and not module.training):
                        return original_forward(
                            hidden_states,
                            *args,
                            cache_params=cache_params,
                            attention_mask=attention_mask,
                            **kwargs,
                        )
                    return _call_with_accelerate_child_hook(
                        module,
                        "conv1d",
                        lambda: _gated_delta_cp_forward(
                            module,
                            hidden_states,
                            metadata,
                            registration.fla_ops,
                            cache_params=cache_params,
                            attention_mask=attention_mask,
                            **kwargs,
                        ),
                    )

                return linear_forward

            self._patch_forward(
                linear_layer,
                linear_forward_attribute,
                make_linear_forward(original_linear_forward),
            )

        self._installed = True
        logger.info(f"[ulysses_sp] installed Qwen3.5 linear-attention CP on {len(linear_layers)} model instances")

    def restore(self) -> None:
        for patch in reversed(self._patches):
            patch.restore()
        self._patches.clear()
        self._installed = False


def _config_uses_linear_attention(hf_model_config, arch_cfg) -> bool:
    for config in (hf_model_config, arch_cfg):
        layer_types = getattr(config, "layer_types", None) or ()
        if any(str(layer_type).lower() == "linear_attention" for layer_type in layer_types):
            return True
    return False


def _is_multimodal_config(hf_model_config, arch_cfg) -> bool:
    return (hf_model_config is not arch_cfg and getattr(hf_model_config, "vision_config", None) is not None
            and getattr(hf_model_config, "text_config", None) is not None)


def prepare_qwen35_linear_attention_cp(
    model,
    hf_model_config,
    arch_cfg,
    core_attn_implementation: str,
    disable_in_eval: bool,
    micro_batch_size=None,
):
    if not _config_uses_linear_attention(hf_model_config, arch_cfg):
        return None

    model_types = {str(getattr(config, "model_type", "") or "").lower() for config in (hf_model_config, arch_cfg)}
    if not model_types.intersection(_SUPPORTED_MODEL_TYPES):
        raise RuntimeError(
            f"Ulysses SP found linear_attention layers for model types {sorted(model_types)}, but only Qwen3.5 and "
            "Qwen3.5-MoE have validated FLA context-parallel adapters.")
    if _is_multimodal_config(hf_model_config, arch_cfg):
        raise RuntimeError(
            "Qwen3.5 multimodal Ulysses SP is not supported because visual-token sequence partitioning is not "
            "implemented. Load the text-only Qwen3.5 causal-LM model/config instead.")
    if micro_batch_size is not None and micro_batch_size != 1:
        raise RuntimeError(
            "Qwen3.5 linear attention CP currently requires micro_batch_size=1 so each rank contributes one "
            "padding-free packed token stream.")
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            "Qwen3.5 linear attention CP requires an instantiated model so DeepSpeed can install model-scoped, "
            "reversible adapters. Pass the model object rather than a model name/path.")

    candidates = []
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and any(hasattr(layer, "linear_attn") for layer in layers):
            candidates.append(module)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Qwen3.5 text backbone with linear attention layers, found {len(candidates)}.")

    text_model = candidates[0]
    decoder_layers = list(text_model.layers)
    linear_layers = [layer.linear_attn for layer in decoder_layers if hasattr(layer, "linear_attn")]
    dense_attention_module_ids = frozenset(
        id(layer.self_attn) for layer in decoder_layers if hasattr(layer, "self_attn"))
    required_attrs = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d",
        "activation",
        "num_v_heads",
        "num_k_heads",
        "head_k_dim",
        "head_v_dim",
        "conv_kernel_size",
        "A_log",
        "dt_bias",
        "norm",
        "out_proj",
    )
    for layer in linear_layers:
        missing = [attribute for attribute in required_attrs if not hasattr(layer, attribute)]
        if missing:
            raise RuntimeError(
                f"Unsupported {type(layer).__module__}.{type(layer).__name__}; missing attributes {missing}.")

    fla_ops = load_fla_cp_ops()
    return {
        "model": weakref.ref(model),
        "text_model": weakref.ref(text_model),
        "decoder_layers": tuple(weakref.ref(layer) for layer in decoder_layers),
        "linear_layers": tuple(weakref.ref(layer) for layer in linear_layers),
        "dense_attention_module_ids": dense_attention_module_ids,
        "fla_ops": fla_ops,
        "core_attn_implementation": core_attn_implementation,
        "disable_in_eval": disable_in_eval,
    }


def create_qwen35_linear_attention_registration(prepared, sp_group, sp_world_size):
    if prepared is None:
        return None
    return Qwen35LinearAttentionCPRegistration(
        sp_group=sp_group,
        sp_world_size=sp_world_size,
        **prepared,
    )
