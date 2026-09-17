# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Route-full / partition-dispatch helpers for AutoEP + AutoTP folding."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import deepspeed.comm as dist

_FOLDING_DIGEST_MOD_A = 2147483647
_FOLDING_DIGEST_MOD_B = 2147483629


@dataclass
class RoutedAssignmentPayload:
    token_indices: torch.Tensor
    expert_indices: torch.Tensor
    assignment_indices: torch.Tensor
    capacity_slots: torch.Tensor
    combine_weights: torch.Tensor
    drop_mask: torch.Tensor
    pad_mask: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]
    extra: dict[str, torch.Tensor]


@dataclass
class RestoreContext:
    original_payload: RoutedAssignmentPayload
    local_indices: torch.Tensor
    tp_rank: int
    tp_size: int
    num_tokens: int
    counters: dict[str, int]


def assignment_ordinals_by_expert(expert_indices: torch.Tensor) -> torch.Tensor:
    """Return stable ordinals within each contiguous expert segment."""
    if expert_indices.numel() == 0:
        return expert_indices.to(torch.long)
    positions = torch.arange(expert_indices.numel(), device=expert_indices.device, dtype=torch.long)
    starts = torch.zeros_like(positions)
    starts[0] = 0
    segment_start = torch.zeros(expert_indices.numel(), device=expert_indices.device, dtype=torch.bool)
    segment_start[0] = True
    segment_start[1:] = expert_indices[1:] != expert_indices[:-1]
    starts = torch.where(segment_start, positions, starts)
    starts = torch.cummax(starts, dim=0).values
    return positions - starts


def _take(payload: RoutedAssignmentPayload, indices: torch.Tensor) -> RoutedAssignmentPayload:
    extra = {
        key:
        value.index_select(0, indices)
        if torch.is_tensor(value) and value.shape[:1] == payload.token_indices.shape[:1] else value
        for key, value in payload.extra.items()
    }
    return RoutedAssignmentPayload(
        token_indices=payload.token_indices.index_select(0, indices),
        expert_indices=payload.expert_indices.index_select(0, indices),
        assignment_indices=payload.assignment_indices.index_select(0, indices),
        capacity_slots=payload.capacity_slots.index_select(0, indices),
        combine_weights=payload.combine_weights.index_select(0, indices),
        drop_mask=payload.drop_mask.index_select(0, indices),
        pad_mask=payload.pad_mask.index_select(0, indices),
        input_splits=list(payload.input_splits),
        output_splits=list(payload.output_splits),
        extra=extra,
    )


def _recompute_input_splits(payload: RoutedAssignmentPayload) -> list[int]:
    destinations = payload.extra.get("destination_ranks")
    if destinations is None:
        return list(payload.input_splits)
    if len(payload.input_splits) == 0:
        return []
    counts = torch.bincount(destinations.to(torch.long), minlength=len(payload.input_splits))
    return [int(value) for value in counts[:len(payload.input_splits)].cpu().tolist()]


def _tensor_digest_words(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.is_floating_point():
        words = torch.nan_to_num(tensor.float(), nan=0.0, posinf=3.4028235e38,
                                 neginf=-3.4028235e38).mul(1000003.0).round().to(torch.long)
    else:
        words = tensor.to(torch.long)
    return words.reshape(-1)


def _digest_words(words: torch.Tensor, *, salt: int, modulus: int) -> torch.Tensor:
    if words.numel() == 0:
        return torch.tensor(salt, device=words.device, dtype=torch.long)
    positions = torch.arange(1, words.numel() + 1, device=words.device, dtype=torch.long)
    positions = positions.add_(salt).remainder_(modulus)
    values = words.remainder(modulus)
    return (values.mul(positions).remainder_(modulus).sum().add_(words.numel() * salt).remainder_(modulus))


def _payload_digest(payload: RoutedAssignmentPayload) -> torch.Tensor:
    device = payload.token_indices.device
    active = (~payload.drop_mask & ~payload.pad_mask).to(torch.long)
    digest = torch.tensor(
        [payload.token_indices.numel(),
         int(sum(payload.input_splits)),
         int(sum(payload.output_splits)), 0, 0],
        device=device,
        dtype=torch.long)
    fields = (
        payload.token_indices,
        payload.expert_indices,
        payload.assignment_indices,
        payload.capacity_slots,
        payload.combine_weights,
        payload.drop_mask,
        payload.pad_mask,
        active,
        payload.extra.get("destination_ranks", torch.empty(0, device=device, dtype=torch.long)),
    )
    for index, field in enumerate(fields, start=1):
        if not torch.is_tensor(field):
            continue
        words = _tensor_digest_words(field)
        digest[3] = digest[3].add(_digest_words(words, salt=17 * index,
                                                modulus=_FOLDING_DIGEST_MOD_A)).remainder_(_FOLDING_DIGEST_MOD_A)
        digest[4] = digest[4].add(_digest_words(words, salt=31 * index,
                                                modulus=_FOLDING_DIGEST_MOD_B)).remainder_(_FOLDING_DIGEST_MOD_B)
    return digest


def _payload_digest_components(payload: RoutedAssignmentPayload) -> dict[str, torch.Tensor]:
    device = payload.token_indices.device
    active = (~payload.drop_mask & ~payload.pad_mask).to(torch.long)
    fields = {
        "token_indices": payload.token_indices,
        "expert_indices": payload.expert_indices,
        "assignment_indices": payload.assignment_indices,
        "capacity_slots": payload.capacity_slots,
        "combine_weights": payload.combine_weights,
        "drop_mask": payload.drop_mask,
        "pad_mask": payload.pad_mask,
        "active": active,
        "destination_ranks": payload.extra.get("destination_ranks", torch.empty(0, device=device, dtype=torch.long)),
    }
    components: dict[str, torch.Tensor] = {}
    for index, (name, field) in enumerate(fields.items(), start=1):
        if not torch.is_tensor(field):
            continue
        words = _tensor_digest_words(field)
        components[name] = torch.stack((
            torch.tensor(words.numel(), device=device, dtype=torch.long),
            _digest_words(words, salt=17 * index, modulus=_FOLDING_DIGEST_MOD_A),
            _digest_words(words, salt=31 * index, modulus=_FOLDING_DIGEST_MOD_B),
        ))
    return components


def _format_payload_debug(payload: RoutedAssignmentPayload, *, digest: torch.Tensor, max_digest: torch.Tensor,
                          min_digest: torch.Tensor, tp_group) -> str:
    if os.environ.get("AUTOEP_FOLDING_DEBUG_PAYLOAD", "0") not in {"1", "true", "TRUE", "yes"}:
        return ""

    differing_fields = []
    for name, component in _payload_digest_components(payload).items():
        component_max = component.clone()
        component_min = component.clone()
        dist.all_reduce(component_max, op=dist.ReduceOp.MAX, group=tp_group)
        dist.all_reduce(component_min, op=dist.ReduceOp.MIN, group=tp_group)
        if not torch.equal(component_max, component_min):
            differing_fields.append({
                "field": name,
                "local": [int(value) for value in component.detach().cpu().tolist()],
                "min": [int(value) for value in component_min.detach().cpu().tolist()],
                "max": [int(value) for value in component_max.detach().cpu().tolist()],
            })

    sample_limit = int(os.environ.get("AUTOEP_FOLDING_DEBUG_SAMPLE_LIMIT", "12"))
    samples = {
        "token_indices": payload.token_indices[:sample_limit].detach().cpu().tolist(),
        "expert_indices": payload.expert_indices[:sample_limit].detach().cpu().tolist(),
        "assignment_indices": payload.assignment_indices[:sample_limit].detach().cpu().tolist(),
        "capacity_slots": payload.capacity_slots[:sample_limit].detach().cpu().tolist(),
        "combine_weights": payload.combine_weights[:sample_limit].detach().float().cpu().tolist(),
    }
    try:
        tp_group_ranks = dist.get_all_ranks_from_group(tp_group)
    except Exception:
        tp_group_ranks = []
    details = {
        "rank": dist.get_rank(),
        "tp_rank": dist.get_rank(group=tp_group),
        "tp_group_ranks": tp_group_ranks,
        "digest": [int(value) for value in digest.detach().cpu().tolist()],
        "digest_min": [int(value) for value in min_digest.detach().cpu().tolist()],
        "digest_max": [int(value) for value in max_digest.detach().cpu().tolist()],
        "differing_fields": differing_fields,
        "samples": samples,
    }
    return f" Debug details: {details}"


def assert_tp_payload_consistent(payload: RoutedAssignmentPayload, *, tp_group, tp_size: int) -> None:
    if tp_size <= 1 or not dist.is_initialized():
        return

    digest = _payload_digest(payload)
    max_digest = digest.clone()
    min_digest = digest.clone()
    dist.all_reduce(max_digest, op=dist.ReduceOp.MAX, group=tp_group)
    dist.all_reduce(min_digest, op=dist.ReduceOp.MIN, group=tp_group)
    if not torch.equal(max_digest, min_digest):
        debug_details = _format_payload_debug(payload,
                                              digest=digest,
                                              max_digest=max_digest,
                                              min_digest=min_digest,
                                              tp_group=tp_group)
        raise RuntimeError("AutoEP+AutoTP routing decisions differ across tensor-parallel lanes. "
                           "Folded dispatch requires identical routed-token payloads before TP partitioning."
                           f"{debug_details}")


def partition_assignments(
    payload: RoutedAssignmentPayload,
    *,
    tp_group,
    tp_rank: int,
    tp_size: int,
) -> tuple[RoutedAssignmentPayload, RestoreContext]:
    """Partition routed assignments across TP peers by stable per-expert ordinal.

    Each peer keeps only ``assignment_index % tp_size == tp_rank`` of the
    (token, expert) assignments and drops the rest *before* the EP dispatch
    all-to-all, so the dispatch carries the full token set exactly once (split
    across peers) instead of ``tp_size`` redundant copies. The dropped work is
    reconstructed afterwards by ``restore_combined``'s all-gather; that
    reconstruction is what makes the folded router/gate gradient replicated
    (AVERAGE) rather than a true SUM partial -- see ``_AllGatherVariableRows``.
    """
    active = ~payload.drop_mask & ~payload.pad_mask
    if tp_size <= 1:
        keep = active
    else:
        keep = (payload.assignment_indices.remainder(tp_size) == tp_rank) & active
    local_indices = torch.nonzero(keep, as_tuple=False).flatten()

    local = _take(payload, local_indices)
    local.input_splits = _recompute_input_splits(local)
    local.output_splits = list(local.input_splits)
    ctx = RestoreContext(
        original_payload=payload,
        local_indices=local_indices,
        tp_rank=tp_rank,
        tp_size=tp_size,
        num_tokens=int(payload.extra.get("num_tokens", torch.tensor(0)).item()) if torch.is_tensor(
            payload.extra.get("num_tokens")) else int(payload.extra.get("num_tokens", 0)),
        counters={
            "assignments_total": int((~payload.drop_mask & ~payload.pad_mask).sum().item()),
            "assignments_local": int(local_indices.numel()),
            "padded": int(payload.pad_mask.sum().item()),
            "dropped": int(payload.drop_mask.sum().item()),
            "split_sum_in": int(sum(local.input_splits)),
            "split_sum_out": int(sum(local.output_splits)),
        },
    )
    return local, ctx


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    if tensor.shape[0] == rows:
        return tensor
    pad_shape = (rows - tensor.shape[0], *tensor.shape[1:])
    return torch.cat((tensor, tensor.new_zeros(pad_shape)), dim=0)


class _AllGatherVariableRows(torch.autograd.Function):
    """Differentiable all-gather of row-variable tensors across the TP folding group.

    Forward concatenates every TP peer's local rows into one tensor that is
    identical on every peer: a replicated full view of the rows that
    ``partition_assignments`` had split across peers before the EP dispatch.

    Backward is the matching reduce-scatter. Because the forward output is
    consumed identically on every peer, each peer holds the same ``grad_output``;
    summing those replicas with ``all_reduce`` and keeping this peer's own
    row-slice is the correct vector-Jacobian product.

    Gradient-reduction consequence (important -- this is why the folded
    router/gate uses AVERAGE, not SUM): the ``all_reduce`` in backward scales
    each peer's slice gradient by ``tp_size``. A parameter whose gradient flows
    through this restore all-gather -- the folded router/gate scores, see
    ``restore_combined`` -- therefore reaches the optimizer's TP reducer carrying
    ``tp_size`` times its own routed-token slice. The TP reducer all_reduce then
    produces ``tp_size * full_grad``, and the AVERAGE strategy in
    ``auto_ep_folding.autoep_folding_gradient_reduction_strategy`` divides by
    ``tp_size`` to recover the true gradient. Reducing with SUM instead leaves
    the uncancelled ``tp_size`` factor -- exactly the 2.0x router/gate gradient
    regression the CPU/Gloo parity tests guard against. The partition is
    reconstructed into a replicated full view here, so it is not a genuine SUM
    partial; a future true-SP path that kept the shard to the loss would be.
    """

    @staticmethod
    def forward(ctx, tensor, group, counts, max_rows):
        ctx.group = group
        ctx.counts = tuple(counts)
        ctx.max_rows = max_rows
        ctx.group_rank = dist.get_rank(group=group)
        if max_rows == 0:
            return tensor.new_empty((0, *tensor.shape[1:]))
        padded = _pad_rows(tensor, max_rows)
        gathered = [torch.zeros_like(padded) for _ in counts]
        dist.all_gather(gathered, padded, group=group)
        return torch.cat([chunk[:count] for chunk, count in zip(gathered, counts)], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        local_count = ctx.counts[ctx.group_rank]
        if ctx.max_rows == 0:
            return grad_output.new_empty((0, *grad_output.shape[1:])), None, None, None
        reduced_chunks = []
        for chunk, count in zip(torch.split(grad_output, ctx.counts, dim=0), ctx.counts):
            grad_padded = grad_output.new_zeros((ctx.max_rows, *grad_output.shape[1:]))
            if count:
                grad_padded[:count].copy_(chunk)
            # grad_output is replicated across TP peers (the gathered full view
            # is consumed identically), so this all_reduce sums tp_size copies
            # and injects the tp_size factor documented in the class docstring.
            dist.all_reduce(grad_padded, group=ctx.group)
            reduced_chunks.append(grad_padded)
        grad_padded = reduced_chunks[ctx.group_rank]
        return grad_padded[:local_count].contiguous(), None, None, None


def _all_gather_variable_rows(tensor: torch.Tensor,
                              group,
                              tp_size: int,
                              *,
                              preserve_grad: bool = False) -> torch.Tensor:
    if tp_size <= 1 or not dist.is_initialized():
        return tensor

    local_rows = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    row_counts = [torch.zeros_like(local_rows) for _ in range(tp_size)]
    dist.all_gather(row_counts, local_rows, group=group)
    counts = [int(item.item()) for item in row_counts]
    max_rows = max(counts) if counts else tensor.shape[0]
    if preserve_grad:
        return _AllGatherVariableRows.apply(tensor, group, tuple(counts), max_rows)
    else:
        padded = _pad_rows(tensor, max_rows)
        gathered = [torch.zeros_like(padded) for _ in range(tp_size)]
        dist.all_gather(gathered, padded, group=group)
    return torch.cat([chunk[:count] for chunk, count in zip(gathered, counts)], dim=0)


def _debug_validate_restore_coverage(payload: RoutedAssignmentPayload, ctx: RestoreContext,
                                     all_token_indices: torch.Tensor, all_expert_indices: torch.Tensor,
                                     all_assignment_indices: torch.Tensor, all_capacity_slots: torch.Tensor) -> None:
    active = ~payload.drop_mask & ~payload.pad_mask
    expected_rows = torch.stack((
        payload.token_indices[active].to(torch.long),
        payload.expert_indices[active].to(torch.long),
        payload.assignment_indices[active].to(torch.long),
        payload.capacity_slots[active].to(torch.long),
    ),
                                dim=1)
    observed_rows = torch.stack((
        all_token_indices.to(torch.long),
        all_expert_indices.to(torch.long),
        all_assignment_indices.to(torch.long),
        all_capacity_slots.to(torch.long),
    ),
                                dim=1)
    if expected_rows.numel() == 0 and observed_rows.numel() == 0:
        return
    if observed_rows.shape[0] != expected_rows.shape[0]:
        raise RuntimeError("AutoEP+AutoTP restore coverage mismatch: gathered assignment count "
                           f"{observed_rows.shape[0]} != expected {expected_rows.shape[0]}")
    if observed_rows.shape[0] <= 4096:
        expected = {tuple(row) for row in expected_rows.detach().cpu().tolist()}
        observed = {tuple(row) for row in observed_rows.detach().cpu().tolist()}
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            duplicate_or_stale = sorted(observed - expected)[:5]
            raise RuntimeError("AutoEP+AutoTP restore coverage mismatch: "
                               f"missing={missing} unexpected={duplicate_or_stale}")


def restore_combined(local_combined: torch.Tensor,
                     ctx: RestoreContext,
                     *,
                     tp_group,
                     validate_coverage: bool = False) -> torch.Tensor:
    """Gather TP-partitioned assignment outputs and combine back by token index.

    The all-gather rebuilds an identical full output on every TP peer, so all
    downstream compute (and the router/gate score gradient) is replicated across
    the folding group. Its differentiable backward injects a ``tp_size`` factor
    (see ``_AllGatherVariableRows``) that the optimizer's TP gradient reducer
    cancels with the AVERAGE strategy. A future true-SP path that kept
    activations sequence-sharded instead of gathering them here would make those
    parameters genuine SUM partials -- the reason the SUM family markers exist
    in ``deepspeed.module_inject.auto_ep_folding``.
    """
    payload = ctx.original_payload
    local_token_indices = payload.token_indices.index_select(0, ctx.local_indices)
    local_capacity_slots = payload.capacity_slots.index_select(0, ctx.local_indices)
    local_weights = payload.combine_weights.index_select(0, ctx.local_indices).to(local_combined.dtype)

    all_outputs = _all_gather_variable_rows(local_combined,
                                            tp_group,
                                            ctx.tp_size,
                                            preserve_grad=local_combined.requires_grad)
    all_token_indices = _all_gather_variable_rows(local_token_indices, tp_group, ctx.tp_size).to(torch.long)
    all_capacity_slots = _all_gather_variable_rows(local_capacity_slots, tp_group, ctx.tp_size).to(torch.long)
    all_weights = _all_gather_variable_rows(local_weights,
                                            tp_group,
                                            ctx.tp_size,
                                            preserve_grad=local_weights.requires_grad).to(local_combined.dtype)
    if validate_coverage:
        local_expert_indices = payload.expert_indices.index_select(0, ctx.local_indices)
        local_assignment_indices = payload.assignment_indices.index_select(0, ctx.local_indices)
        all_expert_indices = _all_gather_variable_rows(local_expert_indices, tp_group, ctx.tp_size).to(torch.long)
        all_assignment_indices = _all_gather_variable_rows(local_assignment_indices, tp_group,
                                                           ctx.tp_size).to(torch.long)
        _debug_validate_restore_coverage(payload, ctx, all_token_indices, all_expert_indices, all_assignment_indices,
                                         all_capacity_slots)

    if ctx.num_tokens <= 0:
        ctx.num_tokens = int(payload.token_indices.max().item()) + 1 if payload.token_indices.numel() else 0
    output = local_combined.new_zeros((ctx.num_tokens, local_combined.shape[-1]))
    if all_outputs.numel() > 0:
        weight_shape = (-1, ) + (1, ) * (all_outputs.dim() - 1)
        weighted_outputs = all_outputs * all_weights.reshape(weight_shape)
        # Add one top-k slot at a time so token accumulation order stays stable
        # without materializing a [tokens, top_k, hidden] buffer.
        for slot in torch.unique(all_capacity_slots, sorted=True).tolist():
            rows = all_capacity_slots == int(slot)
            output.index_add_(0, all_token_indices[rows], weighted_outputs[rows])
    return output


def dispatch_counters(ctx: RestoreContext) -> dict[str, int]:
    return dict(ctx.counters)
