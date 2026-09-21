# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import hashlib
import json
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

EXECUTOR_ARENA_ALIGNMENT = 256
DEFAULT_LIVE_BUDGET = 4_000_000_000
DEFAULT_FUSE_BUDGET = 1_000_000_000


@dataclass(frozen=True)
class ArenaOccurrence:
    """One contained gather lifetime in an executor-local arena plan."""

    ds_id: int
    occurrence: int
    first_use: int
    release: int
    nbytes: int
    dtype: Optional[torch.dtype] = None
    eligible: bool = True
    fallback_reason: Optional[str] = None

    def __post_init__(self):
        if self.occurrence < 0:
            raise ValueError("occurrence must be non-negative")
        if self.first_use < 0 or self.release < self.first_use:
            raise ValueError("arena lifetime must be contained and ordered")
        if self.nbytes <= 0 and self.eligible:
            raise ValueError("eligible arena allocations must have a positive size")
        if self.eligible and self.fallback_reason is not None:
            raise ValueError("eligible occurrences cannot have a fallback reason")
        if not self.eligible and not self.fallback_reason:
            raise ValueError("ineligible occurrences require a fallback reason")


@dataclass(frozen=True)
class ArenaPlanEntry:
    ds_id: int
    occurrence: int
    first_use: int
    release: int
    nbytes: int
    aligned_nbytes: int
    offset: int
    dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class ExecutorArenaPlan:
    alignment: int
    capacity: int
    max_live_bytes: int
    entries: Tuple[ArenaPlanEntry, ...]
    fallbacks: Tuple[ArenaOccurrence, ...]
    digest: str


@dataclass(frozen=True)
class ArenaAdmission:
    accepted: bool
    capacity: int
    demand_profile_bytes: int
    incremental_bytes: int
    live_budget: int
    reason: str


@dataclass(frozen=True)
class FrozenPersistence:
    selected_ds_ids: Tuple[int, ...]
    selected_bytes: int
    available_bytes: int
    reserved_live_bytes: int
    safety_reserve_bytes: int
    unused_bytes: int


@dataclass(frozen=True)
class GraphArenaPlan:
    occurrences: Tuple[ArenaOccurrence, ...]
    packed: ExecutorArenaPlan


@dataclass(frozen=True)
class ArenaRegistration:
    enabled: bool
    reason: str
    signature: str


def align_up(value: int, alignment: int = EXECUTOR_ARENA_ALIGNMENT) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("arena alignment must be a positive power of two")
    return (int(value) + alignment - 1) // alignment * alignment


def _add_free_block(free_blocks, offset: int, size: int) -> None:
    if size == 0:
        return
    free_blocks.append((offset, size))
    free_blocks.sort()

    merged = []
    for block_offset, block_size in free_blocks:
        if merged and merged[-1][0] + merged[-1][1] == block_offset:
            prev_offset, prev_size = merged[-1]
            merged[-1] = (prev_offset, prev_size + block_size)
        else:
            merged.append((block_offset, block_size))
    free_blocks[:] = merged


def _plan_digest(alignment: int, capacity: int, entries, fallbacks) -> str:
    payload = {
        "alignment":
        alignment,
        "capacity":
        capacity,
        "entries": [{
            "ds_id": entry.ds_id,
            "occurrence": entry.occurrence,
            "first_use": entry.first_use,
            "release": entry.release,
            "nbytes": entry.nbytes,
            "aligned_nbytes": entry.aligned_nbytes,
            "offset": entry.offset,
            "dtype": str(entry.dtype),
        } for entry in entries],
        "fallbacks": [{
            "ds_id": occurrence.ds_id,
            "occurrence": occurrence.occurrence,
            "first_use": occurrence.first_use,
            "release": occurrence.release,
            "nbytes": occurrence.nbytes,
            "dtype": str(occurrence.dtype),
            "reason": occurrence.fallback_reason,
        } for occurrence in fallbacks],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pack_executor_arena(occurrences: Iterable[ArenaOccurrence],
                        alignment: int = EXECUTOR_ARENA_ALIGNMENT) -> ExecutorArenaPlan:
    """Pack eligible occurrence intervals with deterministic best-fit reuse."""
    ordered = sorted(tuple(occurrences), key=lambda item: (item.first_use, item.release, item.ds_id, item.occurrence))
    eligible = [item for item in ordered if item.eligible]
    fallbacks = tuple(item for item in ordered if not item.eligible)

    active = []
    free_blocks = []
    entries = []
    capacity = 0
    max_live_bytes = 0

    for occurrence in eligible:
        still_active = []
        for release, offset, size in active:
            if release < occurrence.first_use:
                _add_free_block(free_blocks, offset, size)
            else:
                still_active.append((release, offset, size))
        active = still_active

        aligned_nbytes = align_up(occurrence.nbytes, alignment)
        candidates = [(size, offset, index) for index, (offset, size) in enumerate(free_blocks)
                      if size >= aligned_nbytes]
        if candidates:
            block_size, offset, index = min(candidates)
            del free_blocks[index]
            _add_free_block(free_blocks, offset + aligned_nbytes, block_size - aligned_nbytes)
        else:
            offset = align_up(capacity, alignment)
            capacity = offset + aligned_nbytes

        entry = ArenaPlanEntry(ds_id=occurrence.ds_id,
                               occurrence=occurrence.occurrence,
                               first_use=occurrence.first_use,
                               release=occurrence.release,
                               nbytes=occurrence.nbytes,
                               aligned_nbytes=aligned_nbytes,
                               offset=offset,
                               dtype=occurrence.dtype)
        entries.append(entry)
        active.append((occurrence.release, offset, aligned_nbytes))
        max_live_bytes = max(max_live_bytes, sum(item[2] for item in active))

    entries_tuple = tuple(entries)
    digest = _plan_digest(alignment, capacity, entries_tuple, fallbacks)
    return ExecutorArenaPlan(alignment=alignment,
                             capacity=capacity,
                             max_live_bytes=max_live_bytes,
                             entries=entries_tuple,
                             fallbacks=fallbacks,
                             digest=digest)


def _arena_producer(wait_node):
    producer = wait_node.args[0]
    if producer.target == torch.ops.dc.allgather_param.default:
        return producer, producer
    if producer.target == operator.getitem and producer.meta.get("deepcompile_arena_ds_id") is not None:
        return producer, producer.args[0]
    return None, None


def _arena_dtype(producer):
    if producer.target == torch.ops.dc.allgather_param.default:
        dtype = producer.kwargs.get("dtype")
    else:
        dtype = producer.meta.get("deepcompile_arena_dtype")
    if dtype is None:
        value = producer.meta.get("val")
        dtype = getattr(value, "dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _returns_alias(node) -> bool:
    if node.target == operator.getitem:
        return True
    if node.op == "call_method":
        return node.target in {
            "as_strided", "chunk", "contiguous", "detach", "expand", "flatten", "movedim", "narrow", "permute",
            "reshape", "select", "split", "squeeze", "swapaxes", "swapdims", "t", "transpose", "unbind", "unflatten",
            "unsqueeze", "view"
        }
    schema = getattr(node.target, "_schema", None)
    return schema is not None and any(result.alias_info is not None for result in schema.returns)


def _alias_escape_reason(wait_node, output_nodes) -> Optional[str]:
    pending = list(wait_node.users)
    visited = set()
    while pending:
        node = pending.pop()
        if node in visited or not _returns_alias(node):
            continue
        visited.add(node)
        if node in output_nodes:
            return "graph_output_alias_escape"
        if node.meta.get("deepcompile_saved_tensor"):
            return "saved_tensor_escape"
        pending.extend(node.users)
    return None


def plan_graph_executor_arena(graph) -> GraphArenaPlan:
    """Extract contained demand and fused gather occurrences from semantic FX."""
    nodes = list(graph.nodes)
    node_positions = {node: index for index, node in enumerate(nodes)}
    output_nodes = set()
    releases = defaultdict(list)
    for node in nodes:
        if node.op == "output":
            output_nodes.update(node.all_input_nodes)
        elif node.target == torch.ops.dc.release_param.default:
            releases[int(node.args[2])].append((node_positions[node], node))

    release_cursors = defaultdict(int)
    occurrence_cursors = defaultdict(int)
    previous_release = {}
    occurrences = []
    for wait_node in nodes:
        if wait_node.target != torch.ops.dc.wait_allgather.default:
            continue
        producer, lifetime_start = _arena_producer(wait_node)
        if producer is None:
            continue

        ds_id = int(wait_node.args[2])
        occurrence = occurrence_cursors[ds_id]
        occurrence_cursors[ds_id] += 1
        start = node_positions[lifetime_start]
        release_index = None
        cursor = release_cursors[ds_id]
        while cursor < len(releases[ds_id]):
            candidate, release_node = releases[ds_id][cursor]
            cursor += 1
            if candidate >= node_positions[wait_node]:
                release_index = candidate
                try:
                    release_count = int(release_node.args[3])
                except (IndexError, TypeError, ValueError):
                    release_count = 0
                if release_count <= 0:
                    release_index = None
                    break
                for _ in range(release_count - 1):
                    if cursor >= len(releases[ds_id]):
                        release_index = None
                        break
                    release_index = releases[ds_id][cursor][0]
                    cursor += 1
                break
        release_cursors[ds_id] = cursor

        raw_nbytes = producer.meta.get("allgather_allocation_bytes", producer.meta.get("tensor_size", 0))
        if isinstance(raw_nbytes, torch.SymInt):
            nbytes = 0
        else:
            try:
                nbytes = int(raw_nbytes)
            except (TypeError, ValueError, RuntimeError):
                nbytes = 0
        dtype = _arena_dtype(producer)
        fallback_reason = producer.meta.get("deepcompile_arena_fallback_reason")
        if release_index is None:
            release_index = start
            fallback_reason = fallback_reason or "missing_release"
        if nbytes <= 0:
            fallback_reason = fallback_reason or "dynamic_or_missing_size"
        if dtype is None:
            fallback_reason = fallback_reason or "dynamic_or_missing_dtype"
        if producer in output_nodes or wait_node in output_nodes:
            fallback_reason = fallback_reason or "graph_output_escape"
        fallback_reason = fallback_reason or _alias_escape_reason(wait_node, output_nodes)
        if producer.meta.get("deepcompile_saved_tensor") or wait_node.meta.get("deepcompile_saved_tensor"):
            fallback_reason = fallback_reason or "saved_tensor_escape"
        if previous_release.get(ds_id, -1) >= start:
            fallback_reason = fallback_reason or "overlapping_occurrence"
        previous_release[ds_id] = release_index

        occurrences.append(
            ArenaOccurrence(ds_id=ds_id,
                            occurrence=occurrence,
                            first_use=start,
                            release=release_index,
                            nbytes=max(0, nbytes),
                            dtype=dtype,
                            eligible=fallback_reason is None,
                            fallback_reason=fallback_reason))

    occurrences_tuple = tuple(occurrences)
    return GraphArenaPlan(occurrences=occurrences_tuple, packed=pack_executor_arena(occurrences_tuple))


def admit_executor_arena(plan: ExecutorArenaPlan,
                         demand_profile_bytes: int,
                         live_budget: int = DEFAULT_LIVE_BUDGET) -> ArenaAdmission:
    """Charge only allocation above the z3-only demand-gather profile."""
    if demand_profile_bytes < 0 or live_budget < 0:
        raise ValueError("arena admission inputs must be non-negative")
    incremental_bytes = max(0, plan.capacity - int(demand_profile_bytes))
    accepted = incremental_bytes <= live_budget
    return ArenaAdmission(accepted=accepted,
                          capacity=plan.capacity,
                          demand_profile_bytes=int(demand_profile_bytes),
                          incremental_bytes=incremental_bytes,
                          live_budget=int(live_budget),
                          reason="accepted" if accepted else "live_budget_exceeded")


def freeze_persistence(candidates: Iterable[Tuple[int, int]],
                       headroom_bytes: int,
                       live_budget: int = DEFAULT_LIVE_BUDGET,
                       safety_reserve_bytes: int = 0) -> FrozenPersistence:
    """Freeze persistence after reserving the full generation live budget."""
    if min(headroom_bytes, live_budget, safety_reserve_bytes) < 0:
        raise ValueError("persistence budget inputs must be non-negative")
    available_bytes = max(0, int(headroom_bytes) - int(live_budget) - int(safety_reserve_bytes))
    selected = []
    selected_bytes = 0
    for ds_id, nbytes in candidates:
        if nbytes < 0:
            raise ValueError("persistent candidate size must be non-negative")
        if selected_bytes + nbytes > available_bytes:
            continue
        selected.append(int(ds_id))
        selected_bytes += int(nbytes)

    return FrozenPersistence(selected_ds_ids=tuple(selected),
                             selected_bytes=selected_bytes,
                             available_bytes=available_bytes,
                             reserved_live_bytes=int(live_budget),
                             safety_reserve_bytes=int(safety_reserve_bytes),
                             unused_bytes=available_bytes - selected_bytes)


def executor_plan_signature(plan: Optional[ExecutorArenaPlan], disabled_reason: str = "no_plan") -> str:
    """Return a value every rank can compare at the same consensus point."""
    if plan is None:
        payload = {"enabled": False, "reason": disabled_reason}
    else:
        payload = {
            "enabled": True,
            "alignment": plan.alignment,
            "capacity": plan.capacity,
            "digest": plan.digest,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _signature_int(signature: str) -> int:
    return int(signature[:16], 16) & ((1 << 63) - 1)


def register_executor_arena(nz3,
                            graph_id: int,
                            graph_plan: Optional[GraphArenaPlan],
                            process_group=None,
                            disabled_reason: str = "no_plan",
                            *,
                            bwd: bool = False,
                            admission: Optional[ArenaAdmission] = None) -> ArenaRegistration:
    """Register metadata only; the native executor materializes backing on first real gather."""
    packed = graph_plan.packed if graph_plan is not None else None
    if admission is not None and not admission.accepted:
        graph_plan = None
        packed = None
        disabled_reason = admission.reason
    signature = executor_plan_signature(packed, disabled_reason)
    reason = "accepted" if packed is not None else disabled_reason

    if dist.is_initialized():
        fingerprint = _signature_int(signature)
        device = torch.device(get_accelerator().current_device_name())
        minimum = torch.tensor([fingerprint], device=device, dtype=torch.int64)
        maximum = minimum.clone()
        dist.all_reduce(minimum, dist.ReduceOp.MIN, group=process_group)
        dist.all_reduce(maximum, dist.ReduceOp.MAX, group=process_group)
        if minimum.item() != maximum.item():
            packed = None
            graph_plan = None
            reason = "rank_plan_mismatch"
            signature = executor_plan_signature(None, reason)

    if graph_plan is None or packed is None or packed.capacity == 0:
        nz3.configure_z3_gather_arena(graph_id, bwd, 0, EXECUTOR_ARENA_ALIGNMENT, [], [], [], [], [], signature)
        return ArenaRegistration(enabled=False, reason=reason if packed is None else "empty_plan", signature=signature)

    entries = {(entry.ds_id, entry.occurrence): entry for entry in packed.entries}
    ds_ids = []
    occurrences = []
    offsets = []
    nbytes = []
    dtypes = []
    for occurrence in graph_plan.occurrences:
        entry = entries.get((occurrence.ds_id, occurrence.occurrence))
        ds_ids.append(occurrence.ds_id)
        occurrences.append(occurrence.occurrence)
        offsets.append(entry.offset if entry is not None else -1)
        nbytes.append(occurrence.nbytes)
        dtypes.append(occurrence.dtype if occurrence.dtype is not None else torch.uint8)

    nz3.configure_z3_gather_arena(graph_id, bwd, packed.capacity, packed.alignment, ds_ids, occurrences, offsets,
                                  nbytes, dtypes, signature)
    return ArenaRegistration(enabled=True, reason=reason, signature=signature)
