# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP + AutoTP folding topology helpers.

The functions in this module are pure topology math unless a caller passes
runtime process-group handles into :class:`FoldingGroupHandles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ParallelFoldingSpec:
    world_size: int
    pp_size: int
    stage_size: int
    tp_size: int
    dp_size: int
    ep_size: int
    etp_size: int
    edp_size: int
    mp_mode: str = "tp"


@dataclass(frozen=True)
class FoldingGroupTables:
    tp_groups: tuple[tuple[int, ...], ...]
    dense_dp_groups: tuple[tuple[int, ...], ...]
    ep_groups: tuple[tuple[int, ...], ...]
    edp_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FoldingGroupHandles:
    spec: ParallelFoldingSpec
    tp_group: object
    dense_dp_group: object
    ep_group: object
    edp_group: object
    ep_group_name: str
    tp_ranks: tuple[int, ...]
    dense_dp_ranks: tuple[int, ...]
    ep_ranks: tuple[int, ...]
    edp_ranks: tuple[int, ...]


def _divisors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def build_folding_spec(
    *,
    world_size: int,
    pp_size: int,
    tp_size: int,
    ep_size: int,
    etp_size: int = 1,
    mp_mode: str = "tp",
) -> ParallelFoldingSpec:
    """Build the immutable per-stage folding spec from public config sizes."""
    for name, value in (
        ("world_size", world_size),
        ("pp_size", pp_size),
        ("tensor_parallel.autotp_size", tp_size),
        ("expert_parallel.autoep_size", ep_size),
        ("expert_parallel.expert_tensor_parallel_size", etp_size),
    ):
        _require_positive(name, value)

    if world_size % pp_size != 0:
        raise ValueError(f"pp_size={pp_size} must divide world_size={world_size}. "
                         f"Valid pp_size values: {_divisors(world_size)}")

    stage_size = world_size // pp_size
    if stage_size % tp_size != 0:
        raise ValueError(f"tensor_parallel.autotp_size={tp_size} must divide the stage size "
                         f"(world_size={world_size} / pp_size={pp_size} = {stage_size}). "
                         f"Computed dp would be non-integral. Valid autotp_size values: {_divisors(stage_size)}")

    expert_width = ep_size * etp_size
    if stage_size % expert_width != 0:
        raise ValueError(f"expert_parallel.autoep_size * expert_parallel.expert_tensor_parallel_size "
                         f"({ep_size} * {etp_size} = {expert_width}) must divide the stage size "
                         f"(world_size={world_size} / pp_size={pp_size} = {stage_size}). "
                         f"Computed edp would be non-integral. Valid expert-width values: {_divisors(stage_size)}")

    return ParallelFoldingSpec(
        world_size=world_size,
        pp_size=pp_size,
        stage_size=stage_size,
        tp_size=tp_size,
        dp_size=stage_size // tp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        edp_size=stage_size // expert_width,
        mp_mode=mp_mode,
    )


def expected_folding_group_tables(spec: ParallelFoldingSpec) -> FoldingGroupTables:
    """Derive TP, dense-DP, EP, and EDP rank tables without process groups."""
    tp_groups: list[tuple[int, ...]] = []
    dense_dp_groups: list[tuple[int, ...]] = []
    ep_groups: list[tuple[int, ...]] = []
    edp_groups: list[tuple[int, ...]] = []

    for stage_start in range(0, spec.world_size, spec.stage_size):
        stage_ranks = list(range(stage_start, stage_start + spec.stage_size))

        for dp_idx in range(spec.dp_size):
            start = dp_idx * spec.tp_size
            tp_groups.append(tuple(stage_ranks[start:start + spec.tp_size]))
        for tp_lane in range(spec.tp_size):
            dense_dp_groups.append(tuple(stage_ranks[tp_lane::spec.tp_size]))

        if spec.mp_mode == "tp" and spec.tp_size > 1:
            ordered_stage_ranks = []
            for tp_lane in range(spec.tp_size):
                ordered_stage_ranks.extend(stage_ranks[tp_lane::spec.tp_size])
        else:
            ordered_stage_ranks = stage_ranks

        local_ep_groups = [
            tuple(ordered_stage_ranks[start:start + spec.ep_size])
            for start in range(0, len(ordered_stage_ranks), spec.ep_size)
        ]
        ep_groups.extend(local_ep_groups)
        for pos in range(spec.ep_size):
            edp_groups.append(tuple(group[pos] for group in local_ep_groups))

    return FoldingGroupTables(
        tp_groups=tuple(tp_groups),
        dense_dp_groups=tuple(dense_dp_groups),
        ep_groups=tuple(ep_groups),
        edp_groups=tuple(edp_groups),
    )


def local_folding_ranks(global_rank: int, spec: ParallelFoldingSpec) -> dict[str, tuple[int, ...]]:
    tables = expected_folding_group_tables(spec)
    result = {}
    for name, groups in (
        ("tp", tables.tp_groups),
        ("dense_dp", tables.dense_dp_groups),
        ("ep", tables.ep_groups),
        ("edp", tables.edp_groups),
    ):
        result[name] = next(group for group in groups if global_rank in group)
    return result


def _mpu_world_size(mpu, *names: str) -> int | None:
    if mpu is None:
        return None
    for name in names:
        getter = getattr(mpu, name, None)
        if getter is not None:
            return getter()
    return None


def validate_folding_global(
    spec: ParallelFoldingSpec,
    *,
    zero_stage: int = 0,
    sp_size: int = 1,
    use_data_before_expert_parallel: bool = False,
    mpu=None,
    autoep_enabled: bool = True,
    tp_preset: str | None = None,
    ep_preset: str | None = None,
    zero_offload_optimizer: bool = False,
    zero_offload_param: bool = False,
) -> None:
    """Validate global folding policy before any process group is created."""
    if not autoep_enabled:
        return

    if spec.tp_size > 1 and spec.pp_size > 1:
        raise ValueError("AutoEP+AutoTP folding currently supports pp_size=1 only; "
                         f"got pp_size={spec.pp_size}. Pipeline-parallel validation is planned separately.")

    if spec.tp_size > 1 and sp_size > 1:
        raise ValueError("tensor_parallel.autotp_size and Ulysses sequence parallelism are mutually exclusive "
                         f"for AutoEP folding (autotp_size={spec.tp_size}, sp_size={sp_size}).")

    if spec.etp_size != 1:
        raise ValueError(f"expert_parallel.expert_tensor_parallel_size={spec.etp_size} is reserved for "
                         "expert-internal tensor parallelism and is not supported yet. Use 1; ETP support "
                         "is planned as follow-up work.")

    expert_width = spec.ep_size * spec.etp_size
    if spec.tp_size > 1 and expert_width > spec.dp_size:
        raise ValueError("AutoEP+AutoTP folding does not yet support cross-lane expert-parallel groups where "
                         "expert_parallel.autoep_size * expert_parallel.expert_tensor_parallel_size exceeds "
                         f"the derived dense data-parallel size (ep * etp = {expert_width}, dp = {spec.dp_size}, "
                         f"stage_size = {spec.stage_size}). This is a temporary limitation; use a shape with "
                         "ep * etp <= dp or run a follow-up implementation for cross-lane EP groups.")

    if tp_preset is not None and ep_preset is not None and tp_preset != ep_preset:
        raise ValueError("tensor_parallel.preset_model and expert_parallel.preset_model must match when both "
                         f"are set (tensor_parallel.preset_model={tp_preset!r}, "
                         f"expert_parallel.preset_model={ep_preset!r}).")

    if spec.tp_size > 1 and spec.ep_size == 1:
        raise ValueError("AutoEP+AutoTP folding requires expert_parallel.autoep_size > 1. "
                         "The ep=1 local-computation path would duplicate routed-token gradients across TP lanes.")

    if spec.tp_size > 1 and use_data_before_expert_parallel:
        raise ValueError("expert_parallel with use_data_before_expert_parallel_ is not supported with "
                         "AutoEP+AutoTP folding. Disable use_data_before_expert_parallel_.")

    if spec.tp_size > 1 and zero_stage == 3:
        raise ValueError("AutoEP+AutoTP with ZeRO stage 3 is reserved for the separate ZeRO-3 composition lane. "
                         "Use ZeRO stage 0, 1, or 2 for this folding MVP.")

    if spec.tp_size > 1 and (zero_offload_optimizer or zero_offload_param):
        raise ValueError("ZeRO optimizer/parameter offload with AutoEP+AutoTP folding is not validated yet. "
                         "Disable offload or run a follow-up proof for per-family replica groups.")

    mpu_tp = _mpu_world_size(mpu, "get_tensor_model_parallel_world_size", "get_model_parallel_world_size")
    if mpu_tp not in (None, 1, spec.tp_size):
        raise ValueError(f"mpu tensor/model parallel world size ({mpu_tp}) conflicts with "
                         f"tensor_parallel.autotp_size={spec.tp_size}.")
    mpu_pp = _mpu_world_size(mpu, "get_pipeline_model_parallel_world_size", "get_pipeline_parallel_world_size")
    if mpu_pp not in (None, spec.pp_size):
        raise ValueError(f"mpu pipeline parallel world size ({mpu_pp}) conflicts with pp_size={spec.pp_size}.")


def _normalize_rank_groups(groups: Iterable[Iterable[int]]) -> set[tuple[int, ...]]:
    return {tuple(int(rank) for rank in group) for group in groups}


def assert_group_matches_spec(existing_rank_lists, spec: ParallelFoldingSpec, *, group_kind: str = "ep_edp") -> None:
    """Ensure cached ``ep_size_N`` rank lists match the requested folding spec."""
    tables = expected_folding_group_tables(spec)
    expected_ep = _normalize_rank_groups(tables.ep_groups)
    expected_edp = _normalize_rank_groups(tables.edp_groups)

    if isinstance(existing_rank_lists, dict):
        observed_ep = existing_rank_lists.get("ep", [])
        observed_edp = existing_rank_lists.get("edp", [])
    else:
        observed_ep, observed_edp = existing_rank_lists

    for group in _normalize_rank_groups(observed_ep):
        if group not in expected_ep:
            raise RuntimeError(f"Cached expert-parallel group {group} does not match folding spec {spec}.")
    for group in _normalize_rank_groups(observed_edp):
        if group not in expected_edp:
            raise RuntimeError(f"Cached expert-data-parallel group {group} does not match folding spec {spec}.")
