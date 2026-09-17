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

import torch

AUTOEP_FOLDING_PARAM_FAMILY_ATTR = "ds_autoep_folding_param_family"
AUTOEP_FOLDING_ROUTER_GATE_REPLICATED_PARAM = "router_gate_replicated"
AUTOEP_FOLDING_ROUTER_GATE_PARTIAL_PARAM = "router_gate_partial"
AUTOEP_FOLDING_SP_SHARDED_LAYERNORM_PARAM = "sp_sharded_layernorm"
AUTOEP_FOLDING_GRAD_CORRECTED_ATTR = "ds_autoep_folding_grad_corrected"
AUTOEP_FOLDING_GRAD_REDUCE_SKIP = "skip"
AUTOEP_FOLDING_GRAD_REDUCE_SUM = "sum"
AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE = "average"
# Divide by tp_size with NO TP all_reduce. Used for routed-expert parameters: the
# folded forward all-gathers expert outputs into a replicated full view in
# ``restore_combined``, whose backward injects a ``tp_size`` factor (same factor the
# replicated router cancels via AVERAGE). Routed experts are not TP-replicated, so
# they must not be TP all_reduced; they only need that spurious ``tp_size`` factor
# divided out. The remaining data-parallel reduction is owned by the expert-data
# -parallel (EDP) path, and ``/tp_size`` is linear so it composes with that EDP
# all_reduce in either order.
AUTOEP_FOLDING_GRAD_REDUCE_EXPERT_TP_CANCEL = "expert_tp_cancel"


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

        local_ep_groups = [
            tuple(stage_ranks[start:start + spec.ep_size]) for start in range(0, len(stage_ranks), spec.ep_size)
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
    deepcompile_enabled: bool = False,
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

    if deepcompile_enabled and spec.tp_size > 1:
        raise ValueError("DeepCompile with AutoEP+AutoTP folding is not supported; "
                         "disable compile.deepcompile or use non-folded AutoEP with tensor_parallel.autotp_size=1.")

    if spec.tp_size > 1 and spec.pp_size > 1:
        raise ValueError("AutoEP+AutoTP folding currently supports pp_size=1 only; "
                         f"got pp_size={spec.pp_size}. Pipeline-parallel validation is planned separately.")

    if spec.tp_size > 1 and sp_size > 1:
        raise ValueError("tensor_parallel.autotp_size and Ulysses sequence parallelism are mutually exclusive "
                         f"for AutoEP folding (autotp_size={spec.tp_size}, sp_size={sp_size}).")

    if spec.etp_size != 1:
        raise ValueError(f"expert_parallel.expert_tensor_parallel_size={spec.etp_size} is reserved for "
                         "expert-internal tensor parallelism and is not supported yet. Use "
                         "expert_tensor_parallel_size=1; ETP support is planned as follow-up work.")

    # Cross-lane expert parallelism (expert_width = ep * etp need NOT be a subset of
    # the dense data-parallel size) is supported: ``expected_folding_group_tables``
    # lays EP groups across consecutive stage ranks while dense DP remains TP-lane
    # strided, so an EP group may span TP lanes and dense-DP ranks while preserving
    # node-local EP groups under node-contiguous rank mappings. The only structural
    # requirement is that the expert width tiles the stage cleanly, which
    # ``build_folding_spec`` already enforces (``stage_size % expert_width == 0``,
    # so ``edp`` is integral). The gradient convention holds across the pool
    # because each family's reduction is keyed to its replication structure, not
    # the EP layout: router/gate and dense/LayerNorm AVERAGE over the TP
    # (token-replication) group; routed experts cancel the restore ``tp_size``
    # factor (EXPERT_TP_CANCEL) and reduce data-parallel over
    # the EDP group. The earlier ``expert_width <= dp`` / ``dp % expert_width == 0``
    # fail-fast limitation is therefore removed; only genuinely non-tiling shapes are
    # rejected above (in ``build_folding_spec``).

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


def mark_autoep_folding_router_parameter(param) -> None:
    """Tag a router/gate parameter as the *replicated* folded family (AVERAGE).

    This is the ONLY family marker applied on the live forward path today:
    ``AutoEPMoELayer.__init__`` marks every ``router.*`` parameter with it. The
    folded router runs redundantly on every TP peer (same tokens, same routing)
    and its gradient is reconstructed into a replicated full view by the restore
    all-gather (see ``deepspeed.moe.ep_tp_dispatch._AllGatherVariableRows`` and
    ``restore_combined``). That all-gather backward scales each peer's slice by
    ``tp_size``, so the extra TP reduction must AVERAGE (all_reduce then divide
    by ``tp_size``); SUM would leave the ``tp_size`` factor, i.e. the 2.0x
    parity regression the CPU/Gloo tests guard.
    """
    setattr(param, AUTOEP_FOLDING_PARAM_FAMILY_ATTR, AUTOEP_FOLDING_ROUTER_GATE_REPLICATED_PARAM)


def mark_autoep_folding_partial_router_parameter(param) -> None:
    """Tag a router/gate parameter as a *routed-token partial* family (SUM).

    Forward-looking contract; NOT used on the current forward path -- only the
    unit tests in ``tests/unit/v1/moe/test_autoep_autotp_grad_parity.py`` set
    it. Use it only for a future design where the router's per-token work is
    genuinely partitioned across peers and the slices are NOT all-gathered back
    into a replicated full view, so each peer holds a real partial gradient that
    must be SUMed. Such a router is a SUM partial in any token-partitioned mode
    (``mp_mode in {"tp", "sp"}``) because its partition can ride the existing
    expert-dispatch all-to-all without changing the dense activation layout.
    Prove the SUM with a parity test (like the existing router/gate cases)
    before enabling it on a real forward path.
    """
    setattr(param, AUTOEP_FOLDING_PARAM_FAMILY_ATTR, AUTOEP_FOLDING_ROUTER_GATE_PARTIAL_PARAM)


def mark_autoep_folding_sp_sharded_layernorm_parameter(param) -> None:
    """Tag a LayerNorm parameter as *SP-sequence-sharded* family (SUM under SP).

    Forward-looking contract; NOT used on the current forward path -- only the
    unit tests set it. Unlike the router, a LayerNorm has no adjacent dispatch
    all-to-all to ride on, so the only way to token-partition it is to shard the
    sequence dimension of the dense activations, which is Sequence Parallel by
    definition. It therefore becomes a SUM partial only when ``mp_mode == "sp"``
    and otherwise falls back to the replicated AVERAGE. Today ``tp_size > 1``
    with sequence parallelism is rejected in ``validate_folding_global``; this
    marker is the explicit contract for when that restriction is lifted, and
    must be backed by a parity test before use.
    """
    setattr(param, AUTOEP_FOLDING_PARAM_FAMILY_ATTR, AUTOEP_FOLDING_SP_SHARDED_LAYERNORM_PARAM)


def _is_moe_param_marker(param) -> bool:
    return hasattr(param, "allreduce") and not param.allreduce


def _is_model_parallel_param_marker(param) -> bool:
    return bool(getattr(param, "model_parallel", False) or getattr(param, "tensor_model_parallel", False))


def _autoep_folding_param_family(param, *, param_name: str | None = None) -> str | None:
    """Resolve a parameter's folded reduction family.

    An explicit ``mark_autoep_folding_*`` tag always wins. The ``.router.`` name
    match is only a redundant safety net: ``AutoEPMoELayer`` already tags router
    params, so this fallback merely keeps the conservative *replicated* (AVERAGE)
    classification if some router param ever reaches the reducer untagged. It
    never returns a SUM family by name -- SUM families are opt-in via explicit
    markers only, so any unrecognized replicated/dense/LayerNorm param falls
    through to the AVERAGE default rather than being silently over-scaled.
    """
    family = getattr(param, AUTOEP_FOLDING_PARAM_FAMILY_ATTR, None)
    if family is not None:
        return family
    if param_name is not None and ".router." in param_name:
        return AUTOEP_FOLDING_ROUTER_GATE_REPLICATED_PARAM
    return None


def autoep_folding_gradient_reduction_strategy(
    folding_spec: ParallelFoldingSpec | None,
    param,
    *,
    param_name: str | None = None,
) -> str:
    """Classify one folded TP/SP gradient as ``sum``, ``average``, or ``skip``.

    TP means Tensor Parallel and SP means Sequence Parallel. The parallel mode
    alone is not a safe SUM-vs-AVG selector because different parameter
    families see different backward semantics:

    - Router/gate parameters that are explicitly marked as routed-token
      partials in TP/SP token-partitioned modes receive one partial gradient per
      lane, so their TP/SP reduction is a SUM. The current AutoEP folded router
      gate is marked ``router_gate_replicated`` because the full-flow backward
      reaches this reducer as a lane-replicated gradient; that family uses the
      same AVERAGE normalization as other replicated parameters.
    - Dense and LayerNorm parameters that are merely replicated by TP folding
      are not routed-token partials; blindly SUMing them scales gradients by
      the TP size, so their extra TP reduction is an AVERAGE.
    - A true SP-sharded LayerNorm would be a partial-gradient parameter and
      should SUM. The current AutoEP folding path does not mark runtime
      LayerNorm parameters that way; the marker and strategy boundary exist so
      future SP support has an explicit contract instead of reusing the dense
      replicated default by accident.
    - Model-parallel (genuinely TP-sharded) parameters are SKIP because the
      TP-specific path owns their reduction.
    - Routed-expert parameters are EXPERT_TP_CANCEL: their data-parallel
      reduction is owned by the EP/EDP path, but the folded forward all-gathers
      their outputs into a replicated full view in ``restore_combined`` (whose
      backward injects a ``tp_size`` factor), so the expert-weight gradient
      reaches the optimizer ``tp_size`` times too large. Experts are not
      TP-replicated, so the fix is a plain ``/tp_size`` (no TP all_reduce), which
      is linear and composes with the EDP all_reduce in any order. Without this,
      folded expert gradients are over-scaled by ``tp_size`` -- invisible to
      scale-invariant Adam but real for SGD/Lion/Muon and for gradient clipping
      (it inflates the expert contribution to the global grad norm).

    Underlying rule and mechanism: a folded parameter is replicated (AVERAGE)
    when the forward reconstructs its partitioned work into an identical full
    view inside the layer, and a genuine partial (SUM) only when the shard is
    kept all the way to the loss. Today the router/gate is partitioned across
    TP peers for dispatch but then all-gathered back by ``restore_combined``
    (see ``deepspeed.moe.ep_tp_dispatch``), whose backward scales each peer's
    gradient by ``tp_size``; the TP all_reduce then yields ``tp_size *
    full_grad`` and AVERAGE divides it out. Reducing with SUM would leave that
    factor -- the 2.0x router/gate parity regression the CPU/Gloo tests guard.
    The router can be a SUM partial in either ``tp`` or ``sp`` mode because its
    token partition can ride the existing dispatch all-to-all, whereas a
    LayerNorm becomes a partial only under true ``sp`` (sequence sharding): it
    has no adjacent all-to-all, so partitioning it requires changing the dense
    activation layout, which is Sequence Parallel by definition.

    Both the DeepSpeedEngine path and the ZeRO-2 path call this helper so the
    policy cannot silently drift between optimizers.
    """
    if folding_spec is None or getattr(folding_spec, "tp_size", 1) <= 1:
        return AUTOEP_FOLDING_GRAD_REDUCE_SKIP
    if _is_model_parallel_param_marker(param):
        # Genuinely TP-sharded (column/row-parallel) params: the TP-specific path
        # owns their reduction. Not produced by the folded skip-partition MVP.
        return AUTOEP_FOLDING_GRAD_REDUCE_SKIP
    if _is_moe_param_marker(param):
        # Routed-expert params. Their EP/EDP data-parallel reduction is owned by
        # the expert path, but the folded forward routes their outputs through the
        # ``restore_combined`` all-gather, whose backward leaves a ``tp_size``
        # factor on the expert-weight gradient (the same factor the replicated
        # router cancels with AVERAGE). Experts are NOT TP-replicated, so they must
        # not be TP all_reduced; the factor is cancelled with a plain ``/tp_size``.
        return AUTOEP_FOLDING_GRAD_REDUCE_EXPERT_TP_CANCEL

    family = _autoep_folding_param_family(param, param_name=param_name)
    mp_mode = getattr(folding_spec, "mp_mode", "tp")
    token_partitioned_mode = mp_mode in ("tp", "sp")
    if family == AUTOEP_FOLDING_ROUTER_GATE_PARTIAL_PARAM:
        return AUTOEP_FOLDING_GRAD_REDUCE_SUM if token_partitioned_mode else AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE
    if family == AUTOEP_FOLDING_ROUTER_GATE_REPLICATED_PARAM:
        return AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE
    if family == AUTOEP_FOLDING_SP_SHARDED_LAYERNORM_PARAM and mp_mode == "sp":
        return AUTOEP_FOLDING_GRAD_REDUCE_SUM
    return AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE


def reduce_autoep_folding_gradient(
    folding_spec: ParallelFoldingSpec | None,
    param,
    grad,
    *,
    tp_group,
    param_name: str | None = None,
) -> str:
    strategy = autoep_folding_gradient_reduction_strategy(folding_spec, param, param_name=param_name)
    if strategy == AUTOEP_FOLDING_GRAD_REDUCE_SKIP or grad is None or grad.data.is_sparse:
        return strategy

    from deepspeed import comm as dist

    grad_data = grad.data
    tp_world_size = dist.get_world_size(group=tp_group)

    # Routed experts: cancel the ``tp_size`` factor the restore all-gather leaves,
    # WITHOUT a TP all_reduce (experts are not TP-replicated; cross-TP summation of
    # disjoint expert-token slices is owned by the EDP all_reduce). ``/tp_size`` is
    # linear, so it composes with that EDP reduction in either order.
    if strategy == AUTOEP_FOLDING_GRAD_REDUCE_EXPERT_TP_CANCEL:
        if tp_world_size > 1:
            grad_data.div_(tp_world_size)
        return strategy

    if grad_data.dtype != torch.float32:
        reduced = grad_data.float()
        dist.all_reduce(reduced, group=tp_group)
        if strategy == AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE:
            reduced.div_(tp_world_size)
        grad_data.copy_(reduced.to(grad_data.dtype))
        return strategy

    dist.all_reduce(grad_data, group=tp_group)
    if strategy == AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE:
        grad_data.div_(tp_world_size)
    return strategy


def is_autoep_folding_gradient_corrected(param) -> bool:
    return bool(getattr(param, AUTOEP_FOLDING_GRAD_CORRECTED_ATTR, False))


def clear_autoep_folding_gradient_corrected(param) -> None:
    if hasattr(param, AUTOEP_FOLDING_GRAD_CORRECTED_ATTR):
        setattr(param, AUTOEP_FOLDING_GRAD_CORRECTED_ATTR, False)


def apply_folding_correction_to_grad_buffer(
    folding_spec: ParallelFoldingSpec | None,
    param,
    grad,
    *,
    tp_group,
    param_name: str | None = None,
    use_correction_marker: bool = True,
) -> str:
    if use_correction_marker and is_autoep_folding_gradient_corrected(param):
        return AUTOEP_FOLDING_GRAD_REDUCE_SKIP

    strategy = reduce_autoep_folding_gradient(folding_spec, param, grad, tp_group=tp_group, param_name=param_name)
    if use_correction_marker and strategy != AUTOEP_FOLDING_GRAD_REDUCE_SKIP:
        setattr(param, AUTOEP_FOLDING_GRAD_CORRECTED_ATTR, True)
    return strategy


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
