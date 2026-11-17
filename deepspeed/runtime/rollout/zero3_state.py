# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Moving a ZeRO-3 engine's training state to another ZeRO-3 engine of the same model over other ranks.

An RL loop that changes its layout between phases (e.g. the trainer on half the GPUs while the rollout generates on
the other half, then the trainer on all of them) keeps one engine per trainer layout and moves the state between
them: per trainable parameter, its fp32 master partition and the optimizer's moments (every optimizer state tensor
as large as the partition, e.g. Adam's ``exp_avg`` and ``exp_avg_sq``), with :class:`~.relayout.RelayoutPlan` from
the source engine's storage to the destination's, both described by the affine optimizer storage maps of
:mod:`deepspeed.checkpoint.zero_affine` and read and written through its shard accessors. The destination then takes its bf16/fp16 partitions from
the fp32 ones, as after an optimizer step, and the step counters (optimizer, engine, LR scheduler) of the source.

Both engines keep their memory unless the inactive one gives it back: :func:`release_training_state` frees an
engine's partitions, optimizer state and gradient buffers (their storage, so every view of them stays valid) and
:func:`restore_training_state` allocates them again (their values undefined until a move fills them);
``execute(release_source=True)`` does both around a move. Offloaded optimizer state and parameters are not
supported.
"""

import hashlib
import time
from typing import Dict, List, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.zero_affine import (_global_ranks, zero_optimizer_storage_affine_maps,
                                              zero_optimizer_storage_shard)
from deepspeed.runtime.rollout.relayout import RelayoutPlan

_ENGINE_COUNTERS = ("global_steps", "global_samples", "micro_steps")


def _storages(engine) -> List[torch.UntypedStorage]:
    """The storages of an engine's training state on this rank: low-precision and fp32 partitions, optimizer state
    tensors of a partition's size, the gradient partitions and the reduce buckets (each storage once)."""
    opt = _zero3_optimizer(engine)
    tensors = list(opt.fp16_partitioned_groups_flat) + list(opt.fp32_partitioned_groups_flat)
    for flat in opt.fp32_partitioned_groups_flat:
        for v in opt.optimizer.state.get(flat, {}).values():
            if torch.is_tensor(v) and v.numel() == flat.numel():
                tensors.append(v)
    tensors.append(opt.grad_partitions_flat_buffer)
    tensors += list(getattr(opt, "grad_partitions_flat_buffers", {}).values())
    tensors += [bucket.buffer for bucket in opt.ipg_buckets.values()]
    storages, seen = [], set()
    for t in tensors:
        if t is None or t.untyped_storage().nbytes() == 0:
            continue
        storage = t.untyped_storage()
        if storage.data_ptr() not in seen:
            seen.add(storage.data_ptr())
            storages.append(storage)
    return storages


def release_training_state(engine) -> int:
    """Free ``engine``'s training state on this rank (partitions, optimizer state, gradient buffers) and return the
    bytes freed. The engine must not run until :func:`restore_training_state`; tensors and views keep their shapes
    with no memory behind them."""
    opt = _zero3_optimizer(engine)
    if getattr(opt, "_released_storages", None) is not None:
        return 0
    get_accelerator().synchronize()
    released = []
    for storage in _storages(engine):
        released.append((storage, storage.nbytes()))
        storage.resize_(0)
    opt._released_storages = released
    get_accelerator().empty_cache()
    return sum(n for _, n in released)


def restore_training_state(engine) -> None:
    """Allocate again what :func:`release_training_state` freed; the values are undefined until a move fills them."""
    opt = _zero3_optimizer(engine)
    for storage, nbytes in getattr(opt, "_released_storages", None) or []:
        storage.resize_(nbytes)
    opt._released_storages = None


def _zero3_optimizer(engine):
    opt = engine.optimizer
    if not hasattr(opt, "fp32_partitioned_groups_flat"):
        raise NotImplementedError("Zero3StateMove needs ZeRO stage 3 with an optimizer")
    if opt.offload_optimizer or opt.offload_param or opt.swap_optimizer:
        raise NotImplementedError("Zero3StateMove does not support offloaded optimizer state or parameters")
    return opt


class Zero3StateMove:
    """Plan moving the training state of ``src`` to ``dst`` (every rank builds it; ``src`` / ``dst`` is None on
    ranks outside that engine). Both engines hold the same model (same trainable parameter names and sizes) with
    ZeRO-3 partitions over any ranks. :meth:`execute` moves; the plan can be kept and executed again (e.g. after more
    training on ``src``), and the reverse move is another instance. Plan once ``src`` has stepped: the optimizer
    creates its moments lazily, and :meth:`execute` refuses a plan made without them.

    Args:
        src, dst: DeepSpeed ZeRO-3 engines, or None.
        topology, spread, planner: as for :class:`~.relayout.RelayoutPlan`.
        bucket_numel: full-weight elements per bucket; the buckets' all-to-alls are pipelined.
    """

    def __init__(self,
                 src,
                 dst,
                 topology: str = "flat",
                 spread: bool = False,
                 bucket_numel: int = 1 << 28,
                 planner: str = "rect"):
        self.src, self.dst = src, dst
        mine = {}
        for side, engine in (("src", src), ("dst", dst)):
            if engine is not None:
                params = self._params(engine)
                mine[side] = [(n, p.ds_numel) for n, p in params]
        everyone = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, mine)
        src_params = next((m["src"] for m in everyone if "src" in m), None)
        dst_params = next((m["dst"] for m in everyone if "dst" in m), None)
        if src_params is None or dst_params is None:
            raise ValueError(f"no rank passed a {'source' if src_params is None else 'destination'} engine")
        if src_params != dst_params:
            raise ValueError("the two engines hold different trainable parameters")
        self.info = self._source_state()
        self.moments = self.info["moments"]
        numels = dict(src_params)
        self.weights = [f"{kind}:{n}" for n in numels for kind in ("fp32", ) + tuple(self.moments)]
        shapes = {w: (numels[w.split(":", 1)[1]], ) for w in self.weights}
        holdings = self._pieces(src, "src") if src is not None else {}
        needs = self._pieces(dst, "dst") if dst is not None else {}
        self.plan = RelayoutPlan(holdings,
                                 needs,
                                 shapes,
                                 self.weights,
                                 topology=topology,
                                 spread=spread,
                                 planner=planner)
        self.buckets, size = [[]], 0
        for w in self.weights:
            if size >= bucket_numel:
                self.buckets.append([])
                size = 0
            self.buckets[-1].append(w)
            size += shapes[w][0]

    def _source_state(self) -> dict:
        """The source's state besides the partitions (every rank calls it): its optimizer's state kinds and
        scalars, param-group settings, engine counters and LR scheduler, as the source's first partition has them
        (every source rank steps alike)."""
        mine = None
        if self.src is not None and dist.get_rank(self._params(self.src)[0][1].ds_process_group) == 0:
            opt = _zero3_optimizer(self.src)
            moments, scalars = self._state_kinds(opt)
            scheduler = self.src.lr_scheduler
            mine = {
                "moments": moments,
                "scalars": scalars,
                "groups": [{
                    k: v
                    for k, v in g.items() if k != "params"
                } for g in opt.optimizer.param_groups],
                "counters": {
                    k: getattr(self.src, k)
                    for k in _ENGINE_COUNTERS if hasattr(self.src, k)
                },
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
        everyone = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, mine)
        return next(m for m in everyone if m is not None)

    @staticmethod
    def _params(engine) -> List[Tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in engine.module.named_parameters() if p.requires_grad]

    @staticmethod
    def _state_kinds(opt) -> Tuple[List[str], Dict[str, tuple]]:
        """The optimizer's per-element state keys, and its scalar states (e.g. the step) with how to rebuild them."""
        flat = opt.fp32_partitioned_groups_flat[0]
        moments, scalars = [], {}
        for k, v in opt.optimizer.state.get(flat, {}).items():
            if torch.is_tensor(v) and v.numel() == flat.numel():
                moments.append(k)
            elif torch.is_tensor(v):
                scalars[k] = ("tensor", v.item(), v.dtype, v.device.type)
            else:
                scalars[k] = ("value", v)
        return sorted(moments), scalars

    def _pieces(self, engine, local: str):
        """This rank's ``{weight: [(local, piece)]}`` of ``engine``'s fp32 masters and moments, from its optimizer
        storage maps (metadata only: the state may be released, with no storage to take views of, when planning)."""
        params = self._params(engine)
        rank = dist.get_rank()
        pieces = {}
        for (n, _), storage in zip(params, zero_optimizer_storage_affine_maps([p for _, p in params])):
            mine = [(local, piece) for piece in storage.pieces_by_rank.get(rank, []) if piece.numel]
            for kind in ("fp32", ) + tuple(self.moments):
                pieces[f"{kind}:{n}"] = mine
        return pieces

    def _tensors(self, engine, local: str) -> Dict[str, torch.Tensor]:
        tensors = {}
        for n, p in self._params(engine):
            for kind in ("fp32", ) + tuple(self.moments):
                tensors[f"{local}:{kind}:{n}"] = zero_optimizer_storage_shard(p, None if kind == "fp32" else kind)
        return tensors

    def _prepare_destination(self) -> None:
        """Give the destination optimizer the source's state tensors where it has none yet (never stepped)."""
        opt = _zero3_optimizer(self.dst)
        for flat in opt.fp32_partitioned_groups_flat:
            state = opt.optimizer.state[flat]
            for k in self.moments:
                if k not in state:
                    state[k] = torch.zeros_like(flat)

    @torch.no_grad()
    def execute(self, release_source: bool = False) -> None:
        """Move the state (every rank calls it, between the source's steps). A released destination is restored
        first; ``release_source`` releases the source afterwards (see :func:`release_training_state`)."""
        acc = get_accelerator()
        if self.dst is not None:
            restore_training_state(self.dst)
        self.info = self._source_state()
        if self.info["moments"] != self.moments:
            raise ValueError(f"the source optimizer's state is now {self.info['moments']}, planned for "
                             f"{self.moments}: plan the move again")
        if self.dst is not None:
            self._prepare_destination()
        sources = self._tensors(self.src, "src") if self.src is not None else {}
        dests = self._tensors(self.dst, "dst") if self.dst is not None else {}
        self.plan.execute(sources, dests, buckets=self.buckets, dtype=torch.float32, device=acc.current_device_name())
        if self.dst is not None:
            self._finish_destination()
        acc.synchronize()
        if release_source and self.src is not None:
            release_training_state(self.src)

    def _finish_destination(self) -> None:
        acc = get_accelerator()
        opt = _zero3_optimizer(self.dst)
        for i, flat in enumerate(opt.fp32_partitioned_groups_flat):
            # the forward reads the low-precision partitions, refreshed from the fp32 master as after a step
            opt.fp16_partitioned_groups_flat[i].data.copy_(flat.data)
            state = opt.optimizer.state[flat]
            for k, saved in self.info["scalars"].items():
                if saved[0] == "tensor":
                    device = "cpu" if saved[3] == "cpu" else acc.current_device_name()
                    state[k] = torch.tensor(saved[1], dtype=saved[2], device=device)
                else:
                    state[k] = saved[1]
        for group, saved in zip(opt.optimizer.param_groups, self.info["groups"]):
            group.update(saved)
        for k, v in self.info["counters"].items():
            setattr(self.dst, k, v)
        # The destination may accumulate over another number of micro-batches: start its next window on a multiple
        # of its own count (the engine's boundary is micro_steps modulo the accumulation steps).
        gas = self.dst.gradient_accumulation_steps()
        self.dst.micro_steps = -(-self.dst.micro_steps // gas) * gas
        if self.info["scheduler"] is not None and self.dst.lr_scheduler is not None:
            self.dst.lr_scheduler.load_state_dict(self.info["scheduler"])
        if hasattr(opt, "invalidate_secondary_tensor"):
            opt.invalidate_secondary_tensor()


def _flat_buffers(params, sizes, dtype_of, device) -> Dict[torch.dtype, torch.Tensor]:
    """One zeroed buffer per dtype (``dtype_of(param)``), as long as the partitions of that dtype together, keyed in the
    order the dtypes first appear."""
    totals = {}
    for p in params:
        dtype = dtype_of(p)
        totals[dtype] = totals.get(dtype, 0) + sizes[p.ds_id]
    return {dtype: torch.zeros(numel, dtype=dtype, device=device) for dtype, numel in totals.items()}


def _padded_partition(numel: int, partitions: int) -> Tuple[int, int]:
    """ZeRO-3's partition of ``numel`` elements over ``partitions`` ranks: (aligned numel, partition numel)."""
    aligned = numel + (-numel % partitions)
    return aligned, aligned // partitions


class Zero3Repartition:
    """Re-partition one ZeRO-3 engine's training state over other data-parallel ranks, in place.

    The engine stays: its module, parameters, optimizer, hooks and every tensor object it holds. What changes is the
    storage behind its partitions and the sizes it cached from the old group: the partition sizes and offsets, the flat
    buffers of the low-precision partitions, the fp32 master partitions, the optimizer's moments and the gradient
    partitions. New buffers are allocated and the existing tensor objects are pointed at them, so the optimizer state
    (keyed by them) and the hooks and the parameter coordinator (which hold them) stay valid. Per trainable parameter
    the fp32 master partition and the optimizer's moments move GPU to GPU with
    :class:`~.relayout.RelayoutPlan`, from the old partitions' storage maps to the new ones; the low-precision
    partitions are then refreshed from the fp32 master, as after an optimizer step. The global batch is kept: the
    gradient accumulation steps are recomputed for the new data-parallel size (``train_batch_size`` must divide).

    Every rank of the world builds it and calls :meth:`execute`, passing the engine where it has one (every rank that
    holds the state now or will hold it; the same engine object across changes, so a rank that leaves keeps it with its
    state released and can come back). ``mpu`` is the new layout's (e.g. a ``TrainerMPU`` over the new ranks), created by
    every rank. Without a re-pointable communicator handle, the engine's references to its data-parallel group (the
    parameters', the optimizer's, the parameter coordinator's, the engine's and ``groups.mpu``) are replaced by the new
    group: with a handle that step is the handle's re-point. Only same-process changes (ranks that held the engine
    when it was built) are supported; offloaded state, frozen parameters (e.g. a LoRA base), hpZ, MiCS, sequence
    parallelism and quantized weights are not.

    The plan is made in :meth:`execute` (source maps from the engine as it is, target maps after the install), over the
    world, like :class:`Zero3StateMove`, and kept: executing the same object again from a source layout it has moved
    from before (the same ranks and partition sizes, and the same optimizer state kinds) reuses that plan and skips both
    sides' maps. Keep one object per target layout, as with :class:`Zero3StateMove`; every rank must use the same
    objects, and where the ranks' caches disagree the plan is made anew. :attr:`plan_cached` says whether the last
    :meth:`execute` reused one.
    """

    def __init__(self,
                 engine,
                 mpu,
                 topology: str = "flat",
                 spread: bool = False,
                 bucket_numel: int = 1 << 28,
                 planner: str = "rect"):
        self.engine, self.mpu = engine, mpu
        self.topology, self.spread, self.planner, self.bucket_numel = topology, spread, planner, bucket_numel
        self.new_ranks = sorted(mpu.get_data_parallel_group_ranks())
        self.new_group = mpu.get_data_parallel_group()
        self.times = {}
        # ((source ranks, source partition sizes), state kinds) -> (plan, buckets); the target is this object's layout
        self._plans = {}
        self.plan, self.plan_cached = None, False

    # -- membership ------------------------------------------------------------------------------------------------

    @staticmethod
    def holds_state(engine) -> bool:
        """Whether this rank's ``engine`` holds training state (it is a member of the engine's current layout)."""
        return engine is not None and getattr(_zero3_optimizer(engine), "_repartition_member", True)

    def _check(self, engine, rank, new_member) -> None:
        if new_member and engine is None:
            raise ValueError(f"rank {rank} joins the new layout {self.new_ranks} without an engine")
        if engine is None:
            return
        opt = _zero3_optimizer(engine)
        from deepspeed.utils import groups
        if groups._get_sequence_parallel_world_size() > 1:
            raise NotImplementedError("re-partition does not support sequence parallelism")
        optimized = {id(p) for g in opt.fp16_groups for p in g}
        if any(hasattr(p, "ds_id") and id(p) not in optimized for p in engine.module.parameters()):
            # Frozen parameters (e.g. a LoRA base) are partitioned too and would need their own move.
            raise NotImplementedError("re-partition supports only engines whose ZeRO-3 parameters are all trained")
        if opt.micro_step_id != 0:
            raise RuntimeError("re-partition only between optimizer steps (pending micro-steps)")
        train_batch = engine.train_batch_size()
        if train_batch % (engine.train_micro_batch_size_per_gpu() * len(self.new_ranks)):
            raise ValueError(f"train_batch_size {train_batch} does not split over {len(self.new_ranks)} ranks "
                             f"of {engine.train_micro_batch_size_per_gpu()} per micro-batch")

    # -- the move --------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def execute(self) -> None:
        acc = get_accelerator()
        rank = dist.get_rank()
        engine = self.engine
        old_member = self.holds_state(engine)
        new_member = rank in self.new_ranks
        # Preconditions are checked on every rank and raised on all of them together: a rank that raised alone would
        # leave the others in the collectives below.
        error = None
        try:
            self._check(engine, rank, new_member)
        except (NotImplementedError, RuntimeError, ValueError) as e:
            error = f"rank {rank}: {e}"

        # Names and sizes, and the optimizer state besides the partitions, from the old layout's first rank.
        t = time.perf_counter()
        mine = None
        if error is None and old_member:
            opt = _zero3_optimizer(engine)
            if dist.get_rank(opt.dp_process_group) == 0:
                moments, scalars = Zero3StateMove._state_kinds(opt)
                scheduler = engine.lr_scheduler
                params = Zero3StateMove._params(engine)
                mine = {
                    "params": [(n, p.ds_numel) for n, p in params],
                    # what the source maps depend on (zero_storage_affine_map): the key of a cached plan
                    "layout":
                    (tuple(_global_ranks(opt.dp_process_group)), tuple(p.ds_tensor.ds_numel for _, p in params)),
                    "moments": moments,
                    "scalars": scalars,
                    "groups": [{
                        k: v
                        for k, v in g.items() if k != "params"
                    } for g in opt.optimizer.param_groups],
                    "counters": {
                        k: getattr(engine, k)
                        for k in _ENGINE_COUNTERS if hasattr(engine, k)
                    },
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                }
        cache_id = hashlib.sha1(repr(sorted(self._plans)).encode()).hexdigest()
        everyone = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, (mine, error, cache_id))
        errors = [e for _, e, _ in everyone if e]
        if errors:
            raise RuntimeError("Zero3Repartition: " + "; ".join(errors))
        info = next((m for m, _, _ in everyone if m is not None), None)
        if info is None:
            raise ValueError("no rank holds the engine's state")
        moments = info["moments"]
        kinds = ("fp32", ) + tuple(moments)
        # Reuse a plan only if every rank has it, so that all of them skip (or make) the plan's collectives. The kinds
        # are part of the key: an optimizer has no moments before its first step, and a plan made then moves only the
        # fp32 master.
        key = (info["layout"], kinds)
        cached = self._plans.get(key) if len({c for _, _, c in everyone}) == 1 else None
        self.plan_cached = cached is not None
        weights = [f"{kind}:{n}" for n, _ in info["params"] for kind in kinds]
        shapes = {f"{kind}:{n}": (numel, ) for n, numel in info["params"] for kind in kinds}

        # Sources: the old partitions' maps and views, taken before anything changes (the views keep the old storage).
        holdings, sources = {}, {}
        if old_member:
            # Release gathered copies (persistent parameters) through the optimizer: the engine's method also runs
            # gc.collect() and empties the cache, which execute does once at its end.
            _zero3_optimizer(engine).empty_partition_cache()
            if cached is None:
                holdings = self._pieces(engine, "src", kinds)
            sources = self._shards(engine, "src", kinds)
        self.times["sources"] = time.perf_counter() - t

        # Install the new layout (allocate, re-point, refresh), then the targets' maps and views.
        t = time.perf_counter()
        if new_member:
            self._install(engine, moments)
        acc.synchronize()
        self.times["install"] = time.perf_counter() - t

        t = time.perf_counter()
        if cached is None:
            needs = self._pieces(engine, "dst", kinds) if new_member else {}
            plan = RelayoutPlan(holdings,
                                needs,
                                shapes,
                                weights,
                                topology=self.topology,
                                spread=self.spread,
                                planner=self.planner)
            buckets, size = [[]], 0
            for w in weights:
                if size >= self.bucket_numel:
                    buckets.append([])
                    size = 0
                buckets[-1].append(w)
                size += shapes[w][0]
            self._plans[key] = (plan, buckets)
        else:
            plan, buckets = cached
        self.plan = plan
        self.times["plan"] = time.perf_counter() - t

        t = time.perf_counter()
        dests = self._shards(engine, "dst", kinds) if new_member else {}
        plan.execute(sources, dests, buckets=buckets, dtype=torch.float32, device=acc.current_device_name())
        if new_member:
            self._finish(engine, info)
        acc.synchronize()
        self.times["move"] = time.perf_counter() - t

        # Release what the old layout held: the views above on every rank, the whole state on a rank that leaves.
        del sources, dests
        if old_member and not new_member:
            release_training_state(engine)
            _zero3_optimizer(engine)._repartition_member = False
        acc.empty_cache()

    @staticmethod
    def _pieces(engine, local: str, kinds) -> Dict[str, list]:
        params = Zero3StateMove._params(engine)
        rank = dist.get_rank()
        pieces = {}
        for (n, _), storage in zip(params, zero_optimizer_storage_affine_maps([p for _, p in params])):
            mine = [(local, piece) for piece in storage.pieces_by_rank.get(rank, []) if piece.numel]
            for kind in kinds:
                pieces[f"{kind}:{n}"] = mine
        return pieces

    @staticmethod
    def _shards(engine, local: str, kinds) -> Dict[str, torch.Tensor]:
        tensors = {}
        for n, p in Zero3StateMove._params(engine):
            for kind in kinds:
                tensors[f"{local}:{kind}:{n}"] = zero_optimizer_storage_shard(p, None if kind == "fp32" else kind)
        return tensors

    # -- installing the new layout ---------------------------------------------------------------------------------

    def _repoint(self, engine) -> None:
        """Give every holder of the engine's data-parallel group the new group (a handle's re-point)."""
        from deepspeed.utils import groups
        opt = _zero3_optimizer(engine)
        old = opt.dp_process_group
        group = self.new_group
        for p in engine.module.parameters():
            if hasattr(p, "ds_id"):
                if p.ds_process_group is not old:
                    raise NotImplementedError("a parameter is partitioned over a group other than the engine's")
                p.ds_process_group = group
                p.ds_zero_partition_rank = dist.get_rank(group=group)
                p.ds_zero_partition_world_size = dist.get_world_size(group=group)
        opt.dp_process_group = group
        opt.sub_group_process_groups = [group for _ in opt.sub_group_process_groups]
        offload = opt.parameter_offload
        offload.dp_process_group = group
        coordinator = offload.get_param_coordinator()
        if hasattr(coordinator, "_PartitionedParameterCoordinator__process_group"):
            coordinator._PartitionedParameterCoordinator__process_group = group
        # The fetch trace is each rank's own: complete on a rank that kept training, empty on one that has not stepped
        # in this engine yet, and a trace that is not complete is compared with the other ranks' in a collective at the
        # next step. Every member therefore starts over, so that all of them record the next step together.
        if not coordinator.is_invalid_trace():
            coordinator._invalidate_trace()
        engine.mpu = self.mpu
        groups.mpu = self.mpu
        engine.data_parallel_group = group
        engine.seq_data_parallel_group = group

    def _install(self, engine, moments) -> None:
        acc = get_accelerator()
        device = acc.current_device_name()
        opt = _zero3_optimizer(engine)
        if opt.zero_hpz_partition_size > 1 or getattr(opt, "use_muon", False):
            raise NotImplementedError("re-partition does not support hpZ or Muon")
        self._repoint(engine)
        n = len(self.new_ranks)
        last = dist.get_rank(group=self.new_group) == n - 1

        # Low-precision partitions: one new buffer per partition dtype (ZeRO-3 keeps parameters of keep_in_fp32_modules
        # in fp32 beside the bf16 / fp16 ones, each dtype in its own buffer); every ds_tensor keeps its object and views
        # its slice, in the sub-groups' order, which is how the optimizer's flat groups narrow these buffers.
        params = [p for sub_group in opt.fp16_groups for p in sub_group]
        sizes = {}
        for p in params:
            aligned, part = _padded_partition(p.ds_numel, n)
            sizes[p.ds_id] = part
            p.ds_numel_aligned = aligned
        lp_buffers = _flat_buffers(params, sizes, lambda p: p.ds_tensor.dtype, device)
        offsets = dict.fromkeys(lp_buffers, 0)
        for p in params:
            dtype = p.ds_tensor.dtype
            p.ds_tensor.data = lp_buffers[dtype].narrow(0, offsets[dtype], sizes[p.ds_id])
            p.ds_tensor.ds_numel = sizes[p.ds_id]
            offsets[dtype] += sizes[p.ds_id]
        if hasattr(opt, "lp_param_buffers"):
            opt.lp_param_buffers = lp_buffers
            opt.lp_param_buffer = lp_buffers.get(opt.dtype, next(iter(lp_buffers.values())))
        elif len(lp_buffers) == 1:
            opt.lp_param_buffer = next(iter(lp_buffers.values()))
        else:
            raise NotImplementedError("this ZeRO-3 keeps parameter partitions in one dtype")
        opt.fp16_partitioned_groups_flat.clear()
        opt._set_fp16_partitioned_groups_flat()

        # Cached sizes.
        opt.partition_count = n
        opt.sub_group_partition_counts = [n for _ in opt.sub_group_partition_counts]
        opt.fp16_partitioned_groups_flat_numel = [sum(sizes[p.ds_id] for p in g) for g in opt.fp16_groups]
        opt.groups_padding = [[p.padding_size() if last else 0 for p in g] for g in opt.fp16_groups]
        opt._partition_group_spans_model_parallel.clear()

        # fp32 master partitions and the optimizer's moments: same objects (the optimizer's state is keyed by them).
        opt._point_aliased_fp32_partitions_at_lp_buffer()
        for i, flat in enumerate(opt.fp32_partitioned_groups_flat):
            numel = opt.fp16_partitioned_groups_flat_numel[i]
            if i not in opt.aliased_fp32_sub_groups:
                flat.data = torch.zeros(numel, dtype=flat.dtype, device=device)
            state = opt.optimizer.state[flat]
            for k in moments:
                if k in state:
                    state[k].data = torch.zeros(numel, dtype=state[k].dtype, device=device)
                else:
                    state[k] = torch.zeros(numel, dtype=flat.dtype, device=device)

        # Gradient partitions (one buffer per accumulation dtype, as for the low-precision partitions) and reduce
        # buckets (a released bucket comes back at its own capacity).
        accumulation_dtype = getattr(opt, "_grad_accumulation_dtype", lambda p: opt.grad_partitions_flat_buffer.dtype)
        grad_buffers = _flat_buffers(params, sizes, accumulation_dtype, device)
        offsets = dict.fromkeys(grad_buffers, 0)
        by_id = {}
        for p in params:
            dtype = accumulation_dtype(p)
            by_id[p.ds_id] = grad_buffers[dtype].narrow(0, offsets[dtype], sizes[p.ds_id])
            offsets[dtype] += sizes[p.ds_id]
        opt._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition = by_id
        if hasattr(opt, "grad_partitions_flat_buffers"):
            opt.grad_partitions_flat_buffers = grad_buffers
            opt.grad_partitions_flat_buffer = grad_buffers.get(opt.gradient_accumulation_dtype,
                                                               next(iter(grad_buffers.values())))
        elif len(grad_buffers) == 1:
            opt.grad_partitions_flat_buffer = next(iter(grad_buffers.values()))
        else:
            raise NotImplementedError("this ZeRO-3 keeps gradient partitions in one dtype")
        for bucket in opt.ipg_buckets.values():
            if bucket.buffer is not None and bucket.buffer.untyped_storage().nbytes() == 0:
                bucket.buffer = torch.empty(bucket.buffer.numel(), dtype=bucket.buffer.dtype, device=device)
        opt.grad_position = {}
        opt.set_grad_positions()
        opt.averaged_gradients = {}
        opt._released_storages = None
        opt._repartition_member = True

        # The engine's sizes and the batch arithmetic (same global batch, new accumulation steps).
        engine.dp_world_size = n
        engine.seq_dp_world_size = n
        engine.set_train_batch_size(engine.train_batch_size())
        opt.gradient_accumulation_steps = engine.gradient_accumulation_steps()

    def _finish(self, engine, info) -> None:
        acc = get_accelerator()
        opt = _zero3_optimizer(engine)
        for i, flat in enumerate(opt.fp32_partitioned_groups_flat):
            if i not in opt.aliased_fp32_sub_groups:
                opt.fp16_partitioned_groups_flat[i].data.copy_(flat.data)
            state = opt.optimizer.state[flat]
            for k, saved in info["scalars"].items():
                if saved[0] == "tensor":
                    device = "cpu" if saved[3] == "cpu" else acc.current_device_name()
                    state[k] = torch.tensor(saved[1], dtype=saved[2], device=device)
                else:
                    state[k] = saved[1]
        for group, saved in zip(opt.optimizer.param_groups, info["groups"]):
            group.update(saved)
        for k, v in info["counters"].items():
            setattr(engine, k, v)
        # The engine finds the accumulation boundary from micro_steps modulo the accumulation steps (its only use):
        # start the next window on a multiple of the new count.
        gas = engine.gradient_accumulation_steps()
        engine.micro_steps = -(-engine.micro_steps // gas) * gas
        if info["scheduler"] is not None and engine.lr_scheduler is not None:
            engine.lr_scheduler.load_state_dict(info["scheduler"])
        opt.invalidate_secondary_tensor()
