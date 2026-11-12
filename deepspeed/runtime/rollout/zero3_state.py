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

from typing import Dict, List, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.zero_affine import zero_optimizer_storage_affine_maps, zero_optimizer_storage_shard
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
        if self.info["scheduler"] is not None and self.dst.lr_scheduler is not None:
            self.dst.lr_scheduler.load_state_dict(self.info["scheduler"])
        if hasattr(opt, "invalidate_secondary_tensor"):
            opt.invalidate_secondary_tensor()
