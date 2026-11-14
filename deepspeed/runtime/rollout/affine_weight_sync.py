# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Keep a tensor-parallel inference copy of a training model current, shard to shard, at fixed addresses.

A rollout engine that generates with its own tensor-parallel copy of the weights refreshes that copy after every
optimizer step. Gathering each parameter whole and cutting out the local shard sends every parameter to every rank.
Instead, each parameter is described twice with affine maps (:mod:`deepspeed.checkpoint.affine`): how the trainer
stores it (:func:`zero_param_shaped_storage_map`; ``autotp_storage_affine_maps`` should serve for an AutoTP model,
but that is not tested yet) and which block each inference rank needs (:func:`replicated_layout_map`).
:func:`~deepspeed.checkpoint.affine_transfer.plan_transfers` composes the two into copies from trainer shards straight
into inference shards, once per layout, and every sync runs the copies of all parameters as one
:func:`~deepspeed.checkpoint.affine_transfer_executor.execute_transfers` batch.

Which holder a replicated block is read from changes only the cost, never the result: once both maps pass the
planner's checks, every plan writes the same values whichever replica it reads, provided the replicas hold the same
values. Under ZeRO-1/2 the replicas are the data-parallel copies of each parameter, equal after every step's
all-gather; under ZeRO-3 each element has a single holder.

CUDA graphs record raw device pointers, so a decode graph captured on the inference copy stays valid only while every
update writes into the storage it was captured on. The executor writes targets in place, and :class:`WeightArena`
keeps a rank's shards as views of one buffer that is never reallocated.
"""

from typing import Iterator, Mapping, Optional, Sequence, Set, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.checkpoint.affine import ParamAffineMap, nest_affine_maps, replicated_map
from deepspeed.checkpoint.affine_transfer import PlanBudget, ReplicaPolicy, plan_transfers
from deepspeed.checkpoint.affine_transfer_executor import (StagingBudget, TransferEndpoints, TransferRequest,
                                                           TransferStats, execute_transfers)
from deepspeed.checkpoint.zero_affine import zero_param_storage_affine_map, zero_storage_affine_map

__all__ = ['WeightArena', 'zero_param_shaped_storage_map', 'replicated_layout_map', 'AffineWeightSync']

# A rank's tensor for each parameter it holds, by parameter name.
Buffers = Mapping[str, torch.Tensor]

# Each view starts on a 256-byte boundary, as allocator blocks do, so kernels see the alignment they expect.
_ALIGN_BYTES = 256


def _numel(shape: Sequence[int]) -> int:
    numel = 1
    for size in shape:
        numel *= size
    return numel


class WeightArena:
    """Named views of one flat buffer that is never reallocated.

    Args:
        shapes: ``{name: shape}`` in allocation order.
        dtype, device: of the buffer.
    """

    def __init__(self, shapes: Mapping[str, Sequence[int]], dtype: torch.dtype, device=None):
        element_size = torch.empty(0, dtype=dtype).element_size()
        align = max(1, _ALIGN_BYTES // element_size)
        offsets = {}
        total = 0
        for name, shape in shapes.items():
            total = -(-total // align) * align
            offsets[name] = total
            total += _numel(shape)
        # Zeros, so that padding a kernel may read (rows past the end of a vocabulary shard) holds no garbage. An
        # allocation is not always 256-byte aligned (the CPU allocator aligns to 64 bytes), so allocate one alignment
        # more and start the buffer at its first aligned element.
        storage = torch.zeros(total + align, dtype=dtype, device=device)
        skip = -(storage.data_ptr() // element_size) % align
        self.buffer = storage.narrow(0, skip, total)
        self._views = {}
        for name, offset in offsets.items():
            shape = tuple(shapes[name])
            self._views[name] = self.buffer.narrow(0, offset, _numel(shape)).view(shape)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._views[name]

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return iter(self._views.items())


def zero_param_shaped_storage_map(param) -> ParamAffineMap:
    """What every rank stores of a ZeRO model's ``param``, in the parameter's own shape and keyed by global rank.

    ``zero_param_storage_affine_map`` describes the flattened parameter, while a tensor-parallel layout describes it
    in its own shape, and both sides of a plan must describe the same logical tensor. This rank's shard is still
    ``zero_param_storage_shard(param)``.
    """
    shape = tuple(getattr(param, 'ds_shape', param.shape))
    return nest_affine_maps(replicated_map(shape, 1), {0: zero_param_storage_affine_map(param)})


def replicated_layout_map(layout: ParamAffineMap, replicas: Sequence[Sequence[int]]) -> ParamAffineMap:
    """A parameter's tensor-parallel layout, held by every replica of the inference copy, keyed by global rank.

    ``layout`` describes one replica, keyed by tensor-parallel rank ``0 .. tp - 1``: an AutoTP layer's layout map, or
    one built with :mod:`deepspeed.checkpoint.affine` (``contiguous_split_map``, ``replicated_map``, ...).
    ``replicas[i][j]`` is the global rank that is tensor-parallel rank ``j`` of replica ``i``. A rank's shard is its
    layout shard flattened, so a contiguous tensor of the layout shard's shape holds it.
    """
    tp_size = len(layout.shard_shapes)
    if sorted(layout.shard_shapes) != list(range(tp_size)):
        raise ValueError(f'The layout must be keyed by tensor-parallel ranks 0..{tp_size - 1}, '
                         f'not {sorted(layout.shard_shapes)}.')
    if not replicas or any(len(ranks) != tp_size for ranks in replicas):
        raise ValueError(f'Every replica needs {tp_size} ranks, one per tensor-parallel rank: {list(replicas)}.')
    # Rank j of every replica stores layout shard j whole, as data-parallel replicas store a parameter.
    storage = {}
    for tp_rank, shape in layout.shard_shapes.items():
        storage[tp_rank] = zero_storage_affine_map(_numel(shape), [ranks[tp_rank] for ranks in replicas])
    return nest_affine_maps(layout, storage)


class AffineWeightSync:
    """Copies parameters from where one side stores them into where the other side needs them, planned once.

    Args:
        source_maps: ``{name: map}`` of what the source side stores, keyed by global rank, for example
            :func:`zero_param_shaped_storage_map` of each parameter of a ZeRO model.
        target_maps: ``{name: map}`` of what the target side needs, keyed by global rank, for example from
            :func:`replicated_layout_map`. It names the same parameters as ``source_maps``.
        dtype: of every buffer; ranks that hold no buffer must agree on it too.
        group: the process group the copies cross, the world by default. It must contain every rank the maps name,
            and all of its ranks construct the sync with the same maps. The maps stay keyed by global rank with any
            group; the sync translates them to the group's ranks for the executor.
        budget: bytes staged per peer and direction, see
            :class:`~deepspeed.checkpoint.affine_transfer_executor.StagingBudget`.
        limits: planning limits, see :class:`~deepspeed.checkpoint.affine_transfer.PlanBudget`.
        replica_policy: where a block several ranks hold is read from, see
            :func:`~deepspeed.checkpoint.affine_transfer.plan_transfers`. By default
            :func:`~deepspeed.checkpoint.affine_replica_policy.local_first_least_loaded`: from the target's own process
            when it holds one (a local copy), and otherwise from the holder that sends the fewest bytes so far, over
            all parameters. Every rank must pass a policy that gives the same answers.

    Planning has no communication and is deterministic, so every rank derives the same plans.
    """

    def __init__(self,
                 source_maps: Mapping[str, ParamAffineMap],
                 target_maps: Mapping[str, ParamAffineMap],
                 dtype: torch.dtype,
                 group=None,
                 budget: StagingBudget = StagingBudget(),
                 limits: Optional[PlanBudget] = None,
                 replica_policy: Optional[ReplicaPolicy] = None):
        if set(source_maps) != set(target_maps):
            only_one = sorted(set(source_maps) ^ set(target_maps))
            raise ValueError(f'Source and target maps must name the same parameters; only one side names {only_one}.')
        self.dtype, self.group, self.budget = dtype, group, budget
        self.rank = dist.get_rank()
        if group is None:
            members = list(range(dist.get_world_size()))
        else:
            members = [dist.get_global_rank(group, rank) for rank in range(dist.get_world_size(group=group))]
        process = {rank: index for index, rank in enumerate(members)}
        source_ranks = {rank for affine_map in source_maps.values() for rank in affine_map.shard_shapes}
        target_ranks = {rank for affine_map in target_maps.values() for rank in affine_map.shard_shapes}
        outside = sorted((source_ranks | target_ranks) - set(process))
        if outside:
            raise ValueError(f'Ranks {outside} hold parameters but are not in the transfer group.')
        source_processes = {rank: process[rank] for rank in sorted(source_ranks)}
        target_processes = {rank: process[rank] for rank in sorted(target_ranks)}
        self.endpoints = TransferEndpoints(source=source_processes, target=target_processes)
        # The parameters this rank takes part in on each side.
        self._sources_here = {name for name, affine_map in source_maps.items() if self.rank in affine_map.shard_shapes}
        self._targets_here = {name for name, affine_map in target_maps.items() if self.rank in affine_map.shard_shapes}
        pairs = {name: (target_maps[name], source_maps[name]) for name in source_maps}
        element_size = dict.fromkeys(source_maps, torch.empty(0, dtype=dtype).element_size())
        plans = plan_transfers(pairs,
                               limits,
                               source_processes=source_processes,
                               target_processes=target_processes,
                               element_size=element_size,
                               replica_policy=replica_policy)
        self.plans = {name: plans[name] for name in sorted(plans)}

    def sync(self, source_buffers: Buffers, target_buffers: Buffers) -> TransferStats:
        """Copy every parameter. Collective over ``group``.

        ``source_buffers[name]`` is this rank's source shard of ``name`` and ``target_buffers[name]`` its target
        shard, for every parameter the maps place on this rank; entries for other parameters are ignored. Shards are
        contiguous tensors, which may be views (a view keeps its offset into its storage). The sources must not change
        while the copies run. The sync keeps no view of a shard between calls, so resolve the shards at every call:
        ``reload_states()`` after ``offload_states()`` moves every ZeRO-3 partition to new storage. A rank that lacks a shard refuses the batch in the executor's collective preflight, so
        every rank raises and no data moves.
        """
        requests = []
        for name, plan in self.plans.items():
            sources, source_error = self._local(name, self._sources_here, source_buffers, 'source')
            targets, target_error = self._local(name, self._targets_here, target_buffers, 'target')
            # Raising here would leave the other ranks waiting in the preflight, so the executor raises everywhere.
            error = source_error or target_error
            requests.append(TransferRequest(name, plan, sources, targets, preflight_error=error))
        return execute_transfers(requests, self.endpoints, self.dtype, self.budget, self.group)

    def _local(self, name: str, here: Set[str], buffers: Buffers, side: str):
        """``({rank: shard}, None)`` for this rank's side of ``name``, or ``({}, error)`` when the shard is missing."""
        if name not in here:
            return {}, None
        if name not in buffers:
            return {}, f'rank {self.rank} holds part of {name!r} but was given no {side} shard for it'
        return {self.rank: buffers[name]}, None
