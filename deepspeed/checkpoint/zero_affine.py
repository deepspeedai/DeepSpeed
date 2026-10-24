# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Affine maps of the storage ZeRO keeps for a model parameter.

A transfer planner needs every legal holder of every element. These maps describe a flattened
parameter keyed by global rank, both for the low-precision parameter the model computes with and
for the fp32 master copy and optimizer states that ZeRO partitions across the data-parallel group.
Each map comes with an accessor for the matching local shard, which is a view of ZeRO's storage,
so it can be read as a transfer source or written as a transfer target.

Under AutoTP these maps describe one tensor-parallel shard; ``autotp_storage_affine_maps`` nests
them in each parameter's AutoTP layout to describe the whole parameter in its own shape.
"""

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import AffinePiece, ParamAffineMap, nest_affine_maps, replicated_map

__all__ = [
    'zero_storage_affine_map', 'zero_param_storage_affine_map', 'zero_param_storage_shard',
    'zero_optimizer_storage_affine_maps', 'zero_optimizer_storage_shard', 'autotp_storage_affine_maps'
]


def _flat_piece(numel, start, rank):
    return AffinePiece(shape=(numel, ),
                       source_offset=start,
                       source_strides=(1, ),
                       dest_offset=0,
                       dest_strides=(1, ),
                       locations=[rank])


def _fragment_affine_map(numel, fragments):
    # ``fragments`` lists (rank, start, length): each rank holds one contiguous range, possibly empty.
    shard_shapes = {}
    pieces_by_rank = {}
    for rank, start, length in fragments:
        shard_shapes[rank] = (length, )
        pieces_by_rank[rank] = [_flat_piece(length, start, rank)] if length > 0 else []
    affine_map = ParamAffineMap(logical_shape=(numel, ), shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)
    affine_map.validate()
    return affine_map


def _zero3_bounds(numel, partition_numel, index):
    # The trailing padding of the last partitions holds no parameter elements, so it is not part of the shard.
    start = min(index * partition_numel, numel)
    stop = min(start + partition_numel, numel)
    return start, stop - start


def zero_storage_affine_map(numel, holder_ranks, partition_numel=None):
    """Describe a flattened parameter of ``numel`` elements stored by ``holder_ranks``.

    Without ``partition_numel`` every holder stores the whole parameter, as data-parallel
    replicas do under ZeRO-0/1/2, so the one piece lists every holder. With it, holder ``i``
    stores elements ``[i * n, (i + 1) * n)`` as ZeRO-3 does; the trailing padding holds no
    parameter elements and is not part of the shard, so a holder past the end has an empty shard.
    """
    numel = int(numel)
    holder_ranks = [int(rank) for rank in holder_ranks]
    if numel < 0:
        raise ValueError(f'Parameter numel cannot be negative: {numel}.')
    if not holder_ranks or len(set(holder_ranks)) != len(holder_ranks):
        raise ValueError(f'Holder ranks must be a non-empty list of distinct ranks: {holder_ranks}.')

    if partition_numel is None:
        piece = AffinePiece(shape=(numel, ),
                            source_offset=0,
                            source_strides=(1, ),
                            dest_offset=0,
                            dest_strides=(1, ),
                            locations=holder_ranks)
        affine_map = ParamAffineMap(logical_shape=(numel, ),
                                    shard_shapes={rank: (numel, )
                                                  for rank in holder_ranks},
                                    pieces_by_rank={rank: [piece]
                                                    for rank in holder_ranks})
        affine_map.validate()
        return affine_map

    partition_numel = int(partition_numel)
    if partition_numel < 0 or partition_numel * len(holder_ranks) < numel:
        raise ValueError(f'{len(holder_ranks)} partitions of {partition_numel} elements cannot hold {numel} elements.')
    fragments = []
    for index, rank in enumerate(holder_ranks):
        start, length = _zero3_bounds(numel, partition_numel, index)
        fragments.append((rank, start, length))
    return _fragment_affine_map(numel, fragments)


def _zero3_local_numel(param):
    index = dist.get_rank(group=param.ds_process_group)
    _, length = _zero3_bounds(param.ds_numel, param.ds_tensor.ds_numel, index)
    return length


def _global_ranks(group):
    # dist.get_all_ranks_from_group never ends on the world group: torch maps any rank of the
    # world group to itself instead of raising past the end.
    return [dist.get_global_rank(group, rank) for rank in range(dist.get_world_size(group))]


def _check_zero3_param_storage(param):
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

    if hasattr(param.ds_tensor, 'ds_quant_scale'):
        raise NotImplementedError('Quantized ZeRO-3 parameter storage has no affine storage map.')
    # NVMe offload swaps a partition out of memory and leaves a placeholder in its place.
    if getattr(param.ds_tensor, 'final_location', None) == OffloadDeviceEnum.nvme:
        raise NotImplementedError('NVMe-offloaded ZeRO-3 parameter storage has no affine storage map.')


def zero_param_storage_affine_map(param, replica_group=None):
    """Describe the storage of a live low-precision ``param`` across global ranks.

    A ZeRO-3 parameter is described by its own partition group and ``ds_tensor`` size.
    Otherwise every rank of ``replica_group`` holds a whole copy. It defaults to the group ZeRO
    partitions the parameter's optimizer state over, which is the expert data-parallel group for
    MoE experts, and to the data-parallel group for a parameter outside a ZeRO optimizer. Under
    tensor parallelism the map describes this rank's tensor-parallel shard, which its
    data-parallel peers hold too.

    Quantized and NVMe-offloaded ZeRO-3 parameter storage is not supported.
    """
    if hasattr(param, 'ds_id'):
        _check_zero3_param_storage(param)
        holders = _global_ranks(param.ds_process_group)
        return zero_storage_affine_map(param.ds_numel, holders, param.ds_tensor.ds_numel)
    if replica_group is None:
        replica_group = getattr(param, '_dp_group', None)
    if replica_group is None:
        from deepspeed.utils import groups
        replica_group = groups._get_data_parallel_group()
    return zero_storage_affine_map(param.numel(), _global_ranks(replica_group))


def zero_param_storage_shard(param):
    """This rank's shard of ``param`` as laid out by ``zero_param_storage_affine_map``: a flat view.

    A ZeRO-3 parameter must be partitioned. A gathered copy, which ZeRO-3 keeps for persistent
    parameters between forwards, would go on serving the forward after the shard is written, so
    release gathered copies with ``engine.empty_partition_cache()`` first.
    """
    if hasattr(param, 'ds_id'):
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        _check_zero3_param_storage(param)
        if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
            raise RuntimeError(f'ZeRO-3 parameter {param.ds_id} is gathered, so its shard is not its only copy; '
                               'call engine.empty_partition_cache() first.')
        return param.ds_tensor.data.view(-1).narrow(0, 0, _zero3_local_numel(param))
    return param.data.view(-1)


def _zero3_optimizer_offloaded(param):
    return param._z3_optimizer.offload_optimizer


def zero_optimizer_storage_affine_maps(params):
    """Describe where ZeRO keeps the fp32 master copy and optimizer states of each of ``params``.

    Returns one map per parameter, in order, or ``None`` for a parameter without an fp32 copy,
    such as a frozen one. The master copy and every optimizer state of a parameter share its map.
    Under ZeRO-1/2 (and the bf16 optimizer) each rank of the parameter's data-parallel group holds
    one contiguous range of it, possibly empty, so the ranks exchange their ranges in one all-gather
    per group: every rank must pass the same parameters in the same order. Under ZeRO-3 the
    partitions follow the parameter's own, and no communication is needed.

    Offloaded optimizer states are not supported, since their shards would not be device tensors.
    """
    params = list(params)
    maps = [None] * len(params)
    # ZeRO-1/2 parameters grouped by the group their ranges are exchanged over, in first-seen order
    # so that every rank issues the all-gathers in the same order.
    by_group = {}
    for position, param in enumerate(params):
        if hasattr(param, 'ds_id'):
            # ZeRO-3 keeps no fp32 copy of a frozen parameter.
            if not hasattr(param, '_z3_optimizer') or not param.requires_grad:
                continue
            if _zero3_optimizer_offloaded(param):
                raise NotImplementedError('Offloaded ZeRO-3 optimizer states have no affine storage map.')
            maps[position] = zero_param_storage_affine_map(param)
        elif getattr(param, '_dp_group', None) is not None:
            by_group.setdefault(param._dp_group, []).append(position)

    device = torch.device(get_accelerator().current_device_name())
    for group, positions in by_group.items():
        # Each rank reports (start, length, offloaded) for every parameter; a rank without a range of a
        # parameter reports an empty one. Offload is reported rather than checked locally so that every
        # rank raises together instead of leaving its peers in the all-gather.
        local = torch.zeros(len(positions), 3, dtype=torch.int64, device=device)
        for row, position in enumerate(positions):
            mapping = params[position]._hp_mapping
            if mapping is not None:
                address = mapping.lp_fragment_address
                local[row] = torch.tensor([address.start, address.numel, int(mapping.use_offload)])
        world_size = dist.get_world_size(group=group)
        # Gloo wants the output stacked along the input's first axis, not along a new one.
        gathered = torch.empty(world_size * len(positions), 3, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(gathered, local, group=group)
        gathered = gathered.view(world_size, len(positions), 3).cpu()
        if bool(gathered[:, :, 2].any()):
            raise NotImplementedError('Offloaded ZeRO optimizer states have no affine storage map.')
        ranks = _global_ranks(group)
        for row, position in enumerate(positions):
            fragments = [(rank, int(gathered[index, row, 0]), int(gathered[index, row, 1]))
                         for index, rank in enumerate(ranks)]
            maps[position] = _fragment_affine_map(params[position].numel(), fragments)
    return maps


def zero_optimizer_storage_shard(param, optim_state_key=None):
    """This rank's shard of the fp32 master copy of ``param``, or of its ``optim_state_key`` state.

    The shard is a flat view of ZeRO's storage, laid out as ``zero_optimizer_storage_affine_maps``
    describes it, so writing it updates the optimizer. Returns ``None`` where that has no map.
    """
    if hasattr(param, 'ds_id'):
        if not hasattr(param, '_z3_optimizer') or not param.requires_grad:
            return None
        if _zero3_optimizer_offloaded(param):
            raise NotImplementedError('Offloaded ZeRO-3 optimizer states have no affine storage map.')
        shard = param._z3_optimizer.get_local_fp32_param(param, optim_state_key)
        return shard.view(-1).narrow(0, 0, _zero3_local_numel(param))

    if getattr(param, '_dp_group', None) is None:
        return None
    mapping = param._hp_mapping
    if mapping is None:
        # The rank holds no range of this parameter.
        return torch.empty(0, dtype=torch.float32, device=get_accelerator().current_device_name())
    if mapping.use_offload:
        raise NotImplementedError('Offloaded ZeRO optimizer states have no affine storage map.')
    if optim_state_key is not None and mapping.optim_fragment is None:
        raise RuntimeError('ZeRO links optimizer states to parameters at the first optimizer step; '
                           f'call this after it to read {optim_state_key!r}.')
    return mapping.get_hp_fragment(optim_state_key)


def _autotp_layout_map(name, param, tp_world_size):
    from deepspeed.module_inject.layers import _get_param_uc_conversion_meta

    meta = _get_param_uc_conversion_meta(param)
    if not meta:
        # AutoTP leaves such a parameter whole on every tensor-parallel rank.
        return replicated_map(tuple(getattr(param, 'ds_shape', param.shape)), tp_world_size)
    if meta.get('unsupported_reason'):
        raise NotImplementedError(f"AutoTP parameter '{name}' has no affine layout: {meta['unsupported_reason']}")
    if meta.get('affine_map') is None:
        raise NotImplementedError(f"AutoTP parameter '{name}' publishes no affine layout map.")
    return ParamAffineMap.from_dict(meta['affine_map'])


def autotp_storage_affine_maps(model, optimizer_states=False, tp_group=None):
    """Describe what every process stores of each parameter of an AutoTP ``model`` under ZeRO.

    Returns ``{name: map}`` over ``model.named_parameters()``. Each map nests the parameter's
    AutoTP layout map with the ZeRO storage of every tensor-parallel shard, so it describes the
    parameter in its own shape keyed by global rank, and each rank's shard is the one
    ``zero_param_storage_shard`` returns. With ``optimizer_states`` the maps describe the fp32
    master copy and optimizer states instead, matching ``zero_optimizer_storage_shard``, and a
    parameter without them maps to ``None``. A parameter AutoTP did not partition is replicated
    over the tensor-parallel group, unless it kept the partition over the whole world that
    ``zero.Init`` gave it, which describes it on its own.

    Collective: the tensor-parallel ranks exchange their shards' storage maps in one all-gather
    over ``tp_group`` (the AutoTP group by default), and every rank must call this with the same
    model. A rank's storage of a layout piece that is dense in its shard is at most
    ``2 * ndim - 1`` boxes; any other layout piece keeps one piece per row.
    """
    if tp_group is None:
        from deepspeed.utils import groups
        tp_group = groups.get_tensor_model_parallel_group()
    tp_world_size = dist.get_world_size(group=tp_group)

    # Some AutoTP layers publish their metadata only when asked.
    for module in model.modules():
        marker = getattr(module, '_mark_uc_metadata', None)
        if marker is not None:
            marker()

    named_params = list(model.named_parameters())
    params = [param for _, param in named_params]
    if optimizer_states:
        storage_maps = zero_optimizer_storage_affine_maps(params)
    else:
        storage_maps = [zero_param_storage_affine_map(param) for param in params]
    local = [None if storage_map is None else storage_map.to_dict() for storage_map in storage_maps]
    gathered = [None] * tp_world_size
    dist.all_gather_object(gathered, local, group=tp_group)

    maps = {}
    for position, (name, param) in enumerate(named_params):
        storage_by_tp_rank = [rank_maps[position] for rank_maps in gathered]
        if all(storage is None for storage in storage_by_tp_rank):
            maps[name] = None
            continue
        if any(storage is None for storage in storage_by_tp_rank):
            raise ValueError(f"Parameter '{name}' has storage on some tensor-parallel ranks but not on others.")
        layout = _autotp_layout_map(name, param, tp_world_size)
        inner = {tp_rank: ParamAffineMap.from_dict(storage) for tp_rank, storage in enumerate(storage_by_tp_rank)}
        replicated = layout.to_dict() == replicated_map(layout.logical_shape, tp_world_size).to_dict()
        shared = all(storage == storage_by_tp_rank[0] for storage in storage_by_tp_rank)
        if tp_world_size > 1 and replicated and shared:
            # A parameter AutoTP leaves whole in a model built under zero.Init keeps its partition over
            # the whole world, so every tensor-parallel rank reports the storage of the same one copy.
            layout = replicated_map(layout.logical_shape, 1)
            inner = {0: inner[0]}
        maps[name] = nest_affine_maps(layout, inner)
    return maps
