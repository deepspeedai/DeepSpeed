# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""AutoEP expert placement descriptors and their affine lowering.

The descriptor records placement in EP-local rank coordinates, and
``autoep_placement_to_affine_map`` describes one logical, packed expert tensor
per EP rank. ``autoep_storage_affine_map`` adds the expert-data-parallel and
ZeRO layer on top: it describes what each global rank actually stores.
"""

from deepspeed.checkpoint.affine import AFFINE_MAP_FORMAT_VERSION, AffinePiece, ParamAffineMap, nest_affine_maps
from deepspeed.checkpoint.affine_transfer import plan_transfer
from deepspeed.checkpoint.zero_affine import zero_storage_affine_map
from deepspeed.checkpoint.constants import (AFFINE_MAP, AFFINE_MAP_PARAMS, AFFINE_MAP_VERSION, AUTOEP_AFFINE_MAPS,
                                            AUTOEP_EXPERT_PLACEMENT, AUTOEP_PLACEMENT_EP_SIZE,
                                            AUTOEP_PLACEMENT_EXPERTS, AUTOEP_PLACEMENT_NUM_EXPERTS,
                                            AUTOEP_PLACEMENT_RANK, AUTOEP_PLACEMENT_RANKS, AUTOEP_PLACEMENT_VERSION,
                                            AUTOEP_PLACEMENT_VERSION_KEY)

__all__ = [
    'AUTOEP_PLACEMENT_VERSION',
    'make_autoep_placement_descriptor',
    'validate_autoep_placement_descriptor',
    'autoep_experts_for_rank',
    'legacy_uniform_autoep_placement_descriptor',
    'autoep_placements_equal',
    'autoep_placement_to_affine_map',
    'autoep_storage_affine_map',
    'autoep_metadata_to_affine_map',
    'extract_autoep_rank_tensor',
    'plan_autoep_placement_transfer',
]


def make_autoep_placement_descriptor(num_experts, experts_by_rank):
    """Build a versioned descriptor from expert IDs in rank-local packed order."""
    ranks = [{
        AUTOEP_PLACEMENT_RANK: rank,
        AUTOEP_PLACEMENT_EXPERTS: list(experts),
    } for rank, experts in enumerate(experts_by_rank)]
    descriptor = {
        AUTOEP_PLACEMENT_VERSION_KEY: AUTOEP_PLACEMENT_VERSION,
        AUTOEP_PLACEMENT_NUM_EXPERTS: num_experts,
        AUTOEP_PLACEMENT_EP_SIZE: len(ranks),
        AUTOEP_PLACEMENT_RANKS: ranks,
    }
    validate_autoep_placement_descriptor(descriptor)
    return descriptor


def validate_autoep_placement_descriptor(descriptor):
    """Validate a plain scalar/list/dict AutoEP placement descriptor."""
    if not isinstance(descriptor, dict):
        raise TypeError('AutoEP placement descriptor must be a dict.')

    version = descriptor.get(AUTOEP_PLACEMENT_VERSION_KEY)
    if version != AUTOEP_PLACEMENT_VERSION:
        raise ValueError(f'Unsupported AutoEP placement descriptor version {version!r}; '
                         f'expected {AUTOEP_PLACEMENT_VERSION}.')

    num_experts = _positive_int(descriptor.get(AUTOEP_PLACEMENT_NUM_EXPERTS), AUTOEP_PLACEMENT_NUM_EXPERTS)
    ep_size = _positive_int(descriptor.get(AUTOEP_PLACEMENT_EP_SIZE), AUTOEP_PLACEMENT_EP_SIZE)
    ranks = descriptor.get(AUTOEP_PLACEMENT_RANKS)
    if not isinstance(ranks, list):
        raise TypeError(f'{AUTOEP_PLACEMENT_RANKS} must be a list.')
    if len(ranks) != ep_size:
        raise ValueError(f'AutoEP placement descriptor has {len(ranks)} rank entries; expected exactly {ep_size}.')

    seen_ranks = set()
    covered_experts = set()
    for entry in ranks:
        if not isinstance(entry, dict):
            raise TypeError('Each AutoEP placement rank entry must be a dict.')
        rank = entry.get(AUTOEP_PLACEMENT_RANK)
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < ep_size:
            raise ValueError(f'AutoEP placement rank {rank!r} is outside [0, {ep_size}).')
        if rank in seen_ranks:
            raise ValueError(f'AutoEP placement rank {rank} appears more than once.')
        seen_ranks.add(rank)

        experts = entry.get(AUTOEP_PLACEMENT_EXPERTS)
        if not isinstance(experts, list):
            raise TypeError(f'Experts for AutoEP placement rank {rank} must be a list defining local packed order.')
        local_experts = set()
        for expert_id in experts:
            if not isinstance(expert_id, int) or isinstance(expert_id, bool) or not 0 <= expert_id < num_experts:
                raise ValueError(f'AutoEP placement expert ID {expert_id!r} on rank {rank} is outside '
                                 f'[0, {num_experts}).')
            if expert_id in local_experts:
                raise ValueError(f'AutoEP placement rank {rank} contains duplicate expert ID {expert_id}.')
            local_experts.add(expert_id)
            covered_experts.add(expert_id)

    missing = sorted(set(range(num_experts)) - covered_experts)
    if missing:
        raise ValueError(f'AutoEP placement descriptor does not cover global expert IDs {missing}.')


def autoep_experts_for_rank(descriptor, rank):
    """Return expert IDs in one rank's local packed order."""
    validate_autoep_placement_descriptor(descriptor)
    entries_by_rank = {entry[AUTOEP_PLACEMENT_RANK]: entry for entry in descriptor[AUTOEP_PLACEMENT_RANKS]}
    if not isinstance(rank, int) or isinstance(rank, bool) or rank not in entries_by_rank:
        raise ValueError(f'AutoEP placement rank {rank!r} is outside [0, {len(entries_by_rank)}).')
    return entries_by_rank[rank][AUTOEP_PLACEMENT_EXPERTS]


def legacy_uniform_autoep_placement_descriptor(num_experts, num_local_experts, ep_size):
    """Synthesize the legacy contiguous, uniform AutoEP placement."""
    num_experts = _positive_int(num_experts, 'num_experts')
    num_local_experts = _positive_int(num_local_experts, 'num_local_experts')
    ep_size = _positive_int(ep_size, 'ep_size')
    if num_local_experts * ep_size != num_experts:
        raise ValueError(f'Inconsistent legacy AutoEP metadata: num_local_experts ({num_local_experts}) * '
                         f'ep_size ({ep_size}) != num_experts ({num_experts}).')
    experts_by_rank = []
    for rank in range(ep_size):
        start = rank * num_local_experts
        experts_by_rank.append(list(range(start, start + num_local_experts)))
    return make_autoep_placement_descriptor(num_experts, experts_by_rank)


def autoep_placements_equal(descriptor, other):
    """Return whether two descriptors place the same experts, in the same order, on each rank ID.

    Rank entries are identified by their explicit rank IDs, so listing them in another order does
    not change the placement.
    """
    validate_autoep_placement_descriptor(descriptor)
    validate_autoep_placement_descriptor(other)
    if descriptor[AUTOEP_PLACEMENT_NUM_EXPERTS] != other[AUTOEP_PLACEMENT_NUM_EXPERTS]:
        return False
    if descriptor[AUTOEP_PLACEMENT_EP_SIZE] != other[AUTOEP_PLACEMENT_EP_SIZE]:
        return False
    for rank in range(descriptor[AUTOEP_PLACEMENT_EP_SIZE]):
        if autoep_experts_for_rank(descriptor, rank) != autoep_experts_for_rank(other, rank):
            return False
    return True


def autoep_placement_to_affine_map(descriptor, logical_shape):
    """Lower one ``[num_experts, ...]`` parameter to a :class:`ParamAffineMap`."""
    validate_autoep_placement_descriptor(descriptor)
    logical_shape = tuple(int(dim) for dim in logical_shape)
    num_experts = descriptor[AUTOEP_PLACEMENT_NUM_EXPERTS]
    if not logical_shape or logical_shape[0] != num_experts:
        raise ValueError(f'Expert parameter logical shape must start with num_experts ({num_experts}); '
                         f'got {logical_shape}.')
    if any(dim < 0 for dim in logical_shape):
        raise ValueError(f'Expert parameter logical shape cannot contain negative dimensions: {logical_shape}.')

    entries_by_rank = {entry[AUTOEP_PLACEMENT_RANK]: entry for entry in descriptor[AUTOEP_PLACEMENT_RANKS]}
    holders = _expert_holders(entries_by_rank, num_experts)
    source_strides = _row_major_strides(logical_shape)
    expert_shape = logical_shape[1:]
    expert_numel = _product(expert_shape)
    pieces_by_rank = {}
    shard_shapes = {}

    for rank in range(descriptor[AUTOEP_PLACEMENT_EP_SIZE]):
        experts = entries_by_rank[rank][AUTOEP_PLACEMENT_EXPERTS]
        shard_shape = (len(experts), ) + expert_shape
        shard_shapes[rank] = shard_shape
        dest_strides = _row_major_strides(shard_shape)
        pieces_by_rank[rank] = _pieces_for_rank(experts, holders, expert_shape, expert_numel, source_strides,
                                                dest_strides)

    affine_map = ParamAffineMap(logical_shape=logical_shape, shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)
    # Descriptor validation already proves expert-level coverage. Expanding that
    # check to every tensor element is prohibitively expensive for expert weights.
    affine_map.validate()
    return affine_map


def autoep_storage_affine_map(descriptor, logical_shape, holders_by_ep_rank, partition_numel_by_ep_rank=None):
    """Lower one expert parameter to the storage each global rank actually holds.

    ``holders_by_ep_rank`` maps every rank ID of ``descriptor`` to the global ranks that store that
    EP rank's packed ``[local_experts, ...]`` tensor, in expert-data-parallel (EDP) rank order.

    Without ``partition_numel_by_ep_rank``, every holder stores the whole packed tensor, as EDP
    replicas do under ZeRO-0/1/2. With it, holder ``d`` of EP rank ``r`` stores elements
    ``[d * n, (d + 1) * n)`` of the flattened packed tensor, where ``n`` is
    ``partition_numel_by_ep_rank[r]``, as ZeRO-3 does. ZeRO-3's trailing padding holds no
    parameter elements and is not part of the shard, so a holder whose partition is all padding,
    like every holder of an EP rank without experts, has an empty shard.

    The returned map describes the parameter in its own shape and is keyed by global rank; each
    shard is the holder's flat storage. It nests the placement's layout map with ZeRO's
    storage map of each packed tensor, so every holder of an element, across EDP replicas and
    redundant experts, is listed in the piece's ``locations``; choosing among them is left to the
    transfer planner's replica selection.
    """
    layout = autoep_placement_to_affine_map(descriptor, logical_shape)
    entries_by_rank = {entry[AUTOEP_PLACEMENT_RANK]: entry for entry in descriptor[AUTOEP_PLACEMENT_RANKS]}
    holders_by_ep_rank = _validated_storage_holders(holders_by_ep_rank, entries_by_rank)

    storage_by_ep_rank = {}
    for ep_rank, global_ranks in holders_by_ep_rank.items():
        packed_numel = _product(layout.shard_shapes[ep_rank])
        partition_numel = None
        if partition_numel_by_ep_rank is not None:
            partition_numel = _storage_partition_numel(partition_numel_by_ep_rank, ep_rank, len(global_ranks),
                                                       packed_numel)
        storage_by_ep_rank[ep_rank] = zero_storage_affine_map(packed_numel, global_ranks, partition_numel)
    return nest_affine_maps(layout, storage_by_ep_rank)


def autoep_metadata_to_affine_map(metadata, logical_shape, param_name=None):
    """Load a persisted AutoEP affine map, falling back to legacy placement lowering."""
    serialized_map = metadata.get(AFFINE_MAP)
    if serialized_map is None and param_name is not None:
        affine_maps = metadata.get(AUTOEP_AFFINE_MAPS, {})
        if affine_maps:
            if affine_maps.get(AFFINE_MAP_VERSION) != AFFINE_MAP_FORMAT_VERSION:
                raise ValueError(f"Unsupported AutoEP affine map format version "
                                 f"{affine_maps.get(AFFINE_MAP_VERSION)!r}.")
            serialized_map = affine_maps.get(AFFINE_MAP_PARAMS, {}).get(param_name)

    if serialized_map is not None:
        affine_map = ParamAffineMap.from_dict(serialized_map)
        expected_shape = tuple(int(dim) for dim in logical_shape)
        if affine_map.logical_shape != expected_shape:
            raise ValueError(f"AutoEP affine map logical shape {affine_map.logical_shape} does not match "
                             f"parameter logical shape {expected_shape}.")
        affine_map.validate()
        placement = metadata.get(AUTOEP_EXPERT_PLACEMENT)
        if placement is not None:
            expected_map = autoep_placement_to_affine_map(placement, expected_shape)
            if affine_map.to_dict() != expected_map.to_dict():
                raise ValueError("AutoEP affine map disagrees with the expert placement descriptor.")
        return affine_map

    placement = metadata.get(AUTOEP_EXPERT_PLACEMENT)
    if placement is None:
        raise ValueError("AutoEP metadata has neither an affine map nor an expert placement descriptor.")
    return autoep_placement_to_affine_map(placement, logical_shape)


def extract_autoep_rank_tensor(full_param, target_map, ep_rank):
    """Extract one EP rank's packed local tensor from a universal full tensor."""
    if not isinstance(target_map, ParamAffineMap):
        raise TypeError('target_map must be a ParamAffineMap.')
    if ep_rank not in target_map.shard_shapes:
        raise ValueError(f'EP rank {ep_rank} is not present in the target affine map.')
    return target_map.extract(full_param, ep_rank)


def plan_autoep_placement_transfer(source_placement,
                                   target_placement,
                                   logical_shape,
                                   limits=None,
                                   *,
                                   replica_selector=None):
    """Plan copies that move one expert parameter from ``source_placement`` to ``target_placement``.

    Both placements are lowered with :func:`autoep_placement_to_affine_map`, so plan ranks are
    EP-local ranks of the respective placements, and offsets follow each rank's local packed order.
    The plan only describes the copies; it does not require that the AutoEP runtime can execute
    with ``target_placement``.
    """
    source_map = autoep_placement_to_affine_map(source_placement, logical_shape)
    target_map = autoep_placement_to_affine_map(target_placement, logical_shape)
    return plan_transfer(target_map, source_map, limits, replica_selector=replica_selector)


def _pieces_for_rank(experts, holders, expert_shape, expert_numel, source_strides, dest_strides):
    pieces = []
    run_start = 0
    for local_index in range(1, len(experts) + 1):
        at_end = local_index == len(experts)
        if not at_end:
            previous_expert = experts[local_index - 1]
            current_expert = experts[local_index]
            homogeneous = (current_expert == previous_expert + 1
                           and holders[current_expert] == holders[previous_expert]
                           and len(holders[current_expert]) == 1)
        if at_end or not homogeneous:
            first_expert = experts[run_start]
            run_length = local_index - run_start
            pieces.append(
                AffinePiece(shape=(run_length, ) + expert_shape,
                            source_offset=first_expert * expert_numel,
                            source_strides=source_strides,
                            dest_offset=run_start * expert_numel,
                            dest_strides=dest_strides,
                            locations=holders[first_expert]))
            run_start = local_index
    return pieces


def _validated_storage_holders(holders_by_ep_rank, entries_by_rank):
    if not isinstance(holders_by_ep_rank, dict):
        raise TypeError('holders_by_ep_rank must map each EP rank ID to its global holder ranks.')
    if set(holders_by_ep_rank) != set(entries_by_rank):
        raise ValueError(f'holders_by_ep_rank names EP ranks {sorted(holders_by_ep_rank)}; the placement '
                         f'descriptor names {sorted(entries_by_rank)}.')
    seen = set()
    validated = {}
    for ep_rank in sorted(holders_by_ep_rank):
        global_ranks = list(holders_by_ep_rank[ep_rank])
        if not global_ranks:
            raise ValueError(f'EP rank {ep_rank} has no global holder rank.')
        for global_rank in global_ranks:
            if not isinstance(global_rank, int) or isinstance(global_rank, bool) or global_rank < 0:
                raise ValueError(f'Global holder rank {global_rank!r} of EP rank {ep_rank} is not a '
                                 'non-negative integer.')
            if global_rank in seen:
                raise ValueError(f'Global rank {global_rank} holds storage for more than one EP rank entry.')
            seen.add(global_rank)
        validated[ep_rank] = global_ranks
    return validated


def _storage_partition_numel(partition_numel_by_ep_rank, ep_rank, num_holders, packed_numel):
    partition_numel = partition_numel_by_ep_rank.get(ep_rank)
    if not isinstance(partition_numel, int) or isinstance(partition_numel, bool) or partition_numel < 0:
        raise ValueError(f'Partition size {partition_numel!r} of EP rank {ep_rank} is not a non-negative integer.')
    if partition_numel * num_holders < packed_numel:
        raise ValueError(f'EP rank {ep_rank} packs {packed_numel} elements, but its {num_holders} holders of '
                         f'{partition_numel} elements each cannot store them.')
    return partition_numel


def _expert_holders(entries_by_rank, num_experts):
    holders = {expert_id: set() for expert_id in range(num_experts)}
    for rank, entry in entries_by_rank.items():
        for expert_id in entry[AUTOEP_PLACEMENT_EXPERTS]:
            holders[expert_id].add(rank)
    return holders


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer; got {value!r}.')
    return value


def _product(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def _row_major_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)
