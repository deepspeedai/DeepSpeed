# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


def _log_rank0(msg):
    if dist.get_rank() == 0:
        logger.info(msg)


@torch.jit.script
def scale_tensors(tensors: List[Tensor], scale: int):
    for t in tensors:
        t.div_(scale)


@dataclass
class MiCS_CommGroups:
    """"""
    param_shard_group = None
    param_shard_size = -1
    param_shard_rank = -1

    param_repli_group = None
    param_repli_size = -1
    param_repli_rank = -1

    param_intra_node_group = None
    param_inter_node_shard_group = None


def create_mics_comm_groups(
    shard_size,
    dp_group,
    hierarchical_allgather=False,
    mpu=None,
):
    """
    create shard-group, replicate-group from config_file
    TODO: consider broadcast the config from rank0

    Returns:
        MiCS_CommGroups
    """
    # env var for debugging purpose
    ndevices_per_node = int(os.environ.get("NDEV_PER_NODE", get_accelerator().device_count()))
    _log_rank0(f'creating MiCS communication groups with per node device size {ndevices_per_node}')
    groups = MiCS_CommGroups()

    if mpu is not None:
        assert dp_group == mpu.get_data_parallel_group()

    # full size of the world
    world_size = dist.get_world_size()
    # global rank
    global_rank = dist.get_rank()

    config = _generate_mics_config(world_size, ndevices_per_node, shard_size, 1)
    ranks_of_shard_group = config['shard_groups']
    ranks_of_repli_group = config['replicate_groups']
    if len(ranks_of_repli_group) == 0:
        assert len(ranks_of_shard_group) == 1, "replicate groups are empty only for single shard group"
        for r in ranks_of_shard_group[0]:
            ranks_of_repli_group.append([r])

    # for simplicity
    assert _sizes_all_same(ranks_of_repli_group), "replicate groups must have the same size"
    assert _sizes_all_same(ranks_of_shard_group), "shard groups must have the same size"

    assert sum([len(g) for g in ranks_of_shard_group]) == dist.get_world_size(), "all sharded ranks "
    if len(ranks_of_shard_group) > 1:  # if only shard on one group then no need for replicate groups
        assert len(ranks_of_shard_group) == len(
            ranks_of_repli_group[0]), "number of shard groups must equal to the size of each replicate group"

    global_rank = dist.get_rank()
    # create shard groups
    for shard_ranks in ranks_of_shard_group:
        _group = dist.new_group(shard_ranks)
        if global_rank in shard_ranks:
            groups.param_shard_group = _group
            groups.param_shard_size = len(shard_ranks)
            groups.param_shard_rank = dist.get_rank(_group)
            logger.info(f'rank {global_rank}, shard group'
                        f' {groups.param_shard_rank}/{dist.get_world_size(group=_group)}')

    # create replicate groups
    for repli_ranks in ranks_of_repli_group:
        if len(repli_ranks) > 1:
            _group = dist.new_group(repli_ranks)
            if global_rank in repli_ranks:
                groups.param_repli_group = _group
                groups.param_repli_size = len(repli_ranks)
                groups.param_repli_rank = dist.get_rank(group=_group)
                logger.info(f'rank {global_rank} '
                            f'replicate group {groups.param_repli_rank}/{dist.get_world_size(group=_group)}')
        else:
            groups.param_repli_group = None
            groups.param_repli_size = 1
            groups.param_repli_rank = 0
            logger.info(f'rank {global_rank} replicate group 0/1')

    # assign shard group size as world size
    assert groups.param_shard_size == len(ranks_of_shard_group[0])

    if hierarchical_allgather:
        # create hierarchy inter-node, intra-node groups
        # n_span_nodes = config['shard_span']
        n_span_nodes = config['span_nodes']
        assert n_span_nodes > 1, "sharding spans on single node, no need for hierarchy allgather"
        assert len(ranks_of_shard_group[0]) % n_span_nodes == 0

        n_gpu_per_node = len(ranks_of_shard_group[0]) // n_span_nodes
        intra_node_ranks_group = []
        inter_node_ranks_group = []
        for shard_group in ranks_of_shard_group:
            _intra_node_ranks = []
            for i in range(0, len(shard_group), n_gpu_per_node):
                _intra_node_ranks.append(shard_group[i:i + n_gpu_per_node])
            _inter_node_ranks = []
            for i in range(n_gpu_per_node):
                _ranks = [_g[i] for _g in _intra_node_ranks]
                _inter_node_ranks.append(_ranks)

            intra_node_ranks_group.append(_intra_node_ranks)
            inter_node_ranks_group.append(_inter_node_ranks)

        _log_rank0(f"create for hierarchy all-gather groups: intra nodes {intra_node_ranks_group}")
        _log_rank0(f"create for hierarchy all-gather groups: inter nodes {inter_node_ranks_group}")

        # create communicators
        for shard_group in intra_node_ranks_group:
            for intra_node_ranks in shard_group:
                _group = dist.new_group(intra_node_ranks)
                if global_rank in intra_node_ranks:
                    groups.param_intra_node_group = _group
                _log_rank0(f'create group for intra node ranks {intra_node_ranks}')

        for shard_group in inter_node_ranks_group:
            for inter_node_ranks in shard_group:
                _group = dist.new_group(inter_node_ranks)
                if global_rank in inter_node_ranks:
                    groups.param_inter_node_shard_group = _group
                _log_rank0(f'create group for inter node ranks {inter_node_ranks}')
    return groups


def _generate_mics_config(world_size, ndev_per_node, shard_size, pp_size=1):
    """Generating the configuration for sharding This shard config generation assume
    that the pipeline stages are partitioned in order, i.e., first ranks
    hold the stage0, etc.

    Args:

        shard_size (int): zero3 data-parallel shard size, FIXME:
        change the name later

        pp_size (int): pipeline parallel size, currently, only work with
        pipeline parallelism + zero

    """
    assert world_size % pp_size == 0
    assert (world_size // pp_size) % shard_size == 0, \
        f"dp group size is not dividable by dp_shard_size, "\
        f" (world_size {world_size}, pp_size {pp_size}, dp_shard_size {shard_size})"

    config = {}
    shard_groups = np.arange(world_size).reshape(-1, shard_size)
    replicate_groups = []
    for i in range(shard_size):
        same_shard_ranks = shard_groups[:, i].tolist()
        n_ranks = len(same_shard_ranks)
        replicate_size = n_ranks // pp_size
        replicate_groups.extend([same_shard_ranks[j:j + replicate_size] for j in range(0, n_ranks, replicate_size)])

    config['replicate_groups'] = replicate_groups
    config['shard_groups'] = shard_groups.tolist()
    config["span_nodes"] = len(shard_groups[0]) // ndev_per_node
    return config


def _sizes_all_same(groups):
    """all groups have same length"""
    all_same = True
    for g in groups:
        if len(g) != len(groups[0]):
            return False
    return all_same
