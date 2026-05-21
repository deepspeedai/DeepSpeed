# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist

GROUP_REGISTRY = {}  # int -> dist.ProcessGroup


def register_groups(groups):
    """groups: List[List[int]], e.g. [[0,1],[2,3]]"""
    for gid, ranks in enumerate(groups):
        if gid not in GROUP_REGISTRY:
            GROUP_REGISTRY[gid] = dist.new_group(ranks)


def get_group(gid: int):
    return GROUP_REGISTRY[gid] if gid is not None else dist.get_world_group()


def get_registry():
    return GROUP_REGISTRY


def is_setup():
    return GROUP_REGISTRY['is_reg'] if 'is_reg' in GROUP_REGISTRY else False


def extract_mesh_size(param_dict):
    sp_size = param_dict.get('sequence_parallel_size', 1)
    assert dist.get_world_size() % sp_size == 0, 'World mesh-size should be divisible by SP_SIZE'
    dp_size = dist.get_world_size() // sp_size

    return sp_size, dp_size


def sp_size():
    assert 'SP_SIZE' in GROUP_REGISTRY, 'SP_SIZE not init properly.'

    return GROUP_REGISTRY['SP_SIZE']


def dp_size():
    assert 'DP_SIZE' in GROUP_REGISTRY, 'DP_SIZE not init properly'

    return GROUP_REGISTRY['DP_SIZE']


def populate_registry(SP_SIZE, DP_SIZE):
    """ Populate rank to SP/DP mesh index.  """

    if GROUP_REGISTRY.get('is_reg', False):
        return

    group_listing = []
    offset = 0
    for _ in range(DP_SIZE):
        group_listing.append([i + offset for i in range(SP_SIZE)])
        offset += SP_SIZE

    register_groups(group_listing)

    ## Extraneous metadata required for proper instatiation. ##
    GROUP_REGISTRY['SP_SIZE'] = SP_SIZE
    GROUP_REGISTRY['DP_SIZE'] = DP_SIZE
    GROUP_REGISTRY['is_reg'] = True
