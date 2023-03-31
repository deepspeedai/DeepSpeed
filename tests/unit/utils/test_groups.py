# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.utils.groups import _get_expert_parallel_ranks


def test_get_expert_parallel_ranks():
    """
    Example - E + M + D parallel
    world_size = 16
    model_degree = 2
    expert_degree = 4 # number of experts in same group
    mp_group = [0, 1], [2,3], [4,5] ...
    data_parallel_group =[0,2,4,6,8,10, 12,14],                 [1,3,5,7,9,11,13,15]
    expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
    expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[7,15]
    """
    expert_parallel_groups, expert_data_parallel_groups = _get_expert_parallel_ranks(world_size=16,
                                                                                     model_parallel_size_=2,
                                                                                     expert_parallel_size_=4)
    assert expert_parallel_groups == [
        [0, 2, 4, 6],
        [8, 10, 12, 14],
        [1, 3, 5, 7],
        [9, 11, 13, 15],
    ]
    assert expert_data_parallel_groups == [
        [0, 8],
        [2, 10],
        [4, 12],
        [6, 14],
        [1, 9],
        [3, 11],
        [5, 13],
        [7, 15],
    ]
