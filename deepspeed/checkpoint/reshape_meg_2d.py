# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .reshape_utils import partition_data


class meg_2d_parallel_map(object):

    def __init__(self, pp_degree, tp_degree):
        self.pp_degree = pp_degree
        self.tp_degree = tp_degree
        self.map = {}

    def simple_init(self):
        self.map = {
            self._make_key(i // self.tp_degree, i % self.tp_degree): [i]
            for i in range(self.pp_degree * self.tp_degree)
        }

    def add_data(self, pp_index, tp_index, data):
        self._validate_indices(pp_index, tp_index)
        assert type(data) is list

        key = self._make_key(pp_index, tp_index)
        if not key in self.map.keys():
            self.map[key] = []
        self.map[key] += data

    def get_data(self, pp_index=None, tp_index=None):
        self._validate_indices(pp_index, tp_index)
        pp_indices = list(range(self.pp_degree)) if pp_index is None else [pp_index]
        tp_indices = list(range(self.tp_degree)) if tp_index is None else [tp_index]

        result = []
        for i in pp_indices:
            for j in tp_indices:
                result += self.map[self._make_key(i, j)]

        return result

    def print_data(self, tag):
        print(f'{tag}')
        for key, value in self.map.items():
            print(f'{key} = {value}')

    def _validate_indices(self, pp_index, tp_index):
        assert pp_index is None or pp_index < self.pp_degree
        assert tp_index is None or tp_index < self.tp_degree

    def _make_key(self, i, j):
        return f'{i},{j}'


def _reshape_tp_dimension(old_2d_map, new_tp_degree):
    old_pp_degree = old_2d_map.pp_degree
    new_2d_map = meg_2d_parallel_map(old_pp_degree, new_tp_degree)
    for i in range(old_pp_degree):
        ranks_for_pp_index = old_2d_map.get_data(pp_index=i, tp_index=None)
        split_ranks = partition_data(ranks_for_pp_index, new_tp_degree)
        for j in range(new_tp_degree):
            new_2d_map.add_data(i, j, split_ranks[j])

    return new_2d_map


def _reshape_pp_dimension(old_2d_map, new_pp_degree):
    old_tp_degree = old_2d_map.tp_degree
    new_2d_map = meg_2d_parallel_map(new_pp_degree, old_tp_degree)
    for i in range(old_tp_degree):
        ranks_for_tp_index = old_2d_map.get_data(pp_index=None, tp_index=i)
        split_ranks = partition_data(ranks_for_tp_index, new_pp_degree)
        for j in range(new_pp_degree):
            new_2d_map.add_data(j, i, split_ranks[j])

    return new_2d_map


def reshape_meg_2d_parallel(old_pp_degree, old_tp_degree, new_pp_degree, new_tp_degree, verbose=False):
    assert new_pp_degree <= old_pp_degree
    assert new_tp_degree <= old_tp_degree

    old_2d_map = meg_2d_parallel_map(old_pp_degree, old_tp_degree)
    old_2d_map.simple_init()
    if verbose:
        old_2d_map.print_data(f'original_2d_map:')

    if old_tp_degree != new_tp_degree:
        new_tp_map = _reshape_tp_dimension(old_2d_map, new_tp_degree)
    else:
        new_tp_map = old_2d_map
    if verbose:
        new_tp_map.print_data(f'after_tp_reshape:')

    if old_pp_degree != new_pp_degree:
        final_map = _reshape_pp_dimension(new_tp_map, new_pp_degree)
    else:
        final_map = new_tp_map

    if verbose:
        final_map.print_data(f'final_2d_map:')

    return final_map


def get_mpu_ranks(tp_size=1, pp_size=1, dp_size=1, virtual_pp_size=None):
    """
    Initialize model data parallel groups.

    Arguments:
        tp_size: number of GPUs used to parallelize model tensor.
        pp_size: number of GPUs used to parallelize model pipeline.
        dp_size: number of GPUs used to parallelize model data.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """

    world_size = tp_size * pp_size * dp_size

    print(f"\n\n*** tp={tp_size}, pp={pp_size}, dp={dp_size}, world={world_size}")

    tensor_model_parallel_size = min(tp_size, world_size)
    pipeline_model_parallel_size = min(pp_size, world_size)
    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size

    # Build the data-parallel groups.
    all_dp_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_dp_group_ranks.append(list(ranks))

    print("DP", all_dp_group_ranks)

    # Build the model-parallel groups.
    all_pp_group_ranks = []
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i] for data_parallel_group_ranks in all_dp_group_ranks]
        all_pp_group_ranks.append(list(ranks))

    print(f"PP", all_pp_group_ranks)

    # Build the tensor model-parallel groups.
    all_tp_group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        all_tp_group_ranks.append(list(ranks))

    print(f"TP", all_tp_group_ranks)

    return all_tp_group_ranks, all_pp_group_ranks, all_dp_group_ranks

    # # Build the pipeline model-parallel groups and embedding groups
    # # (first and last rank in each pipeline model-parallel group).
    # for i in range(num_pipeline_model_parallel_groups):
    #     ranks = range(i, world_size,
    #                   num_pipeline_model_parallel_groups)
    #     print(f"EMB{i}", list(ranks))


def reshape(src, tgt):
    """
    reshape([tp_size_src, pp_size_src, dp_size_src],
            [tp_size_tgt, pp_size_tgt, dp_size_tgt])
    """

    print(f"\n\n*** Reshaping: {src} => {tgt}")

    tp_size_src, pp_size_src, dp_size_src = src
    tp_size_tgt, pp_size_tgt, dp_size_tgt = tgt

    tp_ranks1, pp_ranks1, dp_ranks1 = get_mpu_ranks(tp_size=tp_size_src, pp_size=pp_size_src, dp_size=dp_size_src)
    tp_ranks2, pp_ranks2, dp_ranks2 = get_mpu_ranks(tp_size=tp_size_tgt, pp_size=pp_size_src, dp_size=dp_size_src)
    tp_ranks3, pp_ranks3, dp_ranks3 = get_mpu_ranks(tp_size=tp_size_tgt, pp_size=pp_size_tgt, dp_size=dp_size_src)

    # handle tp contraction first
    print("\n*** TP contraction:")

    for i, r in enumerate(tp_ranks1):
        print(f'{tp_ranks1[i]} => {tp_ranks2[i]}')

    # handle pp contraction next

    print("\n*** PP contraction:")

    for i, r in enumerate(pp_ranks1):
        print(f'{pp_ranks2[i]} => {pp_ranks3[i]}')


# easy
#reshape([2,2,1],[1,1,1])

# probably need more logic to suggest how to pack
#reshape([4,4,1],[2,2,1])

#reshape([2,4,2], [8,32,1])

# get_mpu_ranks(2,2,2)
# get_mpu_ranks(4,2,1)
# get_mpu_ranks(2,4,1)
# get_mpu_ranks(1,1,8)
