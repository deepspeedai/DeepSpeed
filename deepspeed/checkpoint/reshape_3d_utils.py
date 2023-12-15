# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .reshape_utils import (get_files, get_files_with_prefix, partition_data, get_zero_files)

from .constants import (MODEL_FILE_PREFIX, LAYER_FILE_PREFIX)

from .reshape_meg_2d import (reshape_meg_2d_parallel, meg_2d_parallel_map)

PP_DIM = 'PP'
TP_DIM = 'TP'
DP_DIM = 'DP'


class model_3d_desc(object):

    def __init__(self, pp_degree=1, tp_degree=1, dp_degree=1):
        self.pp_degree = pp_degree
        self.tp_degree = tp_degree
        self.dp_degree = dp_degree

    def reshape(self, target_3d_desc, verbose=False):
        valid_reshape, reshape_errors = self.can_reshape(target_3d_desc)
        assert valid_reshape, ','.join(reshape_errors)
        tgt_2d_map = reshape_meg_2d_parallel(old_pp_degree=self.pp_degree,
                                             old_tp_degree=self.tp_degree,
                                             new_pp_degree=target_3d_desc.pp_degree,
                                             new_tp_degree=target_3d_desc.tp_degree,
                                             verbose=verbose)

        flat_3d_map = flatten_dp_dimension(meg_2d_map=tgt_2d_map,
                                           src_2d_size=self.pp_degree * self.tp_degree,
                                           dp_degree=self.dp_degree)

        return unflatten_dp_dimension(meg_2d_map=flat_3d_map, dp_degree=target_3d_desc.dp_degree)

    def get_desc(self):
        return f'{PP_DIM},{TP_DIM},{DP_DIM} = ({self.pp_degree}, {self.tp_degree}, {self.dp_degree})'

    def world_size(self):
        return self.pp_degree * self.tp_degree * self.dp_degree

    def is_valid(self, pp_index, tp_index, dp_index):
        err_msg = []
        valid = True
        for index, degree, dim_name in [(pp_index, self.pp_degree, PP_DIM), (tp_index, self.tp_degree, TP_DIM),
                                        (dp_index, self.dp_degree, DP_DIM)]:
            if index >= degree:
                valid = False
                err_msg.append(f'{dim_name} indexing error: index {index} >= degree {degree}')

        return valid, err_msg

    def can_reshape(self, target_3d_desc):
        err_msg = []
        if target_3d_desc.pp_degree > self.pp_degree:
            err_msg.append(
                f'Expansion reshape not supported - {PP_DIM}: {self.pp_degree} ---> {target_3d_desc.pp_degree}')

        if target_3d_desc.tp_degree > self.tp_degree:
            err_msg.append(
                f'Expansion reshape not supported - {TP_DIM}: {self.tp_degree} ---> {target_3d_desc.tp_degree}')

        if target_3d_desc.dp_degree > self.dp_degree:
            err_msg.append(
                f'Expansion reshape not supported - {DP_DIM}: {self.dp_degree} ---> {target_3d_desc.dp_degree}')

        return len(err_msg) == 0, err_msg


def get_model_3d_descriptor(dir):
    file_list = get_files(dir)
    zero_file_list = get_zero_files(dir)
    num_pp0_files = len(get_files_with_prefix(file_list, f'{LAYER_FILE_PREFIX}01'))
    if num_pp0_files > 0:
        tp_degree = num_pp0_files
        pp_degree = len(get_files_with_prefix(file_list, MODEL_FILE_PREFIX)) // tp_degree
        dp_degree = max(1, len(zero_file_list) // (pp_degree * tp_degree))
    else:
        tp_degree = len(get_files_with_prefix(file_list, MODEL_FILE_PREFIX))
        dp_degree = max(1, len(zero_file_list) // tp_degree)
        pp_degree = 1

    return model_3d_desc(pp_degree, tp_degree, dp_degree)


def flatten_dp_dimension(meg_2d_map, src_2d_size, dp_degree):
    new_meg_2d_map = meg_2d_parallel_map(meg_2d_map.pp_degree, meg_2d_map.tp_degree)
    for pp_index in range(meg_2d_map.pp_degree):
        for tp_index in range(meg_2d_map.tp_degree):
            dp0_indices = meg_2d_map.get_data(pp_index, tp_index)
            for idx in dp0_indices:
                dpX_indices = [idx + (i * src_2d_size) for i in range(dp_degree)]
                new_meg_2d_map.add_data(pp_index, tp_index, dpX_indices)
    return new_meg_2d_map


def unflatten_dp_dimension(meg_2d_map, dp_degree):
    pp_degree = meg_2d_map.pp_degree
    tp_degree = meg_2d_map.tp_degree
    meg_2d_map_list = [meg_2d_parallel_map(pp_degree=pp_degree, tp_degree=tp_degree) for _ in range(dp_degree)]
    for pp_index in range(pp_degree):
        for tp_index in range(tp_degree):
            flat_dp_indices = meg_2d_map.get_data(pp_index, tp_index)
            partitioned_dp_indices = partition_data(flat_dp_indices, dp_degree)
            for dp_indices, _2d_map in zip(partitioned_dp_indices, meg_2d_map_list):
                _2d_map.add_data(pp_index, tp_index, dp_indices)

    return meg_2d_map_list
