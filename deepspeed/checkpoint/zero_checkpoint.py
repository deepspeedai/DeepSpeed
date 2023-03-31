# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .constants import (BASE_OPTIMIZER_STATE, GROUP_PADDINGS, OPTIMIZER_STATE_DICT, PARTITION_COUNT)

from .reshape_utils import (basic_folder_validation, get_zero_files, merge_state)

from .reshape_3d_utils import (model_3d_desc, get_model_3d_descriptor)

GROUP_STATE_KEY = 'state'


class ZeROCheckpoint(object):

    def __init__(self, dir):
        basic_folder_validation(dir)
        self.dir = dir
        self.file_list = get_zero_files(dir)
        self.num_files = len(self.file_list)
        assert self.num_files > 0, f'No ZeRO files found in {dir}'

        self.src_3d = get_model_3d_descriptor(dir)
        self.target_3d = model_3d_desc(pp_degree=self.src_3d.pp_degree,
                                       tp_degree=self.src_3d.tp_degree,
                                       dp_degree=self.src_3d.dp_degree)
        self._3d_file_map = self.src_3d.reshape(self.target_3d)

    def get_src_world_size(self):
        return self.src_3d.world_size()

    def get_src_tp_degree(self):
        return self.src_3d.tp_degree

    def get_src_pp_degree(self):
        return self.src_3d.pp_degree

    def get_src_dp_degree(self):
        return self.src_3d.dp_degree

    def get_file_indices_for_rank(self, pp_index, tp_index, dp_index):
        assert dp_index < len(self._3d_file_map), f'DP index {dp_index} >= DP degree {len(self._3d_file_map)}'
        dp_2d_map = self._3d_file_map[dp_index]
        return dp_2d_map.get_data(pp_index, tp_index)

    def get_files_for_rank(self, pp_index, tp_index, dp_index):
        file_idx_list = self.get_file_indices_for_rank(pp_index, tp_index, dp_index)
        return [self.file_list[idx] for idx in file_idx_list]

    def get_state_for_rank(self, pp_index, tp_index, dp_index, keys_to_ignore=[], strip_tensor_paddings=True):
        state_file_list = self.get_files_for_rank(pp_index, tp_index, dp_index)
        merged_sd = None
        for state_file in state_file_list:
            sd = torch.load(state_file, map_location=torch.device('cpu'))
            for key in keys_to_ignore:
                sd.pop(key, None)

            if strip_tensor_paddings:
                self._strip_tensor_paddings(sd)

            if merged_sd is None:
                merged_sd = sd
            else:
                merged_sd = merge_state(merged_sd, sd)

            self._update_partition_count(merged_sd)
            if strip_tensor_paddings:
                self._clear_group_paddings(merged_sd)

        return merged_sd

    def print_3d_index_map(self, tag=None):
        if tag:
            print(f'3D index map: {tag}')
        for dp_index, _2d_map in enumerate(self._3d_file_map):
            _2d_map.print_data(f'dp = {dp_index}')

    def print_3d_file_map(self, tag=None):
        if tag:
            print(f'3D file map: {tag}')
        for dp_index, _2d_map in enumerate(self._3d_file_map):
            for pp_index in _2d_map.pp_degree:
                for tp_index in _2d_map.tp_degree:
                    file_index_list = _2d_map.get_data(pp_index, tp_index)
                    file_list = [self.file_list[idx] for idx in file_index_list]
                    print(f'{pp_index}, {tp_index}, {dp_index} => {file_list}')

    def reshape(self, target_3d_desc: model_3d_desc):
        self.target_3d = target_3d_desc
        self._3d_file_map = self.src_3d.reshape(self.target_3d)

    def _strip_tensor_paddings(self, sd):
        param_group_states = self._get_param_group_states(sd)
        if param_group_states is None:
            return

        group_paddings = self._get_optimizer_state(sd, GROUP_PADDINGS)
        if group_paddings is None:
            return

        for key, group_state in param_group_states.items():
            if group_paddings[key] == 0:
                continue
            for state_name, state_value in group_state.items():
                if torch.is_tensor(state_value):
                    raw_length = state_value.numel() - group_paddings[key]
                    group_state[state_name] = torch.narrow(state_value, 0, 0, raw_length).clone()

    def _clear_group_paddings(self, sd):
        group_paddings = self._get_optimizer_state(sd, GROUP_PADDINGS)
        if group_paddings:
            num_groups = len(group_paddings)
            sd[OPTIMIZER_STATE_DICT][GROUP_PADDINGS] = [0] * num_groups

    def _get_optimizer_state(self, sd, state_key):
        optimizer_state = sd.get(OPTIMIZER_STATE_DICT, None)
        if optimizer_state is None:
            return None

        return optimizer_state.get(state_key, None)

    def _get_param_group_states(self, sd):
        optimizer_state = sd.get(OPTIMIZER_STATE_DICT, None)
        if optimizer_state is None:
            return None

        base_optimizer_state = optimizer_state.get(BASE_OPTIMIZER_STATE, None)
        if base_optimizer_state is None:
            return None

        return base_optimizer_state.get(GROUP_STATE_KEY, None)

    def _update_partition_count(self, sd):
        partition_counts = self._get_optimizer_state(sd, PARTITION_COUNT)
        if partition_counts:
            num_groups = len(partition_counts)
            sd[OPTIMIZER_STATE_DICT][PARTITION_COUNT] = [self.target_3d.dp_degree] * num_groups
