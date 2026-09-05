# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types
from deepspeed.utils import get_full_hp_param, get_full_hp_grad, get_hp_fragment_mapping
from deepspeed.utils import set_full_hp_param, set_full_hp_grad


def link_hp_params(lp_param_list,
                   flat_hp_partition,
                   gradient_dict,
                   offload_gradient_dict,
                   use_offload,
                   param_group_index,
                   partition_start,
                   partition_size,
                   dp_group,
                   param_offsets=None):
    local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group,
                                                       param_offsets)

    for lp_param, lp_start in local_lp_param_and_offset:
        lp_param._hp_mapping = get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict,
                                                       offload_gradient_dict, use_offload, param_group_index,
                                                       partition_start, partition_size)


def lazy_init_hp_params_optimizer_state(lp_param_list, flat_hp_partition, optimizer_state):
    for lp in lp_param_list:
        if lp._hp_mapping is not None:
            lp._hp_mapping.set_optim_state_fragment(flat_hp_partition, optimizer_state[flat_hp_partition])


def _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group, param_offsets=None):
    current_offset = 0
    param_and_offset_list = []
    partition_end = partition_start + partition_size
    index_in_param_group = 0
    gradient_index_by_param = {}
    if param_offsets is not None:
        current_partition_offset = 0
        gradient_list_index = 0
        params_by_offset = sorted(zip(lp_param_list, param_offsets), key=lambda item: item[1])
        for lp_param, param_offset in params_by_offset:
            lp_param_end = param_offset + lp_param.numel()
            in_partition = (partition_start <= param_offset < partition_end
                            or param_offset < partition_start < lp_param_end)
            if not in_partition:
                continue

            fragment_start = max(param_offset, partition_start)
            dest_offset = fragment_start - partition_start
            if dest_offset > current_partition_offset:
                gradient_list_index += 1
            gradient_index_by_param[id(lp_param)] = gradient_list_index
            gradient_list_index += 1
            current_partition_offset = min(lp_param_end, partition_end) - partition_start

    for i, lp_param in enumerate(lp_param_list):
        if param_offsets is not None:
            current_offset = param_offsets[i]
        lp_param._hp_mapping = None
        lp_param._dp_group = dp_group
        lp_param.get_full_hp_param = types.MethodType(get_full_hp_param, lp_param)
        lp_param.get_full_hp_grad = types.MethodType(get_full_hp_grad, lp_param)
        lp_param.set_full_hp_param = types.MethodType(set_full_hp_param, lp_param)
        lp_param.set_full_hp_grad = types.MethodType(set_full_hp_grad, lp_param)

        # lp_param overlaps with partition if both are true
        # 1) current_offset < partition_end,
        # 2) current_offset + lp_param.numel() >= partition_start
        lp_param_end = current_offset + lp_param.numel()
        in_partition = (partition_start <= current_offset < partition_end
                        or current_offset < partition_start < lp_param_end)
        if in_partition:
            if param_offsets is None:
                lp_param._index_in_param_group = index_in_param_group
                index_in_param_group += 1
            else:
                lp_param._index_in_param_group = gradient_index_by_param[id(lp_param)]
            if lp_param.numel() > 0:
                param_and_offset_list.append((lp_param, current_offset))
        if param_offsets is None:
            current_offset += lp_param.numel()

    return param_and_offset_list
