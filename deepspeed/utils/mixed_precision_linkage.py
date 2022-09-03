"""
Copyright 2022 The Microsoft DeepSpeed Team
"""
import types
from deepspeed.utils import get_full_hp_param, get_hp_fragment_mapping


def link_hp_params(lp_param_list,
                   flat_hp_partition,
                   partition_start,
                   partition_size,
                   partition_optimizer_state,
                   dp_group):
    local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list,
                                                       partition_start,
                                                       partition_size,
                                                       dp_group)

    for lp_param, lp_start in local_lp_param_and_offset:
        lp_param._hp_mapping = get_hp_fragment_mapping(lp_param,
                                                       lp_start,
                                                       flat_hp_partition,
                                                       partition_start,
                                                       partition_size,
                                                       partition_optimizer_state)


def _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group):
    current_offset = 0
    param_and_offset_list = []
    partition_end = partition_start + partition_size
    for lp_param in lp_param_list:
        lp_param._hp_mapping = None
        lp_param._dp_group = dp_group
        lp_param.get_full_hp_param = types.MethodType(get_full_hp_param, lp_param)

        # lp_param overlaps with partition if both are true
        # 1) current_offset < partition_end,
        # 2) current_offset + lp_param.numel() >= partition_start
        lp_param_end = current_offset + lp_param.numel()
        if current_offset < partition_end and lp_param_end > partition_start:
            param_and_offset_list.append((lp_param, current_offset))
        current_offset += lp_param.numel()

    return param_and_offset_list
