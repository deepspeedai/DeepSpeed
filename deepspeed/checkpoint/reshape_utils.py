# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
from collections import OrderedDict
from .constants import (ZERO_FILE_PREFIX, FP16_ZERO_FILE_PREFIX, BF16_ZERO_FILE_PREFIX)


def basic_folder_validation(dir):
    assert os.path.exists(dir), f'{dir} path does not exist'
    assert os.path.isdir(dir), f'{dir} is not a folder'


def get_files_with_prefix(all_files, prefix):
    file_list = []
    for file_path in all_files:
        _, fname = os.path.split(file_path)
        if fname.startswith(prefix):
            file_list.append(file_path)

    return sorted(file_list)


def validate_files(file_list):
    for file in file_list:
        if not os.path.isfile(file):
            print(f'Error: {file} is not existent')


def get_files(dir):
    file_list = []
    for root, _, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def get_zero_files(dir):
    file_list = get_files(dir)
    for prefix in [ZERO_FILE_PREFIX, FP16_ZERO_FILE_PREFIX, BF16_ZERO_FILE_PREFIX]:
        zero_files = get_files_with_prefix(file_list, prefix)
        if len(zero_files) > 0:
            return zero_files

    return []


def partition_data(data_list, num_partitions):
    num_elems = len(data_list)
    assert num_elems % num_partitions == 0
    partition_size = num_elems // num_partitions
    partitions_list = [data_list[i:i + partition_size] for i in range(0, num_elems, partition_size)]
    return partitions_list


def _key_list_to_string(key_list):
    return '.'.join(key_list)


def merge_state_dict(dict_a, dict_b, key_list):
    merged_dict = type(dict_a)({})

    for key, value in dict_b.items():
        if key in dict_a.keys():
            merged_dict[key] = merge_state(dict_a[key], dict_b[key], [str(key)])
        else:
            merged_dict[key] = value

    return merged_dict


def merge_state_list(list_a, list_b, key_list):
    if len(list_a) != len(list_b):
        print(f'{_key_list_to_string(key_list)}')
        raise ValueError(f'Cannot merge lists of different lengths, a = {len(list_a)} b = {len(list_b)}')

    return [merge_state(a, b, key_list) for a, b in zip(list_a, list_b)]


def merge_state(state_a, state_b, key_list=[]):
    if type(state_a) != type(state_b):
        key_list_string = _key_list_to_string(key_list)
        print(f'key_list = {key_list_string}')
        raise ValueError(f'Cannot merge two states of types {type(state_a)} and type {type(state_b)}')

    if type(state_a) in (dict, OrderedDict):
        return merge_state_dict(state_a, state_b, key_list)
    elif type(state_a) in (list, tuple):
        return type(state_a)(merge_state_list(state_a, state_b, key_list))
    elif torch.is_tensor(state_a):
        return torch.cat([state_a, state_b], 0)
    else:
        return state_a
