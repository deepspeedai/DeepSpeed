# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
from .constants import (MODEL_FILE_PREFIX, MODEL_FILE_SUFFIX, OPTIM_FILE_SUFFIX, ZERO_FILE_PREFIX)


def get_model_ckpt_name_for_rank(base_folder, mp_rank_str):
    ckpt_name = os.path.join(
        base_folder,
        MODEL_FILE_PREFIX + mp_rank_str + MODEL_FILE_SUFFIX,
    )
    return ckpt_name


def get_zero_ckpt_name_for_rank(base_folder, dp_rank, mp_rank):
    zero_prefix = f'{ZERO_FILE_PREFIX}{dp_rank}'
    mp_rank_string = f'_{MODEL_FILE_PREFIX}{mp_rank:02d}'
    zero_ckpt_name = os.path.join(
        base_folder,
        zero_prefix + mp_rank_string + OPTIM_FILE_SUFFIX,
    )
    return zero_ckpt_name


def get_layer_ckpt_name_for_rank(base_folder, layer_id, tp_rank):
    ckpt_file = f'{layer_id}-model_{tp_rank:02d}{MODEL_FILE_SUFFIX}'
    ckpt_path = os.path.join(base_folder, ckpt_file)
    return ckpt_path


# We pass cloned tensors to torch.save() to avoid checkpoint bloat that occurs when torch.save()
# saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
# This is a problem in DeepSpeed because we often allocate tensors using slices of large flattened buffers.
# Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
# It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
# See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
def clone_tensors_for_torch_save(item, device=torch.device('cpu')):
    """
    Returns a copy of ``item`` with all enclosed tensors replaced by clones on a specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.

    Parameters:
        - ``item``: tensor to clone or (possibly nested) container of tensors to clone.
        - ``device``: target device (defaults to 'cpu')

    Returns:
        - copy of ``item`` with cloned tensors on target device
    """
    if torch.is_tensor(item):
        return item.detach().clone().to(device)
    elif isinstance(item, list):
        return [clone_tensors_for_torch_save(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([clone_tensors_for_torch_save(v, device) for v in item])
    elif isinstance(item, dict):
        return type(item)({k: clone_tensors_for_torch_save(v, device) for k, v in item.items()})
    else:
        return item
