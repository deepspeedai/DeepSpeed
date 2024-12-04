#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This script extracts fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints. It gets
# copied into the top level checkpoint dir, so the user can easily do the conversion at any point in
# the future. Once extracted, the weights don't require DeepSpeed and can be used in any
# application.
#
# example:
#   python zero_to_fp32.py . output_dir/
#   or
#   python zero_to_fp32.py . output_dir/ --safe_serialization

import argparse
import torch
import glob
import math
import os
import re
import gc
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from dataclasses import dataclass

# while this script doesn't use deepspeed to recover data, since the checkpoints are pickled with
# DeepSpeed data structures it has to be available in the current python environment.
from deepspeed.utils import logger
from deepspeed.checkpoint.constants import (DS_VERSION, OPTIMIZER_STATE_DICT, SINGLE_PARTITION_OF_FP32_GROUPS,
                                            FP32_FLAT_GROUPS, ZERO_STAGE, PARTITION_COUNT, PARAM_SHAPES, BUFFER_NAMES,
                                            FROZEN_PARAM_SHAPES, FROZEN_PARAM_FRAGMENTS)


@dataclass
class zero_model_state:
    buffers: dict()
    param_shapes: dict()
    shared_params: list
    ds_version: int
    frozen_param_shapes: dict()
    frozen_param_fragments: dict()


debug = 0

# load to cpu
device = torch.device('cpu')


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_model_state_file(checkpoint_dir, zero_stage):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    # there should be only one file
    if zero_stage <= 2:
        file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    elif zero_stage == 3:
        file = os.path.join(checkpoint_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt")

    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find model states file at '{file}'")

    return file


def get_checkpoint_files(checkpoint_dir, glob_pattern):
    # XXX: need to test that this simple glob rule works for multi-node setup too
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def get_optim_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, "*_optim_states.pt")


def get_model_state_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, "*_model_states.pt")


def parse_model_states(files):
    zero_model_states = []
    for file in files:
        state_dict = torch.load(file, map_location=device, weights_only=False)

        if BUFFER_NAMES not in state_dict:
            raise ValueError(f"{file} is not a model state checkpoint")
        buffer_names = state_dict[BUFFER_NAMES]
        if debug:
            print("Found buffers:", buffer_names)

        # recover just the buffers while restoring them to fp32 if they were saved in fp16
        buffers = {k: v.float() for k, v in state_dict["module"].items() if k in buffer_names}
        param_shapes = state_dict[PARAM_SHAPES]

        # collect parameters that are included in param_shapes
        param_names = []
        for s in param_shapes:
            for name in s.keys():
                param_names.append(name)

        # update with frozen parameters
        frozen_param_shapes = state_dict.get(FROZEN_PARAM_SHAPES, None)
        if frozen_param_shapes is not None:
            if debug:
                print(f"Found frozen_param_shapes: {frozen_param_shapes}")
            param_names += list(frozen_param_shapes.keys())

        # handle shared params
        shared_params = [[k, v] for k, v in state_dict["shared_params"].items()]

        ds_version = state_dict.get(DS_VERSION, None)

        frozen_param_fragments = state_dict.get(FROZEN_PARAM_FRAGMENTS, None)

        z_model_state = zero_model_state(buffers=buffers,
                                         param_shapes=param_shapes,
                                         shared_params=shared_params,
                                         ds_version=ds_version,
                                         frozen_param_shapes=frozen_param_shapes,
                                         frozen_param_fragments=frozen_param_fragments)
        zero_model_states.append(z_model_state)

    return zero_model_states


def parse_optim_states(files, ds_checkpoint_dir):
    total_files = len(files)
    state_dicts = []
    for f in tqdm(files, desc='Loading checkpoint shards'):
        state_dict = torch.load(f, map_location=device, mmap=True, weights_only=False)
        # immediately discard the potentially huge 2 optimizer states as we only care for fp32 master weights
        # and also handle the case where it was already removed by another helper script
        state_dict["optimizer_state_dict"].pop("optimizer_state_dict", None)
        state_dicts.append(state_dict)

    if not ZERO_STAGE in state_dicts[0][OPTIMIZER_STATE_DICT]:
        raise ValueError(f"{files[0]} is not a zero checkpoint")
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]

    # For ZeRO-2 each param group can have different partition_count as data parallelism for expert
    # parameters can be different from data parallelism for non-expert parameters. So we can just
    # use the max of the partition_count to get the dp world_size.

    if type(world_size) is list:
        world_size = max(world_size)

    if world_size != total_files:
        raise ValueError(
            f"Expected {world_size} of '*_optim_states.pt' under '{ds_checkpoint_dir}' but found {total_files} files. "
            "Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
        )

    # the groups are named differently in each stage
    if zero_stage <= 2:
        fp32_groups_key = SINGLE_PARTITION_OF_FP32_GROUPS
    elif zero_stage == 3:
        fp32_groups_key = FP32_FLAT_GROUPS
    else:
        raise ValueError(f"unknown zero stage {zero_stage}")

    fp32_flat_groups = [state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key] for i in range(len(state_dicts))]
    return zero_stage, world_size, fp32_flat_groups


def _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, exclude_frozen_parameters):
    """
    Returns fp32 state_dict reconstructed from ds checkpoint

    Args:
        - ``ds_checkpoint_dir``: path to the deepspeed checkpoint folder (where the optimizer files are)

    """
    print(f"Processing zero checkpoint '{ds_checkpoint_dir}'")

    optim_files = get_optim_files(ds_checkpoint_dir)
    zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files, ds_checkpoint_dir)
    print(f"Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}")

    model_files = get_model_state_files(ds_checkpoint_dir)

    zero_model_states = parse_model_states(model_files)
    print(f'Parsing checkpoint created by deepspeed=={zero_model_states[0].ds_version}')

    if zero_stage <= 2:
        return _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                                          exclude_frozen_parameters)
    elif zero_stage == 3:
        return _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                                          exclude_frozen_parameters)


def _zero2_merge_frozen_params(state_dict, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    frozen_param_shapes = zero_model_states[0].frozen_param_shapes
    frozen_param_fragments = zero_model_states[0].frozen_param_fragments

    if debug:
        num_elem = sum(s.numel() for s in frozen_param_shapes.values())
        print(f'rank 0: {FROZEN_PARAM_SHAPES}.numel = {num_elem}')

        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in frozen_param_fragments.values()])
        print(f'Frozen params: Have {avail_numel} numels to process.')
        print(f'Frozen params: Need {wanted_numel} numels in {wanted_params} params')

    total_params = 0
    total_numel = 0
    for name, shape in frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        state_dict[name] = frozen_param_fragments[name]

        if debug:
            print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} ")

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


def _has_callable(obj, fn):
    attr = getattr(obj, fn, None)
    return callable(attr)


def _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes

    # Reconstruction protocol:
    #
    # XXX: document this

    if debug:
        for i in range(world_size):
            for j in range(len(fp32_flat_groups[0])):
                print(f"{FP32_FLAT_GROUPS}[{i}][{j}].shape={fp32_flat_groups[i][j].shape}")

    # XXX: memory usage doubles here (zero2)
    num_param_groups = len(fp32_flat_groups[0])
    merged_single_partition_of_fp32_groups = []
    for i in range(num_param_groups):
        merged_partitions = [sd[i] for sd in fp32_flat_groups]
        full_single_fp32_vector = torch.cat(merged_partitions, 0)
        merged_single_partition_of_fp32_groups.append(full_single_fp32_vector)
    avail_numel = sum(
        [full_single_fp32_vector.numel() for full_single_fp32_vector in merged_single_partition_of_fp32_groups])

    if debug:
        wanted_params = sum([len(shapes) for shapes in param_shapes])
        wanted_numel = sum([sum(shape.numel() for shape in shapes.values()) for shapes in param_shapes])
        # not asserting if there is a mismatch due to possible padding
        print(f"Have {avail_numel} numels to process.")
        print(f"Need {wanted_numel} numels in {wanted_params} params.")

    # params
    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    total_numel = 0
    total_params = 0
    for shapes, full_single_fp32_vector in zip(param_shapes, merged_single_partition_of_fp32_groups):
        offset = 0
        avail_numel = full_single_fp32_vector.numel()
        for name, shape in shapes.items():

            unpartitioned_numel = shape.numel() if _has_callable(shape, 'numel') else math.prod(shape)
            total_numel += unpartitioned_numel
            total_params += 1

            if debug:
                print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} ")
            state_dict[name] = full_single_fp32_vector.narrow(0, offset, unpartitioned_numel).view(shape)
            offset += unpartitioned_numel

        # Z2 started to align to 2*world_size to improve nccl performance. Therefore both offset and
        # avail_numel can differ by anywhere between 0..2*world_size. Due to two unrelated complex
        # paddings performed in the code it's almost impossible to predict the exact numbers w/o the
        # live optimizer object, so we are checking that the numbers are within the right range
        align_to = 2 * world_size

        def zero2_align(x):
            return align_to * math.ceil(x / align_to)

        if debug:
            print(f"original offset={offset}, avail_numel={avail_numel}")

        offset = zero2_align(offset)
        avail_numel = zero2_align(avail_numel)

        if debug:
            print(f"aligned  offset={offset}, avail_numel={avail_numel}")

        # Sanity check
        if offset != avail_numel:
            raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                               exclude_frozen_parameters):
    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f"added {len(buffers)} buffers")

    if not exclude_frozen_parameters:
        _zero2_merge_frozen_params(state_dict, zero_model_states)

    _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def zero3_partitioned_param_info(unpartitioned_numel, world_size):
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def _zero3_merge_frozen_params(state_dict, world_size, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    if debug:
        for i in range(world_size):
            num_elem = sum(s.numel() for s in zero_model_states[i].frozen_param_fragments.values())
            print(f'rank {i}: {FROZEN_PARAM_SHAPES}.numel = {num_elem}')

        frozen_param_shapes = zero_model_states[0].frozen_param_shapes
        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in zero_model_states[0].frozen_param_fragments.values()]) * world_size
        print(f'Frozen params: Have {avail_numel} numels to process.')
        print(f'Frozen params: Need {wanted_numel} numels in {wanted_params} params')

    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        param_frags = tuple(model_state.frozen_param_fragments[name] for model_state in zero_model_states)
        state_dict[name] = torch.cat(param_frags, 0).narrow(0, 0, unpartitioned_numel).view(shape)

        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print(
                f"Frozen params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


class GatheredTensor:
    """
    A pseudo tensor that collects partitioned weights.
    It is more memory efficient when there are multiple groups.
    """

    def __init__(self, flat_groups, flat_groups_offset, offset, partitioned_numel, shape):
        self.flat_groups = flat_groups
        self.flat_groups_offset = flat_groups_offset
        self.offset = offset
        self.partitioned_numel = partitioned_numel
        self.shape = shape
        self.dtype = self.flat_groups[0][0].dtype

    def contiguous(self):
        """
        Merge partitioned weights from flat_groups into a single tensor.
        """
        end_idx = self.offset + self.partitioned_numel
        world_size = len(self.flat_groups)
        pad_flat_param_chunks = []

        for rank_i in range(world_size):
            # for each rank, we need to collect weights from related group/groups
            flat_groups_at_rank_i = self.flat_groups[rank_i]
            start_group_id = None
            end_group_id = None
            for group_id in range(len(self.flat_groups_offset)):
                if self.flat_groups_offset[group_id] <= self.offset < self.flat_groups_offset[group_id + 1]:
                    start_group_id = group_id
                if self.flat_groups_offset[group_id] < end_idx <= self.flat_groups_offset[group_id + 1]:
                    end_group_id = group_id
                    break
            # collect weights from related group/groups
            for group_id in range(start_group_id, end_group_id + 1):
                flat_tensor = flat_groups_at_rank_i[group_id]
                start_offset = self.offset - self.flat_groups_offset[group_id]
                end_offset = min(end_idx, self.flat_groups_offset[group_id + 1]) - self.flat_groups_offset[group_id]
                pad_flat_param_chunks.append(flat_tensor[start_offset:end_offset])

        # collect weights from all ranks
        pad_flat_param = torch.cat(pad_flat_param_chunks, dim=0)
        param = pad_flat_param[:self.shape.numel()].view(self.shape).contiguous()
        return param


def _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes
    avail_numel = sum([flat_group.numel() for flat_group in fp32_flat_groups[0]]) * world_size

    # Reconstruction protocol: For zero3 we need to zip the partitions together at boundary of each
    # param, re-consolidating each param, while dealing with padding if any

    # merge list of dicts, preserving order
    param_shapes = {k: v for d in param_shapes for k, v in d.items()}

    if debug:
        for i in range(world_size):
            print(f"{FP32_FLAT_GROUPS}[{i}].shape={fp32_flat_groups[i].shape}")

        wanted_params = len(param_shapes)
        wanted_numel = sum(shape.numel() for shape in param_shapes.values())
        # not asserting if there is a mismatch due to possible padding
        avail_numel = fp32_flat_groups[0].numel() * world_size
        print(f"Trainable params: Have {avail_numel} numels to process.")
        print(f"Trainable params: Need {wanted_numel} numels in {wanted_params} params.")

    # params
    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    offset = 0
    total_numel = 0
    total_params = 0
    flat_groups_offset = [0] + list(np.cumsum([flat_tensor.numel() for flat_tensor in fp32_flat_groups[0]]))
    for name, shape in tqdm(param_shapes.items(), desc='Gathering sharded weights'):
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1
        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print(
                f"Trainable params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

        # memory efficient tensor
        tensor = GatheredTensor(fp32_flat_groups, flat_groups_offset, offset, partitioned_numel, shape)
        state_dict[name] = tensor
        offset += partitioned_numel

    offset *= world_size

    # Sanity check
    if offset != avail_numel:
        raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed Trainable fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                               exclude_frozen_parameters):
    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f"added {len(buffers)} buffers")

    if not exclude_frozen_parameters:
        _zero3_merge_frozen_params(state_dict, world_size, zero_model_states)

    _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def to_torch_tensor(state_dict, return_empty_tensor=False):
    """
    Convert state_dict of GatheredTensor to torch tensor
    """
    torch_state_dict = {}
    converted_tensors = {}
    for name, tensor in state_dict.items():
        tensor_id = id(tensor)
        if tensor_id in converted_tensors:  # shared tensors
            shared_tensor = torch_state_dict[converted_tensors[tensor_id]]
            torch_state_dict[name] = shared_tensor
        else:
            converted_tensors[tensor_id] = name
            if return_empty_tensor:
                torch_state_dict[name] = torch.empty(tensor.shape, dtype=tensor.dtype)
            else:
                torch_state_dict[name] = tensor.contiguous()
    return torch_state_dict


def get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir,
                                             tag=None,
                                             exclude_frozen_parameters=False,
                                             lazy_mode=False):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated state_dict that can be loaded with
    ``load_state_dict()`` and used for training without DeepSpeed or shared with others, for example
    via a model hub.

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in 'latest' file. e.g., ``global_step14``
        - ``exclude_frozen_parameters``: exclude frozen parameters
        - ``lazy_mode``: get state_dict in lazy mode. It returns a dict of pesduo tensor instead of torch tensor, which is more memory efficient.
          Convert the pesduo tensor to torch tensor by ``.contiguous()``

    Returns:
        - pytorch ``state_dict``

    A typical usage might be ::

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        # do the training and checkpoint saving
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir) # already on cpu
        model = model.cpu() # move to cpu
        model.load_state_dict(state_dict)
        # submit to model hub or save the model to share with others

    In this example the ``model`` will no longer be usable in the deepspeed context of the same
    application. i.e. you will need to re-initialize the deepspeed engine, since
    ``model.load_state_dict(state_dict)`` will remove all the deepspeed magic from it.

    If you want it all done for you, use ``load_state_dict_from_zero_checkpoint`` instead.

    Note: the above usage may not work if your application doesn't have sufficient free CPU memory.
    You may need to use the offline approach using the ``zero_to_fp32.py`` script that is saved with
    the checkpoint. Or you can load state_dict in lazy mode ::

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, lazy_mode=True) # not on cpu
        for name, lazy_tensor in state_dict.item():
            tensor = lazy_tensor.contiguous()  # to cpu
            print(name, tensor)
            # del tensor to release memory if it no longer in use
    """
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

    state_dict = _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, exclude_frozen_parameters)
    if lazy_mode:
        return state_dict
    else:
        return to_torch_tensor(state_dict)


def convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir,
                                               output_dir,
                                               max_shard_size="5GB",
                                               safe_serialization=False,
                                               tag=None,
                                               exclude_frozen_parameters=False):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict`` file that can be
    loaded with ``torch.load(file)`` + ``load_state_dict()`` and used for training without DeepSpeed.

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder. (one that contains the tag-folder, like ``global_step14``)
        - ``output_dir``: directory to the pytorch fp32 state_dict output files
        - ``max_shard_size``: the maximum size for a checkpoint before being sharded, default value is 5GB
        - ``safe_serialization``:  whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``
        - ``exclude_frozen_parameters``: exclude frozen parameters
    """

    # Dependency pre-check
    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print('If you want to use `safe_serialization`, please `pip install safetensors`')
            raise
    if max_shard_size is not None:
        try:
            from huggingface_hub import split_torch_state_dict_into_shards
        except ImportError:
            print('If you want to use `max_shard_size`, please `pip install huggingface_hub`')
            raise

    # Convert zero checkpoint to state_dict
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir,
                                                          tag,
                                                          exclude_frozen_parameters,
                                                          lazy_mode=True)

    # Shard the model if it is too big.
    weights_name = "model.safetensors" if safe_serialization else "pytorch_model.bin"
    if max_shard_size is not None:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        # an memory-efficient approach for sharding
        empty_state_dict = to_torch_tensor(state_dict, return_empty_tensor=True)
        state_dict_split = split_torch_state_dict_into_shards(empty_state_dict,
                                                              filename_pattern=filename_pattern,
                                                              max_shard_size=max_shard_size)
    else:
        from collections import namedtuple
        StateDictSplit = namedtuple("StateDictSplit", ["is_sharded", "filename_to_tensors"])
        state_dict_split = StateDictSplit(is_sharded=False,
                                          filename_to_tensors={weights_name: list(state_dict.keys())})

    # Save the model by shard
    os.makedirs(output_dir, exist_ok=True)
    filename_to_tensors = state_dict_split.filename_to_tensors.items()
    for shard_file, tensors in tqdm(filename_to_tensors, desc="Saving checkpoint shards"):
        shard_state_dict = {tensor_name: state_dict[tensor_name] for tensor_name in tensors}
        shard_state_dict = to_torch_tensor(shard_state_dict)
        output_path = os.path.join(output_dir, shard_file)
        if safe_serialization:
            save_file(shard_state_dict, output_path, metadata={"format": "pt"})
        else:
            torch.save(shard_state_dict, output_path)
        # release the memory of current shard
        for tensor_name in list(shard_state_dict.keys()):
            del state_dict[tensor_name]
            del shard_state_dict[tensor_name]
        del shard_state_dict
        gc.collect()

    # Save index if sharded
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = "model.safetensors.index.json" if safe_serialization else "pytorch_model.bin.index.json"
        save_index_file = os.path.join(output_dir, save_index_file)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def load_state_dict_from_zero_checkpoint(model, checkpoint_dir, tag=None):
    """
    1. Put the provided model to cpu
    2. Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict``
    3. Load it into the provided model

    Args:
        - ``model``: the model object to update
        - ``checkpoint_dir``: path to the desired checkpoint folder. (one that contains the tag-folder, like ``global_step14``)
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``

    Returns:
        - ``model`: modified model

    Make sure you have plenty of CPU memory available before you call this function. If you don't
    have enough use the ``zero_to_fp32.py`` utility to do the conversion. You will find it
    conveniently placed for you in the checkpoint folder.

    A typical usage might be ::

        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
        # submit to model hub or save the model to share with others

    Note, that once this was run, the ``model`` will no longer be usable in the deepspeed context
    of the same application. i.e. you will need to re-initialize the deepspeed engine, since
    ``model.load_state_dict(state_dict)`` will remove all the deepspeed magic from it.

    """
    logger.info(f"Extracting fp32 weights")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

    logger.info(f"Overwriting model with fp32 weights")
    model = model.cpu()
    model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument("output_dir",
                        type=str,
                        help="directory to the pytorch fp32 state_dict output files"
                        "(e.g. path/checkpoint-12-output/)")
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size"
        "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`"
        "We default it to 5GB in order for models to be able to run easily on free-tier google colab instances"
        "without CPU OOM issues.")
    parser.add_argument(
        "--safe_serialization",
        default=False,
        action='store_true',
        help="Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).")
    parser.add_argument("-t",
                        "--tag",
                        type=str,
                        default=None,
                        help="checkpoint tag used as a unique identifier for checkpoint. e.g., global_step1")
    parser.add_argument("--exclude_frozen_parameters", action='store_true', help="exclude frozen parameters")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    args = parser.parse_args()

    debug = args.debug

    convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint_dir,
                                               args.output_dir,
                                               max_shard_size=args.max_shard_size,
                                               safe_serialization=args.safe_serialization,
                                               tag=args.tag,
                                               exclude_frozen_parameters=args.exclude_frozen_parameters)
