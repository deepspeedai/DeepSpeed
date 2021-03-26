#!/usr/bin/env python

# This script extracts fp32 consolidated weights from a zero 2 and 3 DeepSpeed checkpoints. It gets
# copied into the top level checkpoint dir, so the user can easily do the conversion at any point in
# the future. Once extracted, the weights don't require DeepSpeed and can be used in any
# application.
#
# example: python zero_to_fp32.py global_step1 pytorch_model.bin

import argparse
import torch
import glob
import os
from collections import OrderedDict
import deepspeed

# while this script doesn't use deepspeed to recover data, since the checkpoints are pickled with
# DeepSpeed data structures it has to be available in the current python environment.


def get_optim_files(checkpoint_dir):

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    # XXX: need to test that this simple glob rule works for multi-node setup too
    optim_files = sorted(glob.glob(f"{checkpoint_dir}/*_optim_states.pt"))

    if len(optim_files) == 0:
        raise FileNotFoundError(
            f"can't find '*_optim_states.pt' files in directory '{checkpoint_dir}'")

    return optim_files


def parse_optim_states(files):
    state_dicts = []
    for f in files:
        state_dicts.append(torch.load(f))

    if not "zero_stage" in state_dicts[0]['optimizer_state_dict']:
        raise ValueError(f"non zero checkpoint")
    zero_stage = state_dicts[0]['optimizer_state_dict']["zero_stage"]

    # the groups are named differently in each stage
    if zero_stage == 2:
        fp32_groups_key = "single_partition_of_fp32_groups"
    elif zero_stage == 3:
        fp32_groups_key = "fp32_flat_groups"
    else:
        raise ValueError(f"unknown zero stage {zero_stage}")

    param_shapes = state_dicts[0]["param_shapes"]
    fp32_flat_groups = [
        state_dicts[i]['optimizer_state_dict'][fp32_groups_key][0]
        for i in range(len(state_dicts))
    ]
    world_size = state_dicts[0]['optimizer_state_dict']["partition_count"]

    return zero_stage, world_size, param_shapes, fp32_flat_groups


def zero3_partitioned_param_info(unpartitioned_numel, world_size):
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = int(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def convert_zero_chkpt_to_fp32_consolid_state_dict(checkpoint_dir, output_file):
    """
    Convert zero 2 or 3 checkpoint into a single fp32 consolidated state_dict file that can be
    loaded with ``torch.load(file)`` and used for training without DeepSpeed.

    Args:
        - ``checkpoint_dir``: path to the deepspeed checkpoint folder
        - ``output_file``: path to the pytorch fp32 state_dict output file (e.g. path/pytorch_model.bin)

    """
    print(f"Processing zero checkpoint '{checkpoint_dir}'")

    optim_files = get_optim_files(checkpoint_dir)
    zero_stage, world_size, param_shapes, fp32_flat_groups = parse_optim_states(optim_files)
    print(
        f"Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}")

    # Reconstruction protocol:
    #
    # - for zero2 we just need to concat the partitions back to back and reconsolidate over one huge
    # flat buffer - no need to deal with padding since if there is any it will be only in the tail
    # of the last partition so there it will be just left out
    #
    # - for zero3 we need to zip the partitions together at boundary of each param, re-consolidating
    # each param, while dealing with padding if any

    if zero_stage == 2:
        # XXX: memory usage doubles here (zero2)
        full_single_fp32_vector = torch.cat(fp32_flat_groups, 0)

    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    state_dict = OrderedDict()
    offset = 0
    total_numel = 0
    for name, shape in param_shapes.items():
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        if zero_stage == 2:
            # print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} ")
            state_dict[name] = full_single_fp32_vector.narrow(
                0,
                offset,
                unpartitioned_numel).view(shape)
            offset += unpartitioned_numel

        elif zero_stage == 3:
            partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)
            # print(f"{name} full shape: {shape} partition0 numel {partitioned_numel} partitioned_padding_numel {partitioned_padding_numel}")
            # XXX: memory usage doubles here (zero3)
            state_dict[name] = torch.cat(
                tuple(fp32_flat_groups[i].narrow(0,
                                                 offset,
                                                 partitioned_numel)
                      for i in range(world_size)),
                0).view(shape)
            offset += partitioned_numel + partitioned_padding_numel

    # the job is done
    print(f"Saving fp32 state dict to {output_file} (total_numel={total_numel})")

    torch.save(state_dict, output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help=
        "path to the deepspeed checkpoint folder, e.g., path/checkpoint-1/global_step1")
    parser.add_argument(
        "output_file",
        type=str,
        help=
        "path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-1/pytorch_model.bin)"
    )
    args = parser.parse_args()

    convert_zero_chkpt_to_fp32_consolid_state_dict(args.checkpoint_dir, args.output_file)
