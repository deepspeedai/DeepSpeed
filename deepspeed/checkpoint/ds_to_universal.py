#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from functools import partial
from itertools import chain
import argparse
import glob
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
import os
import re
import shutil
import torch
import tqdm
#from pprint import pprint

from deepspeed.checkpoint import DeepSpeedCheckpoint
from deepspeed.checkpoint import (
    OPTIMIZER_STATE_DICT,
    ZERO_STAGE,
    BASE_OPTIMIZER_STATE,
    SINGLE_PARTITION_OF_FP32_GROUPS,
    PARAM_GROUPS,
    PARAM_SLICE_MAPPINGS,
    PARAM_SHAPES,
    PARAM,
    CAT_DIM,
    PARAM_N_SUB_PARAMS,
    SUB_PARAM_SHAPE,
    VOCAB_TENSOR,
    UNIVERSAL_CHECKPOINT_INFO,
    UNIVERSAL_CHECKPOINT_VERSION_KEY,
    UNIVERSAL_CHECKPOINT_VERSION_VALUE,
    VOCABULARY_PARAMETER_PATTERNS,
    PIPELINE_REPLICATED_PARAMETER_PATTERNS,
    TP_REPLICATED_PARAMETER_PATTERNS,
    PARAMETER_TO_AVERAGE_PATTERNS,
    PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
    PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0,
    PARAMETER_WITH_SUB_PARAMS,
    SubparamShape,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True, help='Input DeepSpeed Checkpoint folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Output DeepSpeed checkpoint folder')
    parser.add_argument('--num_extract_workers',
                        default=4,
                        type=int,
                        help='How many parallel processes to extract zero shards')
    parser.add_argument(
        '--num_merge_workers',
        default=2,
        type=int,
        help=
        'How many parallel processes to merge tp slices (more memory intensive, use much fewer than --num_extract_workers))'
    )
    parser.add_argument('--keep_temp_folder',
                        action='store_true',
                        help='Preserve temporary folder of intermediate checkpoint slice files. Useful for debugging.')
    parser.add_argument('--no_strict',
                        dest='strict',
                        action='store_false',
                        help='Do not perform validity checks on converted checkpoint.')
    parser.add_argument('--inject_missing_state',
                        action='store_true',
                        help='Inject missing checkpoint state into the checkpoint if it is absent.')
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def _create_checkpoint_paths(base_folder, iteration, tp_degree, pp_degree):
    path_list = []
    iter_folder = f'iter_{iteration:07d}'
    for i in range(0, tp_degree):
        path_list.append([])
        for j in range(0, pp_degree):
            rank_folder = f'mp_rank_{i:02d}' if pp_degree == 1 else f'mp_rank_{i:02d}_{j:03d}'
            ckpt_path = os.path.join(rank_folder, 'model_optim_rng.pt')
            path_list[i].append(os.path.join(base_folder, iter_folder, ckpt_path))

    return path_list


def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)


def extract_zero_shards(dir, ds_checkpoint, indices_3D):
    pp_index, tp_index, dp_index = indices_3D
    sd = ds_checkpoint.get_zero_checkpoint_state(pp_index=pp_index, tp_index=tp_index, dp_index=dp_index)

    # pprint(f"Processing {dp_index=} {pp_index=}, {tp_index=}")

    optim_sd = sd[OPTIMIZER_STATE_DICT]
    param_slice_mappings = optim_sd[PARAM_SLICE_MAPPINGS]
    universal_checkpoint_info = ds_checkpoint.get_checkpoint_info(UNIVERSAL_CHECKPOINT_INFO)
    pipeline_replicated_params = universal_checkpoint_info.get(PIPELINE_REPLICATED_PARAMETER_PATTERNS, [])
    # print(f'{pipeline_replicated_params=}')

    # dict
    state_groups = optim_sd[BASE_OPTIMIZER_STATE]["state"]
    # list
    fp32_groups = optim_sd[SINGLE_PARTITION_OF_FP32_GROUPS]
    param_groups_cnt = len(state_groups)

    for param_group_id in range(param_groups_cnt):

        flat_state = dict(
            exp_avg=state_groups[param_group_id]["exp_avg"],
            exp_avg_sq=state_groups[param_group_id]["exp_avg_sq"],
            fp32=fp32_groups[param_group_id],
        )

        if "step" in state_groups[param_group_id]:
            flat_state["step"] = state_groups[param_group_id]["step"]

        for name, fragment_mapping in param_slice_mappings[param_group_id].items():
            if pp_index > 0 and any(re.match(pattern, name) for pattern in pipeline_replicated_params):
                # Skip tied weights that are replicated in first and last pp stages
                continue

            # pprint(f"dpt{dp_index}{pp_index}{tp_index} {param_group_id} {name} => {fragment_mapping.start}:{fragment_mapping.numel}")
            for state_key in flat_state.keys():
                dump_param_fragment(dir, tp_index, dp_index, state_key, flat_state[state_key], name,
                                    fragment_mapping.start, fragment_mapping.numel)


def extract_zero_shards_stage3(optim_files, param_shapes, dp_degree, temp_dir, dp_index):
    state_dict = torch.load(optim_files[dp_index], map_location='cpu', weights_only=False)

    flat_state = dict(
        exp_avg=state_dict[OPTIMIZER_STATE_DICT]['optimizer_state_dict']['state'][0]["exp_avg"],
        exp_avg_sq=state_dict[OPTIMIZER_STATE_DICT]['optimizer_state_dict']['state'][0]["exp_avg_sq"],
        fp32=state_dict[OPTIMIZER_STATE_DICT]['fp32_flat_groups'][0],
    )

    offset = 0
    for name, shape in param_shapes.items():
        unpartitioned_numel = shape.numel()
        partitioned_numel, _ = _zero_partitioned_param_info(unpartitioned_numel, dp_degree)
        padding_free_numel = min(partitioned_numel, abs(unpartitioned_numel - dp_index * partitioned_numel))
        for state_key in flat_state.keys():
            dump_param_fragment(temp_dir, 0, dp_index, state_key, flat_state[state_key], name, offset,
                                padding_free_numel)
        offset += partitioned_numel


cnt = 0


def dp_index_to_str(dp_index):
    return f"{dp_index:0>2d}"


def dump_param_fragment(dir, tp_index, dp_index, state_name, state_flat_tensor, param_name, offset, numel):

    global cnt  # temp hack

    param_base_path = os.path.join(dir, param_name, str(tp_index))
    os.makedirs(param_base_path, exist_ok=True)

    cnt += 1

    path = os.path.join(param_base_path, f"{state_name}.{dp_index_to_str(dp_index)}")

    #print(f"{param_name}: {offset}: {numel} => {path}")

    # State might be a python int or a tensor
    if state_name != "step" and torch.is_tensor(state_flat_tensor):
        state_flat_tensor = state_flat_tensor.narrow(0, offset, numel).clone()
    _save_checkpoint(path, state_flat_tensor)


def _merge_zero_shards(param_base_path, state, tp_degree, slice_shape=None):
    slices = []
    for tp_index in range(tp_degree):
        prefix_path = os.path.join(param_base_path, str(tp_index), f"{state}")
        paths = glob.glob(f"{prefix_path}.*")

        if len(paths) == 0:
            continue

        pattern = re.compile(f"{prefix_path}\\.([0-9]+)")
        dp_indices = set()
        for p in paths:
            m = pattern.match(p)
            if m:
                dp_indices.add(int(m.group(1)))
            else:
                raise ValueError(f"Cannot parse dp_rank from {p}")

        paths = [f"{prefix_path}.{dp_index_to_str(dp_index)}" for dp_index in sorted(list(dp_indices))]
        shards = [torch.load(p, weights_only=False) for p in paths]

        if state == "step":
            assert all(v == shards[0] for v in shards), "All shards must have the same step value"
            slice = shards[0]
        else:
            if slice_shape is None:
                slice = torch.cat(shards, dim=0)
            else:
                slice = torch.cat(shards, dim=0).reshape(slice_shape)

        slices.append(slice)
    return slices


def merge_tp_slices(ds_checkpoint, dir, slice_dir, tp_degree, name_and_shape):

    name, shape = name_and_shape
    slice_base_path = os.path.join(slice_dir, name)
    param_base_path = os.path.join(dir, name)

    universal_checkpoint_info = ds_checkpoint.get_checkpoint_info(UNIVERSAL_CHECKPOINT_INFO)
    replicated_parameters = universal_checkpoint_info.get(TP_REPLICATED_PARAMETER_PATTERNS, [])
    parameters_to_average = universal_checkpoint_info.get(PARAMETER_TO_AVERAGE_PATTERNS, [])
    parameters_with_row_parallelism = universal_checkpoint_info.get(PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, [])
    vocabulary_parameters = universal_checkpoint_info.get(VOCABULARY_PARAMETER_PATTERNS, [])
    parameters_with_2_sub_params_cat_dim_0 = universal_checkpoint_info.get(PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0, [])
    parameter_with_sub_params = universal_checkpoint_info.get(PARAMETER_WITH_SUB_PARAMS, [])

    unmatched_patterns = set(replicated_parameters + parameters_to_average + parameters_with_row_parallelism +
                             vocabulary_parameters + parameters_with_2_sub_params_cat_dim_0)
    unmatched_patterns.update(chain.from_iterable(SubparamShape(**s).patterns for s in parameter_with_sub_params))

    def get_matched_pattern(patterns_, name_):
        matched_ = [pattern_ for pattern_ in patterns_ if re.match(pattern_, name_)]
        assert len(matched_) <= 1, f'Got more than one matching patterns={matched_} for {name_}'
        if matched_:
            pattern_ = matched_[0]
            unmatched_patterns.discard(pattern_)
            return pattern_
        return None

    def get_matched_sub_params_pattern(name_):
        for subparam_shape_dict in parameter_with_sub_params:
            subparam_shape = SubparamShape(**subparam_shape_dict)
            for pattern_ in subparam_shape.patterns:
                if re.match(pattern_, name_):
                    unmatched_patterns.discard(pattern_)
                    return subparam_shape
        return None

    matched_sub_params_shape = get_matched_sub_params_pattern(name)

    step_merged = _merge_zero_shards(slice_base_path, "step", tp_degree, shape)
    if step_merged:
        _save_checkpoint(os.path.join(param_base_path, f"step.pt"), step_merged[0])

    for state in ("fp32", "exp_avg", "exp_avg_sq"):
        slices = _merge_zero_shards(slice_base_path, state, tp_degree, shape)
        final_path = os.path.join(param_base_path, f"{state}.pt")

        #print(f"Expected shape: {shape}")
        #print(f"Fragment sizes:", list(frag.shape for frag in slices))
        ckpt_dict = {}
        if get_matched_pattern(replicated_parameters, name):
            if len(slices) > 1:
                assert all([slices[0].equal(other_slice) for other_slice in slices[1:]])
            param = slices[0]
            # print(f'replicate {name} using first slice')
        elif get_matched_pattern(parameters_to_average, name):
            param = sum(slices) / len(slices)
            # print(f'merge {name} using average')
        elif get_matched_pattern(parameters_with_2_sub_params_cat_dim_0, name):
            cat_dim = 0
            chunked_slices = [torch.chunk(s, 2, dim=cat_dim) for s in slices]
            merged_chunks_0 = torch.cat([s[0] for s in chunked_slices], dim=cat_dim)
            merged_chunks_1 = torch.cat([s[1] for s in chunked_slices], dim=cat_dim)
            param = torch.cat([merged_chunks_0, merged_chunks_1], dim=cat_dim)
            ckpt_dict[CAT_DIM] = cat_dim
            ckpt_dict[PARAM_N_SUB_PARAMS] = 2
        elif matched_sub_params_shape:
            merged_chunks = []
            partition_dim = matched_sub_params_shape.partition_dim

            sub_dim_sizes = matched_sub_params_shape.shape[partition_dim]
            if not isinstance(sub_dim_sizes, tuple):
                sub_dim_sizes = (sub_dim_sizes, )

            partition_shape = [sum(d) if isinstance(d, tuple) else d for d in matched_sub_params_shape.shape]
            partition_shape = [d // tp_degree if i == partition_dim else d for i, d in enumerate(partition_shape)]
            slices = [s.view(partition_shape) for s in slices]

            offset = 0
            for sub_dim_size in sub_dim_sizes:
                part_sub_dim_size = sub_dim_size // tp_degree
                merged_chunks.append(
                    torch.cat([s.narrow(partition_dim, offset, part_sub_dim_size) for s in slices], dim=partition_dim))
                offset += part_sub_dim_size
            param = torch.cat(merged_chunks, dim=partition_dim)
            ckpt_dict[SUB_PARAM_SHAPE] = matched_sub_params_shape
        else:
            cat_dim = 1 if get_matched_pattern(parameters_with_row_parallelism, name) else 0
            # print(f"merge {name} with CAT DIM: {cat_dim}")
            param = torch.cat(slices, dim=cat_dim)
            ckpt_dict[CAT_DIM] = cat_dim

        if get_matched_pattern(vocabulary_parameters, name):
            #print(f"Before {param.shape=}")
            # strip padding
            original_vocab_size = universal_checkpoint_info['original_vocab_size']
            param = param[:original_vocab_size, :]
            ckpt_dict[VOCAB_TENSOR] = True
            #print(f"After {param.shape=}")

        #print(f"Final shape: {param.shape}")
        ckpt_dict[PARAM] = param
        _save_checkpoint(final_path, ckpt_dict)

    return unmatched_patterns


def merge_zero3_slices(dp_degree, dir, slice_dir, name):
    slice_base_path = os.path.join(slice_dir, name)
    param_base_path = os.path.join(dir, name)

    for state in ("fp32", "exp_avg", "exp_avg_sq"):
        slices = _merge_zero_shards(slice_base_path, state, 1)
        final_path = os.path.join(param_base_path, f"{state}.pt")
        _save_checkpoint(final_path, slices[0])


def _do_parallel_work(do_work, work_chunks, num_workers):
    results = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_list = [executor.submit(do_work, work) for work in work_chunks]
            for f in tqdm.tqdm(future_list):
                results.append(f.result())
    else:
        # No parallel pass for unit testing
        # We can't create child processes in tests
        for work in tqdm.tqdm(work_chunks):
            results.append(do_work(work))
    return results


def _extract_zero_shard_files(args, ds_checkpoint, temp_dir):
    _3d_range_list = list(
        itertools.product(range(ds_checkpoint.pp_degree), range(ds_checkpoint.tp_degree),
                          range(ds_checkpoint.dp_degree)))
    #pprint(f'{_3d_range_list=}')

    do_work = partial(extract_zero_shards, temp_dir, ds_checkpoint)
    _do_parallel_work(do_work, _3d_range_list, args.num_extract_workers)


def _extract_zero_shard_files_stage3(args, optim_files, param_shapes, dp_degree, temp_dir):
    do_work = partial(extract_zero_shards_stage3, optim_files, param_shapes, dp_degree, temp_dir)
    _do_parallel_work(do_work, list(range(dp_degree)), args.num_extract_workers)


def _merge_tp_slice_files(args, ds_checkpoint, slice_shapes, temp_dir):
    zero_output_folder = os.path.join(args.output_folder, "zero")
    do_work = partial(merge_tp_slices, ds_checkpoint, zero_output_folder, temp_dir, ds_checkpoint.tp_degree)
    unmatched_patterns_lists = _do_parallel_work(do_work, list(slice_shapes.items()), args.num_merge_workers)

    # verify that all patterns were used
    # if a pattern was not used by any of the workers, then it was not used at all -> assert/alert
    sets = [set(lst) for lst in unmatched_patterns_lists]
    unmatched_patterns = list(set.intersection(*sets))
    if args.strict:
        assert not unmatched_patterns, f'Unused patterns={unmatched_patterns} while merging tp slices'
    elif unmatched_patterns:
        print(f'Warning: Unused patterns={unmatched_patterns} while merging tp slices')


def _merge_zero3_slice_files(args, param_shapes, dp_degree, temp_dir):
    zero_output_folder = os.path.join(args.output_folder, "zero")
    do_work = partial(merge_zero3_slices, dp_degree, zero_output_folder, temp_dir)
    _do_parallel_work(do_work, param_shapes.keys(), args.num_merge_workers)


def _zero_partitioned_param_info(unpartitioned_numel, world_size):
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def _parse_model_states_stage3(files):
    return torch.load(files[0], map_location=torch.device('cpu'), weights_only=False)[PARAM_SHAPES]


def _save_optimizer_state(args, ds_checkpoint):
    sharded_states = [BASE_OPTIMIZER_STATE, PARAM_SLICE_MAPPINGS, SINGLE_PARTITION_OF_FP32_GROUPS]
    sd = ds_checkpoint.get_zero_checkpoint_state(pp_index=0, tp_index=0, dp_index=0)

    optim_sd = sd[OPTIMIZER_STATE_DICT]
    output_sd = {k: v for k, v in optim_sd.items() if k not in sharded_states}
    output_sd[PARAM_GROUPS] = optim_sd[BASE_OPTIMIZER_STATE][PARAM_GROUPS]
    zero_output_folder = os.path.join(args.output_folder, "zero")
    output_file_path = os.path.join(zero_output_folder, f"optimizer_state.pt")
    _save_checkpoint(output_file_path, output_sd)


def _save_optimizer_state_stage3(args, optim_files):
    sd = torch.load(optim_files[0], map_location=torch.device('cpu'), weights_only=False)
    output_sd = sd[OPTIMIZER_STATE_DICT]
    output_sd[PARAM_GROUPS] = output_sd[OPTIMIZER_STATE_DICT][PARAM_GROUPS]
    zero_output_folder = os.path.join(args.output_folder, "zero")
    output_file_path = os.path.join(zero_output_folder, f"optimizer_state.pt")
    _save_checkpoint(output_file_path, output_sd)


def _get_optim_files(checkpoint_dir):
    return _get_checkpoint_files(checkpoint_dir, "*_optim_states.pt")


def _get_model_state_files(checkpoint_dir):
    return _get_checkpoint_files(checkpoint_dir, "*_model_states.pt")


def _get_checkpoint_files(checkpoint_dir, glob_pattern):
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def _get_zero_stage(optim_files):
    state_dict = torch.load(optim_files[0], map_location=torch.device('cpu'), weights_only=False)
    optimizer_state = state_dict[OPTIMIZER_STATE_DICT]
    zero_stage = optimizer_state.get(ZERO_STAGE, 1)
    return zero_stage


def _inject_missing_state(ds_checkpoint):
    if UNIVERSAL_CHECKPOINT_INFO not in ds_checkpoint.global_state:
        sd = torch.load(ds_checkpoint.mp_rank_files[0], map_location=torch.device('cpu'), weights_only=False)
        if UNIVERSAL_CHECKPOINT_INFO not in sd:
            ds_checkpoint.global_state[UNIVERSAL_CHECKPOINT_INFO] = {}
            ds_checkpoint.global_state[UNIVERSAL_CHECKPOINT_INFO][
                UNIVERSAL_CHECKPOINT_VERSION_KEY] = UNIVERSAL_CHECKPOINT_VERSION_VALUE


def _check_for_required_state(ds_checkpoint):
    universal_checkpoint_info = ds_checkpoint.get_checkpoint_info(UNIVERSAL_CHECKPOINT_INFO)
    assert universal_checkpoint_info is not None, f'Required {UNIVERSAL_CHECKPOINT_INFO} state is missing in checkpoint. Verify that client creates this state.'


def main(args):
    print(f'Convert DeepSpeed Checkpoint to Universal Checkpoint')

    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to Universal checkpoint in {args.output_folder}')

    optim_files = _get_optim_files(args.input_folder)
    zero_stage = _get_zero_stage(optim_files)

    if zero_stage <= 2:
        ds_checkpoint = DeepSpeedCheckpoint(args.input_folder)
        if args.inject_missing_state:
            _inject_missing_state(ds_checkpoint)
        else:
            _check_for_required_state(ds_checkpoint)

        iteration = ds_checkpoint.get_iteration()
        #_create_latest_file(args.output_folder, iteration)
        checkpoint_paths = _create_checkpoint_paths(args.output_folder, iteration, ds_checkpoint.tp_degree,
                                                    ds_checkpoint.pp_degree)

        slice_shapes = []
        for mp_rank_file in ds_checkpoint.mp_rank_files:
            mp_sd = torch.load(mp_rank_file, map_location=torch.device('cpu'), weights_only=False)
            slice_shapes += mp_sd[PARAM_SHAPES]

        # fix back to normal flat dict, merge duplicates for tp>1
        slice_shapes = dict((k, v) for d in slice_shapes for k, v in d.items())
        temp_dir = os.path.join(args.output_folder, 'tmp')

        print('*** 1. Extracting ZeRO fragments')
        _extract_zero_shard_files(args, ds_checkpoint, temp_dir)

        print('*** 2. Merging slices .....')
        _merge_tp_slice_files(args, ds_checkpoint, slice_shapes, temp_dir)

        print('*** 3. Saving common optimizer states')
        _save_optimizer_state(args, ds_checkpoint)

        if not args.keep_temp_folder:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Copy mp* files into output folder
        for f in glob.glob(os.path.join(args.input_folder, 'mp*')):
            shutil.copy2(f, args.output_folder)

    else:
        model_files = _get_model_state_files(args.input_folder)
        param_shapes = _parse_model_states_stage3(model_files)
        param_shapes = {k: v for d in param_shapes for k, v in d.items()}
        dp_degree = len(model_files)

        temp_dir = os.path.join(args.output_folder, 'tmp')

        print('*** 1. Extracting ZeRO fragments')
        _extract_zero_shard_files_stage3(args, optim_files, param_shapes, dp_degree, temp_dir)

        print('*** 2. Merging slices .....')
        _merge_zero3_slice_files(args, param_shapes, dp_degree, temp_dir)

        print('*** 3. Saving common optimizer states')
        _save_optimizer_state_stage3(args, optim_files)

        if not args.keep_temp_folder:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Copy *model_states files into output folder
        for f in glob.glob(os.path.join(args.input_folder, '*model_states.pt')):
            shutil.copy2(f, args.output_folder)

    # Update latest to output folder
    checkpoint_root_folder, step_folder = os.path.split(args.output_folder)
    latest_file = os.path.join(checkpoint_root_folder, 'latest_universal')
    with open(latest_file, "w") as f:
        f.write(step_folder)

    print('*** Done!')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
