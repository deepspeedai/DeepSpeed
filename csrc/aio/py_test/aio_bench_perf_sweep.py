# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""
import os
import sys
import argparse
import json
import itertools
import shutil

from ds_aio_job import Job, run_job
from perf_sweep_utils import READ_OP_DESC, WRITE_OP_DESC, BENCH_LOG_DIR, \
    READ_LOG_DIR, WRITE_LOG_DIR
from deepspeed.ops.op_builder import AsyncIOBuilder

OTHER_OPTIONS = '--handle'
PERF_SCRIPT = 'test_ds_aio.py'
DEFAULT_SWEEP_CONFIG = {
    "block_size": ["128K", "1M"],
    "queue_depth": [32, 64, 128],
    "sequential_requests": [True, False],
    "single_submit": [False],
    "io_parallel": [1, 2, 8],
}


class SweepConfig(object):

    def __init__(self, args):
        self.folder_to_device_mapping = get_ftd_map(args.nvme_dir)
        self.search_space = get_sweep_config_dict(args.sweep_config)
        self.search_space.update(self.folder_to_device_mapping)
        self.read = not args.no_read
        self.write = not args.no_write
        self.flush_cache = not args.no_sudo
        self.log_dir = args.log_dir
        self.other_options = f'{OTHER_OPTIONS} --loops {args.loops} --io_size {args.io_size}'
        if args.gpu:
            self.other_options += ' --gpu'
        if args.gds:
            self.other_options += ' --use_gds'


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nvme_dir',
                        nargs='+',
                        required=True,
                        help='Directory in which to perform I/O tests. A writeable directory on a NVMe device.')

    parser.add_argument('--sweep_config', type=str, default=None, help='Performance sweep configuration json file.')

    parser.add_argument('--no_read', action='store_true', help='Disable read performance measurements.')

    parser.add_argument('--no_write', action='store_true', help='Disable write performance measurements.')

    parser.add_argument('--io_size',
                        type=str,
                        default="400M",
                        help='Number of I/O bytes to read/write for performance measurements.')

    parser.add_argument('--gpu', action='store_true', help='Test tensor transfers between GPU device and NVME device.')

    parser.add_argument('--gds', action='store_true', help='Run the sweep over NVIDIA GPUDirectStorage operator')

    parser.add_argument(
        '--no_sudo',
        action='store_true',
        help=
        'Run without sudo access. Page cache will not be flushed and reported read speeds may be higher than actual.')

    parser.add_argument(
        '--log_dir',
        type=str,
        default=BENCH_LOG_DIR,
        help=f'Output directory for performance log files. Default is {os.path.join(".", BENCH_LOG_DIR)}')

    parser.add_argument('--loops', type=int, default=1, help='Count of operation repetitions')

    args = parser.parse_args()
    print(f'args = {args}')

    return args


def dump_cmd_lines(cmd_lines):
    print(f'cmd line count =  {len(cmd_lines)}')
    for i, cmd in enumerate(cmd_lines):
        print(f'{i}:  {cmd}')


def get_ftd_map(nvme_dir_list):
    ftd_list = [f'{dir}:{dev}' for dev, dir in enumerate(nvme_dir_list)]
    ftd_arg = [' '.join(ftd for ftd in ftd_list)]
    return {'folder_to_device_mapping': ftd_arg}


def get_sweep_config_dict(sweep_config_json):
    if sweep_config_json is None:
        return DEFAULT_SWEEP_CONFIG

    with open(sweep_config_json) as fp:
        sweep_config = json.load(fp)
    return sweep_config


def get_sweep_cmd_lines(sweep_config_dict):

    def flatten_options(key, value_list):
        flat_list = []
        for v in value_list:
            if not type(v) is bool:
                flat_list.append(f'--{key} {v}')
            elif v:
                flat_list.append(f'--{key}')
            else:
                flat_list.append(' ')

        return flat_list

    flat_list = [flatten_options(key, value) for key, value in sweep_config_dict.items()]
    cmd_list = list(itertools.product(*flat_list))
    cmd_list = [list(cmd) for cmd in cmd_list]
    #dump_cmd_lines(cmd_list)
    return cmd_list


def launch_sweep(sweep_jobs, sync_job, flush_cache_job):
    for perf_job in sweep_jobs:
        if flush_cache_job is not None:
            run_job(sync_job)
            run_job(flush_cache_job)

        run_job(perf_job)

        run_job(sync_job)


def create_cmd_tags(cmd_line):
    tags = {}
    for param_value in cmd_line:
        fields = param_value.split()
        if len(fields) == 1:
            tags[fields[0]] = None
        elif len(fields) == 2:
            if fields[0] == '--folder_to_device_mapping':
                tags[fields[0]] = len(fields[1:])
            else:
                tags[fields[0]] = fields[1]
        elif len(fields) > 2:
            tags[fields[0]] = len(fields[1:])
    return tags


def get_log_file(io_op_desc, cmd_line):
    QUEUE_DEPTH = "--queue_depth"
    BLOCK_SIZE = "--block_size"
    SINGLE_SUBMIT = "--single_submit"
    SEQUENTIAL_REQUESTS = "--sequential_requests"
    FTD_MAP = "--folder_to_device_mapping"
    IO_PARALLEL = "--io_parallel"

    tag_map = {
        QUEUE_DEPTH: "d",
        BLOCK_SIZE: "bs",
        SINGLE_SUBMIT: "single",
        SEQUENTIAL_REQUESTS: "sequential",
        FTD_MAP: "ftd",
        IO_PARALLEL: "p"
    }

    tag_default = {
        QUEUE_DEPTH: 1,
        BLOCK_SIZE: "1M",
        SINGLE_SUBMIT: "block",
        SEQUENTIAL_REQUESTS: "overlap",
        FTD_MAP: 1,
        IO_PARALLEL: 1
    }

    def get_default_value(tag):
        value = tag_default[tag]
        if tag in [SINGLE_SUBMIT, SEQUENTIAL_REQUESTS]:
            return value
        return f'{tag_map[tag]}{value}'

    def get_config_value(tag, value):
        tag_key = tag_map[tag]
        if value is None:
            return tag_key
        return f'{tag_key}{value}'

    tag_list = [SINGLE_SUBMIT, SEQUENTIAL_REQUESTS, FTD_MAP, QUEUE_DEPTH, BLOCK_SIZE, IO_PARALLEL]
    log_tags = [io_op_desc]
    cmd_tags = create_cmd_tags(cmd_line)
    for tag in tag_list:
        if tag in cmd_tags:
            log_tags.append(get_config_value(tag, cmd_tags[tag]))
        else:
            log_tags.append(get_default_value(tag))

    log_file = '_'.join(log_tags)
    log_file += '.txt'
    return log_file


def create_perf_jobs(io_op_desc, log_dir, cmd_lines):
    py_cmd = ['python', os.path.join(script_path(), PERF_SCRIPT)]

    perf_jobs = []
    for cmd in cmd_lines:
        log_file = os.path.join(log_dir, get_log_file(io_op_desc, cmd))
        job = Job(cmd_line=py_cmd + cmd, output_file=log_file)
        perf_jobs.append(job)

    return perf_jobs


def script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def async_io_setup():
    return AsyncIOBuilder().is_compatible()


def remove_folder(folder):
    assert os.path.isdir(folder), f"Error: cannot remove {folder} - folder not found"
    shutil.rmtree(folder)


def run_read_sweep(sweep_config, flush_cache_job, sync_job, cmd_lines):
    read_cmd_lines = [[f'--read {sweep_config.other_options}'] + cmd for cmd in cmd_lines]
    #dump_cmd_lines(cmd_lines)

    log_folder = os.path.join(sweep_config.log_dir, f'{READ_LOG_DIR}')
    os.makedirs(log_folder, exist_ok=True)

    perf_jobs = create_perf_jobs(io_op_desc=READ_OP_DESC, log_dir=log_folder, cmd_lines=read_cmd_lines)

    launch_sweep(sweep_jobs=perf_jobs, sync_job=sync_job, flush_cache_job=flush_cache_job)


def run_write_sweep(sweep_config, flush_cache_job, sync_job, cmd_lines):
    write_cmd_lines = [[f'{sweep_config.other_options}'] + cmd for cmd in cmd_lines]
    #dump_cmd_lines(write_cmd_lines)

    log_folder = os.path.join(sweep_config.log_dir, f'{WRITE_LOG_DIR}')
    os.makedirs(log_folder, exist_ok=True)

    perf_jobs = create_perf_jobs(io_op_desc=WRITE_OP_DESC, log_dir=log_folder, cmd_lines=write_cmd_lines)

    launch_sweep(sweep_jobs=perf_jobs, sync_job=sync_job, flush_cache_job=flush_cache_job)


def main():
    print("Running performance sweep of deepspeed nvme library")

    if not async_io_setup():
        error_msg = """
            Failing because environment is not properly configured for deepspeed async i/o module.
            Possible fix: apt install libaio-dev.
        """
        print(error_msg)
        quit()

    args = parse_arguments()
    sweep_config = SweepConfig(args)
    cmd_lines = get_sweep_cmd_lines(sweep_config.search_space)

    if sweep_config.flush_cache:
        flush_cache_job = Job(cmd_line=['sudo', 'bash -c', "'echo 1 > /proc/sys/vm/drop_caches'"])
    else:
        flush_cache_job = None

    sync_job = Job(cmd_line=['sync'])

    if sweep_config.read:
        run_read_sweep(sweep_config, flush_cache_job, sync_job, cmd_lines)

    if sweep_config.write:
        run_write_sweep(sweep_config, flush_cache_job, sync_job, cmd_lines)


if __name__ == "__main__":
    main()
