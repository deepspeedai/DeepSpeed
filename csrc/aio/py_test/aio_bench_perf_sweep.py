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
import subprocess
import shutil

from test_ds_aio_utils import refine_integer_value
from perf_sweep_utils import READ_OP_DESC, WRITE_OP_DESC, BENCH_LOG_DIR, \
    READ_IO_DIR, WRITE_IO_DIR, READ_LOG_DIR, WRITE_LOG_DIR
from deepspeed.ops.op_builder import AsyncIOBuilder

OTHER_OPTIONS = '--handle'
PERF_SCRIPT = 'test_ds_aio.py'
DEFAULT_SWEEP_CONFIG = {
    "block_size": ["128K", "256K"],
    "queue_depth": [4, 16, 32],
    "overlap_events": [True, False],
    "io_parallel": [2, 8],
    "single_submit": [False]
}


class Job(object):

    def __init__(self, cmd_line, output_file=None, work_dir=None):
        self.cmd_line = cmd_line
        self.output_file = output_file
        self.work_dir = work_dir
        self.output_fd = None

    def cmd(self):
        return self.cmd_line

    def get_stdout(self):
        return self.output_fd

    def get_stderr(self):
        return self.output_fd

    def get_cwd(self):
        return self.work_dir

    def open_output_file(self):
        if self.output_file is not None:
            self.output_fd = open(self.output_file, 'w')

    def close_output_file(self):
        if self.output_fd is not None:
            self.output_fd.close()
            self.output_fd = None


class SweepConfig(object):

    def __init__(self, args):
        self.nvme_dir = args.nvme_dir
        self.io_size = args.io_size
        self.search_space = get_sweep_config_dict(args.sweep_config)
        self.read = not args.no_read
        self.write = not args.no_write
        self.flush_cache = not args.no_sudo
        self.log_dir = args.log_dir
        self.loops = args.loops
        self.other_options = f'{OTHER_OPTIONS} --loops {args.loops}'


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nvme_dir',
                        required=True,
                        type=str,
                        help='Directory in which to perform I/O tests. A writeable directory on a NVMe device.')

    parser.add_argument('--sweep_config', type=str, default=None, help='Performance sweep configuration json file.')

    parser.add_argument('--no_read', action='store_true', help='Disable read performance measurements.')

    parser.add_argument('--no_write', action='store_true', help='Disable write performance measurements.')

    parser.add_argument('--io_size',
                        type=str,
                        default="400M",
                        help='Number of I/O bytes to read/write for performance measurements.')

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


def run_job(job):
    args = ' '.join(job.cmd())
    print(f'args = {args}')
    job.open_output_file()
    proc = subprocess.run(args=args, shell=True, stdout=job.get_stdout(), stderr=job.get_stderr(), cwd=job.get_cwd())
    job.close_output_file()
    assert proc.returncode == 0, \
    f"This command failed: {job.cmd()}"


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
            tags[fields[0]] = fields[1]
    return tags


def get_log_file(io_op_desc, cmd_line):
    QUEUE_DEPTH = "--queue_depth"
    BLOCK_SIZE = "--block_size"
    SINGLE_SUBMIT = "--single_submit"
    OVERLAP_EVENTS = "--overlap_events"
    THREAD_COUNT = "--threads"
    IO_PARALLEL = "--io_parallel"

    tag_map = {
        QUEUE_DEPTH: "d",
        BLOCK_SIZE: "bs",
        SINGLE_SUBMIT: "single",
        OVERLAP_EVENTS: "overlap",
        THREAD_COUNT: "t",
        IO_PARALLEL: "p"
    }

    tag_default = {
        QUEUE_DEPTH: 1,
        BLOCK_SIZE: "1M",
        SINGLE_SUBMIT: "block",
        OVERLAP_EVENTS: "sequential",
        THREAD_COUNT: 1,
        IO_PARALLEL: 1
    }

    def get_default_value(tag):
        value = tag_default[tag]
        if tag in [SINGLE_SUBMIT, OVERLAP_EVENTS]:
            return value
        return f'{tag_map[tag]}{value}'

    def get_config_value(tag, value):
        tag_key = tag_map[tag]
        if value is None:
            return tag_key
        return f'{tag_key}{value}'

    tag_list = [SINGLE_SUBMIT, OVERLAP_EVENTS, THREAD_COUNT, IO_PARALLEL, QUEUE_DEPTH, BLOCK_SIZE]
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


def get_block_size_and_count(io_bytes):
    block_size = 1
    block_count = io_bytes
    bytes_in_KB = 1024

    while block_count % bytes_in_KB == 0:
        block_size *= bytes_in_KB
        block_count /= bytes_in_KB

    return int(block_size), int(block_count)


def create_read_file(sweep_config):
    read_folder = os.path.join(sweep_config.nvme_dir, f'{READ_IO_DIR}')
    os.makedirs(read_folder, exist_ok=True)
    read_file_name = os.path.join(read_folder, f'random_{sweep_config.io_size}B.pt')
    block_size, block_count = get_block_size_and_count(refine_integer_value(sweep_config.io_size))
    dd_job = Job(cmd_line=[f'dd if=/dev/urandom of={read_file_name} bs={block_size} count={block_count}'])
    print(f'[Start] Create read file of {sweep_config.io_size} bytes by running {dd_job.cmd()} ....')
    run_job(dd_job)
    print(f'[Done] Create read file of {sweep_config.io_size} bytes by running {dd_job.cmd()} ....')
    return read_folder, read_file_name


def remove_folder(folder):
    assert os.path.isdir(folder), f"Error: cannot remove {folder} - folder not found"
    shutil.rmtree(folder)


def run_read_sweep(sweep_config, flush_cache_job, sync_job, cmd_lines):
    read_folder, read_file_name = create_read_file(sweep_config)
    read_option = f'--read_file {read_file_name}'
    read_cmd_lines = [[f'{read_option} {sweep_config.other_options}'] + cmd for cmd in cmd_lines]
    #dump_cmd_lines(read_cmd_lines)

    log_folder = os.path.join(sweep_config.log_dir, f'{READ_LOG_DIR}')
    os.makedirs(log_folder, exist_ok=True)

    perf_jobs = create_perf_jobs(io_op_desc=READ_OP_DESC, log_dir=log_folder, cmd_lines=read_cmd_lines)

    launch_sweep(sweep_jobs=perf_jobs, sync_job=sync_job, flush_cache_job=flush_cache_job)

    remove_folder(read_folder)


def run_write_sweep(sweep_config, flush_cache_job, sync_job, cmd_lines):
    write_folder = os.path.join(sweep_config.nvme_dir, f'{WRITE_IO_DIR}')
    os.makedirs(write_folder, exist_ok=True)
    write_file_name = os.path.join(write_folder, f'random_{sweep_config.io_size}B.pt')
    write_option = f'--write_size {sweep_config.io_size} --write_file {write_file_name}'
    write_cmd_lines = [[f'{write_option} {sweep_config.other_options}'] + cmd for cmd in cmd_lines]
    #dump_cmd_lines(write_cmd_lines)

    log_folder = os.path.join(sweep_config.log_dir, f'{WRITE_LOG_DIR}')
    os.makedirs(log_folder, exist_ok=True)

    perf_jobs = create_perf_jobs(io_op_desc=WRITE_OP_DESC, log_dir=log_folder, cmd_lines=write_cmd_lines)

    launch_sweep(sweep_jobs=perf_jobs, sync_job=sync_job, flush_cache_job=flush_cache_job)

    remove_folder(write_folder)


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
