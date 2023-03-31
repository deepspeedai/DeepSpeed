# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os
import argparse
import multiprocessing as mp
from ds_aio_basic import aio_basic_multiprocessing
from ds_aio_handle import aio_handle_multiprocessing
from test_ds_aio_utils import refine_args


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--read_file', type=str, default=None, help='Read file.')

    parser.add_argument('--write_file', type=str, default=None, help='Write file.')

    parser.add_argument('--write_size', type=str, default=None, help='Number of bytes to write.')

    parser.add_argument('--block_size', type=str, default='1M', help='I/O block size.')

    parser.add_argument('--queue_depth', type=int, default=32, help='I/O queue depth.')

    parser.add_argument('--threads', type=int, default=1, help='Thread parallelism count.')

    parser.add_argument('--single_submit',
                        action='store_true',
                        help='Submit I/O requests in singles (default is submit queue_depth amount at once.).')

    parser.add_argument('--overlap_events',
                        action='store_true',
                        help='Overlap I/O submission and completion requests.')

    parser.add_argument('--validate', action='store_true', help='Perform validation in library.')

    parser.add_argument('--handle', action='store_true', help='Use AIO handle.')

    parser.add_argument('--loops', type=int, default=1, help='Count of operation repetitions')

    parser.add_argument('--io_parallel', type=int, default=None, help='Per iop parallelism')

    parser.add_argument('--gpu', action='store_true', help='Use GPU memory')

    parser.add_argument('--use_accelerator_pin_memory',
                        action='store_true',
                        help='Obtain pinned (CPU page-locked) tensors from accelerator')

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def validate_args(args):
    if args.read_file and not os.path.isfile(args.read_file):
        print(f'args validation error: {args.read_file} not found')
        return False

    return True


def main():
    print(f'Testing deepspeed_aio python frontend')

    args = parse_arguments()
    refine_args(args)
    if not validate_args(args):
        quit()

    mp.set_start_method('spawn')
    multiprocess_function = aio_handle_multiprocessing if args.handle else aio_basic_multiprocessing
    if args.read_file:
        multiprocess_function(args, True)

    if args.write_file:
        multiprocess_function(args, False)


if __name__ == "__main__":
    main()
