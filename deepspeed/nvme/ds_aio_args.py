# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import argparse
import os
from .test_ds_aio_utils import refine_integer_value
from deepspeed.accelerator import get_accelerator

MAPPING_DELIMITER = ':'


def refine_args(args):
    if args.io_size and type(args.io_size) == str:
        args.io_size = refine_integer_value(args.io_size)

    if args.block_size and type(args.block_size) == str:
        args.block_size = refine_integer_value(args.block_size)

    return args


def _get_mapping_dict(args):
    if args.folder is not None:
        d = {i: args.folder for i in range(args.multi_process)}
    else:
        d = {}
        for m in args.folder_to_device_mapping:
            fields = m.split(MAPPING_DELIMITER)
            d[fields[1]] = fields[0]

    return d


def _validate_folder_mapping(args):
    no_error = True
    error_messages = []
    invalid_mappings = [m for m in args.folder_to_device_mapping if MAPPING_DELIMITER not in m]
    if len(invalid_mappings) > 0:
        error_messages.append(
            f'Missing delimiter ({MAPPING_DELIMITER}) in folder_to_device_mapping {invalid_mappings}')
        no_error = False

    folder_list = [m.split(MAPPING_DELIMITER)[0] for m in args.folder_to_device_mapping]
    invalid_folders = [d for d in folder_list if not os.path.exists(d)]
    if len(invalid_folders) > 0:
        error_messages.append(f'Invalid folders in folder_to_device_mapping: {invalid_folders}')
        no_error = False

    if args.gpu:
        device_list = [int(m.split(MAPPING_DELIMITER)[1]) for m in args.folder_to_device_mapping]
        invalid_device_list = [dev_id for dev_id in device_list if not dev_id < get_accelerator().device_count()]
        if len(invalid_device_list) > 0:
            error_messages.append(f'Invalid device ids in folder_to_device_mapping: {invalid_device_list}')
            no_error = False

    return no_error, error_messages


def validate_args(args):
    no_error = True
    error_messages = []

    if args.folder is not None and len(args.folder_to_device_mapping) > 0:
        error_messages.append(f'--folder and --folder_to_device_mapping cannot be specified together.')
        no_error = False
    elif args.folder is None and len(args.folder_to_device_mapping) == 0:
        error_messages.append(f'At least one of --folder or --folder_to_device_mapping must be specified.')
        no_error = False

    # Validate --folder
    if args.folder is not None and not os.path.exists(args.folder):
        no_error = False
        error_messages.append(f'Invalid folder in --folder: {args.folder} ')

    # Validate --folder_mapping_to_device
    if len(args.folder_to_device_mapping) > 0:
        no_mapping_error, mapping_error_messages = _validate_folder_mapping(args)
        no_error = no_error and no_mapping_error
        error_messages += mapping_error_messages

    # Validate --gpu, --use_gds
    if args.use_gds and not args.gpu:
        error_messages.append(f'--gpu must be set to transfer with --use_gds')
        no_error = False

    if not no_error:
        print(f'Found {len(error_messages)} validation errors')
        for i, msg in enumerate(error_messages):
            print(f'{i+1}: {msg}')

    return no_error


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', default=None, type=str, help='Folder to use for I/O.')

    parser.add_argument('--folder_to_device_mapping',
                        default=[],
                        nargs='+',
                        help='Specification of mapping of folder to (gpu) device id, (ignored for cpu accesses).'
                        'Can be specified multiple times for multi-process runs,'
                        'e.g. --folder_to_device_mapping /mnt/nvme0:0 --folder_to_device_mapping /mnt/nvme1:15 --gpu'
                        'means access /mnt/nvme0 with gpu 0 and /mnt/nvme1 with gpu 15')

    parser.add_argument('--io_size', type=str, default=None, required=True, help='Number of bytes to read or write.')

    parser.add_argument('--read', action='store_true', help='Perform read I/O (default is write)')

    parser.add_argument('--multi_process',
                        type=int,
                        default=1,
                        help='Number of parallel processes doing I/O (default 1).')

    parser.add_argument('--block_size',
                        type=str,
                        default='1M',
                        help='I/O block size. Can use K, M, or G suffix (default 1M for 1 megabytes).')

    parser.add_argument('--queue_depth', type=int, default=32, help='I/O queue depth (default 32).')

    parser.add_argument('--single_submit',
                        action='store_true',
                        help='Submit I/O requests in singles (default is submit queue_depth amount at once.).')

    parser.add_argument(
        '--sequential_requests',
        action='store_true',
        help=
        'Delay I/O request submission until completion of prior requests (default is overlap I/O submission and completion requests.).'
    )

    parser.add_argument('--validate', action='store_true', help='Perform validation of I/O transfer in library.')

    parser.add_argument('--handle', action='store_true', help='Use AIO handle.')

    parser.add_argument('--loops', type=int, default=3, help='Count of operation repetitions')

    parser.add_argument('--io_parallel', type=int, default=None, help='Per iop parallelism')

    parser.add_argument('--gpu', action='store_true', help='Use GPU memory')

    parser.add_argument('--use_gds', action='store_true', help='Enable GDS AIO')

    parser.add_argument('--slow_bounce_buffer',
                        action='store_true',
                        help='For GPU memory transfers, measure impact of bounce buffer pinning on critical path.')

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def get_validated_args():
    args = parse_arguments()
    args = refine_args(args)
    if not validate_args(args):
        quit()
    print(f'Successful validation of command line arguments')

    peer_tag = 'gpu' if args.gpu else 'process'
    args.mapping_dict = _get_mapping_dict(args)
    args.mapping_list = [(device_id, folder) for device_id, folder in args.mapping_dict.items()]
    assert len(args.mapping_dict) == len(args.mapping_list)
    print(f'Configuring {len(args.mapping_list)} {peer_tag} to folder mapping')
    for i, (device_id, folder) in enumerate(args.mapping_list):
        print(f'[{i}]: {peer_tag} {device_id} <----> {folder}')

    return args
