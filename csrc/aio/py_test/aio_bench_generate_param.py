# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""
import os
import argparse
import json
from parse_aio_stats import READ_SPEED, WRITE_SPEED, get_sorted_results
from perf_sweep_utils import BENCH_LOG_DIR, READ_LOG_DIR, WRITE_LOG_DIR


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir',
                        type=str,
                        default=BENCH_LOG_DIR,
                        help=f'Folder of performance sweep logs. Default is {os.path.join(".", BENCH_LOG_DIR)}')

    args = parser.parse_args()
    print(f'args = {args}')

    return args


def validate_args(args):
    for d in [READ_LOG_DIR, WRITE_LOG_DIR]:
        log_dir = os.path.join(args.log_dir, d)
        if not os.path.isdir(log_dir):
            print(f'{log_dir} folder is not existent')
            return False

    return True


def convert_to_param(key):
    assert len(key) == 6
    return {
        "single_submit": "true" if key[0] == "single" else "false",
        "overlap_events": "true" if key[1] == "overlap" else "false",
        "thread_count": int(key[3]),
        "queue_depth": int(key[4]),
        "block_size": int(key[5])
    }


def generate_aio_param(read_log_dir, write_log_dir):
    _, read_results = get_sorted_results(read_log_dir, READ_SPEED)
    _, write_results = get_sorted_results(write_log_dir, WRITE_SPEED)
    combined_perf = {key[1:]: value for key, value in read_results.items()}

    for key, value in write_results.items():
        new_key = key[1:]
        if new_key in combined_perf:
            combined_perf[new_key] += value
        else:
            combined_perf[new_key] = 0

    optimal_key = None
    optimal_perf = 0.0
    for key, value in combined_perf.items():
        if value > optimal_perf:
            optimal_perf = value
            optimal_key = key

    aio_param = {"aio": convert_to_param(optimal_key)}

    read_perf_keys = {key[1:]: key for key in read_results.keys()}
    write_perf_keys = {key[1:]: key for key in write_results.keys()}
    optimal_config_read = read_results.get(read_perf_keys[optimal_key], None)
    optimal_config_write = write_results.get(write_perf_keys[optimal_key], None)

    print(f'Best performance (GB/sec): read = {optimal_config_read:5.2f}, write = {optimal_config_write:5.2f}')
    print(json.dumps(aio_param, indent=3))


def main():
    print('Generate aio param')
    args = parse_arguments()
    if not validate_args(args):
        quit()

    read_log_dir = os.path.join(args.log_dir, READ_LOG_DIR)
    write_log_dir = os.path.join(args.log_dir, WRITE_LOG_DIR)
    generate_aio_param(read_log_dir, write_log_dir)


if __name__ == "__main__":
    main()
