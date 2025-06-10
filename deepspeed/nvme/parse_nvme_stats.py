# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os
import argparse

READ_SPEED = 'read_speed'
WRITE_SPEED = 'write_speed'

PERF_METRICS = [READ_SPEED, WRITE_SPEED]

METRIC_SEARCH = {READ_SPEED: 'E2E Read Speed', WRITE_SPEED: 'E2E Write Speed'}


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, required=True, help='Folder of statistics logs')

    parser.add_argument('--metric',
                        type=str,
                        required=True,
                        help='Performance metric to report: [read_speed|write_speed]')

    args = parser.parse_args()
    print(f'args = {args}')

    return args


def extract_value(key, file):
    INVALID_PREFIXES = ["ds"]
    for p in INVALID_PREFIXES:
        if key.startswith(p):
            return key
    try:
        if key[0] in ['t', 'd', 'p']:
            return int(key[1:])
        if key.startswith("bs"):
            if key.endswith('K'):
                v = key[2:].split('K')
                return int(v[0]) * 1024
            elif key.endswith('M'):
                v = key[2:].split('M')
                return int(v[0]) * 1024 * 1024
            else:
                return int(key[2:])
    except:
        print(f"{file}: extract_value fails on {key}")
        return None

    return key


def get_file_key(file):
    f, _ = os.path.splitext(os.path.basename(file))
    fields = f.split('_')
    values = [extract_value(k, file) for k in fields]
    return tuple(values)


def get_thread_count(file):
    f, _ = os.path.splitext(os.path.basename(file))
    fields = f.split('_')
    for key in fields:
        if key[0] == 't':
            return int(key[1:])
    return 1


"""
Extract performance metric from log file.
Sample file lines are:
Task Read Latency = 0.031647682189941406 sec
Task Read Speed = 12.342926020792527 GB/sec
E2E Read Latency = 0.031697988510131836 sec
E2E Read Speed = 12.323337169333062 GB/sec

For the above sample, -metric = "read_speed" corresponds to "E2E Read Speed", and 12.32 will be returned
"""


def get_metric(file, metric):
    thread_count = get_thread_count(file)
    with open(file) as f:
        for line in f.readlines():
            if line.startswith(METRIC_SEARCH[metric]):
                if metric in [READ_SPEED, WRITE_SPEED]:
                    fields = line.split()
                    return float(fields[-2])
                else:
                    fields = line.split('=')
                    return float(fields[-1])

    return None


def validate_args(args):
    if not args.metric in PERF_METRICS:
        print(f'{args.metric} is not a valid performance metrics')
        return False

    if not os.path.isdir(args.log_dir):
        print(f'{args.log_dir} folder is not existent')
        return False

    return True


def get_results(log_files, metric):
    results = {}
    for f in log_files:
        file_key = get_file_key(f)
        value = get_metric(f, metric)
        results[file_key] = value

    return results


def get_sorted_results(log_dir, metric):
    log_files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]

    log_files_path = [os.path.join(log_dir, f) for f in log_files]
    results = get_results(log_files_path, metric)
    result_keys = list(results.keys())
    sorted_keys = sorted(result_keys)
    return sorted_keys, results


def main():
    print("Parsing aio statistics")
    args = parse_arguments()

    if not validate_args(args):
        quit()

    sorted_keys, results = get_sorted_results(args.log_dir, args.metric)
    for k in sorted_keys:
        print(f'{k} = {results[k]}')


if __name__ == "__main__":
    main()
