"""
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os
import argparse
import re

RAW_RATE = 'raw_rate'
E2E_RATE = 'e2e_rate'
SUBMIT_LATENCY = 'submit_latency'
COMPLETE_LATENCY = 'complete_latency'
READ_SPEED = 'read_speed'
WRITE_SPEED = 'write_speed'

TASK_READ_SPEED = 'task_read_speed'

PERF_METRICS = [
    RAW_RATE,
    E2E_RATE,
    SUBMIT_LATENCY,
    COMPLETE_LATENCY,
    READ_SPEED,
    WRITE_SPEED
]
METRIC_SEARCH = {
    RAW_RATE: 'ds_raw_time',
    E2E_RATE: 'ds_time',
    SUBMIT_LATENCY: 'aggr: submit',
    COMPLETE_LATENCY: 'aggr: complete',
    READ_SPEED: 'E2E Read Speed',
    WRITE_SPEED: 'E2E Write Speed'
}

NUM_BYTES = (400 * 1024 * 1024)
NUM_GIGA_BYTES = (1024 * 1024 * 1024)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir',
                        type=str,
                        required=True,
                        help='Folder of statistics logs')

    parser.add_argument(
        '--metric',
        type=str,
        required=True,
        help=
        'Performance metric to report: [raw_rate|e2e_rate|submit_latency|complete_latency]'
    )

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
    f, _ = os.path.splitext(file)
    fields = f.split('_')
    for key in fields:
        if key[0] == 't':
            return int(key[1:])
    return 1


def get_metric(file, metric):
    thread_count = get_thread_count(file)
    num_giga_bytes = NUM_BYTES / NUM_GIGA_BYTES
    with open(file) as f:
        for line in f.readlines():
            if line.startswith(METRIC_SEARCH[metric]):
                if metric == RAW_RATE:
                    fields = line.split()
                    raw_time_sec = float(fields[2]) / 1e06
                    raw_rate = (thread_count * num_giga_bytes * 1.0) / raw_time_sec
                    return raw_rate
                elif metric in [READ_SPEED, WRITE_SPEED]:
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

    if not os.path.isdir(args.logdir):
        print(f'{args.logdir} folder is not existent')
        return False

    return True


def get_results(log_files, metric):
    results = {}
    for f in log_files:
        file_key = get_file_key(f)
        value = get_metric(f, metric)
        results[file_key] = value

    return results


def main():
    print("Parsing aio statistics")
    args = parse_arguments()

    if not validate_args(args):
        quit()

    log_files = [
        f for f in os.listdir(args.logdir)
        if os.path.isfile(os.path.join(args.logdir,
                                       f))
    ]

    log_files_path = [os.path.join(args.logdir, f) for f in log_files]
    results = get_results(log_files_path, args.metric)
    result_keys = list(results.keys())
    sorted_keys = sorted(result_keys)
    for k in sorted_keys:
        print(f'{k} = {results[k]}')


if __name__ == "__main__":
    main()
