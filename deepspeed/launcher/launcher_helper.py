# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import sys
import argparse
import subprocess
from deepspeed.utils import logger
from deepspeed.launcher.constants import MPICH_LAUNCHER


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="DeepSpeed launcher helper to map environment variables for"
                                     "multi-node/multi-gpu training jobs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--launcher",
                        default=MPICH_LAUNCHER,
                        type=str,
                        help="(optional) choose launcher backend for multi-node "
                        "training. Options currently include MPICH.")

    parser.add_argument("--module",
                        action="store_true",
                        help="Change each process to interpret the launch "
                        "script as a Python module, executing with the same "
                        "behavior as 'python -m'.")

    parser.add_argument("--no_python",
                        action="store_true",
                        help="Skip prepending the training script with "
                        "'python' - just execute it directly.")

    parser.add_argument("user_script", type=str, help="User script to launch, followed by any required "
                        "arguments.")

    parser.add_argument('user_args', nargs=argparse.REMAINDER)

    parser.add_argument("--bind_cores_to_rank",
                        action="store_true",
                        help="Bind each rank to different cores of the host")

    parser.add_argument("--bind_core_list",
                        type=str,
                        default=None,
                        help="List of cores to bind to with comma separated list of "
                        "numbers and range. i.e. 1,3-5,7 => [1,3,4,5,7].  When not "
                        "specified, all cores on system would be used rank binding")

    return parser.parse_args(args=args)


def env_mapping(env, rank_name_list=None, local_rank_name_list=None):
    rank = None
    for rank_name in rank_name_list:
        if rank_name in env:
            if rank == None:
                rank = env.get(rank_name)
            elif rank != env.get(rank_name):
                raise EnvironmentError(f"rank number doesn't match!")
    if rank == None:
        raise EnvironmentError(f"rank number is not in current env!")
    env['RANK'] = rank

    local_rank = None
    for local_rank_name in local_rank_name_list:
        if local_rank_name in env:
            if local_rank == None:
                local_rank = env.get(local_rank_name)
            elif local_rank != env.get(local_rank_name):
                raise EnvironmentError(f"local_rank number doesn't match!")
    if local_rank == None:
        raise EnvironmentError(f"rank number is not in current env!")
    env['LOCAL_RANK'] = local_rank

    return env


def main(args=None):
    args = parse_args(args)

    env = os.environ.copy()

    args.launcher = args.launcher.lower()
    if args.launcher == MPICH_LAUNCHER:
        rank_name_list = ["PMIX_RANK"] + ["PMI_RANK"]
        local_rank_name_list = ["PALS_LOCAL_RANKID"] + ["MPI_LOCALRANKID"]
        env = env_mapping(env, rank_name_list=rank_name_list, local_rank_name_list=local_rank_name_list)
    else:
        raise NotImplementedError(f"Unknown launcher {args.launcher}")

    python_exec = []
    if not args.no_python:
        python_exec += [sys.executable, "-u"]
        if args.module:
            python_exec.append("-m")
    cmd = python_exec + [args.user_script] + args.user_args

    logger.info(f"launcher_helper cmd = {' '.join(cmd)}")

    result = subprocess.Popen(cmd, env=env, close_fds=False)
    result.wait()


if __name__ == "__main__":
    main()
