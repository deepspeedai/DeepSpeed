# Copyright 2020 The Microsoft DeepSpeed Team
"""
DeepSpeed launcher, this is similar to torch.distributed.launch but supports
additional features such as abitrary gpu exclusion.

deepspeed.launcher.launch is intended to be run on a single worker node and
will spawn several worker sub-processes depending on how many devices/ranks
are on the worker.
"""

import sys
import subprocess
import os
import json
import base64
import time
import signal
from collections import defaultdict
from argparse import ArgumentParser, REMAINDER

from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from ..utils import logger


def parse_args():
    parser = ArgumentParser(description="DeepSpeed distributed training launch"
                            " utility that creates multiple distributed"
                            " processes on a single node")

    # Optional arguments for the launch helper
    parser.add_argument("--node_rank",
                        type=int,
                        default=0,
                        help="The rank of the node for multi-node distributed "
                        "training")
    parser.add_argument("--master_addr",
                        default="127.0.0.1",
                        type=str,
                        help="Master node (rank 0)'s address, should be either"
                        " the IP address or the hostname of node 0, for"
                        " single node multi-proc training, the"
                        " --master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communication during distributed "
                        "training")
    parser.add_argument("--world_info",
                        default="None",
                        type=str,
                        help="world info base64 encoded dictionary")

    # positional
    parser.add_argument("training_script",
                        type=str,
                        help="The full path to the single GPU training "
                        "program/script to be launched in parallel, "
                        "followed by all the arguments for the "
                        "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    current_env = os.environ.copy()

    for k in current_env.keys():
        if "NCCL" in k:
            logger.info("%s %s %s", args.node_rank, k, current_env[k])

    world_info = None
    assert args.world_info != "None", "must provide world info dict"
    world_info = base64.urlsafe_b64decode(args.world_info)
    world_info = json.loads(world_info)

    logger.info("WORLD INFO DICT: {}".format(world_info))
    node_list = list(world_info.keys())
    args.nnodes = len(node_list)
    local_node = node_list[args.node_rank]
    local_gpu_ids = world_info[local_node]
    num_local_procs = len(local_gpu_ids)
    logger.info(
        "nnodes={}, num_local_procs={}, node_rank={}".format(args.nnodes,
                                                             num_local_procs,
                                                             args.node_rank),
    )

    global_rank_mapping = defaultdict(list)
    curr_global_rank = 0
    dist_world_size = 0
    for node_id in node_list:
        gids = world_info[node_id]
        dist_world_size += len(gids)
        for gid in gids:
            global_rank_mapping[node_id].append(curr_global_rank)
            curr_global_rank += 1
    logger.info("global_rank_mapping={}".format(global_rank_mapping))
    logger.info("dist_world_size={}".format(dist_world_size))
    current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
    logger.info("Setting CUDA_VISIBLE_DEVICES={}".format(
        current_env["CUDA_VISIBLE_DEVICES"]))
    exclusion_counts_per_node = None

    # set PyTorch distributed related environmental variables
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []
    for local_rank in range(0, num_local_procs):
        # each process's rank
        dist_rank = global_rank_mapping[local_node][local_rank]
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [
            sys.executable,
            "-u",
            args.training_script,
            "--local_rank={}".format(local_rank)
        ] + args.training_script_args

        sig_names = {2: "SIGINT", 15: "SIGTERM"}
        last_return_code = None

        def sigkill_handler(signum, frame):
            for process in processes:
                print(f"Killing subprocess {process.pid}")
                try:
                    process.kill()
                except Exception as e:
                    pass
            if last_return_code is not None:
                raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
            if signum in sig_names:
                print(f"Main process received {sig_names[signum]}, exiting")
            sys.exit(1)

        # pass SIGINT/SIGTERM to children if the parent is being terminated
        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    alive_processes = set(processes)
    while len(alive_processes):
        finished_processes = []
        for process in alive_processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                if process.returncode != 0:
                    last_return_code = process.returncode  # for sigkill_handler
                    sigkill_handler(signal.SIGTERM, None)  # not coming back
                else:
                    # exited cleanly
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)


if __name__ == "__main__":
    main()
