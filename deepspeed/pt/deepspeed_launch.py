"""
Copyright 2020 The Microsoft DeepSpeed Team: deepspeed@microsoft.com
"""

import sys
import subprocess
import os
import socket
import json
import base64
from collections import defaultdict
from argparse import ArgumentParser, REMAINDER

import torch


def parse_args():
    parser = ArgumentParser(description="DeepSpeed distributed training launch "
                            "utilty that creates multiple distributed "
                            "processes on a single node")

    # Optional arguments for the launch helper
    parser.add_argument("--node_rank",
                        type=int,
                        default=0,
                        help="The rank of the node for multi-node distributed "
                        "training")
    parser.add_argument("--master_addr",
                        default="127.0.0.1",
                        type=str,
                        help="Master node (rank 0)'s address, should be either "
                        "the IP address or the hostname of node 0, for "
                        "single node multi-proc training, the "
                        "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port",
                        default=29500,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communciation during distributed "
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
            print(args.node_rank, k, current_env[k], flush=True)

    world_info = None
    assert args.world_info != "None", "must provide world info dict"
    world_info = base64.urlsafe_b64decode(args.world_info)
    world_info = json.loads(world_info)

    print("WORLD INFO DICT: {}".format(world_info), flush=True)
    node_list = list(world_info.keys())
    args.nnodes = len(node_list)
    local_node = node_list[args.node_rank]
    local_gpu_ids = world_info[local_node]
    num_local_procs = len(local_gpu_ids)
    print("nnodes={}, num_local_procs={}, node_rank={}".format(
        args.nnodes,
        num_local_procs,
        args.node_rank),
          flush=True)

    global_rank_mapping = defaultdict(list)
    curr_global_rank = 0
    dist_world_size = 0
    for node_id in node_list:
        gids = world_info[node_id]
        dist_world_size += len(gids)
        for gid in gids:
            global_rank_mapping[node_id].append(curr_global_rank)
            curr_global_rank += 1
    print("global_rank_mapping={}".format(global_rank_mapping), flush=True)
    print("dist_world_size={}".format(dist_world_size), flush=True)
    current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
    print("Setting CUDA_VISIBLE_DEVICES={}".format(current_env["CUDA_VISIBLE_DEVICES"]),
          flush=True)
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

        # spawn the processes
        cmd = [
            sys.executable,
            "-u",
            args.training_script,
            "--local_rank={}".format(local_rank)
        ] + args.training_script_args
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
