# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed launcher, this is similar to torch's distributed.launch but supports
additional features such as arbitrary gpu exclusion.

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
import psutil
from collections import defaultdict
from typing import Dict
from argparse import ArgumentParser, REMAINDER
from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from ..nebula.constants import DLTS_POD_ENV_PATH
from ..utils import logger, get_numactl_cmd
from ..elasticity import is_torch_elastic_compatible
from .constants import ELASTIC_TRAINING_ID_DEFAULT

PID_FILE_BASEPATH = "/tmp"


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
    parser.add_argument("--world_info", default="None", type=str, help="world info base64 encoded dictionary")

    parser.add_argument("--module",
                        action="store_true",
                        help="Change each process to interpret the launch "
                        "script as a Python module, executing with the same "
                        "behavior as 'python -m'.")

    parser.add_argument("--no_python",
                        action="store_true",
                        help="Skip prepending the training script with "
                        "'python' - just execute it directly.")

    parser.add_argument("--enable_elastic_training", action="store_true", help="Enable elastic training support.")

    parser.add_argument("--min_elastic_nodes", type=int, default=-1, help="Min number of nodes in elastic training.")

    parser.add_argument("--max_elastic_nodes", type=int, default=-1, help="Max number of nodes in elastic training.")

    parser.add_argument("--no_local_rank",
                        action="store_true",
                        help="Do not pass local_rank as an argument when calling "
                        "the user's training script.")

    parser.add_argument("--save_pid",
                        type=int,
                        default=0,
                        help="main launching process pid, for internal pid tracking")

    parser.add_argument("--enable_each_rank_log",
                        default="None",
                        type=str,
                        help="redirect the stdout and stderr from each rank into different log files")

    parser.add_argument("--bind_cores_to_rank",
                        action="store_true",
                        help="Bind each rank to different cores of the host. "
                        "This improves host efficiency especially for CPU backend")

    parser.add_argument("--bind_core_list",
                        type=str,
                        default=None,
                        help="List of cores to bind to with comma separated list of "
                        "numbers and range. i.e. 1,3-5,7 => [1,3,4,5,7].  When not "
                        "specified, all cores on system would be used rank binding")

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


# Adapted from https://psutil.readthedocs.io/en/latest/#kill-process-tree
def terminate_process_tree(pid):
    process = psutil.Process(pid)
    children = process.children(recursive=True)
    children.append(process)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=30)
    for p in alive:
        p.kill()


def main():
    args = parse_args()
    current_env = os.environ.copy()

    for k in current_env.keys():
        if "NCCL" in k:
            logger.info(f"{args.node_rank} {k}={current_env[k]}")

    if args.world_info == "None":
        raise ValueError("world_info can not be None")
    world_info = base64.urlsafe_b64decode(args.world_info)
    world_info = json.loads(world_info)

    logger.info(f"WORLD INFO DICT: {world_info}")
    node_list = list(world_info.keys())
    args.nnodes = len(node_list)
    local_node = node_list[args.node_rank]
    local_gpu_ids = world_info[local_node]
    num_local_procs = len(local_gpu_ids)
    logger.info(f"nnodes={args.nnodes}, num_local_procs={num_local_procs}, node_rank={args.node_rank}")

    global_rank_mapping = defaultdict(list)
    curr_global_rank = 0
    dist_world_size = 0
    for node_id in node_list:
        gids = world_info[node_id]
        dist_world_size += len(gids)
        for gid in gids:
            global_rank_mapping[node_id].append(curr_global_rank)
            curr_global_rank += 1
    logger.info(f"global_rank_mapping={global_rank_mapping}")
    logger.info(f"dist_world_size={dist_world_size}")
    current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
    logger.info(f"Setting CUDA_VISIBLE_DEVICES={current_env['CUDA_VISIBLE_DEVICES']}")

    # set PyTorch distributed related environmental variables
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["CROSS_RANK"] = str(args.node_rank)
    current_env["CROSS_SIZE"] = str(args.nnodes)
    current_env["LOCAL_SIZE"] = str(num_local_procs)

    if args.save_pid:
        print(f"launcher pid: {os.getpid()}")

    pid_file = None
    if args.save_pid:
        launcher_pid = os.getpid()
        pid_file = os.path.join(PID_FILE_BASEPATH, f"{args.save_pid}.deepspeed")
        assert not os.path.isfile(pid_file), "pid file exists but shouldn't"
        with open(pid_file, 'w') as fd:
            fd.write(f"{launcher_pid}")

    if not is_torch_elastic_compatible():
        if args.enable_elastic_training:
            logger.info(f"Disabling elastic training support as \
                    PyTorch version should be greater than 1.11.x")
            args.enable_elastic_training = False

    if os.path.exists(DLTS_POD_ENV_PATH):
        with open(DLTS_POD_ENV_PATH) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                if line.startswith('export FC_TASKROLE_NAME') or line.startswith('export FC_TASK_INDEX'):
                    key_val = line.split()[1]
                    key, val = key_val.split('=')
                    current_env[key] = val

    processes = []
    cmd = []

    if not args.enable_elastic_training:
        if args.enable_each_rank_log != "None":
            # prepare the log path and the file name prefix
            if os.path.isfile(args.enable_each_rank_log):
                raise ValueError(f"{args.enable_each_rank_log} should not be a file, it should be a directory.")
            if not os.path.exists(args.enable_each_rank_log):
                try:
                    os.makedirs(args.enable_each_rank_log)
                except Exception as e:
                    print(e)
                    raise ValueError(f"unable to create directory {args.enable_each_rank_log} for each rank log.")
            log_name_prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())

        for local_proc in range(0, num_local_procs):
            # each process's rank
            dist_rank = global_rank_mapping[local_node][local_proc]
            local_rank = dist_rank % num_local_procs
            current_env["RANK"] = str(dist_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            cmd = []
            if args.bind_cores_to_rank:
                cores_per_rank, numactl_cmd = get_numactl_cmd(args.bind_core_list, num_local_procs, local_rank)
                current_env["OMP_NUM_THREADS"] = f"{cores_per_rank}"
                cmd = cmd + numactl_cmd
            if not args.no_python:
                cmd.append(sys.executable)
                cmd.append("-u")
                if args.module:
                    cmd.append("-m")
            else:
                if args.module:
                    raise ValueError("Don't use both the '--no_python' flag"
                                     " and the '--module' flag at the same time.")
            cmd.append(args.training_script)
            # A user may not want to pass local_rank as a keyword arg so we make this optional.
            if not args.no_local_rank:
                cmd.append(f"--local_rank={local_rank}")
            cmd += args.training_script_args

            if args.enable_each_rank_log != "None":
                log_file = os.path.join(args.enable_each_rank_log, f"{log_name_prefix}_rank{dist_rank}.log")
                log_fd = open(log_file, 'w')
                process = subprocess.Popen(cmd, env=current_env, stdout=log_fd, stderr=log_fd)
            else:
                process = subprocess.Popen(cmd, env=current_env)

            processes.append(process)
    else:
        from ..elasticity import DSElasticAgent
        from torch.distributed.elastic.rendezvous import RendezvousParameters
        from torch.distributed.elastic.agent.server.api import WorkerSpec
        import torch.distributed.elastic.rendezvous.registry as rdzv_registry
        from torch.distributed.elastic.multiprocessing import Std

        if args.min_elastic_nodes == -1:
            args.min_elastic_nodes = 1
        if args.max_elastic_nodes == -1:
            args.max_elastic_nodes = args.nnodes
        assert args.max_elastic_nodes > 0 and args.min_elastic_nodes > 0, "Max and Min nodes should be positive"

        current_env["NCCL_ASYNC_ERROR_HANDLING"] = str(1)

        # Get config and arguments
        cmd = []
        if not args.no_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag"
                                 " and the '--module' flag at the same time.")
        cmd.append(args.training_script)
        cmd += args.training_script_args
        cmd_args = cmd[1:]

        rdzv_configs: Dict[str, str] = {'timeout': 100}
        run_id = os.environ.get("ELASTIC_RUN_ID", ELASTIC_TRAINING_ID_DEFAULT)

        # Creating config for rendezvous class
        rdzv_parameters = RendezvousParameters(backend='c10d',
                                               endpoint=args.master_addr + ":" + str(args.master_port),
                                               run_id=run_id,
                                               min_nodes=args.min_elastic_nodes,
                                               max_nodes=args.max_elastic_nodes,
                                               **rdzv_configs)

        spec = WorkerSpec(
            role='trainer',
            local_world_size=num_local_procs,
            entrypoint=cmd[0],
            args=cmd[1:],
            rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
            max_restarts=100,
            monitor_interval=5,
            redirects=Std.from_str("0"),
            tee=Std.from_str("0"),
            master_addr=None,
            master_port=None,
        )
        agent = DSElasticAgent(spec, current_env)
        agent.run()

    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum, frame):
        for process in processes:
            logger.info(f"Killing subprocess {process.pid}")
            try:
                terminate_process_tree(process.pid)
            except Exception:
                pass
        if last_return_code is not None:
            logger.error(f"{cmd} exits with return code = {last_return_code}")
            sys.exit(last_return_code)
        if signum in sig_names:
            logger.info(f"Main process received {sig_names[signum]}, exiting")
        if args.save_pid:
            if os.path.isfile(pid_file):
                os.remove(pid_file)
        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

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
                    logger.info(f"Process {process.pid} exits successfully.")
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)


if __name__ == "__main__":
    main()
