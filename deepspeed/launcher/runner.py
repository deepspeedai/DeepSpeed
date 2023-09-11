# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed runner is the main front-end to launching multi-worker
training jobs with DeepSpeed. By default this uses pdsh to parallel
ssh into multiple worker nodes and launch all the necessary processes
per rank for training.
"""

import os
import re
import sys
import shlex
import json
import base64
import argparse
import subprocess
import collections
from copy import deepcopy
import signal
import time

from .multinode_runner import PDSHRunner, OpenMPIRunner, MVAPICHRunner, SlurmRunner, MPICHRunner, IMPIRunner
from .constants import PDSH_LAUNCHER, OPENMPI_LAUNCHER, MVAPICH_LAUNCHER, SLURM_LAUNCHER, MPICH_LAUNCHER, IMPI_LAUNCHER
from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from ..nebula.constants import NEBULA_EXPORT_ENVS
from ..utils import logger

from ..autotuning import Autotuner
from deepspeed.accelerator import get_accelerator

DLTS_HOSTFILE = "/job/hostfile"
EXPORT_ENVS = ['MLFLOW', 'NCCL', 'PYTHON', 'MV2', 'UCX']
EXPORT_ENVS += NEBULA_EXPORT_ENVS
DEEPSPEED_ENVIRONMENT_NAME = os.getenv("DS_ENV_FILE", ".deepspeed_env")
DEEPSPEED_ENVIRONMENT_PATHS = [os.path.expanduser("~"), '.']
PDSH_MAX_FAN_OUT = 1024

# On AISC compute, each node sets environment variables independently, want to prevent
# exporting rank-0 env variables in case of heterogeneous compute.
EXCLUDE_ENVS = {'AISC_JOB_NAME': ['NCCL_IB_HCA', 'UCX_NET_DEVICES']}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="DeepSpeed runner to help launch distributed "
                                     "multi-node/multi-gpu training jobs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-H",
                        "--hostfile",
                        type=str,
                        default=DLTS_HOSTFILE,
                        help="Hostfile path (in MPI style) that defines the "
                        "resource pool available to the job (e.g., "
                        "worker-0 slots=4)")

    parser.add_argument("-i",
                        "--include",
                        type=str,
                        default="",
                        help='''Specify hardware resources to use during execution.
                        String format is
                                NODE_SPEC[@NODE_SPEC ...],
                        where
                                NODE_SPEC=NAME[:SLOT[,SLOT ...]].
                        If :SLOT is omitted, include all slots on that host.
                        Example: -i "worker-0@worker-1:0,2" will use all slots
                        on worker-0 and slots [0, 2] on worker-1.
                        ''')

    parser.add_argument("-e",
                        "--exclude",
                        type=str,
                        default="",
                        help='''Specify hardware resources to NOT use during execution.
                        Mutually exclusive with --include. Resource formatting
                        is the same as --include.
                        Example: -e "worker-1:0" will use all available
                        resources except slot 0 on worker-1.
                        ''')

    parser.add_argument("--num_nodes",
                        type=int,
                        default=-1,
                        help="Total number of worker nodes to run on, this will use "
                        "the top N hosts from the given hostfile.")

    parser.add_argument("--min_elastic_nodes",
                        type=int,
                        default=-1,
                        help="Minimum number of nodes to run elastic training on. "
                        "Default is 1 when elastic training is enabled")

    parser.add_argument("--max_elastic_nodes",
                        type=int,
                        default=-1,
                        help="Maximum number of nodes to run elastic training on. "
                        "Default is num_nodes when elastic training is enabled")

    parser.add_argument("--num_gpus",
                        "--num_accelerators",
                        type=int,
                        default=-1,
                        help="Max number of GPUs to use on each node, will use "
                        "[0:N) GPU ids on each node.")

    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="(optional) Port used by PyTorch distributed for "
                        "communication during training.")

    parser.add_argument("--master_addr",
                        default="",
                        type=str,
                        help="(optional) IP address of node 0, will be "
                        "inferred via 'hostname -I' if not specified.")

    parser.add_argument("--launcher",
                        default=PDSH_LAUNCHER,
                        type=str,
                        help="(optional) choose launcher backend for multi-node "
                        "training. Options currently include PDSH, OpenMPI, MVAPICH, SLURM, MPICH, IMPI.")

    parser.add_argument("--launcher_args",
                        default="",
                        type=str,
                        help="(optional) pass launcher specific arguments as a "
                        "single quoted argument.")

    parser.add_argument("--module",
                        action="store_true",
                        help="Change each process to interpret the launch "
                        "script as a Python module, executing with the same "
                        "behavior as 'python -m'.")

    parser.add_argument("--no_python",
                        action="store_true",
                        help="Skip prepending the training script with "
                        "'python' - just execute it directly.")

    parser.add_argument("--no_local_rank",
                        action="store_true",
                        help="Do not pass local_rank as an argument when calling "
                        "the user's training script.")

    parser.add_argument("--no_ssh_check",
                        action="store_true",
                        help="Do not perform ssh check in multi-node launcher model")

    parser.add_argument("--force_multi",
                        action="store_true",
                        help="Force multi-node launcher mode, helps in cases where user "
                        "wants to launch on single remote node.")

    parser.add_argument("--save_pid",
                        action="store_true",
                        help="Save file containing launcher process id (pid) at /tmp/<main-pid>.ds, "
                        "where <main-pid> is the pid of the first process that invoked `deepspeed`. "
                        "Useful when launching deepspeed processes programmatically.")

    parser.add_argument("--enable_each_rank_log",
                        default="None",
                        type=str,
                        help="redirect the stdout and stderr from each rank into different log files")

    parser.add_argument("--autotuning",
                        default="",
                        choices=["tune", "run"],
                        type=str,
                        help="Run DeepSpeed autotuner to discover optimal configuration parameters "
                        "before running job.")

    parser.add_argument("--elastic_training",
                        action="store_true",
                        help="Enable elastic training support in DeepSpeed.")

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

    parser.add_argument("--ssh_port", type=int, default=None, help="SSH port to use for remote connections")

    return parser.parse_args(args=args)


def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        logger.warning("Unable to find hostfile, will proceed with training "
                       "with local resources only.")
        return None

    # e.g., worker-0 slots=16
    with open(hostfile_path, 'r') as fd:
        hostfile_text = fd.readlines()

    return _parse_hostfile(hostfile_text)


def _parse_hostfile(hostfile_lines):
    # Regex matches one or more non-whitespace characters (\S+) at the start of
    # the line, followed by one or more whitespace characters (\s+), followed
    # by the string "slots=", followed by one or more digits (\d+).
    pattern = r'^(\S+)\s+slots=(\d+)'

    resource_pool = collections.OrderedDict()

    for line in hostfile_lines:
        line = line.strip()
        match = re.search(pattern, line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            if host in resource_pool:
                logger.error(f"Bad hostfile text: {hostfile_lines}")
                raise ValueError(f"Hostfile contains multiple entries for {host}, unable to proceed with launching")
            resource_pool[host] = num_slots
        else:
            logger.error(f"Bad hostfile text: {hostfile_lines}")
            raise ValueError(f"Hostfile contains a bad entry: {line}, unable to proceed with launching")

    if len(resource_pool) == 0:
        logger.error(f"Bad hostfile text: {hostfile_lines}")
        raise ValueError("Hostfile is empty or not formatted correctly, unable to proceed with launching.")

    return resource_pool


def _stable_remove_duplicates(data):
    # Create a new list in the same order as original but with duplicates
    # removed, should never be more than ~16 elements so simple is best
    new_list = []
    for x in data:
        if x not in new_list:
            new_list.append(x)
    return new_list


def parse_resource_filter(host_info, include_str="", exclude_str=""):
    '''Parse an inclusion or exclusion string and filter a hostfile dictionary.

    String format is NODE_SPEC[@NODE_SPEC ...], where
        NODE_SPEC = NAME[:SLOT[,SLOT ...]].
    If :SLOT is omitted, include/exclude all slots on that host.

    Examples:
        include_str="worker-0@worker-1:0,2" will use all slots on worker-0 and
          slots [0, 2] on worker-1.
        exclude_str="worker-1:0" will use all available resources except
          slot 0 on worker-1.
    '''

    # Constants that define our syntax
    NODE_SEP = '@'
    SLOT_LIST_START = ':'
    SLOT_SEP = ','

    # Ensure include/exclude are mutually exclusive
    if (include_str != "") and (exclude_str != ""):
        raise ValueError('include_str and exclude_str are mutually exclusive.')

    # no-op
    if (include_str == "") and (exclude_str == ""):
        return host_info

    # Either build from scratch or remove items
    filtered_hosts = dict()
    if include_str:
        parse_str = include_str
    if exclude_str != "":
        filtered_hosts = deepcopy(host_info)
        parse_str = exclude_str

    # foreach node in the list
    for node_config in parse_str.split(NODE_SEP):
        # Node can either be alone or node:slot,slot,slot
        if SLOT_LIST_START in node_config:
            hostname, slots = node_config.split(SLOT_LIST_START)
            slots = [int(x) for x in slots.split(SLOT_SEP)]

            # sanity checks
            if hostname not in host_info:
                raise ValueError(f"Hostname '{hostname}' not found in hostfile")
            for slot in slots:
                if slot not in host_info[hostname]:
                    raise ValueError(f"No slot '{slot}' specified on host '{hostname}'")

            # If include string, build the list from here
            if include_str:
                filtered_hosts[hostname] = slots
            elif exclude_str:
                for slot in slots:
                    logger.info(f'removing {slot} from {hostname}')
                    filtered_hosts[hostname].remove(slot)

        # User just specified the whole node
        else:
            hostname = node_config
            # sanity check hostname
            if hostname not in host_info:
                raise ValueError(f"Hostname '{hostname}' not found in hostfile")

            if include_str:
                filtered_hosts[hostname] = host_info[hostname]
            elif exclude_str:
                filtered_hosts[hostname] = []

    # Post-processing to remove duplicates and empty nodes
    del_keys = []
    for hostname in filtered_hosts:
        # Remove duplicates
        filtered_hosts[hostname] = _stable_remove_duplicates(filtered_hosts[hostname])
        # Remove empty hosts
        if len(filtered_hosts[hostname]) == 0:
            del_keys.append(hostname)
    for name in del_keys:
        del filtered_hosts[name]

    # Lastly, go over filtered_hosts and convert to a OrderedDict() to ensure
    # we map ranks to nodes correctly by maintaining host_info ordering.
    ordered_hosts = collections.OrderedDict()
    for host in host_info:
        if host in filtered_hosts:
            ordered_hosts[host] = filtered_hosts[host]

    return ordered_hosts


def parse_inclusion_exclusion(resource_pool, inclusion, exclusion):
    active_resources = collections.OrderedDict()
    for hostname, slots in resource_pool.items():
        active_resources[hostname] = list(range(slots))

    return parse_resource_filter(active_resources, include_str=inclusion, exclude_str=exclusion)


def encode_world_info(world_info):
    world_info_json = json.dumps(world_info).encode('utf-8')
    world_info_base64 = base64.urlsafe_b64encode(world_info_json).decode('utf-8')
    return world_info_base64


def run_autotuning(args, active_resources):
    tuner = Autotuner(args, active_resources)
    logger.info("[Start] Running autotuning")

    tuner.tune()
    tuner.print_tuning_results()

    logger.info("[End] Running autotuning")
    tuner.write_optimal_config()

    if args.autotuning == "run":
        tuner.run_after_tuning()


def parse_num_nodes(str_num_nodes: str, elastic_training: bool):
    node_list = str_num_nodes.split(":")

    if len(node_list) == 1:
        min_nodes, max_nodes = int(node_list[0]), -1
    elif len(node_list) == 2 and elastic_training:
        min_nodes, max_nodes = int(node_list[0]), int(node_list[1])
    elif len(node_list) == 2 and not elastic_training:
        raise RuntimeError("MIN:MAX format is only supported in elastic training")
    else:
        raise RuntimeError("num_nodes {} is not in MIN:MAX format".format(str_num_nodes))

    return min_nodes, max_nodes


def main(args=None):
    args = parse_args(args)

    # For when argparse interprets remaining args as a single string
    args.user_args = shlex.split(" ".join(list(map(lambda x: x if x.startswith("-") else f"'{x}'", args.user_args))))

    if args.elastic_training:
        assert args.master_addr != "", "Master Addr is required when elastic training is enabled"

    resource_pool = fetch_hostfile(args.hostfile)

    # respect CUDA_VISIBLE_DEVICES for a single node and no explicit resource filters
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not resource_pool and len(cuda_visible_devices):
        detected_str = f"Detected CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
        if len(args.include) or len(args.exclude) or args.num_nodes > 1 or args.num_gpus > 0:
            print(
                f"{detected_str} but ignoring it because one or several of --include/--exclude/--num_gpus/--num_nodes cl args were used. If you want to use CUDA_VISIBLE_DEVICES don't pass any of these arguments to deepspeed."
            )
        else:
            args.include = f"localhost:{cuda_visible_devices}"
            print(f"{detected_str}: setting --include={args.include}")
        del os.environ["CUDA_VISIBLE_DEVICES"]

    if args.num_nodes >= 0 or args.num_gpus >= 0:
        if args.include != "" or args.exclude != "":
            raise ValueError("Cannot specify num_nodes/gpus with include/exclude")

    multi_node_exec = True
    if not resource_pool:
        resource_pool = {}
        device_count = get_accelerator().device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool['localhost'] = device_count
        args.master_addr = "127.0.0.1"
        multi_node_exec = False

    if not multi_node_exec and args.num_nodes > 1:
        raise ValueError("Num nodes is >1 but no extra nodes available via hostfile")

    active_resources = parse_inclusion_exclusion(resource_pool, args.include, args.exclude)
    env = os.environ.copy()

    # validate that passwordless-ssh is workly properly with this hostfile
    if multi_node_exec and not args.no_ssh_check:
        first_host = list(active_resources.keys())[0]
        try:
            ssh_check_cmd = "ssh -o PasswordAuthentication=no "
            if args.ssh_port is not None:
                ssh_check_cmd += f"-p {args.ssh_port} "
            ssh_check_cmd += f"{first_host} hostname"
            subprocess.check_call(ssh_check_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"Using hostfile at {args.hostfile} but host={first_host} was not reachable via ssh. If you are running with a single node please remove {args.hostfile} or setup passwordless ssh."
            )

    if not args.master_addr:
        assert multi_node_exec
        first_host = list(active_resources.keys())[0]
        hostname_cmd = [f"ssh {first_host} hostname -I"]
        try:
            result = subprocess.check_output(hostname_cmd, shell=True)
        except subprocess.CalledProcessError as err:
            logger.error(
                "Unable to detect suitable master address via `hostname -I`, please manually specify one via --master_addr"
            )
            raise err
        args.master_addr = result.decode('utf-8').split()[0]
        if not args.master_addr:
            raise RuntimeError(
                f"Unable to detect suitable master address via `hostname -I`, please manually specify one via --master_addr"
            )
        logger.info(f"Using IP address of {args.master_addr} for node {first_host}")

    if args.autotuning != "":
        run_autotuning(args, active_resources)
        return

    if args.num_nodes > 0:
        updated_active_resources = collections.OrderedDict()
        for count, hostname in enumerate(active_resources.keys()):
            if args.num_nodes == count:
                break
            updated_active_resources[hostname] = active_resources[hostname]
        active_resources = updated_active_resources

    if args.num_gpus > 0:
        updated_active_resources = collections.OrderedDict()
        for hostname in active_resources.keys():
            updated_active_resources[hostname] = list(range(args.num_gpus))
        active_resources = updated_active_resources

    if args.elastic_training:
        assert not args.no_local_rank, "--no_local_rank argument is not supported in Elastic training"

    # encode world info as base64 to make it easier to pass via command line
    world_info_base64 = encode_world_info(active_resources)

    multi_node_exec = args.force_multi or len(active_resources) > 1

    if not multi_node_exec:
        deepspeed_launch = [
            sys.executable, "-u", "-m", "deepspeed.launcher.launch", f"--world_info={world_info_base64}",
            f"--master_addr={args.master_addr}", f"--master_port={args.master_port}"
        ]
        if args.no_python:
            deepspeed_launch.append("--no_python")
        if args.module:
            deepspeed_launch.append("--module")
        if args.no_local_rank:
            deepspeed_launch.append("--no_local_rank")
        if args.save_pid:
            deepspeed_launch += ["--save_pid", f"{os.getpid()}"]
        if args.enable_each_rank_log:
            deepspeed_launch.append(f"--enable_each_rank_log={args.enable_each_rank_log}")
        if args.elastic_training:
            deepspeed_launch.append("--enable_elastic_training")
            deepspeed_launch.append(f"--max_elastic_nodes={args.max_elastic_nodes}")
            deepspeed_launch.append(f"--min_elastic_nodes={args.min_elastic_nodes}")
        if args.bind_cores_to_rank:
            deepspeed_launch.append("--bind_cores_to_rank")
        if args.bind_core_list is not None:
            deepspeed_launch.append(f"--bind_core_list={args.bind_core_list}")
        cmd = deepspeed_launch + [args.user_script] + args.user_args
    else:
        args.launcher = args.launcher.lower()
        if args.launcher == PDSH_LAUNCHER:
            runner = PDSHRunner(args, world_info_base64)
        elif args.launcher == OPENMPI_LAUNCHER:
            runner = OpenMPIRunner(args, world_info_base64, resource_pool)
        elif args.launcher == MPICH_LAUNCHER:
            runner = MPICHRunner(args, world_info_base64, resource_pool)
        elif args.launcher == IMPI_LAUNCHER:
            runner = IMPIRunner(args, world_info_base64, resource_pool)
        elif args.launcher == MVAPICH_LAUNCHER:
            runner = MVAPICHRunner(args, world_info_base64, resource_pool)
        elif args.launcher == SLURM_LAUNCHER:
            runner = SlurmRunner(args, world_info_base64, resource_pool)
        else:
            raise NotImplementedError(f"Unknown launcher {args.launcher}")

        if not runner.backend_exists():
            raise RuntimeError(f"launcher '{args.launcher}' not installed.")

        curr_path = os.path.abspath('.')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = curr_path + ":" + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = curr_path

        excluded_vars = []
        for exclude_key, var_list in EXCLUDE_ENVS.items():
            if exclude_key in env.keys():
                # key exists in launcher env -> var list should be used
                excluded_vars += var_list

        exports = ""
        for var in env.keys():
            if any([var.startswith(name) for name in EXPORT_ENVS]):
                if not any([var == name for name in excluded_vars]):
                    runner.add_export(var, env[var])

        for environ_path in DEEPSPEED_ENVIRONMENT_PATHS:
            environ_file = DEEPSPEED_ENVIRONMENT_NAME
            # handle if users to enter path for `DS_ENV_FILE`
            if not os.path.isfile(environ_file):
                environ_file = os.path.join(environ_path, DEEPSPEED_ENVIRONMENT_NAME)
            if os.path.isfile(environ_file):
                logger.info(f"deepspeed_env file = {environ_file}")
                with open(environ_file, 'r') as fd:
                    for var in fd.readlines():
                        key, val = var.split('=', maxsplit=1)
                        runner.add_export(key, val)

        if args.launcher == PDSH_LAUNCHER:
            cmd, kill_cmd, env = runner.get_cmd(env, active_resources)
        else:
            cmd = runner.get_cmd(env, active_resources)

    logger.info(f"cmd = {' '.join(cmd)}")
    result = subprocess.Popen(cmd, env=env)

    def sigkill_handler(signum, frame):
        result.send_signal(signal.SIGINT)
        time.sleep(0.1)
        result.send_signal(signal.SIGTERM)
        result_kill = subprocess.Popen(kill_cmd, env=env)
        result_kill.wait()
        time.sleep(1)
        sys.exit(1)

    if args.launcher == PDSH_LAUNCHER and multi_node_exec:
        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

    result.wait()

    # In case of failure must propagate the error-condition back to the caller (usually shell). The
    # actual error and traceback should have been printed in the subprocess, so in order to avoid
    # unnecessary noise we just quietly exit here with the same code as the subprocess
    if result.returncode > 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
