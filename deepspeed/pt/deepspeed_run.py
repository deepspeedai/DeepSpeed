"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

import os
import sys
import json
import pynvml
import shutil
import base64
import logging
import argparse
import subprocess
import collections
from copy import deepcopy

DLTS_HOSTFILE = "/job/hostfile"


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="DeepSpeed runner to help launch distributed "
        "multi-node/multi-gpu training jobs.")

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

    parser.add_argument("--num_nodes", type=int, default=-1, help="")

    parser.add_argument("--num_gpus", type=int, default=-1, help="")

    parser.add_argument("--master_port",
                        default=29500,
                        type=int,
                        help="(optional) Port used by PyTorch distributed for "
                        "communication during training.")

    parser.add_argument("--master_addr",
                        default="",
                        type=str,
                        help="(optional) IP address of node 0, will be "
                        "inferred via 'hostname -I' if not specified.")

    parser.add_argument("user_script",
                        type=str,
                        help="User script to launch, followed by any required "
                        "arguments.")
    parser.add_argument('user_args', nargs=argparse.REMAINDER)
    return parser.parse_args(args=args)


def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        logging.warning("Unable to find hostfile, will proceed with training "
                        "with local resources only.")
        return None

    # e.g., worker-0 slots=16
    with open(hostfile_path, 'r') as fd:

        resource_pool = collections.OrderedDict()
        for line in fd.readlines():
            try:
                hostname, slots = line.split()
                _, slot_count = slots.split("=")
                slot_count = int(slot_count)
            except ValueError as err:
                logging.error("Hostfile is not formatted correctly, unable to "
                              "proceed with training.")
                raise err
            if hostname in resource_pool:
                logging.error("Hostfile contains duplicate hosts, unable to "
                              "proceed with training.")
                raise ValueError("host {} is already defined".format(hostname))
            resource_pool[hostname] = slot_count

    return resource_pool


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
                raise ValueError("Hostname '{}' not found in hostfile".format(hostname))
            for s in slots:
                if s not in host_info[hostname]:
                    raise ValueError("No slot '{}' specified on host '{}'".format(
                        s,
                        hostname))

            # If include string, build the list from here
            if include_str:
                filtered_hosts[hostname] = slots
            elif exclude_str:
                for s in slots:
                    print('removing {} from {}'.format(s, hostname))
                    filtered_hosts[hostname].remove(s)

        # User just specified the whole node
        else:
            hostname = node_config
            # sanity check hostname
            if hostname not in host_info:
                raise ValueError("Hostname '{}' not found in hostfile".format(hostname))

            if include_str:
                filtered_hosts[hostname] = host_info[hostname]
            elif exclude_str:
                filtered_hosts[hostname] = []

    # Post-processing to remove duplicates and empty nodes
    del_keys = []
    for hostname in filtered_hosts:
        # Remove duplicates
        filtered_hosts[hostname] = list(set(filtered_hosts[hostname]))
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

    return parse_resource_filter(active_resources,
                                 include_str=inclusion,
                                 exclude_str=exclusion)


def local_gpu_count():
    device_count = None
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print("device count", device_count)
        return device_count
    except pynvml.NVMLError:
        logging.error("Unable to get GPU count information, perhaps there are "
                      "no GPUs on this host?")
        return device_count


def encode_world_info(world_info):
    world_info_json = json.dumps(world_info).encode('utf-8')
    world_info_base64 = base64.urlsafe_b64encode(world_info_json).decode('utf-8')
    return world_info_base64


def main(args=None):
    args = parse_args(args)

    if args.num_nodes >= 0 or args.num_gpus >= 0:
        if args.include != "" or args.exclude != "":
            raise ValueError("Cannot specify num_nodes/gpus with include/exclude")

    multi_node_exec = True
    resource_pool = fetch_hostfile(args.hostfile)
    if not resource_pool:
        resource_pool = {}
        device_count = local_gpu_count()
        if device_count is None:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool['localhost'] = device_count
        args.master_addr = "127.0.0.1"
        multi_node_exec = False

    if not multi_node_exec and args.num_nodes > 1:
        raise ValueError("Num nodes is >1 but no extra nodes available via hostfile")

    active_resources = parse_inclusion_exclusion(resource_pool,
                                                 args.include,
                                                 args.exclude)

    if multi_node_exec and not shutil.which('pdsh'):
        raise RuntimeError("pdsh is not installed, unable to proceed")

    env = os.environ.copy()

    if not args.master_addr:
        first_host = list(active_resources.keys())[0]
        hostname_cmd = ["ssh {} hostname -I".format(first_host)]
        result = subprocess.check_output(hostname_cmd, shell=True)
        args.master_addr = result.decode('utf-8').split()[0]
        logging.info("Using IP address of {} for node {}".format(
            args.master_addr,
            first_host))

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

    # encode world info as base64 to make it easier to pass via command line
    world_info_base64 = encode_world_info(active_resources)

    if not multi_node_exec:
        deepspeed_launch = [
            sys.executable,
            "-u",
            "-m",
            "deepspeed.pt.deepspeed_launch",
            "--world_info={}".format(world_info_base64)
        ]
        cmd = deepspeed_launch + [args.user_script] + args.user_args
    else:
        env['PDSH_RCMD_TYPE'] = 'ssh'

        active_workers = ",".join(active_resources.keys())
        logging.info("Running on the following workers: %s" % active_workers)

        pdsh_cmd_args = ['pdsh', '-w', active_workers]

        num_nodes = len(active_resources.keys())
        num_gpus_per_node = None

        curr_path = os.path.abspath('.')

        nccl_export = ""
        for nccl_var in filter(lambda x: "NCCL_" in x, env.keys()):
            nccl_export += "export {}={}; ".format(nccl_var, env[nccl_var])

        deepspeed_launch = [
            nccl_export,
            "cd {};".format(curr_path),
            sys.executable,
            "-u",
            "-m",
            "deepspeed.pt.deepspeed_launch",
            '--world_info={}'.format(world_info_base64),
            "--node_rank=%n",
            "--master_addr={}".format(args.master_addr),
            "--master_port={}".format(args.master_port)
        ]
        cmd = pdsh_cmd_args + deepspeed_launch + [args.user_script] + args.user_args
    print("cmd={}".format(cmd), flush=True)
    result = subprocess.Popen(cmd, env=env)
    result.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s %(asctime)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
