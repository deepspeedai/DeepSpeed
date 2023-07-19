# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# return a list of list for cores to numa mapping
# [
#     [ cores for numa 0 ]
#     [ cores belong to numa 1 ]
#     ...
# ]

import distutils
import os
import psutil
import subprocess


# return a list of list for cores to numa mapping
# [
#     [ cores for numa 0 ]
#     [ cores belong to numa 1 ]
#     ...
# ]
def get_numa_cores():
    ret = []
    output = subprocess.check_output(['numactl', '--hardware']).decode("utf-8")
    lines = output.split('\n')
    for line in lines:
        if line.startswith('available:'):
            num_numas = int(line.split(' ')[1])
            break
    for numa in range(num_numas):
        for line in lines:
            if line.startswith(f'node {numa} cpus:'):
                cores = line.split(' ')[3:]
                ret.append([int(core) for core in cores])
    return ret


def check_for_numactl_pkg():
    libs = dict(
        dpkg=["-l", "numactl", "apt"],
        pacman=["-Q", "numactl", "pacman"],
        rpm=["-q", "numactl", "yum"],
    )

    found = False
    for pkgmgr, data in libs.items():
        flag, lib, tool = data
        path = distutils.spawn.find_executable(pkgmgr)
        if path is not None:
            cmd = f"{pkgmgr} {flag} {lib}"
            result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            if result.wait() == 0:
                found = True
            else:
                print(f"please install the {lib} package with {tool}")
            break
    return found


def parse_range(rng):
    try:
        value = int(rng)
        return range(value, value + 1)
    except ValueError:
        # value is not a single number
        parts = rng.split('-')
        if len(parts) != 2:
            raise ValueError("Bad range: '%s', range must be either a number or two number separated by dash" %
                             (rng, ))
        start = int(parts[0])
        end = int(parts[1])
        if start > end:
            raise ValueError("Bad range: '%s', range end must larger than or equal to start" % (rng, ))
        return range(start, end + 1)


# parse comma and dash separated range list into list
# i.e. "0,2-4,6" --> [0, 2, 3, 4, 6]
# rules:
# 1. Range list number be comma separated, each item are either a single number,
#    or a range marked by two numbers (both number are included in the range)
# 2. Sub ranges must be in ascend order and not overlap with each other
# 3. No space in the range expression
def parse_range_list(range_str):
    number_list = []
    last = -1
    range_list = range_str.split(',')
    for sub_range in range_list:
        sub_number_list = parse_range(sub_range)
        if sub_number_list[0] <= last:
            raise ValueError(
                "Bad range: '%s', sub ranges must not overlap with each other and should be in ascend order" %
                (range_str, ))
        last = sub_number_list[-1]
        number_list.extend(sub_number_list)
    return number_list


def get_numactl_cmd(bind_core_list, num_local_procs, local_rank):
    numactl_cmd = []
    check_for_numactl_pkg()
    if 'KMP_AFFINITY' in os.environ.keys():
        raise ValueError("Environment variable KMP_AFFINITY conflicts with numactl "
                         "because it interfere with how many CPU cores numactl can set. "
                         "Unset KMP_AFFINITY before launching deepspeed.\n\n"
                         "\t$ unset KMP_AFFINITY\n"
                         "\t$ deepspeed <deepspeed command parameters>")
    if bind_core_list is not None:
        core_list = parse_range_list(bind_core_list)
        total_cores = len(core_list)
    else:
        total_cores = psutil.cpu_count(logical=False)
        core_list = range(total_cores)
    cores_per_rank = total_cores // num_local_procs
    assert cores_per_rank >= 1, "At least one core needs to be assigned to each rank"
    core_list_for_rank = core_list[cores_per_rank * local_rank:cores_per_rank * (local_rank + 1)]
    numactl_cmd.append("numactl")

    # check if all cores belong to same numa, if true, bind process to that numa domain with -m parameter
    numa_cores = get_numa_cores()
    num_numas = len(numa_cores)

    numa_mode = "normal"

    non_empty_numa_list = []
    empty_numa_list = []
    previous_numa_cores = []
    numa_node_list = []
    numa_node_list_list = []
    for i in range(num_numas):
        # look for empty numa which is HBM numa
        if numa_cores[i] == []:
            empty_numa_list.append(i)
        else:
            non_empty_numa_list.append(i)

            # check for fakenuma
            if numa_cores[i] == previous_numa_cores:
                if numa_node_list == []:
                    #first duplication, add previous node into list
                    numa_node_list.append(i - 1)
                numa_node_list.append(i)
            else:
                if numa_node_list != []:
                    numa_node_list_list.append(numa_node_list)
                    numa_node_list = []
        previous_numa_cores = numa_cores[i]
    if numa_node_list != []:
        numa_node_list_list.append(numa_node_list)

    if empty_numa_list != [] and len(empty_numa_list) == len(non_empty_numa_list):
        numa_mode = "flat_hbm"
        numa_dict = dict(zip(non_empty_numa_list, empty_numa_list))
    elif numa_node_list_list != []:
        numa_mode = "fake"

    if numa_mode == "normal":
        for i in range(num_numas):
            if set(core_list_for_rank) <= set(numa_cores[i]):
                numactl_cmd.append("-m")
                numactl_cmd.append(f"{i}")
                break
    elif numa_mode == "flat_hbm":
        for i in range(num_numas):
            if set(core_list_for_rank) <= set(numa_cores[i]):
                numactl_cmd.append("-p")
                numactl_cmd.append(f"{numa_dict[i]}")
                break
    elif numa_mode == "fake":
        for i in range(num_numas):
            if set(core_list_for_rank) <= set(numa_cores[i]):
                for nodes in numa_node_list_list:
                    if i in nodes:
                        numactl_cmd.append("-m")
                        numactl_cmd.append(f"{','.join(map(str, nodes))}")
                        break
                # the following construct break the outer loop if inner loop breaks
                else:
                    continue
                break

    numactl_cmd.append("-C")
    last_core = core_list_for_rank[0]
    first_core = last_core
    core_list_str = f"{last_core}"
    for core_id in core_list_for_rank[1:]:
        if core_id == last_core + 1:
            last_core = core_id
            continue
        else:
            if first_core == last_core:
                core_list_str = f"{core_list_str},{core_id}"
            else:
                core_list_str = f"{core_list_str}-{last_core},{core_id}"
            first_core = core_id
            last_core = core_id
    if first_core != last_core:
        core_list_str = f"{core_list_str}-{last_core}"
    numactl_cmd.append(f"{core_list_str}")
    return cores_per_rank, numactl_cmd
