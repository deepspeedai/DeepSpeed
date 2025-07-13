# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from deepspeed.utils import log_dist


def get_caller_func(frame=3):
    import sys
    return sys._getframe(frame).f_code.co_name


def print_rank_0(message):
    import deepspeed.comm as dist
    if dist.get_rank() == 0:
        print(message)


# Helper function to pretty-print message sizes
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# Helper function to calculate algbw and busbw.
# See https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36 and https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
def calc_bw_log(comm_op, size, duration):
    import deepspeed.comm as dist

    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all_single":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather" or comm_op == "all_gather_into_tensor" or comm_op == "reduce_scatter" or comm_op == "reduce_scatter_tensor":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce" or comm_op == "all_reduce_coalesced" or comm_op == "inference_all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "send" or comm_op == "recv" or comm_op == "isend" or comm_op == "irecv" or comm_op == "broadcast" or comm_op == "reduce" or comm_op == "gather" or comm_op == "scatter" or comm_op == "barrier":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("wrong comm_op specified")  # noqa: F821
        exit(0)

    # convert to Gbps
    tput *= 8
    busbw *= 8

    tput /= 1e6
    busbw /= 1e6

    return tput, busbw


class CommsLogger:

    def __init__(self):
        from deepspeed.comm.constants import COMMS_LOGGER_VERBOSE_DEFAULT, COMMS_LOGGER_DEBUG_DEFAULT, COMMS_LOGGER_PROF_OPS_DEFAULT, COMMS_LOGGER_PROF_ALL_DEFAULT, COMMS_LOGGER_ENABLED_DEFAULT
        self.comms_dict = {}
        self.verbose = COMMS_LOGGER_VERBOSE_DEFAULT
        self.debug = COMMS_LOGGER_DEBUG_DEFAULT
        self.prof_ops = COMMS_LOGGER_PROF_OPS_DEFAULT
        self.prof_all = COMMS_LOGGER_PROF_ALL_DEFAULT
        self.enabled = COMMS_LOGGER_ENABLED_DEFAULT

    def configure(self, comms_config):
        self.enabled = comms_config.comms_logger_enabled
        if self.enabled:
            self.verbose = comms_config.comms_logger.verbose
            self.debug = comms_config.comms_logger.debug
            self.prof_ops = comms_config.comms_logger.prof_ops
            self.prof_all = comms_config.comms_logger.prof_all

    # There are three settings for the op profiler:
    # - Global profiling (profile all comms)
    # - Op-type profiling (e.g. profile all all_reduce comms)
    # - Op profiling (e.g. profile a specific all_reduce op)
    def start_profiling_comms(self):
        self.prof_all = True

    def stop_profiling_comms(self):
        self.prof_all = True

    # E.g. start_profiling_op('all_reduce')
    def start_profiling_op(self, op_name_list):
        self.prof_ops = list(set(self.prof_ops) | set(op_name_list))

    def stop_profiling_op(self, op_name_list):
        self.prof_ops = [op for op in self.prof_ops if op not in op_name_list]

    # Add log entry
    def append(self, raw_name, record_name, latency, msg_size):
        algbw, busbw = calc_bw_log(raw_name, msg_size, latency)
        if record_name in self.comms_dict.keys():
            # If this comm_op has already been logged with this message size, just add to existing record
            if msg_size in self.comms_dict[record_name].keys():
                self.comms_dict[record_name][msg_size][0] += 1
                self.comms_dict[record_name][msg_size][1].append(latency)
                self.comms_dict[record_name][msg_size][2].append(algbw)
                self.comms_dict[record_name][msg_size][3].append(busbw)
            # If this is a new message size for this comm_op, add new record under existing comm_op
            else:
                self.comms_dict[record_name][msg_size] = [1, [latency], [algbw], [busbw]]
        else:
            # Create entirely new record
            self.comms_dict[record_name] = {msg_size: [1, [latency], [algbw], [busbw]]}
        # If verbose, print every comm op
        # TODO: Add to tensorboard
        if self.verbose:
            log_str = f"comm op: {record_name} | time (ms): {latency:.2f} | msg size: {convert_size(msg_size)} | algbw (Gbps): {algbw:.2f} | busbw (Gbps): {busbw:.2f}"
            log_dist(log_str, [0])

    def get_raw_data(self):
        """
        Get the raw communication data dictionary.

        Returns:
            dict: Raw communication data in format {record_name: {msg_size: [count, [latencies], [algbws], [busbws]]}}
        """
        return self.comms_dict.copy()

    def has_data(self):
        """
        Check if any communication data has been logged.

        Returns:
            bool: True if communication data exists, False otherwise
        """
        return len(self.comms_dict) > 0

    def reset_data(self):
        """
        Clear all logged communication data.
        """
        self.comms_dict.clear()

    def get_operation_names(self):
        """
        Get list of all logged communication operation names.

        Returns:
            list: List of operation names that have been logged
        """
        return list(self.comms_dict.keys())

    def get_total_operations(self):
        """
        Get total number of communication operations logged across all types.

        Returns:
            int: Total count of operations
        """
        total = 0
        for record_name in self.comms_dict:
            for msg_size in self.comms_dict[record_name]:
                total += self.comms_dict[record_name][msg_size][0]  # count is at index 0
        return total

    def get_operation_summary(self, operation_name):
        """
        Get summary statistics for a specific operation type.

        Args:
            operation_name (str): Name of the communication operation

        Returns:
            dict: Summary statistics for the operation, or None if operation not found
        """
        if operation_name not in self.comms_dict:
            return None

        from deepspeed.utils.timer import trim_mean

        # Create a snapshot to avoid concurrent modification issues
        op_data = self.comms_dict[operation_name].copy()
        summary = {}

        for msg_size, vals in op_data.items():
            count = vals[0]
            total_lat = sum(vals[1])
            avg_lat = trim_mean(vals[1], 0.1)
            avg_algbw = trim_mean(vals[2], 0.1)
            avg_busbw = trim_mean(vals[3], 0.1)

            summary[msg_size] = {
                "count": count,
                "total_latency_ms": total_lat,
                "avg_latency_ms": avg_lat,
                "tput_avg_gbps": avg_algbw,
                "busbw_avg_gbps": avg_busbw,
                "msg_size_bytes": msg_size,
                "msg_size_str": convert_size(msg_size)
            }

        return summary

    # Print summary at end of iteration, epoch, or training
    def log_all(self, print_log=True, show_straggler=False, return_dict=False):
        """
        Print and/or return communication operation statistics.

        Args:
            print_log (bool, optional): Whether to print the summary to console. Defaults to True.
            show_straggler (bool, optional): Whether to include straggler effect analysis. Defaults to False.
            return_dict (bool, optional): Whether to return statistics as a dictionary. Defaults to False.

        Returns:
            dict or None: If return_dict=True, returns a comprehensive dictionary with the following structure:
            {
                "summary": {
                    "operation_name": {
                        message_size_bytes: {
                            "count": int,                    # Number of operations with this message size
                            "total_latency_ms": float,      # Sum of all latencies for this message size
                            "avg_latency_ms": float,        # Average latency (outliers trimmed)
                            "tput_avg_gbps": float,         # Average algorithmic bandwidth in Gbps
                            "busbw_avg_gbps": float,        # Average bus bandwidth in Gbps
                            "msg_size_bytes": int,          # Message size in bytes
                            "msg_size_str": str             # Human-readable message size (e.g., "678.86 MB")
                        }
                    }
                },
                "straggler_analysis": {                     # Only present if show_straggler=True
                    "operation_name": {
                        message_size_bytes: {
                            "count": int,                    # Number of operations
                            "total_comm_lat_ms": float,     # Total communication latency (min across ranks)
                            "total_straggler_ms": float,    # Total straggler effect
                            "avg_comm_lat_ms": float,       # Average communication latency
                            "avg_straggler_ms": float,      # Average straggler effect
                            "msg_size_bytes": int,          # Message size in bytes
                            "msg_size_str": str             # Human-readable message size
                        }
                    }
                } if show_straggler else None,
                "metadata": {
                    "world_size": int,                      # Number of processes in distributed setup
                    "rank": int,                            # Current process rank
                    "timestamp": str                        # ISO format timestamp when log_all was called
                }
            }

            Returns None if return_dict=False.

        Note:
            - Statistics use trimmed mean (10% trimmed from both ends) to remove outliers
            - Straggler analysis requires distributed communication and may impact performance
            - All bandwidth values are in Gbps (Gigabits per second)
            - Latency values are in milliseconds
        """
        import torch
        from deepspeed.utils.timer import trim_mean
        import deepspeed.comm as dist
        from deepspeed.comm.reduce_op import ReduceOp
        from deepspeed.accelerator import get_accelerator
        from datetime import datetime

        # Create a snapshot of the dictionary to avoid concurrent modification issues
        # This prevents "dictionary changed size during iteration" errors when
        # communication operations are happening in other threads
        comms_dict_snapshot = self.comms_dict.copy()

        # Initialize return dictionary structure
        result_dict = {
            "summary": {},
            "straggler_analysis": None,
            "metadata": {
                "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                "rank": dist.get_rank() if dist.is_initialized() else 0,
                "timestamp": datetime.now().isoformat()
            }
        } if return_dict else None

        if print_log:
            print(
                f"{'Comm. Op': <20}{'Message Size': <20}{'Count': <20}{'Total Latency(ms)': <20}{'Avg Latency(ms)': <20}{'tput_avg (Gbps)': <20}{'busbw_avg (Gbps)': <20}"
            )

        for record_name in comms_dict_snapshot.keys():
            if print_log:
                print(record_name)

            # Initialize operation entry in result dict
            if return_dict:
                result_dict["summary"][record_name] = {}

            for msg_size, vals in sorted(comms_dict_snapshot[record_name].items()):
                # vals[0] is the count for each msg size
                count = vals[0]
                # vals[1] is a list of latency records for each msg size
                total_lat = sum(vals[1])
                # vals[2] and vals[3] are the lists of algbw and busbw, respectively
                # Get rid of outliers when we print
                avg_lat = trim_mean(vals[1], 0.1)
                avg_algbw = trim_mean(vals[2], 0.1)
                avg_busbw = trim_mean(vals[3], 0.1)

                # Store data in result dictionary
                if return_dict:
                    result_dict["summary"][record_name][msg_size] = {
                        "count": count,
                        "total_latency_ms": total_lat,
                        "avg_latency_ms": avg_lat,
                        "tput_avg_gbps": avg_algbw,
                        "busbw_avg_gbps": avg_busbw,
                        "msg_size_bytes": msg_size,
                        "msg_size_str": convert_size(msg_size)
                    }

                if print_log:
                    print(
                        f"{' ': <20}{convert_size(msg_size): <20}{count: <20}{total_lat: <20.2f}{avg_lat: <20.2f}{avg_algbw: <20.2f}{avg_busbw: <20.2f}"
                    )

        if show_straggler:
            if return_dict:
                result_dict["straggler_analysis"] = {}

            if print_log:
                print("_______________________________")
                print("Breakdown with straggler effect")
                print("-------------------------------")
                print(
                    f"{'Comm. Op': <20}{'Message Size': <20}{'Count': <20}{'Total comm lat(ms)': <20}{'Total straggler(ms)': <20}{'Avg comm lat(ms)': <20}{'Avg straggler(ms)': <20}"
                )

            device = get_accelerator().current_device_name()
            for record_name in comms_dict_snapshot.keys():
                if print_log:
                    print(record_name)

                # Initialize operation entry in straggler dict
                if return_dict:
                    result_dict["straggler_analysis"][record_name] = {}

                for msg_size, vals in sorted(comms_dict_snapshot[record_name].items()):
                    # vals[0] is the count for each msg size
                    count = vals[0]
                    # vals[1] is a list of latency records for each msg size
                    lats = torch.tensor(vals[1], device=device)
                    min_lats = torch.tensor(vals[1], device=device)
                    dist.all_reduce(min_lats, op=ReduceOp.MIN)
                    total_lat = min_lats.sum().item()
                    total_straggler = (lats - min_lats).sum().item()
                    avg_lat = trim_mean(min_lats.tolist(), 0.1)
                    avg_straggler = trim_mean((lats - min_lats).tolist(), 0.1)

                    # Store straggler data in result dictionary
                    if return_dict:
                        result_dict["straggler_analysis"][record_name][msg_size] = {
                            "count": count,
                            "total_comm_lat_ms": total_lat,
                            "total_straggler_ms": total_straggler,
                            "avg_comm_lat_ms": avg_lat,
                            "avg_straggler_ms": avg_straggler,
                            "msg_size_bytes": msg_size,
                            "msg_size_str": convert_size(msg_size)
                        }

                    if print_log:
                        print(
                            f"{' ': <20}{convert_size(msg_size): <20}{count: <20}{total_lat: <20.2f}{total_straggler: <20.2f}{avg_lat: <20.2f}{avg_straggler: <20.2f}"
                        )

        # Return the dictionary if requested
        return result_dict if return_dict else None
