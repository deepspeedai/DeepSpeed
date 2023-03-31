# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

from numpy import BUFSIZE
import json
import subprocess
import sys
import threading
import time
import base64

import os
import hjson
from tqdm import tqdm

from ..utils import logger
from .constants import AUTOTUNING, AUTOTUNING_METRIC_PATH
from .utils import get_val_by_key, search_error, was_interruptted
"""
thread-0: loop over experiment queue dispatching experiments if they become available
thread-N: start each experiment in its own thread
"""

from deepspeed import comm as dist

TIMEOUT = 5


class ResourceManager:

    def __init__(self, args, hosts, num_gpus_per_node, results_dir, exps_dir, arg_mappings):
        self.results_dir = results_dir
        self.exps_dir = exps_dir

        self.nodes = []
        self.num_gpus_per_node = num_gpus_per_node
        for host in hosts:
            self.nodes.append(Node(host, num_gpus_per_node))

        self.experiment_queue = []
        self.running_experiments = {}
        self.finished_experiments = {}
        self.experiment_count = 0
        self.exp_paths = set()
        self.args = args

        self.arg_mappings = {}
        if arg_mappings is not None:
            for k, v in arg_mappings.items():
                k = k.strip()
                v = v.strip()
                if k not in self.arg_mappings:
                    self.arg_mappings[k] = v

    def schedule_experiments(self, exp_paths):
        for exp_path in exp_paths:
            if exp_path in self.exp_paths:
                continue
            else:
                self.exp_paths.add(exp_path)
                with open(exp_path, "r") as fd:
                    exp = hjson.load(fd)
                    exp["exp_id"] = self.experiment_count
                    self.experiment_count += 1

                    result_dir = exp["result_dir"] = os.path.join(self.results_dir, exp['name'])
                    if AUTOTUNING in exp["ds_config"]:
                        metric_file = os.path.join(result_dir, "metrics.json")
                        exp["ds_config"][AUTOTUNING][AUTOTUNING_METRIC_PATH] = metric_file
                    stderr_file = os.path.join(result_dir, "stderr.log")
                    model_info_file = os.path.join(result_dir, "model_info.json")
                    metric_file = os.path.join(result_dir, "metrics.json")

                    # skip existing experiments (except for the ones that were interrupted)
                    if os.path.exists(result_dir) and os.path.exists(stderr_file):
                        if not was_interruptted(stderr_file):
                            err = search_error(stderr_file)
                            exp_id = exp["exp_id"]
                            self.finished_experiments[exp_id] = (exp, err)
                            if err or os.path.exists(metric_file) or os.path.exists(model_info_file):
                                logger.info(f"Skipping exp {exp['name']} whose result already exists")
                                continue

                    self.experiment_queue.append(exp)

    def run_job(self, exp: dict, reservations):
        exp_id = exp["exp_id"]
        exp["master_port"] = self.args.master_port + exp_id
        exp["result_dir"] = os.path.join(self.results_dir, exp['name'])
        user_script = self.args.user_script
        user_args = self.args.user_args

        # overwrite the user arg in the arg_mappings
        for key, val in self.arg_mappings.items():
            nval = get_val_by_key(exp, key)
            if nval and str(nval) != "auto":
                if val in user_args:
                    idx = user_args.index(val)
                    user_args[idx + 1] = str(nval)
                else:
                    user_args.append(val)
                    user_args.append(str(nval))

        t = threading.Thread(target=run_experiment, args=(exp, reservations, user_script, user_args))
        t.start()
        self.running_experiments[exp_id] = (t, exp, reservations, time.time())

    def experiment_check(self, pbar):
        finished_exps = []
        for exp_id, exp_data in self.running_experiments.items():
            thread, exp_json, reservations, start_time = exp_data
            logger.debug(f"Checking exp_id = {exp_id}, alive = {thread.is_alive()}")
            thread.join(timeout=TIMEOUT)
            if not thread.is_alive():
                exp_dir = exp_json["result_dir"]
                stderr_file = os.path.join(exp_dir, "stderr.log")
                err = search_error(stderr_file)
                finished_exps.append((exp_id, reservations))
                self.finished_experiments[exp_id] = (exp_json, err)
                duration = time.time() - start_time
                logger.debug(f"Finished exp_id = {exp_id}, duration={duration:.2f} sec")
                pbar.update(len(finished_exps))
        for exp_id, reservations in finished_exps:
            for reservation in reservations:
                reservation.restore_slots()
            self.running_experiments.pop(exp_id)
        time.sleep(TIMEOUT)

    def resource_request(self, exp):
        num_gpus, num_nodes = exp['num_gpus'], exp['num_nodes']
        slot_request = num_gpus
        reservations = []
        for node in self.nodes:
            if num_nodes == 0:
                break
            slots = node.reserve_slots(slot_request=slot_request)
            if slots:
                reservations.append(Reservation(node=node, slots=slots))
                num_nodes -= 1

        if num_nodes == 0:
            # request satisfied
            return reservations
        else:
            # request not satisfied
            for reservation in reservations:
                reservation.restore_slots()

    def status(self):
        status = ""
        for node in self.nodes:
            status += f"{node.host} ({len(node.idle_slots)} idle gpus), "
        return status[:-1]

    def run(self):
        pbar = tqdm(total=len(self.experiment_queue))

        while len(self.experiment_queue) > 0:
            exp = self.experiment_queue.pop(0)
            logger.debug(f'Popped exp_id = {exp["exp_id"]} from the queue')
            logger.debug(f'Resource status: {self.status()}')
            reservations = self.resource_request(exp)

            if not reservations:
                logger.debug(f'Unable to schedule exp_id = {exp["exp_id"]}')
                self.experiment_queue.insert(0, exp)
                logger.debug(f'Put exp_id = {exp["exp_id"]} back into the queue')
                self.experiment_check(pbar)
            else:
                desc = ""
                for reservation in reservations:
                    reservation.slots.sort()
                    slots = ",".join(map(str, reservation.slots))
                    desc += f"{reservation.node.host}:{slots}@"
                desc = desc[:-1]
                logger.debug(f'Running exp_id = {exp["exp_id"]} on {desc}')
                self.run_job(exp, reservations)

        # All pending experiments are scheduled, waiting for them to complete
        while len(self.running_experiments) > 0:
            self.experiment_check(pbar)

    def save_exp_results_to_database(self, message, ranks=None, path=None):
        """Print message when one of following condition meets

        + not dist.is_initialized()
        + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
            message (str)
            ranks (list)
            path (str)

        """
        should_log = not dist.is_initialized()
        ranks = ranks or []
        my_rank = dist.get_rank() if dist.is_initialized() else -1
        if ranks and not should_log:
            should_log = ranks[0] == -1
            should_log = should_log or (my_rank in set(ranks))
        logger.debug(f"*** Should log: {should_log}")
        if should_log:
            message['rank'] = my_rank
            with open(path, 'a') as outfile:
                json.dump(message, outfile)
                outfile.write('\n')

    def parse_results(self, metric):
        """ Parses the metric file of the finished experiments to select the optimal DeepSpeed configuration.

        Args:
            finished_experiments (dcit): a dictionary of experiment id and experiment description.

        Returns:
            The path to the result folder of the experiment with the optimal configuration.
        """
        max_throughput = sys.float_info.min
        best_exp_id = -1
        for exp_id, (exp, err) in self.finished_experiments.items():
            if err:
                logger.info(
                    f"The experiment exp_id = {exp_id}, exp_name = {exp['name']}, did not run successfully with error = {err}, thus a metrics.txt does not exist for it. Check the stderr.log in {exp['result_dir']}"
                )
                continue

            metric_file = exp["ds_config"][AUTOTUNING][AUTOTUNING_METRIC_PATH]

            if os.path.exists(metric_file):
                with open(metric_file, 'r') as f:
                    results = hjson.load(f)
                    curr_throughput = results[metric]
                    if curr_throughput > max_throughput:
                        max_throughput = curr_throughput
                        best_exp_id = exp_id
                    exp['results'] = results

        if best_exp_id != -1:
            best_exp, _ = self.finished_experiments[best_exp_id]
            return best_exp, max_throughput

        return exp, None

    def clear(self):
        """Clear experiment queues, does not reset self.experiment_count
        """
        self.experiment_queue = []
        # clean up the running experiments
        for exp_id, exp_data in self.running_experiments.items():
            thread, exp_json, reservations, start_time = exp_data
            clean_up(exp_json, reservations)
        self.running_experiments = {}
        self.finished_experiments = {}
        self.exp_paths = set()


class Node:

    def __init__(self, host, max_slots):
        self.host = host
        self.max_slots = max_slots
        self.idle_slots = list(range(max_slots))

    def reserve_slots(self, slot_request: int) -> list:
        if len(self.idle_slots) >= slot_request:
            return [self.idle_slots.pop(0) for _ in range(slot_request)]

    def restore_slots(self, slots: list):
        self.idle_slots += slots


class Reservation:

    def __init__(self, node, slots):
        self.node = node
        self.slots = slots

    def restore_slots(self):
        self.node.restore_slots(self.slots)

    def desc(self):
        slots = ",".join(map(str, self.slots))
        return f"{self.node.host}:{slots}@"


def get_job_id():
    # Infrastructure-specific job-id
    infra_job_id = None
    if "DLWS_JOB_ID" in os.environ:
        infra_job_id = os.environ["DLWS_JOB_ID"]
    elif "DLTS_JOB_ID" in os.environ:
        infra_job_id = os.environ["DLTS_JOB_ID"]
    else:
        infra_job_id = "unknown-job-id"

    return infra_job_id


def get_user():
    user = None
    if "USER" in os.environ:
        user = os.environ["USER"]
    else:
        user = "unknown-user"
    return user


def run_experiment(exp: dict, reservations, user_script, user_args):
    include_str = ""
    for reservation in reservations:
        reservation.slots.sort()
        slots = ",".join(map(str, reservation.slots))
        include_str += f"{reservation.node.host}:{slots}@"
    include_str = include_str[:-1]
    master_port = exp["master_port"]
    exp["launcher_args"] = [
        "--include",
        f"{include_str}",
        "--master_port",
        str(master_port),
    ]
    logger.debug(f'launcher args={exp["launcher_args"]}')

    exp["user"] = get_user()
    exp["job_id"] = get_job_id()
    exp_dir = exp["result_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    ds_config_path = os.path.join(exp_dir, "ds_config.json")
    exp["ds_config_path"] = ds_config_path

    ds_config = copy.deepcopy(exp["ds_config"])
    ds_config_json = json.dumps(ds_config).encode('utf-8')

    exp["ds_config_base64"] = base64.urlsafe_b64encode(ds_config_json).decode('utf-8')

    with open(exp["ds_config_path"], "w", buffering=BUFSIZE) as fd:
        json.dump(ds_config, fd)
        fd.flush()
        os.fsync(fd)
        path = exp["ds_config_path"]
        logger.info(f"Scheduler wrote ds_config to {path}, {os.path.abspath(path)}")

    with open(os.path.join(exp_dir, "exp.json"), "w", buffering=BUFSIZE) as fd:
        json.dump(exp, fd)
        fd.flush()
        os.fsync(fd)
        path = os.path.join(exp_dir, "exp.json")
        logger.info(f"Scheduler wrote exp to {path}, {os.path.abspath(path)}")

    # remove "--deepspeed_config ds_config.json" from user_args
    if user_args:
        if "--deepspeed_config" in user_args:
            idx = user_args.index("--deepspeed_config")
        # "--deepspeed_config" is omitted in HF
        elif "--deepspeed" in user_args:
            idx = user_args.index("--deepspeed")
        assert idx < len(user_args), "there is no ds_config file specified after --deepspeed_config or --deepspeed"
        # user_args[idx + 1] = exp["ds_config_path"]
        # pass base64 serialized ds_config to launcher
        user_args[idx + 1] = exp["ds_config_base64"]

    exp["user_script"] = user_script
    exp["user_args"] = user_args

    cmd = ["deepspeed"] + exp["launcher_args"] + [user_script] + user_args

    assert len(exp["launcher_args"]) > 0, "must provide launcher args"

    with open(os.path.join(exp_dir, "cmd.txt"), "w", buffering=BUFSIZE) as fd:
        fd.write(" ".join(cmd))
        fd.write("\n")
        fd.flush()
        os.fsync(fd)

    logger.info(
        f"Launching exp_id = {exp['exp_id']}, exp_name = {exp['name']}, with resource = {include_str}, and ds_config = {os.path.abspath(ds_config_path)}"
    )

    with open(os.path.join(exp_dir, "stdout.log"), "wb") as out, open(os.path.join(exp_dir, "stderr.log"),
                                                                      "wb") as err:
        result = subprocess.Popen(cmd, stdout=out, stderr=err)
        result.wait()
        out.flush()
        err.flush()
        os.fsync(out)
        os.fsync(err)

    clean_up(exp, reservations)

    logger.info(f"Done running exp_id = {exp['exp_id']}, exp_name = {exp['name']}, with resource = {include_str}")


PDSH_MAX_FAN_OUT = 1024


def clean_up(exp: dict, reservations):
    env = os.environ.copy()
    env['PDSH_RCMD_TYPE'] = 'ssh'

    nodes_str = ""
    for reservation in reservations:
        nodes_str += f"{reservation.node.host},"
    nodes_str = nodes_str[:-1]
    logger.debug(f"Cleaning up exp_id = {exp['exp_id']} on the following workers: {nodes_str}")

    # PDSH flags for max node fan out and specific hosts to launch on
    # See https://linux.die.net/man/1/pdsh for flag details
    pdsh_cmd = ['pdsh', '-f', str(PDSH_MAX_FAN_OUT), '-w', nodes_str]

    kill_cmd = [
        'pkill',
        '-f',
        exp['name'],
    ]
    cmd = pdsh_cmd + kill_cmd
    logger.debug("cmd = {}".format(' '.join(cmd)))

    result = subprocess.Popen(cmd, env=env)
    result.wait()

    # In case of failure must propagate the error-condition back to the caller (usually shell). The
    # actual error and traceback should have been printed in the subprocess, so in order to avoid
    # unnecessary noise we just quietly exit here with the same code as the subprocess
    if result.returncode > 0:
        sys.exit(result.returncode)

    logger.info(f"Done cleaning up exp_id = {exp['exp_id']} on the following workers: {nodes_str}")
