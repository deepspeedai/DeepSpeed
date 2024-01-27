# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import sys
import shutil
import subprocess
import warnings
from shlex import split
from abc import ABC, abstractmethod
from deepspeed.accelerator import get_accelerator
from ..utils import logger, get_numactl_cmd
from .constants import PDSH_MAX_FAN_OUT, MVAPICH_TMP_HOSTFILE


class MultiNodeRunner(ABC):

    def __init__(self, args, world_info_base64):
        self.args = args
        self.validate_args()
        self.user_arguments = self.parse_user_args()
        self.user_script = args.user_script
        self.world_info_base64 = world_info_base64
        self.exports = {}

    @abstractmethod
    def backend_exists(self):
        """Return whether the corresponding backend exists"""

    @abstractmethod
    def get_cmd(self, environment, active_resources):
        """Return the command to execute on node"""

    def add_export(self, key, var):
        self.exports[key.strip()] = var.strip()

    def parse_user_args(self):
        return self.args.user_args

    @property
    def name(self):
        """Return the name of the backend"""
        return self.__class__.__name__

    def validate_args(self):
        """Validate self.args"""


class PDSHRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64):
        super().__init__(args, world_info_base64)

    def backend_exists(self):
        return shutil.which('pdsh')

    def parse_user_args(self):
        processed_args = []
        for arg in self.args.user_args:
            # With pdsh, if we are passing a string as an argument, it will get
            # split on whitespace. To avoid this and support strings that
            # contain '"', we do this extra processing step:
            if " " in arg:
                arg = '"{}"'.format(arg.replace('"', '\\"'))
            processed_args.append(arg)
        return processed_args

    @property
    def name(self):
        return "pdsh"

    def get_cmd(self, environment, active_resources):
        environment['PDSH_RCMD_TYPE'] = 'ssh'
        if self.args.ssh_port is not None:  # only specify ssh port if it is specified
            environment["PDSH_SSH_ARGS_APPEND"] += f" -p {self.args.ssh_port}"

        active_workers = ",".join(active_resources.keys())
        logger.info("Running on the following workers: %s" % active_workers)

        # PDSH flags for max node fan out and specific hosts to launch on
        # See https://linux.die.net/man/1/pdsh for flag details
        pdsh_cmd_args = ['pdsh', '-S', '-f', str(PDSH_MAX_FAN_OUT), '-w', active_workers] + split(
            self.args.launcher_args)

        exports = ""
        for key, val in self.exports.items():
            exports += "export {}={}; ".format(key, val)

        # https://linux.die.net/man/1/pdsh
        # %n will be replaced by pdsh command
        deepspeed_launch = [
            exports, f"cd {os.path.abspath('.')};", sys.executable, "-u", "-m", "deepspeed.launcher.launch",
            f'--world_info={self.world_info_base64}', "--node_rank=%n", f"--master_addr={self.args.master_addr}",
            f"--master_port={self.args.master_port}"
        ]
        if self.args.no_python:
            deepspeed_launch.append("--no_python")
        if self.args.module:
            deepspeed_launch.append("--module")
        if self.args.no_local_rank:
            deepspeed_launch.append("--no_local_rank")
        if self.args.save_pid:
            deepspeed_launch += ["--save_pid", f"{os.getpid()}"]
        if self.args.elastic_training:
            deepspeed_launch.append("--enable_elastic_training")
            deepspeed_launch.append(f"--max_elastic_nodes={self.args.max_elastic_nodes}")
            deepspeed_launch.append(f"--min_elastic_nodes={self.args.min_elastic_nodes}")

        cmd_to_search = [i + "\\" for i in deepspeed_launch[2:6]]

        kill_command = pdsh_cmd_args + ["pkill -f ", " ".join(cmd_to_search)[:-2]]
        return pdsh_cmd_args + deepspeed_launch + [self.user_script] + self.user_arguments, kill_command, environment


class OpenMPIRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool
        self.add_export('UCX_TLS', 'tcp')

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mvapich
        return shutil.which('ompi_info')

    @property
    def name(self):
        return "openmpi"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")
        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        total_process_count = sum(self.resource_pool.values())

        mpirun_cmd = [
            'mpirun',
            '-n',
            f'{total_process_count}',
            '-hostfile',
            f'{self.args.hostfile}',
            '--mca',
            'btl',
            '^openib',
            '--mca',
            'btl_tcp_if_include',
            'eth0',
        ] + split(self.args.launcher_args)

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-x', "{}={}".format(k, v)]

        python_exec = []
        if not self.args.no_python:
            python_exec = [sys.executable, "-u"]
            if self.args.module:
                python_exec.append("-m")

        return mpirun_cmd + export_cmd + python_exec + [self.user_script] + self.user_arguments


class MPICHRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mpich
        return shutil.which('mpirun')  #mpich_info

    @property
    def name(self):
        return "mpich"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")

        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        devices_per_node = self.resource_pool.values()
        total_process_count = sum(devices_per_node)
        process_per_node = list(devices_per_node)[0]
        if not all([n == process_per_node for n in devices_per_node]):
            raise ValueError("MPICH requires same number of devices per node")

        mpirun_cmd = [
            'mpirun',
            '-n',
            f'{total_process_count}',
            '-ppn',
            f'{process_per_node}',
        ] + split(self.args.launcher_args)
        export_cmd = []

        for k, v in self.exports.items():
            export_cmd += ['-genv', "{}={}".format(k, v)]

        export_cmd += ['-genv', 'MASTER_ADDR', str(self.args.master_addr)]
        export_cmd += ['-genv', 'MASTER_PORT', str(self.args.master_port)]
        export_cmd += ['-genv', 'WORLD_SIZE', str(total_process_count)]
        export_cmd += ['-genv', 'LOCAL_SIZE', str(process_per_node)]

        export_cmd += ['-hosts']
        hosts = ""
        for i, host in enumerate(self.resource_pool.keys()):
            if i == 0:
                hosts = f"{host}"
            else:
                hosts += f",{host}"
        export_cmd += [hosts]

        helper_args = ["--launcher"] + [self.args.launcher]
        python_exec = []
        if not self.args.no_python:
            python_exec += [sys.executable, "-u"]
            if self.args.module:
                python_exec.append("-m")
                helper_args.append("--module")
        else:
            helper_args.append("--no_python")

        helper_cmd = str(os.path.dirname(os.path.realpath(__file__))) + '/launcher_helper.py'
        helper_cmd = [helper_cmd] + helper_args + [self.user_script] + self.user_arguments

        return mpirun_cmd + export_cmd + python_exec + helper_cmd


class IMPIRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mpich
        return shutil.which('mpirun')  #mpich_info

    @property
    def name(self):
        return "impi"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")

        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        devices_per_node = self.resource_pool.values()
        total_process_count = sum(devices_per_node)
        process_per_node = list(devices_per_node)[0]
        if not all([n == process_per_node for n in devices_per_node]):
            raise ValueError("Intel MPI requires same number of devices per node")

        mpirun_cmd = [
            'mpirun',
            '-ppn',
            f'{process_per_node}',
        ] + split(self.args.launcher_args)
        export_cmd = []

        for k, v in self.exports.items():
            export_cmd += ['-genv', f'{k}', f'{v}']

        if self.args.bind_cores_to_rank:
            cores_per_rank, _ = get_numactl_cmd(self.args.bind_core_list, process_per_node, 0)
            export_cmd += ['-genv', 'OMP_NUM_THREADS', str(cores_per_rank)]

        export_cmd += ['-genv', 'MASTER_ADDR', str(self.args.master_addr)]
        export_cmd += ['-genv', 'MASTER_PORT', str(self.args.master_port)]
        export_cmd += ['-genv', 'WORLD_SIZE', str(total_process_count)]
        export_cmd += ['-genv', 'LOCAL_SIZE', str(process_per_node)]

        # turn off IMPI core binding, use deepspeed's own core binding
        export_cmd += ['-genv', 'I_MPI_PIN', '0']

        export_cmd += ['-hosts']
        hosts = ""
        for i, host in enumerate(self.resource_pool.keys()):
            if i == 0:
                hosts = f"{host}"
            else:
                hosts += f",{host}"
        export_cmd += [hosts]

        per_host_cmd = []

        for i in range(total_process_count):
            local_rank = i % process_per_node
            python_exec = []
            if self.args.bind_cores_to_rank:
                _, numactl_cmd = get_numactl_cmd(self.args.bind_core_list, process_per_node, local_rank)
                python_exec += numactl_cmd

            if not self.args.no_python:
                python_exec += [sys.executable, "-u"]
                if self.args.module:
                    python_exec.append("-m")
            env_mapping = ['-env', 'RANK', str(i)]
            env_mapping += ['-env', 'LOCAL_RANK', str(local_rank)]
            if i == 0:
                per_host_cmd = ['-n', '1'] + env_mapping + python_exec + [self.user_script] + self.user_arguments
            else:
                per_host_cmd = per_host_cmd + [':', '-n', '1'] + env_mapping + python_exec + [self.user_script
                                                                                              ] + self.user_arguments
        print(mpirun_cmd + export_cmd + per_host_cmd)
        return mpirun_cmd + export_cmd + per_host_cmd


class SlurmRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

    def backend_exists(self):
        return shutil.which('sinfo')

    @property
    def name(self):
        return 'slurm'

    def get_cmd(self, environment, active_resources):
        assert not getattr(self.args, 'detect_nvlink_pairs',
                           False), "slurm backend does not support remapping visible devices"
        total_process_count = sum(self.resource_pool.values())
        srun_cmd = [
            'srun',
            '-n',
            f'{total_process_count}',
        ] + split(self.args.launcher_args)

        if getattr(self.args, 'slurm_comment', ''):
            srun_cmd += ['--comment', self.args.slurm_comment]

        if self.args.include != "":
            srun_cmd.append('--include')
            srun_cmd.append(f'{self.args.include}')
        if self.args.exclude != "":
            srun_cmd.append('--exclude')
            srun_cmd.append(f'{self.args.exclude}')
        if self.args.num_nodes > 0:
            srun_cmd.append('--nodes')
            srun_cmd.append(f'{self.args.num_nodes}')
        if self.args.num_gpus > 0:
            srun_cmd.append('--gpus')
            srun_cmd.append(f'{self.args.num_gpus}')

        exports = '--export=ALL'
        for key, val in self.exports.items():
            exports += f",{key}={val}"

        python_exec = [sys.executable, "-u"]
        command = srun_cmd + [exports] + python_exec + [self.user_script] + self.user_arguments
        return command


class MVAPICHRunner(MultiNodeRunner):

    def __init__(self, args, world_info_base64, resource_pool):
        super().__init__(args, world_info_base64)
        self.resource_pool = resource_pool

        # Disable the CMA kernel module, not available on Ubuntu systems
        self.add_export('MV2_SMP_USE_CMA', '0')

        # If we fail this will output more verbose logging
        self.add_export('MV2_DEBUG_SHOW_BACKTRACE', '1')

        # Enabled cuda-aware communication
        if get_accelerator().device_name() == 'cuda':
            self.add_export('MV2_USE_CUDA', '1')

        # Support deep learning frameworks: http://hidl.cse.ohio-state.edu/userguide/horovod/
        self.add_export('MV2_SUPPORT_DL', '1')

        # Support MPI_THREAD_MULTIPLE
        self.add_export('MV2_ENABLE_AFFINITY', '0')

        # Performance tuning flags for allgather
        self.add_export('MV2_INTER_ALLGATHER_TUNING', '5')
        self.add_export('MV2_CUDA_USE_NAIVE', '0')

    def backend_exists(self):
        #TODO: if IB is available we should suggestion mvapich
        mpiname_exists = shutil.which('mpiname')
        exists = False
        if not mpiname_exists:
            warnings.warn("mpiname does not exist, mvapich is not installed properly")
        else:
            results = subprocess.check_output('mpiname', shell=True)
            mpiname_results = results.decode('utf-8').strip()
            if "MVAPICH2-GDR" in mpiname_results:
                exists = True
            else:
                warnings.warn(f"Expected MVAPICH2-GDR as return for mpiname but received {mpiname_results}")
        return exists

    @property
    def name(self):
        return "mvapich"

    def validate_args(self):
        super().validate_args()
        #TODO: Allow for include/exclude at node-level but not gpu-level
        if self.args.include != "" or self.args.exclude != "":
            raise ValueError(f"{self.name} backend does not support worker include/exclusion")
        if self.args.num_nodes != -1 or self.args.num_gpus != -1:
            raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")

    def get_cmd(self, environment, active_resources):
        devices_per_node = self.resource_pool.values()
        total_process_count = sum(devices_per_node)
        process_per_node = list(devices_per_node)[0]
        if not all([n == process_per_node for n in devices_per_node]):
            raise ValueError("mvapich requires same number of devices per node")

        with open(MVAPICH_TMP_HOSTFILE, 'w') as fd:
            for host in self.resource_pool.keys():
                fd.write(f'{host}\n')

        mpirun_cmd = [
            'mpirun',
            '-np',
            f'{total_process_count}',
            '-ppn',
            f'{process_per_node}',
            '--hostfile',
            f'{MVAPICH_TMP_HOSTFILE}',
        ] + split(self.args.launcher_args)

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-env', "{}={}".format(k, v)]

        python_exec = []
        if not self.args.no_python:
            python_exec = [sys.executable, "-u"]
            if self.args.module:
                python_exec.append("-m")

        return mpirun_cmd + export_cmd + python_exec + [self.user_script] + self.user_arguments
