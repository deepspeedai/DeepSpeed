'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import os
import torch
from datetime import timedelta

from .logging import logger
from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT, default_pg_timeout


def init_distributed(dist_backend="nccl",
                     auto_mpi_discovery=True,
                     distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT,
                     verbose=True,
                     timeout=default_pg_timeout,
                     init_method=None):
    """Initialize torch.distributed backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: Optional (str). torch distributed backend, e.g., nccl, mpi, gloo

        auto_mpi_discovery Optional (bool). if distributed environment variables are not set, attempt to discover them from MPI

        distributed_port: Optional (int). torch distributed backend port

        verbose: Optional (bool). verbose logging

        timeout: Optional (timedelta). Timeout for operations executed against the process group. Default value equals 30 minutes.

        init_method: Optional (string). Torch distributed, URL specifying how to initialize the process group. Default is “env://” if no init_method or store is specified.
    """
    required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):
        if verbose:
            logger.info(
                "Not using the DeepSpeed or torch.distributed launchers, attempting to detect MPI environment..."
            )
        if in_aml() and not in_dlts():
            patch_aml_env_for_torch_nccl_backend(verbose=verbose)
        else:
            mpi_discovery(distributed_port=distributed_port, verbose=verbose)

    if not torch.distributed.is_initialized():
        if verbose:
            logger.info(
                "Initializing torch distributed with backend: {}".format(dist_backend))
        assert isinstance(timeout, timedelta)
        torch.distributed.init_process_group(backend=dist_backend,
                                             timeout=timeout,
                                             init_method=init_method)


def mpi_discovery(distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, verbose=True):
    """
    Discovery MPI environment via mpi4py and map to relevant torch.distributed state
    """
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[0]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)

    if verbose:
        logger.info(
            "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'],
                    os.environ['LOCAL_RANK'],
                    os.environ['WORLD_SIZE'],
                    os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))

    if torch.distributed.is_initialized():
        assert torch.distributed.get_rank() == rank, "MPI rank {} does not match torch rank {}".format(
            rank, torch.distributed.get_rank())
        assert torch.distributed.get_world_size() == world_size, "MPI world size {} does not match torch world size {}".format(
            world_size, torch.distributed.get_world_size())


def in_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def in_dlts():
    # Are we running on a DLTS cluster?
    return 'DLTS_JOB_ID' in os.environ


def patch_aml_env_for_torch_nccl_backend(master_port=6105, verbose=True):
    """Helper routine to get and set environment variables.
    This is adapted from Azure ML's documentation available from:
    https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training/#environment-variables-from-openmpi
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(
        os.environ["WORLD_SIZE"])

    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "54965"

    if verbose:
        logger.info("NCCL_SOCKET_IFNAME original value = {}".format(
            os.environ["NCCL_SOCKET_IFNAME"]))

    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

    if verbose:
        logger.info(
            "Discovered AzureML settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'],
                    os.environ['LOCAL_RANK'],
                    os.environ['WORLD_SIZE'],
                    os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))
