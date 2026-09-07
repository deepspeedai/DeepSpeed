# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""`deepspeed.initialize` on a machine with no launcher must not require mpi4py.

`init_distributed` fills in the distributed environment when a launcher did not, and its only
route for that was `mpi_discovery`, which imports mpi4py. Running `python train.py` on a single
accelerator - no launcher, no MPI, no mpi4py - therefore ended at `ModuleNotFoundError: No
module named 'mpi4py'` before `deepspeed.initialize` returned.
"""

import os

import pytest

from deepspeed.comm.comm import MPI_WORLD_SIZE_ENV_VARS, in_multi_rank_mpi_job, single_process_discovery

LAUNCHER_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture
def clean_env(monkeypatch):
    for name in LAUNCHER_ENV + MPI_WORLD_SIZE_ENV_VARS + ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        monkeypatch.delenv(name, raising=False)


def test_no_mpi_variables_is_not_a_multi_rank_job(clean_env):
    assert in_multi_rank_mpi_job() is False


@pytest.mark.parametrize("var", MPI_WORLD_SIZE_ENV_VARS)
def test_each_launcher_size_variable_above_one_marks_a_multi_rank_job(clean_env, monkeypatch, var):
    """OpenMPI, MPICH/Intel MPI, PMIx, MVAPICH and srun each export a different one."""
    monkeypatch.setenv(var, "4")

    assert in_multi_rank_mpi_job() is True


@pytest.mark.parametrize("var", MPI_WORLD_SIZE_ENV_VARS)
def test_a_launcher_reporting_one_task_is_not_a_multi_rank_job(clean_env, monkeypatch, var):
    """`srun -n1` is a launcher and a single process at once; it wants the fallback, not an error."""
    monkeypatch.setenv(var, "1")

    assert in_multi_rank_mpi_job() is False


def test_a_rank_variable_alone_does_not_make_it_multi_rank(clean_env, monkeypatch):
    """A rank says a launcher is present, not that the world is bigger than one."""
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    assert in_multi_rank_mpi_job() is False


def test_an_unparseable_size_is_not_taken_as_multi_rank(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "")

    assert in_multi_rank_mpi_job() is False


def test_single_process_discovery_fills_the_environment(clean_env):
    single_process_discovery(distributed_port=29501, verbose=False)

    assert os.environ["RANK"] == "0"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert os.environ["MASTER_PORT"] == "29501"


def test_single_process_discovery_leaves_what_the_caller_set(clean_env, monkeypatch):
    """A partially set environment is the caller's, not something to overwrite."""
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.7")

    single_process_discovery(distributed_port=29501, verbose=False)

    assert os.environ["MASTER_PORT"] == "12345"
    assert os.environ["MASTER_ADDR"] == "10.0.0.7"
    assert os.environ["WORLD_SIZE"] == "1"


def test_a_multi_rank_job_without_mpi4py_says_so(clean_env, monkeypatch):
    """Falling back to a single process there would silently run one rank of a many-rank job."""
    import deepspeed.comm.comm as comm

    def no_mpi4py(*args, **kwargs):
        raise ImportError("No module named 'mpi4py'")

    monkeypatch.setattr(comm, "mpi_discovery", no_mpi4py)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

    with pytest.raises(ImportError, match="mpi4py"):
        comm.init_distributed(dist_backend="gloo", auto_mpi_discovery=True, dist_init_required=True)

    assert os.environ.get("WORLD_SIZE") != "1", "the environment must not be filled in for a multi-rank job"


def test_a_single_task_slurm_step_without_mpi4py_falls_back(clean_env, monkeypatch):
    """`srun -n1 python train.py` with no mpi4py: the case ebarkhordar raised on the PR."""
    import deepspeed.comm.comm as comm

    def no_mpi4py(*args, **kwargs):
        raise ImportError("No module named 'mpi4py'")

    monkeypatch.setattr(comm, "mpi_discovery", no_mpi4py)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "1")

    reached = {}
    real = comm.single_process_discovery

    def wrapped(*args, **kwargs):
        reached["yes"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(comm, "single_process_discovery", wrapped)

    comm.init_distributed(dist_backend="gloo", auto_mpi_discovery=True, dist_init_required=True)

    assert reached.get("yes"), "a one-task step took the mpi4py error instead of the fallback"
    assert os.environ["WORLD_SIZE"] == "1"
