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

from deepspeed.comm.comm import (MPI_RANK_ENV_VARS, MPI_WORLD_SIZE_ENV_VARS, launched_by_mpi, mpi_world_size_from_env,
                                 single_process_discovery)

LAUNCHER_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture
def clean_env(monkeypatch):
    for name in LAUNCHER_ENV + MPI_WORLD_SIZE_ENV_VARS + MPI_RANK_ENV_VARS + ("PMIX_NAMESPACE", ):
        monkeypatch.delenv(name, raising=False)


def test_a_bare_environment_reports_no_launcher_and_no_size(clean_env):
    assert mpi_world_size_from_env() is None
    assert launched_by_mpi() is False


@pytest.mark.parametrize("var", MPI_WORLD_SIZE_ENV_VARS)
def test_each_launcher_size_variable_is_read(clean_env, monkeypatch, var):
    """OpenMPI, MPICH/Intel MPI, MVAPICH and srun each export a different one."""
    monkeypatch.setenv(var, "4")

    assert mpi_world_size_from_env() == 4


@pytest.mark.parametrize("var", MPI_WORLD_SIZE_ENV_VARS)
def test_a_launcher_reporting_one_task_reports_one(clean_env, monkeypatch, var):
    """`srun -n1` is a launcher and a single process at once; it wants the fallback, not an error."""
    monkeypatch.setenv(var, "1")

    assert mpi_world_size_from_env() == 1


@pytest.mark.parametrize("var", MPI_RANK_ENV_VARS)
def test_a_rank_variable_marks_a_launcher_but_gives_no_size(clean_env, monkeypatch, var):
    """A rank says a launcher is present, not how big the world is."""
    monkeypatch.setenv(var, "0")

    assert launched_by_mpi() is True
    assert mpi_world_size_from_env() is None


def test_an_unparseable_size_falls_through_to_the_next_variable(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "")

    assert mpi_world_size_from_env() is None


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


def test_a_launcher_that_reports_no_size_is_refused(clean_env, monkeypatch):
    """PMIx launched directly sets PMIX_RANK and no size at all.

    `prterun -n4` and `prterun -n1` are indistinguishable from the environment, so falling back
    would turn the four-rank case into four separate world-size-1 runs. Refusing costs the
    one-rank case an error naming mpi4py, which is the recoverable half of that trade.
    """
    import deepspeed.comm.comm as comm

    def no_mpi4py(*args, **kwargs):
        raise ImportError("No module named 'mpi4py'")

    monkeypatch.setattr(comm, "mpi_discovery", no_mpi4py)
    monkeypatch.setenv("PMIX_RANK", "0")
    monkeypatch.setenv("PMIX_NAMESPACE", "prterun-host-1234@1")

    with pytest.raises(ImportError, match="does not report a world size"):
        comm.init_distributed(dist_backend="gloo", auto_mpi_discovery=True, dist_init_required=True)

    assert "WORLD_SIZE" not in os.environ


def test_a_bare_environment_initializes_end_to_end(clean_env, monkeypatch):
    """`python train.py` with nothing set, all the way through init_distributed.

    The helper tests above pass on a version of this that shadows `init_distributed`'s own
    `world_size` parameter and hands `None` to the backend, because they never reach the
    backend. This one does.
    """
    import deepspeed.comm.comm as comm

    def no_mpi4py(*args, **kwargs):
        raise ImportError("No module named 'mpi4py'")

    monkeypatch.setattr(comm, "mpi_discovery", no_mpi4py)

    comm.init_distributed(dist_backend="gloo", auto_mpi_discovery=True, dist_init_required=True)

    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["RANK"] == "0"
