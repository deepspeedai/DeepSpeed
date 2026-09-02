# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
from deepspeed.launcher import multinode_runner as mnrunner
from deepspeed.launcher.runner import encode_world_info, parse_args
import os
import pytest


@pytest.fixture
def runner_info():
    hosts = {'worker-0': 4, 'worker-1': 4}
    world_info = encode_world_info(hosts)
    env = deepcopy(os.environ)
    args = parse_args(['test_launcher.py'])
    return env, hosts, world_info, args


@pytest.fixture
def mock_mpi_env(monkeypatch):
    # Provide the 3 required MPI variables:
    monkeypatch.setenv('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '0')
    monkeypatch.setenv('OMPI_COMM_WORLD_SIZE', '1')


def test_pdsh_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.PDSHRunner(args, world_info)
    cmd, kill_cmd, env = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'pdsh'
    assert env['PDSH_RCMD_TYPE'] == 'ssh'


def test_openmpi_runner(runner_info, mock_mpi_env):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'
    assert 'eth0' in cmd


def test_btl_nic_openmpi_runner(runner_info, mock_mpi_env):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(['--launcher_arg', '-mca btl_tcp_if_include eth1', 'test_launcher.py'])
    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert 'eth0' not in cmd
    assert 'eth1' in cmd


def test_btl_nic_two_dashes_openmpi_runner(runner_info, mock_mpi_env):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(['--launcher_arg', '--mca btl_tcp_if_include eth1', 'test_launcher.py'])
    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert 'eth0' not in cmd
    assert 'eth1' in cmd


def test_setup_mpi_environment_success():
    """Test that _setup_mpi_environment correctly sets environment variables when MPI variables exist."""
    os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = '0'
    os.environ['OMPI_COMM_WORLD_RANK'] = '1'
    os.environ['OMPI_COMM_WORLD_SIZE'] = '2'

    args = parse_args(['--launcher_arg', '--mca btl_tcp_if_include eth1', 'test_launcher.py'])

    runner = mnrunner.OpenMPIRunner(args, None, None)
    # Set up the MPI environment
    runner._setup_mpi_environment()

    assert os.environ['LOCAL_RANK'] == '0'
    assert os.environ['RANK'] == '1'
    assert os.environ['WORLD_SIZE'] == '2'

    # Clean up environment
    del os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    del os.environ['OMPI_COMM_WORLD_RANK']
    del os.environ['OMPI_COMM_WORLD_SIZE']
    del os.environ['LOCAL_RANK']
    del os.environ['RANK']
    del os.environ['WORLD_SIZE']


def test_setup_mpi_environment_missing_variables():
    """Test that _setup_mpi_environment raises an EnvironmentError when MPI variables are missing."""

    # Clear relevant environment variables
    os.environ.pop('OMPI_COMM_WORLD_LOCAL_RANK', None)
    os.environ.pop('OMPI_COMM_WORLD_RANK', None)
    os.environ.pop('OMPI_COMM_WORLD_SIZE', None)

    args = parse_args(['--launcher_arg', '--mca btl_tcp_if_include eth1', 'test_launcher.py'])

    with pytest.raises(EnvironmentError, match="MPI environment variables are not set"):
        mnrunner.OpenMPIRunner(args, None, None)


def test_setup_mpi_environment_fail():
    """Test that _setup_mpi_environment fails if only partial MPI variables are provided."""
    os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = '0'
    os.environ.pop('OMPI_COMM_WORLD_RANK', None)  # missing variable
    os.environ['OMPI_COMM_WORLD_SIZE'] = '2'

    args = parse_args(['--launcher_arg', '--mca btl_tcp_if_include eth1', 'test_launcher.py'])

    with pytest.raises(EnvironmentError, match="MPI environment variables are not set"):
        runner = mnrunner.OpenMPIRunner(args, None, None)

    # Clean up environment
    del os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    del os.environ['OMPI_COMM_WORLD_SIZE']


def test_mpich_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.MPICHRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'


def test_slurm_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.SlurmRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'srun'


def test_mvapich_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.MVAPICHRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'
