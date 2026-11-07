# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
from deepspeed.launcher import multinode_runner as mnrunner
from deepspeed.launcher import runner as ds_runner
from deepspeed.launcher.runner import (encode_world_info, parse_args, parse_inclusion_exclusion,
                                       apply_num_nodes_and_gpus)
import os
import pytest


@pytest.fixture
def runner_info():
    hosts = {'worker-0': 4, 'worker-1': 4}
    world_info = encode_world_info(hosts)
    env = deepcopy(os.environ)
    args = parse_args(['test_launcher.py'])
    return env, hosts, world_info, args


def test_pdsh_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.PDSHRunner(args, world_info)
    cmd, kill_cmd, env = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'pdsh'
    assert env['PDSH_RCMD_TYPE'] == 'ssh'


def test_openmpi_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'
    assert 'eth0' in cmd


def test_btl_nic_openmpi_runner(runner_info):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(['--launcher_arg', '-mca btl_tcp_if_include eth1', 'test_launcher.py'])
    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert 'eth0' not in cmd
    assert 'eth1' in cmd


def test_btl_nic_two_dashes_openmpi_runner(runner_info):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(['--launcher_arg', '--mca btl_tcp_if_include eth1', 'test_launcher.py'])
    runner = mnrunner.OpenMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert 'eth0' not in cmd
    assert 'eth1' in cmd


def test_mpich_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.MPICHRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'


def test_impi_binding_without_local_numactl(runner_info, monkeypatch):
    # MPI executes numactl on the workers, so its absence on the login node must
    # not prevent building rank commands or setting their thread count.
    env, resource_pool, world_info, args = runner_info
    args.bind_cores_to_rank = True
    args.bind_core_list = '0-7'
    monkeypatch.delenv('KMP_AFFINITY', raising=False)
    monkeypatch.setattr(mnrunner.shutil, 'which', lambda name: None)

    def missing_command(*args, **kwargs):
        raise FileNotFoundError('numactl is not installed on the login node')

    monkeypatch.setattr(mnrunner.subprocess, 'check_output', missing_command)
    runner = mnrunner.IMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)

    assert cmd[0] == 'mpirun'
    assert cmd[cmd.index('-hosts') + 1] == 'worker-0,worker-1'
    assert cmd[cmd.index('OMP_NUM_THREADS') + 1] == '2'
    assert cmd.count('numactl') == 8
    assert cmd.index('numactl') > cmd.index('-hosts')
    assert [cmd[i + 1] for i, token in enumerate(cmd) if token == '-C'] == ['0-1', '2-3', '4-5', '6-7'] * 2


def test_slurm_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    active_resources = parse_inclusion_exclusion(resource_pool, args.include, args.exclude)
    runner = mnrunner.SlurmRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, active_resources)
    assert cmd[0] == 'srun'
    assert cmd[cmd.index('-n') + 1] == '8'


@pytest.mark.parametrize('resource_filter, expected_hosts, expected_node_count, expected_process_count',
                         [(['--exclude', 'worker-1'], 'worker-0', '1', '4'),
                          (['--include', 'worker-0:0,1@worker-1:0,1'], 'worker-0,worker-1', '2', '4')])
def test_slurm_runner_resource_filter(runner_info, resource_filter, expected_hosts, expected_node_count,
                                      expected_process_count):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(resource_filter + ['test_launcher.py'])
    active_resources = parse_inclusion_exclusion(resource_pool, args.include, args.exclude)
    runner = mnrunner.SlurmRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, active_resources)
    assert '--include' not in cmd
    assert cmd[cmd.index('--nodelist') + 1] == expected_hosts
    # Without --nodes, srun may satisfy -n from a subset of --nodelist and drop a kept host.
    assert cmd[cmd.index('--nodes') + 1] == expected_node_count
    assert cmd[cmd.index('-n') + 1] == expected_process_count


@pytest.mark.parametrize('resource_filter, expected_error', [(['--include', 'worker-1:0,2'], 'specific device ids'),
                                                             (['--exclude', 'worker-1:0'], 'specific device ids'),
                                                             (['--exclude', 'worker-1:1,2,3'], 'same slot count')])
def test_slurm_runner_rejects_unsupported_filter(runner_info, resource_filter, expected_error):
    # srun cannot pin tasks to device ids or vary the count per host, so these filters have
    # to fail loudly instead of launching a job that ignores them.
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(resource_filter + ['test_launcher.py'])
    active_resources = parse_inclusion_exclusion(resource_pool, args.include, args.exclude)
    runner = mnrunner.SlurmRunner(args, world_info, resource_pool)
    with pytest.raises(ValueError, match=expected_error):
        runner.get_cmd(env, active_resources)


@pytest.mark.parametrize('resource_flag, expected_srun_flag, expected_process_count',
                         [(['--num_gpus', '2'], ('--gpus-per-node', '2'), '4'),
                          (['--num_nodes', '1'], ('--nodes', '1'), '4')])
def test_slurm_runner_num_nodes_and_gpus(runner_info, resource_flag, expected_srun_flag, expected_process_count):
    # main() trims active_resources for these two flags as well, so sizing the job from it
    # moves their task count too. They are mutually exclusive with --include/--exclude, so the
    # resource-filter branch must stay silent and cannot append a second --nodes.
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(resource_flag + ['test_launcher.py'])
    active_resources = parse_inclusion_exclusion(resource_pool, args.include, args.exclude)
    active_resources = apply_num_nodes_and_gpus(active_resources, args.num_nodes, args.num_gpus)
    runner = mnrunner.SlurmRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, active_resources)
    assert '--nodelist' not in cmd
    assert cmd.count(expected_srun_flag[0]) == 1
    assert cmd[cmd.index(expected_srun_flag[0]) + 1] == expected_srun_flag[1]
    assert cmd[cmd.index('-n') + 1] == expected_process_count


@pytest.mark.parametrize('force_multi', [[], ['--force_multi']])
def test_runner_main_rejects_unsupported_filter(tmp_path, monkeypatch, force_multi):
    # --include worker-1:0,2 leaves one host, so without --force_multi main() takes the
    # local-launch path and never builds SlurmRunner. srun still cannot honor that filter, so
    # the launcher the user asked for must not be dropped silently: main() rejects it either way.
    # Popen is blocked so a regression shows up as this assertion rather than as a real launch.
    monkeypatch.setattr(ds_runner.subprocess, 'Popen',
                        lambda *a, **kw: pytest.fail(f'main() launched instead of rejecting: {a[0]}'))
    hostfile = tmp_path / 'hostfile'
    hostfile.write_text('worker-0 slots=4\nworker-1 slots=4\n')
    argv = force_multi + [
        '--hostfile',
        str(hostfile), '--no_ssh_check', '--master_addr', '127.0.0.1', '--launcher', 'slurm', '--include',
        'worker-1:0,2', 'test_launcher.py'
    ]
    with pytest.raises(ValueError, match='specific device ids'):
        ds_runner.main(argv)


def test_validate_active_resources_default_is_a_no_op():
    # Only the slurm backend constrains which filters it can express. The others place tasks
    # per device id themselves, so the hook stays a no-op for them and main() lets them through.
    for cls in (mnrunner.PDSHRunner, mnrunner.OpenMPIRunner, mnrunner.MPICHRunner, mnrunner.IMPIRunner,
                mnrunner.MVAPICHRunner):
        cls.validate_active_resources({'worker-0': [0, 2], 'worker-1': [1]})


def test_mvapich_runner(runner_info):
    env, resource_pool, world_info, args = runner_info
    runner = mnrunner.MVAPICHRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    assert cmd[0] == 'mpirun'
