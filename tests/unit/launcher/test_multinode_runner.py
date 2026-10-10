# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
from deepspeed.launcher import multinode_runner as mnrunner
from deepspeed.launcher import runner as ds_runner
from deepspeed.launcher.runner import (encode_world_info, parse_args, parse_inclusion_exclusion,
                                       apply_num_nodes_and_gpus)
import os
import sys
import json
import subprocess
from pathlib import Path
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


@pytest.fixture
def target_numactl(tmp_path, monkeypatch):
    # Model the numactl CLI, including exec and host-specific topology/permissions.
    fake_numactl = tmp_path / 'numactl'
    fake_numactl.write_text(f'''#!{sys.executable}
import json
import os
import sys
with open(os.environ['NUMACTL_LOG'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:] == ['--hardware']:
    node = int(os.environ.get('NUMACTL_NODE', '0'))
    print('available: 2 nodes (0-1)')
    for i in range(2):
        start = 0 if i == node else 8
        print('node %d cpus: %s' % (i, ' '.join(str(c) for c in range(start, start + 8))))
    sys.exit(0)
args = sys.argv[1:]
while args and args[0] in ('-m', '-p', '-C'):
    option = args[0]
    if option in os.environ.get('NUMACTL_DENIED', '').split():
        error = os.environ.get('NUMACTL_CPU_ERROR') if option == '-C' else None
        print(error or os.environ.get('NUMACTL_ERROR', 'Operation not permitted'), file=sys.stderr)
        sys.exit(1)
    args = args[2:]
os.environ['NUMACTL_BINDING'] = json.dumps(sys.argv[1:len(sys.argv) - len(args)])
os.execvp(args[0], args)
''')
    fake_numactl.chmod(0o755)
    monkeypatch.setenv('PATH', f"{tmp_path}:{os.environ['PATH']}")
    repo_root = Path(mnrunner.__file__).resolve().parents[2]
    monkeypatch.setenv('PYTHONPATH', f"{repo_root}:{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setenv('NUMACTL_LOG', str(tmp_path / 'launcher-numactl.log'))
    monkeypatch.setenv('NUMACTL_DENIED', '-m')
    monkeypatch.setenv('NUMACTL_ERROR', 'invalid NUMA node')
    return tmp_path


@pytest.mark.parametrize('launch_mode', [[], ['--module'], ['--no_python']])
@pytest.mark.parametrize('denied_options', ['', '-m', '-m -C'])
def test_impi_binding_uses_target_host(runner_info, target_numactl, launch_mode, denied_options):
    env, resource_pool, world_info, _ = runner_info
    script = target_numactl / 'rank_payload.py'
    script.write_text(f'''#!{sys.executable}
import json
import os
import sys
with open(sys.argv[1], 'w') as output:
    json.dump(dict(pid=os.getpid(), rank=os.environ['RANK'], local_rank=os.environ['LOCAL_RANK'],
                   threads=os.environ['OMP_NUM_THREADS'], arguments=sys.argv[2:],
                   binding=json.loads(os.environ.get('NUMACTL_BINDING', '[]'))), output)
sys.exit(17)
''')
    script.chmod(0o755)
    user_script = 'rank_payload' if '--module' in launch_mode else str(script)
    output = target_numactl / 'rank.json'
    args = parse_args(['--bind_cores_to_rank', '--bind_core_list', '0-7'] + launch_mode +
                      [user_script, str(output), 'argument with spaces'])
    runner = mnrunner.IMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    # Command construction must neither probe nor depend on the launcher's permissions.
    assert not (target_numactl / 'launcher-numactl.log').exists()
    groups = []
    group = []
    for arg in cmd[cmd.index('-n'):]:
        if arg == ':':
            groups.append(group)
            group = []
        else:
            group.append(arg)
    groups.append(group)

    # Execute the MPI rank commands with different target-host environments.
    for rank, node, denied in [(1, '0', ''), (5, '1', denied_options)]:
        rank_env = os.environ.copy()
        rank_env.update(RANK=str(rank),
                        LOCAL_RANK='1',
                        LOCAL_SIZE='4',
                        NUMACTL_NODE=node,
                        NUMACTL_DENIED=denied,
                        NUMACTL_ERROR='Operation not permitted',
                        NUMACTL_LOG=str(target_numactl / f'target-{rank}.log'))
        proc = subprocess.Popen(groups[rank][8:],
                                env=rank_env,
                                cwd=target_numactl,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        _, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 17, stderr.decode()
        result = json.loads(output.read_text())
        if '-C' in denied:
            binding = []
        elif '-m' in denied:
            binding = ['-C', '2-3']
        else:
            binding = ['-m', node, '-C', '2-3']
        assert result == dict(pid=proc.pid,
                              rank=str(rank),
                              local_rank='1',
                              threads='2',
                              arguments=['argument with spaces'],
                              binding=binding)


@pytest.mark.parametrize('error', ['invalid NUMA node', 'invalid CPU list'])
def test_impi_helper_rejects_invalid_binding(target_numactl, error):
    output = target_numactl / 'must-not-launch'
    helper = Path(mnrunner.numa.__file__)
    env = os.environ.copy()
    env.update(RANK='0', LOCAL_RANK='0', LOCAL_SIZE='2', NUMACTL_DENIED='-m -C')
    if error == 'invalid CPU list':
        env.update(NUMACTL_ERROR='Operation not permitted', NUMACTL_CPU_ERROR=error)
    else:
        env['NUMACTL_ERROR'] = error
    result = subprocess.run([
        sys.executable,
        str(helper), '--num_local_procs', '2', '--local_rank', '0', '--bind_core_list', '0-7', '--', 'touch',
        str(output)
    ],
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=30)
    assert result.returncode != 0
    assert error in result.stderr.decode()
    assert not output.exists()


@pytest.mark.parametrize('launch_mode', [[], ['--module'], ['--no_python']])
def test_impi_without_binding_skips_numa_wrapper(runner_info, launch_mode):
    env, resource_pool, world_info, _ = runner_info
    args = parse_args(launch_mode + ['training_script', '--training-arg'])
    runner = mnrunner.IMPIRunner(args, world_info, resource_pool)
    cmd = runner.get_cmd(env, resource_pool)
    expected = []
    if not args.no_python:
        expected = [sys.executable, '-u']
        if args.module:
            expected.append('-m')
    expected += ['training_script', '--training-arg']
    first_rank_command = cmd.index('-n') + 8
    assert cmd[first_rank_command:first_rank_command + len(expected)] == expected
