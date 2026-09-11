# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from deepspeed.utils.comms_logging import CommsLogger


def test_stop_profiling_comms_disables_prof_all():
    # start_profiling_comms()/stop_profiling_comms() toggle the global comm
    # profiling flag prof_all. stop_profiling_comms() must clear it; otherwise
    # global comm profiling can never be turned off once it has been started.
    comms_logger = CommsLogger()

    comms_logger.start_profiling_comms()
    assert comms_logger.prof_all is True

    comms_logger.stop_profiling_comms()
    assert comms_logger.prof_all is False


def test_get_operation_summary_does_not_reorder_the_stored_records():
    # comms_dict stores parallel lists per message size: [count, latencies, algbws,
    # busbws], where index i is the i-th recorded op. trim_mean used to sort in
    # place, so summarising sorted each of those lists independently and destroyed
    # the correspondence between them: get_raw_data() then paired the fastest op
    # with the lowest bandwidth. Summarising must not mutate what was recorded.
    comms_logger = CommsLogger()
    latencies = [3.0, 1.0, 2.0]
    # algbw is a function of latency, so latency[i] * algbw[i] is constant.
    algbws = [10.0 / latency for latency in latencies]
    busbws = [algbw * 2 for algbw in algbws]
    comms_logger.comms_dict = {"all_reduce": {1024: [3, list(latencies), list(algbws), list(busbws)]}}

    summary = comms_logger.get_operation_summary("all_reduce")

    stored = comms_logger.get_raw_data()["all_reduce"][1024]
    assert stored[1] == latencies
    assert stored[2] == algbws
    assert stored[3] == busbws
    assert all(latency * algbw == 10.0 for latency, algbw in zip(stored[1], stored[2]))
    # the trimmed mean itself is unaffected
    assert summary[1024]["avg_latency_ms"] == 2.0


def test_trim_mean_does_not_mutate_its_argument():
    from deepspeed.utils.timer import trim_mean

    data = [3.0, 1.0, 2.0]
    assert trim_mean(data, 0.1) == 2.0
    assert data == [3.0, 1.0, 2.0]


def test_timed_op_falls_back_to_the_op_name_when_log_name_is_missing(monkeypatch):
    # timed_op looks up func_args['log_name'], so an op whose signature does not
    # declare log_name used to raise KeyError as soon as profiling was turned on.
    # Such an op must still be logged, under its own name.
    from deepspeed.comm import comm

    monkeypatch.setattr(comm, 'comms_logger', CommsLogger())
    monkeypatch.setattr(
        comm, 'cdb', SimpleNamespace(using_mpi=False, is_initialized=lambda: True,
                                     get_world_size=lambda group=None: 1))
    monkeypatch.setattr(comm, 'get_accelerator', lambda: SimpleNamespace(synchronize=lambda: None))

    @comm.timed_op
    def barrier():
        return 'done'

    comm.comms_logger.enabled = True
    comm.comms_logger.start_profiling_comms()

    assert barrier() == 'done'
    assert 'barrier' in comm.comms_logger.comms_dict


def test_timed_op_profiles_default_log_name_with_prof_ops(monkeypatch):
    from deepspeed.comm import comm

    monkeypatch.setattr(comm, 'comms_logger', CommsLogger())
    monkeypatch.setattr(
        comm, 'cdb', SimpleNamespace(using_mpi=False, is_initialized=lambda: True,
                                     get_world_size=lambda group=None: 1))
    monkeypatch.setattr(comm, 'get_accelerator', lambda: SimpleNamespace(synchronize=lambda: None))

    @comm.timed_op
    def barrier(log_name='barrier'):
        return 'done'

    comm.comms_logger.enabled = True
    comm.comms_logger.prof_ops = ['barrier']

    assert barrier() == 'done'
    assert 'barrier' in comm.comms_logger.comms_dict


@pytest.mark.parametrize('op_name', ['broadcast_object_list', 'all_to_all'])
@pytest.mark.parametrize('positional', [False, True])
@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('profile_mode', ['prof_all', 'prof', 'prof_ops_default', 'prof_ops_custom', 'unselected'])
def test_collective_profiling(monkeypatch, op_name, positional, debug, profile_mode):
    from deepspeed.comm import comm

    backend_op = Mock(return_value='done')
    backend = SimpleNamespace(using_mpi=False, is_initialized=lambda: True, get_world_size=lambda group=None: 2)
    setattr(backend, op_name, backend_op)
    monkeypatch.setattr(comm, 'cdb', backend)
    monkeypatch.setattr(comm, 'get_accelerator', lambda: SimpleNamespace(synchronize=lambda: None))
    op_timer = Mock()
    op_timer.elapsed.return_value = 1.0
    timers = Mock(return_value=op_timer)
    monkeypatch.setattr(comm, 'timers', timers)

    monkeypatch.setattr(comm, 'comms_logger', CommsLogger())
    comm.comms_logger.enabled = True
    comm.comms_logger.debug = debug
    comm.comms_logger.prof_all = profile_mode == 'prof_all'
    log_name = 'custom_collective' if profile_mode in ('prof', 'prof_ops_custom') else op_name
    comm.comms_logger.prof_ops = [log_name] if profile_mode.startswith('prof_ops') else []

    if op_name == 'broadcast_object_list':
        object_list = [{'value': 1}]
        args = (object_list, 0, None, None)
        expected_size = 0
    else:
        input_list = [torch.ones(4), torch.ones(4)]
        output_list = [torch.empty_like(tensor) for tensor in input_list]
        args = (output_list, input_list, None, False)
        expected_size = sum(tensor.numel() * tensor.element_size() for tensor in input_list)

    kwargs = {}
    if profile_mode in ('prof', 'prof_ops_custom'):
        prof = profile_mode == 'prof'
        if positional:
            args += (prof, log_name)
        else:
            kwargs = {'prof': prof, 'log_name': log_name}

    assert getattr(comm, op_name)(*args, **kwargs) == 'done'
    if op_name == 'broadcast_object_list':
        backend_op.assert_called_once_with(object_list=object_list, src=0, group=None, device=None)
    else:
        backend_op.assert_called_once_with(output_list, input_list, group=None, async_op=False)

    if profile_mode == 'unselected':
        assert comm.comms_logger.comms_dict == {}
        timers.assert_not_called()
    else:
        record_name, = comm.comms_logger.comms_dict
        if debug:
            assert record_name.startswith(log_name + ' | [Caller Func: ')
        else:
            assert record_name == log_name
        record = comm.comms_logger.comms_dict[record_name][expected_size]
        assert record[0] == 1
        assert record[1] == [1.0]
        op_timer.start.assert_called_once_with()
        op_timer.stop.assert_called_once_with()
        op_timer.elapsed.assert_called_once_with(reset=False)
        assert all(call.args == (record_name, ) for call in timers.call_args_list)


def test_timed_op_disabled_does_not_access_profiling_state(monkeypatch):
    from deepspeed.comm import comm

    @comm.timed_op
    def barrier(prof=False, log_name='barrier'):
        return 'done'

    # No other logger attributes or backend should be needed on the disabled path.
    monkeypatch.setattr(comm, 'comms_logger', SimpleNamespace(enabled=False))
    monkeypatch.setattr(comm, 'cdb', None)
    timers = Mock()
    monkeypatch.setattr(comm, 'timers', timers)

    assert barrier(True, 'custom_barrier') == 'done'
    timers.assert_not_called()


@pytest.mark.parametrize('world_size', [1, 2, 4])
@pytest.mark.parametrize('comm_op', ['broadcast_object_list', 'all_to_all'])
def test_calc_bw_log_supports_object_and_list_collectives(monkeypatch, comm_op, world_size):
    import deepspeed.comm as dist
    from deepspeed.utils.comms_logging import calc_bw_log

    monkeypatch.setattr(dist, 'get_world_size', lambda group=None: world_size)

    tput, busbw = calc_bw_log(comm_op, 1024, 2.0)
    expected_tput = 1024 / 2.0 * 8 / 1e6
    expected_busbw = expected_tput
    if comm_op == 'all_to_all':
        expected_busbw *= (world_size - 1) / world_size
    assert tput == pytest.approx(expected_tput)
    assert busbw == pytest.approx(expected_busbw)
