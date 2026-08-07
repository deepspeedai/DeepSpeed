# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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
