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
