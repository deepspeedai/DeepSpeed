# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import os

import pytest

from deepspeed.nvme.perf_generate_param import generate_main
from deepspeed.nvme.perf_sweep_utils import READ_LOG_DIR, WRITE_LOG_DIR
from deepspeed.runtime.swap_tensor.aio_config import get_aio_config

# Log names as written by perf_run_sweep.get_log_file: <op>_<submit>_<ordering>_ftd<N>_d<depth>_bs<size>_p<parallel>
FAST_SWEEP_POINT = "single_overlap_ftd1_d32_bs1M_p8"
SLOW_SWEEP_POINT = "block_sequential_ftd1_d128_bs8M_p2"


def _write_sweep_logs(log_dir, speeds):
    for sub_dir, op, metric in ((READ_LOG_DIR, "read", "E2E Read Speed"), (WRITE_LOG_DIR, "write", "E2E Write Speed")):
        os.makedirs(os.path.join(log_dir, sub_dir))
        for sweep_point, speed in speeds.items():
            with open(os.path.join(log_dir, sub_dir, f"{op}_{sweep_point}.txt"), "w") as f:
                f.write(f"{metric} = {speed} GB/sec\n")


# ds_nvme_tune prints its recommendation as a ds_config "aio" section, so it must survive the config parser
# unchanged: every key spelled as get_aio_config reads it, and the flags as JSON booleans.
@pytest.mark.parametrize("winner, expected", [
    (FAST_SWEEP_POINT, {
        "block_size": 1048576,
        "queue_depth": 32,
        "intra_op_parallelism": 8,
        "single_submit": True,
        "overlap_events": True
    }),
    (SLOW_SWEEP_POINT, {
        "block_size": 8388608,
        "queue_depth": 128,
        "intra_op_parallelism": 2,
        "single_submit": False,
        "overlap_events": False
    }),
])
def test_recommended_aio_section_round_trips_through_config(tmpdir, capsys, winner, expected):
    speeds = {FAST_SWEEP_POINT: 1.0, SLOW_SWEEP_POINT: 1.0}
    speeds[winner] = 5.0
    _write_sweep_logs(str(tmpdir), speeds)

    generate_main(str(tmpdir))
    printed = capsys.readouterr().out
    recommended = json.loads(printed[printed.index("{"):])

    aio_config = get_aio_config(recommended)
    assert {key: aio_config[key] for key in expected} == expected
