# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import os
import pytest
from unit.simple_model import create_config_from_dict
from deepspeed.launcher import runner as dsrun
from deepspeed.autotuning.autotuner import Autotuner
from deepspeed.autotuning.constants import GLOBAL_TUNING_SPACE
from deepspeed.autotuning.scheduler import ResourceManager
from deepspeed.autotuning.tuner.base_tuner import BaseTuner
from deepspeed.autotuning.utils import get_val_by_key, metric_value, set_val_by_key, stage_metric_regressed

RUN_OPTION = 'run'
TUNE_OPTION = 'tune'


def test_command_line():
    '''Validate handling of command line arguments'''
    for opt in [RUN_OPTION, TUNE_OPTION]:
        dsrun.parse_args(args=f"--num_nodes 1 --num_gpus 1 --autotuning {opt} foo.py".split())

    for error_opts in [
            "--autotuning --num_nodes 1 --num_gpus 1 foo.py".split(),
            "--autotuning test --num_nodes 1 -- num_gpus 1 foo.py".split(), "--autotuning".split()
    ]:
        with pytest.raises(SystemExit):
            dsrun.parse_args(args=error_opts)


@pytest.mark.parametrize("arg_mappings",
                        [
                            None,
                            {
                            },
                            {
                                "train_micro_batch_size_per_gpu": "--per_device_train_batch_size"
                            },
                            {
                                "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
                                "gradient_accumulation_steps": "--gradient_accumulation_steps"
                            },
                            {
                                "train_batch_size": "-tbs"
                            }
                        ]) # yapf: disable
def test_resource_manager_arg_mappings(arg_mappings):
    rm = ResourceManager(args=None,
                         hosts="worker-0, worker-1",
                         num_gpus_per_node=4,
                         results_dir=None,
                         exps_dir=None,
                         arg_mappings=arg_mappings)

    if arg_mappings is not None:
        for k, v in arg_mappings.items():
            assert k.strip() in rm.arg_mappings.keys()
            assert arg_mappings[k.strip()].strip() == rm.arg_mappings[k.strip()]


@pytest.mark.parametrize("active_resources",
                        [
                           {"worker-0": [0, 1, 2, 3]},
                           {"worker-0": [0, 1, 2, 3], "worker-1": [0, 1, 2, 3]},
                           {"worker-0": [0], "worker-1": [0, 1, 2], "worker-2": [0, 1, 2]},
                           {"worker-0": [0, 1], "worker-2": [4, 5]}
                        ]
                        ) # yapf: disable
def test_autotuner_resources(tmpdir, active_resources):
    config_dict = {"autotuning": {"enabled": True, "exps_dir": os.path.join(tmpdir, 'exps_dir'), "arg_mappings": {}}}
    config_path = create_config_from_dict(tmpdir, config_dict)
    args = dsrun.parse_args(args=f'--autotuning {TUNE_OPTION} foo.py --deepspeed_config {config_path}'.split())
    tuner = Autotuner(args=args, active_resources=active_resources)

    expected_num_nodes = len(list(active_resources.keys()))
    assert expected_num_nodes == tuner.exp_num_nodes

    expected_num_gpus = min([len(v) for v in active_resources.values()])
    assert expected_num_gpus == tuner.exp_num_gpus


def test_get_best_space_record_ignores_runs_without_a_metric(tmpdir):
    # A run that fails to produce a metric (e.g. OOM) is recorded with metric_val=None.
    # get_best_space_record used to crash with "TypeError: '>' not supported between
    # instances of 'NoneType' and 'NoneType'" whenever every run in a space had no
    # metric, discarding all previously gathered results for the whole tuning run.
    config_dict = {"autotuning": {"enabled": True, "exps_dir": os.path.join(tmpdir, 'exps_dir'), "arg_mappings": {}}}
    config_path = create_config_from_dict(tmpdir, config_dict)
    args = dsrun.parse_args(args=f'--autotuning {TUNE_OPTION} foo.py --deepspeed_config {config_path}'.split())
    tuner = Autotuner(args=args, active_resources={"worker-0": [0, 1]})

    tuner.update_records("z0_space", {"name": "exp1"}, None, 1)
    tuner.update_records("z0_space", {"name": "exp2"}, None, 1)
    # tune_space's callers already fall back to a default when this is None,
    # so returning None here (instead of a record carrying a None metric) must
    # stay a safe, handled outcome rather than a new crash.
    assert tuner.get_best_space_record("z0_space") is None

    tuner.update_records("z0_space", {"name": "exp3"}, 3.5, 1)
    best = tuner.get_best_space_record("z0_space")
    assert best[0]["name"] == "exp3"
    assert best[1] == 3.5
    assert best[2] == 3


def test_get_val_by_key_searches_all_nested_subdicts():
    # get_val_by_key must mirror its sibling set_val_by_key: both walk every
    # nested subdict, not just the first one. Here 'device' lives in the SECOND
    # top-level subdict, so the old first-subdict-only search returned None.
    exp = {"optimizer": {"type": "Adam"}, "zero_optimization": {"offload_optimizer": {"device": "cpu"}}}
    assert get_val_by_key(exp, "device") == "cpu"

    # A genuinely absent key still returns None (no false positives).
    assert get_val_by_key(exp, "missing_key") is None

    # A key in the first subdict is unaffected.
    assert get_val_by_key(exp, "type") == "Adam"

    # The getter agrees with the setter, which already reaches this field.
    set_val_by_key(exp, "device", "nvme")
    assert get_val_by_key(exp, "device") == "nvme"


def _make_tuner(tmpdir, metric):
    config_dict = {
        "autotuning": {
            "enabled": True,
            "metric": metric,
            "exps_dir": os.path.join(tmpdir, "exps_dir"),
            "results_dir": os.path.join(tmpdir, "results_dir"),
            "arg_mappings": {}
        }
    }
    config_path = create_config_from_dict(tmpdir, config_dict)
    args = dsrun.parse_args(args=f'--autotuning {TUNE_OPTION} foo.py --deepspeed_config {config_path}'.split())
    return Autotuner(args=args, active_resources={"worker-0": [0, 1]})


def test_stage_metric_regressed_respects_direction_and_unset_sentinel():
    # 0 is the "no previous stage" sentinel. A positive latency must not look
    # regressed against that sentinel, or bucket search never runs for latency.
    assert stage_metric_regressed("latency", 10.0, 0) is False
    assert stage_metric_regressed("throughput", 10.0, 0) is False
    assert stage_metric_regressed("latency", 10.0, 4.0) is True
    assert stage_metric_regressed("latency", 4.0, 10.0) is False
    assert stage_metric_regressed("throughput", 4.0, 10.0) is True
    assert stage_metric_regressed("throughput", 10.0, 4.0) is False
    assert stage_metric_regressed("FLOPS", 1.0, 9.0) is True


def test_metric_value_reads_flops_aliases():
    # The engine writes FLOPS_per_gpu. The README tells users to set "FLOPS",
    # and AUTOTUNING_METRIC_FLOPS is "flops". All three names are one metric.
    results = {"FLOPS_per_gpu": 12.0, "throughput": 3.0, "latency": 9.0}
    assert metric_value(results, "FLOPS") == 12.0
    assert metric_value(results, "flops") == 12.0
    assert metric_value(results, "FLOPS_per_gpu") == 12.0
    assert metric_value(results, "latency") == 9.0
    assert metric_value(results, "throughput") == 3.0
    # An exact key wins when a file contains both names.
    assert metric_value({"flops": 1.0, "FLOPS_per_gpu": 2.0}, "flops") == 1.0
    with pytest.raises(KeyError):
        metric_value(results, "not-a-metric")


def test_get_best_space_record_latency_prefers_the_smaller_value(tmpdir):
    tuner = _make_tuner(tmpdir, "latency")
    tuner.update_records("z0", {"name": "slow"}, 30.0, 1)
    tuner.update_records("z0", {"name": "fast"}, 8.0, 1)
    tuner.update_records("z0", {"name": "oom"}, None, 1)
    best = tuner.get_best_space_record("z0")
    assert best[0]["name"] == "fast"
    assert best[1] == 8.0

    tuner.update_records("z1", {"name": "mid"}, 12.0, 1)
    global_best = tuner.get_best_space_records()[GLOBAL_TUNING_SPACE]
    assert global_best[0]["name"] == "fast"
    assert global_best[1] == 8.0


def test_get_best_space_record_throughput_still_prefers_the_larger_value(tmpdir):
    tuner = _make_tuner(tmpdir, "throughput")
    tuner.update_records("z0", {"name": "slow"}, 30.0, 1)
    tuner.update_records("z0", {"name": "fast"}, 8.0, 1)
    best = tuner.get_best_space_record("z0")
    assert best[0]["name"] == "slow"
    assert best[1] == 30.0

    tuner.update_records("z1", {"name": "faster"}, 40.0, 1)
    global_best = tuner.get_best_space_records()[GLOBAL_TUNING_SPACE]
    assert global_best[0]["name"] == "faster"
    assert global_best[1] == 40.0


def _write_metric(path, payload):
    with open(path, "w") as fd:
        json.dump(payload, fd)


def _finished_exp(name, metric_path):
    return {"name": name, "ds_config": {"autotuning": {"metric_path": str(metric_path)}}}


def test_parse_results_latency_and_flops_alias(tmpdir):
    slow_path = os.path.join(tmpdir, "slow.json")
    fast_path = os.path.join(tmpdir, "fast.json")
    _write_metric(slow_path, {"latency": 30.0, "throughput": 100.0, "FLOPS_per_gpu": 1.0})
    _write_metric(fast_path, {"latency": 10.0, "throughput": 40.0, "FLOPS_per_gpu": 9.0})
    rm = ResourceManager(args=None,
                         hosts=["worker-0"],
                         num_gpus_per_node=1,
                         results_dir=None,
                         exps_dir=None,
                         arg_mappings=None)
    rm.finished_experiments = {
        0: (_finished_exp("slow", slow_path), None),
        1: (_finished_exp("fast", fast_path), None),
    }

    exp, value = rm.parse_results("latency")
    assert exp["name"] == "fast"
    assert value == 10.0

    exp, value = rm.parse_results("throughput")
    assert exp["name"] == "slow"
    assert value == 100.0

    exp, value = rm.parse_results("FLOPS")
    assert exp["name"] == "fast"
    assert value == 9.0

    exp, value = rm.parse_results("flops")
    assert exp["name"] == "fast"
    assert value == 9.0


class _ScriptedTuner(BaseTuner):

    def next_batch(self, sample_size):
        batch = self.all_exps[:sample_size]
        self.all_exps = self.all_exps[sample_size:]
        return batch


class _ScriptedResourceManager:

    def __init__(self, samples, exps_dir):
        self.samples = samples
        self.exps_dir = exps_dir
        self.calls = 0

    def schedule_experiments(self, paths):
        pass

    def run(self):
        pass

    def parse_results(self, metric):
        exp, value = self.samples[self.calls]
        self.calls += 1
        return exp, value

    def clear(self):
        pass


def _sample(name, value):
    return ({"name": name}, value)


def test_base_tuner_latency_keeps_the_smaller_value(monkeypatch, tmpdir):
    monkeypatch.setattr("deepspeed.autotuning.tuner.base_tuner.write_experiments", lambda exps, exps_dir: [])
    samples = [_sample("slow", 20.0), _sample("missing", None), _sample("fast", 5.0)]
    exps = [{"name": "slow"}, {"name": "missing"}, {"name": "fast"}]
    tuner = _ScriptedTuner(exps, _ScriptedResourceManager(samples, str(tmpdir)), "latency")
    tuner.tune(sample_size=1, n_trials=3)
    assert tuner.best_exp["name"] == "fast"
    assert tuner.best_metric_val == 5.0

    throughput_samples = [_sample("high", 20.0), _sample("missing", None), _sample("low", 5.0)]
    throughput_exps = [{"name": "high"}, {"name": "missing"}, {"name": "low"}]
    throughput = _ScriptedTuner(throughput_exps, _ScriptedResourceManager(throughput_samples, str(tmpdir)),
                                "throughput")
    throughput.tune(sample_size=1, n_trials=3)
    assert throughput.best_exp["name"] == "high"
    assert throughput.best_metric_val == 20.0
