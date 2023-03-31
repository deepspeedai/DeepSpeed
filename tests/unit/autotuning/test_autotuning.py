# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pytest
from unit.simple_model import create_config_from_dict
from deepspeed.launcher import runner as dsrun
from deepspeed.autotuning.autotuner import Autotuner
from deepspeed.autotuning.scheduler import ResourceManager

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
