# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from deepspeed.ops.op_builder import FusedLambBuilder

from unit.common import DistributedTest
from unit.simple_model import *

from unit.checkpoint.common import checkpoint_correctness_verification

import pytest


class TestCheckpointWriterResume(DistributedTest):
    world_size = [1, 2]
    non_daemonic_procs = True

    @pytest.mark.parametrize('decoupled', [False, True])
    @pytest.mark.parametrize('serialization', [False, True])
    def test_resume_training(self, tmpdir, monkeypatch, decoupled, serialization):
        # These launcher variables describe the single machine used by DistributedTest.
        monkeypatch.setenv('CROSS_RANK', '0')
        monkeypatch.setenv('CROSS_SIZE', '1')
        config = {
            'train_micro_batch_size_per_gpu': 1,
            'zero_allow_untested_optimizer': True,
            'zero_optimization': {
                'stage': 3,
                'reduce_bucket_size': 1000,
                'stage3_prefetch_bucket_size': 1000,
            },
            'checkpoint': {
                'checkpoint_serialization': serialization,
                'writer': {
                    'type': 'python',
                    'decoupled': decoupled,
                },
            },
        }

        def make_engine():
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            return deepspeed.initialize(model=model, optimizer=optimizer, config=config)[0]

        def train_step(engine):
            loss = engine(torch.ones(1, 4, device=engine.device)).square().mean()
            engine.backward(loss)
            engine.step()

        def snapshot(engine):
            with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                return {name: value.detach().clone() for name, value in engine.module.state_dict().items()}

        source = make_engine()
        target = None
        try:
            train_step(source)
            saved_weights = snapshot(source)
            saved_steps = source.global_steps
            source.save_checkpoint(tmpdir, tag='resume', client_state={'label': 'resume'})
            # The next optimizer step commits pending asynchronous checkpoint writes.
            train_step(source)
            target = make_engine()
            load_path, client_state = target.load_checkpoint(tmpdir, tag='resume')
            assert load_path is not None
            assert client_state['label'] == 'resume'
            assert saved_steps == target.global_steps
            for name, value in snapshot(target).items():
                torch.testing.assert_close(value, saved_weights[name], rtol=0, atol=0)

            # The next update checks restoration of Adam's moments and step counter,
            # not just restoration of model weights.
            train_step(target)
            expected = snapshot(source)
            for name, value in snapshot(target).items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        finally:
            if target is not None:
                target.destroy()
            source.destroy()


class TestOtherOptimizerCheckpoint(DistributedTest):
    world_size = 2

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME], reason="lamb is not compatible")
    def test_checkpoint_unfused_optimizer(self, tmpdir):
        #if not get_accelerator().is_fp16_supported():
        #    pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0,
            "scheduler": {
                "type": "OneCycle",
                "params": {
                    "cycle_first_step_size": 1000,
                    "cycle_first_stair_count": 500,
                    "cycle_second_step_size": 1000,
                    "cycle_second_stair_count": 500,
                    "decay_step_size": 1000,
                    "cycle_min_lr": 0.0001,
                    "cycle_max_lr": 0.0010,
                    "decay_lr_rate": 0.001,
                    "cycle_min_mom": 0.85,
                    "cycle_max_mom": 0.99,
                    "decay_mom_rate": 0.0
                }
            }
        }
        dtype = torch.float
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}
            dtype = torch.float16

        # with bf16 fails with: DeepSpeed lamb optimizer requires dynamic loss scaling
        # if get_accelerator().is_bf16_supported():
        #     config_dict["bf16"] = {"enabled": True}

        args = args_from_dict(tmpdir, config_dict)
        hidden_dim = 10
        models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        # Load & verify optimizer states
        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            dtype=dtype)

        # Ignore optimizer states
        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=False,
                                            dtype=dtype)

    def test_checkpoint_fused_optimizer(self, tmpdir):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
        }
        dtype = torch.float
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}
            dtype = torch.float16

        args = args_from_dict(tmpdir, config_dict)
        hidden_dim = 10
        models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        # Load & verify optimizer states
        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            dtype=dtype)

        # Ignore optimizer states
        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=False,
                                            dtype=dtype)

    def test_checkpoint_fp32_optimizer(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": False
            }
        }

        args = args_from_dict(tmpdir, config_dict)
        hidden_dim = 10
        models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            dtype=torch.float32)
