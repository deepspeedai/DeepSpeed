# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from unit.common import DistributedTest
from unit.simple_model import *
from unit.checkpoint.common import checkpoint_correctness_verification
from unit.util import skip_on_arch

import pytest


class TestPipelineCheckpoint(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize('compile_mode', [True, False])
    @pytest.mark.parametrize("zero_stage", [0, 1])
    def test_checkpoint_pipe_engine(self, zero_stage, tmpdir, compile_mode):
        skip_on_arch(min_arch=7)

        config_dict = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "fp16": {
                "enabled": zero_stage > 0
            },
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

        models = [LinearStackPipe(num_stages=2) for _ in range(2)]
        checkpoint_correctness_verification(config_dict=config_dict,
                                            models=models,
                                            hidden_dim=models[0].hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=True,
                                            train_batch=True,
                                            dtype=torch.float16 if zero_stage > 0 else torch.float32,
                                            compile_mode=compile_mode)

    @pytest.mark.parametrize(
        "base_topo,test_topo",
        [
            #(PipeTopo(num_pp=1,
            #          num_dp=4),
            # PipeTopo(num_pp=4,
            #          num_dp=1)),
            #(PipeTopo(num_pp=2,
            #          num_dp=2),
            # PipeTopo(num_pp=2,
            #          num_dp=2)),
            #(PipeTopo(num_pp=4,
            #          num_dp=1),
            # PipeTopo(num_pp=2,
            #          num_dp=2)),
        ])
    def test_checkpoint_pipe_module(self, base_topo, test_topo, tmpdir):
        checkpoint_engine = TorchCheckpointEngine()
        base_model = LinearStackPipe(topology=base_topo)
        base_model.save_state_dict(tmpdir, checkpoint_engine=checkpoint_engine)

        dist.barrier()

        test_model = LinearStackPipe(topology=test_topo)
        test_model.load_state_dir(tmpdir, checkpoint_engine=checkpoint_engine)

        # Base and test can have different lengths, so make sure we map from the
        # smaller to larger model
        if len(base_model.forward_funcs) < len(test_model.forward_funcs):
            A = base_model
            B = test_model
        else:
            A = test_model
            B = base_model

        # Compare layers individually since partitions are different
        for idx, A_layer in enumerate(A.forward_funcs):
            if not hasattr(A_layer, 'parameters'):
                # Skip functionals, etc.
                continue

            # Find the corresponding layer in B
            global_idx = idx + A._local_start
            B_local_idx = global_idx - B._local_start
            B_layer = B.forward_funcs[B_local_idx]

            # Compare layer parameters
            for p0, p1 in zip(A_layer.parameters(), B_layer.parameters()):
                assert torch.allclose(p0, p1, atol=1e-07), f"Model state {p0} is not equal to {p1}"
