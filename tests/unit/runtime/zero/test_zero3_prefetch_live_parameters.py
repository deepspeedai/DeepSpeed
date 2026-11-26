# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch

import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from unit.common import DistributedTest


class TestZero3PrefetchLiveParameters(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("max_live_parameters,prefetch_large_weight", [(48, False), (80, True)])
    def test_prefetch_respects_live_parameter_budget(self, max_live_parameters, prefetch_large_weight):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 16, bias=False))
        reference = copy.deepcopy(model)
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": 32,
                "stage3_max_live_parameters": max_live_parameters,
                "stage3_param_persistence_threshold": 0,
                "stage3_max_reuse_distance": 0,
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                               config=config)
        reference.to(engine.device)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        inputs = torch.arange(8, device=engine.device, dtype=torch.float32).reshape(2, 4) / 8

        # The root has no parameters of its own, so its pre-hook can only fetch these weights speculatively.
        first_weight, second_weight = model.parameters()
        observed = []
        hook = model.register_forward_pre_hook(lambda *_: observed.append(
            (first_weight.ds_status, second_weight.ds_status)))
        try:
            for _ in range(4):
                reference_optimizer.zero_grad()
                expected = reference(inputs).square().mean()
                expected.backward()
                reference_optimizer.step()

                loss = engine(inputs).square().mean()
                torch.testing.assert_close(loss, expected)
                engine.backward(loss)
                engine.step()

            # 16 + 64 elements fit the live budget only at 80. At 48 the second weight still
            # has to be gathered on demand, even though that weight alone exceeds the limit.
            first_status, second_status = observed[-1]
            assert first_status != ZeroParamStatus.NOT_AVAILABLE, observed
            assert (second_status != ZeroParamStatus.NOT_AVAILABLE) == prefetch_large_weight, observed
            for param, expected_param in zip(model.parameters(), reference.parameters()):
                with deepspeed.zero.GatheredParameters([param]):
                    torch.testing.assert_close(param, expected_param)
        finally:
            hook.remove()
            engine.destroy()
