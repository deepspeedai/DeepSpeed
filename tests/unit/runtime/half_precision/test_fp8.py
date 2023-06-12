# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
from unit.common import DistributedTest
from unit.util import skip_on_arch

try:
    import transformer_engine.pytorch as transformer_engine
    from transformer_engine.common import recipe
except ImportError:
    pytest.skip("Transformer Engine package is missing, skipping tests", allow_module_level=True)


@pytest.mark.parametrize("base_datatype", ["fp16", "bf16", "fp32"])
class TestFp8ComposabilityAcrossZero(DistributedTest):
    world_size = 1

    def test(self, base_datatype):
        skip_on_arch(min_arch=9)

        def run_zero(stage, model_dtype):
            num_batches = 128
            batch_size = 16
            hidden_dim = 768
            # Have to set seed before model
            torch.random.manual_seed(42)
            enable_fp16 = model_dtype == torch.float16
            enable_bf16 = model_dtype == torch.bfloat16
            # TransformerEngine Model
            model = transformer_engine.Linear(hidden_dim, hidden_dim, bias=True, params_dtype=model_dtype)

            # Create FP8 recipe. Note: All input args are optional.
            fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
                                               amax_history_len=16,
                                               amax_compute_algo="max")
            config = {
                "train_batch_size": batch_size,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 0.00001
                    }
                },
                "zero_optimization": {
                    "stage": stage
                },
                "fp16": {
                    "enabled": enable_fp16,
                    "loss_scale": 0.1
                },
                "bf16": {
                    "enabled": enable_bf16
                }
            }
            # Init DeepSpeed
            model, optimizer, _, _ = deepspeed.initialize(args=None,
                                                          model=model,
                                                          model_parameters=model.parameters(),
                                                          config=config)

            batches = torch.randn(num_batches, batch_size, hidden_dim, device=model.device, dtype=model_dtype)
            for batch in batches:
                # Enables autocasting for the forward pass
                with transformer_engine.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    out = model(batch)
                loss = out.mean()
                model.backward(loss)
                model.step()
            return loss

        if base_datatype == "fp16":
            model_dtype = torch.float16
        elif base_datatype == "bf16":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32

        # config
        zero_stage = [0, 1, 2, 3]
        losses = []
        for stage in zero_stage:
            loss = run_zero(stage, model_dtype)
            losses.append(loss)
        all_equal = all(torch.allclose(loss, losses[0], 1e-07, 1e-05) for loss in losses)
        assert (all_equal)
