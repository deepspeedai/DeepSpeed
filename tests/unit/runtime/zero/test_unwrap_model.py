# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from deepspeed.runtime.zero import unwrap_model_for_generation
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest
from unit.simple_model import SimpleModel

config = {
    "train_batch_size": 2,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}

if get_accelerator().is_fp16_supported():
    config["fp16"] = {"enabled": True, "loss_scale": 138.}
elif get_accelerator().is_bf16_supported():
    config["bf16"] = {"enabled": True}


class TestUnwrapModel(DistributedTest):
    # gather across more than 1 gpu
    world_size = 2

    def test(self):

        def hooks_exist(engine):
            if engine.optimizer is not None and hasattr(engine.optimizer, "parameter_offload"):
                optimizer_offload = engine.optimizer.parameter_offload
            elif engine.optimizer is not None:
                optimizer_offload = engine.optimizer

            hooks = 0
            for hook in optimizer_offload.forward_hooks:
                hooks += 1
            if hooks > 0:
                return True
            return False

        model = SimpleModel(hidden_dim=100)
        engine, _, _, _ = deepspeed.initialize(args=None, model=model, config=config)

        with unwrap_model_for_generation(engine):
            # assert no hooks
            assert not hooks_exist(engine)
            # assert parameters gathered
            assert model.linears[0].weight.numel() != 0, "GatheredParameters should give a non-0-sized tensor"

        # assert hooks
        assert hooks_exist(engine)
