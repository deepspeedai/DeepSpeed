# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import pytest

from deepspeed.ops.lion import FusedLion
from deepspeed.ops.lion import DeepSpeedCPULion
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.ops.op_builder import CPULionBuilder, FusedLionBuilder

# Each config builds a different optimizer, so each one is gated on the op it needs.
requires_fused_lion = pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLionBuilder.NAME],
                                         reason="FusedLionBuilder has not been implemented on this system.")
requires_cpu_lion = pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[CPULionBuilder.NAME],
                                       reason="CPULionBuilder has not been implemented on this system.")

# yapf: disable
#'optimizer, zero_offload, resulting_optimizer
lion_configs = [pytest.param("Lion", False, FusedLion,        marks=requires_fused_lion),
                pytest.param("Lion", True,  DeepSpeedCPULion, marks=requires_cpu_lion)]

@pytest.mark.parametrize(
    'optimizer, zero_offload, resulting_optimizer',
    lion_configs)
class TestLionConfigs(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    def test(self,
             optimizer,
             zero_offload,
             resulting_optimizer):
        # Precision does not affect which optimizer is built; fp32 keeps the test independent of fp16/bf16 support.
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": optimizer,
                "params": {
                    "lr": 0.00015,
                }
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "cpu_offload": zero_offload
            }
        }
        model = SimpleModel(10)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        # get base optimizer under zero
        ds_optimizer = model.optimizer.optimizer
        opt_class = resulting_optimizer
        assert isinstance(ds_optimizer, opt_class)
