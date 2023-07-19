# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)
# yapf: disable
#'optimizer, zero_offload, torch_adam, adam_w_mode, resulting_optimizer
adam_configs = [["AdamW", False, False, False, (FusedAdam, True)],
                ["AdamW", False, True,  False, (torch.optim.AdamW, None)],
                ["AdamW", True,  False, False, (DeepSpeedCPUAdam, True)],
                ["AdamW", True,  True,  False, (torch.optim.AdamW, None)],
                ["AdamW", False, False, True,  (FusedAdam, True)],
                ["AdamW", False, True,  True,  (torch.optim.AdamW, None)],
                ["AdamW", True,  False, True,  (DeepSpeedCPUAdam, True)],
                ["AdamW", True,  True,  True,  (torch.optim.AdamW, None)],
                ["Adam",  False, False, False, (FusedAdam, False)],
                ["Adam",  False, True,  False, (torch.optim.Adam, None)],
                ["Adam",  True,  False, False, (DeepSpeedCPUAdam, False)],
                ["Adam",  True,  True,  False, (torch.optim.Adam, None)],
                ["Adam",  False, False, True,  (FusedAdam, True)],
                ["Adam",  False, True,  True,  (torch.optim.AdamW, None)],
                ["Adam",  True,  False, True,  (DeepSpeedCPUAdam, True)],
                ["Adam",  True,  True,  True,  (torch.optim.AdamW, None)]]

@pytest.mark.parametrize(
    'optimizer, zero_offload, torch_adam, adam_w_mode, resulting_optimizer',
    adam_configs)
class TestAdamConfigs(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    def test(self,
             optimizer,
             zero_offload,
             torch_adam,
             adam_w_mode,
             resulting_optimizer):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": optimizer,
                "params": {
                    "lr": 0.00015,
                    "torch_adam": torch_adam,
                    "adam_w_mode": adam_w_mode
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            },
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
        opt_class, adam_w_mode = resulting_optimizer
        assert isinstance(ds_optimizer, opt_class)
        if adam_w_mode in [True, False]:
            assert ds_optimizer.adam_w_mode == adam_w_mode
