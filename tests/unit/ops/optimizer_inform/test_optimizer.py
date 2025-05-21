# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import pytest

from unit.common import DistributedTest
from unit.simple_model import SimpleModel

# yapf: disable
#optimizer,  torch_adam, adam_w_mode
adam_configs = [["AdamW", False, False],
                ["AdamW", True,  False],
                ["AdamW", False, True ],
                ["AdamW", True,  True ],
                ["Adam",  False, False],
                ["Adam",  True,  False],
                ["Adam",  False, True ],
                ["Adam",  True,  True ]]

@pytest.mark.parametrize(
    'optimizer, torch_adam, adam_w_mode',
    adam_configs)
class TestAdamConfigs(DistributedTest):
    world_size = 1

    def test(self,
             optimizer,
             torch_adam,
             adam_w_mode):
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
            }
        }
        model = SimpleModel(10)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        # get basic optimizer and full basic optimizer
        basic_optimizer_name = model.basic_optimizer
        full_basic_optimizer_name = model.full_basic_optimizer_name

        provider_name = type(basic_optimizer_name).__module__.split('.')[0]
        optimizer_name = basic_optimizer_name.__class__.__name__
        fuse_status = None
        if 'Fuse' or 'fuse' in optimizer_name:
            fuse_status = 'Fused'
        else:
            fuse_status = 'NonFused'
        target_optimizer_name = f'{provider_name}_{optimizer_name}_{fuse_status}'

        assert full_basic_optimizer_name == target_optimizer_name
