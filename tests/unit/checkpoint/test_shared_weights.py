# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn as nn

import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from unit.common import DistributedTest


class ModelWithSharedWeights(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(100, 100)
        self.layer1 = nn.Linear(200, 200)
        self.layer2 = nn.Linear(300, 300)
        # tie layer 1 and layer 2
        self.layer1.weight = self.layer2.weight


class TestCheckpointSharedWeights(DistributedTest):
    world_size = 2

    def test_checkpoint_shared_weights(self, tmp_path):
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 2
            },
        }
        model = ModelWithSharedWeights()
        optimizer = torch.optim.Adam(model.parameters())

        deepspeed_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            optimizer=optimizer,
        )
        filename = tmp_path / "checkpoint.pt"
        deepspeed_engine.save_checkpoint(filename, tag="checkpoint")

        model = ModelWithSharedWeights()
        state_dict = get_fp32_state_dict_from_zero_checkpoint(filename, tag="checkpoint")
        model.load_state_dict(state_dict, strict=True)
