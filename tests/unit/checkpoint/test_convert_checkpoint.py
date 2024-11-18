# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn as nn

import deepspeed
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from unit.common import DistributedTest


class ModelWithSharedWeights(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(100, 100)
        self.layer1 = nn.Linear(200, 200)
        self.layer2 = nn.Linear(300, 300)
        # tie layer 1 and layer 2
        self.layer1.weight = self.layer2.weight


class TestCheckpointConvert(DistributedTest):
    world_size = 2

    def test_convert_zero_checkpoint_to_fp32_state_dict(self, tmpdir):
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3
            },
        }
        model = ModelWithSharedWeights()
        optimizer = torch.optim.Adam(model.parameters())

        deepspeed_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            optimizer=optimizer,
        )
        ds_save_dir = tmpdir / "checkpoint_ds"
        deepspeed_engine.save_checkpoint(ds_save_dir, tag="checkpoint")

        model = ModelWithSharedWeights()

        # save checkpoint
        fp32_save_dir = tmpdir / "checkpoint_fp32"
        convert_zero_checkpoint_to_fp32_state_dict(ds_save_dir, fp32_save_dir)

        # load state_dict from fp32 checkpoint
        state_dict = torch.load(fp32_save_dir / 'pytorch_model.bin')

        # check shared tensor
        assert id(state_dict['layer1.weight']) == id(state_dict['layer2.weight'])

        # load state_dict into model
        model.load_state_dict(state_dict, strict=True)
