# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from unit.common import DistributedTest

from transformers import VisionEncoderDecoderModel
from transformers.integrations.deepspeed import HfDeepSpeedConfig

import deepspeed


class TestNestingInit(DistributedTest):
    world_size = 1

    def test_nesting_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = torch.nn.Linear(4, 4)

        # ensure that zero3 processed the parameter
        assert hasattr(model.weight, "ds_id")

        deepspeed_engine, *_ = deepspeed.initialize(model=model, config_params=ds_config)


class TestShutdownInNestingInit(DistributedTest):
    world_size = 1

    def test_shutdown_in_nesting_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model1 = torch.nn.Linear(4, 4)

            assert hasattr(model1.weight, "ds_id")
            deepspeed_engine1, *_ = deepspeed.initialize(model=model1, config_params=ds_config)
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model2 = torch.nn.Linear(4, 4)

        # ensure that zero3 processed the parameter
        assert hasattr(model2.weight, "ds_id")
        deepspeed_engine2, *_ = deepspeed.initialize(model=model2, config_params=ds_config)


class TestNestedParallelInit(DistributedTest):
    world_size = 1

    # Testing a model with composed and nested zero.Inits, with 3 zero.Init contexts, 1 parent and 2 children.
    # The skeleton of the model is like so
    #
    # class VisionEncoderDecoderModel(...)::
    #     def __init__(self):
    #             encoder = AutoModel.from_config(config.encoder)
    #             decoder = AutoModelForCausalLM.from_config(config.decoder)
    #
    # And the user calls like below:
    # VisionEncoderDecoderModel.from_pretrained(...)
    # which calls this constructor inside zero.Init

    def test_nested_parallel_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))
        dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
        model = VisionEncoderDecoderModel.from_pretrained(
            "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2")
        assert all([hasattr(p, 'ds_id') for p in model.parameters()])
