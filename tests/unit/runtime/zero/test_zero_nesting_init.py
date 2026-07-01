# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from unit.common import DistributedTest

from transformers import GPT2Config, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

import deepspeed


def _create_tiny_vision_encoder_decoder_model(model_path):
    encoder_config = ViTConfig(image_size=8,
                               patch_size=4,
                               num_hidden_layers=1,
                               hidden_size=8,
                               num_attention_heads=2,
                               intermediate_size=16)
    decoder_config = GPT2Config(vocab_size=32,
                                n_positions=16,
                                n_embd=8,
                                n_layer=1,
                                n_head=2,
                                bos_token_id=0,
                                eos_token_id=1,
                                add_cross_attention=True,
                                is_decoder=True)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    model = VisionEncoderDecoderModel(config)
    model.save_pretrained(model_path, safe_serialization=False)


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

    def test_nested_parallel_init(self, tmp_path):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))
        _create_tiny_vision_encoder_decoder_model(tmp_path)
        dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
        model = VisionEncoderDecoderModel.from_pretrained(str(tmp_path), local_files_only=True)
        assert all([hasattr(p, 'ds_id') for p in model.parameters()])
