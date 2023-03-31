# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import numpy as np
import torch
import pytest
import random
import copy
from torch import nn
from unit.modelingpreln import BertEncoder as BertEncoderPreln
from unit.modeling import BertLayerNorm, BertConfig, BertEncoder as BertEncoderPostln
from deepspeed import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


def check_equal(first, second, atol=1e-2, verbose=False):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x[0].cpu().detach().numpy()
        y = y[0].cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i), atol=atol)


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


device = torch.device(get_accelerator().device_name())
kwargs_fp32 = {'dtype': torch.float, 'device': device, 'requires_grad': True}
kwargs_fp16 = {'dtype': torch.half, 'device': device, 'requires_grad': True}


class DSEncoder(nn.Module):

    def __init__(self, config, weights, biases):
        super(DSEncoder, self).__init__()
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.layer = nn.ModuleList([
            copy.deepcopy(DeepSpeedTransformerLayer(config, weights, biases)) for _ in range(config.num_hidden_layers)
        ])
        self.grads = []
        self.pre_or_post = config.pre_layer_norm

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, checkpoint_activations=False):
        all_encoder_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            raise NotImplementedError("`checkpoint` below is not defined")
            #l = 0
            #num_layers = len(self.layer)
            #chunk_length = math.ceil(math.sqrt(num_layers))
            #while l < num_layers:
            #    hidden_states = checkpoint.checkpoint(
            #        custom(
            #            l,  # noqa: F821
            #            l + chunk_length),
            #        hidden_states,
            #        attention_mask * 1)
            #    l += chunk_length
            # decoder layers
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            if (self.pre_or_post):
                hidden_states = self.FinalLayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def create_models(ds_config):
    bert_config = BertConfig(vocab_size_or_config_json_file=119547,
                             hidden_size=ds_config.hidden_size,
                             num_hidden_layers=ds_config.num_hidden_layers,
                             num_attention_heads=ds_config.heads,
                             batch_size=ds_config.batch_size,
                             intermediate_size=ds_config.intermediate_size,
                             hidden_act="gelu",
                             hidden_dropout_prob=ds_config.hidden_dropout_ratio,
                             attention_probs_dropout_prob=ds_config.attn_dropout_ratio,
                             max_position_embeddings=512,
                             type_vocab_size=2,
                             initializer_range=ds_config.initializer_range,
                             fp16=ds_config.fp16)

    weights = []
    biases = []

    for i in range(4):
        weights.append(nn.Parameter(torch.Tensor(ds_config.hidden_size, ds_config.hidden_size)))
        weights[i].data.normal_(mean=0.0, std=ds_config.initializer_range)

    weights.append(nn.Parameter(torch.Tensor(ds_config.hidden_size)))
    weights[4].data.fill_(1.0)
    weights.append(nn.Parameter(torch.Tensor(ds_config.intermediate_size, ds_config.hidden_size)))
    weights[5].data.normal_(mean=0.0, std=ds_config.initializer_range)
    weights.append(nn.Parameter(torch.Tensor(ds_config.hidden_size, ds_config.intermediate_size)))
    weights[6].data.normal_(mean=0.0, std=ds_config.initializer_range)
    weights.append(nn.Parameter(torch.Tensor(ds_config.hidden_size)))
    weights[7].data.fill_(1.0)

    biases.append(nn.Parameter(torch.Tensor(ds_config.hidden_size)))
    biases[0].data.zero_()
    for i in range(4):
        biases.append(nn.Parameter(torch.Tensor(ds_config.hidden_size)))
        biases[i + 1].data.zero_()
    biases.append(nn.Parameter(torch.Tensor(ds_config.intermediate_size)))
    biases[5].data.zero_()
    biases.append(nn.Parameter(torch.Tensor(ds_config.hidden_size)))
    biases[6].data.zero_()
    biases.append(nn.Parameter(torch.Tensor(ds_config.hidden_size)))
    biases[7].data.zero_()

    if (ds_config.pre_layer_norm):
        bert_encoder = BertEncoderPreln(bert_config, weights, biases)
    else:
        bert_encoder = BertEncoderPostln(bert_config, weights, biases)
    ds_encoder = DSEncoder(ds_config, weights, biases)

    if ds_config.fp16:
        bert_encoder.half()
        ds_encoder.half()

    bert_encoder.to(get_accelerator().device_name())
    ds_encoder.to(get_accelerator().device_name())

    return bert_encoder, ds_encoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_forward(ds_config, seq_len, atol=1e-2, verbose=False, test_bsz=None):
    set_seed(123)
    bert_encoder, ds_encoder = create_models(ds_config)

    bsz = ds_config.batch_size if test_bsz is None else test_bsz

    # prepare test data
    kwargs = kwargs_fp16 if ds_config.fp16 else kwargs_fp32
    hidden_states = torch.randn(bsz, seq_len, ds_config.hidden_size, **kwargs)
    input_mask = torch.randn(bsz, 1, 1, seq_len, **kwargs)

    # run baseline
    base_results = bert_encoder(hidden_states,
                                input_mask,
                                output_all_encoded_layers=False,
                                checkpoint_activations=False)

    # run ds
    ds_results = ds_encoder(hidden_states, input_mask, output_all_encoded_layers=False, checkpoint_activations=False)

    # check forward evaluation
    check_equal(base_results, ds_results, atol=atol, verbose=verbose)


# FP16 test cases can only run on the devices support FP16.
@pytest.mark.sequential
@pytest.mark.parametrize('batch_size, hidden_size, seq_len, heads, num_layers, is_preln, use_fp16',
                         [
                             (64,160,128,2,24,False,True),
                             #(8,2048,2048,32,1,True,True),
                             (8,160,128,2,3,True,True),
                             (8,160,128,2,3,False,True),
                             (8,1600,128,2,3,True,True),
                             (8,1600,128,25,3,True,True),
                             (8,1600,128,25,3,False,True),
                             (8,256,52,4,3,True,True),
                             (3,1024,51,16,3,True,False),
                             (3,1024,54,16,3,True,True),
                             (8,1024,381,16,3,True,False),
                             (8,1024,384,16,3,True,True),
                             (8,1024,384,16,3,True,True),
                             (8,1024,119,16,3,True,False),
                             (8,1024,120,16,3,True,True),
                             (8,1024,509,16,3,True,False),
                             (8,1024,512,16,3,True,True),
                             (64,1024,56,16,3,False,False),
                             (64,1024,53,16,3,False,True),
                             (64,1024,24,16,3,False,False),
                             (64,1024,21,16,3,False,True),
                             (8,1024,384,16,3,False,False),
                             (8,1024,384,16,3,False,True),
                             (8,1024,512,16,3,False,False),
                             (8,1024,511,16,3,False,True),
                             (8,1536,128,24,3,False,False),
                             (8,1536,128,24,3,False,True),
                             (8,2048,128,32,3,False,False),
                             (8,2048,128,32,3,False,True),
                             (8,2560,128,40,3,False,False),
                             (8,2560,128,40,3,False,True),
                             (8,128,128,2,3,True,False),
                             (8,128,128,2,3,True,True),
                             (8,4096,128,64,3,True,True),
                             (8,8192,128,64,3,False,True),
                             (1,256,2048,32,3,True,True),
                         ]) # yapf: disable
class TestCUDAForward(DistributedTest):
    world_size = 1

    def test_forward(self, batch_size, hidden_size, seq_len, heads, num_layers, is_preln, use_fp16):
        # Only run fp16 test cases on devices with FP16 capability.
        if not get_accelerator().is_fp16_supported() and use_fp16 is True:
            return

        ds_config = DeepSpeedTransformerConfig()
        ds_config.layer_id = None
        ds_config.batch_size = batch_size
        ds_config.hidden_size = hidden_size
        ds_config.intermediate_size = 4 * hidden_size
        ds_config.heads = heads
        ds_config.attn_dropout_ratio = 0.0
        ds_config.hidden_dropout_ratio = 0.0
        ds_config.num_hidden_layers = num_layers
        ds_config.pre_layer_norm = is_preln
        ds_config.initializer_range = 0.02
        ds_config.fp16 = use_fp16

        run_forward(ds_config, seq_len, atol=3e-2)


@pytest.mark.parametrize('batch_size, small_bsz, hidden_size, seq_len, heads, num_layers, is_preln, use_fp16',
                         [
                             (8,3,1024,512,16,3,True,False),
                             (8,7,1024,512,16,3,True,True),
                             (8,3,1024,512,16,3,False,False),
                             (8,7,1024,512,16,3,False,True),
                         ]) # yapf: disable
class TestCUDAForwardSmallBatchSize(DistributedTest):
    world_size = 1

    def test_forward_with_small_bsz(self, batch_size, small_bsz, hidden_size, seq_len, heads, num_layers, is_preln,
                                    use_fp16):
        # Only run fp16 test cases on devices with FP16 capability.
        if not get_accelerator().is_fp16_supported() and use_fp16 is True:
            return

        ds_config = DeepSpeedTransformerConfig()
        ds_config.layer_id = None
        ds_config.batch_size = batch_size
        ds_config.hidden_size = hidden_size
        ds_config.intermediate_size = 4 * hidden_size
        ds_config.heads = heads
        ds_config.attn_dropout_ratio = 0.0
        ds_config.hidden_dropout_ratio = 0.0
        ds_config.num_hidden_layers = num_layers
        ds_config.pre_layer_norm = is_preln
        ds_config.initializer_range = 0.02
        ds_config.fp16 = use_fp16

        run_forward(ds_config, seq_len, atol=3e-2, test_bsz=small_bsz)

@pytest.mark.parametrize('batch_size, hidden_size, seq_len, heads, num_layers, is_preln, use_fp16',
                         [
                             #(64,1024,128,16,3,True,False),
                             #(64,1024,128,16,3,True,True),
                             #(64,1024,128,16,3,False,False),
                             #(64,1024,128,16,3,False,True),
                         ]) # yapf: disable
class TestCUDAForwardStochastic(DistributedTest):
    world_size = 1

    def test_forward_stochastic(self, batch_size, hidden_size, seq_len, heads, num_layers, is_preln, use_fp16):
        # Only run fp16 test cases on devices with FP16 capability.
        if not get_accelerator().is_fp16_supported() and use_fp16 is True:
            return

        ds_config = DeepSpeedTransformerConfig()
        ds_config.layer_id = None
        ds_config.batch_size = batch_size
        ds_config.hidden_size = hidden_size
        ds_config.intermediate_size = 4 * hidden_size
        ds_config.heads = heads
        ds_config.attn_dropout_ratio = 0.0
        ds_config.hidden_dropout_ratio = 0.0
        ds_config.num_hidden_layers = num_layers
        ds_config.pre_layer_norm = is_preln
        ds_config.initializer_range = 0.02
        ds_config.fp16 = use_fp16
        ds_config.stochastic_mode = True

        run_forward(ds_config, seq_len, atol=7e-2)
