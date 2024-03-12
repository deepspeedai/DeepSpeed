# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import numpy as np
import torch
import pytest
import random
import copy
import os
from torch import nn
from deepspeed import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from deepspeed.accelerator import get_accelerator
from unit.modeling import BertConfig, BertLayerNorm, BertEncoder as BertEncoderPostln
from unit.modelingpreln import BertEncoder as BertEncoderPreln
from unit.common import DistributedTest, is_rocm_pytorch

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)


def check_equal(first, second, atol=1e-2, verbose=False):
    diction_x = {}
    diction_y = {}

    if verbose:
        for i, (x, y) in enumerate(zip(first, second)):
            print(x[1], y[1])

    for i, (x, y) in enumerate(zip(first, second)):
        k = 0
        while (diction_x.get((k, x[1])) is not None):
            k = k + 1
        diction_x[k, x[1]] = x[0]
        k = 0
        while (diction_y.get((k, y[1])) is not None):
            k = k + 1
        diction_y[k, y[1]] = y[0]
    if verbose:
        print()
        for i, (x, y) in enumerate(zip(diction_x, diction_y)):
            print(x, y)

    for i, (x, y) in enumerate(zip(diction_x, diction_y)):
        if (x[0] == 1): continue
        if verbose:
            print("checking ", x[1], ":")
        y = diction_y[x[0], x[1]]
        x = diction_x[x[0], x[1]]

        if verbose:
            print(((x == float('inf')).nonzero(as_tuple=True)[0]))
            print(((y == float('inf')).nonzero(as_tuple=True)[0]))
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

        avgx = np.sum(abs(x), dtype=float)
        countx = x.shape[0]
        for i in range(len(x.shape) - 1):
            countx *= x.shape[i + 1]
            avgx = np.sum(avgx)
        tolerance = 1
        if avgx != float('inf') and avgx != -float('inf'):
            avgx = avgx / countx
            tolerance = avgx * atol
        if verbose:
            print("tolerance is ", tolerance)
            x = x.flatten()
            y = y.flatten()
            print("x = {}".format(x))
            print("y = {}".format(y))
            if any(x == float('inf')) or any(x == -float('inf')):
                print("found infinity in x")
            if any(y == float('inf')) or any(y == -float('inf')):
                print("found infinity in y")
            print(np.linalg.norm(x.astype('float64')))
            print(np.linalg.norm(y.astype('float64')))
            print('-' * 80)
        #toler = np.linalg.norm(x.astype('float64')) * 0.0005
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i), atol=tolerance)


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
            raise NotImplementedError("`checkpoint` is not defined below")
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
                hidden_states = layer_module(hidden_states, attention_mask, grads=self.grads)
                hidden_states.register_hook(lambda x, self=self: self.grads.append([x, "hidden_state"]))

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            if (self.pre_or_post):
                hidden_states = self.FinalLayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

    def get_grads(self):
        return self.grads


def create_models(ds_config):
    bert_config = BertConfig(vocab_size_or_config_json_file=119547,
                             hidden_size=ds_config.hidden_size,
                             num_hidden_layers=ds_config.num_hidden_layers,
                             num_attention_heads=ds_config.heads,
                             intermediate_size=ds_config.intermediate_size,
                             hidden_act="gelu",
                             hidden_dropout_prob=ds_config.hidden_dropout_ratio,
                             attention_probs_dropout_prob=ds_config.attn_dropout_ratio,
                             max_position_embeddings=512,
                             type_vocab_size=2,
                             initializer_range=ds_config.initializer_range)

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


def run_backward(ds_config, seq_len, atol=1e-2, verbose=False):
    set_seed(123)
    bert_encoder, ds_encoder = create_models(ds_config)

    # prepare test data
    kwargs = kwargs_fp16 if ds_config.fp16 else kwargs_fp32
    hidden_states = torch.randn(ds_config.batch_size, seq_len, ds_config.hidden_size, **kwargs)
    input_mask = torch.randn(ds_config.batch_size, 1, 1, seq_len, **kwargs)
    Y = torch.randn(ds_config.batch_size, seq_len, ds_config.hidden_size, **kwargs)

    # run baseline
    base_results = bert_encoder(hidden_states,
                                input_mask,
                                output_all_encoded_layers=False,
                                checkpoint_activations=False)

    loss = (Y - base_results[0]).pow(2).sum() / 64
    loss.backward()
    base_grads = bert_encoder.get_grads()

    # run ds
    ds_results = ds_encoder(hidden_states, input_mask, output_all_encoded_layers=False, checkpoint_activations=False)

    loss = (Y - ds_results[0]).pow(2).sum() / 64
    loss.backward()
    ds_grads = ds_encoder.get_grads()

    # check grads
    check_equal(base_grads, ds_grads, atol=atol, verbose=verbose)


# NOTE: Keep these different params as they have helped find divergence in behavior between AMD and NVIDIA.
@pytest.mark.parametrize('batch_size, hidden_size, seq_len, heads, num_layers, is_preln, use_fp16, atol',
                         [
                             (64,160,128,2,24,False,True, 0.2),
                             (64,1600,128,2,4,False,True, 0.2),
                             (8,1600,128,25,3,True,True, 0.05),
                             (8,160,128,2,3,True,True, 0.1),
                             (8,1600,128,2,3,True,True, 0.05),
                         ]) # yapf: disable
class TestCUDABackward(DistributedTest):
    world_size = 1
    if is_rocm_pytorch():
        #This is to flush denorms in forward pass. Please refer to https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.rst#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
        os.environ['ROCBLAS_INTERNAL_FP16_ALT_IMPL'] = '1'

    def test_backward(self, is_preln, use_fp16, batch_size, hidden_size, seq_len, heads, num_layers, atol):
        # Only run fp16 test cases on devices with FP16 capability.
        if not get_accelerator().is_fp16_supported() and (use_fp16 is True or is_preln is False):
            return

        ds_config = DeepSpeedTransformerConfig()
        ds_config.layer_id = None
        ds_config.batch_size = batch_size
        ds_config.hidden_size = hidden_size
        ds_config.intermediate_size = hidden_size
        ds_config.heads = heads
        ds_config.attn_dropout_ratio = 0.0
        ds_config.hidden_dropout_ratio = 0.0
        ds_config.num_hidden_layers = num_layers
        ds_config.pre_layer_norm = is_preln
        ds_config.initializer_range = 0.02
        ds_config.fp16 = use_fp16

        run_backward(ds_config, seq_len, atol=atol, verbose=True)
