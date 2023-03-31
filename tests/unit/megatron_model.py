# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
import sys
import math

from .common import get_test_path
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.accelerator import get_accelerator


def get_megatron_version():
    p = os.popen("pip list --format=columns | grep megatron-lm")
    pip_list = p.read()
    assert 'megatron-lm' in pip_list, 'Please install Megatron-LM before getting its version'
    ver_str = pip_list.split()[1]
    return float(ver_str[0])


def get_gpt2_model(args_others, mp_size=1):
    from megatron.model import GPT2Model
    from megatron.initialize import initialize_megatron

    args_defaults = {
        'vocab_file': get_test_path('gpt2-vocab.json'),
        'merge_file': get_test_path('gpt2-merges.txt'),
        'tokenizer_type': 'GPT2BPETokenizer',
    }

    args_defaults.update(args_others)

    # setting "make-vocab-size-divisible-by" to avoid word-embedding size change in resizing testing.
    sys.argv.extend(['--model-parallel-size', str(mp_size), '--make-vocab-size-divisible-by', str(1)])

    initialize_megatron(args_defaults=args_defaults, ignore_unknown_args=True)
    model = GPT2Model(num_tokentypes=0, parallel_output=False)
    model.to(get_accelerator().device_name())
    from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
    from megatron import mpu
    i = get_accelerator().current_device_name()
    model = torchDDP(model, device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group())

    return model


class MockGPT2ModelPipe(PipelineModule):

    def __init__(self, num_layers, mp_size, args_others, topo, **kwargs):
        from megatron.initialize import initialize_megatron

        args_defaults = {
            'vocab_file': get_test_path('gpt2-vocab.json'),
            'merge_file': get_test_path('gpt2-merges.txt'),
            'tokenizer_type': 'GPT2BPETokenizer',
        }

        args_defaults.update(args_others)

        # setting "make-vocab-size-divisible-by" to avoid word-embedding size change in resizing testing.
        sys.argv.extend(['--model-parallel-size', str(mp_size), '--make-vocab-size-divisible-by', str(1)])

        initialize_megatron(args_defaults=args_defaults, ignore_unknown_args=True)

        from megatron.model.transformer import ParallelTransformerLayer

        class ParallelTransformerLayerPipe(ParallelTransformerLayer):

            def forward(self, args):
                # hardcode attn mask for testing, PP requires the attn_mask to be stashed
                attention_mask = torch.tensor([[True]], device=get_accelerator().current_device_name())
                return super().forward(args, attention_mask)

        layers = []
        for x in range(num_layers):
            layers.append(
                LayerSpec(ParallelTransformerLayerPipe, self.gpt2_attention_mask_func, self.init_method_normal(0.02),
                          self.scaled_init_method_normal(0.02, num_layers), x))
        super().__init__(layers=layers, loss_fn=torch.nn.CrossEntropyLoss(), topology=topo, **kwargs)

    def gpt2_attention_mask_func(self, attention_scores, ltor_mask):
        attention_scores.masked_fill_(ltor_mask, -10000.0)
        return attention_scores

    def init_method_normal(self, sigma):
        """Init method based on N(0, sigma)."""

        def init_(tensor):
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

        return init_

    def scaled_init_method_normal(self, sigma, num_layers):
        """Init method based on N(0, sigma/sqrt(2*num_layers)."""
        std = sigma / math.sqrt(2.0 * num_layers)

        def init_(tensor):
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)

        return init_
