# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .optimized_linear import LoRAOptimizedLinear, OptimizedLinear

import torch

try:
    import transformers
except ImportError:
    transformers = None


def init_lora(model):
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRAOptimizedLinear):
            m.init_lora()


class Init(object):
    """
    Init context wrapper similar in style to zero.Init. Allows for injecting OptimizedLinear during model
    construction which will shard base weights and reduce overall memory usage during model init. Primarily
    useful when initializing a model via transformers.AutoModelForCausalLM.

    Example usage:
        lora_config = deepspeed.linear.LoRAConfig(..)
        quant_config = deepspeed.linear.QuantizationConfig(..)
        with deepspeed.linear.Init(lora_config=lora_config, quant_config=quant_config):
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-405B")

    """

    def __init__(self, lora_config=None, quant_config=None):
        self._orig_nn_linear = torch.nn.Linear
        self._orig_causallm_pretrained = None
        if transformers != None:
            self._orig_causallm_pretrained = transformers.AutoModelForCausalLM.from_pretrained
            self._orig_causallm_config = transformers.AutoModelForCausalLM.from_config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self._post_init_complete = False

    def __enter__(self):

        class OptLinearWrapper:
            _orig_nn_linear = self._orig_nn_linear
            _lora_config = self.lora_config
            _quant_config = self.quant_config

            def __new__(self, *args, **kwargs):
                self._lora_config.delay_lora_init = True
                kwargs['lora_config'] = self._lora_config
                kwargs['quantization_config'] = self._quant_config
                kwargs['linear_cls'] = self._orig_nn_linear
                return OptimizedLinear(*args, **kwargs)

        def _model_init(model):
            if self.lora_config != None:
                init_lora(model)
            self._post_init_complete = True
            return model

        # ensures non-lora params are frozen and lora weights are initialized
        def from_pretrained(*args, **kwargs):
            model = self._orig_causallm_pretrained(*args, **kwargs)
            return _model_init(model)

        def from_config(*args, **kwargs):
            model = self._orig_causallm_config(*args, **kwargs)
            return _model_init(model)

        torch.nn.Linear = OptLinearWrapper
        if transformers != None:
            transformers.AutoModelForCausalLM.from_pretrained = from_pretrained
            transformers.AutoModelForCausalLM.from_config = from_config

    def __exit__(self, *args, **kwargs):
        torch.nn.Linear = self._orig_nn_linear
        if not self._post_init_complete:
            print('WARNING: For some reason LoRA modules are not initialized, this is usually done automatically '
                  'if using transformers via (AutoModelForCausalLM from_pretrained/from_config). '
                  'You must call `init_lora` on each module in order to use DeepSpeed LoRA, otherwise '
                  'you will error out during runtime.')
        else:
            transformers.AutoModelForCausalLM.from_pretrained = self._orig_causallm_pretrained
            transformers.AutoModelForCausalLM.from_config = self._orig_causallm_config
