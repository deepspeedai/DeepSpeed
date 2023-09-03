# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


def quantize_transformer_layer(orig_layer_impl, model, megatron=False, preln=False):
    """ Quantize bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.models.bert.modeling_bert.BertLayer or transformers.BertLayer
        model (torch.nn.Module): user's nn.module representing their model

        megatron (bool): megatron model-parallel implementation (this is supported for inference only)
        preln (bool): does the original layer implementation do pre or post layer norm?

        Note: For Bert kind of models, we inject based on the DeepSpeed-Example models, if not setting huggingface flag.

    Returns:
        Updated nn.module with quantized transformer layers
    """

    def quantize_weight(weight):
        return weight.to(torch.int8)

    def megatron_layer_quantize(layer):
        layer.attention.query_key_value.weight.data = quantize_weight(layer.attention.query_key_value.weight.data)
        layer.attention.dense.weight.data = quantize_weight(layer.attention.dense.weight.data)
        layer.mlp.dense_h_to_4h.weight.data = quantize_weight(layer.mlp.dense_h_to_4h.weight.data)
        layer.mlp.dense_4h_to_h.weight.data = quantize_weight(layer.mlp.dense_4h_to_h.weight.data)

    def bert_layer_quantize(layer):
        layer.attention.self.query.weight.data = quantize_weight(layer.attention.self.query.weight.data)
        layer.attention.self.key.weight.data = quantize_weight(layer.attention.self.key.weight.data)
        layer.attention.self.value.weight.data = quantize_weight(layer.attention.self.value.weight.data)
        layer.attention.output.dense.weight.data = quantize_weight(layer.attention.output.dense.weight.data)
        if preln:
            layer.intermediate.dense_act.weight.data = quantize_weight(layer.intermediate.dense_act.weight.data)
        else:
            layer.intermediate.dense.weight.data = quantize_weight(layer.intermediate.dense.weight.data)
        layer.output.dense.weight.data = quantize_weight(layer.output.dense.weight.data)

    def quantize_fn(child):
        if megatron:
            # Quantize megatron GPT2 / GPT3 trained model
            megatron_layer_quantize(child)
        else:
            # Quantize either DeepSpeed or HuggingFace trained model
            bert_layer_quantize(child)

        return child

    return quantize_module(model=model, orig_class=orig_layer_impl, quantize_fn=quantize_fn)


def quantize_module(model, orig_class, quantize_fn):
    policy = {orig_class: quantize_fn}
    return _quantize_module(model, policy)


def _quantize_module(model, policies):
    for name, child in model.named_children():
        if child.__class__ in policies:
            orig = repr(child)
            setattr(model, name, policies[child.__class__](child))
            new = getattr(model, name)
        else:
            _quantize_module(child, policies)

    return model
