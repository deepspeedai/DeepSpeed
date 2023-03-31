# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..module_inject.replace_policy import HFBertLayerPolicy, replace_policies
from deepspeed.accelerator import get_accelerator


class WeightQuantization(object):

    def __init__(self, mlp_extra_grouping=True, mp_size=1):
        self.dense_scales = []
        self.qkv_scales = []
        self.mlp4hh_scales = []
        self.mlph4h_scales = []
        self.mlp_extra_grouping = mlp_extra_grouping
        self.mp_size = mp_size

    def quantize_data(self, data, quantize_bits, groups, key=None):
        data_groups = torch.split(data.float().view(-1), data.numel() // groups)
        max_d = [max(g.max(), g.min().abs()) for g in data_groups]
        data_scale = [float(1 << quantize_bits) / (2 * mx + 1e-5) for mx in max_d]
        data_int = [(g * s) for g, s in zip(data_groups, data_scale)]
        data_int = [
            di.round().clamp(-(1 << (quantize_bits - 1)), (((1 << (quantize_bits - 1)) - 1))) for di in data_int
        ]
        data_int = torch.cat(data_int).reshape(data.shape)
        data_int = data_int.to(torch.int8)
        data_scale = torch.cat([s.unsqueeze(0).unsqueeze(0) for s in data_scale])
        return data_int, data_scale

    def is_mlp(self, data, merge_count=1):
        return ((self.mp_size *data.shape[0] * merge_count) / data.shape[1] == 4 or \
                (self.mp_size *data.shape[1] * merge_count) / data.shape[0] == 4)

    def is_qkv(self, data):
        return ((self.mp_size * data.shape[0]) / data.shape[1] == 3 or \
                (self.mp_size * data.shape[1]) / data.shape[0] == 3)

    def Quantize(self, value_list, quantize_bits, groups, key, merge_dim=0):
        if self.mlp_extra_grouping and self.is_mlp(value_list[0], merge_count=len(value_list)):
            groups *= 2
        q_scale = []
        index = 0
        for data in value_list:
            data_int, data_scale = self.quantize_data(data, quantize_bits, groups, key)
            q_scale.append(data_scale)
            value_list[index] = data_int
            index += 1
        q_scale = (1 /
                   torch.cat(q_scale, dim=merge_dim).to(get_accelerator().current_device_name()).view(-1).unsqueeze(0))
        if "mlp.dense_4h_to_h.weight" in key:
            self.mlp4hh_scales.append(q_scale)
        elif "mlp.dense_h_to_4h.weight" in key:
            self.mlph4h_scales.append(q_scale)
        elif "attention.query_key_value.weight" in key:
            self.qkv_scales.append(q_scale)
        else:
            self.dense_scales.append(q_scale)
        return value_list

    def merge_layer_scales(self, layer_scales):
        max_dim = max([s.shape[-1] for s in layer_scales])
        layer_scales = [
            torch.cat((s, torch.zeros((1, max_dim - s.shape[-1]), device=get_accelerator().current_device_name())),
                      dim=-1) if s.shape[-1] < max_dim else s for s in layer_scales
        ]
        return torch.cat(layer_scales).unsqueeze(0)

    def merge_scales(self):
        all_scales = []
        for dense_scale, qkv_scale, m4hh_scale, mh4h_scale in \
            zip(self.dense_scales, self.qkv_scales, self.mlp4hh_scales, self.mlph4h_scales):
            all_scales.append(self.merge_layer_scales([qkv_scale, dense_scale, mh4h_scale, m4hh_scale]))
        return torch.cat(all_scales)

    def merge_scales_split(self, split_count):
        all_scales = [[] for _ in range(split_count)]
        for dense_scale, qkv_scale, m4hh_scale, mh4h_scale in \
            zip(self.dense_scales, self.qkv_scales, self.mlp4hh_scales, self.mlph4h_scales):
            dense_scale = torch.split(dense_scale, dense_scale.numel() // split_count)
            qkv_scale = torch.split(qkv_scale, qkv_scale.numel() // split_count)
            m4hh_scale = torch.split(m4hh_scale, m4hh_scale.numel() // split_count)
            mh4h_scale = torch.split(mh4h_scale, mh4h_scale.numel() // split_count)
            for s in range(split_count):
                all_scales[s].append(
                    torch.cat([
                        torch.cat((qkv_scale[s], torch.zeros_like(qkv_scale[s])), dim=1),
                        torch.cat((dense_scale[s], torch.zeros_like(dense_scale[s])), dim=1), mh4h_scale[s],
                        m4hh_scale[s]
                    ]).unsqueeze(0))
            for scales_a in all_scales:
                torch.cat(scales_a)
        return all_scales

    def sd_quantize_megatron(self, sd, quantize_bits, groups):
        keys = sd.keys()
        for key in keys:
            value_list = [sd[key]]
            if "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key or \
                "mlp.dense_h_to_4h.weight" in key or "attention.query_key_value.weight" in key:
                value_list = self.Quantize(value_list, quantize_bits, groups, key=key)
            sd[key] = value_list[0]

        all_scales = self.merge_scales()
        return sd, all_scales

    def model_quantize(self, model, quantize_policy, quantize_bits, groups):
        all_scales = []

        def quantize_fn(layer, policy_cls):
            policy = policy_cls(layer)

            _, qkvw, _, dense_w, _, _ = policy.attention()
            _, _h4h_w, _, _4hh_w, _ = policy.mlp()
            keys = [qkvw, dense_w, _h4h_w, _4hh_w]
            layer_scales = []

            for key in range(len(keys)):
                if self.mlp_extra_grouping and self.is_mlp(keys[key]):
                    data_quantized, data_scale = self.quantize_data(keys[key], quantize_bits, groups * 2)
                elif policy_cls is HFBertLayerPolicy and self.is_qkv(keys[key]):
                    data_quantized, data_scale = self.quantize_data(keys[key], quantize_bits, groups * 3)
                else:
                    data_quantized, data_scale = self.quantize_data(keys[key], quantize_bits, groups)
                keys[key].copy_(data_quantized)
                layer_scales.append((1 / data_scale.to(get_accelerator().current_device_name()).view(-1).unsqueeze(0)))
            all_scales.append(self.merge_layer_scales(layer_scales))
            return layer

        def _quantize_module(model, policies):
            for name, child in model.named_children():
                if child.__class__ in policies:
                    quantize_fn, replace_policy = policies[child.__class__]
                    setattr(model, name, quantize_fn(child, replace_policy))
                else:
                    _quantize_module(child, policies)

            return model

        policy = {}
        if quantize_policy is not None:
            for layer_name, replace_policy in quantize_policy.items():
                policy.update({layer_name: (quantize_fn, replace_policy)})
        else:
            for plcy in replace_policies:
                policy.update({plcy._orig_layer_class: (quantize_fn, plcy)})

        quantized_module = _quantize_module(model, policy)

        return quantized_module, torch.cat(all_scales)
