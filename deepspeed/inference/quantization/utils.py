# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
from torch import Tensor
from typing import Tuple
import torch.nn as nn
from typing import Dict, Callable, Union
from deepspeed.accelerator import get_accelerator
import functools

device = get_accelerator().device_name() if get_accelerator().is_available() else 'cpu'

quantizer_cuda_module = None


def get_quantizer_cuda_module():
    global quantizer_cuda_module
    if quantizer_cuda_module is None:
        quantizer_cuda_module = deepspeed.ops.op_builder.QuantizerBuilder().load()
    return quantizer_cuda_module


def tensor_clamp(tensor: Tensor, min, max) -> Tensor:
    if tensor.device.type == 'cpu' and tensor.dtype == torch.float16:
        # CPU does not support FP16 clamp
        return tensor.to(dtype=torch.float32).clamp_(min, max).to(dtype=torch.float16)
    else:
        return tensor.clamp_(min, max)


def tensor_round(tensor: Tensor) -> Tensor:
    if tensor.device.type == 'cpu' and tensor.dtype == torch.float16:
        # CPU does not support FP16 round
        return tensor.to(dtype=torch.float32).round_().to(dtype=torch.float16)
    else:
        return tensor.round_()


class Quantizer:

    def __init__(self, config: Dict) -> None:
        self.config = config
        assert self.config['num_bits'] == 4 or self.config[
            'num_bits'] == 8, 'Only INT4 and INT8 quantization is supported.'
        assert self.config['symmetric'] == False, 'Only asymmetric quantization is supported at this moment.'

    def quantize(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        assert tensor.shape[self.config['group_dim']] % self.config['group_size'] == 0 \
            , f'Tensor shape: {tensor.shape} quantization config {self.config}'

        tensor = torch.clone(tensor)

        shape = tensor.shape
        num_groups = shape[self.config['group_dim']] // self.config['group_size']
        new_shape = (shape[:self.config['group_dim']] + (num_groups, self.config['group_size']) +
                     shape[self.config['group_dim'] + 1:])
        tensor = tensor.view(new_shape)

        quantized_tensor, scale, min_value = self._quantize_int8(tensor)
        quantized_tensor = quantized_tensor.view(shape)

        if self.config['num_bits'] == 4:
            return self._compress_uint8_to_uint4(quantized_tensor), scale, min_value
        if self.config['num_bits'] == 8:
            return quantized_tensor, scale, min_value

        assert False, 'Unsupported quantization bits {}'.format(self.config['num_bits'])

    def _quantize_int8(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        q_range = 2**self.config['num_bits'] - 1
        min_value = tensor.amin(dim=self.config['group_dim'] + 1, keepdim=True)
        max_value = tensor.amax(dim=self.config['group_dim'] + 1, keepdim=True)

        scale = q_range / (max_value - min_value)

        tensor = tensor.sub_(min_value).mul_(scale)
        tensor = tensor_round(tensor_clamp(tensor, 0, q_range)).to(torch.uint8)
        return tensor, scale, min_value

    def _compress_uint8_to_uint4(self, tensor: Tensor) -> Tensor:
        assert tensor.shape[-1] % 2 == 0

        new_data_shape = list(tensor.shape)
        new_data_shape[-1] = new_data_shape[-1] // 2

        data = torch.empty(new_data_shape, dtype=torch.uint8, device=tensor.device)
        data = torch.bitwise_or(tensor[..., 0::2].bitwise_left_shift(4), tensor[..., 1::2])

        return data


class DeQuantizer:

    def __init__(self, config: Dict, dtype: torch.dtype) -> None:
        self.config = config
        self.dtype = dtype
        assert self.config['num_bits'] == 4 or self.config[
            'num_bits'] == 8, 'Only INT4 and INT8 quantization is supported.'
        assert self.config['symmetric'] == False, 'Only asymmetric quantization is supported at this moment.'

    def dequantize(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        # Use customized CUDA quantization kernel if possible.
        if self.config['group_size'] % 8 == 0 and \
                (self.config['num_bits'] == 4 or self.config['num_bits'] == 8) and \
                self.config['group_dim'] == len(tensor.shape) - 1 and \
                    self.dtype == torch.float16 and device == 'cuda':

            last_dimension_size = self.config['group_size']
            if self.config['num_bits'] == 4:
                last_dimension_size = last_dimension_size // 2
                quantized_tensor = get_quantizer_cuda_module().dequantize_int4_to_half_experimental(
                    tensor.reshape(-1, last_dimension_size), quant_scale, quant_min,
                    tensor.numel() // last_dimension_size, self.config['group_size'])
                shape = list(tensor.shape)
                shape[-1] = shape[-1] * 2
            elif self.config['num_bits'] == 8:
                # last_dimension_size = last_dimension_size // 2
                quantized_tensor = get_quantizer_cuda_module().dequantize_int8_to_half_experimental(
                    tensor.reshape(-1, last_dimension_size), quant_scale, quant_min,
                    tensor.numel() // last_dimension_size, self.config['group_size'])
                shape = list(tensor.shape)

            return quantized_tensor.reshape(shape)

        if self.config['num_bits'] == 4:
            tensor = self._decompress_uint4_to_uint8(tensor)
        elif self.config['num_bits'] != 8:
            assert False, 'Unsupported quantization bits {}'.format(self.config['num_bits'])

        shape = tensor.shape
        num_groups = shape[self.config['group_dim']] // self.config['group_size']
        new_shape = (shape[:self.config['group_dim']] + (num_groups, self.config['group_size']) +
                     shape[self.config['group_dim'] + 1:])
        tensor = tensor.view(new_shape)

        dequantized_tensor = self._dequantize_int8(tensor, quant_scale, quant_min).view(shape)
        return dequantized_tensor

    def _dequantize_int8(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        assert tensor.dtype == torch.uint8
        data = torch.zeros_like(tensor, dtype=self.dtype, device=tensor.device)
        data = data.copy_(tensor)
        data = data.div_(quant_scale).add_(quant_min)

        return data

    def _decompress_uint4_to_uint8(self, tensor: Tensor) -> Tensor:
        new_data_shape = list(tensor.shape)
        new_data_shape[-1] = new_data_shape[-1] * 2
        data = torch.empty(new_data_shape, dtype=torch.uint8, device=tensor.device)
        data[..., 0::2] = tensor.bitwise_right_shift(4)
        data[..., 1::2] = tensor.bitwise_and(0xF)

        return data


def get_AsyncPartitionedParameterSwapper(model: nn.Module):
    for param_name, param in model.named_parameters():
        if hasattr(param, 'nvme_swapper') and param.nvme_swapper is not None:
            return param.nvme_swapper
    return None


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


def concat_to_compat_param(quantized_weight: Tensor,
                           quant_scale: Tensor,
                           quant_min: Tensor,
                           return_param: bool = True) -> Union[nn.Parameter, Tensor]:
    shape_wieght = quantized_weight.shape
    shape_scale = quant_scale.shape
    shape_min = quant_min.shape

    quantized_weight = torch.flatten(quantized_weight)
    quant_scale = torch.flatten(quant_scale)
    quant_min = torch.flatten(quant_min)

    def deconcat_individual_tensors(shape_wieght: torch.Size, shape_scale: torch.Size,
                                    shape_min: torch.Size) -> Callable:

        def fn(compat_tensor: nn.Parameter) -> Tuple[Tensor, Tensor, Tensor]:
            weight = torch.narrow(compat_tensor, 0, 0, shape_wieght.numel()).view(shape_wieght)
            scale = torch.narrow(compat_tensor, 0, shape_wieght.numel(), shape_scale.numel()).view(shape_scale)
            min_val = torch.narrow(compat_tensor, 0,
                                   shape_wieght.numel() + shape_scale.numel(), shape_min.numel()).view(shape_min)

            return weight, scale, min_val

        return fn

    compat_tensor = torch.concat([quantized_weight, quant_scale, quant_min])
    if return_param:
        compat_tensor = nn.Parameter(compat_tensor, requires_grad=False)
    compat_tensor.deconcat = deconcat_individual_tensors(shape_wieght, shape_scale, shape_min)

    return compat_tensor


def _quantize_param(param: nn.Parameter, quant_config: Dict):
    assert not hasattr(param, 'weight_quantized'), 'Parameter has already been quantized.'
    quantizer = Quantizer(quant_config)
    dequantizer = DeQuantizer(quant_config, param.dtype)

    quantized_weight, quant_scale, quant_min = quantizer.quantize(param.data)

    quantized_weight = quantized_weight.view(param.dtype)
    quant_scale = quant_scale.view(param.dtype)
    quant_min = quant_min.view(param.dtype)

    quantized_compat_tensor = concat_to_compat_param(quantized_weight, quant_scale, quant_min)
    param.data = quantized_compat_tensor
    param.deconcat = quantized_compat_tensor.deconcat

    param.quantizer = quantizer
    param.dequantizer = dequantizer
    setattr(param, 'weight_quantized', True)


def wrap_quantized_functional(f):

    @functools.wraps(f)
    def wrapper(input: Tensor, weight: nn.Parameter, *args, **kwargs) -> Tensor:
        if hasattr(weight, 'weight_quantized') and getattr(weight, 'weight_quantized'):
            quantized_weight, quant_scale, quant_min = weight.deconcat(weight)
            temp_dequantized_weight = weight.dequantizer.dequantize(quantized_weight.view(torch.uint8), quant_scale,
                                                                    quant_min)
            return f(input, temp_dequantized_weight, *args, **kwargs)
        else:
            return f(input, weight, *args, **kwargs)

    return wrapper


def wrap_load_from_state_dict(f):

    @functools.wraps(f)
    def wrapper(model, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        replaced_old_value = None
        key = None
        # We may have nested wrappers if we launch multiple initialization context.
        # Use state_dict_quantized flag to quantize state_dict only once
        if hasattr(model.weight, 'weight_quantized') and getattr(
                model.weight, 'weight_quantized') and not hasattr(model.weight, 'state_dict_quantized'):
            setattr(model.weight, 'state_dict_quantized', True)
            key = prefix + 'weight'
            if key in state_dict:
                quantized_weight, quant_scale, quant_min = model.weight.quantizer.quantize(state_dict[key])
                quantized_weight = quantized_weight.view(model.weight.dtype)
                quant_scale = quant_scale.view(model.weight.dtype)
                quant_min = quant_min.view(model.weight.dtype)

                replaced_old_value = state_dict[key]

                state_dict[key] = concat_to_compat_param(quantized_weight, quant_scale, quant_min)

        f(model, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        if replaced_old_value is not None:
            state_dict[key] = replaced_old_value
            delattr(model.weight, 'state_dict_quantized')

    return wrapper


WEIGHT_QUANTIZATION_LAYERS = (
    nn.Linear,
    nn.Embedding,
)
