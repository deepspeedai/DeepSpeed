# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import math
from deepspeed.utils import logger
from deepspeed.ops.quantizer import ds_quantizer

TWO_D_PARAMS = 6


class Quantizer(object):

    def __init__(self,
                 q_groups=1,
                 q_mixed_fp16=False,
                 q_change_ratio=0.01,
                 q_type=0,
                 q_rounding=0,
                 q_verbose=False,
                 q_eigenvalue=False,
                 use_quantizer_kernel=False,
                 layer_num=0):

        self.q_groups = q_groups
        self.q_mixed_fp16 = q_mixed_fp16
        self.q_change_ratio = q_change_ratio
        self.q_type = q_type
        self.qsteps = 0
        self.quantize_real_ratio = 1.000
        self.q_verbose = q_verbose
        self.q_eigenvalue = q_eigenvalue
        self.use_quantizer_kernel = use_quantizer_kernel
        self.q_rounding = q_rounding
        self.layer_num = layer_num

    def any_precision_switch(self):
        # Temporary disabled functionality
        if self.layer_num == 0:
            return True
        result = False
        for index in range(self.layer_num):
            if self.q_start_bits[index] != self.q_target_bits:
                next_step = self.qsteps + (TWO_D_PARAMS * (self.layer_num if self.layer_num != 0 else 1))
                if next_step >= self.q_period[index]:
                    result = True
        return result

    def quantize(self, parameter_group, overflow, eigenvalue_enabled, block_eigenvalue={}):

        if overflow and not eigenvalue_enabled:
            return

        self.step()

        self.update_fp16_ratio()

        for i in range(len(parameter_group)):
            for p in parameter_group[i]:
                if len(p.size()) > 1 and hasattr(p, "start_bits") and p.start_bits:
                    param_id = id(p)
                    if block_eigenvalue is None:
                        eigenvalue, layer_id = None, 0
                    else:
                        eigenvalue, layer_id = block_eigenvalue[param_id] if param_id in block_eigenvalue else (None,
                                                                                                                0)
                    if eigenvalue is not None:
                        factor = 1 + math.floor(eigenvalue * 4)
                        p.data = self.compute_quantization(p.data, layer_id, factor)
                    else:
                        p.data = self.compute_quantization(p, layer_id)

    def step(self):
        self.qsteps += 1

    def quantize_highbit(self, inputs, num_bits):

        q_range = 2**num_bits
        input_flat = inputs.reshape(self.q_groups, -1)
        g_min = input_flat.amin(dim=-1, keepdim=True)
        g_max = input_flat.amax(dim=-1, keepdim=True)

        # Random number generator (Uniform)
        if self.q_rounding == 'nearest':
            p = 0.
        else:
            p = input_flat.new(input_flat.shape).uniform_(-0.5, 0.5)

        if self.q_type == 'symmetric':
            scale = 2 * torch.max(torch.abs(g_min), torch.abs(g_max)) / q_range
            zero_point = 0.
            input_flat = (input_flat / scale + p).round().clamp(-(q_range >> 1), (q_range >> 1) - 1) * scale
        elif self.q_type == 'asymmetric':
            scale = (g_max - g_min) / q_range
            zero_point = (g_min / scale).round() * scale
            input_flat = ((input_flat - zero_point) / scale + p).round().clamp(0, (q_range - 1)) * scale + zero_point
        output = input_flat.reshape(inputs.shape).contiguous()
        return output

    def quantize_tenary(self, inputs):
        input_flat = inputs.reshape(self.q_groups, -1)
        n = input_flat.shape[1]
        m = input_flat.norm(p=1, dim=1).div(n)
        thres = (0.7 * m).view(-1, 1)  #.expand_as(input_flat)
        pos = (input_flat > thres).type(inputs.type())
        neg = (input_flat < -thres).type(inputs.type())
        mask = (input_flat.abs() > thres).type(inputs.type())
        alpha = ((mask * input_flat).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
        output = alpha * pos - alpha * neg
        output = output.reshape(inputs.shape).contiguous()
        return output

    def quantize_binary(self, inputs):
        input_flat = inputs.reshape(self.q_groups, -1)
        n = input_flat.shape[1]
        m = input_flat.norm(p=1, dim=1, keepdim=True).div(n)
        output = input_flat.sign().mul(m)
        output = output.reshape(inputs.shape).contiguous()
        return output

    def mixed_fp16_quantize(self, input, input_q, index):
        if self.q_mixed_fp16 and self.q_start_bits[index] >= (self.q_target_bits - 1):
            input_q = input * self.quantize_real_ratio + (1 - self.quantize_real_ratio) * input_q
            return input_q
        return input_q

    def compute_quantization(self, input, index=0, factor=1):
        # fixing the quantization bits based on the training steps
        # when reducing 1 bit at each period, we increase the period
        # to go slowly toward the target quantization bits
        # the period and starting bit can be configured

        if input.start_bits != input.target_bits:
            if self.qsteps >= input.q_period:
                self.quantize_real_ratio = 1.0
                input.q_period <<= 1
                input.q_period *= factor
                input.start_bits -= 1
                if self.q_verbose:
                    logger.info(
                        f'Quantization settings: current bit-precision = {input.start_bits}, step = {self.qsteps}, quantization period = {input.q_period}, index = {index}'
                    )
        assert (input.start_bits >= input.target_bits), \
            'Quantization bit is lower than target precision bits!'

        if self.use_quantizer_kernel:
            if input.start_bits <= 2:
                raise ValueError('Quantization bit is too low, please do it without quantization kernel!')
            input_q = ds_quantizer(input.data.clone(),
                                   self.q_groups,
                                   input.start_bits,
                                   asym=False if self.q_type == 'symmetric' else True,
                                   sr=False if self.q_rounding == 'nearest_neighbor' else True)
        else:
            if input.start_bits >= 3:
                input_flat = self.quantize_highbit(input.data, input.start_bits)
            elif input.start_bits == 2:
                assert self.q_type == 'symmetric', 'Quantization type is not symmetric!'
                assert self.q_rounding == 'nearest', 'Quantization rounding is not nearest_neighbor!'
                input_flat = self.quantize_tenary(input.data)
            elif input.start_bits == 1:
                assert self.q_type == 'symmetric', 'Quantization type is not symmetric!'
                assert self.q_rounding == 'nearest', 'Quantization rounding is not nearest_neighbor!'
                input_flat = self.quantize_binary(input.data)
        if self.use_quantizer_kernel:
            return self.mixed_fp16_quantize(input.data, input_q, index)
        else:
            if self.q_mixed_fp16 and input.start_bits >= input.target_bits - 1:
                input_flat = self.quantize_real_ratio * input.data + \
                              (1 - self.quantize_real_ratio) * input_flat
            return input_flat

    def update_fp16_ratio(self):
        if self.q_mixed_fp16:
            if self.quantize_real_ratio > 0:
                self.quantize_real_ratio -= self.q_change_ratio
            else:
                self.quantize_real_ratio = 0.000
