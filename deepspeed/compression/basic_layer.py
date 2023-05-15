# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import math
from torch import nn
from torch.nn import init
import deepspeed.comm as dist
from .utils import TopKBinarizer, SymQuantizer, AsymQuantizer, TernaryQuantizer, BinaryQuantizer
from deepspeed.utils import logger

g_mpu = None


class QuantAct(nn.Module):
    """
    Class to quantize given activations. Note that when using this function, the input activation quantization range will be fixed for all
    tokens/images for inference. This generally will affect some accuracy but achieve better latency performance.
    Parameters:
    ----------
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    quant_mode : str, default 'symmetric'
    """

    def __init__(self, act_range_momentum=0.95, quant_mode='symmetric'):
        super(QuantAct, self).__init__()

        self.act_range_momentum = act_range_momentum
        self.quant_mode = quant_mode
        if quant_mode == 'symmetric':
            self.act_function = SymQuantizer.apply
        else:
            self.act_function = AsymQuantizer.apply

        self.register_buffer('x_min_max', torch.zeros(2))

    def forward(self, x, num_bits, *args):
        """
        x: the activation that we need to quantize
        num_bits: the number of bits we need to quantize the activation to
        *args: some extra arguments that are useless but needed for align with the interface of other quantization functions
        """

        if self.training:
            x_min = x.data.min()
            x_max = x.data.max()

            # Initialization
            if self.x_min_max[0] == self.x_min_max[1]:
                self.x_min_max[0] = x_min
                self.x_min_max[1] = x_max

            # if do not need momentum, please set self.act_range_momentum = 0
            self.x_min_max[0] = self.x_min_max[0] * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
            self.x_min_max[1] = self.x_min_max[1] * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        x_q = self.act_function(x, num_bits, self.x_min_max[0], self.x_min_max[1])

        return x_q


class Embedding_Compress(nn.Embedding):

    def __init__(self, *kargs):
        super(Embedding_Compress, self).__init__(*kargs)
        self.weight.start_bits = None
        self.weight.target_bits = None
        self.weight.q_period = None
        self.weight_quantization_enabled_in_forward = False
        self.weight_quantization_enabled = False

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, weight_quantization={}'.format(
            self.num_embeddings, self.embedding_dim, self.weight.target_bits)

    def enable_weight_quantization(self, start_bits, target_bits, quantization_period,
                                   weight_quantization_enabled_in_forward, quantization_type, num_groups):
        self.weight.start_bits = start_bits
        self.weight.target_bits = target_bits
        self.weight.q_period = quantization_period
        self.weight_quantization_enabled_in_forward = weight_quantization_enabled_in_forward
        if self.weight_quantization_enabled_in_forward:
            logger.warning(
                "************ A lot of MoQ features are not supported in quantize_weight_in_forward mode, please consider to use DS-FP16 optimizer************"
            )
            if self.weight.target_bits >= 3:
                if quantization_type == 'symmetric':
                    self.weight_quantizer = SymQuantizer.apply
                else:
                    self.weight_quantizer = AsymQuantizer.apply
            elif self.weight.target_bits == 2:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for ternary weight quantization'
                self.weight_quantizer = TernaryQuantizer.apply
            elif self.weight.target_bits == 1:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for binary weight quantization'
                self.weight_quantizer = BinaryQuantizer.apply
            # for embedding, we always use token-wise quantization
            self.weight_quantize_num_groups = self.weight.size(0)

    def fix_weight_quantization(self):
        self.weight.data = self.weight_quantizer(self.weight, self.weight.target_bits, None, None,
                                                 self.weight_quantize_num_groups).data
        self.weight_quantization_enabled_in_forward = False
        return None

    def forward(self, input):
        if self.weight_quantization_enabled_in_forward and self.weight_quantization_enabled:
            weight = self.weight_quantizer(self.weight, self.weight.target_bits, None, None,
                                           self.weight_quantize_num_groups)
        else:
            weight = self.weight

        out = nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type,
                                      self.scale_grad_by_freq, self.sparse)
        return out


class LinearLayer_Compress(nn.Linear):
    """
    Linear layer with compression.
    """

    def __init__(self, *kargs, bias=True):
        super(LinearLayer_Compress, self).__init__(*kargs, bias=bias)
        self.sparse_pruning_method = None
        self.row_pruning_method = None
        self.head_pruning_method = None
        self.activation_quantization_method = None
        self.weight.start_bits = None
        self.weight.target_bits = None
        self.weight.q_period = None
        self.weight_quantization_enabled_in_forward = False
        self.weight_quantization_enabled = False
        self.sparse_pruning_enabled = False
        self.row_pruning_enabled = False
        self.head_pruning_enabled = False
        self.activation_quantization_enabled = False

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sparse pruning={}, row pruning={}, head pruning={}, activation quantization={}, weight_quantization={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.sparse_pruning_method is not None, \
            self.row_pruning_method is not None, self.head_pruning_method is not None, self.activation_quantization_method is not None, self.weight.target_bits)

    def enable_sparse_pruning(self, ratio, method):
        # Here, we support two cases: L1 norm based pruning and topk based pruning
        self.sparse_pruning_ratio = ratio
        self.sparse_pruning_method = method
        if method == 'l1':
            weight_norm = torch.abs(self.weight.data)
            mask = TopKBinarizer.apply(weight_norm, self.sparse_pruning_ratio, False)
            mask = mask.view(self.weight.size())
            mask = mask.to(self.weight.device)
        elif method == 'topk':
            self.sparse_mask_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            self.sparse_mask_scores.data = self.sparse_mask_scores.data.to(self.weight.device)
            init.kaiming_uniform_(self.sparse_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError

        self.register_buffer('sparse_pruning_mask', mask)

    def enable_row_pruning(self, ratio, method):
        # Here, we support two cases: L1 norm based pruning and topk based pruning
        self.row_pruning_ratio = ratio
        self.row_pruning_method = method

        if method == 'l1':
            # compute the l1 norm of each column
            weight_norm = torch.norm(self.weight.data, p=1, dim=1)
            mask = TopKBinarizer.apply(weight_norm, self.row_pruning_ratio, False)
            mask = mask.view(-1, 1)
            mask = mask.to(self.weight.device)
        elif method == 'topk':
            self.row_mask_scores = nn.Parameter(torch.Tensor(self.weight.size(0), 1))
            self.row_mask_scores.data = self.row_mask_scores.data.to(self.weight.device)
            init.kaiming_uniform_(self.row_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError

        self.register_buffer('row_pruning_mask', mask)

    def enable_head_pruning(self, ratio, method, num_heads):
        # Here, we support only topk based pruning
        self.num_heads = num_heads
        self.head_pruning_ratio = ratio
        self.head_pruning_method = method

        if method not in ['topk']:
            raise NotImplementedError
        else:
            self.head_pruning_ratio = ratio
            self.head_pruning_scores = nn.Parameter(torch.Tensor(1,
                                                                 self.num_heads))  # we apply the pruning to O matrix
            self.head_pruning_scores.data = self.head_pruning_scores.data.to(self.weight.device)
            init.kaiming_uniform_(self.head_pruning_scores, a=math.sqrt(5))

    def fix_sparse_pruning_helper(self):
        mask = self.get_mask(pruning_type='sparse')
        self.weight.data = self.weight.data * mask
        del self.sparse_pruning_mask
        if self.sparse_pruning_method == 'topk':
            del self.sparse_mask_scores
        self.sparse_pruning_method = None
        self.sparse_pruning_enabled = False
        return None

    def fix_row_col_pruning_helper(self, mask=None, dim_reduction=False):
        # This function is used for row/col pruning
        # particularly, if we have two back-to-back layers, F1 and F2; when
        # we remove rows from F1, we also need to remove columns from F2
        # However, if we only have one layer, F1, then we only need to mask pruned
        # rows as 0 in F1
        if mask is None:
            mask = self.get_mask(pruning_type='row').bool()
            if dim_reduction:
                start_bits = self.weight.start_bits
                target_bits = self.weight.target_bits
                q_period = self.weight.q_period
                self.weight = nn.Parameter(self.weight.data[mask.view(-1), :])
                self.weight.start_bits = start_bits
                self.weight.target_bits = target_bits
                self.weight.q_period = q_period
                if self.bias is not None:
                    self.bias = nn.Parameter(self.bias.data[mask.view(-1)])
                self.out_features = self.weight.size(0)
            else:
                self.weight.data = self.weight.data * mask.view(-1, 1)
                if self.bias is not None:
                    self.bias.data = self.bias.data * mask.view(-1)

            del self.row_pruning_mask
            if self.row_pruning_method == 'topk':
                del self.row_mask_scores
            self.row_pruning_method = None
        else:
            # this is generally for column pruning
            start_bits = self.weight.start_bits
            target_bits = self.weight.target_bits
            q_period = self.weight.q_period
            self.weight = nn.Parameter(self.weight.data[:, mask.view(-1)])
            self.weight.start_bits = start_bits
            self.weight.target_bits = target_bits
            self.weight.q_period = q_period
            self.in_features = self.weight.size(1)
            mask = None
        self.row_pruning_enabled = False
        return mask

    def fix_head_pruning_helper(self, mask=None, num_heads=None, dim_reduction=False):
        # similar as row/col pruning, head pruning also needs to prune QKV which is associated with O matrix
        num_heads = num_heads if num_heads else self.num_heads
        if mask is None:
            if self.head_pruning_method == 'topk':
                mask = self.get_mask(pruning_type='head').bool()
                if dim_reduction:
                    shape = self.weight.size(0)
                    start_bits = self.weight.start_bits
                    target_bits = self.weight.target_bits
                    q_period = self.weight.q_period
                    self.weight = nn.Parameter(self.weight.data.t().reshape(num_heads,
                                                                            -1)[mask.view(-1), :].reshape(-1,
                                                                                                          shape).t())
                    self.weight.start_bits = start_bits
                    self.weight.target_bits = target_bits
                    self.weight.q_period = q_period
                else:

                    shape = self.weight.size()
                    self.weight.data = (self.weight.data.t().reshape(self.num_heads, -1) * mask.view(-1, 1)).reshape(
                        shape[1], shape[0]).t()

                if self.head_pruning_method == 'topk':
                    del self.head_pruning_scores
                self.head_pruning_method = None
            else:
                raise NotImplementedError
        else:
            start_bits = self.weight.start_bits
            target_bits = self.weight.target_bits
            q_period = self.weight.q_period
            shape = self.weight.size(1)
            self.weight = nn.Parameter(self.weight.data.reshape(num_heads, -1)[mask.view(-1), :].reshape(-1, shape))
            self.weight.start_bits = start_bits
            self.weight.target_bits = target_bits
            self.weight.q_period = q_period
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias.data.reshape(num_heads, -1)[mask.view(-1), :].reshape(-1))
        self.head_pruning_enabled = False
        return mask

    def get_mask(self, pruning_type='row'):
        if pruning_type == 'sparse':
            if self.sparse_pruning_method == 'l1':
                return self.sparse_pruning_mask.to(self.weight.device)
            elif self.sparse_pruning_method == 'topk':
                return TopKBinarizer.apply(self.sparse_mask_scores, self.sparse_pruning_ratio, False)
            else:
                raise NotImplementedError
        if pruning_type == 'row':
            if self.row_pruning_method == 'l1':
                return self.row_pruning_mask.to(self.weight.device)
            elif self.row_pruning_method == 'topk':
                return TopKBinarizer.apply(self.row_mask_scores, self.row_pruning_ratio, False)
            else:
                raise NotImplementedError
        elif pruning_type == 'head':
            if self.head_pruning_method == 'topk':
                return TopKBinarizer.apply(self.head_pruning_scores, self.head_pruning_ratio, False)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def enable_weight_quantization(self, start_bits, target_bits, quantization_period,
                                   weight_quantization_enabled_in_forward, quantization_type, num_groups):
        self.weight.start_bits = start_bits
        self.weight.target_bits = target_bits
        self.weight.q_period = quantization_period
        self.weight_quantization_enabled_in_forward = weight_quantization_enabled_in_forward
        if self.weight_quantization_enabled_in_forward:
            logger.warning(
                "************ A lot of MoQ features are not supported in quantize_weight_in_forward mode, please consider to use DS-FP16 optimizer************"
            )
            if self.weight.target_bits >= 3:
                if quantization_type == 'symmetric':
                    self.weight_quantizer = SymQuantizer.apply
                else:
                    self.weight_quantizer = AsymQuantizer.apply
            elif self.weight.target_bits == 2:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for ternary weight quantization'
                self.weight_quantizer = TernaryQuantizer.apply
            elif self.weight.target_bits == 1:
                assert quantization_type == 'symmetric', 'Only symmetric quantization is supported for binary weight quantization'
                self.weight_quantizer = BinaryQuantizer.apply
            self.weight_quantize_num_groups = num_groups

    def fix_weight_quantization(self):
        self.weight.data = self.weight_quantizer(self.weight, self.weight.target_bits, None, None,
                                                 self.weight_quantize_num_groups).data
        self.weight_quantization_enabled_in_forward = False
        return None

    def enable_activation_quantization(self, bits, quantization_type, range_calibration):
        assert bits in [4, 8], 'Only 4/8 bits activation quantization are supported for now'
        self.activation_quantization_bits = bits
        self.activation_quantization_method = f"{quantization_type}_{range_calibration}"
        if range_calibration == 'static':
            self.activation_quantizer = QuantAct(quant_mode=quantization_type)
        else:
            if quantization_type == 'symmetric':
                self.activation_quantizer = SymQuantizer.apply
            else:
                self.activation_quantizer = AsymQuantizer.apply

    def head_pruning_reshape(self, w, mask):
        shape = w.shape
        return (w.t().reshape(self.num_heads, -1) * mask.view(-1, 1)).reshape(shape[1], shape[0]).t()

    def forward(self, input, skip_bias_add=False):

        if self.weight_quantization_enabled_in_forward and self.weight_quantization_enabled:
            weight = self.weight_quantizer(self.weight, self.weight.target_bits, None, None,
                                           self.weight_quantize_num_groups)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.sparse_pruning_enabled and self.sparse_pruning_method:
            mask = self.get_mask(pruning_type='sparse')
            weight = weight * mask.view(self.weight.size())

        if self.row_pruning_enabled and self.row_pruning_method:
            mask = self.get_mask(pruning_type='row')
            weight = weight * mask.view(-1, 1)
            if bias is not None:
                bias = bias * mask.view(-1)

        if self.head_pruning_enabled and self.head_pruning_method:
            mask = self.get_mask(pruning_type='head')
            weight = self.head_pruning_reshape(weight, mask)

        if self.activation_quantization_enabled:
            if 'dynamic' in self.activation_quantization_method:
                num_groups = input.numel() // input.size(-1)
            else:
                num_groups = 1
            input = self.activation_quantizer(input, self.activation_quantization_bits, None, None, num_groups)

        if skip_bias_add:
            # used for mpu linear layers
            output = nn.functional.linear(input, weight, None)
            return output, bias
        else:
            output = nn.functional.linear(input, weight, bias)
            return output


class Conv2dLayer_Compress(nn.Conv2d):
    """
    Conv2D layer with compression.
    """

    def __init__(self, *kargs):
        super(Conv2dLayer_Compress, self).__init__(*kargs)
        self.sparse_pruning_method = None
        self.channel_pruning_method = None
        self.activation_quantization_method = None
        self.weight.start_bits = None
        self.weight.target_bits = None
        self.weight.q_period = None
        self.weight_quantization_enabled_in_forward = False
        self.sparse_pruning_enabled = False
        self.channel_pruning_enabled = False
        self.activation_quantization_enabled = False

    def __repr__(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        output = s.format(**self.__dict__)

        return output + ' sparse pruning={}, channel pruning={}, activation quantization={}, weight_quantization={}'.format(
            self.sparse_pruning_method is not None, self.channel_pruning_method is not None,
            self.activation_quantization_method is not None, self.weight.target_bits)

    def enable_sparse_pruning(self, ratio, method):
        self.sparse_pruning_ratio = ratio
        self.sparse_pruning_method = method
        if method == 'l1':
            weight_norm = torch.abs(self.weight.data)
            mask = TopKBinarizer.apply(weight_norm, self.sparse_pruning_ratio, False)
            mask = mask.view(self.weight.size())
            mask = mask.to(self.weight.device)
        elif method == 'topk':
            self.sparse_mask_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            self.sparse_mask_scores.data = self.sparse_mask_scores.data.to(self.weight.device)
            init.kaiming_uniform_(self.sparse_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError

        self.register_buffer('sparse_pruning_mask', mask)

    def enable_channel_pruning(self, ratio, method):
        # Here, we support two cases: L1 norm based pruning and topk based pruning
        self.channel_pruning_ratio = ratio
        self.channel_pruning_method = method

        if method == 'l1':
            # compute the l1 norm of each conv2d kernel (the last three dimension)
            weight_norm = torch.norm(self.weight.data, p=1, dim=[1, 2, 3])
            mask = TopKBinarizer.apply(weight_norm, self.channel_pruning_ratio, False)
            mask = mask.view(-1, 1, 1, 1)
            mask = mask.to(self.weight.device)
        elif method == 'topk':
            self.channel_mask_scores = nn.Parameter(torch.Tensor(self.weight.size(0), 1, 1, 1))
            self.channel_mask_scores.data = self.channel_mask_scores.data.to(self.weight.device)
            init.kaiming_uniform_(self.channel_mask_scores, a=math.sqrt(5))
            mask = None
        else:
            raise NotImplementedError

        self.register_buffer('channel_pruning_mask', mask)

    def fix_sparse_pruning_helper(self):
        mask = self.get_mask(pruning_type='sparse')
        self.weight.data = self.weight.data * mask
        del self.sparse_pruning_mask
        if self.sparse_pruning_method == 'topk':
            del self.sparse_mask_scores
        self.sparse_pruning_method = None
        self.sparse_pruning_enabled = False
        return None

    def fix_channel_pruning_helper(self, mask=None, dim_reduction=False):
        if mask is None:
            if self.channel_pruning_method in ['l1', 'topk']:
                mask = self.get_mask(pruning_type='channel').bool()
                if dim_reduction:
                    start_bits = self.weight.start_bits
                    target_bits = self.weight.target_bits
                    q_period = self.weight.q_period
                    self.weight = nn.Parameter(self.weight.data[mask.view(-1), ...])
                    self.weight.start_bits = start_bits
                    self.weight.target_bits = target_bits
                    self.weight.q_period = q_period
                    if self.bias is not None:
                        self.bias = nn.Parameter(self.bias.data[mask.view(-1)])
                else:
                    self.weight.data = self.weight.data * mask.view(-1, 1, 1, 1)
                    if self.bias is not None:
                        self.bias.data = self.bias.data * mask.view(-1)
                del self.channel_pruning_mask
                if self.channel_pruning_method == 'topk':
                    del self.channel_mask_scores
                self.channel_pruning_method = None
            else:
                raise NotImplementedError
        else:
            start_bits = self.weight.start_bits
            target_bits = self.weight.target_bits
            q_period = self.weight.q_period
            self.weight = nn.Parameter(self.weight.data[:, mask.view(-1), ...])
            self.weight.start_bits = start_bits
            self.weight.target_bits = target_bits
            self.weight.q_period = q_period
            mask = None
        self.channel_pruning_enabled = False
        return mask

    def get_mask(self, pruning_type='sparse'):
        if pruning_type == 'sparse':
            if self.sparse_pruning_method == 'l1':
                return self.sparse_pruning_mask.to(self.weight.device)
            elif self.sparse_pruning_method == 'topk':
                return TopKBinarizer.apply(self.sparse_mask_scores, self.sparse_pruning_ratio, False)
            else:
                raise NotImplementedError
        elif pruning_type == 'channel':
            if self.channel_pruning_method == 'l1':
                return self.channel_pruning_mask.to(self.weight.device)
            elif self.channel_pruning_method == 'topk':
                return TopKBinarizer.apply(self.channel_mask_scores, self.channel_pruning_ratio, False)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def fix_weight_quantization(self):
        self.weight.data = self.weight_quantizer(self.weight, self.weight.target_bits, None, None,
                                                 self.weight_quantize_num_groups).data
        self.weight_quantization_enabled_in_forward = False
        return None

    def enable_weight_quantization(self, start_bits, target_bits, quantization_period,
                                   weight_quantization_enabled_in_forward, quantization_type, num_groups):
        self.weight.start_bits = start_bits
        self.weight.target_bits = target_bits
        self.weight.q_period = quantization_period
        self.weight_quantization_enabled_in_forward = weight_quantization_enabled_in_forward
        if self.weight_quantization_enabled_in_forward:
            assert self.weight.target_bits >= 4, 'Only >=4 bits weight quantization are supported during forward pass for now'
            logger.warning(
                "************ A lot of MoQ features are not supported in quantize_weight_in_forward mode, please consider to use DS-FP16 optimizer************"
            )
            if quantization_type == 'symmetric':
                self.weight_quantizer = SymQuantizer.apply
            else:
                self.weight_quantizer = AsymQuantizer.apply
            self.weight_quantize_num_groups = num_groups

    def enable_activation_quantization(self, bits, quantization_type, range_calibration):
        assert bits in [4, 8], 'Only 4/8 bits activation quantization are supported for now'
        self.activation_quantization_bits = bits
        self.activation_quantization_method = f"{quantization_type}_{range_calibration}"
        if range_calibration == 'static':
            self.activation_quantizer = QuantAct(quant_mode=quantization_type)
        else:
            if quantization_type == 'symmetric':
                self.activation_quantizer = SymQuantizer.apply
            else:
                self.activation_quantizer = AsymQuantizer.apply

    def forward(self, input):

        if self.weight_quantization_enabled_in_forward and self.weight_quantization_enabled:
            weight = self.weight_quantizer(self.weight, self.weight.target_bits, None, None,
                                           self.weight_quantize_num_groups)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.sparse_pruning_enabled and self.sparse_pruning_method:
            mask = self.get_mask(pruning_type='sparse')
            weight = weight * mask.view(self.weight.size())

        if self.channel_pruning_enabled:
            mask = self.get_mask(pruning_type='channel')
            weight = weight * mask.view(-1, 1, 1, 1)
            if bias is not None:
                bias = bias * mask.view(-1)

        if self.activation_quantization_enabled:
            if 'dynamic' in self.activation_quantization_method:
                num_groups = input.numel() // input[0].numel()
            else:
                num_groups = 1
            input = self.activation_quantizer(input, self.activation_quantization_bits, None, None, num_groups)

        return nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class BNLayer_Compress(nn.BatchNorm2d):

    def fix_channel_pruning_helper(self, mask, dim_reduction=True):
        self.weight = nn.Parameter(self.weight.data[mask.view(-1)])
        self.bias = nn.Parameter(self.bias.data[mask.view(-1)])
        self.running_mean = self.running_mean[mask.view(-1)]
        self.running_var = self.running_var[mask.view(-1)]


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = g_mpu.get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    dist.all_reduce(input_, group=group)

    return input_


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = g_mpu.get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = dist.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatenate along the last dimension."""
    group = g_mpu.get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    dist.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------


def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


class ColumnParallelLinear_Compress(LinearLayer_Compress):

    def __init__(self, mpu, input_size, output_size, bias=True, gather_output=True, skip_bias_add=False):
        # Keep input parameters
        global g_mpu
        g_mpu = mpu
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        assert output_size % world_size == 0
        self.output_size_per_partition = output_size // world_size

        super(ColumnParallelLinear_Compress, self).__init__(self.input_size, self.output_size_per_partition, bias=bias)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        if self.skip_bias_add:
            output_parallel, bias = super().forward(input_parallel, True)
        else:
            output_parallel = super().forward(input_parallel)
            bias = None
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output, bias


class RowParallelLinear_Compress(LinearLayer_Compress):

    def __init__(self, mpu, input_size, output_size, bias=True, input_is_parallel=False, skip_bias_add=False):
        # Keep input parameters
        global g_mpu
        g_mpu = mpu
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add

        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        assert input_size % world_size == 0
        self.input_size_per_partition = input_size // world_size

        super(RowParallelLinear_Compress, self).__init__(self.input_size_per_partition, self.output_size, bias=bias)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel, bias = super().forward(input_parallel, True)

        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            if bias is not None:
                output = output_ + bias
            else:
                output = output_
            output_bias = None
        else:
            output = output_
            output_bias = bias
        return output, output_bias
