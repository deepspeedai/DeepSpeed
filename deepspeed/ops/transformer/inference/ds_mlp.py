'''
Copyright 2022 The Microsoft DeepSpeed Team
'''

import torch
from torch.autograd import Function
from deepspeed.utils.types import ActivationFuncType
from deepspeed import comm as dist
import torch.nn as nn
import math
from ... import op_builder

inference_cuda_module = None


class DeepSpeedMLPFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                residual,
                residual_norm,
                bias,
                inter_w,
                inter_b,
                attn_nw,
                attn_nb,
                config,
                mp_group,
                output_b,
                output_w,
                q_scales,
                q_groups,
                merge_count,
                mlp_gemm_func,
                fused_gemm_gelu,
                vector_matmul_func,
                bias_residual_func,
                residual_add_func,
                activation_func_type=ActivationFuncType.GELU):

        if attn_nw is None:
            output = fused_gemm_gelu(residual_norm,
                                     inter_w,
                                     inter_w.scale,
                                     inter_b,
                                     output_w,
                                     output_w.scale,
                                     config.epsilon,
                                     config.pre_layer_norm,
                                     config.q_int8,
                                     False)
        else:
            output, residual_add = mlp_gemm_func(input,
                                             residual,
                                             bias,
                                             inter_w,
                                             output_w,
                                             inter_b,
                                             attn_nw,
                                             attn_nb,
                                             config.epsilon,
                                             config.pre_layer_norm,
                                             config.mlp_after_attn,
                                             inter_w.scale,
                                             output_w.scale,
                                             config.q_int8,
                                             config.mlp_act_func_type)
        residual = residual if config.pre_layer_norm else residual_add
        residual_add_func(
            output,                # hidden state
            residual,              # residual
            input,                 # attention output
            bias if bias is not None else output_b,
            output_b,
            config.mp_size,         # model parallel size
            config.mlp_after_attn,  # whether mlp is after attention (GPTJ model architecture runs the MLP layer in parallel with attention)
            bias is not None,       # whether bias addition is fused
            config.pre_layer_norm)  # whether the layer norm is applied before attention
        if mp_group is not None and dist.get_world_size(group=mp_group) > 1:
            dist.all_reduce(residual, group=mp_group)
        return residual

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedMLP(nn.Module):
    def __init__(self,
                 config,
                 mp_group=None,
                 q_scales=None,
                 q_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        super(DeepSpeedMLP, self).__init__()

        self.config = config
        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        device = torch.cuda.current_device()  #if config.bigscience_bloom else 'cpu'
        self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
        self.attn_nb = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
        intm_size_per_partition = self.config.intermediate_size // self.config.mp_size
        self.inter_w = nn.Parameter(torch.empty(self.config.hidden_size,
                                                intm_size_per_partition,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.inter_b = nn.Parameter(torch.empty(intm_size_per_partition,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
        self.output_w = nn.Parameter(torch.empty(intm_size_per_partition,
                                                 self.config.hidden_size,
                                                 dtype=data_type,
                                                 device=device),
                                     requires_grad=False)
        self.output_b = nn.Parameter(torch.empty(self.config.hidden_size,
                                                 dtype=data_type_fp,
                                                 device=device),
                                     requires_grad=False)

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))

        # load the cuda module
        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()

        self.mp_group = mp_group
        self.mlp_gemm_func = inference_cuda_module.mlp_gemm_fp16 if config.fp16 else \
                                    inference_cuda_module.mlp_gemm_fp32
        self.vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else \
                                inference_cuda_module.vector_matmul_fp32
        self.fused_gemm_gelu = inference_cuda_module.fused_gemm_gelu_fp16 if config.fp16 else \
                                    inference_cuda_module.fused_gemm_gelu_fp32

        self.bias_residual_func = inference_cuda_module.bias_residual_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.bias_residual_fp32

        self.residual_add_func = inference_cuda_module.residual_add_bias_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.residual_add_bias_fp32

    def forward(self, input, residual, residual_norm, bias):
        return DeepSpeedMLPFunction.apply(input,
                                          residual,
                                          residual_norm,
                                          bias,
                                          self.inter_w,
                                          self.inter_b,
                                          self.attn_nw,
                                          self.attn_nb,
                                          self.config,
                                          self.mp_group,
                                          self.output_b,
                                          self.output_w,
                                          self.q_scales,
                                          self.q_groups,
                                          self.merge_count,
                                          self.mlp_gemm_func,
                                          self.fused_gemm_gelu,
                                          self.vector_matmul_func,
                                          self.bias_residual_func,
                                          self.residual_add_func)
