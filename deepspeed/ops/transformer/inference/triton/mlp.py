# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import math
import torch.nn as nn
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from ..op_binding import MLPGemmOp, VectorMatMulOp, GELUGemmOp, ResidualAddOp


class TritonMLP(nn.Module):

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False):
        super(TritonMLP, self).__init__()

        self.config = config
        data_type = self.config.dtype
        data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        device = get_accelerator().current_device_name()
        self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                    requires_grad=False)
        self.attn_nb = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                    requires_grad=False)
        intm_size_per_partition = self.config.intermediate_size // self.config.mp_size
        self.inter_w = nn.Parameter(torch.empty(self.config.hidden_size,
                                                intm_size_per_partition,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.inter_b = nn.Parameter(torch.empty(intm_size_per_partition, dtype=data_type_fp, device=device),
                                    requires_grad=False)
        self.output_w = nn.Parameter(torch.empty(intm_size_per_partition,
                                                 self.config.hidden_size,
                                                 dtype=data_type,
                                                 device=device),
                                     requires_grad=False)
        self.output_b = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                     requires_grad=False)

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))
        self.mp_group = mp_group

        self.mlp_gemm_func = MLPGemmOp(config)
        self.vector_matmul_func = VectorMatMulOp(config)
        self.fused_gemm_gelu = GELUGemmOp(config)
        self.residual_add_func = ResidualAddOp(config)

    def forward(self, input, residual, residual_norm, bias):
        residual_add = None
        if self.attn_nw is None:
            output = self.fused_gemm_gelu(input=residual_norm,
                                          weight=self.inter_w,
                                          bias=self.inter_b,
                                          weight_out=self.output_w)
        else:
            output, residual_add = self.mlp_gemm_func(input=input,
                                                      residual=residual,
                                                      input_bias=bias,
                                                      weight_interm=self.inter_w,
                                                      weight_out=self.output_w,
                                                      bias=self.inter_b,
                                                      gamma=self.attn_nw,
                                                      beta=self.attn_nb)
        residual = self.residual_add_func(hidden_state=output,
                                          residual=residual,
                                          attention_output=input,
                                          attention_bias=bias if bias is not None else self.output_b,
                                          final_bias=self.output_b,
                                          add_bias=bias is not None,
                                          residual_add=residual_add)

        if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(residual, group=self.mp_group)

        return residual
