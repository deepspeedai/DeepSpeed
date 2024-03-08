# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ....inference_utils import DtypeEnum
from ....logging import inference_logger
from deepspeed.ops.op_builder import InferenceCoreBuilder
from ... import DSKernelBase


class CUDAWf6Af16Linear(DSKernelBase):
    """
    Wrapper around the CUDA kernel of Wf6Af16 quantized linear.

    Performs z = x @ y
    """
    supported_dtypes = [DtypeEnum.fp16]

    def __init__(self):
        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.kernel = self.inf_module.cuda_wf6af16_linear
        # The split_k_map is profiled on A100-80G GPU for some common shapes.
        # It is an array of dictionaries, where the array index is the tokens chunk id.
        # The dictionary is the mapping from the output channel to the split-K size.
        self.split_k_map = [
            {  # tokens: [1, 64]
                3072: 18,
                4096: 13,
                5120: 10,
                6144: 9,
                8192: 6,
                10240: 5,
                14336: 7,
                28672: 7,
                57344: 7
            },
            {  # tokens: [65:128]
                3072: 9,
                4096: 6,
                5120: 5,
                6144: 9,
                8192: 3,
                10240: 5,
                14336: 7,
                28672: 7,
                57344: 6
            },
            {  # tokens: [129:192]
                3072: 6,
                4096: 4,
                5120: 7,
                6144: 3,
                8192: 2,
                10240: 5,
                14336: 5,
                28672: 5,
                57344: 4
            },
            {  # tokens: [193:256]
                3072: 9,
                4096: 3,
                5120: 5,
                6144: 2,
                8192: 5,
                10240: 4,
                14336: 8,
                28672: 6,
                57344: 4
            },
            {  # tokens: [257:320]
                3072: 7,
                4096: 5,
                5120: 2,
                6144: 5,
                8192: 4,
                10240: 1,
                14336: 3,
                28672: 3,
                57344: 4
            },
            {  # tokens: [321:384]
                3072: 3,
                4096: 2,
                5120: 5,
                6144: 3,
                8192: 1,
                10240: 8,
                14336: 3,
                28672: 4,
                57344: 3
            },
            {  # tokens: [385:448]
                3072: 5,
                4096: 7,
                5120: 3,
                6144: 5,
                8192: 7,
                10240: 3,
                14336: 1,
                28672: 1,
                57344: 3
            },
            {  # tokens: [449:512]
                3072: 2,
                4096: 5,
                5120: 4,
                6144: 1,
                8192: 5,
                10240: 2,
                14336: 6,
                28672: 4,
                57344: 1
            },
            {  # tokens: [513:576]
                3072: 2,
                4096: 3,
                5120: 1,
                6144: 1,
                8192: 3,
                10240: 3,
                14336: 3,
                28672: 1,
                57344: 1
            },
            {  # tokens: [577:640]
                3072: 5,
                4096: 4,
                5120: 1,
                6144: 4,
                8192: 2,
                10240: 1,
                14336: 1,
                28672: 1,
                57344: 1
            },
            {  # tokens: [641:704]
                3072: 3,
                4096: 1,
                5120: 2,
                6144: 2,
                8192: 1,
                10240: 2,
                14336: 1,
                28672: 1,
                57344: 1
            },
            {  # tokens: [705:768]
                3072: 3,
                4096: 1,
                5120: 3,
                6144: 2,
                8192: 1,
                10240: 1,
                14336: 1,
                28672: 1,
                57344: 1
            }
        ]

    def __call__(self, output: torch.Tensor, hidden_states: torch.Tensor, weights_2bit: torch.Tensor,
                 weights_4bit: torch.Tensor, scale: torch.Tensor, out_channels, tokens, in_channels) -> torch.Tensor:
        """
        Matmul kernel of FP6 weight-only quantized linear. All inputs should be contiguous.
        It does not support batched-matmul.

        Parameters:
            output (torch.Tensor): Output tensor. Shape is of [token_number, out_features]
            hidden_states (torch.Tensor): Input tensor. Shape is of [token_number, in_features]
            weights_2bit (torch.Tensor): Input tensor of the 2-bit slice. Shape is of [out_features*2/8, in_features]
            weights_4bit (torch.Tensor): Input tensor of the 4-bit slice. Shape is of [out_features*4/8, in_features]
            scale (torch.Tensor): Input tensor. Shape is of [out_features], since the scale is per output channel
            out_channels (int): The number of output channels
            tokens (int): The number of tokens
            in_channels (int): The number of input channels
        """

        if out_channels % 256 != 0 or in_channels % 64 != 0:
            raise ValueError("The out and in channel should be multiple of 256 and 64 respectively.")

        # TODO: add a more general heuristic to determine the split-K.
        split_k = -1  # not initialized
        if tokens <= 768:
            # Try to find the split-K from the pre-profiled map.
            tokens_chunk_id = (tokens - 1) // 64
            split_k = self.split_k_map[tokens_chunk_id].get(out_channels, -1)
        if split_k == -1:
            split_k = 1
            inference_logger().warning(
                f"The split-K setting may be suboptimal for shape {tokens}x{in_channels}x{out_channels}...")

        workspace = self.get_workspace(out_channels, tokens, in_channels, split_k, torch.float, hidden_states.device)
        self.kernel(output, hidden_states, weights_2bit, weights_4bit, scale, workspace, out_channels, tokens,
                    in_channels, split_k)

    def get_workspace(self, out_channels: int, tokens: int, in_channels: int, split_k: int, dtype,
                      device) -> torch.Tensor:
        """
        Allocate workspace for the kernel. The workspace is used to store the intermediate results of the matmul before
        split-K. The split-K size is determined by the size of the matmul.
        """
        workspace = torch.empty((split_k, out_channels, tokens), dtype=dtype, device=device)

        return workspace
