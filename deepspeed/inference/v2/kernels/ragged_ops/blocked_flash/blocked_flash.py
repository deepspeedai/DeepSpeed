# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.accelerator import get_accelerator
from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import RaggedOpsBuilder

from ... import DSKernelBase


def get_q_block_size(head_size: int) -> int:
    """
    Returns the query block size required by the kernel given a head size.
    """
    cc_major, cc_minor = torch.cuda.get_device_capability(get_accelerator().current_device())  #ignore-cuda

    if cc_major < 8:
        raise RuntimeError("Blocked attention requires CUDA compute capability >= 8.0")

    if head_size <= 64:
        return 128
    elif head_size <= 160:
        if cc_minor != 0:
            return 64
        else:
            return 128
    elif head_size == 192:
        return 128
    elif head_size == 224:
        if cc_minor != 0:
            return 64
        else:
            return 128
    else:
        if cc_major == 8 and cc_minor == 0:
            return 128
        else:
            return 64


def get_kv_block_size(head_size: int) -> int:
    """
    Return preferred granulatity for blocked KV-cache implementation.
    """
    cc_major, cc_minor = torch.cuda.get_device_capability(get_accelerator().current_device())  #ignore-cuda

    if cc_major < 8:
        raise RuntimeError("Blocked attention requires CUDA compute capability >= 8.0")

    if head_size <= 64:
        return 128
    elif head_size != 160 or cc_minor != 0:
        return 64
    else:
        return 32


class BlockedFlashAttn(DSKernelBase):
    """
    Modified implementation of flash-attn-2 tuned for inference on blocked KV-cache and wider
    range of input sequence lengths.
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]

    def __init__(self, head_size: int, dtype: DtypeEnum) -> None:
        """
        Triggers any compilation of the kernels.
        """
        if not isinstance(dtype, DtypeEnum):
            dtype = DtypeEnum(dtype)

        if dtype not in BlockedFlashAttn.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported data types are {}".format(
                dtype, BlockedFlashAttn.supported_dtypes))

        # For testing, need to revert to 32
        if head_size % 16 != 0:
            raise ValueError("Head size must be divisible by 32 (configured with {})".format(head_size))

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.flash_attn_by_atoms

    def __call__(self, out: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, atoms: torch.Tensor,
                 softmax_scale: float) -> torch.Tensor:
        """
        Flash attention implementation atop a blocked KV-cache. Atoms should be pre-populated.
        See attention_atom.h for further details on the structure of the information.

        Arguments:
            out (torch.Tensor): Output tensor of shape [tokens, hidden_size]
            q (torch.Tensor): Query tensor of shape [tokens, hidden_size]
            k (torch.Tensor): Key cache tensor of shape [n_blocks, block_size, n_heads_kv, head_size]. This Tensor only needs to be contiguous on the final dimension.
            v (torch.Tensor): Value cache tensor of shape [n_blocks, block_size, n_heads_kv, head_size]. This Tensor only needs to be contiguous on the final dimension.
            atoms (torch.Tensor): Atom information tensor of shape [num_atoms, 8] and type int32.
                Not all data is readable in this format. See attention_atom.h for further details.
            softmax_scale (float): Softmax scale factor.

        Returns:
            out (torch.Tensor): Output tensor of shape [tokens, hidden_size]
        """
        self.kernel(out, q, k, v, atoms, softmax_scale, True)
        return out
