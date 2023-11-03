# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import torch

from ... import DSKernelBase
from deepspeed.ops.op_builder import RaggedOpsBuilder
from ....ragged import RaggedBatchWrapper


class AtomBuilder(DSKernelBase):
    """
    C++ implementation to populate the attention atoms for the blocked attention
    kernel.
    """

    def __init__(self) -> None:
        """
        Triggers compilation of the C++ implementation.
        """
        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.build_atoms

    def __call__(self, atoms: torch.Tensor, ragged_batch: RaggedBatchWrapper, q_block_size: int,
                 kv_block_size: int) -> Tuple[torch.Tensor, int]:
        """
        Populates the attention atoms for the blocked attention kernel.

        Args:
            atoms (torch.Tensor): Pre-allocated int32 tensor of shape [max_atoms, 8]
            ragged_batch (torch.Tensor): Wrapper for the ragged batch.
            q_block_size (int): The block size for the queries (as determined by the
                attention implementation)
            kv_block_size (int): The block size for the keys/values (as determined by the
                attention implementation)

        Returns:

        """
        if atoms.device != torch.device("cpu"):
            raise RuntimeError("AtomBuilder must be called on tensors")

        n_atoms = self.kernel(atoms, ragged_batch.batch_metadata_buffer(on_device=False),
                              ragged_batch.inflight_seq_descriptors(on_device=False),
                              ragged_batch.kv_ptrs(on_device=False), q_block_size, kv_block_size)
        return atoms, n_atoms
