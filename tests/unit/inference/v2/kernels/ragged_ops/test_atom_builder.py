# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.inference.v2.kernels.ragged_ops import AtomBuilder
from .ragged_testing_utils import build_complex_batch

Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 128


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('seq_params', [(1, 0, 0), (1, 228, 0), (383, 0, 0), (1, 494, 0)])
def test_single_sequence(seq_params) -> None:
    seq_len, n_seen_tokens, _ = seq_params

    batch, _ = build_complex_batch([seq_params], kv_block_size=KV_BLOCK_SIZE, padding=False)
    atom_builder = AtomBuilder()

    atoms = torch.empty((8, 8), dtype=torch.int32, device=torch.device("cpu"))
    atoms, n_atoms = atom_builder(atoms, batch, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    calc_n_atoms = (seq_len + 127) // 128

    assert n_atoms == calc_n_atoms

    for i, atom in enumerate(atoms[:n_atoms]):
        # Since the ptr was 0, first 2 elements should be 0
        assert atom[0] == 0
        assert atom[1] == 0

        # Since we have a single sequence, the q_start_idx should always be
        # whichever atom we're on multiplied by the block size
        assert atom[2] == i * Q_BLOCK_SIZE
        assert atom[3] == min(Q_BLOCK_SIZE, seq_len - i * Q_BLOCK_SIZE)
        total_toks = i * Q_BLOCK_SIZE + min(Q_BLOCK_SIZE, seq_len - i * Q_BLOCK_SIZE)

        assert atom[4] == (total_toks + n_seen_tokens + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
        assert atom[5] == (total_toks + n_seen_tokens)

        assert atom[6] == n_seen_tokens + i * Q_BLOCK_SIZE
