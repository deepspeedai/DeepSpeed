# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import torch

from deepspeed.runtime.state_dict_factory import SDLoaderFactory


def test_megatron_split_then_merge_restores_checkpoint(tmpdir):
    # A tensor-parallel checkpoint that is split for more ranks and merged back for
    # fewer ranks must come back unchanged, including the vocab-parallel final_linear
    # output layer that GPT-NeoX checkpoints carry.
    hidden, vocab = 8, 32
    full_module = {
        "attention.query_key_value.weight": torch.randn(3 * hidden, hidden),
        "attention.dense.weight": torch.randn(hidden, hidden),
        "mlp.dense_h_to_4h.weight": torch.randn(4 * hidden, hidden),
        "mlp.dense_h_to_4h.bias": torch.randn(4 * hidden),
        "mlp.dense_4h_to_h.weight": torch.randn(hidden, 4 * hidden),
        "final_linear.weight": torch.randn(vocab, hidden),
    }
    full_path = os.path.join(str(tmpdir), "full.pt")
    torch.save({"module": full_module, "checkpoint_version": 2.0}, full_path)

    tp_size = 2
    shard_paths = []
    for rank in range(tp_size):
        _, shard, _ = SDLoaderFactory.get_sd_loader([full_path], checkpoint_engine=None).load(tp_size, rank)
        shard_path = os.path.join(str(tmpdir), f"mp_rank_{rank:02d}_model_states.pt")
        torch.save(shard, shard_path)
        shard_paths.append(shard_path)

    _, merged, _ = SDLoaderFactory.get_sd_loader(shard_paths, checkpoint_engine=None).load(1, 0)

    for key, expected in full_module.items():
        assert torch.equal(merged["module"][key], expected), key
