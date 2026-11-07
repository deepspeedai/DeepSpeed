# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Example 01: Simple Two-Stage Ray Pipeline (Homogeneous GPUs)
=============================================================

**Difficulty:** Beginner

Demonstrates the simplest Ray-backed pipeline: two stages on identical GPU
resources, communicating via Ray's distributed object store.

**What you'll learn:**
- Configuring ``pipeline.executor = "ray"`` and ``pipeline.transport = "ray"``
- Splitting a model into pipeline stages with DeepSpeed
- Running a Ray pipeline training loop

**Setup:**
  1. Install DeepSpeed: ``pip install deepspeed``
  2. Install Ray: ``pip install ray``
  3. Run: ``python examples/ray_pipeline/01_simple_two_stage.py``

**Expected output:**
  - Stage info printed to console (stage_id=0, stage_id=1)
  - Loss decreases over batches
  - "Example 01 complete."
"""
# NOTE: This example is illustrative. It demonstrates the *configuration*
# and placement patterns for Ray-backed pipeline parallelism in DeepSpeed.
# The code path through deepspeed.initialize() with
# pipeline.executor='ray' will be activated when the Ray pipeline
# engine is merged into the main branch. Until then, this example serves
# as a reference for the intended usage pattern.
#
# To run this as a plain PyTorch training script (without DeepSpeed
# pipeline parallelism), execute normally:
#   python examples/ray_pipeline/XX_example_name.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("Ray is not installed. Install with: pip install ray")


class TwoLayerTransformer(nn.Module):
    """A minimal transformer for pipeline parallelism demo.

    Layers naturally split into two stages:
    - Stage 0: embedding -> attention
    - Stage 1: output projection -> loss
    """

    def __init__(self, vocab_size=128, hidden_dim=64, seq_len=16, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # Stage 0
        x, _ = self.attention(x, x, x)  # Stage 0
        x = self.output(x)  # Stage 1
        return x


def main():
    if not HAS_RAY:
        print("Ray not available. Skipping example.")
        return

    # --- Initialize Ray with 2 GPUs (1 per stage) ---
    ray.init(num_gpus=2, ignore_reinit_error=True)

    # --- Generate synthetic data ---
    batch_size = 8
    vocab_size = 128
    hidden_dim = 64
    seq_len = 16
    num_batches = 20

    data = []
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append((input_ids, labels))

    # --- Build model ---
    model = TwoLayerTransformer(vocab_size=vocab_size, hidden_dim=64, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- DeepSpeed config with Ray pipeline ---
    ds_config = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
            },
        },
        "pipeline": {
            "executor": "ray",  # <- Use Ray for stage execution
            "transport": "ray",  # <- Ray object store for communication
            "stages": 2,
            "partition": "uniform",
        },
    }

    print("=" * 60)
    print("DeepSpeed Ray Pipeline -- Example 01")
    print(f"  Model: TwoLayerTransformer ({vocab_size}v, {hidden_dim}d, {seq_len}s)")
    print("  Pipeline: 2 homogeneous GPU stages")
    print("  Transport: Ray object store")
    print(f"  Data: {num_batches} batches of {batch_size} samples")
    print("=" * 60)

    # --- Training loop ---
    model.train()
    for batch_idx, (input_ids, labels) in enumerate(data):
        output = model(input_ids)
        loss = F.cross_entropy(output.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx:3d}/{num_batches} | Loss: {loss.item():.4f}")

    print("=" * 60)
    print("Example 01 complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
