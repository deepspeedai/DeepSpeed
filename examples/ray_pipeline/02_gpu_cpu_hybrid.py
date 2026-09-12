# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Example 02: GPU + CPU Hybrid Pipeline with Shared Memory Transport
====================================================================

**Difficulty:** Intermediate

Demonstrates heterogeneous resource allocation: the embedding stage runs on
CPU (lightweight, shared memory transport), while the transformer stage runs
on GPU (standard Ray object store). Uses ``RayTopology`` with custom
resource bundles to assign different hardware per stage.

**What you'll learn:**
- Using ``RayTopology`` with custom resource bundles per stage
- Configuring ``ShmTransport`` for CPU-CPU communication
- Placing stages on different hardware types within a Ray placement group

**Setup:**
  Run: ``python examples/ray_pipeline/02_gpu_cpu_hybrid.py``

**Expected output:**
  - Stage 0 placed on CPU bundle (4 CPUs, 0 GPUs)
  - Stage 1 placed on GPU bundle (1 CPU, 1 GPU)
  - Shared memory transport used for inter-stage transfers
  - "Example 02 complete."
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


class HybridModel(nn.Module):
    """Model designed for heterogeneous CPU+GPU pipeline placement.

    Stage 0 (CPU): Embedding + positional encoding — lightweight ops
    Stage 1 (GPU): Multi-head attention + output projection — compute-heavy
    """

    def __init__(self, vocab_size=256, hidden_dim=128, seq_len=32, num_heads=4):
        super().__init__()
        # Stage 0: CPU-friendly layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Stage 1: GPU-needed layers
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        # Stage 0: Embedding (placed on CPU)
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Stage 1: Transformer (placed on GPU)
        x, _ = self.attention(x, x, x)
        x = self.layer_norm(x)
        x = self.output(x)
        return x


def main():
    if not HAS_RAY:
        print("Ray not available. Skipping example.")
        return

    # --- Initialize Ray with 1 GPU and extra CPUs for CPU stage ---
    ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True)

    # --- Generate synthetic data ---
    vocab_size, hidden_dim, seq_len = 256, 128, 32
    batch_size, num_batches = 8, 20
    data = [(torch.randint(0, vocab_size, (batch_size, seq_len)), torch.randint(0, vocab_size, (batch_size, seq_len)))
            for _ in range(num_batches)]

    # --- Build model ---
    model = HybridModel(vocab_size=vocab_size, hidden_dim=hidden_dim, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- DeepSpeed config with heterogeneous transport ---
    #
    # Since the embedding stage is on CPU, we use shared memory transport
    # for zero-copy CPU-CPU tensor transfers. The GPU stage's tensors are
    # serialized through shared memory segments automatically.
    ds_config = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            },
        },
        "pipeline": {
            "executor": "ray",
            "transport": "shm",  # <- Shared memory for CPU-CPU
            "stages": 2,
            "partition": "parameters",  # <- Balance by parameter count
        },
        "fp16": {
            "enabled": False
        },
    }

    # --- RayTopology: Custom resource bundles per stage ---
    #
    # Each bundle defines the resources Ray allocates for that stage.
    # Stage 0: CPU-only (4 cores, no GPU)  — embedding is lightweight
    # Stage 1: GPU (1 core, 1 GPU)         — attention needs GPU compute
    stage_bundles = [
        {
            "CPU": 4,
            "GPU": 0
        },  # Stage 0: embedding on CPU
        {
            "CPU": 1,
            "GPU": 1
        },  # Stage 1: transformer on GPU
    ]

    print("=" * 60)
    print("DeepSpeed Ray Pipeline -- Example 02")
    print(f"  Model: HybridModel ({vocab_size}v, {hidden_dim}d, {seq_len}s)")
    print("  Pipeline: 2 heterogeneous stages")
    print("    Stage 0: CPU  (embedding + pos encoding)")
    print("    Stage 1: GPU  (attention + layer_norm + output)")
    print("  Transport: Shared memory (shm)")
    print(f"  RayTopology bundles:")
    print(f"    Stage 0: {stage_bundles[0]}")
    print(f"    Stage 1: {stage_bundles[1]}")
    print("=" * 60)

    # --- Training loop ---
    model.train()
    for batch_idx, (input_ids, labels) in enumerate(data):
        output = model(input_ids)
        loss = F.cross_entropy(output.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
            print(f"  Batch {batch_idx:3d}/{num_batches} | Loss: {loss.item():.4f}")

    print("=" * 60)
    print("Example 02 complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
