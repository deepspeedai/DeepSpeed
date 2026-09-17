# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Example 03: Multi-Accelerator Pipeline with TCP Transport
==========================================================

**Difficulty:** Intermediate

Demonstrates cross-accelerator communication using TCP transport. Stage 0
runs on GPU, Stage 1 runs on a simulated NPU (CPU for development). TCP
sockets bridge the two hardware types, serializing tensors into
length-prefixed byte streams.

**What you'll learn:**
- Configuring ``TcpTransport`` with send/recv port assignment
- Running stages on different accelerator types via custom RayTopology bundles
- Cross-platform tensor serialization via TCP

**Setup:**
  Run: ``python examples/ray_pipeline/03_multi_accelerator.py``

**Expected output:**
  - Stage 0 uses GPU, Stage 1 uses CPU (simulated NPU)
  - TcpTransport ports configured and logged
  - "Example 03 complete."
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


class CrossPlatformModel(nn.Module):
    """Model designed for cross-accelerator pipeline.

    Stage 0 (GPU): Heavy attention computation
    Stage 1 (NPU, simulated as CPU): Output projection
    """

    def __init__(self, vocab_size=512, hidden_dim=256, seq_len=64, num_heads=8):
        super().__init__()
        # Stage 0: GPU layers — compute-bound
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)

        # Stage 1: NPU/CPU layers — lighter projection
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        # Stage 0: GPU
        x = self.embedding(input_ids)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)

        # Stage 1: NPU (simulated as CPU)
        x = self.layer_norm(x)
        x = self.output(x)
        return x


def main():
    if not HAS_RAY:
        print("Ray not available. Skipping example.")
        return

    # --- Initialize Ray with 1 GPU and extra CPUs for simulated NPU ---
    ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True)

    # --- Generate synthetic data ---
    vocab_size, hidden_dim, seq_len = 512, 256, 64
    batch_size, num_batches = 4, 15
    data = [(torch.randint(0, vocab_size, (batch_size, seq_len)), torch.randint(0, vocab_size, (batch_size, seq_len)))
            for _ in range(num_batches)]

    # --- Build model ---
    model = CrossPlatformModel(vocab_size=vocab_size, hidden_dim=hidden_dim, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- TCP port configuration ---
    #
    # TcpTransport uses separate send/recv ports per stage:
    # - send_port: port used when sending tensors downstream
    # - recv_port: port bound for receiving tensors from upstream
    tcp_send_port = 20000
    tcp_recv_port = 20001

    # --- DeepSpeed config with TCP transport ---
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
            "transport": "tcp",  # <- TCP for cross-accelerator
            "tcp_send_port": tcp_send_port,  # Stage sends via this port
            "tcp_recv_port": tcp_recv_port,  # Stage receives via this port
            "tcp_host": "127.0.0.1",
            "stages": 2,
            "partition": "parameters",
        },
        "fp16": {
            "enabled": False
        },
    }

    # --- RayTopology: Different hardware per stage ---
    #
    # Stage 0 needs GPU for attention computation
    # Stage 1 simulates an NPU with CPU-only placement (real NPU would
    # use custom resources like {"NPU": 1} or {"TPU": 1})
    stage_bundles = [
        {
            "CPU": 1,
            "GPU": 1
        },  # Stage 0: GPU for attention
        {
            "CPU": 4,
            "GPU": 0
        },  # Stage 1: CPU (simulated NPU)
    ]

    print("=" * 60)
    print("DeepSpeed Ray Pipeline -- Example 03")
    print(f"  Model: CrossPlatformModel ({vocab_size}v, {hidden_dim}d, {seq_len}s)")
    print("  Pipeline: 2 stages on different accelerators")
    print("    Stage 0: GPU  -- attention computation")
    print("    Stage 1: CPU  -- output projection (simulated NPU)")
    print(f"  Transport: TCP")
    print(f"    send_port: {tcp_send_port}")
    print(f"    recv_port: {tcp_recv_port}")
    print(f"    host: 127.0.0.1")
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
    print("Example 03 complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
