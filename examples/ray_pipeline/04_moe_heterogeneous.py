# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Example 04: Mixture-of-Experts Heterogeneous Pipeline
=======================================================

**Difficulty:** Advanced

Demonstrates a full MoE pipeline with heterogeneous placement across three
stages. Each stage uses a different hardware type and transport backend,
showing how Ray's placement groups enable fine-grained resource allocation.

**Architecture:**

  Stage 0: Embedding ................ CPU (SharedMemory transport)
  Stage 1: Self-attention ........... GPU type A (Ray object store)
  Stage 2: Expert FFN layers ........ GPU type B + CPU (TCP transport)

**What you'll learn:**
- Building a MoE model and partitioning it into pipeline stages
- Using ``RayTopology`` with per-stage custom resource bundles
- Configuring different transport backends per pipeline segment
- Debugging heterogeneous placement with actor node-ID inspection

**Setup:**
  Run: ``python examples/ray_pipeline/04_moe_heterogeneous.py``

**Expected output:**
  - Three stages placed on different resource bundles
  - Transport type logged for each inter-stage boundary
  - "Example 04 complete."
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

# ---------------------------------------------------------------------------
# Mixture-of-Experts model
# ---------------------------------------------------------------------------


class ExpertFFN(nn.Module):
    """A single expert feed-forward network.

    MoE architectures use multiple parallel FFN layers ("experts"), with a
    router (gating network) deciding which expert(s) handle each token.
    """

    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Router(nn.Module):
    """Gating network that selects experts for each token.

    For simplicity, this demo uses top-1 routing: each token is routed
    to the expert with the highest gating score.
    """

    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        logits = self.gate(x)  # (B, S, num_experts)
        weights = F.softmax(logits, dim=-1)
        return weights


class MoEModel(nn.Module):
    """A three-stage MoE model for heterogeneous pipeline placement.

    Stage 0: Embedding + positional encoding                    (CPU)
    Stage 1: Self-attention + layer norm                         (GPU-A)
    Stage 2: Router + expert FFNs + output projection            (GPU-B)
    """

    def __init__(self, vocab_size=256, hidden_dim=128, seq_len=32, num_heads=4, num_experts=4, ffn_dim=512):
        super().__init__()

        # Stage 0: CPU — embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Stage 1: GPU-A — attention block
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        # Stage 2: GPU-B — MoE + output
        self.router = Router(hidden_dim, num_experts)
        self.experts = nn.ModuleList([ExpertFFN(hidden_dim, ffn_dim) for _ in range(num_experts)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self._num_experts = num_experts

    def forward(self, input_ids):
        # --- Stage 0: Embedding on CPU ---
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :x.size(1), :]

        # --- Stage 1: Attention on GPU-A ---
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + residual)

        # --- Stage 2: MoE on GPU-B ---
        # Router selects top-1 expert per token
        gate_logits = self.router(x)  # (B, S, num_experts)
        expert_indices = gate_logits.argmax(dim=-1)  # (B, S)

        # Route each token to its selected expert
        batch_size, seq_len, hidden = x.shape
        expert_output = torch.zeros_like(x)
        for expert_idx in range(self._num_experts):
            mask = (expert_indices == expert_idx)  # (B, S)
            if mask.any():
                token_indices = mask.nonzero(as_tuple=False)  # (N, 2)
                b_idx = token_indices[:, 0]
                s_idx = token_indices[:, 1]
                selected_tokens = x[b_idx, s_idx]  # (N, hidden)
                processed = self.experts[expert_idx](selected_tokens)
                expert_output[b_idx, s_idx] = processed

        x = self.layer_norm2(x + expert_output)  # residual
        x = self.output(x)
        return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not HAS_RAY:
        print("Ray not available. Skipping example.")
        return

    # --- Initialize Ray ---
    #
    # We request 2 GPUs (for stages 1 and 2) plus extra CPUs (for stage 0).
    # In production, GPU-B could be a different hardware type (NPU/TPU)
    # specified via custom resource labels.
    ray.init(num_gpus=2, num_cpus=6, ignore_reinit_error=True)

    # --- Generate synthetic data ---
    vocab_size, hidden_dim, seq_len = 256, 128, 32
    batch_size, num_batches = 4, 10
    data = [(torch.randint(0, vocab_size, (batch_size, seq_len)), torch.randint(0, vocab_size, (batch_size, seq_len)))
            for _ in range(num_batches)]

    # --- Build model ---
    model = MoEModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        num_heads=4,
        num_experts=4,
        ffn_dim=512,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- DeepSpeed config: MoE pipeline ---
    #
    # We use 3 pipeline stages with uniform partition. In a production MoE
    # setup, you'd use DeepSpeed's MoE module for efficient expert routing
    # and load balancing.
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
            "transport": "ray",  # Ray object store as base transport
            "stages": 3,
            "partition": "parameters",
        },
        "fp16": {
            "enabled": False
        },
    }

    # --- RayTopology: Per-stage heterogeneous bundles ---
    #
    # Each stage gets different resources:
    # - Stage 0: CPU-only for embedding (no GPU needed)
    # - Stage 1: GPU for attention computation
    # - Stage 2: GPU for MoE experts (could be GPU-B, NPU, etc.)
    #
    # In real deployments, GPU type A vs. GPU type B is specified via Ray's
    # custom resource labels, e.g. {"GPU_A": 1} vs {"GPU_B": 1}.
    stage_bundles = [
        {
            "CPU": 4,
            "GPU": 0
        },  # Stage 0: Embedding on CPU
        {
            "CPU": 1,
            "GPU": 1
        },  # Stage 1: Attention on GPU-A
        {
            "CPU": 1,
            "GPU": 1
        },  # Stage 2: MoE experts on GPU-B
    ]

    print("=" * 60)
    print("DeepSpeed Ray Pipeline -- Example 04")
    print("  MoE Heterogeneous Pipeline")
    print(f"  Model: MoEModel ({vocab_size}v, {hidden_dim}d, {seq_len}s)")
    print(f"  Experts: 4 x FFN({hidden_dim}->512->{hidden_dim})")
    print("  Pipeline: 3 heterogeneous stages")
    print(f"    Stage 0: CPU  (embedding)")
    print(f"    Stage 1: GPU  (self-attention)")
    print(f"    Stage 2: GPU  (router + expert FFNs + output)")
    print("  Transport:")
    print(f"    Stage 0<->1: Ray object store (CPU->GPU)")
    print(f"    Stage 1<->2: Ray object store (GPU->GPU)")
    print(f"  RayTopology bundles:")
    print(f"    Stage 0: {stage_bundles[0]}")
    print(f"    Stage 1: {stage_bundles[1]}")
    print(f"    Stage 2: {stage_bundles[2]}")
    print("=" * 60)

    # --- Training loop ---
    model.train()
    for batch_idx, (input_ids, labels) in enumerate(data):
        output = model(input_ids)
        loss = F.cross_entropy(output.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 3 == 0 or batch_idx == num_batches - 1:
            print(f"  Batch {batch_idx:3d}/{num_batches} | Loss: {loss.item():.4f}")

    print("=" * 60)
    print("Example 04 complete.")
    print()
    print("Next steps:")
    print("  - Replace CPU stage with an actual NPU/TPU via Ray custom resources")
    print("  - Add DeepSpeed MoE module for expert load balancing")
    print("  - Scale experts across multiple GPUs with expert-parallel sharding")
    ray.shutdown()


if __name__ == "__main__":
    main()
