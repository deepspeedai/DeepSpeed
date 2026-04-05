# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Integration tests for AutoEP (multi-GPU, requires distributed backend)."""

import pytest
import torch
import torch.nn as nn
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest

# ---------------------------------------------------------------------------
# Mock model fixtures
# ---------------------------------------------------------------------------


class MockHFConfig:
    model_type = "mixtral"
    num_local_experts = 4
    num_experts_per_tok = 2
    hidden_size = 64
    intermediate_size = 128


class MockMoEExperts(nn.Module):
    """Mimics HF transformers 5.0.0+ fused expert storage for Mixtral."""

    def __init__(self):
        super().__init__()
        # gate_up_proj shape: [num_experts, 2 * ffn_hidden, hidden_size]
        self.gate_up_proj = nn.Parameter(torch.randn(4, 256, 64))
        # down_proj shape: [num_experts, hidden_size, ffn_hidden]
        self.down_proj = nn.Parameter(torch.randn(4, 64, 128))


class MockMoEBlock(nn.Module):
    """Mimics model.layers.N.mlp for a Mixtral-like model."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(64, 4, bias=False)
        self.experts = MockMoEExperts()


class MockMoETransformer(nn.Module):
    """Synthetic 2-layer MoE transformer for integration testing.

    Uses small dimensions (hidden=64, ffn=128, 4 experts, top-2)
    to keep memory and compute requirements minimal.
    """

    def __init__(self):
        super().__init__()
        self.config = MockHFConfig()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([self._make_layer() for _ in range(2)])
        self.lm_head = nn.Linear(64, 100)

    def _make_layer(self):
        layer = nn.Module()
        layer.self_attn = nn.MultiheadAttention(64, 1, batch_first=True)
        layer.mlp = MockMoEBlock()
        layer.input_layernorm = nn.LayerNorm(64)
        layer.post_attention_layernorm = nn.LayerNorm(64)
        return layer

    def forward(self, x):
        """Forward pass.

        Args:
            x: [B, S, H] input tensor.

        Returns:
            logits: [B, S, V] where V=100.
        """
        for layer_module in self.model.layers:
            residual = x
            x = layer_module.input_layernorm(x)
            x, _ = layer_module.self_attn(x, x, x)
            x = residual + x
            residual = x
            x = layer_module.post_attention_layernorm(x)
            x = layer_module.mlp(x)  # Replaced by AutoEPMoELayer during init
            x = residual + x
        logits = self.lm_head(x)
        return logits


def _make_autoep_config(zero_stage=0, ep_size=2):
    """Build a DeepSpeed JSON config dict for AutoEP integration tests."""
    return {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
            },
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": ep_size,
            "preset_model": "mixtral",
        },
        "zero_optimization": {
            "stage": zero_stage,
        },
    }


def _seed_everything(seed=1234):
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(seed)
    get_accelerator().manual_seed_all(seed)


def _run_training_steps(engine, num_steps=3, seq_len=8, hidden_dim=64):
    """Run forward + backward + step for the given number of iterations.

    Returns:
        losses: list of scalar loss values (one per step).
        grad_norms: list of total gradient norms (one per step, measured after backward before step).
    """
    losses = []
    grad_norms = []
    for _ in range(num_steps):
        x = torch.randn(1, seq_len, hidden_dim, device=engine.device)
        logits = engine(x)
        # Simple loss: mean of logits
        loss = logits.mean()
        engine.backward(loss)

        # Compute total grad norm BEFORE step (step zeros gradients)
        total_norm = 0.0
        for p in engine.module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.float().norm(2).item()**2
        total_norm = total_norm**0.5
        grad_norms.append(total_norm)

        engine.step()
        losses.append(loss.item())

    return losses, grad_norms


# ---------------------------------------------------------------------------
# Test class: EP-only (world_size=2)
# ---------------------------------------------------------------------------


class TestAutoEPOnly(DistributedTest):
    world_size = 2

    def test_ep_only_2gpu(self):
        """Basic EP training with ep_size=2, ZeRO-0.

        Verifies:
        - deepspeed.initialize succeeds with AutoEP config
        - MoE layers are replaced with AutoEPMoELayer
        - 3 training steps produce finite losses
        - Gradient norms are positive (gradients flow through the model)
        """
        _seed_everything(1234)

        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=0, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        # Verify AutoEPMoELayer replacement occurred
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        replaced_count = 0
        for _, module in engine.module.named_modules():
            if isinstance(module, AutoEPMoELayer):
                replaced_count += 1
        assert replaced_count == 2, (f"Expected 2 MoE layers replaced, found {replaced_count}")

        # Run training steps
        losses, grad_norms = _run_training_steps(engine, num_steps=3)

        # All losses must be finite
        for i, loss_val in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss_val)), (f"Loss at step {i} is not finite: {loss_val}")

        # At least one step must have non-zero gradients
        assert any(gn > 0 for gn in grad_norms), (f"All gradient norms are zero: {grad_norms}")

    def test_zero2_ep_2gpu(self):
        """EP with ZeRO-2 training.

        Verifies EP and ZeRO Stage 2 work together: finite losses
        and parameters actually update across training steps.
        Note: ZeRO-2 partitions gradients, so p.grad may be None on some ranks.
        """
        _seed_everything(1234)

        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=2, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        # Verify replacement
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        replaced_count = sum(1 for _, m in engine.module.named_modules() if isinstance(m, AutoEPMoELayer))
        assert replaced_count == 2, (f"Expected 2 MoE layers replaced with ZeRO-2, found {replaced_count}")

        # Snapshot parameter values before training
        params_before = {n: p.data.clone().float() for n, p in engine.module.named_parameters() if p.requires_grad}

        # Run training steps (ignore grad norms since ZeRO-2 partitions them)
        losses, _ = _run_training_steps(engine, num_steps=3)

        for i, loss_val in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss_val)), (f"Loss at step {i} is not finite: {loss_val}")

        # Verify at least some parameters changed (optimizer step took effect)
        params_changed = 0
        for n, p in engine.module.named_parameters():
            if n in params_before and not torch.equal(p.data.float(), params_before[n]):
                params_changed += 1
        assert params_changed > 0, "No parameters changed after 3 training steps with ZeRO-2"

    def test_zero3_ep_rejected_2gpu(self):
        """EP with ZeRO-3 should trigger an assertion error.

        ZeRO Stage 3 is incompatible with MoE. The engine should raise
        an AssertionError with the message 'MoE not supported with Stage 3'.
        """
        _seed_everything(1234)

        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=3, ep_size=2)

        with pytest.raises(AssertionError, match="MoE not supported with Stage 3"):
            deepspeed.initialize(model=model, config=config)
