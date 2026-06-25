# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Integration tests for AutoEP (multi-GPU, requires distributed backend)."""

import pytest
import torch
import deepspeed
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    make_autoep_integration_config as _make_autoep_config,
    run_training_steps as _run_training_steps,
    seed_everything as _seed_everything,
)
from unit.common import DistributedTest

# ---------------------------------------------------------------------------
# Test class: AutoEP integration (world_size=2)
# ---------------------------------------------------------------------------


class TestAutoEPOnly(DistributedTest):
    world_size = 2

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
