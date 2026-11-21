# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
import weakref

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.runtime.zero.parameter_offload as po_mod
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from unit.common import DistributedTest, preferred_dtype


class _TwoLayerModel(torch.nn.Module):

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _DynamicReuseModel(torch.nn.Module):

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.a = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.c = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reuse_a = True
        self.a_param_holder = []
        self.a_status_at_b = []

    def forward(self, x):
        x = self.a(x)
        self.a_status_at_b.append(self.a_param_holder[0].ds_status)
        x = self.b(x)
        if self.reuse_a:
            return self.a(x)
        return self.c(x)


class _RecordStatusAfterOuterBackward(torch.autograd.Function):
    """Record a parameter's ZeRO status right after the downstream module's backward completes."""

    @staticmethod
    def forward(ctx, x, param_holder, observations):
        ctx.param_holder = param_holder
        ctx.observations = observations
        return x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.observations.append(ctx.param_holder[0].ds_status)
        return grad_output, None, None


class _SharedLinear(torch.nn.Module):

    def __init__(self, hidden_dim, shared_weight):
        super().__init__()
        self.shared_weight = shared_weight
        self.own_weight = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

    def forward(self, x):
        return F.linear(F.linear(x, self.own_weight), self.shared_weight)


class _CheckpointedOuter(torch.nn.Module):

    def __init__(self, hidden_dim, shared_weight):
        super().__init__()
        self.shared_weight = shared_weight
        self.outer_weight = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.inner = _SharedLinear(hidden_dim, shared_weight)

    def _ckpt_body(self, x):
        return torch.tanh(self.inner(x))

    def forward(self, x):
        x = checkpoint(self._ckpt_body, x, use_reentrant=False)
        return F.linear(x, self.outer_weight)


class _CheckpointRecomputeModel(torch.nn.Module):

    def __init__(self, hidden_dim=8):
        super().__init__()
        shared_weight = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1, requires_grad=False)
        self.early = _SharedLinear(hidden_dim, shared_weight)
        self.outer = _CheckpointedOuter(hidden_dim, shared_weight)
        self.shared_holder = []
        self.status_after_outer_bwd = []

    def forward(self, x):
        x = self.early(x)
        x = _RecordStatusAfterOuterBackward.apply(x, self.shared_holder, self.status_after_outer_bwd)
        return self.outer(x)


def _zero3_config(*, max_reuse_distance=10**9, use_preferred_dtype=False):
    dtype = preferred_dtype() if use_preferred_dtype else torch.float32
    return {
        "train_batch_size": 1,
        "fp16": {
            "enabled": dtype is torch.float16
        },
        "bf16": {
            "enabled": dtype is torch.bfloat16
        },
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
            "stage3_prefetch_bucket_size": 0,
            "stage3_max_reuse_distance": max_reuse_distance,
        },
    }


class TestZero3CoordinatorCacheLifecycle(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    def test_release_sub_module_follows_retraced_order(self):
        hidden = 8
        numel = hidden * hidden
        model = _DynamicReuseModel(hidden)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Allow reuse across one intermediate module (`b`), but not two (`b` + `c`).
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=_zero3_config(max_reuse_distance=numel + 1),
        )
        coordinator = engine.optimizer.parameter_offload.get_param_coordinator()
        model.a_param_holder.append(engine.module.a.weight)

        try:
            # First step records `[root, a, b, a, ...]`; second step runs under COMPLETE trace.
            for _ in range(2):
                x = torch.randn(2, hidden, device=engine.device, requires_grad=True)
                engine.backward(engine(x).sum())
                assert x.grad is not None and torch.isfinite(x.grad).all()
                engine.step()

            assert coordinator.is_complete_trace()
            assert model.a_status_at_b[-1] == ZeroParamStatus.AVAILABLE

            # Switch control flow to `a -> b -> c`: step 0 invalidates the trace via trace_prologue(c),
            # step 1 records the new trace, and step 2 runs under the new COMPLETE trace where `a` is
            # no longer reused within `stage3_max_reuse_distance`.
            model.reuse_a = False
            for step_idx in range(3):
                x = torch.randn(2, hidden, device=engine.device, requires_grad=True)
                loss = engine(x).sum()
                if step_idx == 0:
                    assert coordinator.is_invalid_trace()
                engine.backward(loss)
                assert x.grad is not None and torch.isfinite(x.grad).all()
                engine.step()

            assert coordinator.is_complete_trace()
            assert model.a_status_at_b[-1] == ZeroParamStatus.NOT_AVAILABLE

            # Switch back to `a -> b -> a` and verify retracing restores reuse retention.
            model.reuse_a = True
            for _ in range(3):
                x = torch.randn(2, hidden, device=engine.device, requires_grad=True)
                engine.backward(engine(x).sum())
                assert x.grad is not None and torch.isfinite(x.grad).all()
                engine.step()

            assert coordinator.is_complete_trace()
            assert model.a_status_at_b[-1] == ZeroParamStatus.AVAILABLE
        finally:
            engine.destroy()
            po_mod.FWD_MODULE_STACK.clear()

    def test_backward_recompute_release_does_not_mutate_cached_forward_release(self):
        hidden = 8
        numel = hidden * hidden
        model = _CheckpointRecomputeModel(hidden)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=_zero3_config(max_reuse_distance=4 * numel),
        )
        coordinator = engine.optimizer.parameter_offload.get_param_coordinator()
        model.shared_holder.append(engine.module.outer.shared_weight)

        try:
            # Step 0 records the trace and Step 1 runs under COMPLETE trace while `shared_weight` is frozen.
            # During `outer` backward, checkpoint recompute of `inner` attributes `shared_weight` to
            # `outer.ds_recompute_parameters`, so `outer` releases `shared_weight` at the backward boundary.
            for _ in range(2):
                x = torch.randn(2, hidden, device=engine.device, requires_grad=True)
                engine.backward(engine(x).sum())
                assert x.grad is not None and torch.isfinite(x.grad).all()
                engine.step()

            assert coordinator.is_complete_trace()
            assert model.status_after_outer_bwd[-1] == ZeroParamStatus.NOT_AVAILABLE

            # Unfreeze `shared_weight` so checkpoint recompute in Step 2 no longer populates
            # `outer.ds_recompute_parameters`, while keeping the same submodule trace. Because `early`
            # uses `shared_weight` next in backward within `stage3_max_reuse_distance`, `shared_weight`
            # must remain AVAILABLE after `outer` backward instead of inheriting a mutated cached release set.
            model.shared_holder[0].requires_grad_(True)
            x = torch.randn(2, hidden, device=engine.device, requires_grad=True)
            engine.backward(engine(x).sum())
            assert x.grad is not None and torch.isfinite(x.grad).all()
            engine.step()

            assert coordinator.is_complete_trace()
            assert model.status_after_outer_bwd[-1] == ZeroParamStatus.AVAILABLE
        finally:
            engine.destroy()
            po_mod.FWD_MODULE_STACK.clear()

    def test_engine_coordinator_released_after_complete_trace(self):
        hidden = 16
        dtype = preferred_dtype()
        model = _TwoLayerModel(hidden)
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=_zero3_config(use_preferred_dtype=True),
            model_parameters=list(model.parameters()),
        )

        x = torch.randn(2, hidden, dtype=dtype, device=engine.device)
        # First forward enters RECORD and records the trace; second forward enters COMPLETE
        # and populates the coordinator's release cache.
        engine(x)
        engine(x)

        coordinator = engine.optimizer.get_param_coordinator()
        assert coordinator.is_complete_trace()
        coordinator_ref = weakref.ref(coordinator)
        submodule_ref = weakref.ref(engine.module.fc1)

        engine.destroy()
        po_mod.FWD_MODULE_STACK.clear()
        del coordinator, engine, optimizer, model, x
        gc.collect()

        assert coordinator_ref() is None
        assert submodule_ref() is None
