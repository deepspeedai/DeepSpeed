# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.linear import zero3_gathered_param, zero3_linear_module_forward, zero3_linear_wrap
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils import safe_get_full_grad

from unit.common import DistributedTest
from unit.v1.zero.test_zero_user_backward import get_config_dict, initialize_distributed

# These two frozen params are released at the recompute owner-module / engine-epilogue boundary
# rather than at each param's true last use, so they stay gathered until then. Correct and
# leak-free, but not prompt; prompt per-param recompute release is a follow-up.
_RECOMPUTE_RELEASE_TIMING = pytest.mark.xfail(
    reason="Recompute params release at the owner-module/epilogue boundary, not at each frozen "
    "param's last use, so they stay gathered until then (follow-up).",
    strict=False)


class _StatusProbe:
    """Holder for an autograd boundary probe."""

    def __init__(self, parameter, observations):
        self.parameter = parameter
        self.observations = observations


class _RecordStatusAfterConsumer(torch.autograd.Function):
    """Record the probed param's ZeRO status when this boundary's backward runs.

    Inserted before a module, so its backward fires after that module's backward and its
    DeepSpeed input wrapper -- separating release at the real consumer from outer cleanup.
    """

    @staticmethod
    def forward(ctx, value, probe):
        ctx.probe = probe
        return value

    @staticmethod
    def backward(ctx, grad_output):
        parameter = ctx.probe.parameter
        ctx.probe.observations.append({
            "status": parameter.ds_status,
            "active_sub_modules": set(parameter.ds_active_sub_modules),
        })
        return grad_output, None


class _RaiseOnceInBackward(torch.autograd.Function):
    """Abort one backward after checkpoint recompute has built real ZeRO state."""

    @staticmethod
    def forward(ctx, value, control):
        ctx.control = control
        return value

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.control["raise"]:
            ctx.control["raise"] = False
            raise RuntimeError("injected incomplete checkpoint backward")
        return grad_output, None


def _zero3_config(*, gradient_accumulation_steps=1, dtype=torch.float32, module_granularity_threshold=0):
    config = get_config_dict(3, gradient_accumulation_steps=gradient_accumulation_steps, force_fp32=True)
    # Zero reuse window + no prefetch so residency reflects the current consumer, not a future reuse.
    config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
    config["zero_optimization"]["stage3_max_reuse_distance"] = 0
    config["zero_optimization"]["stage3_module_granularity_threshold"] = module_granularity_threshold
    if dtype == torch.float16:
        config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        config["bf16"] = {"enabled": True}
    return config


def _initialize_zero3(model, *, gradient_accumulation_steps=1, dtype=torch.float32, module_granularity_threshold=0):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    engine, _, _, _ = deepspeed.initialize(config=_zero3_config(
        gradient_accumulation_steps=gradient_accumulation_steps,
        dtype=dtype,
        module_granularity_threshold=module_granularity_threshold),
                                           model=model,
                                           model_parameters=trainable_parameters)
    return engine


def _synchronize():
    get_accelerator().synchronize()
    dist.barrier()


def _assert_checkpoint_state_clean(engine, *, require_partitioned=True):
    """Assert no recompute params / active consumers remain and params are partitioned."""
    for module_name, module in engine.module.named_modules():
        recompute_parameters = getattr(module, "ds_recompute_parameters", set())
        assert not recompute_parameters, (
            f"module {module_name or '<root>'} kept recompute parameters after the lifecycle boundary: "
            f"{[parameter.ds_id for parameter in recompute_parameters]}")

    for parameter_name, parameter in engine.module.named_parameters():
        assert not parameter.ds_active_sub_modules, (
            f"parameter {parameter_name} kept active ZeRO consumers after the lifecycle boundary: "
            f"{sorted(parameter.ds_active_sub_modules)}")
        if (require_partitioned and not parameter.ds_persist and not parameter.is_external_param):
            assert parameter.ds_status == ZeroParamStatus.NOT_AVAILABLE, (
                f"parameter {parameter_name} remained gathered after the lifecycle boundary: "
                f"status={parameter.ds_status}")


class _RecursiveFrozenBlock(torch.nn.Module):
    """Recursively invoke one module instance so its ZeRO ds_id overlaps with itself."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
        torch.nn.init.orthogonal_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, value, remaining_depth):
        value = torch.tanh(F.linear(value, self.weight, self.bias))
        if remaining_depth:
            # self(...) not a new module: nested invocations share one ds_id.
            value = self(value, remaining_depth - 1)
        return value


class _RecursiveCheckpointModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.shared = _RecursiveFrozenBlock(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, value):
        value = checkpoint(self.shared, value, 2, use_reentrant=False)
        return self.head(value)


class _ReleaseTimingModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.observations = []

    def _checkpointed(self, value):
        probe = _StatusProbe(self.frozen.weight, self.observations)
        # Probe backward runs right after self.frozen's backward consumer.
        value = _RecordStatusAfterConsumer.apply(value, probe)
        value = torch.tanh(self.frozen(value))
        return torch.tanh(self.trainable(value))

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False))


class _NoGradInputModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.adapter = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def _checkpointed(self, value):
        return self.adapter(torch.tanh(self.frozen(value)))

    def forward(self, value):
        value = checkpoint(self._checkpointed, value, use_reentrant=False)
        return self.head(torch.tanh(value))


class _EarlyStopCheckpointModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.recompute_started = 0
        self.recompute_reached_tail = 0

    def _checkpointed(self, value):
        in_recompute = torch._C._current_graph_task_id() != -1
        if in_recompute:
            self.recompute_started += 1
        value = torch.sin(self.frozen(value))
        value = self.trainable(value)
        # Discarded tail after the tensors backward needs; exercises non-reentrant early-stop.
        torch.cos(value.detach())
        if in_recompute:
            self.recompute_reached_tail += 1
        return value

    def forward(self, value):
        with set_checkpoint_early_stop(True):
            value = checkpoint(self._checkpointed, value, use_reentrant=False)
        return self.head(value)


class _RecomputeExceptionModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.raise_during_recompute = True

    def _checkpointed(self, value):
        value = self.frozen(value)
        if self.raise_during_recompute and torch._C._current_graph_task_id() != -1:
            raise RuntimeError("injected checkpoint recompute forward")
        return self.trainable(torch.tanh(value))

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False))


class _IncompleteBackwardModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.backward_control = {"raise": True}

    def _checkpointed(self, value):
        value = self.trainable(torch.tanh(self.frozen(value)))
        return _RaiseOnceInBackward.apply(value, self.backward_control)

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False))


class _FrozenNoConsumerModel(torch.nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.embedding.weight.requires_grad_(False)
        self.grad_token = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.observations = []

    def _checkpointed(self, indices):
        value = self.embedding(indices) + self.grad_token
        probe = _StatusProbe(self.embedding.weight, self.observations)
        # grad_token makes the boundary differentiable; the frozen embedding still has no backward consumer.
        value = _RecordStatusAfterConsumer.apply(value, probe)
        return self.projection(value)

    def forward(self, indices):
        value = checkpoint(self._checkpointed, indices, use_reentrant=False)
        return self.head(torch.tanh(value)).sum()


class _ExternalBiasLinear(torch.nn.Linear):

    def forward(self, value):
        output = F.linear(value, self.weight, self.bias)
        return output, self.bias


class _ExternalCheckpointModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.producer = _ExternalBiasLinear(hidden_dim, hidden_dim)
        self.producer.bias.requires_grad_(False)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.external_consumer_statuses = []

    def _checkpointed(self, value):
        value, external_bias = self.producer(value)
        parameter = external_bias if hasattr(external_bias, "ds_status") else external_bias.ds_param_alias
        self.external_consumer_statuses.append(parameter.ds_status)
        # Child returns its bias as a ZeRO external param, consumed by the enclosing checkpoint fn.
        return torch.tanh(value + external_bias)

    def forward(self, value):
        return self.head(checkpoint(self._checkpointed, value, use_reentrant=False)).sum()


class TestZero3ActivationCheckpointLifecycle(DistributedTest):
    """Black-box ZeRO-3 activation-checkpoint lifecycle checks for PR #8148."""

    world_size = 1

    @pytest.mark.parametrize("fast_sharding", [False, True])
    def test_reused_checkpointed_module_invocations_release_independently(self, fast_sharding):
        """Nested calls to one module must retire independent hook invocations despite sharing a ds_id."""
        device, _, _ = initialize_distributed()
        engine = _initialize_zero3(_RecursiveCheckpointModel(hidden_dim=8),
                                   module_granularity_threshold=1_000_000 if fast_sharding else 0)
        assert F.linear is zero3_linear_wrap
        assert bool(getattr(engine.module, "_z3_leaf", False)) is fast_sharding
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        engine.backward(engine(value).sum())
        _synchronize()

        assert value.grad is not None and torch.isfinite(value.grad).all()
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    @_RECOMPUTE_RELEASE_TIMING
    def test_recomputed_parameter_releases_at_last_activation_consumer(self):
        """Frozen param should release at its real backward consumer, not at outer cleanup."""
        device, _, _ = initialize_distributed()
        model = _ReleaseTimingModel(hidden_dim=8)
        engine = _initialize_zero3(model)
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        engine.backward(engine(value).sum())
        _synchronize()

        assert model.observations, "the last-consumer autograd boundary did not execute"
        for observation in model.observations:
            assert observation["status"] == ZeroParamStatus.NOT_AVAILABLE, (
                "frozen parameter stayed gathered after its last activation consumer: "
                f"status={observation['status']}, active={sorted(observation['active_sub_modules'])}")
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_no_grad_checkpoint_input_direct_backward_releases(self):
        """Direct Tensor.backward must run the no-grad-input checkpoint cleanup after each microbatch."""
        device, _, _ = initialize_distributed()
        engine = _initialize_zero3(_NoGradInputModel(hidden_dim=8), gradient_accumulation_steps=2)

        for microbatch in range(2):
            value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=False)
            engine(value).sum().backward()
            _synchronize()
            _assert_checkpoint_state_clean(engine)
            engine.step()

        engine.destroy()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_no_grad_checkpoint_input_scaled_backward_releases(self, dtype):
        """engine.scale(loss).backward() must also drain no-grad checkpoint leftovers."""
        if dtype == torch.float16 and not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
        if dtype == torch.bfloat16 and not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")
        device, _, _ = initialize_distributed()
        engine = _initialize_zero3(_NoGradInputModel(hidden_dim=8), dtype=dtype)
        param_dtype = next(engine.module.parameters()).dtype
        value = torch.randn(2, 8, device=device, dtype=param_dtype, requires_grad=False)

        engine.scale(engine(value).sum()).backward()
        _synchronize()

        _assert_checkpoint_state_clean(engine)
        engine.step()
        engine.destroy()

    def test_nonreentrant_checkpoint_early_stop_unwinds_for_clean_retry(self):
        """Non-reentrant early-stop must balance hooks and release replay params for reuse."""
        device, _, _ = initialize_distributed()
        model = _EarlyStopCheckpointModel(hidden_dim=8)
        engine = _initialize_zero3(model)

        for _ in range(2):
            value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
            engine.backward(engine(value).sum())
            _synchronize()
            _assert_checkpoint_state_clean(engine)
            engine.step()

        assert model.recompute_started == 2, "checkpoint replay did not execute once per backward"
        assert model.recompute_reached_tail == 0, "the test topology did not trigger non-reentrant early-stop"
        engine.destroy()

    def test_recompute_forward_exception_unwinds_for_clean_retry(self):
        """An exception during replay must unwind the GraphTask so a retry starts clean."""
        device, _, _ = initialize_distributed()
        model = _RecomputeExceptionModel(hidden_dim=8)
        engine = _initialize_zero3(model)
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        try:
            engine.backward(engine(value).sum())
        except RuntimeError as error:
            assert "injected checkpoint recompute forward" in str(error)
        else:
            raise AssertionError("the injected recompute exception did not execute")

        model.raise_during_recompute = False
        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine(retry).sum().backward()
        _synchronize()

        assert retry.grad is not None and torch.isfinite(retry.grad).all()
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_reset_step_drains_incomplete_backward_state(self):
        """The next root forward's reset must drain state left by an aborted backward."""
        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        engine = _initialize_zero3(model)
        value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)

        try:
            engine.backward(engine(value).sum())
        except RuntimeError as error:
            assert "injected incomplete checkpoint backward" in str(error)
        else:
            raise AssertionError("the injected incomplete backward did not execute")

        observe_reset = {"enabled": True}

        def _observe_after_deepspeed_reset(module, unused_inputs):
            if observe_reset["enabled"]:
                _assert_checkpoint_state_clean(engine)
                observe_reset["enabled"] = False

        handle = engine.module.register_forward_pre_hook(_observe_after_deepspeed_reset)
        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine(retry).sum().backward()
        _synchronize()
        handle.remove()

        assert not observe_reset["enabled"], "the post-reset observer did not execute"
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    @_RECOMPUTE_RELEASE_TIMING
    def test_frozen_parameter_without_backward_consumer_releases_at_last_use(self):
        """Frozen param with no backward consumer should release at its last use, not at epilogue."""
        device, _, _ = initialize_distributed()
        model = _FrozenNoConsumerModel(vocab_size=16, hidden_dim=8)
        engine = _initialize_zero3(model)
        indices = torch.randint(0, 16, (2, 4), device=device)

        engine.backward(engine(indices))
        _synchronize()

        assert model.observations, "the frozen-parameter last-use boundary did not execute"
        for observation in model.observations:
            assert observation["status"] == ZeroParamStatus.NOT_AVAILABLE, (
                "frozen parameter with no backward consumer stayed gathered after its last use: "
                f"status={observation['status']}, active={sorted(observation['active_sub_modules'])}")
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_checkpoint_external_parameter_lifecycle(self):
        """A checkpoint-replayed external param must stay AVAILABLE at each consumer and clean up after."""
        device, _, _ = initialize_distributed()
        model = _ExternalCheckpointModel(hidden_dim=8)
        engine = _initialize_zero3(model)

        for _ in range(2):
            value = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
            engine.backward(engine(value))
            _synchronize()
            assert value.grad is not None and torch.isfinite(value.grad).all()
            _assert_checkpoint_state_clean(engine, require_partitioned=False)
            engine.step()

        assert model.external_consumer_statuses
        assert all(status == ZeroParamStatus.AVAILABLE for status in model.external_consumer_statuses)
        assert model.producer.bias.is_external_param
        engine.destroy()


class _SwiGLUBlock(torch.nn.Module):
    """Qwen3.5-style SwiGLU: down(silu(gate(x)) * up(x)) with intermediate >> hidden."""

    def __init__(self, hidden_dim, intermediate):
        super().__init__()
        self.gate = torch.nn.Linear(hidden_dim, intermediate, bias=False)
        self.up = torch.nn.Linear(hidden_dim, intermediate, bias=False)
        self.down = torch.nn.Linear(intermediate, hidden_dim, bias=False)

    def forward(self, value):
        return self.down(F.silu(self.gate(value)) * self.up(value))


class _SwiGLUCheckpointModel(torch.nn.Module):

    def __init__(self, hidden_dim, intermediate):
        super().__init__()
        self.block = _SwiGLUBlock(hidden_dim, intermediate)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, value):
        return self.head(checkpoint(self.block, value, use_reentrant=False)).sum()


class _SqueezedConvFn(torch.autograd.Function):
    """Stand-in for GDN causal_conv1d_fn(..., conv1d.weight.squeeze(1))."""

    @staticmethod
    def forward(ctx, value, squeezed_weight):
        ctx.save_for_backward(value, squeezed_weight)
        scale = squeezed_weight.sum(dim=-1)
        return value * scale.view(1, -1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        value, squeezed_weight = ctx.saved_tensors
        scale = squeezed_weight.sum(dim=-1)
        grad_value = grad_output * scale.view(1, -1, 1)
        grad_scale = (grad_output * value).sum(dim=(0, 2))
        grad_squeezed = grad_scale.unsqueeze(-1).expand_as(squeezed_weight)
        return grad_value, grad_squeezed


class _GDNLikeConvBlock(torch.nn.Module):

    def __init__(self, channels, kernel):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(channels, channels, kernel, groups=channels)
        self.proj = torch.nn.Linear(channels, channels)

    def forward(self, value):
        # Gather the Parameter before squeeze so a partitioned view cannot hit C++ empty.
        weight = zero3_gathered_param(self.conv1d.weight)
        conv_out = _SqueezedConvFn.apply(value, weight.squeeze(1))
        return self.proj(conv_out.transpose(1, 2)).transpose(1, 2)


class _GDNLikeCheckpointModel(torch.nn.Module):

    def __init__(self, channels, kernel):
        super().__init__()
        self.block = _GDNLikeConvBlock(channels, kernel)
        self.head = torch.nn.Linear(channels, 1)

    def forward(self, value):
        hidden = checkpoint(self.block, value, use_reentrant=False)
        return self.head(hidden.mean(dim=-1)).sum()


class TestZero3CheckpointEmptyLinear(DistributedTest):
    """ZeRO-3 + checkpoint must rematerialize gathered Linear/conv weights, not empty shards."""

    world_size = 2

    def _dtype(self):
        if get_accelerator().is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def test_linear_wrap_gathers_partitioned_weight_in_forward(self):
        """Rematerialize can hit Linear after the post-hook emptied W; forward must gather."""
        device, _, _ = initialize_distributed()
        hidden_dim = 64
        intermediate = 192
        dtype = self._dtype()
        engine = _initialize_zero3(torch.nn.Linear(hidden_dim, intermediate, bias=False), dtype=dtype)
        param_dtype = next(engine.module.parameters()).dtype
        value = torch.randn(2, hidden_dim, device=device, dtype=param_dtype)
        engine(value)
        _synchronize()
        weight = engine.module.weight
        assert weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
        output = zero3_linear_wrap(value, weight, None)
        assert output.shape[-1] == intermediate
        engine.destroy()

    def test_backward_keeps_trainable_weight_full_for_accumulate(self):
        """Backward must not re-partition a just-gathered trainable weight before AccumulateGrad.

        Mirrors the Qwen3.5 GDN out_proj crash: the weight is full when the Linear
        records its autograd node (so the returned grad shape is valid), but gets
        partitioned again before backward. The wrap re-gathers it to compute the
        grad; partitioning it back in the finally shrank it to numel 0 before
        AccumulateGrad ran on the same parameter, giving
        "size of tensor a (0) must match tensor b (N)".
        """
        device, _, _ = initialize_distributed()
        hidden_dim = 64
        out_dim = 96
        dtype = self._dtype()
        engine = _initialize_zero3(torch.nn.Linear(hidden_dim, out_dim, bias=False), dtype=dtype)
        weight = engine.module.weight
        param_dtype = next(engine.module.parameters()).dtype
        value = torch.randn(2, hidden_dim, device=device, dtype=param_dtype, requires_grad=True)
        engine(value)
        _synchronize()

        # Record the autograd node while the weight is full, then partition it so the
        # backward has to re-gather -- reproducing the recompute ordering exactly.
        weight.all_gather()
        output = zero3_linear_wrap(value, weight, None)
        weight.partition()
        assert weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
        output.sum().backward()
        _synchronize()

        grad = safe_get_full_grad(weight)
        assert grad is not None and torch.isfinite(grad).all()
        engine.destroy()

    def test_swiglu_checkpoint_backward_sees_gathered_weights(self):
        """SwiGLU rematerialize must not hit 0 vs intermediate_size after Linear partition."""
        device, _, _ = initialize_distributed()
        hidden_dim = 64
        intermediate = 192
        dtype = self._dtype()
        engine = _initialize_zero3(_SwiGLUCheckpointModel(hidden_dim, intermediate), dtype=dtype)
        assert F.linear is zero3_linear_wrap
        assert torch.nn.Linear.forward is zero3_linear_module_forward
        param_dtype = next(engine.module.parameters()).dtype
        value = torch.randn(2, hidden_dim, device=device, dtype=param_dtype, requires_grad=True)

        engine.backward(engine(value))
        _synchronize()

        assert value.grad is not None and torch.isfinite(value.grad).all()
        for parameter in engine.module.parameters():
            if parameter.requires_grad:
                grad = safe_get_full_grad(parameter)
                assert grad is not None
                assert torch.isfinite(grad).all()
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    def test_conv_squeeze_checkpoint_gathers_before_view(self):
        """GDN-shaped Conv1d.weight.squeeze(1) must gather the Parameter before the view."""
        device, _, _ = initialize_distributed()
        channels = 4
        kernel = 3
        seq_len = 8
        dtype = self._dtype()
        engine = _initialize_zero3(_GDNLikeCheckpointModel(channels, kernel), dtype=dtype)
        param_dtype = next(engine.module.parameters()).dtype
        value = torch.randn(2, channels, seq_len, device=device, dtype=param_dtype, requires_grad=True)

        engine.backward(engine(value))
        _synchronize()

        assert value.grad is not None and torch.isfinite(value.grad).all()
        for parameter in engine.module.parameters():
            if parameter.requires_grad:
                grad = safe_get_full_grad(parameter)
                assert grad is not None
                assert torch.isfinite(grad).all()
        _assert_checkpoint_state_clean(engine)
        engine.destroy()
