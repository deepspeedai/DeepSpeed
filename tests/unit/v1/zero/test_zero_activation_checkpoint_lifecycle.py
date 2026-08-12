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
from deepspeed.runtime.zero.linear import zero3_linear_wrap
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

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


def _assert_backward_state_clean(engine):
    """Assert that the next root forward discarded an incomplete backward."""
    optimizer = engine.optimizer
    assert optimizer._backward_active_depth == 0
    assert optimizer._remaining_grad_acc_hooks == 0
    for bucket in optimizer.ipg_buckets.values():
        assert not bucket.params
        assert bucket.elements == 0
    reduced_flags = optimizer.params_already_reduced
    if isinstance(reduced_flags, dict):
        reduced_flags = reduced_flags.values()
    assert not any(reduced_flags)
    assert not getattr(optimizer, "extra_large_param_to_reduce", {})
    groups = getattr(optimizer, "bit16_groups", getattr(optimizer, "fp16_groups", ()))
    for group in groups:
        for parameter in group:
            assert parameter.grad is None
            assert getattr(parameter, "grad_accum", None) is None


def _snapshot_trainable_parameters(engine):
    parameters = [parameter for parameter in engine.module.parameters() if parameter.requires_grad]
    with deepspeed.zero.GatheredParameters(parameters):
        return [parameter.detach().float().cpu().clone() for parameter in parameters]


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
                _assert_backward_state_clean(engine)
                observe_reset["enabled"] = False

        handle = engine.module.register_forward_pre_hook(_observe_after_deepspeed_reset)
        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine(retry).sum().backward()
        _synchronize()
        handle.remove()

        assert not observe_reset["enabled"], "the post-reset observer did not execute"
        _assert_checkpoint_state_clean(engine)
        engine.destroy()

    @pytest.mark.parametrize("zero_stage", [2, 3], ids=["zero2", "zero3"])
    def test_partial_reduction_before_aborted_backward_requires_restart(self, zero_stage):
        """A retry must fail closed once an aborted backward has reduced a partial bucket."""
        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        model.backward_control["raise"] = False
        config = get_config_dict(zero_stage, gradient_accumulation_steps=2, force_fp32=True)
        config["zero_optimization"]["reduce_bucket_size"] = 8
        if zero_stage == 3:
            config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
            config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)

        first = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine.backward(engine(first).sum())
        engine.step()

        model.backward_control["raise"] = True
        failing = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            engine.backward(engine(failing).sum())

        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine(retry)
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine(retry)

        engine.destroy()

    def test_zero3_leaf_partial_reduction_before_abort_requires_restart(self):
        """A multi-parameter ZeRO-3 leaf hook must record every bucket drain."""
        from deepspeed.utils import set_z3_leaf_modules

        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        model.backward_control["raise"] = False
        set_z3_leaf_modules(model, [torch.nn.Linear])
        config = get_config_dict(3, gradient_accumulation_steps=2, force_fp32=True)
        config["zero_optimization"]["reduce_bucket_size"] = 8
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
        config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)

        first = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine.backward(engine(first).sum())
        engine.step()

        model.backward_control["raise"] = True
        failing = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            engine.backward(engine(failing).sum())

        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine(retry)
        engine.destroy()

    @pytest.mark.parametrize("zero_stage", [2, 3], ids=["zero2", "zero3"])
    def test_step_after_aborted_backward_requires_restart(self, zero_stage):
        """An optimizer step cannot bypass aborted-backward cleanup at the next forward."""
        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        config = get_config_dict(zero_stage, gradient_accumulation_steps=1, force_fp32=True)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)

        failing = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            engine.backward(engine(failing).sum())
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine.step()

        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine(retry)
        engine.destroy()

    @pytest.mark.parametrize("zero_stage", [2, 3], ids=["zero2", "zero3"])
    def test_non_boundary_step_after_aborted_backward_requires_restart(self, zero_stage):
        """A non-boundary GAS step must reject an incomplete backward without advancing."""
        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        config = get_config_dict(zero_stage, gradient_accumulation_steps=2, force_fp32=True)
        config["zero_optimization"]["reduce_bucket_size"] = 1_000_000
        if zero_stage == 3:
            config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
            config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)

        failing = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            engine.backward(engine(failing).sum())

        initial_micro_steps = engine.micro_steps
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine.step()
        assert engine.micro_steps == initial_micro_steps

        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine(retry)
        engine.destroy()

    def test_runtime_step_replacement_keeps_aborted_backward_guard(self):
        """Instance-level ZeRO step replacement must retain the incomplete-backward guard."""
        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        config = get_config_dict(3, gradient_accumulation_steps=1, force_fp32=True)
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
        config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)
        replacement_calls = []
        engine.optimizer.step = lambda closure=None: replacement_calls.append(closure)

        failing = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            engine.backward(engine(failing).sum())
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine.step()

        assert not replacement_calls
        engine.destroy()

    def test_zero2_oversized_unreduced_gradient_is_cleared_for_retry(self):
        """Discard an oversized ZeRO-2 bucket side-table entry after a safe abort."""
        device, _, _ = initialize_distributed()
        torch.manual_seed(1234)
        model = _IncompleteBackwardModel(hidden_dim=8)
        model.head.bias.requires_grad_(False)
        config = get_config_dict(2, gradient_accumulation_steps=1, force_fp32=True)
        config["zero_optimization"]["reduce_bucket_size"] = 1
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)

        failing = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            engine.backward(engine(failing).sum())
        assert engine.optimizer.extra_large_param_to_reduce

        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        engine.backward(engine(retry).sum())
        engine.step()

        _assert_backward_state_clean(engine)
        engine.destroy()

    def test_bf16_immediate_update_aborted_backward_requires_restart(self):
        """BF16 hooks that reached the persistent FP32 buffer make retry unsafe."""
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bfloat16 is not supported on this accelerator")

        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        config = get_config_dict(1, gradient_accumulation_steps=1)
        config["bf16"] = {"enabled": True, "immediate_grad_update": True}
        config["data_types"] = {"grad_accum_dtype": "fp32"}
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)

        failing = torch.randn(2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            engine.backward(engine(failing).sum())
        assert any(any(group) for group in engine.optimizer.fp32_groups_has_gradients)

        retry = torch.randn(2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            engine(retry)
        engine.destroy()

    def test_zero1_prior_accumulation_aborted_backward_requires_restart(self):
        """Keep a valid earlier ZeRO-1 microbatch from being silently discarded."""
        device, _, _ = initialize_distributed()
        engines = []

        for _ in range(2):
            torch.manual_seed(1234)
            model = _IncompleteBackwardModel(hidden_dim=8)
            model.backward_control["raise"] = False
            config = get_config_dict(1, gradient_accumulation_steps=2, force_fp32=True)
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)
            engines.append(engine)

        control, aborted = engines
        first = torch.randn(2, 8, device=device, dtype=torch.float32)
        second = torch.randn(2, 8, device=device, dtype=torch.float32)
        for engine in engines:
            engine.backward(engine(first.clone().requires_grad_(True)).sum())
            engine.step()

        control.backward(control(second.clone().requires_grad_(True)).sum())
        control.step()

        aborted.module.backward_control["raise"] = True
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            aborted.backward(aborted(second.clone().requires_grad_(True)).sum())
        with pytest.raises(RuntimeError, match="restart the accumulation window"):
            aborted(second.clone().requires_grad_(True))

        control.destroy()
        aborted.destroy()

    @pytest.mark.parametrize("zero_stage", [2, 3], ids=["zero2", "zero3"])
    def test_coalesced_prior_backward_abort_requires_restart(self, zero_stage):
        """Keep an earlier coalesced backward from being silently discarded."""
        device, _, _ = initialize_distributed()
        model = _IncompleteBackwardModel(hidden_dim=8)
        model.backward_control["raise"] = False
        config = get_config_dict(zero_stage, gradient_accumulation_steps=1, force_fp32=True)
        if zero_stage == 3:
            config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
            config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)

        first = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        failing = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        retry = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=True)
        with engine.coalesce_grad_reduction():
            engine.backward(engine(first).sum())
            model.backward_control["raise"] = True
            with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
                engine.backward(engine(failing).sum())
            with pytest.raises(RuntimeError, match="restart the accumulation window"):
                engine(retry)

        engine.destroy()

    @pytest.mark.parametrize("zero_stage", [2, 3], ids=["zero2", "zero3"])
    def test_pre_reduction_abort_retry_matches_clean_control(self, zero_stage):
        """Discard unreduced failed gradients without losing earlier GAS contributions."""
        device, _, _ = initialize_distributed()
        first = torch.randn(2, 8, device=device, dtype=torch.float32)
        second = torch.randn(2, 8, device=device, dtype=torch.float32)
        engines = []

        for _ in range(2):
            torch.manual_seed(1234)
            model = _IncompleteBackwardModel(hidden_dim=8)
            model.backward_control["raise"] = False
            config = get_config_dict(zero_stage, gradient_accumulation_steps=2, force_fp32=True)
            config["zero_optimization"]["reduce_bucket_size"] = 1_000_000
            if zero_stage == 3:
                config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
                config["zero_optimization"]["stage3_max_reuse_distance"] = 0
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)
            engines.append(engine)

        control, recovered = engines
        for engine in engines:
            engine.backward(engine(first.clone().requires_grad_(True)).sum())
            engine.step()

        control.backward(control(second.clone().requires_grad_(True)).sum())
        control.step()

        recovered.module.backward_control["raise"] = True
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            recovered.backward(recovered(second.clone().requires_grad_(True)).sum())
        recovered.backward(recovered(second.clone().requires_grad_(True)).sum())
        recovered.step()

        for control_parameter, recovered_parameter in zip(_snapshot_trainable_parameters(control),
                                                          _snapshot_trainable_parameters(recovered)):
            torch.testing.assert_close(recovered_parameter, control_parameter, rtol=0, atol=0)

        control.destroy()
        recovered.destroy()

    def test_zero2_cpu_offload_pre_reduction_abort_retry_matches_clean_control(self):
        """A safe retry must not import CPU gradients from the previous optimizer step."""
        device, _, dtype = initialize_distributed()
        first_window = [torch.randn(2, 8, device=device, dtype=dtype) for _ in range(2)]
        second_window = [torch.randn(2, 8, device=device, dtype=dtype) for _ in range(2)]
        engines = []

        for _ in range(2):
            torch.manual_seed(1234)
            model = _IncompleteBackwardModel(hidden_dim=8)
            model.backward_control["raise"] = False
            config = get_config_dict(2, gradient_accumulation_steps=2)
            config["zero_optimization"]["reduce_bucket_size"] = 1_000_000
            config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
            config["zero_force_ds_cpu_optimizer"] = False
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=trainable_parameters)
            engines.append(engine)

        control, recovered = engines
        for microbatch in first_window:
            for engine in engines:
                engine.backward(engine(microbatch.clone().requires_grad_(True)).sum())
                engine.step()
        for engine in engines:
            assert engine.optimizer.accumulated_grads_in_cpu

        control.backward(control(second_window[0].clone().requires_grad_(True)).sum())
        control.step()

        recovered.module.backward_control["raise"] = True
        with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
            recovered.backward(recovered(second_window[0].clone().requires_grad_(True)).sum())
        recovered.backward(recovered(second_window[0].clone().requires_grad_(True)).sum())
        recovered.step()

        for engine in engines:
            engine.backward(engine(second_window[1].clone().requires_grad_(True)).sum())
            engine.step()

        for control_parameter, recovered_parameter in zip(_snapshot_trainable_parameters(control),
                                                          _snapshot_trainable_parameters(recovered)):
            torch.testing.assert_close(recovered_parameter, control_parameter, rtol=0, atol=0)

        control.destroy()
        recovered.destroy()

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
