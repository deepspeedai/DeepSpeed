# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.compile import backend as backend_mod
import deepspeed.compile.patch_fake_tensor as patch_fake_tensor_mod
from deepspeed.compile.init_z3 import _allow_dynamo_dynamic_parameter_shapes_for_z3, _resolve_expected_grad_dtype
from deepspeed.compile.patch_fake_tensor import _resolve_zero3_guarded_value, patch_fake_tensor
from deepspeed.compile.patch_compiled_func import (get_backward_inputs, pop_backward_input, register_backward_frame)
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.parameter_offload import ZeROOrderedDict
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.torch import required_torch_version


def test_missing_grad_dtype_attribute_falls_back_to_param_dtype():

    class FakeParam:
        dtype = torch.bfloat16

    assert _resolve_expected_grad_dtype(FakeParam()) is torch.bfloat16


def test_explicit_none_grad_dtype_allows_raw_grad_dtype():
    param = torch.empty((2, 3), dtype=torch.bfloat16)
    param.grad_dtype = None

    assert _resolve_expected_grad_dtype(param) is None


def test_explicit_grad_dtype_is_preserved():
    param = torch.empty((2, 3), dtype=torch.bfloat16)
    param.grad_dtype = torch.float32

    assert _resolve_expected_grad_dtype(param) is torch.float32


def test_zero3_allows_dynamo_dynamic_parameter_shapes(monkeypatch):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = True

    class FakeDynamo:
        config = FakeDynamoConfig()

    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)

    restore = _allow_dynamo_dynamic_parameter_shapes_for_z3({})
    assert restore
    try:
        assert FakeDynamo.config.force_parameter_static_shapes is False
        assert FakeDynamo.config.force_nn_module_property_static_shapes is False
    finally:
        restore()


def test_zero3_tensor_match_delegates_non_zero_tensor(monkeypatch):
    from torch._dynamo.guards import GuardBuilder
    from torch._subclasses import FakeTensorMode

    calls = []
    sentinel = object()

    def original_tensor_match(builder, guard, value=None):
        calls.append((builder, guard, value))
        return sentinel

    monkeypatch.setattr(GuardBuilder, "TENSOR_MATCH", original_tensor_match)
    monkeypatch.setattr(
        torch._dynamo.variables.builder,
        "wrap_to_fake_tensor_and_record",
        torch._dynamo.variables.builder.wrap_to_fake_tensor_and_record,
    )
    monkeypatch.setattr(FakeTensorMode, "from_tensor", FakeTensorMode.from_tensor)
    patch_fake_tensor()
    wrapped_tensor_match = GuardBuilder.TENSOR_MATCH

    class Builder:

        def get(self, _guard):
            return value

    builder = Builder()
    guard = object()
    value = torch.ones(1)

    assert wrapped_tensor_match(builder, guard, value) is sentinel
    assert calls == [(builder, guard, value)]


def test_zero3_semantic_guard_ignores_transient_physical_state_changes(monkeypatch):

    class Accelerator:

        def is_pinned(self, _tensor):
            return False

    monkeypatch.setattr(patch_fake_tensor_mod, "get_accelerator", Accelerator)

    class Module(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0))
            self.weight.ds_id = 1
            self.weight.ds_shape = torch.Size((2, 2))
            self.weight.ds_status = ZeroParamStatus.NOT_AVAILABLE

            params = ZeROOrderedDict(parent_module=self)
            params.update(self._parameters)
            params._in_forward = True
            self._parameters = params

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight)

    module = Module()
    param = next(module.parameters())
    backend_calls = []

    def backend(_gm, _inputs):
        backend_calls.append(None)
        param.data = torch.ones(2, 2, device=param.device)
        return lambda _weight, value: (value, )

    patch_fake_tensor()
    compiled = torch.compile(module, backend=backend)
    value = torch.ones(1, 2, device=param.device)

    assert torch.equal(compiled(value), value)
    param.data = torch.empty(0, device=param.device)
    param.ds_status = ZeroParamStatus.AVAILABLE
    assert torch.equal(compiled(value), value)
    assert len(backend_calls) == 1


def test_zero3_semantic_guard_resolves_real_parameter_from_dummy_value():
    param = torch.nn.Parameter(torch.empty(0))
    param.ds_id = 1
    param.ds_shape = torch.Size((2, 2))

    class Guard:
        name = "param"

    guard = Guard()

    class Builder:

        def get(self, guard_or_name):
            if guard_or_name is guard:
                raise TypeError("older PyTorch expects the source expression")
            assert guard_or_name == guard.name
            return param

    dummy = torch.nn.Parameter(torch.empty(2, 2))

    assert _resolve_zero3_guarded_value(Builder(), guard, dummy) is param


def test_zero_ordered_dict_does_not_read_ds_status_while_compiling(monkeypatch):

    class StatusTrap:

        @property
        def ds_status(self):
            raise AssertionError("volatile ds_status must not be read while compiling")

    module = torch.nn.Module()
    params = ZeROOrderedDict(parent_module=module)
    param = StatusTrap()
    params["weight"] = param
    params._in_forward = True
    module._parameters = params
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    assert module._parameters["weight"] is param


@pytest.mark.parametrize("first_owner_to_restore", [0, 1])
def test_zero3_dynamo_config_restores_after_last_overlapping_owner(monkeypatch, first_owner_to_restore):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = False

    class FakeDynamo:
        config = FakeDynamoConfig()

    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)
    restores = [_allow_dynamo_dynamic_parameter_shapes_for_z3({}), _allow_dynamo_dynamic_parameter_shapes_for_z3({})]

    assert all(restores)
    restores[first_owner_to_restore]()
    assert FakeDynamo.config.force_parameter_static_shapes is False
    restores[1 - first_owner_to_restore]()
    assert FakeDynamo.config.force_parameter_static_shapes is True
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False


@pytest.mark.parametrize("first_owner_to_destroy", [0, 1])
def test_zero3_dynamo_config_restores_when_overlapping_engines_are_destroyed(monkeypatch, first_owner_to_destroy):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = False

    class FakeDynamo:
        config = FakeDynamoConfig()

    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)
    engines = [object.__new__(DeepSpeedEngine), object.__new__(DeepSpeedEngine)]
    for engine in engines:
        torch.nn.Module.__init__(engine)
        engine._deepcompile_active = False
        engine._deepcompile_dynamo_config_restore = _allow_dynamo_dynamic_parameter_shapes_for_z3({})

    engines[first_owner_to_destroy].destroy()
    assert FakeDynamo.config.force_parameter_static_shapes is False
    engines[1 - first_owner_to_destroy].destroy()
    assert FakeDynamo.config.force_parameter_static_shapes is True
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False


@pytest.mark.parametrize("first_owner_to_destroy", [0, 1])
def test_destroy_releases_only_owner_with_overlapping_frame_ids(first_owner_to_destroy):
    original_autograd_function = torch.autograd.Function
    engines = [object.__new__(DeepSpeedEngine), object.__new__(DeepSpeedEngine)]
    owners = [object(), object()]
    frame_id = 17
    for owner, engine in zip(owners, engines):
        torch.nn.Module.__init__(engine)
        engine._deepcompile_active = False
        engine._deepcompile_owned_frames = {(owner, frame_id)}

    backend_mod.frames_needing_bwd.clear()
    frame_keys = [(owners[0], frame_id), (owners[1], frame_id)]
    backend_mod.frames_needing_bwd.update(frame_keys)
    backend_mod.patch_compiled_func()
    for frame_key in frame_keys:
        get_backward_inputs()[frame_key] = [(torch.ones(1), )]

    try:
        engines[first_owner_to_destroy].destroy()
        surviving_frame = frame_keys[1 - first_owner_to_destroy]
        assert backend_mod.frames_needing_bwd == {surviving_frame}
        assert set(get_backward_inputs()) == {surviving_frame}
        assert torch.autograd.Function is not original_autograd_function

        engines[1 - first_owner_to_destroy].destroy()
        assert backend_mod.frames_needing_bwd == set()
        assert get_backward_inputs() == {}
        assert torch.autograd.Function is original_autograd_function
    finally:
        backend_mod.frames_needing_bwd.clear()
        backend_mod.unpatch_compiled_func()


def test_destroyed_owner_inputs_cannot_be_consumed_by_survivor():
    original_autograd_function = torch.autograd.Function
    engines = [object.__new__(DeepSpeedEngine), object.__new__(DeepSpeedEngine)]
    owners = [object(), object()]
    frame_keys = [(owners[0], 17), (owners[1], 17)]
    for frame_key, engine in zip(frame_keys, engines):
        torch.nn.Module.__init__(engine)
        engine._deepcompile_active = False
        engine._deepcompile_owned_frames = {frame_key}

    backend_mod.frames_needing_bwd.clear()
    backend_mod.frames_needing_bwd.update(frame_keys)
    backend_mod.patch_compiled_func()
    register_backward_frame(frame_keys[0])

    class CompiledFunction(torch.autograd.Function):
        compiled_bw = object()

        @staticmethod
        def _backward_impl(ctx, all_args):
            return all_args

        @staticmethod
        def _backward_prologue(ctx, *grad_outputs):
            return grad_outputs

    owner_a_inputs = (torch.ones(1), )
    if required_torch_version(min_version=2.7):
        CompiledFunction._backward_impl(None, owner_a_inputs)
    else:
        CompiledFunction._backward_prologue(None, *owner_a_inputs)

    try:
        assert len(get_backward_inputs(frame_keys[0])) == 1
        assert get_backward_inputs(frame_keys[1]) == []

        engines[0].destroy()

        assert backend_mod.frames_needing_bwd == {frame_keys[1]}
        assert get_backward_inputs(frame_keys[0]) == []
        assert pop_backward_input(frame_keys[1]) is None
        assert torch.autograd.Function is not original_autograd_function

        engines[1].destroy()
        assert get_backward_inputs() == {}
        assert torch.autograd.Function is original_autograd_function
    finally:
        backend_mod.frames_needing_bwd.clear()
        backend_mod.unpatch_compiled_func()


def test_deactivation_releases_only_the_engine_owned_state(monkeypatch):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = False

    class FakeDynamo:
        config = FakeDynamoConfig()

    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)
    engine = object.__new__(DeepSpeedEngine)
    torch.nn.Module.__init__(engine)
    engine._deepcompile_active = True
    engine.module_forward_pre_hook = object()
    engine.module_forward_post_hook = object()
    engine._deepcompile_dynamo_config_restore = _allow_dynamo_dynamic_parameter_shapes_for_z3({})
    original_autograd_function = torch.autograd.Function
    owner = object()
    other_owner = object()
    frame_key = (owner, 17)
    other_frame_key = (other_owner, 18)
    backend_mod.frames_needing_bwd.clear()
    backend_mod.frames_needing_bwd.update((frame_key, other_frame_key))
    engine._deepcompile_owned_frames = {frame_key}
    backend_mod.patch_compiled_func()
    get_backward_inputs()[frame_key] = [(torch.ones(1), )]
    get_backward_inputs()[other_frame_key] = [(torch.ones(1), )]

    try:
        engine._set_deepcompile_active(False)

        assert FakeDynamo.config.force_parameter_static_shapes is True
        assert FakeDynamo.config.force_nn_module_property_static_shapes is False
        assert not hasattr(engine, "_deepcompile_dynamo_config_restore")
        assert engine._deepcompile_owned_frames == set()
        assert backend_mod.frames_needing_bwd == {other_frame_key}
        assert set(get_backward_inputs()) == {other_frame_key}
        assert torch.autograd.Function is not original_autograd_function
        assert engine.is_deepcompile_active() is False
    finally:
        backend_mod.frames_needing_bwd.clear()
        backend_mod.unpatch_compiled_func()
