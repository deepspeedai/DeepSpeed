# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.compile import backend as backend_mod
from deepspeed.compile.init_z3 import (_allow_dynamo_dynamic_parameter_shapes_for_z3,
                                       _deactivate_deepcompile_on_backend_failure, _resolve_expected_grad_dtype)
from deepspeed.runtime.engine import DeepSpeedEngine


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

    compile_kwargs = {}
    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)

    restore = _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs)
    assert restore
    try:
        assert "dynamic" not in compile_kwargs
        assert FakeDynamo.config.force_parameter_static_shapes is False
        assert FakeDynamo.config.force_nn_module_property_static_shapes is False
    finally:
        restore()


def test_zero3_preserves_explicit_dynamo_dynamic_setting(monkeypatch):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = True

    class FakeDynamo:
        config = FakeDynamoConfig()

    compile_kwargs = {"dynamic": False}
    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)

    restore = _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs)
    assert restore
    try:
        assert compile_kwargs["dynamic"] is False
    finally:
        restore()


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
    assert FakeDynamo.config.force_parameter_static_shapes is False
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False

    restores[first_owner_to_restore]()
    assert FakeDynamo.config.force_parameter_static_shapes is False
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False

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

    assert FakeDynamo.config.force_parameter_static_shapes is False
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False

    engines[first_owner_to_destroy].destroy()
    assert FakeDynamo.config.force_parameter_static_shapes is False
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False

    engines[1 - first_owner_to_destroy].destroy()
    assert FakeDynamo.config.force_parameter_static_shapes is True
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False


@pytest.mark.parametrize("first_owner_to_destroy", [0, 1])
def test_destroy_releases_only_the_engine_owned_compiled_backward_frames(first_owner_to_destroy):
    original_autograd_function = torch.autograd.Function
    engines = [object.__new__(DeepSpeedEngine), object.__new__(DeepSpeedEngine)]
    for frame_id, engine in enumerate(engines, start=17):
        torch.nn.Module.__init__(engine)
        engine._deepcompile_active = False
        engine._deepcompile_owned_frames = {frame_id}

    backend_mod.frames_needing_bwd.clear()
    backend_mod.frames_needing_bwd.update((17, 18))
    backend_mod.patch_compiled_func()
    backend_mod.get_backward_inputs().append((torch.ones(1), ))

    try:
        engines[first_owner_to_destroy].destroy()
        surviving_frame = 18 if first_owner_to_destroy == 0 else 17
        assert backend_mod.frames_needing_bwd == {surviving_frame}
        assert len(backend_mod.get_backward_inputs()) == 1
        assert torch.autograd.Function is not original_autograd_function

        engines[1 - first_owner_to_destroy].destroy()
        assert backend_mod.frames_needing_bwd == set()
        assert backend_mod.get_backward_inputs() == []
        assert torch.autograd.Function is original_autograd_function
    finally:
        backend_mod.frames_needing_bwd.clear()
        backend_mod.unpatch_compiled_func()


def test_zero3_compile_failure_deactivation_restores_dynamo_config(monkeypatch):

    class FakeDynamoConfig:
        force_parameter_static_shapes = True
        force_nn_module_property_static_shapes = False

    class FakeDynamo:
        config = FakeDynamoConfig()

    monkeypatch.setattr(torch, "_dynamo", FakeDynamo)
    restore = _allow_dynamo_dynamic_parameter_shapes_for_z3({})
    cleanup_states = []
    fake_engine = type(
        "FakeEngine", (), {
            "_deepcompile_active": True,
            "module_forward_pre_hook": None,
            "module_forward_post_hook": None,
            "_create_module_forward_pre_hook": lambda self: object(),
            "_create_module_forward_post_hook": lambda self: object(),
        })()
    fake_handle = type("FakeDeepCompileHandle", (),
                       {"cleanup": lambda self: cleanup_states.append(fake_engine._deepcompile_active)})()
    monkeypatch.setattr("deepspeed.compile.init_z3.get_deepcompile_handle", lambda: fake_handle)
    fake_engine._deepcompile_dynamo_config_restore = restore
    original_autograd_function = torch.autograd.Function
    backend_mod.frames_needing_bwd.clear()
    backend_mod.frames_needing_bwd.update((17, 18))
    fake_engine._deepcompile_owned_frames = {17}
    backend_mod.patch_compiled_func()
    backend_mod.get_backward_inputs().append((torch.ones(1), ))

    assert FakeDynamo.config.force_parameter_static_shapes is False
    assert FakeDynamo.config.force_nn_module_property_static_shapes is False

    def failing_backend():
        raise RuntimeError("compile failed")

    backend = _deactivate_deepcompile_on_backend_failure(fake_engine, failing_backend)
    fake_engine._release_deepcompile_compiled_backward_state = (
        lambda: DeepSpeedEngine._release_deepcompile_compiled_backward_state(fake_engine))
    fake_engine._release_deepcompile_dynamo_config = (
        lambda: DeepSpeedEngine._release_deepcompile_dynamo_config(fake_engine))
    fake_engine._set_deepcompile_active = lambda active: DeepSpeedEngine._set_deepcompile_active(fake_engine, active)
    fake_engine.is_deepcompile_active = lambda: fake_engine._deepcompile_active

    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="compile failed"):
                backend()

        assert FakeDynamo.config.force_parameter_static_shapes is True
        assert FakeDynamo.config.force_nn_module_property_static_shapes is False
        assert not hasattr(fake_engine, "_deepcompile_dynamo_config_restore")
        assert backend_mod.frames_needing_bwd == {18}
        assert len(backend_mod.get_backward_inputs()) == 1
        assert torch.autograd.Function is not original_autograd_function
        assert cleanup_states == [True]

        fake_engine.optimizer = None
        fake_engine.checkpoint_engine = None
        DeepSpeedEngine.destroy(fake_engine)
        assert cleanup_states == [True]
    finally:
        backend_mod.frames_needing_bwd.clear()
        backend_mod.unpatch_compiled_func()
