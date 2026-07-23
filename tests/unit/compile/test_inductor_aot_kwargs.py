# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed.compile.inductor as inductor


def _compiler(name):

    def compiler(*args, **kwargs):
        return name, args, kwargs

    compiler.__name__ = name
    return compiler


def _install_patch_spies(monkeypatch):
    compiler_calls = []
    partition_calls = []

    def fake_patch_compiler(original_compiler, dc_compiler, z3_partition, graph_id, graph_param_manager, bwd):
        wrapped = {
            "original_compiler": original_compiler,
            "dc_compiler": dc_compiler,
            "z3_partition": z3_partition,
            "graph_id": graph_id,
            "graph_param_manager": graph_param_manager,
            "bwd": bwd,
        }
        compiler_calls.append(wrapped)
        return wrapped

    def fake_wrap_partition_fn(z3_partition, partition_fn, real_inputs, param_indices, frame_id, frames_partitioned):
        wrapped = {
            "z3_partition": z3_partition,
            "partition_fn": partition_fn,
            "real_inputs": real_inputs,
            "param_indices": param_indices,
            "frame_id": frame_id,
            "frames_partitioned": frames_partitioned,
        }
        partition_calls.append(wrapped)
        return wrapped

    monkeypatch.setattr(inductor, "patch_compiler", fake_patch_compiler)
    monkeypatch.setattr(inductor, "wrap_partition_fn", fake_wrap_partition_fn)
    return compiler_calls, partition_calls


def _patch_kwargs(kwargs, monkeypatch):
    compiler_calls, partition_calls = _install_patch_spies(monkeypatch)
    make_fw_graph = object()
    make_bw_graph = object()
    real_inputs = object()
    param_indices = object()
    param_manager = object()
    frames_partitioned = set()

    applied = inductor._patch_deepcompile_aot_kwargs(kwargs,
                                                     graph_id=7,
                                                     z3_partition=True,
                                                     make_fw_graph=make_fw_graph,
                                                     make_bw_graph=make_bw_graph,
                                                     real_inputs=real_inputs,
                                                     param_indices=param_indices,
                                                     param_manager=param_manager,
                                                     frame_id=11,
                                                     frames_partitioned=frames_partitioned)

    return {
        "applied": applied,
        "compiler_calls": compiler_calls,
        "partition_calls": partition_calls,
        "make_fw_graph": make_fw_graph,
        "make_bw_graph": make_bw_graph,
        "real_inputs": real_inputs,
        "param_indices": param_indices,
        "param_manager": param_manager,
        "frames_partitioned": frames_partitioned,
    }


def test_legacy_inductor_shape_wraps_explicit_bw_compiler(monkeypatch):
    fw_compiler = _compiler("fw")
    bw_compiler = _compiler("bw")
    inference_compiler = _compiler("inference")
    partition_fn = _compiler("partition")
    kwargs = {
        "fw_compiler": fw_compiler,
        "bw_compiler": bw_compiler,
        "inference_compiler": inference_compiler,
        "partition_fn": partition_fn,
    }

    result = _patch_kwargs(kwargs, monkeypatch)

    assert result["applied"] is True
    assert len(result["compiler_calls"]) == 2
    assert result["compiler_calls"][0]["original_compiler"] is fw_compiler
    assert result["compiler_calls"][0]["dc_compiler"] is result["make_fw_graph"]
    assert result["compiler_calls"][0]["bwd"] is False
    assert result["compiler_calls"][1]["original_compiler"] is bw_compiler
    assert result["compiler_calls"][1]["dc_compiler"] is result["make_bw_graph"]
    assert result["compiler_calls"][1]["bwd"] is True
    assert kwargs["fw_compiler"] is result["compiler_calls"][0]
    assert kwargs["bw_compiler"] is result["compiler_calls"][1]
    assert kwargs["inference_compiler"] is result["compiler_calls"][0]
    assert kwargs["partition_fn"] is result["partition_calls"][0]
    assert result["partition_calls"][0]["partition_fn"] is partition_fn


def test_missing_bw_compiler_uses_original_fw_compiler_for_backward(monkeypatch):
    fw_compiler = _compiler("fw")
    partition_fn = _compiler("partition")
    kwargs = {
        "fw_compiler": fw_compiler,
        "partition_fn": partition_fn,
    }

    result = _patch_kwargs(kwargs, monkeypatch)

    assert result["applied"] is True
    assert result["compiler_calls"][0]["original_compiler"] is fw_compiler
    assert result["compiler_calls"][0]["bwd"] is False
    assert result["compiler_calls"][1]["original_compiler"] is fw_compiler
    assert result["compiler_calls"][1]["dc_compiler"] is result["make_bw_graph"]
    assert result["compiler_calls"][1]["bwd"] is True
    assert kwargs["bw_compiler"] is result["compiler_calls"][1]


def test_torchxla_openxla_shape_passes_through_unchanged(monkeypatch):
    kwargs = {"fw_compiler": _compiler("openxla_eval_boxed")}
    original_kwargs = dict(kwargs)

    result = _patch_kwargs(kwargs, monkeypatch)

    assert result["applied"] is False
    assert result["compiler_calls"] == []
    assert result["partition_calls"] == []
    assert kwargs == original_kwargs


def test_deepcompile_z3_inductor_config_patch_disables_available_reduction_heuristics():
    config = torch._inductor.config
    triton_config = config.triton
    original_values = {
        config_name: getattr(triton_config,
                             config_name.split(".", 1)[1])
        for config_name in inductor._DEEP_COMPILE_Z3_INDUCTOR_REDUCTION_CONFIG
        if hasattr(triton_config,
                   config_name.split(".", 1)[1])
    }
    assert original_values

    with inductor.deepcompile_z3_inductor_config_patch(enabled=True):
        for config_name in original_values:
            assert getattr(triton_config, config_name.split(".", 1)[1]) is False

    for config_name, original_value in original_values.items():
        assert getattr(triton_config, config_name.split(".", 1)[1]) is original_value

    with inductor.deepcompile_z3_inductor_config_patch(enabled=False):
        for config_name, original_value in original_values.items():
            assert getattr(triton_config, config_name.split(".", 1)[1]) is original_value


def test_deepcompile_z3_inductor_config_patch_skips_unavailable_reduction_heuristics(monkeypatch):

    class FakeTritonConfig:
        persistent_reductions = True

    class FakePatch:

        def __init__(self, overrides):
            self.overrides = overrides

        def __enter__(self):
            assert self.overrides == {"triton.persistent_reductions": False}
            FakeTritonConfig.persistent_reductions = False

        def __exit__(self, exc_type, exc, tb):
            FakeTritonConfig.persistent_reductions = True

    class FakeConfig:
        triton = FakeTritonConfig()

        @staticmethod
        def patch(overrides):
            return FakePatch(overrides)

    class FakeInductor:
        config = FakeConfig()

    monkeypatch.setattr(torch, "_inductor", FakeInductor())

    with inductor.deepcompile_z3_inductor_config_patch(enabled=True):
        assert FakeTritonConfig.persistent_reductions is False

    assert FakeTritonConfig.persistent_reductions is True


def test_patch_compiler_applies_z3_inductor_config_during_original_compile(monkeypatch):
    events = []

    class ConfigContext:

        def __init__(self, enabled):
            self.enabled = enabled

        def __enter__(self):
            events.append(("enter", self.enabled))

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", self.enabled))

    monkeypatch.setattr(inductor, "deepcompile_z3_inductor_config_patch", lambda enabled: ConfigContext(enabled))

    class ParamManager:
        param_names = []

    def original_compiler(gm, inputs):
        events.append(("compile", tuple(inputs)))
        return "compiled"

    gm = torch.fx.symbolic_trace(lambda: torch.ones(()))
    wrapped = inductor.patch_compiler(original_compiler,
                                      dc_compiler=lambda gm, inputs: gm.graph,
                                      z3_partition=True,
                                      graph_id=7,
                                      graph_param_manager={7: ParamManager()},
                                      bwd=False)

    assert wrapped(gm, ()) == "compiled"
    assert events == [("enter", True), ("compile", ()), ("exit", True)]
