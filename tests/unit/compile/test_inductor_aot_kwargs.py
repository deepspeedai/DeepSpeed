# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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
