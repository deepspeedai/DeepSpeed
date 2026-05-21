# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import torch

import deepspeed.compile.passes.selective_gather as selective_gather_pass
from deepspeed.compile.profilers import ProfilingResult


class FakeAccelerator:

    def __init__(self, total_mem=1000, available_mem=250, device="cpu"):
        self._total_mem = total_mem
        self._available_mem = available_mem
        self._device = device

    def total_memory(self):
        return self._total_mem

    def available_memory(self):
        return self._available_mem

    def current_device(self):
        return self._device


class FakeDeepCompileHandle:

    def __init__(self):
        self.persistent_ds_ids = []

    def set_persistent(self, ds_id):
        self.persistent_ds_ids.append(ds_id)


def _make_param(numel, ds_persist=False):
    return SimpleNamespace(numel=numel,
                           dtype=torch.float32,
                           param=SimpleNamespace(ds_persist=ds_persist, ds_shape=(numel, )))


def test_compute_persistence_budget_prefers_peak_resident_alloc():
    budget = selective_gather_pass._compute_persistence_budget(all_graph_mem_records=[[("fwd", 700, 0, 980)],
                                                                                      [("bwd", 720, 20, 800)]],
                                                               total_mem=1000,
                                                               mem_margin=0.1)

    assert budget["usable_mem"] == 900
    assert budget["peak_resident_alloc"] == 720
    assert budget["transient_peak"] == 980
    assert budget["available_mem"] == 180
    assert budget["profiled_list_count"] == 2


def test_compute_persistence_budget_clamps_when_resident_alloc_exceeds_budget():
    budget = selective_gather_pass._compute_persistence_budget(all_graph_mem_records=[[("fwd", 920, 0, 980)],
                                                                                      [("bwd", 910, -10, 950)]],
                                                               total_mem=1000,
                                                               mem_margin=0.1)

    assert budget["usable_mem"] == 900
    assert budget["peak_resident_alloc"] == 920
    assert budget["available_mem"] == 0


def test_selective_gather_sets_persistent_params_when_resident_headroom_exists(monkeypatch):
    fake_handle = FakeDeepCompileHandle()

    monkeypatch.setattr(selective_gather_pass, "get_accelerator", lambda: FakeAccelerator(available_mem=220))
    monkeypatch.setattr(selective_gather_pass, "get_deepcompile_handle", lambda: fake_handle)
    monkeypatch.setattr(selective_gather_pass.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(selective_gather_pass.dist, "all_reduce", lambda tensor, op: tensor)

    profiling_results = {
        0:
        ProfilingResult(fwd_graph=SimpleNamespace(nodes=[]),
                        bwd_graph=SimpleNamespace(nodes=[]),
                        fwd_mem=[("fwd", 700, 0, 950)],
                        bwd_mem=[("bwd", 680, -20, 720)])
    }
    param_manager = {
        0:
        SimpleNamespace(params={
            "small": _make_param(25),
            "large": _make_param(60),
        },
                        ds_ids={
                            "small": 1,
                            "large": 2,
                        })
    }
    gm = object()

    returned = selective_gather_pass.selective_gather(gm,
                                                      graph_id=0,
                                                      graph_order=[(0, True)],
                                                      profiling_results=profiling_results,
                                                      create_inputs_fn=None,
                                                      mem_budget=0.0,
                                                      param_manager=param_manager,
                                                      bwd=True)

    assert returned is gm
    assert fake_handle.persistent_ds_ids == [1]
