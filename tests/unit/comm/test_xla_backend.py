# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import sys
import types
from datetime import timedelta

import pytest
import torch

from deepspeed.comm.constants import XLA_BACKEND


def _install_fake_torch_xla(monkeypatch, local_ordinal=0):
    torch_xla = types.ModuleType("torch_xla")
    torch_xla_core = types.ModuleType("torch_xla.core")
    torch_xla_xla_model = types.ModuleType("torch_xla.core.xla_model")
    torch_xla_distributed = types.ModuleType("torch_xla.distributed")
    torch_xla_backend = types.ModuleType("torch_xla.distributed.xla_backend")

    torch_xla_xla_model.get_local_ordinal = lambda: local_ordinal

    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", torch_xla_core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", torch_xla_xla_model)
    monkeypatch.setitem(sys.modules, "torch_xla.distributed", torch_xla_distributed)
    monkeypatch.setitem(sys.modules, "torch_xla.distributed.xla_backend", torch_xla_backend)


def test_torch_backend_uses_xla_init_method(monkeypatch):
    _install_fake_torch_xla(monkeypatch, local_ordinal=3)
    import deepspeed.comm.torch as ds_torch

    init_calls = []
    dist_pkg = getattr(torch, 'distributed')

    class FakeAccelerator:

        @staticmethod
        def device_name():
            return 'xla'

    monkeypatch.delenv('LOCAL_RANK', raising=False)
    monkeypatch.setattr(ds_torch, "build_shm_op", lambda: None)
    monkeypatch.setattr(ds_torch, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(dist_pkg, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist_pkg,
        "init_process_group",
        lambda backend, **kwargs: init_calls.append((backend, kwargs)),
    )
    monkeypatch.setattr(dist_pkg, "get_rank", lambda: 1)
    monkeypatch.setattr(dist_pkg, "get_world_size", lambda: 8)
    monkeypatch.setattr(dist_pkg, "get_backend", lambda: XLA_BACKEND)

    backend = ds_torch.TorchBackend(XLA_BACKEND, timedelta(seconds=5), None)

    assert backend.is_initialized()
    assert init_calls[0][0] == XLA_BACKEND
    assert init_calls[0][1]["init_method"] == "xla://"
    assert os.environ["LOCAL_RANK"] == "3"
    assert os.environ["RANK"] == "1"
    assert os.environ["WORLD_SIZE"] == "8"


def test_init_distributed_skips_mpi_discovery_for_xla(monkeypatch):
    import deepspeed.comm.comm as ds_comm

    calls = []

    class FakeAccelerator:

        @staticmethod
        def communication_backend_name():
            return XLA_BACKEND

    class FakeTorchBackend:

        def __init__(self, backend, timeout, init_method, rank=-1, world_size=-1):
            calls.append((backend, timeout, init_method, rank, world_size))

        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setattr(ds_comm, "cdb", None)
    monkeypatch.setattr(ds_comm, "configure", lambda deepspeed_config=None: None)
    monkeypatch.setattr(ds_comm, "init_deepspeed_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(ds_comm, "set_backend", lambda: None)
    monkeypatch.setattr(ds_comm, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(ds_comm, "TorchBackend", FakeTorchBackend)
    monkeypatch.setattr(ds_comm, "mpi_discovery",
                        lambda *args, **kwargs: pytest.fail("mpi_discovery should not run for the xla backend"))

    for env_var in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]:
        monkeypatch.delenv(env_var, raising=False)

    ds_comm.init_distributed(dist_backend=XLA_BACKEND,
                             auto_mpi_discovery=True,
                             verbose=False,
                             timeout=timedelta(seconds=5))

    assert calls == [(XLA_BACKEND, timedelta(seconds=5), None, -1, -1)]
