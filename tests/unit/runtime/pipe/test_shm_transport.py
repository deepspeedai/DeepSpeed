# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for ShmTransport shared-memory pipeline communication."""

import importlib.util
import sys
from pathlib import Path

import torch
import pytest

# ---------------------------------------------------------------------------
# Load transport and shm_transport modules directly, bypassing the full
# DeepSpeed package __init__ chain (which fails on Python 3.12 + torch 2.2).
# ---------------------------------------------------------------------------
_pipe_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "deepspeed" / "runtime" / "pipe"

_transport_spec = importlib.util.spec_from_file_location("deepspeed.runtime.pipe.transport",
                                                         _pipe_dir / "transport.py")
_transport_mod = importlib.util.module_from_spec(_transport_spec)
sys.modules["deepspeed.runtime.pipe.transport"] = _transport_mod
_transport_spec.loader.exec_module(_transport_mod)

_shm_spec = importlib.util.spec_from_file_location("deepspeed.runtime.pipe.shm_transport",
                                                   _pipe_dir / "shm_transport.py")
_shm_mod = importlib.util.module_from_spec(_shm_spec)
sys.modules["deepspeed.runtime.pipe.shm_transport"] = _shm_mod
_shm_spec.loader.exec_module(_shm_mod)

ShmTransport = _shm_mod.ShmTransport


class TestShmTransportValidation:
    """Validate ShmTransport init and error handling."""

    def test_send_before_init_raises(self):
        """send() before initialize() raises RuntimeError."""
        transport = ShmTransport()
        with pytest.raises(RuntimeError, match="not initialized"):
            transport.send(torch.zeros(1), dest_stage=1)

    def test_recv_before_init_raises(self):
        """recv() before initialize() raises RuntimeError."""
        transport = ShmTransport()
        with pytest.raises(RuntimeError, match="not initialized"):
            transport.recv(torch.zeros(1), src_stage=0)


class TestShmTransportIntegration:
    """Integration: send and recv via shared memory in a single process."""

    def _make_transport(self):
        t = ShmTransport()
        t.initialize(None)
        return t

    def test_send_recv_roundtrip_1d(self):
        """1D tensor round-trips correctly."""
        transport = self._make_transport()
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        transport.send(tensor, dest_stage=0)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert torch.allclose(tensor, received)
        assert received.dtype == torch.float32

    def test_send_recv_multidimensional(self):
        """3D tensor round-trips with correct shape."""
        transport = self._make_transport()
        tensor = torch.randn(3, 4, 8, dtype=torch.float32)
        transport.send(tensor, dest_stage=0)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert received.shape == (3, 4, 8)
        assert torch.allclose(tensor, received)

    def test_send_recv_dtype_float64(self):
        """float64 dtype is preserved."""
        transport = self._make_transport()
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        transport.send(tensor, dest_stage=0)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert received.dtype == torch.float64
        assert torch.allclose(tensor, received)

    def test_send_recv_dtype_int64(self):
        """int64 dtype is preserved."""
        transport = self._make_transport()
        tensor = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        transport.send(tensor, dest_stage=0)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert received.dtype == torch.int64
        assert torch.allclose(tensor, received)

    def test_send_recv_scalar(self):
        """Scalar (0D) tensor round-trips."""
        transport = self._make_transport()
        tensor = torch.tensor(42.0)
        transport.send(tensor, dest_stage=0)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert torch.allclose(tensor, received)

    def test_shutdown_clears_state(self):
        """shutdown sets _initialized=False."""
        transport = ShmTransport()
        transport.initialize(None)
        transport.shutdown()
        assert not transport._initialized
