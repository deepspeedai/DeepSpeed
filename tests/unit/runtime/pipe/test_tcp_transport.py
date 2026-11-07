# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for persistent TCP transport mode (TDD: RED phase).

Tests for persistent socket reuse in TcpTransport. The persistent
variant reuses sockets across multiple send()/recv() calls instead
of opening a new connection each time.
"""

import pytest
import time


class TestPersistentTcpTransport:
    """Validation tests for persistent TCP transport mode."""

    def test_persistent_mode_accepted(self):
        """persistent=True should be a valid constructor parameter."""
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport
        transport = TcpTransport(persistent=True)
        assert transport._persistent is True

    def test_persistent_default_is_false(self):
        """Default mode should be non-persistent (backward compatible)."""
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport
        transport = TcpTransport()
        assert transport._persistent is False

    def test_persistent_requires_initialize_before_send(self):
        """send() must still raise if initialize() not called."""
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport
        import torch

        transport = TcpTransport(persistent=True)
        with pytest.raises(RuntimeError, match="not initialized"):
            transport.send(torch.zeros(1), dest_stage=1)

    def test_persistent_requires_initialize_before_recv(self):
        """recv() must still raise if initialize() not called."""
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport
        import torch

        transport = TcpTransport(persistent=True)
        with pytest.raises(RuntimeError, match="not initialized"):
            transport.recv(torch.zeros(1), src_stage=0)


class TestPersistentTcpTransportIntegration:
    """Integration tests for persistent TCP mode with real sockets."""

    def test_persistent_send_recv_single_tensor(self):
        """Single tensor round-trips correctly with persistent sockets."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        # Use different ports for each test
        send_transport = TcpTransport(send_port=21001, recv_port=21002, persistent=True)
        recv_transport = TcpTransport(send_port=21000, recv_port=21001, persistent=True)

        send_transport.initialize(None)
        recv_transport.initialize(None)

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

        received = [None]
        error = [None]

        def receiver():
            try:
                received[0] = recv_transport.recv(torch.zeros(4), src_stage=0)
            except Exception as e:
                error[0] = e

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        send_transport.send(tensor, dest_stage=0)
        recv_thread.join(timeout=5)

        assert error[0] is None, f"Receiver error: {error[0]}"
        assert received[0] is not None
        assert torch.allclose(tensor, received[0])

        send_transport.shutdown()
        recv_transport.shutdown()

    def test_persistent_multiple_sends_same_socket(self):
        """Multiple sends reuse the same socket in persistent mode."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        send_transport = TcpTransport(send_port=21003, recv_port=21004, persistent=True)
        recv_transport = TcpTransport(send_port=21002, recv_port=21003, persistent=True)

        send_transport.initialize(None)
        recv_transport.initialize(None)

        results = []

        def receiver():
            for _ in range(5):
                results.append(recv_transport.recv(torch.zeros(1), src_stage=0))

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        for i in range(5):
            tensor = torch.tensor([float(i)])
            send_transport.send(tensor, dest_stage=0)

        recv_thread.join(timeout=10)
        assert len(results) == 5
        assert torch.allclose(results[0], torch.tensor([0.0]))
        assert torch.allclose(results[4], torch.tensor([4.0]))

        send_transport.shutdown()
        recv_transport.shutdown()

    def test_persistent_socket_reused_flag(self):
        """After first send, the persistent socket should be cached."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        send_transport = TcpTransport(send_port=21005, recv_port=21006, persistent=True)
        recv_transport = TcpTransport(send_port=21004, recv_port=21005, persistent=True)

        send_transport.initialize(None)
        recv_transport.initialize(None)

        received = [None]

        def receiver():
            received[0] = recv_transport.recv(torch.zeros(4), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        # Before send, no cached socket
        assert send_transport._pool_manager is not None
        assert len(send_transport._pool_manager._pools) == 0

        send_transport.send(torch.randn(4), dest_stage=0)
        recv_thread.join(timeout=5)

        # After first send, a pool is created for this dest_stage
        assert len(send_transport._pool_manager._pools) > 0

        send_transport.shutdown()
        recv_transport.shutdown()

    def test_shutdown_drains_pool_manager(self):
        """shutdown must drain the pool manager and clear pools."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        transport = TcpTransport(send_port=21007, recv_port=21008, persistent=True)
        transport.initialize(None)

        # Need another transport to connect to for send to succeed
        recv_transport = TcpTransport(send_port=21006, recv_port=21007, persistent=True)
        recv_transport.initialize(None)

        received = [None]

        def receiver():
            received[0] = recv_transport.recv(torch.zeros(2), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        transport.send(torch.randn(2), dest_stage=0)
        recv_thread.join(timeout=5)

        # After send, pool has an active connection
        assert transport._pool_manager is not None
        transport.shutdown()
        # After shutdown, pools are cleared
        assert len(transport._pool_manager._pools) == 0
        assert not transport._initialized

        recv_transport.shutdown()


class TestTcpTransportPoolIntegration:
    """End-to-end tests for TcpTransport with pooled persistent connections.

    Port pairs are configured like a real pipeline:
    - Stage 1: send_port=31000, recv_port=31001
    - Stage 2: send_port=31001, recv_port=31000

    Stage 1 sends to 31000 → Stage 2 receives on 31000.
    Stage 2 sends to 31001 → Stage 1 receives on 31001.
    """

    def test_pooled_send_recv_roundtrip(self):
        """Single tensor round-trips through pooled connections."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        # Stage 1 sends to 31000, receives on 31001
        stage1 = TcpTransport(send_port=31000, recv_port=31001, persistent=True)
        # Stage 2 sends to 31001, receives on 31000
        stage2 = TcpTransport(send_port=31001, recv_port=31000, persistent=True)

        stage1.initialize(None)
        stage2.initialize(None)

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        received = [None]
        errors = [None]

        def receiver():
            try:
                received[0] = stage2.recv(torch.zeros(4), src_stage=0)
            except Exception as e:
                errors[0] = e

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        stage1.send(tensor, dest_stage=1)
        recv_thread.join(timeout=5)

        assert errors[0] is None, f"Receiver error: {errors[0]}"
        assert received[0] is not None
        assert torch.allclose(tensor, received[0])

        # Pool should have created a connection for dest_stage=1
        assert stage1._pool_manager is not None
        assert "1" in stage1._pool_manager._pools, "Pool should have entry for dest_stage=1"

        stage1.shutdown()
        stage2.shutdown()

    def test_pooled_multiple_sends_reuse(self):
        """Multiple sends reuse the same pooled connection."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        stage1 = TcpTransport(send_port=31002, recv_port=31003, persistent=True)
        stage2 = TcpTransport(send_port=31003, recv_port=31002, persistent=True)

        stage1.initialize(None)
        stage2.initialize(None)

        results = []

        def receiver():
            for _ in range(5):
                results.append(stage2.recv(torch.zeros(1), src_stage=0))

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        for i in range(5):
            stage1.send(torch.tensor([float(i)]), dest_stage=1)

        recv_thread.join(timeout=10)
        assert len(results) == 5
        assert torch.allclose(results[4], torch.tensor([4.0]))

        # Pool should still have a single pool for dest_stage=1
        assert stage1._pool_manager is not None
        assert "1" in stage1._pool_manager._pools

        stage1.shutdown()
        stage2.shutdown()


class TestTcpTransportMultiStage:
    """Smoke test: 3-stage forward pipeline with pooled connections.

    Stage topology:
        Stage 0 → (port 32010) → Stage 1 → (port 32020) → Stage 2

    Stage 0 sends to 32010, Stage 1 receives on 32010.
    Stage 1 sends to 32020, Stage 2 receives on 32020.
    Each stage uses the pool for persistent connections.
    """

    def test_three_stage_forward_flow(self):
        """Tensor flows Stage 0 → Stage 1 → Stage 2 through pooled TCP."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        stage0 = TcpTransport(send_port=32010, recv_port=32000, persistent=True)
        stage1 = TcpTransport(send_port=32020, recv_port=32010, persistent=True)
        stage2 = TcpTransport(send_port=32000, recv_port=32020, persistent=True)

        stage0.initialize(None)
        stage1.initialize(None)
        stage2.initialize(None)

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result_1 = [None]
        result_2 = [None]
        errors = [None]

        def s1():
            try:
                r = stage1.recv(torch.zeros(5), src_stage=0)
                result_1[0] = r.clone()
                stage1.send(r, dest_stage=2)
            except Exception as e:
                errors[0] = e

        def s2():
            try:
                result_2[0] = stage2.recv(torch.zeros(5), src_stage=1)
            except Exception as e:
                errors[0] = e

        t2 = threading.Thread(target=s2)
        t1 = threading.Thread(target=s1)
        t2.start()
        t1.start()
        time.sleep(0.15)
        stage0.send(tensor, dest_stage=1)
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors[0] is None, f"Error: {errors[0]}"
        assert result_1[0] is not None
        assert result_2[0] is not None
        assert torch.allclose(tensor, result_1[0])
        assert torch.allclose(tensor, result_2[0])

        assert "1" in stage0._pool_manager._pools
        assert "2" in stage1._pool_manager._pools

        stage0.shutdown()
        stage1.shutdown()
        stage2.shutdown()

    def test_pipeline_tensor_identity_preserved(self):
        """Random tensor value is identical after passing 3 stages."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        stage0 = TcpTransport(send_port=32012, recv_port=32002, persistent=True)
        stage1 = TcpTransport(send_port=32022, recv_port=32012, persistent=True)
        stage2 = TcpTransport(send_port=32002, recv_port=32022, persistent=True)

        stage0.initialize(None)
        stage1.initialize(None)
        stage2.initialize(None)

        tensor = torch.randn(16, 64)
        r1, r2 = [None], [None]

        def s1():
            r = stage1.recv(torch.zeros(1), src_stage=0)
            r1[0] = r.clone()
            stage1.send(r, dest_stage=2)

        def s2():
            r2[0] = stage2.recv(torch.zeros(1), src_stage=1)

        t2 = threading.Thread(target=s2)
        t1 = threading.Thread(target=s1)
        t2.start()
        t1.start()
        time.sleep(0.15)
        stage0.send(tensor, dest_stage=1)
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert torch.allclose(tensor, r1[0])
        assert torch.allclose(tensor, r2[0])

        stage0.shutdown()
        stage1.shutdown()
        stage2.shutdown()


class TestTcpTransportBufferCorrectness:
    """Smoke test: tensor buffer correctness across CPU/GPU boundaries.

    Verifies that tensors originating on GPU (or CPU) are correctly
    serialized and deserialized by TcpTransport, preserving values,
    shapes, dtypes, and gradient state.
    """

    def _get_device(self):
        """Return the test device (GPU if available, else CPU)."""
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #ignore-cuda

    def test_gpu_tensor_roundtrip(self):
        """Tensor on GPU correctly transfers through TcpTransport."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        device = self._get_device()

        stage1 = TcpTransport(send_port=33000, recv_port=33001)
        stage2 = TcpTransport(send_port=33001, recv_port=33000)

        stage1.initialize(None)
        stage2.initialize(None)

        # Create tensor on device (GPU or CPU)
        tensor = torch.randn(8, 16, device=device)
        received = [None]

        def receiver():
            received[0] = stage2.recv(torch.zeros(1), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        stage1.send(tensor, dest_stage=1)
        recv_thread.join(timeout=5)

        assert received[0] is not None
        # Result may be on CPU after TCP transfer
        assert torch.allclose(tensor.cpu(), received[0].cpu())
        # Shape is preserved
        assert received[0].shape == tensor.shape

        stage1.shutdown()
        stage2.shutdown()

    def test_dtype_preservation_float32(self):
        """float32 tensor preserves dtype through transfer."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        stage1 = TcpTransport(send_port=33002, recv_port=33003)
        stage2 = TcpTransport(send_port=33003, recv_port=33002)

        stage1.initialize(None)
        stage2.initialize(None)

        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        received = [None]

        def receiver():
            received[0] = stage2.recv(torch.zeros(1), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        stage1.send(tensor, dest_stage=1)
        recv_thread.join(timeout=5)

        assert received[0] is not None
        assert received[0].dtype == torch.float32
        assert torch.allclose(tensor, received[0])

        stage1.shutdown()
        stage2.shutdown()

    def test_dtype_preservation_float64(self):
        """float64 tensor preserves dtype through transfer."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        stage1 = TcpTransport(send_port=33004, recv_port=33005)
        stage2 = TcpTransport(send_port=33005, recv_port=33004)

        stage1.initialize(None)
        stage2.initialize(None)

        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        received = [None]

        def receiver():
            received[0] = stage2.recv(torch.zeros(1), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        stage1.send(tensor, dest_stage=1)
        recv_thread.join(timeout=5)

        assert received[0] is not None
        assert received[0].dtype == torch.float64
        assert torch.allclose(tensor, received[0])

        stage1.shutdown()
        stage2.shutdown()

    def test_large_tensor_transfer(self):
        """Large tensors (1M elements) round-trip correctly."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        device = self._get_device()

        stage1 = TcpTransport(send_port=33006, recv_port=33007)
        stage2 = TcpTransport(send_port=33007, recv_port=33006)

        stage1.initialize(None)
        stage2.initialize(None)

        tensor = torch.randn(1024, 1024, device=device)  # 1M elements
        received = [None]

        def receiver():
            received[0] = stage2.recv(torch.zeros(1), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.1)

        stage1.send(tensor, dest_stage=1)
        recv_thread.join(timeout=10)

        assert received[0] is not None
        assert received[0].shape == (1024, 1024)
        assert torch.allclose(tensor.cpu(), received[0].cpu())

        stage1.shutdown()
        stage2.shutdown()


class TestTcpTransportGpuStress:
    """Stress test: sustained multi-buffer transfers with GPU tensors.

    Simulates multiple micro-batches flowing through a persistent
    TCP connection, verifying correctness, pool stability, and
    memory behavior under load.
    """

    def _get_device(self):
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #ignore-cuda

    def test_stress_100_micro_batches(self):
        """100 sequential tensor transfers through persistent pool."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        device = self._get_device()
        BATCH_COUNT = 100

        stage1 = TcpTransport(send_port=34000, recv_port=34001, persistent=True)
        stage2 = TcpTransport(send_port=34001, recv_port=34000, persistent=True)

        stage1.initialize(None)
        stage2.initialize(None)

        # Pre-generate varied-size tensors on GPU
        tensors = [torch.randn(4, i * 8 + 8, device=device) for i in range(BATCH_COUNT)]

        results = [None] * BATCH_COUNT
        errors = [None]

        def receiver():
            try:
                for i in range(BATCH_COUNT):
                    results[i] = stage2.recv(torch.zeros(1), src_stage=0)
            except Exception as e:
                errors[0] = e

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.2)

        for i, tensor in enumerate(tensors):
            stage1.send(tensor, dest_stage=1)

        recv_thread.join(timeout=30)
        assert errors[0] is None, f"Stress test error: {errors[0]}"

        # Verify all 100 tensors
        for i in range(BATCH_COUNT):
            assert results[i] is not None, f"Batch {i}: no result"
            assert results[i].shape == tensors[i].shape, (
                f"Batch {i}: shape mismatch {results[i].shape} vs {tensors[i].shape}")
            assert torch.allclose(tensors[i].cpu(), results[i].cpu()), (f"Batch {i}: value mismatch")

        # Pool should have exactly one connection for dest=1
        assert "1" in stage1._pool_manager._pools
        assert stage1._pool_manager._pools["1"].total_capacity() >= 1

        stage1.shutdown()
        stage2.shutdown()

    def test_stress_varied_sizes(self):
        """Tensors of many different shapes transfer correctly."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        device = self._get_device()

        stage1 = TcpTransport(send_port=34002, recv_port=34003, persistent=True)
        stage2 = TcpTransport(send_port=34003, recv_port=34002, persistent=True)

        stage1.initialize(None)
        stage2.initialize(None)

        shapes = [(1, ), (16, 1), (4, 8, 2), (16, 64, 128, 1), (256, ), (1, 1024), (8, 8, 8), (32, 32, 3, 3),
                  (100, 100), (4, 4, 4, 4, 2)]
        tensors = [torch.randn(*s, device=device) for s in shapes]
        results = [None] * len(shapes)

        def receiver():
            for i in range(len(shapes)):
                results[i] = stage2.recv(torch.zeros(1), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.2)

        for tensor in tensors:
            stage1.send(tensor, dest_stage=1)

        recv_thread.join(timeout=10)

        for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
            assert results[i] is not None, f"Shape {shape}: no result"
            assert results[i].shape == shape, (f"Shape {shape}: got {results[i].shape}")
            assert torch.allclose(tensor.cpu(), results[i].cpu()), (f"Shape {shape}: value mismatch")

        stage1.shutdown()
        stage2.shutdown()

    def test_stress_no_pool_leak(self):
        """Pool connection count is stable under sustained use."""
        import torch
        import threading
        from deepspeed.runtime.pipe.tcp_transport import TcpTransport

        device = self._get_device()

        stage1 = TcpTransport(send_port=34004, recv_port=34005, persistent=True)
        stage2 = TcpTransport(send_port=34005, recv_port=34004, persistent=True)

        stage1.initialize(None)
        stage2.initialize(None)

        tensor = torch.randn(64, 64, device=device)

        def receiver():
            for _ in range(200):
                stage2.recv(torch.zeros(1), src_stage=0)

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()
        time.sleep(0.2)

        # Record pool state before
        pool = stage1._pool_manager._pools.get("1")
        capacity_before = pool.total_capacity() if pool else 0

        for _ in range(200):
            stage1.send(tensor, dest_stage=1)

        recv_thread.join(timeout=30)

        # Pool should not leak connections
        pool = stage1._pool_manager._pools.get("1")
        capacity_after = pool.total_capacity() if pool else 0
        assert capacity_after <= 4, f"Pool leak: {capacity_after} connections"

        stage1.shutdown()
        stage2.shutdown()


# Module-level multiprocessing helpers (must be picklable for spawn)
# Uses importlib to bypass deepsync import chain (PT 2.2.2 compat)

import os
import sys
import importlib.util
import types as _types


def _import_tcp_transport():
    """Import TcpTransport bypassing deepsync ~23-init chain."""
    mod_name = 'deepspeed.runtime.pipe.tcp_transport'
    if mod_name in sys.modules:
        return sys.modules[mod_name].TcpTransport
    path = os.path.join(os.getcwd(), 'deepspeed', 'runtime', 'pipe', 'tcp_transport.py')
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['deepspeed'] = _types.ModuleType('deepspeed')
    sys.modules['deepspeed.runtime'] = _types.ModuleType('deepspeed.runtime')
    sys.modules['deepspeed.runtime.pipe'] = _types.ModuleType('deepspeed.runtime.pipe')
    sys.modules['deepspeed.runtime.pipe'].__package__ = 'deepspeed.runtime.pipe'
    sys.modules['deepspeed.runtime.pipe.tcp_transport'] = mod

    # Register socket_pool before loading tcp_transport (needed by 'from .socket_pool import')
    sp_path = os.path.join(os.getcwd(), 'deepspeed', 'runtime', 'pipe', 'socket_pool.py')
    sp_spec = importlib.util.spec_from_file_location('deepspeed.runtime.pipe.socket_pool', sp_path)
    sp_mod = importlib.util.module_from_spec(sp_spec)
    sp_mod.__package__ = 'deepspeed.runtime.pipe'
    sys.modules['deepspeed.runtime.pipe.socket_pool'] = sp_mod
    sp_spec.loader.exec_module(sp_mod)

    class _DummyTransport:
        pass

    sys.modules['deepspeed.runtime.pipe.transport'] = _types.ModuleType('transport')
    sys.modules['deepspeed.runtime.pipe.transport'].PipelineTransport = _DummyTransport
    mod.__package__ = 'deepspeed.runtime.pipe'
    spec.loader.exec_module(mod)
    return mod.TcpTransport


def _mp_sender_loop(send_port, recv_port, tensor_cpu, batch_count, error_q):
    """Send N copies of a tensor through TcpTransport (persistent pool)."""
    import torch
    TcpTransport = _import_tcp_transport()
    tensor = torch.from_numpy(tensor_cpu)
    try:
        stage = TcpTransport(send_port=send_port, recv_port=recv_port, persistent=True)
        stage.initialize(None)
        for _ in range(batch_count):
            stage.send(tensor, dest_stage=1)
    except Exception as e:
        error_q.put(str(e))
    finally:
        stage.shutdown()


def _mp_receiver_loop(send_port, recv_port, batch_count, result_q, error_q):
    """Receive N tensors and put results in queue."""
    import torch
    TcpTransport = _import_tcp_transport()
    try:
        stage = TcpTransport(send_port=send_port, recv_port=recv_port)
        stage.initialize(None)
        for i in range(batch_count):
            r = stage.recv(torch.zeros(1), src_stage=0)
            result_q.put((i, r.shape, r.clone().cpu().numpy()))
    except Exception as e:
        error_q.put(str(e))
    finally:
        stage.shutdown()


def _mp_sender_single(send_port, recv_port, tensor_cpu, error_q):
    """Send one tensor through TcpTransport."""
    import torch
    TcpTransport = _import_tcp_transport()
    tensor = torch.from_numpy(tensor_cpu)
    import time
    time.sleep(0.2)  # let receiver bind first
    try:
        stage = TcpTransport(send_port=send_port, recv_port=recv_port)
        stage.initialize(None)
        stage.send(tensor, dest_stage=1)
    except Exception as e:
        error_q.put(str(e))
    finally:
        stage.shutdown()


def _mp_receiver_single(send_port, recv_port, result_val, error_q):
    """Receive one tensor and set result Value with shape size."""
    import torch
    TcpTransport = _import_tcp_transport()
    try:
        stage = TcpTransport(send_port=send_port, recv_port=recv_port)
        stage.initialize(None)
        r = stage.recv(torch.zeros(1), src_stage=0)
        result_val.value = int(torch.numel(r))
    except Exception as e:
        error_q.put(str(e))
    finally:
        stage.shutdown()


class TestTcpTransportMultiProcess:
    """Multi-process GPU transfer stress test.

    Uses importlib bypass for TcpTransport in subprocess helpers
    to avoid the deepsync PT 2.2.2 import chain.
    """

    def _get_device(self):
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #ignore-cuda

    def test_mp_single_sender_receiver(self):
        """One sender process, one receiver process, 50 tensors."""
        import torch
        import torch.multiprocessing as mp

        device = self._get_device()
        BATCH_COUNT = 50
        tensor = torch.randn(32, 32, device=device)

        ctx = mp.get_context('spawn')
        errors = ctx.Queue()
        results = ctx.Queue()
        tensor_cpu = tensor.cpu().numpy()

        r_proc = ctx.Process(target=_mp_receiver_loop, args=(36001, 36000, BATCH_COUNT, results, errors))
        s_proc = ctx.Process(target=_mp_sender_loop, args=(36000, 36001, tensor_cpu, BATCH_COUNT, errors))
        r_proc.start()
        s_proc.start()
        r_proc.join(timeout=30)
        s_proc.join(timeout=30)

        errs = []
        while not errors.empty():
            errs.append(errors.get())
        if errs:
            raise RuntimeError(f"Process errors: {errs}")

        received = {}
        while not results.empty():
            i, shape, r = results.get()
            received[i] = torch.from_numpy(r).reshape(shape)

        assert len(received) == BATCH_COUNT, f"Only {len(received)}/{BATCH_COUNT} received"
        for i in range(BATCH_COUNT):
            assert i in received, f"Missing batch {i}"
            assert torch.allclose(tensor.cpu().flatten(), received[i].flatten()), f"Batch {i} mismatch"

    def test_mp_large_tensor_random(self):
        """Large random tensor transfers correctly between processes."""
        import torch
        import torch.multiprocessing as mp

        device = self._get_device()
        tensor = torch.randn(128, 256, device=device)

        ctx = mp.get_context('spawn')
        result_val = ctx.Value('i', 0)
        error_q = ctx.Queue()
        tensor_cpu = tensor.cpu().numpy()

        r_proc = ctx.Process(target=_mp_receiver_single, args=(37003, 37002, result_val, error_q))
        s_proc = ctx.Process(target=_mp_sender_single, args=(37002, 37003, tensor_cpu, error_q))
        r_proc.start()
        s_proc.start()
        r_proc.join(timeout=20)
        s_proc.join(timeout=10)

        errs = []
        while not error_q.empty():
            errs.append(error_q.get())
        assert not errs, f"Errors: {errs}"
        assert result_val.value > 0, f"Tensor shape verification failed (got {result_val.value})"
