# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
import importlib
from datetime import timedelta
from types import SimpleNamespace
import weakref

import pytest

ds_comm = importlib.import_module("deepspeed.comm.comm")
torch_backend = importlib.import_module("deepspeed.comm.torch")


class FakeBackend:

    def __init__(self):
        self.called = None
        self.work = object()

    def send(self, **kwargs):
        self.called = "send"
        return self.work

    def recv(self, **kwargs):
        self.called = "recv"
        return self.work

    def isend(self, **kwargs):
        self.called = "isend"
        return self.work

    def irecv(self, **kwargs):
        self.called = "irecv"
        return self.work


class FakeTensor:

    def __init__(self, value, device_type):
        self.value = value
        self.device = SimpleNamespace(type=device_type)
        self.copy_count = 0

    def to(self, device_type):
        return FakeTensor(self.value, device_type)

    def copy_(self, other):
        self.value = other.value
        self.copy_count += 1


class FakeWork:

    def __init__(self, tensor=None, wait_result=True, successful=True, work_result=None):
        self.tensor = tensor
        self.completed = False
        self.wait_timeouts = []
        self.wait_result = wait_result
        self.successful = successful
        self.work_result = work_result
        self.result_error = None

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.completed = self.wait_result
        if self.completed and self.tensor is not None:
            self.tensor.value = 42
        return self.wait_result

    def is_completed(self):
        return self.completed

    def is_success(self):
        return self.successful

    def source_rank(self):
        return 1

    def synchronize(self):
        return None

    def result(self):
        if self.result_error is not None:
            raise self.result_error
        return [self.tensor]

    def get_future(self):
        future = torch_backend.torch.futures.Future()
        future.set_result([self.tensor])
        return future

    def get_future_result(self):
        future = torch_backend.torch.futures.Future()
        future.set_result(self.work_result)
        return future


@pytest.mark.parametrize(("operation", "peer_arg"), [("isend", "dst"), ("irecv", "src")])
def test_nonblocking_p2p_uses_nonblocking_backend(monkeypatch, operation, peer_arg):
    backend = FakeBackend()
    monkeypatch.setattr(ds_comm, "cdb", backend)
    monkeypatch.setattr(ds_comm.comms_logger, "enabled", False)

    work = getattr(ds_comm, operation)(object(), **{peer_arg: 1})

    assert backend.called == operation
    assert work is backend.work


def configure_mps_staging(monkeypatch, operation, fake_operation):
    monkeypatch.setattr(torch_backend, "_needs_cpu_staging",
                        lambda tensor: isinstance(tensor, FakeTensor) and tensor.device.type == "mps")
    torch_dist = getattr(torch_backend.torch, "distributed")
    monkeypatch.setattr(torch_dist, operation, fake_operation)
    backend = object.__new__(torch_backend.TorchBackend)
    return backend, FakeTensor(0, "mps")


def test_inherently_async_mps_isend_retains_staging_without_copy_back(monkeypatch):
    holder = {}

    def fake_isend(tensor, **kwargs):
        holder["tensor"] = weakref.ref(tensor)
        holder["work"] = FakeWork()
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "isend", fake_isend)

    work = backend.isend(device_tensor, dst=1)

    assert isinstance(work, torch_backend.StagedWork)
    assert device_tensor.copy_count == 0
    device_tensor_ref = weakref.ref(device_tensor)
    del device_tensor
    gc.collect()
    assert device_tensor_ref() is None
    assert holder["tensor"]() is not None

    timeout = timedelta(seconds=1)
    assert work.wait(timeout) is True

    assert holder["work"].wait_timeouts == [timeout]
    gc.collect()
    assert holder["tensor"]() is None


def test_inherently_async_mps_irecv_copies_back_once_after_wait(monkeypatch):
    holder = {}

    def fake_irecv(tensor, **kwargs):
        holder["work"] = FakeWork(tensor)
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)

    work = backend.irecv(device_tensor, src=1)

    assert isinstance(work, torch_backend.StagedWork)
    assert work.is_completed() is False
    assert device_tensor.copy_count == 0
    assert device_tensor.value == 0

    timeout = timedelta(seconds=1)
    assert work.wait(timeout=timeout) is True

    assert holder["work"].wait_timeouts == [timeout]
    assert work.is_completed() is True
    assert work.source_rank() == 1
    assert device_tensor.copy_count == 1
    assert device_tensor.value == 42
    assert work.result() == [device_tensor]

    assert work.wait(timeout) is True
    assert holder["work"].wait_timeouts == [timeout, timeout]
    assert device_tensor.copy_count == 1
    assert device_tensor.value == 42


def test_staged_irecv_wait_false_discards_copy_back(monkeypatch):
    holder = {}

    def fake_irecv(tensor, **kwargs):
        holder["work"] = FakeWork(tensor, wait_result=False)
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)
    work = backend.irecv(device_tensor, src=1)

    assert work.wait() is False
    holder["work"].completed = True

    assert work.is_completed() is True
    assert device_tensor.copy_count == 0


def test_staged_irecv_synchronize_waits_for_completion(monkeypatch):
    holder = {}

    def fake_irecv(tensor, **kwargs):
        holder["work"] = FakeWork(tensor)
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)
    work = backend.irecv(device_tensor, src=1)

    work.synchronize()
    assert device_tensor.copy_count == 0

    holder["work"].tensor.value = 42
    holder["work"].completed = True
    work.synchronize()
    work.synchronize()

    assert device_tensor.copy_count == 1
    assert device_tensor.value == 42


@pytest.mark.parametrize("completion_api", ["result", "get_future", "get_future_result"])
def test_staged_irecv_completion_apis_copy_back_once(monkeypatch, completion_api):
    holder = {}
    work_result = 0

    def fake_irecv(tensor, **kwargs):
        tensor.value = 42
        holder["work"] = FakeWork(tensor, work_result=work_result)
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)
    work = backend.irecv(device_tensor, src=1)
    holder["work"].completed = True

    result = getattr(work, completion_api)()
    if completion_api == "result":
        assert result == [device_tensor]
    elif completion_api == "get_future":
        assert result.wait() == [device_tensor]
    else:
        assert result.wait() == work_result

    assert device_tensor.copy_count == 1
    assert device_tensor.value == 42
    assert work.wait() is True
    assert device_tensor.copy_count == 1


@pytest.mark.parametrize("work_result", [1, 2, 100])
def test_staged_irecv_failed_future_result_discards_copy_back(monkeypatch, work_result):

    def fake_irecv(tensor, **kwargs):
        tensor.value = 42
        return FakeWork(tensor, successful=False, work_result=work_result)

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)
    result = backend.irecv(device_tensor, src=1).get_future_result().wait()

    assert result == work_result
    assert device_tensor.copy_count == 0
    assert device_tensor.value == 0


def test_staged_irecv_copy_back_disables_autograd(monkeypatch):

    class GradTensor(FakeTensor):

        def __init__(self, value, device_type, requires_grad):
            super().__init__(value, device_type)
            self.tensor = torch_backend.torch.tensor(value,
                                                     dtype=torch_backend.torch.float32,
                                                     requires_grad=requires_grad)

        def to(self, device_type):
            return GradTensor(self.value, device_type, requires_grad=False)

        def copy_(self, other):
            self.tensor.copy_(other.tensor)
            super().copy_(other)

    def fake_irecv(tensor, **kwargs):
        tensor.tensor.fill_(42)
        tensor.value = 42
        return FakeWork(tensor)

    monkeypatch.setattr(torch_backend, "_needs_cpu_staging",
                        lambda tensor: isinstance(tensor, GradTensor) and tensor.device.type == "mps")
    torch_dist = getattr(torch_backend.torch, "distributed")
    monkeypatch.setattr(torch_dist, "irecv", fake_irecv)
    backend = object.__new__(torch_backend.TorchBackend)
    device_tensor = GradTensor(0, "mps", requires_grad=True)

    assert backend.irecv(device_tensor, src=1).wait() is True
    assert device_tensor.tensor.item() == 42


def test_staged_work_polling_finishes_copy_back_once(monkeypatch):
    holder = {}

    def fake_irecv(tensor, **kwargs):
        holder["work"] = FakeWork(tensor)
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)
    work = backend.irecv(device_tensor, src=1)
    holder["work"].tensor.value = 42
    holder["work"].completed = True

    assert work.is_completed() is True
    assert work.is_completed() is True
    assert device_tensor.copy_count == 1
    assert device_tensor.value == 42


def test_staged_work_failed_completion_discards_copy_back(monkeypatch):
    holder = {}

    def fake_irecv(tensor, **kwargs):
        holder["work"] = FakeWork(tensor, successful=False)
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)
    work = backend.irecv(device_tensor, src=1)
    holder["work"].tensor.value = 42
    holder["work"].completed = True

    assert work.is_completed() is True
    work.synchronize()

    assert device_tensor.copy_count == 0
    assert device_tensor.value == 0


def test_staged_work_unsupported_result_preserves_copy_back(monkeypatch):
    holder = {}

    def fake_irecv(tensor, **kwargs):
        holder["work"] = FakeWork(tensor)
        holder["work"].result_error = RuntimeError("result is unsupported")
        return holder["work"]

    backend, device_tensor = configure_mps_staging(monkeypatch, "irecv", fake_irecv)
    work = backend.irecv(device_tensor, src=1)

    with pytest.raises(RuntimeError, match="result is unsupported"):
        work.result()

    assert work.wait() is True
    assert device_tensor.copy_count == 1
    assert device_tensor.value == 42
