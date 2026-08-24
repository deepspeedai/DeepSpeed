# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from unittest.mock import MagicMock, sentinel

import pytest

import deepspeed.comm.comm as comm
from deepspeed.comm.ccl import CCLBackend


@pytest.mark.parametrize("call_kwargs", [{}, {"device_ids": [0]}])
def test_public_barrier_forwards_device_ids(monkeypatch, call_kwargs):
    backend = MagicMock()
    backend.barrier.return_value = sentinel.work
    monkeypatch.setattr(comm, "cdb", backend)
    monkeypatch.setattr(comm.comms_logger, "enabled", False)
    expected_device_ids = call_kwargs.get("device_ids")

    result = comm.barrier(**call_kwargs)

    assert result is sentinel.work
    backend.barrier.assert_called_once_with(group=None, async_op=False, device_ids=expected_device_ids)
    assert backend.barrier.call_args.kwargs["device_ids"] is expected_device_ids


@pytest.mark.parametrize("call_kwargs", [{}, {"device_ids": [0]}])
def test_ccl_barrier_accepts_device_ids(call_kwargs):
    backend = CCLBackend.__new__(CCLBackend)
    backend.run_collective = MagicMock(return_value=sentinel.work)

    result = backend.barrier(**call_kwargs)

    assert result is sentinel.work
    backend.run_collective.assert_called_once_with(name="barrier", group=None, async_op=False)
