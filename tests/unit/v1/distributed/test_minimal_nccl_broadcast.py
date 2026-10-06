# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Minimal two-rank NCCL broadcast diagnostic with a DeepSpeed-free child."""

from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

_CHILD_FLAG = "--run-minimal-nccl-child"
_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT_SECONDS = 60
_SUBPROCESS_TIMEOUT_SECONDS = 120
_TERMINATION_GRACE_SECONDS = 5
_EXPECTED_VALUE = 8654


def _rank_main(rank: int, rendezvous_path: str) -> None:
    import torch

    assert not any(name == "deepspeed" or name.startswith("deepspeed.") for name in sys.modules)
    cuda = getattr(torch, "cuda")
    dist = getattr(torch, "distributed")
    requested_device = rank
    os.environ.update({
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(_WORLD_SIZE),
    })
    cuda.set_device(requested_device)
    print(
        json.dumps(
            {
                "event": "before_init",
                "rank": rank,
                "requested_device": requested_device,
                "current_device": cuda.current_device(),
                "device_count": cuda.device_count(),
            },
            sort_keys=True),
        flush=True,
    )

    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{rendezvous_path}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        tensor = torch.full(
            (4, 4),
            float(_EXPECTED_VALUE if rank == 0 else -1),
            dtype=torch.float32,
            device=f"cuda:{rank}",
        ).contiguous()
        print(
            json.dumps(
                {
                    "event": "before_broadcast",
                    "rank": rank,
                    "requested_device": requested_device,
                    "current_device": cuda.current_device(),
                    "tensor_device": str(tensor.device),
                    "tensor_dtype": str(tensor.dtype),
                    "tensor_shape": list(tensor.shape),
                    "tensor_contiguous": tensor.is_contiguous(),
                },
                sort_keys=True),
            flush=True,
        )
        dist.broadcast(tensor, src=0)
        expected = torch.full((4, 4), float(_EXPECTED_VALUE), dtype=torch.float32)
        actual = tensor.cpu()
        if not torch.equal(actual, expected):
            raise AssertionError(f"rank {rank} received {actual.tolist()}, expected {_EXPECTED_VALUE}")
        print(json.dumps({"event": "broadcast_complete", "rank": rank}, sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_child() -> int:
    import torch

    assert not any(name == "deepspeed" or name.startswith("deepspeed.") for name in sys.modules)
    with tempfile.TemporaryDirectory(prefix="ds-minimal-nccl-") as directory:
        rendezvous_path = str(Path(directory) / "rendezvous")
        torch.multiprocessing.spawn(_rank_main, args=(rendezvous_path, ), nprocs=_WORLD_SIZE, join=True)
    return 0


def _signal_owned_process_group(process: subprocess.Popen[str], requested_signal: int) -> None:
    try:
        os.killpg(process.pid, requested_signal)
    except OSError:
        if process.poll() is None:
            raise


def test_two_rank_minimal_nccl_broadcast() -> None:
    command = [sys.executable, str(Path(__file__).resolve()), _CHILD_FLAG]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=_SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_owned_process_group(process, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_owned_process_group(process, signal.SIGKILL)
            output, _ = process.communicate()
        print(output, end="")
        raise AssertionError(f"minimal NCCL child exceeded {_SUBPROCESS_TIMEOUT_SECONDS}s")

    print(output, end="")
    assert process.returncode == 0, f"minimal NCCL child exited with {process.returncode}"


if __name__ == "__main__":
    if sys.argv[1:] != [_CHILD_FLAG]:
        raise SystemExit(f"expected exactly {_CHILD_FLAG}")
    raise SystemExit(_run_child())
