# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Minimal two-rank NCCL broadcast diagnostic with a DeepSpeed-free child."""

from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

_CHILD_FLAG = "--run-minimal-nccl-child"
_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT_SECONDS = 60
_SUBPROCESS_TIMEOUT_SECONDS = 120
_EXPECTED_VALUE = 8654


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _rank_main(rank: int, master_port: int) -> None:
    import torch

    cuda = getattr(torch, "cuda")
    dist = getattr(torch, "distributed")
    requested_device = rank
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(_WORLD_SIZE),
    })
    cuda.set_device(requested_device)
    tensor = torch.tensor([_EXPECTED_VALUE if rank == 0 else -1], dtype=torch.int64, device=f"cuda:{rank}")
    print(
        json.dumps(
            {
                "event": "before_init",
                "rank": rank,
                "requested_device": requested_device,
                "current_device": cuda.current_device(),
                "tensor_device": str(tensor.device),
                "device_count": cuda.device_count(),
            },
            sort_keys=True),
        flush=True,
    )

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        dist.broadcast(tensor, src=0)
        actual = tensor.item()
        if actual != _EXPECTED_VALUE:
            raise AssertionError(f"rank {rank} received {actual}, expected {_EXPECTED_VALUE}")
        print(json.dumps({"event": "broadcast_complete", "rank": rank, "value": actual}, sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_child() -> int:
    import torch

    torch.multiprocessing.spawn(_rank_main, args=(_available_port(), ), nprocs=_WORLD_SIZE, join=True)
    return 0


def test_two_rank_minimal_nccl_broadcast() -> None:
    command = [sys.executable, str(Path(__file__).resolve()), _CHILD_FLAG]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, end="")
        raise AssertionError(f"minimal NCCL child exceeded {_SUBPROCESS_TIMEOUT_SECONDS}s") from exc

    print(result.stdout, end="")
    assert result.returncode == 0, f"minimal NCCL child exited with {result.returncode}"


if __name__ == "__main__":
    if sys.argv[1:] != [_CHILD_FLAG]:
        raise SystemExit(f"expected exactly {_CHILD_FLAG}")
    raise SystemExit(_run_child())
