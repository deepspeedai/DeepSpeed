# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""mori SDMA backend for the ZeRO-3 all_gather_into_tensor hot path.

Encapsulates every mori-specific import, handle construction and dtype
dispatch so ``deepspeed/runtime/zero/partition_parameters.py`` only needs
to call:

    mori.init(max_numel)                          # one-shot, idempotent
    work = mori.allgather_into_tensor(in_, out_)  # returns None on fallback

The backend silently fails (no exceptions, ``init`` leaves the handle
unset, ``allgather_into_tensor`` returns ``None``) when mori is missing,
the platform isn't AMD/ROCm, or shmem initialization fails.  Callers are
expected to fall back to ``dist.allgather_fn`` in that case.
"""

from typing import Optional

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

_handle = None
_dtype_map = None
_init_attempted = False
_call_failed_warned = False


class _SdmaWork:
    """Duck-type compatible with ``torch.distributed.Work``.

    Mirrors NCCL ``Work.wait()`` semantics: CPU-level blocking AND
    GPU-level stream dependency so the current compute stream sees
    SDMA-written data.
    """

    def __init__(self, event):
        self._event = event

    def wait(self):
        self._event.synchronize()
        get_accelerator().current_stream().wait_event(self._event)

    def is_completed(self) -> bool:
        return self._event.query()


def _ensure_default_pg_registered():
    """Register the WORLD process group as 'default' in PyTorch's C++ GroupRegistry.

    mori's shmem layer looks up the PG by name "default"; the standard
    DeepSpeed init path doesn't register it under that label.
    """
    world_group = torch.distributed.group.WORLD
    assert world_group is not None, "torch.distributed must be initialized before SDMA allgather"
    torch._C._distributed_c10d._register_process_group("default", world_group)


def _build_dtype_map():
    """torch.dtype -> mori_cpp.DataType (NCCL-style enum)."""
    from mori.ccl import DataType
    return {
        torch.uint8: DataType.Uint8,
        torch.int8: DataType.Int8,
        torch.int16: DataType.Int16,
        torch.int32: DataType.Int32,
        torch.int64: DataType.Int64,
        torch.float16: DataType.Float16,
        torch.bfloat16: DataType.BFloat16,
        torch.float32: DataType.Float32,
        torch.float64: DataType.Float64,
    }


def init(max_numel: int = 64 * 1024 * 1024) -> None:
    """Best-effort, idempotent SDMA handle construction.

    Builds one ``mori_cpp.AllGatherIntoTensor`` (NCCL/RCCL-style C++
    dispatcher) sized for the largest expected per-rank shard.  All
    subsequent allgather calls reuse this handle.

    Safe to call unconditionally: any failure (mori not installed,
    non-AMD/ROCm runtime, shmem init error, ...) leaves ``_handle``
    unset and logs a single rank-0 warning, so callers transparently
    fall back to RCCL/NCCL via ``dist.allgather_fn``.
    """
    global _handle, _dtype_map, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    try:
        _ensure_default_pg_registered()
        import mori.shmem as shmem
        from mori.ccl import AllGatherIntoTensor

        shmem.shmem_torch_process_group_init("default")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        # Per-rank input transit buffer must hold the largest shard we'll
        # ever see; output transit buffer = npes * input.  4 B/element is
        # the SDMA kernel's uint32 lane width.
        input_bytes = max_numel * 4
        _handle = AllGatherIntoTensor(
            my_pe=my_pe,
            npes=npes,
            input_buffer_size=input_bytes,
            output_buffer_size=input_bytes * npes,
            copy_output_to_user=True,
        )
        _dtype_map = _build_dtype_map()
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info("SDMA allgather enabled via mori_cpp.AllGatherIntoTensor")
    except Exception as e:
        _handle = None
        _dtype_map = None
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.warning(f"SDMA allgather unavailable ({type(e).__name__}: {e}); "
                           f"falling back to dist.allgather_fn")


def is_enabled() -> bool:
    return _handle is not None


def allgather_into_tensor(input_tensor: torch.Tensor,
                          output_tensor: torch.Tensor) -> Optional[_SdmaWork]:
    """Run one allgather_into_tensor through the SDMA handle.

    Returns an ``_SdmaWork`` (Work-compatible) on success.  Returns
    ``None`` if SDMA is disabled or the call fails for any reason — the
    caller should then fall back to ``dist.allgather_fn``.
    """
    global _call_failed_warned
    if _handle is None:
        return None
    try:
        stream = get_accelerator().current_stream()
        dtype = _dtype_map[input_tensor.dtype]
        ok = _handle(input_tensor.data_ptr(), output_tensor.data_ptr(),
                     input_tensor.numel(), dtype, stream.cuda_stream)
        if not ok:
            return None
        event = get_accelerator().Event()
        event.record(stream)
        return _SdmaWork(event)
    except Exception as e:
        if not _call_failed_warned and dist.is_initialized() and dist.get_rank() == 0:
            logger.warning(f"SDMA allgather failed ({e}); falling back to dist.allgather")
            _call_failed_warned = True
        return None
