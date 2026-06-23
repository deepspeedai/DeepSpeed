# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""mori SDMA backend, plugged into ``TorchBackend.all_gather_into_tensor``.

When the user opts in, ``deepspeed.comm`` routes ``all_gather_into_tensor``
on the WORLD process group through ``mori_cpp.AllGatherIntoTensor``
(intra-node SDMA copy on AMD MI300).  Any failure (mori missing,
non-AMD/ROCm runtime, shmem init error, oversized call, non-WORLD group)
yields ``None`` and the caller falls back to the underlying RCCL/NCCL
allgather.

User-visible controls (env vars, no ``ds_config`` field):

* ``DS_SDMA_ALLGATHER=1``              opt in to the SDMA path.  Required:
                                        even when mori is installed, the
                                        SDMA fast-path stays off unless
                                        the user sets this explicitly.
                                        When set, ``MORI_ENABLE_SDMA=1`` is
                                        auto-exported on the user's behalf
                                        so mori allocates uncached transit
                                        buffers.
* ``DS_SDMA_ALLGATHER_MAX_NUMEL=N``   override the transit buffer size in
                                       elements (default 64M = 256 MiB
                                       per-rank input, ~2 GiB output on 8
                                       ranks)
"""

import os
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

_handle = None
_dtype_map = None
_max_numel = 0
_init_attempted = False
_call_failed_warned = False


class _SdmaWork:
    """Duck-type compatible with ``torch.distributed.Work``.

    ``wait()`` issues a stream-level dependency only and does NOT block the
    CPU, mirroring RCCL ``Work.wait()`` semantics.  ZeRO-3's prefetch
    pipeline relies on the CPU staying free so the next bucket can be
    queued ahead of time while bucket N is in flight.
    """

    def __init__(self, event):
        self._event = event

    def wait(self):
        get_accelerator().current_stream().wait_event(self._event)

    def is_completed(self) -> bool:
        return self._event.query()


def _ensure_default_pg_registered():
    """Register the WORLD process group as 'default' in PyTorch's C++ GroupRegistry.

    mori's shmem layer looks up the PG by the name "default"; the standard
    DeepSpeed init path doesn't register WORLD under that label.
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


_TRUTHY = {"1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"}


def _is_enabled_by_env() -> bool:
    """User must explicitly opt in via ``DS_SDMA_ALLGATHER=1``.

    Default is off even when mori happens to be importable: mori is an
    external dependency and we don't want DeepSpeed's collective backend
    to silently change behaviour based on which extra packages are
    installed.  Keeping this opt-in also makes A/B baselines against the
    stock RCCL path trivial without having to uninstall mori.
    """
    return os.environ.get("DS_SDMA_ALLGATHER", "0") in _TRUTHY


def _resolve_max_numel(default: int) -> int:
    raw = os.environ.get("DS_SDMA_ALLGATHER_MAX_NUMEL")
    if raw is None:
        return default
    try:
        return max(int(raw), 0)
    except ValueError:
        return default


def init(max_numel: int = 64 * 1024 * 1024) -> None:
    """Best-effort, idempotent SDMA handle construction.

    Builds one ``mori_cpp.AllGatherIntoTensor`` (NCCL/RCCL-style C++
    dispatcher) sized for the largest expected per-rank shard.  All
    subsequent allgather calls reuse this handle.  Safe to call
    unconditionally: any failure leaves ``_handle`` unset and logs a
    single rank-0 info line, so callers transparently fall back to
    RCCL/NCCL.
    """
    global _handle, _dtype_map, _max_numel, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True

    is_rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    if not _is_enabled_by_env():
        # Silent no-op: SDMA stays off and dist.allgather is used.  We
        # don't log here because most users never set DS_SDMA_ALLGATHER and
        # rank-0 spam on every backend init is noise.
        return

    max_numel = _resolve_max_numel(max_numel)
    # mori's SymmMemManager only allocates the uncached transit buffers
    # required by the SDMA kernel when MORI_ENABLE_SDMA is set; setdefault
    # so users who already exported it (or want to override) win.
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")

    try:
        _ensure_default_pg_registered()
        import mori.shmem as shmem
        from mori.ccl import AllGatherIntoTensor

        shmem.shmem_torch_process_group_init("default")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        # Per-rank input transit buffer must hold the largest shard we'll
        # ever see; output buffer = npes * input.  4 B/element is the SDMA
        # kernel's uint32 lane width.
        input_bytes = max_numel * 4
        _handle = AllGatherIntoTensor(
            my_pe=my_pe,
            npes=npes,
            input_buffer_size=input_bytes,
            output_buffer_size=input_bytes * npes,
            copy_output_to_user=True,
        )
        _dtype_map = _build_dtype_map()
        _max_numel = max_numel
        if is_rank0:
            logger.info(f"SDMA allgather enabled via mori_cpp.AllGatherIntoTensor "
                        f"(max_numel={max_numel})")
    except Exception as e:
        _handle = None
        _dtype_map = None
        _max_numel = 0
        if is_rank0:
            logger.info(f"SDMA allgather unavailable ({type(e).__name__}: {e}); "
                        f"using RCCL/NCCL allgather")


def is_enabled() -> bool:
    return _handle is not None


def supports(input_tensor: torch.Tensor, group=None) -> bool:
    """Cheap pre-check used by ``TorchBackend.all_gather_into_tensor``.

    SDMA is only safe when:
        - the backend is initialised (``_handle`` set),
        - the call is on the WORLD process group (mori's shmem layer was
          bound to "default"/WORLD at init time),
        - the per-rank shard fits inside the pre-allocated transit buffer,
        - the input dtype is in ``_dtype_map``.
    """
    if _handle is None:
        return False
    if group is not None and group is not torch.distributed.group.WORLD:
        return False
    if input_tensor.numel() > _max_numel:
        return False
    if _dtype_map is None or input_tensor.dtype not in _dtype_map:
        return False
    return True


def allgather_into_tensor(input_tensor: torch.Tensor, output_tensor: torch.Tensor, group=None) -> Optional[_SdmaWork]:
    """Run one allgather_into_tensor through the SDMA handle.

    Returns an ``_SdmaWork`` (Work-compatible) on success.  Returns
    ``None`` when SDMA is not applicable for this call (uninitialised,
    non-WORLD group, dtype not supported, shard larger than the transit
    buffer) or the call fails for any reason — the caller falls back to
    ``dist.allgather_fn``.
    """
    global _call_failed_warned
    if not supports(input_tensor, group):
        return None
    try:
        stream = get_accelerator().current_stream()
        dtype = _dtype_map[input_tensor.dtype]
        ok = _handle(input_tensor.data_ptr(), output_tensor.data_ptr(), input_tensor.numel(), dtype,
                     stream.cuda_stream)
        if not ok:
            return None
        event = get_accelerator().Event()
        event.record(stream)
        return _SdmaWork(event)
    except Exception as e:
        if (not _call_failed_warned and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
            logger.warning(f"SDMA allgather failed ({e}); falling back to dist.allgather")
            _call_failed_warned = True
        return None
