# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import struct
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory
from .transport import PipelineTransport


class ShmTransport(PipelineTransport):
    """Pipeline transport using shared memory for same-node CPU-CPU communication.

    Uses ``multiprocessing.shared_memory.SharedMemory`` for zero-copy tensor
    transfer. ``send()`` creates a named segment with a packed header (numel,
    dtype code, ndim, shape) followed by raw tensor bytes. ``recv()`` opens
    the segment by name, reads the header to reconstruct shape/dtype, then
    unlinks the segment.

    Args:
        name_prefix (str): Prefix for shared memory segment names.
            Default ``"deepspeed_pp"``.
    """

    _DTYPE_CODES = {
        torch.float32: 0,
        torch.float64: 1,
        torch.int32: 2,
        torch.int64: 3,
        torch.float16: 4,
        torch.bfloat16: 5,
        torch.uint8: 6,
        torch.int8: 7,
        torch.int16: 8,
        torch.bool: 9,
    }
    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_CODES.items()}
    _DTYPE_NP = {
        0: np.float32,
        1: np.float64,
        2: np.int32,
        3: np.int64,
        4: np.float16,
        5: np.float16,
        6: np.uint8,
        7: np.int8,
        8: np.int16,
        9: np.bool_,
    }

    # 8 int32 values for header: numel, dtype_code, ndim, shape[0..4]
    HEADER_SIZE = 32

    def __init__(self, name_prefix="deepspeed_pp", stage_id=0):
        self._name_prefix = f"{name_prefix}_{id(self)}"
        self._initialized = False
        self._seq = 0
        self._stage_id = stage_id
        self._pending = {}

    def send(self, tensor, dest_stage):
        if not self._initialized:
            raise RuntimeError("ShmTransport not initialized. Call initialize() first.")

        arr = tensor.cpu().detach().numpy()
        dtype_code = self._DTYPE_CODES.get(tensor.dtype, 0)
        numel = int(arr.size)
        ndim = arr.ndim
        shape = arr.shape

        header = struct.pack(
            "!iiiiiiii",
            numel,
            dtype_code,
            ndim,
            shape[0] if ndim > 0 else 0,
            shape[1] if ndim > 1 else 0,
            shape[2] if ndim > 2 else 0,
            shape[3] if ndim > 3 else 0,
            shape[4] if ndim > 4 else 0,
        )

        # NOTE: Single-process naming. Multi-process deployments need
        # an out-of-band name exchange or deterministic scheme.
        name = f"{self._name_prefix}_{self._stage_id}_{self._seq}"
        self._seq += 1

        total_size = self.HEADER_SIZE + arr.nbytes
        shm = SharedMemory(name=name, create=True, size=total_size)
        buf = np.ndarray(total_size, dtype=np.uint8, buffer=shm.buf)
        buf[:self.HEADER_SIZE] = np.frombuffer(header, dtype=np.uint8)
        buf[self.HEADER_SIZE:] = arr.ravel().view(np.uint8)
        shm.close()

        # Store name for receiver
        self._pending[dest_stage] = name

    def recv(self, tensor, src_stage):
        if not self._initialized:
            raise RuntimeError("ShmTransport not initialized. Call initialize() first.")

        # Get the name from pending dict
        name = self._pending.pop(src_stage, None)
        if name is None:
            raise RuntimeError(f"No pending receive from stage {src_stage}. "
                               "Ensure send() was called before recv().")

        shm = SharedMemory(name=name)

        header = struct.unpack("!iiiiiiii", bytes(shm.buf[:self.HEADER_SIZE]))
        numel, dtype_code, ndim = header[0], header[1], header[2]
        shape = tuple(header[3:3 + ndim])

        np_dtype = self._DTYPE_NP.get(dtype_code, np.float32)
        dtype = self._CODE_TO_DTYPE.get(dtype_code, torch.float32)

        # Copy raw bytes out of shared memory before closing it
        data_start = self.HEADER_SIZE
        data_end = data_start + numel * np.dtype(np_dtype).itemsize
        raw_bytes = bytes(shm.buf[data_start:data_end])
        shm.close()
        shm.unlink()

        arr = np.frombuffer(raw_bytes, dtype=np_dtype)
        result = torch.from_numpy(arr.reshape(shape).copy())
        return result

    def initialize(self, topology):
        if self._initialized:
            return
        self._initialized = True

    def shutdown(self):
        self._pending.clear()
        self._initialized = False
