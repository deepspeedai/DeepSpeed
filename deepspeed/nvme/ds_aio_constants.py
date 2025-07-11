# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

AIO_HANDLE = 'aio_handle'
AIO_BASIC = 'aio_basic'
TORCH_IO = 'torch_io'
TORCH_FAST_IO = 'torch_fastio'
VALID_ENGINES = [AIO_HANDLE, AIO_BASIC, TORCH_IO, TORCH_FAST_IO]

BUFFER = 'buffer'
BOUNCE_BUFFER = 'bounce_buffer'
NUM_BYTES = 'num_bytes'
FILE = 'file'
HANDLE = 'handle'
ELAPSED_SEC = 'elapsed_sec'
FAST_IO_BUFFER = 'fast_io_buffer'
USE_CPU_LOCKED_TENSOR = 'cpu_locked_tensor'
USE_GDS = 'gds'
