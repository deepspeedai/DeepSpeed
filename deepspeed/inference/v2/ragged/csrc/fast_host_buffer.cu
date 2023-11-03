// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "ds_kernel_utils.h"
#include "fast_host_buffer.h"

void* get_cuda_fast_buffer(int64_t size)
{
    void* buffer_ptr;
    // Host allocation flags that should minimize the host -> accelerator copy latency
    unsigned int alloc_flags =
        cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined;

    cudaHostAlloc(&buffer_ptr, size, alloc_flags);
    return buffer_ptr;
}
