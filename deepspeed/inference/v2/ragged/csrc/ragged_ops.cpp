// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "fast_host_buffer.h"

/*
Similar to doing an empty_like to replicate a Tensor on the host, but will
attempt to optimize for faster host -> accelerator copies. Since this is on the critical
path for the forward pass, this should directly improve performance.
Allocates the shadow buffers for the input_ids, batch, seq and kv_ids tensors.

Arguments:
    device_mirror: A tensor on the accelerator that should be mirrored by the host.

Returns:
    A tensor on the host of the same size and datatype optimized for fast host -> accelerator
copies.
*/
torch::Tensor allocate_fast_host_buffer(torch::Tensor device_mirror)
{
#ifdef __HIP_PLATFORM_HCC__
    auto options =
        torch::TensorOptions().device(torch::kCPU).pinned_memory(true).dtype(device_mirror.dtype());
    auto buffer = torch::empty(device_mirror.sizes(), options);
#else

    void* buffer_ptr = get_cuda_fast_buffer(device_mirror.numel() * device_mirror.element_size());

    auto options = torch::TensorOptions().device(torch::kCPU).dtype(device_mirror.dtype());
    auto buffer = torch::from_blob(buffer_ptr, device_mirror.sizes(), options);
#endif
    return buffer;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("allocate_fast_host_buffer",
          &allocate_fast_host_buffer,
          "Allocate a host mirror of an accelerator Tensor.");
}
