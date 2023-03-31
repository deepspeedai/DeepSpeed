// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Copyright NVIDIA/apex
This file is adapted from fused adam in NVIDIA/apex, commit a109f85
*/

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "compat.h"

#include <assert.h>

// #include <iostream>

// This header is the one-stop shop for all your multi-tensor apply needs.

// TODO:  Kernel arg size limit may be <4KB for some other cards (ie Jetson)
constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};

template <int n>
struct TensorListMetadata {
    void* addresses[n][depth_to_max_tensors[n - 1]];
    int sizes[depth_to_max_tensors[n - 1]];
    unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
    int block_to_chunk[depth_to_max_blocks[n - 1]];  // I fear this needs to be a full int.
    int start_tensor_this_launch;
};

template <typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(int chunk_size,
                                          volatile int* noop_flag,
                                          T tl,
                                          U callable,
                                          ArgTypes... args)
{
    // Hand the chunk information to the user-supplied functor to process however it likes.
    callable(chunk_size, noop_flag, tl, args...);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(int block_size,
                        int chunk_size,
                        const at::Tensor& noop_flag,
                        const std::vector<std::vector<at::Tensor>>& tensor_lists,
                        T callable,
                        ArgTypes... args)
{
    TORCH_CHECK(tensor_lists.size() == depth, "tensor_lists.size() != depth");
    int len0 = tensor_lists[0].size();
    TORCH_CHECK(len0 > 0, "tensor_lists[0].size() is not > 0");
    auto ref_device = tensor_lists[0][0].device();
    TORCH_CHECK(ref_device.type() == at::kCUDA, "expected input to be on cuda");
    for (int l = 0; l < tensor_lists.size(); l++)  // No range-based for because I need indices
    {
        TORCH_CHECK(tensor_lists[l].size() == len0, "Size mismatch among tensor lists");
        for (int t = 0; t < tensor_lists[l].size(); t++) {
            // TODO:  Print which tensor fails.
            bool contiguous_memory = tensor_lists[l][t].is_contiguous();
#ifdef VERSION_GE_1_5
            contiguous_memory = (contiguous_memory ||
                                 tensor_lists[l][t].is_contiguous(at::MemoryFormat::ChannelsLast));
#endif
            TORCH_CHECK(contiguous_memory, "A tensor was not contiguous.");
            TORCH_CHECK(tensor_lists[l][t].device() == ref_device,
                        "A tensor was not on the same device as the first tensor");
            TORCH_CHECK(tensor_lists[l][t].numel() == tensor_lists[0][t].numel(), "Size mismatch");
        }
    }

    int ntensors = tensor_lists[0].size();

    TensorListMetadata<depth> tl;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(tensor_lists[0][0]));
    auto stream = at::cuda::getCurrentCUDAStream();

    tl.start_tensor_this_launch = 0;
    int loc_block_info = 0;
    int loc_tensor_info = 0;
    for (int t = 0; t < ntensors; t++) {
        tl.sizes[loc_tensor_info] = tensor_lists[0][t].numel();
        for (int d = 0; d < depth; d++)
            tl.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
        loc_tensor_info++;

        int chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;

        for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
            // std::cout << chunks_this_tensor << std::endl;
            tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
            tl.block_to_chunk[loc_block_info] = chunk;
            loc_block_info++;

            bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth - 1] &&
                                 chunk == chunks_this_tensor - 1);
            bool blocks_full = (loc_block_info == depth_to_max_blocks[depth - 1]);
            bool last_chunk = (t == ntensors - 1 && chunk == chunks_this_tensor - 1);
            if (tensors_full || blocks_full || last_chunk) {
                // using accscalar_t = acc_type<scalar_t, true>;
                multi_tensor_apply_kernel<<<loc_block_info, block_size, 0, stream>>>(
                    chunk_size, noop_flag.DATA_PTR<int>(), tl, callable, args...);

                AT_CUDA_CHECK(cudaGetLastError());

                // Reset.  The control flow possibilities here make my brain hurt.
                loc_block_info = 0;
                if (chunk == chunks_this_tensor - 1) {
                    // std::cout << "Hit case 1 " << cond1 << " " << cond2 << " " << cond3 <<
                    // std::endl;
                    loc_tensor_info = 0;
                    tl.start_tensor_this_launch = t + 1;
                } else {
                    // std::cout << "Hit case 2 " << cond1 << " " << cond2 << " " << cond3 <<
                    // std::endl;
                    tl.sizes[0] = tl.sizes[loc_tensor_info - 1];
                    for (int d = 0; d < depth; d++)
                        tl.addresses[d][0] = tl.addresses[d][loc_tensor_info - 1];
                    loc_tensor_info = 1;
                    tl.start_tensor_this_launch = t;
                }
            }
        }
    }
}
