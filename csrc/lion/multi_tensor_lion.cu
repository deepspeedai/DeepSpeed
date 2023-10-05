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
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "multi_tensor_apply.cuh"
#include "type_shim.h"

#define BLOCK_SIZE 512
#define ILP 4

using MATH_T = float;

template <typename T>
struct LionFunctor {
    __device__ __forceinline__ void operator()(int chunk_size,
                                               volatile int* noop_gmem,
                                               TensorListMetadata<3>& tl,
                                               const float beta1,
                                               const float beta2,
                                               const float lr,
                                               const float decay)
    {
        // I'd like this kernel to propagate infs/nans.
        // if(*noop_gmem == 1)
        //   return;

        int tensor_loc = tl.block_to_tensor[blockIdx.x];

        // potentially use to pass in list of scalar
        // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.sizes[tensor_loc];

        T* g = (T*)tl.addresses[0][tensor_loc];
        g += chunk_idx * chunk_size;

        T* p = (T*)tl.addresses[1][tensor_loc];
        p += chunk_idx * chunk_size;

        T* m = (T*)tl.addresses[2][tensor_loc];
        m += chunk_idx * chunk_size;

        n -= chunk_idx * chunk_size;

        MATH_T after_decay = 1.0f - lr * decay;

        // see note in multi_tensor_scale_kernel.cu
        for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
            MATH_T r_g[ILP];
            MATH_T r_p[ILP];
            MATH_T r_m[ILP];
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    r_g[ii] = g[i];
                    r_p[ii] = p[i];
                    r_m[ii] = m[i];
                } else {
                    r_g[ii] = MATH_T(0);
                    r_p[ii] = MATH_T(0);
                    r_m[ii] = MATH_T(0);
                }
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                MATH_T c = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                MATH_T update = c > 0 ? (-lr) : lr;
                r_p[ii] = r_p[ii] * after_decay + update;
                r_m[ii] = beta2 * r_m[ii] + (1 - beta2) * r_g[ii];
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    p[i] = r_p[ii];
                    m[i] = r_m[ii];
                }
            }
        }
    }
};

void multi_tensor_lion_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const int step,
                            const float weight_decay)
{
    using namespace at;

    // Assume single type across p,g,m1,m2 now
    DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(),
                                   0,
                                   "lion",
                                   multi_tensor_apply<3>(BLOCK_SIZE,
                                                         chunk_size,
                                                         noop_flag,
                                                         tensor_lists,
                                                         LionFunctor<scalar_t_0>(),
                                                         beta1,
                                                         beta2,
                                                         lr,
                                                         weight_decay);)

    AT_CUDA_CHECK(cudaGetLastError());
}
