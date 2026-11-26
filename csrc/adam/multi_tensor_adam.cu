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

typedef enum : int {
    ADAM_MODE_0 = 0,  // L2 regularization mode
    ADAM_MODE_1 = 1   // Decoupled weight decay mode(AdamW)
} adamMode_t;

using MATH_T = float;

// One element's Adam update. Every FusedAdam functor calls this, so the fp32 and the
// param-remainder layouts get the same arithmetic, including the compiler's choice of fused
// multiply-adds.
__device__ __forceinline__ void adam_update(MATH_T& r_g,
                                            MATH_T& r_p,
                                            MATH_T& r_m,
                                            MATH_T& r_v,
                                            const float beta1,
                                            const float beta2,
                                            const float beta1_correction,
                                            const float beta2_correction,
                                            const float epsilon,
                                            const float lr,
                                            adamMode_t mode,
                                            const float decay)
{
    if (mode == ADAM_MODE_0) {  // L2
        r_g = r_g + (decay * r_p);
        r_m = beta1 * r_m + (1 - beta1) * r_g;
        r_v = beta2 * r_v + (1 - beta2) * r_g * r_g;
        MATH_T next_m_unbiased = r_m / beta1_correction;
        MATH_T next_v_unbiased = r_v / beta2_correction;
        MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
        MATH_T update = next_m_unbiased / denom;
        r_p = r_p - (lr * update);
    } else {  // weight decay
        r_m = beta1 * r_m + (1 - beta1) * r_g;
        r_v = beta2 * r_v + (1 - beta2) * r_g * r_g;
        MATH_T next_m_unbiased = r_m / beta1_correction;
        MATH_T next_v_unbiased = r_v / beta2_correction;
        MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
        MATH_T update = (next_m_unbiased / denom) + (decay * r_p);
        r_p = r_p - (lr * update);
    }
}

template <typename T, typename index_t>
struct AdamFunctor {
    __device__ __forceinline__ void operator()(int chunk_size,
                                               volatile int* noop_gmem,
                                               TensorListMetadata<4>& tl,
                                               const float beta1,
                                               const float beta2,
                                               const float beta1_correction,
                                               const float beta2_correction,
                                               const float epsilon,
                                               const float lr,
                                               adamMode_t mode,
                                               const float decay)
    {
        // I'd like this kernel to propagate infs/nans.
        // if(*noop_gmem == 1)
        //   return;

        index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

        // potentially use to pass in list of scalar
        // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

        index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
        index_t n = tl.sizes[tensor_loc];

        T* g = (T*)tl.addresses[0][tensor_loc];
        g += chunk_idx * chunk_size;

        T* p = (T*)tl.addresses[1][tensor_loc];
        p += chunk_idx * chunk_size;

        T* m = (T*)tl.addresses[2][tensor_loc];
        m += chunk_idx * chunk_size;

        T* v = (T*)tl.addresses[3][tensor_loc];
        v += chunk_idx * chunk_size;

        n -= chunk_idx * chunk_size;

        // see note in multi_tensor_scale_kernel.cu
        for (index_t i_start = 0; i_start < n && i_start < chunk_size;
             i_start += blockDim.x * ILP) {
            MATH_T r_g[ILP];
            MATH_T r_p[ILP];
            MATH_T r_m[ILP];
            MATH_T r_v[ILP];
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    r_g[ii] = g[i];
                    r_p[ii] = p[i];
                    r_m[ii] = m[i];
                    r_v[ii] = v[i];
                } else {
                    r_g[ii] = MATH_T(0);
                    r_p[ii] = MATH_T(0);
                    r_m[ii] = MATH_T(0);
                    r_v[ii] = MATH_T(0);
                }
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                adam_update(r_g[ii],
                            r_p[ii],
                            r_m[ii],
                            r_v[ii],
                            beta1,
                            beta2,
                            beta1_correction,
                            beta2_correction,
                            epsilon,
                            lr,
                            mode,
                            decay);
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    p[i] = r_p[ii];
                    m[i] = r_m[ii];
                    v[i] = r_v[ii];
                }
            }
        }
    }
};

void multi_tensor_adam_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const float epsilon,
                            const int step,
                            const int mode,
                            const int bias_correction,
                            const float weight_decay)
{
    using namespace at;

    // Handle bias correction mode
    float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
    if (bias_correction == 1) {
        bias_correction1 = 1 - std::pow(beta1, step);
        bias_correction2 = 1 - std::pow(beta2, step);
    }

    size_t max_size = 0;
    bool requires_64bit_indexing = false;
    for (auto it = tensor_lists.begin(); it != tensor_lists.end(); it++) {
        for (auto it2 = it->begin(); it2 != it->end(); it2++) {
            if (it2->numel() > max_size) {
                max_size = it2->numel();
                if (max_size >= INT_MAX) {
                    requires_64bit_indexing = true;
                    break;
                }
            }
        }
        if (requires_64bit_indexing) { break; }
    }

    // Assume single type across p,g,m1,m2 now
    if (requires_64bit_indexing) {
        DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(),
                                       0,
                                       "adam",
                                       multi_tensor_apply<4>((int64_t)BLOCK_SIZE,
                                                             (int64_t)chunk_size,
                                                             noop_flag,
                                                             tensor_lists,
                                                             AdamFunctor<scalar_t_0, int64_t>(),
                                                             beta1,
                                                             beta2,
                                                             bias_correction1,
                                                             bias_correction2,
                                                             epsilon,
                                                             lr,
                                                             (adamMode_t)mode,
                                                             weight_decay);)
    } else {
        DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(),
                                       0,
                                       "adam",
                                       multi_tensor_apply<4>(BLOCK_SIZE,
                                                             chunk_size,
                                                             noop_flag,
                                                             tensor_lists,
                                                             AdamFunctor<scalar_t_0, int32_t>(),
                                                             beta1,
                                                             beta2,
                                                             bias_correction1,
                                                             bias_correction2,
                                                             epsilon,
                                                             lr,
                                                             (adamMode_t)mode,
                                                             weight_decay);)
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

/*
Adam on fp32 master weights stored as a bf16 tensor (the upper 16 bits, rounded to nearest with ties
away from zero) plus an int16 remainder (the lower 16 bits). The fp32 value is rebuilt exactly in
registers, updated, and split back, so no fp32 copy of the weights exists in memory. Same layout as
Transformer Engine's FusedAdam with store_param_remainders (NVIDIA/TransformerEngine,
transformer_engine/common/multi_tensor/adam.cu, AdamFunctorMasterParamRemainder), and as
deepspeed/utils/param_remainders.py.
*/
template <typename GRAD_T, typename index_t>
struct AdamParamRemainderFunctor {
    __device__ __forceinline__ void operator()(int chunk_size,
                                               volatile int* noop_gmem,
                                               TensorListMetadata<5>& tl,
                                               const float beta1,
                                               const float beta2,
                                               const float beta1_correction,
                                               const float beta2_correction,
                                               const float epsilon,
                                               const float lr,
                                               adamMode_t mode,
                                               const float decay)
    {
        index_t tensor_loc = tl.block_to_tensor[blockIdx.x];
        index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
        index_t n = tl.sizes[tensor_loc];

        GRAD_T* g = (GRAD_T*)tl.addresses[0][tensor_loc];
        g += chunk_idx * chunk_size;

        // The bf16 weights, read as their bit patterns.
        int16_t* p = (int16_t*)tl.addresses[1][tensor_loc];
        p += chunk_idx * chunk_size;

        float* m = (float*)tl.addresses[2][tensor_loc];
        m += chunk_idx * chunk_size;

        float* v = (float*)tl.addresses[3][tensor_loc];
        v += chunk_idx * chunk_size;

        int16_t* p_remainder = (int16_t*)tl.addresses[4][tensor_loc];
        p_remainder += chunk_idx * chunk_size;

        n -= chunk_idx * chunk_size;

        for (index_t i_start = 0; i_start < n && i_start < chunk_size;
             i_start += blockDim.x * ILP) {
            union fp32_bits {
                float fp32;
                int16_t int16[2];  // little endian: [0] is the lower half, [1] the upper half
            };
            fp32_bits master[ILP];
            MATH_T r_g[ILP];
            MATH_T r_m[ILP];
            MATH_T r_v[ILP];
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                int16_t upper = 0;
                int16_t lower = 0;
                if (i < n && i < chunk_size) {
                    r_g[ii] = static_cast<MATH_T>(g[i]);
                    r_m[ii] = m[i];
                    r_v[ii] = v[i];
                    upper = p[i];
                    lower = p_remainder[i];
                } else {
                    r_g[ii] = MATH_T(0);
                    r_m[ii] = MATH_T(0);
                    r_v[ii] = MATH_T(0);
                }
                // A negative remainder means the bf16 half was rounded up; undo that to rebuild the
                // fp32 value.
                if (lower < 0) upper--;
                master[ii].int16[1] = upper;
                master[ii].int16[0] = lower;
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                MATH_T r_p = master[ii].fp32;
                adam_update(r_g[ii],
                            r_p,
                            r_m[ii],
                            r_v[ii],
                            beta1,
                            beta2,
                            beta1_correction,
                            beta2_correction,
                            epsilon,
                            lr,
                            mode,
                            decay);
                master[ii].fp32 = r_p;
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                int16_t upper = master[ii].int16[1];
                int16_t lower = master[ii].int16[0];
                // Round the bf16 half to nearest: up when the lower half is 0x8000 or more.
                if (lower < 0) upper++;
                // Rounding up can carry a NaN's bit pattern into a zero or an infinity; keep the
                // weight a NaN.
                if (isnan(master[ii].fp32)) {
                    upper = 0x7FC0;
                    lower = 0;
                }
                if (i < n && i < chunk_size) {
                    p[i] = upper;
                    p_remainder[i] = lower;
                    m[i] = r_m[ii];
                    v[i] = r_v[ii];
                }
            }
        }
    }
};

void multi_tensor_adam_param_remainder_cuda(int chunk_size,
                                            at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr,
                                            const float beta1,
                                            const float beta2,
                                            const float epsilon,
                                            const int step,
                                            const int mode,
                                            const int bias_correction,
                                            const float weight_decay)
{
    using namespace at;

    // Lists: gradients, bf16 weights, exp_avg, exp_avg_sq, remainders.
    TORCH_CHECK(tensor_lists.size() == 5, "expected 5 tensor lists, got ", tensor_lists.size());
    for (size_t j = 0; j < tensor_lists[1].size(); j++) {
        TORCH_CHECK(tensor_lists[1][j].scalar_type() == at::ScalarType::BFloat16,
                    "param remainders need bf16 weights");
        TORCH_CHECK(tensor_lists[2][j].scalar_type() == at::ScalarType::Float &&
                        tensor_lists[3][j].scalar_type() == at::ScalarType::Float,
                    "param remainders need fp32 exp_avg and exp_avg_sq");
        TORCH_CHECK(tensor_lists[4][j].scalar_type() == at::ScalarType::Short,
                    "param remainders must be int16");
        TORCH_CHECK(tensor_lists[0][j].scalar_type() == tensor_lists[0][0].scalar_type(),
                    "all gradients must have the same dtype");
    }

    float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
    if (bias_correction == 1) {
        bias_correction1 = 1 - std::pow(beta1, step);
        bias_correction2 = 1 - std::pow(beta2, step);
    }

    bool requires_64bit_indexing = false;
    for (auto it = tensor_lists.begin(); it != tensor_lists.end() && !requires_64bit_indexing;
         it++) {
        for (auto it2 = it->begin(); it2 != it->end(); it2++) {
            if (it2->numel() >= INT_MAX) {
                requires_64bit_indexing = true;
                break;
            }
        }
    }

    if (requires_64bit_indexing) {
        DISPATCH_FLOAT_AND_HALF(
            tensor_lists[0][0].scalar_type(),
            0,
            "adam_param_remainder",
            multi_tensor_apply<5>((int64_t)BLOCK_SIZE,
                                  (int64_t)chunk_size,
                                  noop_flag,
                                  tensor_lists,
                                  AdamParamRemainderFunctor<scalar_t_0, int64_t>(),
                                  beta1,
                                  beta2,
                                  bias_correction1,
                                  bias_correction2,
                                  epsilon,
                                  lr,
                                  (adamMode_t)mode,
                                  weight_decay);)
    } else {
        DISPATCH_FLOAT_AND_HALF(
            tensor_lists[0][0].scalar_type(),
            0,
            "adam_param_remainder",
            multi_tensor_apply<5>(BLOCK_SIZE,
                                  chunk_size,
                                  noop_flag,
                                  tensor_lists,
                                  AdamParamRemainderFunctor<scalar_t_0, int32_t>(),
                                  beta1,
                                  beta2,
                                  bias_correction1,
                                  bias_correction2,
                                  epsilon,
                                  lr,
                                  (adamMode_t)mode,
                                  weight_decay);)
    }

    AT_CUDA_CHECK(cudaGetLastError());
}
