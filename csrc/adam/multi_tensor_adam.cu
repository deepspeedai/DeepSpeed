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
#include <cmath>

#include "multi_tensor_apply.cuh"
#include "type_shim.h"

#define BLOCK_SIZE 512
#define ILP 4

typedef enum : int {
    ADAM_MODE_0 = 0,  // L2 regularization mode
    ADAM_MODE_1 = 1   // Decoupled weight decay mode(AdamW)
} adamMode_t;

using MATH_T = float;

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
                if (mode == ADAM_MODE_0) {  // L2
                    r_g[ii] = r_g[ii] + (decay * r_p[ii]);
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                    MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                    MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
                    MATH_T update = next_m_unbiased / denom;
                    r_p[ii] = r_p[ii] - (lr * update);
                } else {  // weight decay
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                    MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                    MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
                    MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
                    r_p[ii] = r_p[ii] - (lr * update);
                }
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

template <typename GRAD_T, typename index_t>
struct MixedPrecisionAdamFunctor {
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
                                               const float decay,
                                               const float inv_grad_scale)
    {
        index_t tensor_loc = tl.block_to_tensor[blockIdx.x];
        index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
        index_t n = tl.sizes[tensor_loc];

        GRAD_T* g = (GRAD_T*)tl.addresses[0][tensor_loc];
        g += chunk_idx * chunk_size;
        float* p = (float*)tl.addresses[1][tensor_loc];
        p += chunk_idx * chunk_size;
        float* m = (float*)tl.addresses[2][tensor_loc];
        m += chunk_idx * chunk_size;
        float* v = (float*)tl.addresses[3][tensor_loc];
        v += chunk_idx * chunk_size;
        GRAD_T* output = (GRAD_T*)tl.addresses[4][tensor_loc];
        output += chunk_idx * chunk_size;

        n -= chunk_idx * chunk_size;

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
                    r_g[ii] = MATH_T(g[i]) * inv_grad_scale;
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
                if (mode == ADAM_MODE_0) {
                    r_g[ii] = r_g[ii] + (decay * r_p[ii]);
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                    MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                    MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
                    MATH_T update = next_m_unbiased / denom;
                    r_p[ii] = r_p[ii] - (lr * update);
                } else {
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                    MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                    MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
                    MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
                    r_p[ii] = r_p[ii] - (lr * update);
                }
            }
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    p[i] = r_p[ii];
                    m[i] = r_m[ii];
                    v[i] = r_v[ii];
                    output[i] = r_p[ii];
                }
            }
        }
    }
};

void multi_tensor_adam_mixed_precision_cuda(int chunk_size,
                                            at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr,
                                            const float beta1,
                                            const float beta2,
                                            const float epsilon,
                                            const int step,
                                            const int mode,
                                            const int bias_correction,
                                            const float weight_decay,
                                            const float grad_scale)
{
    TORCH_CHECK(tensor_lists.size() == 5, "expected five tensor lists");
    TORCH_CHECK(!tensor_lists[0].empty(), "tensor lists must not be empty");
    TORCH_CHECK(std::isfinite(grad_scale) && grad_scale > 0.0f,
                "grad_scale must be finite and positive");
    const auto tensor_count = tensor_lists[0].size();
    const auto grad_type = tensor_lists[0][0].scalar_type();
    TORCH_CHECK(grad_type == at::kHalf || grad_type == at::kBFloat16,
                "gradients must be fp16 or bf16");
    const auto device = tensor_lists[0][0].device();
    TORCH_CHECK(device.is_cuda(), "tensors must be CUDA tensors");
    TORCH_CHECK(noop_flag.device() == device,
                "noop flag and tensor lists must be on the same device");

    for (const auto& list : tensor_lists) {
        TORCH_CHECK(list.size() == tensor_count, "tensor lists must have equal lengths");
    }
    for (size_t tensor_idx = 0; tensor_idx < tensor_count; ++tensor_idx) {
        const auto& grad = tensor_lists[0][tensor_idx];
        const auto& output = tensor_lists[4][tensor_idx];
        TORCH_CHECK(grad.scalar_type() == grad_type, "all gradients must have the same dtype");
        TORCH_CHECK(output.scalar_type() == grad_type, "output dtype must match gradient dtype");
        for (size_t list_idx = 1; list_idx < 4; ++list_idx) {
            TORCH_CHECK(tensor_lists[list_idx][tensor_idx].scalar_type() == at::kFloat,
                        "master parameters and moments must be fp32");
        }
        for (size_t list_idx = 0; list_idx < tensor_lists.size(); ++list_idx) {
            const auto& tensor = tensor_lists[list_idx][tensor_idx];
            TORCH_CHECK(tensor.device() == device, "all tensors must be on the same device");
            TORCH_CHECK(tensor.is_contiguous(), "all tensors must be contiguous");
            TORCH_CHECK(tensor.sizes() == grad.sizes(),
                        "corresponding tensors must have the same shape");
        }
    }

    float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
    if (bias_correction == 1) {
        bias_correction1 = 1 - std::pow(beta1, step);
        bias_correction2 = 1 - std::pow(beta2, step);
    }

    bool requires_64bit_indexing = false;
    for (const auto& tensor : tensor_lists[0]) {
        if (tensor.numel() >= INT_MAX) {
            requires_64bit_indexing = true;
            break;
        }
    }

    const float inv_grad_scale = 1.0f / grad_scale;
    if (requires_64bit_indexing) {
        DISPATCH_FLOAT_AND_HALF(
            grad_type,
            0,
            "mixed_precision_adam",
            multi_tensor_apply<5>((int64_t)BLOCK_SIZE,
                                  (int64_t)chunk_size,
                                  noop_flag,
                                  tensor_lists,
                                  MixedPrecisionAdamFunctor<scalar_t_0, int64_t>(),
                                  beta1,
                                  beta2,
                                  bias_correction1,
                                  bias_correction2,
                                  epsilon,
                                  lr,
                                  (adamMode_t)mode,
                                  weight_decay,
                                  inv_grad_scale);)
    } else {
        DISPATCH_FLOAT_AND_HALF(
            grad_type,
            0,
            "mixed_precision_adam",
            multi_tensor_apply<5>(BLOCK_SIZE,
                                  chunk_size,
                                  noop_flag,
                                  tensor_lists,
                                  MixedPrecisionAdamFunctor<scalar_t_0, int32_t>(),
                                  beta1,
                                  beta2,
                                  bias_correction1,
                                  bias_correction2,
                                  epsilon,
                                  lr,
                                  (adamMode_t)mode,
                                  weight_decay,
                                  inv_grad_scale);)
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

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
