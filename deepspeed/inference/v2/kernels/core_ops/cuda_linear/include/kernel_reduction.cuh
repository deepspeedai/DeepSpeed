// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef DEEPSPEED_CUDA_LINEAR_KERNEL_REDUCTION_CUH
#define DEEPSPEED_CUDA_LINEAR_KERNEL_REDUCTION_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define REDUCTION_ELEMENT_PER_THREADBLOCK 256
#define HALF_PER_128BIT 8

__global__ void SplitK_Reduction(half* C,
                                 float* Reduction_Workspace,
                                 size_t M_Global,
                                 size_t N_Global,
                                 int Split_K)
{
    half* WARP_GPTR_C = C + REDUCTION_ELEMENT_PER_THREADBLOCK * blockIdx.x;
    float* WARP_GPTR_R = Reduction_Workspace + REDUCTION_ELEMENT_PER_THREADBLOCK * blockIdx.x;
    half* THREAD_GPTR_C = WARP_GPTR_C + threadIdx.x * HALF_PER_128BIT;
    float* THREAD_GPTR_R = WARP_GPTR_R + threadIdx.x * HALF_PER_128BIT;
    // Initializing Thread-Local Results
    float Results[HALF_PER_128BIT];
#pragma unroll
    for (int i = 0; i < HALF_PER_128BIT; i++) Results[i] = 0.0f;
    // Reduction
    for (int i = 0; i < Split_K; i++) {
#pragma unroll
        for (int j = 0; j < HALF_PER_128BIT; j++) Results[j] += THREAD_GPTR_R[j];
        THREAD_GPTR_R += M_Global * N_Global;
    }
// Writing to global memory
#pragma unroll
    for (int i = 0; i < HALF_PER_128BIT; i++) THREAD_GPTR_C[i] = __float2half_rn(Results[i]);
}

#endif
