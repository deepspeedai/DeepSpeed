// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include <curand_kernel.h>

#include "context.h"
#include "cublas_wrappers.h"

#define THREADS 256
#define TILE_DIM 32

#define minus_infinity -1 * std::numeric_limits<float>::infinity()

#define FINAL_MASK 0xffffffff

template <typename T>
void launch_fused_add2(T* out,
                       const T* inp1,
                       const T* inp2,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       cudaStream_t& stream);

template <typename T>
void launch_fused_add4(T* out,
                       const T* inp1,
                       const T* inp2,
                       const T* inp3,
                       const T* inp4,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       cudaStream_t& stream);

template <typename T>
void launch_fused_add3(T* out,
                       const T* inp1,
                       const T* inp2,
                       const T* inp3,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       cudaStream_t& stream);
