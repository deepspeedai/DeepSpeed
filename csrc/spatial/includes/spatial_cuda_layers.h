// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#if __CUDA_ARCH__ >= 530
#define HALF_PRECISION_AVAILABLE = 1
#endif

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif

#include <cuda.h>
#include <cuda_fp16.h>

/*********** Group Norm Kernels, Structs, and Helpers ************/

struct {
    int64_t batch_size;
    int64_t seq_len;
    int64_t channels;
} typedef ChannelsLastProblem;

void launch_opt_bias_add(__half* result,
                         const __half* activation,
                         const __half* bias,
                         const __half* other,
                         const __half* other_bias,
                         int batch_size,
                         int seq_len,
                         int channels,
                         cudaStream_t stream);
