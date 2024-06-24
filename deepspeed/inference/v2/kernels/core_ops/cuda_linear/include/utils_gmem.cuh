// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef DEEPSPEED_CUDA_LINEAR_UTILS_GMEM_CUH
#define DEEPSPEED_CUDA_LINEAR_UTILS_GMEM_CUH

#include <assert.h>
#include "configs.h"
#include "ptx_cp.async.cuh"

/*
 * Copying A1/A2 from global memory to shared memory.
 * Usually 1024 or 2048 Bytes
 */
template <int SMEM_SIZE_IN_BYTES_PER_WARP>
__device__ __forceinline__ void CopyFromGlobalToShared_A(uint32_t* SPTR,
                                                         const uint4* GPTR,
                                                         bool pred_guard = true)
{
#ifdef DEBUG_MODE
    static_assert(SMEM_SIZE_IN_BYTES_PER_WARP / WARP_SIZE % 16 == 0);
#endif
    int lane_id = threadIdx.x % WARP_SIZE;
    half* SPTR_HALF = reinterpret_cast<half*>(SPTR);
    const half* GPTR_HALF = reinterpret_cast<const half*>(GPTR);
    SPTR_HALF += lane_id * 8;
    GPTR_HALF += lane_id * 8;
#pragma unroll
    for (int i = 0; i < SMEM_SIZE_IN_BYTES_PER_WARP / WARP_SIZE / 16; i++) {
        cp_async<16>(SPTR_HALF, GPTR_HALF, pred_guard);
        SPTR_HALF += 256;  // Forward 512 Bytes
        GPTR_HALF += 256;  // Forward 512 Bytes
    }
}

/*
 * Copying 64 Quant Scales (FP16) from global memory to shared memory.
 */
__device__ __forceinline__ void CopyFromGlobalToShared_Scales(half* SPTR_QuantScales,
                                                              const half* GPTR_A_Scales)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int Offset_Shared = lane_id * 2;
    int Offset_Global = lane_id / 4 + (lane_id % 4) * 16;
    for (int i = 0; i < 2; i++)
        SPTR_QuantScales[Offset_Shared + i] = GPTR_A_Scales[Offset_Global + i * 8];
}

/*
 * (1) Copying X  rows * 64 columns of FP16 values, originally in row    major
 * (2) Copying 64 rows * X  columns of FP16 values, originally in column major
 * 16 Bytes per thread -> 512 Bytes per WARP = 4 line per WARP = 1 line per 8 Threads
 */
template <int MaxNumOfLinesToCopy, int BLOCK_WARPS>
__device__ __forceinline__ void CopyFromGlobalToShared(
    half (*__restrict__ SharedPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    const half* GlobalPTR,
    const int GlobalStride,
    const int NumOfLinesLeft,  // To support arbitrary N dimensions.
    bool Pred = true)
{
    // static parameters: 1 Group (8 Threads) can copy 1 line (64 FP16) each time
    const int NumOfThreads = BLOCK_WARPS * WARP_SIZE;
    const int NumOfGroups = NumOfThreads / 8;
    const int MaxIteration = (MaxNumOfLinesToCopy - 1) / NumOfGroups + 1;
    // runtime variables
    const int line_id = threadIdx.x / 8;
    const int line_offset = (threadIdx.x % 8) * 8;
    // PTR for source global memory and target shared memory
    GlobalPTR += line_id * GlobalStride + line_offset;
    SharedPTR += line_id;
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        bool AsyncCopyPred = (line_id + i * NumOfGroups) < NumOfLinesLeft && Pred;
        cp_async<16>(&(*SharedPTR)[line_offset], GlobalPTR, AsyncCopyPred);
        //
        GlobalPTR += NumOfGroups * GlobalStride;
        SharedPTR += NumOfGroups;
    }
}

#endif
