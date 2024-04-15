// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef DEEPSPEED_CUDA_LINEAR_UTILS_CORE_CUH
#define DEEPSPEED_CUDA_LINEAR_UTILS_CORE_CUH

#include <assert.h>

#include "configs.h"
#include "ptx_mma.cuh"
#include "utils_paralleldequant.cuh"

#ifdef PIPELINE_LEVEL_SMEM
template <int NUM_INT_PER_THREAD>
__device__ __forceinline__ void CopyFromSharedToRegister_AFrag(uint32_t Reg[],
                                                               uint32_t* SPTR,
                                                               int slice_id)
{
    SPTR += slice_id * (NUM_INT_PER_THREAD * WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_INT_PER_THREAD; i++) { Reg[i] = SPTR[lane_id + i * WARP_SIZE]; }
}

template <typename TilingConfig>
__device__ __forceinline__ void initialize_mma_slice(
    uint32_t (*a)[4],
    uint32_t (*b)[4],
    uint32_t* __restrict__ A1_SPTR_read,
    uint32_t* __restrict__ A2_SPTR_read,
    half __restrict__ (*B_SPTR_read)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    uint32_t* RPTR_Scales)
{
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6
    // register per thread;
    uint32_t a_1[2];  // NO double buffer
    uint32_t a_2[4];  // NO double buffer
    CopyFromSharedToRegister_AFrag<2>(a_1, A1_SPTR_read, 0);
    CopyFromSharedToRegister_AFrag<4>(a_2, A2_SPTR_read, 0);
    Dequant_32FP6_4Way(a, a_1, a_2, RPTR_Scales);  // SIMT Dequant: dequantizing FP6 to FP16 at
                                                   // register level, dequantizing a slice each time
    B_FromSharedToReg<TilingConfig>(b, B_SPTR_read, 0);  // Loading B from shared to registers
}

template <typename TilingConfig>
__device__ __forceinline__ void core_mma_slice(
    float c[][REG_PER_THREAD_C_TENSOR_16_16],
    uint32_t (*a)[4],
    uint32_t (*b)[4],
    uint32_t* __restrict__ A1_SPTR_read,
    uint32_t* __restrict__ A2_SPTR_read,
    half __restrict__ (*B_SPTR_read)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    uint32_t* RPTR_Scales,
    int slice_id)  // writing slice[slice_id] to registers, k=0 -> slice_id=1 for prefetching
{
#ifdef DEBUG_MODE
    assert(
        (TilingConfig::WARP_COL_MMA_TENSORS == 1) ||
        (TilingConfig::WARP_COL_MMA_TENSORS % 2 ==
         0));  // if WARP_COL_MMA_TENSORS == 1, B tile in registers is padded to a 16*16 MMA block
#endif
    const int NumRegSets_a =
        WARP_ROW_MMA_TENSORS;  // 1 set = 4 registers, containing a 16*16 MMA block
    const int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS == 1)
                                 ? 1
                                 : TilingConfig::WARP_COL_MMA_TENSORS /
                                       2;  // 1 set = 4 registers, containing a 16*16 MMA block
    uint32_t(*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] =
        reinterpret_cast<uint32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(
            c);  // Registers for accumulated FP32 results

    // Setting RPTRs for double buffers
    uint32_t(*a_read)[4] = a;
    uint32_t(*a_write)[4] = a;
    uint32_t(*b_read)[4] = b;
    uint32_t(*b_write)[4] = b;
    if (slice_id % 2 == 1) {
        b_write += NumRegSets_b;
        a_write += NumRegSets_a;
    } else {
        b_read += NumRegSets_b;
        a_read += NumRegSets_a;
    }

// Reading registers and issuing core tensor core computations (a slice of A and B tile in shared
// memory)
#pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
        if (TilingConfig::WARP_COL_MMA_TENSORS == 1) {
            MMA_FP16_M16N8K16(c_uint_ptr[i], a_read[i], b_read[0]);
        } else {
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS / 2; j++) {
                MMA_FP16_M16N8K16(c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS], a_read[i], b_read[j]);
                MMA_FP16_M16N8K16(c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS] + 4,
                                  a_read[i],
                                  b_read[j] + 2);  // c+4; b+2
            }
        }
    }

    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6
    // register per thread;
    uint32_t a_1[2];  // NO double buffer
    uint32_t a_2[4];  // NO double buffer
    CopyFromSharedToRegister_AFrag<2>(a_1, A1_SPTR_read, slice_id);
    CopyFromSharedToRegister_AFrag<4>(a_2, A2_SPTR_read, slice_id);
    Dequant_32FP6_4Way(
        a_write, a_1, a_2, RPTR_Scales);  // SIMT Dequant: dequantizing FP6 to FP16 at register
                                          // level, dequantizing a slice each time
    B_FromSharedToReg<TilingConfig>(
        b_write, B_SPTR_read, slice_id);  // Loading B from shared to registers
}

#else
// Old version with naive pipeline design
template <int NUM_INT_PER_THREAD>
__device__ __forceinline__ void CopyFromSharedToRegister_AFrag(uint32_t Reg[], uint32_t* SPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_INT_PER_THREAD; i++) { Reg[i] = SPTR[lane_id + i * WARP_SIZE]; }
}
template <typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreLoop(
    float c[][REG_PER_THREAD_C_TENSOR_16_16],
    half __restrict__ (*read_SPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8],
    uint32_t* __restrict__ read_SPTR_Frag1,
    uint32_t* __restrict__ read_SPTR_Frag2,
    uint32_t* RPTR_Scales)
{
#ifdef DEBUG_MODE
    assert(
        (TilingConfig::WARP_COL_MMA_TENSORS == 1) ||
        (TilingConfig::WARP_COL_MMA_TENSORS % 2 ==
         0));  // if WARP_COL_MMA_TENSORS == 1, B tile in registers is padded to a 16*16 MMA block
#endif
    const int NumRegSets_a =
        WARP_ROW_MMA_TENSORS;  // 1 set = 4 registers, containing a 16*16 MMA block
    const int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS == 1)
                                 ? 1
                                 : TilingConfig::WARP_COL_MMA_TENSORS /
                                       2;  // 1 set = 4 registers, containing a 16*16 MMA block

    // Registers to store FP32 results
    uint32_t(*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] =
        reinterpret_cast<uint32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(c);
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6
    // register per thread;
    uint32_t a_1[2 * 2];  // double buffer is used
    uint32_t a_2[4 * 2];  // double buffer is used
    // Registers to store decompressed FP6
    uint32_t a[NumRegSets_a * 1][4];  // No double buffer
    // Register to store FP16 B matrix (a slice)
    uint32_t b[NumRegSets_b * 2][4];  // double buffer is used

    // Overlapped Smem and TC pipeline: pre-loading from shared to registers
    CopyFromSharedToRegister_AFrag<2>(a_1, read_SPTR_Frag1);
    CopyFromSharedToRegister_AFrag<4>(a_2, read_SPTR_Frag2);
    B_FromSharedToReg<TilingConfig>(b, read_SPTR, 0);

#pragma unroll
    for (int k = 0; k < WARP_K_MMA_TENSORS; k++) {
        uint32_t(*b_read)[4] = b;
        uint32_t(*b_write)[4] = b;
        uint32_t* a_1_read = a_1;
        uint32_t* a_1_write = a_1;
        uint32_t* a_2_read = a_2;
        uint32_t* a_2_write = a_2;
        if (k % 2 == 0) {
            b_write += NumRegSets_b;
            a_1_write += 2;
            a_2_write += 4;
        } else {
            b_read += NumRegSets_b;
            a_1_read += 2;
            a_2_read += 4;
        }
        // data loading
        if (k + 1 < WARP_K_MMA_TENSORS) {
            // updating SPTR for fragment1 and fragment2
            read_SPTR_Frag1 += 2 * WARP_SIZE;
            read_SPTR_Frag2 += 4 * WARP_SIZE;
            CopyFromSharedToRegister_AFrag<2>(a_1_write, read_SPTR_Frag1);
            CopyFromSharedToRegister_AFrag<4>(a_2_write, read_SPTR_Frag2);
            B_FromSharedToReg<TilingConfig>(b_write, read_SPTR, (k + 1) * MMA_16);
        }
        // SIMT Dequant + Tensor Core computations
        Dequant_32FP6_4Way(
            a, a_1_read, a_2_read, RPTR_Scales);  // Dequantizing FP6 to FP16 at register level,
                                                  // dequantizing a slice each time
#pragma unroll
        for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
            if (TilingConfig::WARP_COL_MMA_TENSORS == 1)
                MMA_FP16_M16N8K16(c_uint_ptr[i], a[i], b_read[0]);
            else {
#pragma unroll
                for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS / 2; j++) {
                    MMA_FP16_M16N8K16(c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS], a[i], b_read[j]);
                    MMA_FP16_M16N8K16(c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS] + 4,
                                      a[i],
                                      b_read[j] + 2);  // c+4; b+2
                }
            }
        }
    }
}
#endif  // #ifdef PIPELINE_LEVEL_SMEM

template <typename TilingConfig>
__device__ __forceinline__ void StoreToSharedMemoryFromRegister(
    float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4],
    float c[][REG_PER_THREAD_C_TENSOR_16_16])
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    int warp_row_offset = warpId * (MMA_16 * WARP_ROW_MMA_TENSORS);
#pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS;
             j++) {  // Dealing with one 16*8 Tensor
            int RegSetID = i + (j / 2) * WARP_ROW_MMA_TENSORS;
            int RegOffset = (j % 2) * (REG_PER_THREAD_C_TENSOR_16_16 / 2);
            int Tensor_row_offset = warp_row_offset + i * MMA_16;
            int Tensor_col_offset = j * MMA_8;
#pragma unroll
            for (int r = 0; r < REG_PER_THREAD_C_TENSOR_16_16 / 2; r++) {
                int row_offset = lane_id / 4;
                if (r >= 2) row_offset += 8;
                int col_offset = (lane_id % 4) * 2;
                if (r % 2 == 1) col_offset += 1;
                smem_CFrag[Tensor_col_offset + col_offset][Tensor_row_offset + row_offset] =
                    c[RegSetID][r + RegOffset];
            }
        }
    }
}

#endif
