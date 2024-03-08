// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef CONFIGS_H
#define CONFIGS_H

// #define DEBUG_MODE
#define PIPELINE_LEVEL_GMEM 2
#define PIPELINE_LEVEL_SMEM 2  // only support 2

/************************ Hardware Parameters ************************/
#define WARP_SIZE 32
#define REG_BIT_WIDTH 32
// mma: M=16 K=16 N=8
#define MMA_8 8
#define MMA_16 16
// for memory access
#define THREAD_OPT_ACCESS_BIT_WIDTH_128 128  // LDS.128, cp_async.128, ...
#define BIT_WIDTH_PER_HALF 16                // Half precision: FP16

/******************** Register Allocation For GEMM ********************/
#define REG_PER_THREAD_C_TENSOR_16_16 8  // 8 for FP32 Accumulation
/********************** Memory Padding Parameters **********************/
// Eliminating bank-conflict
#define PADDING_BYTES_16 16  // Padding 16 bytes each column
#define PADDING_SHARED_MEM_FOR_B_8 \
    8  // Padding 8 half  each column, during CopyFromGlobalToShared() for B
#define PADDING_SHARED_MEM_FOR_C_4 \
    4  // Padding 4 float each column, during StoreToSharedMemoryFromRegister() for C
/************************* WARP Tiling part-1 *************************/
#define WARP_ROW_MMA_TENSORS 4
#define WARP_M (WARP_ROW_MMA_TENSORS * MMA_16)  // 64
#define WARP_K_MMA_TENSORS 4
#define WARP_K (WARP_K_MMA_TENSORS * MMA_16)  // 64
template <int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_MMA_TENSORS_>
struct TilingConfig {
    // Depending on "n" dimension of the GEMM
    static constexpr int BLOCK_ROW_WARPS = BLOCK_ROW_WARPS_;
    static constexpr int BLOCK_COL_WARPS = BLOCK_COL_WARPS_;
    static constexpr int WARP_COL_MMA_TENSORS = WARP_COL_MMA_TENSORS_;
    /************************* WARP Tiling part-2 *************************/
    static constexpr int WARP_N = WARP_COL_MMA_TENSORS * MMA_8;
    /*************************Thread Block Tiling *************************/
    static constexpr int TILE_M = WARP_M * BLOCK_ROW_WARPS;
    static constexpr int TILE_N = MMA_8 * WARP_COL_MMA_TENSORS * BLOCK_COL_WARPS;
    static constexpr int TILE_K = WARP_K;
    /********************** #Thread per Thread Block **********************/
    static constexpr int BLOCK_WARPS = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
    /******************************* Others *******************************/
    static constexpr int SMEM_SIZE_B_TILE = TILE_N * (TILE_K + PADDING_BYTES_16) * 2 *
                                            PIPELINE_LEVEL_GMEM;  // sizeof(half)=2, doubleBuffer=2
    static constexpr int SMEM_SIZE_C_TILE =
        TILE_N * (TILE_M + PADDING_BYTES_16) * 4;  // sizeof(float)=4
};

/************************ General Config for Quant-LLM **********************/
#define WEIGHT_FRAG1_BIT_WIDTH 2
#define WEIGHT_FRAG2_BIT_WIDTH 4
#define WEIGHT_BIT_WIDTH (WEIGHT_FRAG1_BIT_WIDTH + WEIGHT_FRAG2_BIT_WIDTH)  // 6
// #define QUANT_GROUP_SIZE_DIVIDED_BY_64  4                                                   //
//  QuantGroupSize: 4*64 = 256
/*************************** 64*64 Weghts of A WARP *************************/
#define WEIGHT_PER_UNIT (WARP_M * WARP_K)  // 64*64
#define SMEM_SIZE_IN_BYTES_PER_WARP_A1          \
    (WEIGHT_PER_UNIT * WEIGHT_FRAG1_BIT_WIDTH / \
     8)  // 1024 Bytes   #doubleBuffer not takedn into consideration
#define SMEM_SIZE_IN_BYTES_PER_WARP_A2          \
    (WEIGHT_PER_UNIT * WEIGHT_FRAG2_BIT_WIDTH / \
     8)  // 2048 Bytes   #doubleBuffer not takedn into consideration
#define SMEM_SIZE_A1_TILE                 \
    (SMEM_SIZE_IN_BYTES_PER_WARP_A1 * 4 * \
     PIPELINE_LEVEL_GMEM)  // #WARP=4, #Trible-Buffer for 3-level pipeline for A = 12 KB; double
                           // buffer for 2-level pipeline A= 8  KB.
#define SMEM_SIZE_A2_TILE                 \
    (SMEM_SIZE_IN_BYTES_PER_WARP_A2 * 4 * \
     PIPELINE_LEVEL_GMEM)  // #WARP=4, #Trible-Buffer for 3-level pipeline for A = 24 KB; double
                           // buffer for 2-level pipeline A= 16 KB.
/******************** Global Memory Layout For QUANTIZED DATA ******************/
#define NUM_INT4_PER_UNIT_2BIT_FRAG (WEIGHT_PER_UNIT * WEIGHT_FRAG1_BIT_WIDTH / 128)  // 64
#define NUM_INT4_PER_UNIT_4BIT_FRAG (WEIGHT_PER_UNIT * WEIGHT_FRAG2_BIT_WIDTH / 128)  // 128
/******************** Register Allocation For QUANTIZED DATA ******************/
#define WEIGHT_PER_THREAD (WEIGHT_PER_UNIT / WARP_SIZE)                   // 128
#define REG_PER_THREAD_2BIT_FRAG (WEIGHT_PER_THREAD / REG_BIT_WIDTH * 2)  // 8
#define REG_PER_THREAD_4BIT_FRAG (WEIGHT_PER_THREAD / REG_BIT_WIDTH * 4)  // 16
/******************** Register Allocation For QUANT Scales ******************/
#define WARP_REG_QUANT_SCALE 4  // 8 rows per thread -> 8 FP16 scales -> 4 registers
#define WARP_REG_QUANT_SCALE_DISTRIBUTED \
    1  // T0-T3, T4-T7, ..., T28-T31 share the same scales, using shfl to get all the scales for
       // each thread

#endif  // CONFIGS_H
