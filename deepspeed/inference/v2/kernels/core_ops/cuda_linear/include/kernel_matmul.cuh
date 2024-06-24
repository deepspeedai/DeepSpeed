// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef DEEPSPEED_CUDA_LINEAR_KERNEL_MATMUL_CUH
#define DEEPSPEED_CUDA_LINEAR_KERNEL_MATMUL_CUH

#include "configs.h"
#include "utils_core.cuh"
#include "utils_gmem.cuh"

/*
 * C = A*B
 * A: row major with ahead-of-time layout transformation, FP6
 * B: col major, FP16
 * C: col major, FP16
 */
template <typename TilingConfig, typename OutputDataType>
__global__ void QUANT_GEMM_Kernel(const uint4* Weight1,
                                  const uint4* Weight2,
                                  const half* Scales,
                                  const half* B,
                                  OutputDataType* C,
                                  const size_t M_Global,
                                  const size_t N_Global,
                                  const size_t K_Global,
                                  int Split_K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900

#ifdef DEBUG_MODE
    assert(K_Global % TilingConfig::TILE_K == 0);
    assert(M_Global % TilingConfig::TILE_M == 0);
    assert(gridDim.y == Split_K * (M_Global / TilingConfig::TILE_M));
#endif
    extern __shared__ __align__(128)
        half smem[];  // Dynamic shared memory for FP16 A tiles， 128 Bytes aligned
    half(*smem_array)[WARP_K + PADDING_SHARED_MEM_FOR_B_8] =
        reinterpret_cast<half(*)[WARP_K + PADDING_SHARED_MEM_FOR_B_8]>(
            smem +
            (SMEM_SIZE_A1_TILE + SMEM_SIZE_A2_TILE) / 2);  // Dynamic shared memory for FP16 B tiles
    __shared__ half QuantScales[64 * TilingConfig::BLOCK_WARPS];  // static shared memory for
                                                                  // quantization scales, 64 row per
                                                                  // warp * 4 warps = 512 Bytes
    // Thread Block Mapping, considering SplitK
    const size_t BatchID = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const size_t x = blockIdx.x;  // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
    const size_t y =
        blockIdx.y %
        (M_Global / TilingConfig::TILE_M);  // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
    const size_t Tile_Start_M = y * TilingConfig::TILE_M;
    const size_t Tile_Start_N = x * TilingConfig::TILE_N;
    const size_t NumColumnToCopy = (N_Global - Tile_Start_N) < TilingConfig::TILE_N
                                       ? (N_Global - Tile_Start_N)
                                       : TilingConfig::TILE_N;
    const size_t NumBlock_K = K_Global / TilingConfig::TILE_K;
    const size_t AverageNumBlock_K = NumBlock_K / Split_K;
    const size_t ExtraNumBlock_K = NumBlock_K - AverageNumBlock_K * Split_K;
    size_t NumIter = AverageNumBlock_K;
    if (BatchID < ExtraNumBlock_K) NumIter++;
    size_t StartBlockID_K = AverageNumBlock_K * BatchID;
    if (BatchID < ExtraNumBlock_K)
        StartBlockID_K += BatchID;
    else
        StartBlockID_K += ExtraNumBlock_K;
    // Warp ID.
    const int warpId = threadIdx.x / WARP_SIZE;
    int WARP_i =
        warpId / TilingConfig::BLOCK_COL_WARPS;  // WARP_i: row number;  WARP_j: column number
    // int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    // Global Memory Address for Matrix A (Weight)
    // ///////////////////////////////////////////////////////////////////////// StartPTR for each
    // ThreadBlock(TB)
    const uint4* TB_StartGPTR_A1 =
        Weight1 + (y * TilingConfig::BLOCK_ROW_WARPS) * NumBlock_K * NUM_INT4_PER_UNIT_2BIT_FRAG;
    const uint4* TB_StartGPTR_A2 =
        Weight2 + (y * TilingConfig::BLOCK_ROW_WARPS) * NumBlock_K * NUM_INT4_PER_UNIT_4BIT_FRAG;
    // StartPTR for each WARP.
    const uint4* WARP_StartGPTR_A1 =
        TB_StartGPTR_A1 + WARP_i * NumBlock_K * NUM_INT4_PER_UNIT_2BIT_FRAG;
    const uint4* WARP_StartGPTR_A2 =
        TB_StartGPTR_A2 + WARP_i * NumBlock_K * NUM_INT4_PER_UNIT_4BIT_FRAG;
    // StartPTR for each WARP, considering SplitK
    const size_t WARP_Start_UnitID_K = StartBlockID_K;
    WARP_StartGPTR_A1 += WARP_Start_UnitID_K * NUM_INT4_PER_UNIT_2BIT_FRAG;
    WARP_StartGPTR_A2 += WARP_Start_UnitID_K * NUM_INT4_PER_UNIT_4BIT_FRAG;
    // Copying A tile from Global to Shared, using double-buffer
    // ////////////////////////////////////////////////////////// StartSPTR for each ThreadBlock
    uint32_t* AFrag_2BIT_SPTR = reinterpret_cast<uint32_t*>(smem);
    uint32_t* AFrag_4BIT_SPTR =
        AFrag_2BIT_SPTR +
        SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 4 * TilingConfig::BLOCK_WARPS *
            PIPELINE_LEVEL_GMEM;  // 8 buffers including double buffers, 12 for trible buffers
    // StartSPTR for each WARP
    AFrag_2BIT_SPTR += warpId * SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 4;
    AFrag_4BIT_SPTR += warpId * SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 4;
    // Pre-fetch of A tile
    for (int i = 0; i < PIPELINE_LEVEL_GMEM - 1; i++) {
        CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A1>(
            AFrag_2BIT_SPTR + i * SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 4 * 4, WARP_StartGPTR_A1);
        CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A2>(
            AFrag_4BIT_SPTR + i * SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 4 * 4, WARP_StartGPTR_A2);
        WARP_StartGPTR_A1 += SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 16;
        WARP_StartGPTR_A2 += SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 16;
    }
    // Global Memory Address for Matrix A (QuantScale)
    // /////////////////////////////////////////////////////////////////////
    const half* TB_StartGPTR_A_Scale = Scales + (y * TilingConfig::BLOCK_ROW_WARPS) * 64;
    const half* WARP_StartGPTR_A_Scales = TB_StartGPTR_A_Scale + WARP_i * 64;
    CopyFromGlobalToShared_Scales(QuantScales + WARP_i * 64, WARP_StartGPTR_A_Scales);
    // Copying B tile from Global to Shared, considering SplitK
    // /////////////////////////////////////////////////////////////
    const half* BTile_GPTR = B + Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K;
    for (int i = 0; i < PIPELINE_LEVEL_GMEM - 1; i++) {
        CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS>(
            smem_array + i * TilingConfig::TILE_N, BTile_GPTR, K_Global, NumColumnToCopy);
        BTile_GPTR += TilingConfig::TILE_K;
    }
    // Register Allocation for A,B, and C, Initilazed to Zeros
    // /////////////////////////////////////////////////////////////////////
    constexpr int NumRegSets_a =
        WARP_ROW_MMA_TENSORS;  // 1 set = 4 registers, containing a 16*16 MMA block
    constexpr int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS == 1)
                                     ? 1
                                     : TilingConfig::WARP_COL_MMA_TENSORS /
                                           2;  // 1 set = 4 registers, containing a 16*16 MMA block
#ifdef PIPELINE_LEVEL_SMEM
    uint32_t a[NumRegSets_a * PIPELINE_LEVEL_SMEM]
              [4];  // double/Trible buffer is used // Registers to store decompressed FP6
    uint32_t b[NumRegSets_b * PIPELINE_LEVEL_SMEM]
              [4];  // double/Triple buffer is used // Register to store FP16 B matrix (a slice)
#endif
    float c[NumRegSets_a * NumRegSets_b][REG_PER_THREAD_C_TENSOR_16_16];
    for (int i = 0; i < NumRegSets_a * NumRegSets_b; i++)
        for (int j = 0; j < REG_PER_THREAD_C_TENSOR_16_16; j++) c[i][j] = 0.0f;
    //
    cp_async_wait_all();
    __syncthreads();

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t Scales_RPTR[4];  // 4 Registers per thread for Quantization Scales
    ExtractFromSharedToReg_Scales(Scales_RPTR, QuantScales + WARP_i * 64);
#ifdef PIPELINE_LEVEL_SMEM
    // Initializing the Software Pipeline: writing registers.
    // ////////////////////////////////////////////////////////////////////////////////////////////////
    initialize_mma_slice<TilingConfig>(
        a, b, AFrag_2BIT_SPTR, AFrag_4BIT_SPTR, smem_array, Scales_RPTR);
#endif
// The outer loop.
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma unroll(1)
    for (size_t tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        // Trible-Buffer for A Tile
        uint32_t* __restrict__ read_SPTR_Frag1 =
            AFrag_2BIT_SPTR + ((tile_id_k + 0) % PIPELINE_LEVEL_GMEM) *
                                  SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 4 *
                                  4;  // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
        uint32_t* __restrict__ read_SPTR_Frag2 =
            AFrag_4BIT_SPTR + ((tile_id_k + 0) % PIPELINE_LEVEL_GMEM) *
                                  SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 4 *
                                  4;  // 2048 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
#ifdef PIPELINE_LEVEL_SMEM
        uint32_t* __restrict__ read2_SPTR_Frag1 =
            AFrag_2BIT_SPTR +
            ((tile_id_k + 1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 4 * 4;
        uint32_t* __restrict__ read2_SPTR_Frag2 =
            AFrag_4BIT_SPTR +
            ((tile_id_k + 1) % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 4 * 4;
#endif
        uint32_t* __restrict__ write_SPTR_Frag1 =
            AFrag_2BIT_SPTR + ((tile_id_k + (PIPELINE_LEVEL_GMEM - 1)) % PIPELINE_LEVEL_GMEM) *
                                  SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 4 *
                                  4;  // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
        uint32_t* __restrict__ write_SPTR_Frag2 =
            AFrag_4BIT_SPTR + ((tile_id_k + (PIPELINE_LEVEL_GMEM - 1)) % PIPELINE_LEVEL_GMEM) *
                                  SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 4 *
                                  4;  // 2048 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
        // Trible-Buffer for B Tile
        half(*__restrict__ read_SPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8] =
            smem_array + ((tile_id_k + 0) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
#ifdef PIPELINE_LEVEL_SMEM
        half(*__restrict__ read2_SPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8] =
            smem_array + ((tile_id_k + 1) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
#endif
        half(*__restrict__ write_SPTR)[WARP_K + PADDING_SHARED_MEM_FOR_B_8] =
            smem_array +
            ((tile_id_k + (PIPELINE_LEVEL_GMEM - 1)) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
        //
        bool GlobalCopy = (tile_id_k + PIPELINE_LEVEL_GMEM - 1) < NumIter;
        // Copying A tile from Global to Register, Bypassing L1, using double-buffer
        CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A1>(
            write_SPTR_Frag1, WARP_StartGPTR_A1, GlobalCopy);
        CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A2>(
            write_SPTR_Frag2, WARP_StartGPTR_A2, GlobalCopy);
        // copying B tile from GlobalMemory to SharedMemory
        CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS>(
            write_SPTR, BTile_GPTR, K_Global, NumColumnToCopy, GlobalCopy);
        cp_async_group_commit();
#ifdef PIPELINE_LEVEL_SMEM
        core_mma_slice<TilingConfig>(c,
                                     a,
                                     b,
                                     read_SPTR_Frag1,
                                     read_SPTR_Frag2,
                                     read_SPTR,
                                     Scales_RPTR,
                                     1);  // read_SPTR_Frag1, read_SPTR_Frag2 are different for each
                                          // WARP; read_SPTR is shared among WARPs
        core_mma_slice<TilingConfig>(
            c, a, b, read_SPTR_Frag1, read_SPTR_Frag2, read_SPTR, Scales_RPTR, 2);
        core_mma_slice<TilingConfig>(
            c, a, b, read_SPTR_Frag1, read_SPTR_Frag2, read_SPTR, Scales_RPTR, 3);
        // Barriers and Synchronizations
        cp_async_wait_group<PIPELINE_LEVEL_GMEM - 2>();
        __syncthreads();
        core_mma_slice<TilingConfig>(
            c, a, b, read2_SPTR_Frag1, read2_SPTR_Frag2, read2_SPTR, Scales_RPTR, 0);
        // Updating global PTRs
        WARP_StartGPTR_A1 +=
            SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
        WARP_StartGPTR_A2 +=
            SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
        BTile_GPTR += TilingConfig::TILE_K;
#else
        PipelinedCoreLoop<TilingConfig>(
            c,
            read_SPTR,
            read_SPTR_Frag1,
            read_SPTR_Frag2,
            Scales_RPTR);  // read_SPTR_Frag1, read_SPTR_Frag2 are different for each WARP;
                           // read_SPTR is shared among WARPs
        // Updating global PTRs
        WARP_StartGPTR_A1 +=
            SMEM_SIZE_IN_BYTES_PER_WARP_A1 / 16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
        WARP_StartGPTR_A2 +=
            SMEM_SIZE_IN_BYTES_PER_WARP_A2 / 16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
        BTile_GPTR += TilingConfig::TILE_K;
        // Barriers and Synchronizations
        cp_async_wait_group<PIPELINE_LEVEL_GMEM - 2>();
        __syncthreads();
#endif
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    OutputDataType* BlockGlobalPTR =
        C + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
    for (size_t i = warpId; i < NumColumnToCopy; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (size_t j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M;
             j += WARP_SIZE)  // j-th row
        {
            if constexpr (std::is_same<OutputDataType, half>::value)
                BlockGlobalPTR[j + i * M_Global] = __float2half_rn(smem_CFrag[i][j]);
            else
                BlockGlobalPTR[j + i * M_Global] = smem_CFrag[i][j];
        }

#else
    assert(("The FP6 functions are only available on Ampere GPUs.", false));
#endif
}

#endif
