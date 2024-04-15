// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef DEEPSPEED_CUDA_LINEAR_UTILS_PARALLELDEQUANT_CUH
#define DEEPSPEED_CUDA_LINEAR_UTILS_PARALLELDEQUANT_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/*
 * Input:   R1
 * Outputs: R1, R2
 * Note:    Simplified Exponent calculation is applied.
 */
__device__ __forceinline__ void FP6_FP16_Cast_4Way(u_int32_t* R1, u_int32_t* R2)
{
    *R2 = *R1 & 0x80808080;
    *R1 = *R1 >> 2;
    *R1 = *R1 & 0x1f1f1f1f;
    *R2 = *R2 | *R1;
    *R1 = *R2 & 0x9f009f00;
    *R2 = *R2 & 0x009f009f;
    *R2 = *R2 << 8;
}

/*
 * Input:   R1
 * Outputs: R1, R2
 * Note:    Simplified Exponent calculation is NOT applied.
 */
__device__ __forceinline__ void FP6_FP16_Cast_4Way_Naive(u_int32_t* R1, u_int32_t* R2)
{
    //*R2 = *R1 & 0x80808080;
    *R2 = *R1 & 0xc0c0c0c0;
    *R1 = *R1 >> 2;
    //*R1 = *R1 & 0x1f1f1f1f;
    *R1 = *R1 & 0x0f0f0f0f;
    *R2 = *R2 | *R1;
    //
    //*R1 = *R2 & 0x9f009f00;
    //*R2 = *R2 & 0x009f009f;
    *R1 = *R2 & 0xcf00cf00;
    if (!(*R1 & 0x40000000) && (*R1 & 0x0c000000)) *R1 = *R1 | 0x30000000;
    if (!(*R1 & 0x00004000) && (*R1 & 0x00000c00)) *R1 = *R1 | 0x00003000;
    *R2 = *R2 & 0x00cf00cf;
    if (!(*R2 & 0x00400000) && (*R2 & 0x000c0000)) *R2 = *R2 | 0x00300000;
    if (!(*R2 & 0x00000040) && (*R2 & 0x0000000c)) *R2 = *R2 | 0x00000030;
    //
    *R2 = *R2 << 8;
    //*R1 = 0x3c003c00;
    //*R2 = 0x3c003c00;
}

__device__ __forceinline__ u_int32_t MultScale(u_int32_t PackedFP16Pair, half Scale)
{
    half* FP16_1 = reinterpret_cast<half*>(&PackedFP16Pair);
    half* FP16_2 = FP16_1 + 1;
    uint32_t output;
    half* output_half_ptr = reinterpret_cast<half*>(&output);
    output_half_ptr[0] = __hmul(__hmul(*FP16_1, __float2half(4096.0f)), Scale);
    output_half_ptr[1] = __hmul(__hmul(*FP16_2, __float2half(4096.0f)), Scale);
    return output;
}

__device__ __forceinline__ void Dequant_32FP6_4Way(u_int32_t __restrict__ Reg[][4],
                                                   u_int32_t __restrict__* read_RPTR_Frag1,
                                                   u_int32_t __restrict__* read_RPTR_Frag2,
                                                   u_int32_t* Scales)
{
    u_int32_t* OutputRegs = reinterpret_cast<u_int32_t*>(Reg);
    u_int32_t* Frag1_PTR = read_RPTR_Frag1;
    u_int32_t* Frag2_PTR = read_RPTR_Frag2;
    half* Scale_RPTR = reinterpret_cast<half*>(Scales);
    u_int32_t Packed_FP6 = 0;
    u_int32_t tmp = 0;
// Dequantizing 32 FP6, each Loop dequantizing 4 FP6
#pragma unroll(8)
    for (int i = 0; i < 8; i++) {
        // Frag1
        Packed_FP6 = (*Frag1_PTR) & 0xc0c0c0c0;
        if (i % 4 == 3)
            Frag1_PTR++;
        else
            (*Frag1_PTR) = (*Frag1_PTR) << 2;
        // Frag2
        tmp = (*Frag2_PTR) & 0xf0f0f0f0;
        tmp = tmp >> 2;
        if (i % 2 == 1)
            Frag2_PTR++;
        else
            (*Frag2_PTR) = (*Frag2_PTR) << 4;
        // Packed_FP6
        Packed_FP6 = Packed_FP6 | tmp;
        //
        FP6_FP16_Cast_4Way(&Packed_FP6, &tmp);
        //
        *OutputRegs = MultScale(Packed_FP6, Scale_RPTR[0]);  // Muliply FP16 scales
        OutputRegs += 1;
        *OutputRegs = MultScale(tmp, Scale_RPTR[1]);  // Muliply FP16 scales
        OutputRegs += 1;
        // Updating offset for FP16 scales for every two iterations
        if (i % 2 == 1) Scale_RPTR += 2;
    }
}

/*
 *
 */
__device__ __forceinline__ void ExtractFromSharedToReg_Scales(uint32_t* Scales,
                                                              half* WARP_SPTR_Scales)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    uint32_t* SPTR_uint = reinterpret_cast<uint32_t*>(WARP_SPTR_Scales);
    uint32_t tmpReg = SPTR_uint[lane_id];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        // T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
        Scales[i] = __shfl_sync(0xffffffff, tmpReg, i, 4);
    }
}

#endif
