// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// NOTE:
// This shared-memory implementation targets AArch64 CPUs.
// Minimum supported architecture is ARMv8-A with NEON (Advanced SIMD) support.
// Systems without NEON are not supported.
//
// This version processes 8 bf16/fp16 elements per iteration by using two
// float32x4_t registers (2 x 128-bit) as the intermediate fp32 representation.
// VLOAD_U16 returns uint16x8_t (128-bit load), CVT_BF16_TO_FP32 widens that to
// a float32x4x2_t (2 x float32x4_t), and CVT_FP32_TO_BF16 narrows back to
// uint16x8_t.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <cmath>

// 128 bits = 16 bytes -> fits 8 fp16/bf16 or 4 fp32 elements.
static int vector_length_in_bytes = 16;

// Convert 8 bf16 (uint16x8_t) -> 8 fp32 (2 x float32x4_t)
static inline float32x4x2_t cvt_bf16_to_fp32(const uint16x8_t input)
{
    float32x4x2_t result;
    // Split 8 bf16 elements into low/high halves
    uint16x4_t low = vget_low_u16(input);
    uint16x4_t high = vget_high_u16(input);
    // Zero-extend 16-bit to 32-bit and shift left by 16 bits
    // BF16 has the same exponent/sign bits as FP32, just missing lower mantissa bits
    result.val[0] = vreinterpretq_f32_u32(vshll_n_u16(low, 16));
    result.val[1] = vreinterpretq_f32_u32(vshll_n_u16(high, 16));
    return result;
}

// Convert 8 fp16 (float16x8_t) -> 8 fp32 (2 x float32x4_t)
static inline float32x4x2_t cvt_fp16_to_fp32(float16x8_t input)
{
    float32x4x2_t result;
    // Split 8 fp16 elements into low/high halves and convert each to fp32
    result.val[0] = vcvt_f32_f16(vget_low_f16(input));
    result.val[1] = vcvt_f32_f16(vget_high_f16(input));
    return result;
}

// Convert 8 fp32 (2 x float32x4_t) -> 8 bf16 (uint16x8_t) with RNE rounding and NaN handling
static inline uint16x8_t cvt_fp32_to_bf16(float32x4x2_t src)
{
    // Reinterpret float32 bits as uint32
    uint32x4_t u32_lo = vreinterpretq_u32_f32(src.val[0]);
    uint32x4_t u32_hi = vreinterpretq_u32_f32(src.val[1]);

    const uint32x4_t ones = vdupq_n_u32(0x1);
    const uint32x4_t vec_bias =
        vdupq_n_u32(0x7FFF);  // one less than half of the dropped bits range
    const uint16x8_t nan_bf16 = vdupq_n_u16(0xFFFF);

    // RNE for low half: lsb = (input >> 16) & 1
    uint32x4_t lsb_lo = vandq_u32(vshrq_n_u32(u32_lo, 16), ones);
    uint32x4_t bias_lo = vaddq_u32(vec_bias, lsb_lo);
    u32_lo = vaddq_u32(u32_lo, bias_lo);

    // RNE for high half
    uint32x4_t lsb_hi = vandq_u32(vshrq_n_u32(u32_hi, 16), ones);
    uint32x4_t bias_hi = vaddq_u32(vec_bias, lsb_hi);
    u32_hi = vaddq_u32(u32_hi, bias_hi);

    // Narrow: shift right 16 and narrow to 16-bit
    // vshrn_n_u32 produces the low 4 elements (uint16x4_t)
    // vshrn_high_n_u32 appends the high 4 elements to form uint16x8_t
    uint16x4_t bf16_lo = vshrn_n_u32(u32_lo, 16);
    uint16x8_t bf16 = vshrn_high_n_u32(bf16_lo, u32_hi, 16);

    // NaN handling: notnan mask is all-ones for normal numbers, all-zeros for NaN
    uint32x4_t notnan_lo = vceqq_f32(src.val[0], src.val[0]);
    uint32x4_t notnan_hi = vceqq_f32(src.val[1], src.val[1]);
    uint16x4_t mask_lo = vmovn_u32(notnan_lo);
    uint16x4_t mask_hi = vmovn_u32(notnan_hi);
    uint16x8_t mask = vcombine_u16(mask_lo, mask_hi);

    // Select bf16 where not NaN, nan_bf16 where NaN
    return vbslq_u16(mask, bf16, nan_bf16);
}

// Convert 8 fp32 (2 x float32x4_t) -> 8 fp16 (float16x8_t)
static inline float16x8_t cvt_fp32_to_fp16(float32x4x2_t input)
{
    float16x4_t lo = vcvt_f16_f32(input.val[0]);
    float16x4_t hi = vcvt_f16_f32(input.val[1]);
    return vcombine_f16(lo, hi);
}

// Add two paired fp32 vectors element-wise (8 fp32 adds using 2 NEON instructions)
static inline float32x4x2_t vadd_f32_2vl(float32x4x2_t a, float32x4x2_t b)
{
    float32x4x2_t result;
    result.val[0] = vaddq_f32(a.val[0], b.val[0]);
    result.val[1] = vaddq_f32(a.val[1], b.val[1]);
    return result;
}

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  128bit vector ==> 16 bytes. sticking to NEON 128 bit

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers);
void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers);
void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers);

void parallel_memcpy(void* to, void* from, size_t n_bytes);

// Load 128-bit (8 x uint16 or 8 x fp16) -- full Q-register load
#define VLOAD_U8(X) vld1q_u8((uint8_t*)(X))
#define VLOAD_U16(X) vld1q_u16((uint16_t*)(X))
#define VLOAD_F16(X) vld1q_f16((float16_t*)(X))
#define VLOAD_F32(X) vld1q_f32((float32_t*)(X))

// Store 128-bit (8 x uint16 or 8 x fp16) -- full Q-register store
#define VSTORE_U8(A, B) vst1q_u8((uint8_t*)(A), B)
#define VSTORE_U16(A, B) vst1q_u16((uint16_t*)(A), B)
#define VSTORE_F16(A, B) vst1q_f16((float16_t*)(A), B)
#define VSTORE_F32(A, B) vst1q_f32((float32_t*)(A), B)


// for x86 - it uses 256 bit registers to add 8 fp32 values. 8x32 = 256
// in x86, fp32 vector conversions and additions happen for 8 fp32 elements,
#define VADD_F32(A, B) vaddq_f32(A, B)


// used in bf16/fp16 reduce
// uses 512 bit registers to add 16 bf16/16 values that are converted to f32 before addition
#define VADD_F32_2VL(A, B) vadd_f32_2vl(A, B)

// vector_length_in_bytes =32 for x86, 
// so 8 fp32 elemets can be preocessed at a time.
// 16 bf16/fp16 elements can be processed at a time.

#define CVT_BF16_TO_FP32(X) cvt_bf16_to_fp32(X)
#define CVT_FP16_TO_FP32(X) cvt_fp16_to_fp32(X)
#define CVT_FP32_TO_BF16(X) cvt_fp32_to_bf16(X)
#define CVT_FP32_TO_FP16(X) cvt_fp32_to_fp16(X)
