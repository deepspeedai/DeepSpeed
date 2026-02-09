// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// NOTE:
// This shared-memory implementation targets AArch64 CPUs.
// Minimum supported architecture is ARMv8-A with NEON (Advanced SIMD) support.
// Systems without NEON are not supported.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <cmath>

// 128 bits = 16 bytes -> fits 8 fp16/bf16 or 4 fp32 elements.
static int vector_length_in_bytes = 16;
// When widening fp16/bf16 -> fp32, 4 elements fit in one 128-bit register.
// Using 8 would require two 128-bit registers, so limit to 4.
static constexpr int full_precision_elements_in_fixed_vector = 4;

static inline float32x4_t cvt_bf16_to_fp32(const uint16x4_t input)
{
    // Zero-extend 16-bit to 32-bit and shift left by 16 bits
    // BF16 has the same exponent/sign bits as FP32, just missing lower mantissa bits
    uint32x4_t result_32 = vshll_n_u16(input, 16);
    return vreinterpretq_f32_u32(result_32);
}

static inline float32x4_t cvt_fp16_to_fp32(float16x4_t input)
{
    // Converts 4 FP16 values to 4 FP32 values
    return vcvt_f32_f16(input);
}

// While converting fp32 to fp16, before truncating lsb, it should be rounded to nearest even and
// Converts 4 float32 -> 4 bfloat16 with round-to-nearest-even (RNE) and NaN handling
static inline uint16x4_t cvt_fp32_to_bf16(float32x4_t src)
{
    // Reinterpret float32 bits as uint32
    uint32x4_t u32 = vreinterpretq_u32_f32(src);

    const uint32x4_t ones = vdupq_n_u32(0x1);
    const uint32x4_t vec_bias =
        vdupq_n_u32(0x7FFF);  // one less than half of the dropped bits range
    const uint16x4_t nan_bf16 = vdup_n_u16(0xFFFF);

    // RNE: lsb = (input >> 16) & 1
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u32, 16), ones);

    // rounding_bias = 0x7FFF + lsb, lsb can be 0 or 1.
    uint32x4_t bias = vaddq_u32(vec_bias, lsb);

    // input += rounding_bias
    u32 = vaddq_u32(u32, bias);

    // >> 16 to get bfloat16
    // vshrq_n_u32 - keeps 32 bit width after shift
    // vshrn_n_u32 - keeps 16 bits width after shift
    uint16x4_t bf16 = vshrn_n_u32(u32, 16);

    // vmvnq_u32 is bitwise NOT
    // NaN mask: ~(src == src) -> 1 if NaN
    // for normal num, ~(src == src) -> 0
    uint32x4_t isnan = vmvnq_u32(vceqq_f32(src, src));

    // Select nan_bf16 if isnan (use 16-bit mask)
    uint16x4_t mask = vreinterpret_u16_u32(vget_low_u32(isnan));
    return vbsl_u16(mask, nan_bf16, bf16);
}

// fp32 and fp16 are IEEE formats.
// converting fp32 to fp16 is handled by vcvt_f16_f32 internally without arbitrarily truncating the
// lsb but rounds to nearest.
static inline float16x4_t cvt_fp32_to_fp16(float32x4_t input)
{
    // Converts 4 FP32 values to 4 FP16 values with rounding
    return vcvt_f16_f32(input);
}

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  128bit vector ==> 16 bytes. sticking to NEON 128 bit

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers);
void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers);
void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers);

void parallel_memcpy(void* to, void* from, size_t n_bytes);

#define VLOAD_U8(X) vld1q_u8((uint8_t*)(X))
#define VLOAD_U16(X) vld1_u16((uint16_t*)(X))
#define VLOAD_F16(X) vld1_f16((float16_t*)(X))
#define VLOAD_F32(X) vld1q_f32((float32_t*)(X))

#define VSTORE_U8(A, B) vst1q_u8((uint8_t*)(A), B)
#define VSTORE_U16(A, B) vst1_u16((uint16_t*)(A), B)
#define VSTORE_F16(A, B) vst1_f16((float16_t*)(A), B)  // fp16 supported from armv8.2-a+fp16
#define VSTORE_F32(A, B) vst1q_f32((float32_t*)(A), B)

#define VADD_F32(A, B) vaddq_f32(A, B)
#define VADD_F32_2VL(A, B) vaddq_f32(A, B)

#define CVT_BF16_TO_FP32(X) cvt_bf16_to_fp32(X)
#define CVT_FP16_TO_FP32(X) cvt_fp16_to_fp32(X)
#define CVT_FP32_TO_BF16(X) cvt_fp32_to_bf16(X)
#define CVT_FP32_TO_FP16(X) cvt_fp32_to_fp16(X)
