// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#define TILE (128 * 1024 * 1024)
#if defined(__AVX512__) or defined(__AVX256__) or defined(__NEON__)
#if defined(__NEON__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

template <typename T>
inline T readAs(const void* src)
{
    T res;
    std::memcpy(&res, src, sizeof(T));
    return res;
}
template <typename T>
inline void writeAs(void* dst, const T& val)
{
    std::memcpy(dst, &val, sizeof(T));
}

#define ROUND_DOWN(size, step) ((size) & ~((step) - 1))

#if defined(__AVX512__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_ADD(x, y) _mm512_add_ps(x, y)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_AND(x, y) _mm512_and_ps(x, y)
#define SIMD_ANDNOT(x, y) _mm512_andnot_ps(x, y)
#define SIMD_OR(x, y) _mm512_or_ps(x, y)
#define SIMD_XOR(x, y) _mm512_xor_ps(x, y)
#define SIMD_WIDTH 16

static __m512 load_16_bf16_as_f32(const void* data)
{
    __m256i a = readAs<__m256i>(data);     // use memcpy to avoid aliasing
    __m512i b = _mm512_cvtepu16_epi32(a);  // convert 8 u16 to 8 u32
    __m512i c = _mm512_slli_epi32(b, 16);  // logical shift left of all u32 by
                                           // 16 bits (representing bf16->f32)
    return readAs<__m512>(&c);             // use memcpy to avoid aliasing
}

static void store_16_f32_as_bf16_nearest(__m512 v, void* data)
{
    __m512i u32 = readAs<__m512i>(&v);

    // flow assuming non-nan:

    // uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    __m512i b = _mm512_srli_epi32(u32, 16);
    __m512i lsb_mask = _mm512_set1_epi32(0x00000001);
    __m512i c = _mm512_and_si512(b, lsb_mask);
    __m512i bias_constant = _mm512_set1_epi32(0x00007fff);
    __m512i rounding_bias = _mm512_add_epi32(c, bias_constant);

    // uint16_t res = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    __m512i d = _mm512_add_epi32(u32, rounding_bias);
    __m512i e = _mm512_srli_epi32(d, 16);
    __m256i non_nan_res = _mm512_cvtusepi32_epi16(e);

    // handle nan (exp is all 1s and mantissa != 0)
    // if ((x & 0x7fffffffU) > 0x7f800000U)
    __m512i mask_out_sign = _mm512_set1_epi32(0x7fffffff);
    __m512i non_sign_bits = _mm512_and_si512(u32, mask_out_sign);
    __m512i nan_threshold = _mm512_set1_epi32(0x7f800000);
    __mmask16 nan_mask = _mm512_cmp_epi32_mask(non_sign_bits, nan_threshold, _MM_CMPINT_GT);

    // mix in results with nans as needed
    __m256i nans = _mm256_set1_epi16(0x7fc0);
    __m256i res = _mm256_mask_mov_epi16(non_nan_res, nan_mask, nans);

    writeAs(data, res);
}
#define SIMD_LOAD_BF16(x) load_16_bf16_as_f32(x)
#define SIMD_STORE_BF16(x, d) store_16_f32_as_bf16_nearest(d, x)

#define SIMD_LOAD_FP16(x) _mm512_cvtph_ps(_mm256_castps_si256(_mm256_loadu_ps(x)))
#define SIMD_STORE_FP16(x, d) \
    _mm256_store_ps(x, _mm256_castsi256_ps(_mm512_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT)))

#define INTV __m256i
#elif defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_ADD(x, y) _mm256_add_ps(x, y)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_AND(x, y) _mm256_and_ps(x, y)
#define SIMD_ANDNOT(x, y) _mm256_andnot_ps(x, y)
#define SIMD_OR(x, y) _mm256_or_ps(x, y)
#define SIMD_XOR(x, y) _mm256_xor_ps(x, y)
#define SIMD_WIDTH 8

#define SIMD_LOAD_BF16(x) static_assert(false && "AVX256 does not support BFloat16")
#define SIMD_STORE_BF16(x, d) static_assert(false && "AVX256 does not support BFloat16")
#define SIMD_LOAD_FP16(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)x))
#define SIMD_STORE_FP16(x, d) \
    _mm_store_ps(x, _mm_castsi128_ps(_mm256_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT)))

#define INTV __m128i
#elif defined(__NEON__)
#define SIMD_STORE(a, d) vst1q_f32(a, d)
#define SIMD_LOAD(x) vld1q_f32(x)
#define SIMD_SET(x) vdupq_n_f32(x)
#define SIMD_ADD(x, y) vaddq_f32(x, y)
#define SIMD_MUL(x, y) vmulq_f32(x, y)
// vfmaq accumulates into its first operand, while the x86 macros take the addend last.
#define SIMD_FMA(x, y, c) vfmaq_f32(c, x, y)
#define SIMD_SQRT(x) vsqrtq_f32(x)
#define SIMD_DIV(x, y) vdivq_f32(x, y)
#define SIMD_AND(x, y) \
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))
// x86 andnot(x, y) computes ~x & y; NEON bic(a, b) computes a & ~b, so the operands swap.
#define SIMD_ANDNOT(x, y) \
    vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x)))
#define SIMD_OR(x, y) \
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))
#define SIMD_XOR(x, y) \
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))
#define SIMD_WIDTH 4

static inline float32x4_t load_4_bf16_as_f32(const void* data)
{
    // bf16 is the high half of an fp32 pattern, so widen and shift into place.
    uint16x4_t a = vld1_u16((const uint16_t*)data);
    uint32x4_t b = vshlq_n_u32(vmovl_u16(a), 16);
    return vreinterpretq_f32_u32(b);
}

static inline void store_4_f32_as_bf16_nearest(float32x4_t v, void* data)
{
    // Same round-to-nearest-even-with-nan-quieting flow as the AVX512 version above.
    uint32x4_t u32 = vreinterpretq_u32_f32(v);

    // uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u32, 16), vdupq_n_u32(1));
    uint32x4_t rounding_bias = vaddq_u32(lsb, vdupq_n_u32(0x7fff));
    // vaddhn keeps the high halves of the sums, which is exactly (u32 + bias) >> 16 narrowed.
    uint16x4_t non_nan_res = vaddhn_u32(u32, rounding_bias);

    // if ((x & 0x7fffffffU) > 0x7f800000U) the result is a quiet nan
    uint32x4_t non_sign_bits = vandq_u32(u32, vdupq_n_u32(0x7fffffff));
    uint16x4_t nan_mask = vmovn_u32(vcgtq_u32(non_sign_bits, vdupq_n_u32(0x7f800000)));
    uint16x4_t res = vbsl_u16(nan_mask, vdup_n_u16(0x7fc0), non_nan_res);

    vst1_u16((uint16_t*)data, res);
}
#define SIMD_LOAD_BF16(x) load_4_bf16_as_f32(x)
#define SIMD_STORE_BF16(x, d) store_4_f32_as_bf16_nearest(d, x)

#define SIMD_LOAD_FP16(x) vcvt_f32_f16(vld1_f16((const float16_t*)(x)))
#define SIMD_STORE_FP16(x, d) vst1_f16((float16_t*)(x), vcvt_f16_f32(d))

#define INTV uint16x4_t
#endif

union AVX_Data {
#if defined(__AVX512__)
    __m512 data;
#elif defined(__AVX256__)
    __m256 data;
#elif defined(__NEON__)
    float32x4_t data;
#endif
    // float data_f[16];
};

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::Half>, void> simd_store(T* dst,
                                                                                AVX_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) { SIMD_STORE_FP16((float*)(dst + width * i), src[i].data); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> simd_store(T* dst,
                                                                                    AVX_Data* src)
{
#if defined(__AVX512__) or defined(__NEON__)
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) { SIMD_STORE_BF16((float*)(dst + width * i), src[i].data); }
#else
    throw std::runtime_error("AVX512 required for BFloat16");
#endif
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_store(T* dst, AVX_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) { SIMD_STORE(dst + width * i, src[i].data); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::Half>, void> simd_load(AVX_Data* dst,
                                                                               T* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_LOAD_FP16((float*)(src + width * i)); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> simd_load(AVX_Data* dst,
                                                                                   T* src)
{
#if defined(__AVX512__) or defined(__NEON__)
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_LOAD_BF16((float*)(src + width * i)); }
#else
    throw std::runtime_error("AVX512 required for BFloat16");
#endif
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_load(AVX_Data* dst, T* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_LOAD(src + width * i); }
}

template <int span>
inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data* src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a[i].data);
    }
}
template <int span>
inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a.data);
    }
}
template <int span>
inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data* src_m_r, AVX_Data* src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r[i].data, src_a[i].data);
    }
}
template <int span>
inline void simd_sqrt(AVX_Data* dst, AVX_Data* src)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_SQRT(src[i].data); }
}
template <int span>
inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r[i].data); }
}
template <int span>
inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r[i].data); }
}
template <int span>
inline void simd_div(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_DIV(src_a_l[i].data, src_a_r[i].data); }
}
template <int span>
inline void simd_and(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_AND(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_and(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_AND(src_a_l[i].data, src_a_r[i].data); }
}
template <int span>
inline void simd_andnot(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ANDNOT(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_andnot(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_ANDNOT(src_a_l[i].data, src_a_r[i].data);
    }
}
template <int span>
inline void simd_or(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_OR(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_or(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_OR(src_a_l[i].data, src_a_r[i].data); }
}
template <int span>
inline void simd_xor(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_XOR(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_xor(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_XOR(src_a_l[i].data, src_a_r[i].data); }
}

#endif
