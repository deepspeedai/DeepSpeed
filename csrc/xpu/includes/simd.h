// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#define TILE (128 * 1024 * 1024)
#if defined(__AVX512__) or defined(__AVX256__)

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))

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

#define SIMD_LOAD2(x, h) \
    ((h) ? _mm512_cvtph_ps(_mm256_castps_si256(_mm256_loadu_ps(x))) : _mm512_loadu_ps(x))
#define SIMD_STORE2(x, d, h)                                                                      \
    ((h) ? _mm256_store_ps(x, _mm256_castsi256_ps(_mm512_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT))) \
         : _mm512_storeu_ps(x, d))

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

#define SIMD_LOAD2(x, h) \
    ((h) ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)x)) : _mm256_loadu_ps(x))
#define SIMD_STORE2(x, d, h)                                                                \
    ((h) ? _mm_store_ps(x, _mm_castsi128_ps(_mm256_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT))) \
         : _mm256_storeu_ps(x, d))

#define INTV __m128i
#endif

union AVX_Data {
#if defined(__AVX512__)
    __m512 data;
#elif defined(__AVX256__)
    __m256 data;
#endif
    // float data_f[16];
};

template <int span>
inline void simd_store(float* dst, AVX_Data* src, bool half_precision)
{
    size_t width = (half_precision ? SIMD_WIDTH / 2 : SIMD_WIDTH);
#pragma unroll
    for (size_t i = 0; i < span; ++i) { SIMD_STORE2(dst + width * i, src[i].data, half_precision); }
}
template <int span>
inline void simd_load(AVX_Data* dst, float* src, bool half_precision)
{
    size_t width = (half_precision ? 1 : SIMD_WIDTH);
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_LOAD2(src + width * i, half_precision); }
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
