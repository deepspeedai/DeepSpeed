// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

#pragma once

#include "simd.h"

#if defined(__AVX512__)
#define REFLOW_STREAM_PACKED(dst, value) _mm256_stream_si256(dst, value)
#define REFLOW_PACK_FP16(value) _mm512_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT)
#elif defined(__AVX256__)
#define REFLOW_STREAM_PACKED(dst, value) _mm_stream_si128(dst, value)
#define REFLOW_PACK_FP16(value) _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT)
static __m256 reflow_load_8_bf16_as_f32(const void* data)
{
    __m128i a = readAs<__m128i>(data);     // 8 x uint16 (bf16)
    __m256i b = _mm256_cvtepu16_epi32(a);  // zero-extend to 8 x uint32
    __m256i c = _mm256_slli_epi32(b, 16);  // bf16 -> fp32 bits (bf16 is the high 16 bits of fp32)
    return readAs<__m256>(&c);
}

/* Pack 8 uint32 (each already holding a value in its low 16 bits) down to 8 uint16. AVX2 lacks a
 * direct 32->16 narrowing like AVX-512's _mm512_cvtusepi32_epi16, so use packus (which works
 * per-128-bit lane) plus a 64-bit permute to gather the 8 results into the low 128 bits.
 */
static inline __m128i reflow_pack_low16_x8(__m256i x)
{
    __m256i p = _mm256_packus_epi32(x, x);
    p = _mm256_permute4x64_epi64(p, 0xD8);
    return _mm256_castsi256_si128(p);
}

static inline __m128i pack_f32_as_bf16_nearest(__m256 v)
{
    __m256i u32 = readAs<__m256i>(&v);
    // round to nearest, ties to even: rounding_bias = ((u32 >> 16) & 1) + 0x7fff
    __m256i b = _mm256_srli_epi32(u32, 16);
    __m256i lsb = _mm256_and_si256(b, _mm256_set1_epi32(0x00000001));
    __m256i rounding_bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x00007fff));
    __m256i e = _mm256_srli_epi32(_mm256_add_epi32(u32, rounding_bias), 16);
    __m128i non_nan_res = reflow_pack_low16_x8(e);
    // NaN (exp all 1s, mantissa != 0): (u32 & 0x7fffffff) > 0x7f800000 -> emit a quiet NaN 0x7fc0
    __m256i non_sign = _mm256_and_si256(u32, _mm256_set1_epi32(0x7fffffff));
    __m256i nan_mask = _mm256_cmpgt_epi32(non_sign, _mm256_set1_epi32(0x7f800000));
    __m128i nan_mask16 = reflow_pack_low16_x8(_mm256_srli_epi32(nan_mask, 16));
    return _mm_blendv_epi8(non_nan_res, _mm_set1_epi16((short)0x7fc0), nan_mask16);
}

// The stock AVX2 helpers support FP16 and FP32. ReFlow also transfers and accumulates BF16.
template <int span>
inline void reflow_simd_store(c10::BFloat16* dst, AVX_Data* src)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        writeAs(dst + SIMD_WIDTH * i, pack_f32_as_bf16_nearest(src[i].data));
    }
}

template <int span>
inline void reflow_simd_load(AVX_Data* dst, c10::BFloat16* src)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = reflow_load_8_bf16_as_f32(src + SIMD_WIDTH * i);
    }
}
#endif

#if defined(__AVX512__) or defined(__AVX256__)
template <int span, typename T>
inline bool reflow_stream_ready_output(T* dst)
{
    // The eight-vector loop writes the bulk. Short vector tails and misaligned views use cached
    // stores.
    return span == 8 && reinterpret_cast<uintptr_t>(dst) % sizeof(INTV) == 0;
}

template <int span, typename T>
inline void reflow_simd_store_ready(T* dst, AVX_Data* src, bool stream_output)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        INTV packed;
        if constexpr (std::is_same_v<T, c10::BFloat16>) {
            packed = pack_f32_as_bf16_nearest(src[i].data);
        } else {
            static_assert(std::is_same_v<T, c10::Half>);
            packed = REFLOW_PACK_FP16(src[i].data);
        }
        T* output = dst + SIMD_WIDTH * i;
        if (stream_output) {
            REFLOW_STREAM_PACKED(reinterpret_cast<INTV*>(output), packed);
        } else {
            writeAs(output, packed);
        }
    }
}

template <int span, typename T>
inline void reflow_simd_store(T* dst, AVX_Data* src)
{
    simd_store<span>(dst, src);
}

template <int span, typename T>
inline void reflow_simd_load(AVX_Data* dst, T* src)
{
    simd_load<span>(dst, src);
}
#endif
