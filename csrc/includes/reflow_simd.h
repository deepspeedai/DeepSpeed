// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

#pragma once

#include "simd.h"

#if defined(__AVX512__)
#define SIMD_STREAM_STORE(a, d) _mm512_stream_ps(a, d)
// Reflow BF16 aliases (the stock AVX-512 BF16 conversion works as-is).
#define SIMD_LOAD_BF16_REFLOW(x) load_16_bf16_as_f32(x)
#define SIMD_STORE_BF16_REFLOW(x, d) store_16_f32_as_bf16_nearest(d, x)

static void stream_store_16_f32_as_bf16_nearest(__m512 v, void* data)
{
    __m512i u32 = readAs<__m512i>(&v);

    __m512i b = _mm512_srli_epi32(u32, 16);
    __m512i lsb_mask = _mm512_set1_epi32(0x00000001);
    __m512i c = _mm512_and_si512(b, lsb_mask);
    __m512i bias_constant = _mm512_set1_epi32(0x00007fff);
    __m512i rounding_bias = _mm512_add_epi32(c, bias_constant);

    __m512i d = _mm512_add_epi32(u32, rounding_bias);
    __m512i e = _mm512_srli_epi32(d, 16);
    __m256i non_nan_res = _mm512_cvtusepi32_epi16(e);

    __m512i mask_out_sign = _mm512_set1_epi32(0x7fffffff);
    __m512i non_sign_bits = _mm512_and_si512(u32, mask_out_sign);
    __m512i nan_threshold = _mm512_set1_epi32(0x7f800000);
    __mmask16 nan_mask = _mm512_cmp_epi32_mask(non_sign_bits, nan_threshold, _MM_CMPINT_GT);

    __m256i nans = _mm256_set1_epi16(0x7fc0);
    __m256i res = _mm256_mask_mov_epi16(non_nan_res, nan_mask, nans);

    _mm256_storeu_si256((__m256i*)data, res);
}
#define SIMD_STREAM_STORE_BF16(x, d) stream_store_16_f32_as_bf16_nearest(d, x)

#define SIMD_STREAM_STORE_FP16(x, d) \
    _mm256_storeu_ps(x, _mm256_castsi256_ps(_mm512_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT)))

#elif defined(__AVX256__)
#define SIMD_STREAM_STORE(a, d) _mm256_stream_ps(a, d)
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

static void reflow_store_8_f32_as_bf16_nearest(__m256 v, void* data)
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
    __m128i res = _mm_blendv_epi8(non_nan_res, _mm_set1_epi16((short)0x7fc0), nan_mask16);
    writeAs(data, res);
}

static void reflow_stream_store_8_f32_as_bf16_nearest(__m256 v, void* data)
{
    __m256i u32 = readAs<__m256i>(&v);
    __m256i b = _mm256_srli_epi32(u32, 16);
    __m256i lsb = _mm256_and_si256(b, _mm256_set1_epi32(0x00000001));
    __m256i rounding_bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x00007fff));
    __m256i e = _mm256_srli_epi32(_mm256_add_epi32(u32, rounding_bias), 16);
    __m128i non_nan_res = reflow_pack_low16_x8(e);
    __m256i non_sign = _mm256_and_si256(u32, _mm256_set1_epi32(0x7fffffff));
    __m256i nan_mask = _mm256_cmpgt_epi32(non_sign, _mm256_set1_epi32(0x7f800000));
    __m128i nan_mask16 = reflow_pack_low16_x8(_mm256_srli_epi32(nan_mask, 16));
    __m128i res = _mm_blendv_epi8(non_nan_res, _mm_set1_epi16((short)0x7fc0), nan_mask16);
    /* Match the AVX-512 BF16 path (stream_store_16_f32_as_bf16_nearest), which uses a regular
     * store: a non-temporal _mm_stream_si128 here is weakly ordered and not fenced before the
     * OMP workers' half_params writes are read, yielding non-deterministic output.
     */
    _mm_storeu_si128((__m128i*)data, res);
}

// Reflow-specific AVX-256 BF16 support (the stock SIMD_*_BF16 above are AVX-512-only).
#define SIMD_LOAD_BF16_REFLOW(x) reflow_load_8_bf16_as_f32(x)
#define SIMD_STORE_BF16_REFLOW(x, d) reflow_store_8_f32_as_bf16_nearest(d, x)
#define SIMD_STREAM_STORE_BF16(x, d) reflow_stream_store_8_f32_as_bf16_nearest(d, x)
#define SIMD_STREAM_STORE_FP16(x, d) \
    _mm_storeu_ps(x, _mm_castsi128_ps(_mm256_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT)))
#endif

// Streaming stores and BF16 helpers are available only in the AVX builds.
#if defined(__AVX512__) or defined(__AVX256__)
// Non-temporal (streaming) store per dtype.
template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> reflow_simd_store_stream(
    T* dst,
    AVX_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) { SIMD_STREAM_STORE(dst + width * i, src[i].data); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::Half>, void> reflow_simd_store_stream(
    T* dst,
    AVX_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        SIMD_STREAM_STORE_FP16((float*)(dst + width * i), src[i].data);
    }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> reflow_simd_store_stream(
    T* dst,
    AVX_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        SIMD_STREAM_STORE_BF16((float*)(dst + width * i), src[i].data);
    }
}

template <int span, typename T>
inline typename std::enable_if_t<!std::is_same_v<T, float> && !std::is_same_v<T, c10::Half> &&
                                     !std::is_same_v<T, c10::BFloat16>,
                                 void>
reflow_simd_store_stream(T* dst, AVX_Data* src)
{
    simd_store<span, T>(dst, src);
}
#endif

#if defined(__AVX512__) or defined(__AVX256__)
// Reflow load/store: BF16 uses the Reflow BF16 path (incl. AVX-256); other dtypes delegate
// to the stock simd_load/simd_store so the standard DeepSpeedCPUAdam path is untouched.
template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> reflow_simd_store(
    T* dst,
    AVX_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        SIMD_STORE_BF16_REFLOW((float*)(dst + width * i), src[i].data);
    }
}
template <int span, typename T>
inline typename std::enable_if_t<!std::is_same_v<T, c10::BFloat16>, void> reflow_simd_store(
    T* dst,
    AVX_Data* src)
{
    simd_store<span>(dst, src);
}
template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> reflow_simd_load(
    AVX_Data* dst,
    T* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_LOAD_BF16_REFLOW((float*)(src + width * i));
    }
}
template <int span, typename T>
inline typename std::enable_if_t<!std::is_same_v<T, c10::BFloat16>, void> reflow_simd_load(
    AVX_Data* dst,
    T* src)
{
    simd_load<span>(dst, src);
}
#endif
