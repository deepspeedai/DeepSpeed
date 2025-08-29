// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <immintrin.h>

inline __m512 cvt_bf16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_bf16_to_fp32(const __m256i src)
{
    auto y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_bf16(const __m512 src)
{
    __m512i value = _mm512_castps_si512(src);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    // uint32_t lsb = (input >> 16) & 1;
    auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
    // uint32_t rounding_bias = 0x7fff + lsb;
    t_value = _mm512_add_epi32(t_value, vec_bias);
    // input += rounding_bias;
    t_value = _mm512_add_epi32(t_value, value);
    // input = input >> 16;
    t_value = _mm512_srli_epi32(t_value, 16);
    // Check NaN before converting back to bf16
    t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
    return _mm512_cvtusepi32_epi16(t_value);
}

inline __m512 cvt_fp16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_fp16_to_fp32(const __m256i src) { return _mm512_cvtph_ps(src); }

inline __m256i cvt_fp32_to_fp16(const __m512 src) __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_fp16(const __m512 src)
{
    return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  256bit vector ==> 32 bytes, 512bit vector ==> 64 bytes
// If you change implementation of reduce_bf16_buffers, etc. , check whether this number needs
// to be changed
static int vector_length_in_bytes = 32;

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));
void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));
void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));

void parallel_memcpy(void* to, void* from, size_t n_bytes) __attribute__((target("avx512bw")));

#define VLOAD_U8(X) _mm256_loadu_si256((__m256i*)(X))
#define VLOAD_U16(X) _mm256_loadu_si256((__m256i*)(X))
#define VLOAD_F16(X) _mm256_loadu_si256((__m256i*)(X))
#define VLOAD_F32(X) _mm256_loadu_ps((float*)(X))

#define VSTORE_U8(A, B) _mm256_storeu_si256((__m256i*)(A), B)
#define VSTORE_U16(A, B) _mm256_storeu_si256((__m256i*)(A), B)
#define VSTORE_F16(A, B) _mm256_storeu_si256((__m256i*)(A), B)
#define VSTORE_F32(A, B) _mm256_storeu_ps((float*)(A), B)

#define VADD_F32(A, B) _mm256_add_ps(A, B)
#define VADD_F32_2VL(A, B) _mm512_add_ps(A, B)

#define CVT_BF16_TO_FP32(X) cvt_bf16_to_fp32(X)
#define CVT_FP16_TO_FP32(X) cvt_fp16_to_fp32(X)
#define CVT_FP32_TO_BF16(X) cvt_fp32_to_bf16(X)
#define CVT_FP32_TO_FP16(X) cvt_fp32_to_fp16(X)
