// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include "shm.h"

inline __m512 cvt_bf16_to_fp32(const __m256i src)
{
    auto y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

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

inline __m512 cvt_fp16_to_fp32(const __m256i src) { return _mm512_cvtph_ps(src); }

inline __m256i cvt_fp32_to_fp16(const __m512 src)
{
    return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

#define CVT_ADD_BF16(x)                                                                      \
    do {                                                                                     \
        auto in##x##_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[x] + i))); \
        inout_val = _mm512_add_ps(inout_val, in##x##_val);                                   \
    } while (0)

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  256bit vector ==> 32 bytes, 512bit vector ==> 64 bytes
// If you change implementation of reduce_bf16_buffers, etc. , check whether this number needs
// to be changed
#define VECTOR_LENGTH_IN_BYTES 32

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[0] + i)));
        switch (world_size) {
            case 16: CVT_ADD_BF16(15);
            case 15: CVT_ADD_BF16(14);
            case 14: CVT_ADD_BF16(13);
            case 13: CVT_ADD_BF16(12);
            case 12: CVT_ADD_BF16(11);
            case 11: CVT_ADD_BF16(10);
            case 10: CVT_ADD_BF16(9);
            case 9: CVT_ADD_BF16(8);
            case 8: CVT_ADD_BF16(7);
            case 7: CVT_ADD_BF16(6);
            case 6: CVT_ADD_BF16(5);
            case 5: CVT_ADD_BF16(4);
            case 4: CVT_ADD_BF16(3);
            case 3: CVT_ADD_BF16(2);
            case 2: CVT_ADD_BF16(1);
            case 1: break;
            default:
                for (int j = 1; j < world_size; j++) {
                    auto in_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[j] + i)));
                    inout_val = _mm512_add_ps(inout_val, in_val);
                }
        }
        _mm256_storeu_si256((__m256i*)(to_buffer + i), cvt_fp32_to_bf16(inout_val));
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < world_size; j++) { val += *(at::BFloat16*)(buffers[j] + i); }
        *(at::BFloat16*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

#define CVT_ADD_FP16(x)                                                                      \
    do {                                                                                     \
        auto in##x##_val = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[x] + i))); \
        inout_val = _mm512_add_ps(inout_val, in##x##_val);                                   \
    } while (0)

void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[0] + i)));
        switch (world_size) {
            case 16: CVT_ADD_FP16(15);
            case 15: CVT_ADD_FP16(14);
            case 14: CVT_ADD_FP16(13);
            case 13: CVT_ADD_FP16(12);
            case 12: CVT_ADD_FP16(11);
            case 11: CVT_ADD_FP16(10);
            case 10: CVT_ADD_FP16(9);
            case 9: CVT_ADD_FP16(8);
            case 8: CVT_ADD_FP16(7);
            case 7: CVT_ADD_FP16(6);
            case 6: CVT_ADD_FP16(5);
            case 5: CVT_ADD_FP16(4);
            case 4: CVT_ADD_FP16(3);
            case 3: CVT_ADD_FP16(2);
            case 2: CVT_ADD_FP16(1);
            case 1: break;
            default:
                for (int j = 1; j < world_size; j++) {
                    auto in_val = cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[j] + i)));
                    inout_val = _mm512_add_ps(inout_val, in_val);
                }
        }
        _mm256_storeu_si256((__m256i*)(to_buffer + i), cvt_fp32_to_fp16(inout_val));
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < world_size; j++) { val += *(at::Half*)(buffers[j] + i); }
        *(at::Half*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

#define CVT_ADD_F32(x)                                                \
    do {                                                              \
        auto in##x##_val = _mm256_loadu_ps((float*)(buffers[x] + i)); \
        inout_val = _mm256_add_ps(inout_val, in##x##_val);            \
    } while (0)

void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 4;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)(buffers[0] + i));
        switch (world_size) {
            case 16: CVT_ADD_F32(15);
            case 15: CVT_ADD_F32(14);
            case 14: CVT_ADD_F32(13);
            case 13: CVT_ADD_F32(12);
            case 12: CVT_ADD_F32(11);
            case 11: CVT_ADD_F32(10);
            case 10: CVT_ADD_F32(9);
            case 9: CVT_ADD_F32(8);
            case 8: CVT_ADD_F32(7);
            case 7: CVT_ADD_F32(6);
            case 6: CVT_ADD_F32(5);
            case 5: CVT_ADD_F32(4);
            case 4: CVT_ADD_F32(3);
            case 3: CVT_ADD_F32(2);
            case 2: CVT_ADD_F32(1);
            case 1: break;
            default:
                for (int j = 1; j < world_size; j++) {
                    auto in_val = _mm256_loadu_ps((float*)(buffers[j] + i));
                    inout_val = _mm256_add_ps(inout_val, in_val);
                }
        }
        _mm256_storeu_ps((float*)(to_buffer + i), inout_val);
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < world_size; j++) { val += *(float*)(buffers[j] + i); }
        *(float*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

void parallel_memcpy(void* to, void* from, size_t n_bytes)
{
    auto aligned_bytes = n_bytes - (n_bytes % VECTOR_LENGTH_IN_BYTES);
    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < aligned_bytes; i += VECTOR_LENGTH_IN_BYTES) {
        auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
        _mm256_storeu_si256((__m256i*)((char*)to + i), val);
    }

    // process remaining part
    for (int i = aligned_bytes; i < n_bytes; i++) { *((char*)to + i) = *((char*)from + i); }
}
