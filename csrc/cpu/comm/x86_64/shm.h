// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <immintrin.h>

inline __m512 cvt_bf16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_bf16(const __m512 src) __attribute__((target("avx512bw")));
inline __m512 cvt_fp16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_fp16(const __m512 src) __attribute__((target("avx512bw")));

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));
void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));
void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));

void parallel_memcpy(void* to, void* from, size_t n_bytes) __attribute__((target("avx512bw")));

extern int world_size;
