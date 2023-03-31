// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_copy.h"
#include <omp.h>

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))

#if defined(__AVX512__) or defined(__AVX256__)
union AVX_Data {
#if defined(__AVX512__)
    __m512 data;
#else
    __m256 data;
#endif
};
#endif

static void helper_memcpy_1(float* dest, float* src, size_t param_size)
{
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    rounded_size = ROUND_DOWN(param_size, SIMD_WIDTH);

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH) {
            AVX_Data src_4;
            src_4.data = SIMD_LOAD(src + i);

            SIMD_STORE(dest + i, src_4.data);
        }
    }

#endif

    if (param_size > rounded_size) {
#pragma omp parallel for
        for (size_t k = rounded_size; k < param_size; k++) { dest[k] = src[k]; }
    }
}

static void helper_memcpy_4(float* dest, float* src, size_t param_size)
{
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    rounded_size = ROUND_DOWN(param_size, (SIMD_WIDTH << 2));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2)) {
            AVX_Data src_4[4];
            src_4[0].data = SIMD_LOAD(src + i);
            src_4[1].data = SIMD_LOAD(src + i + SIMD_WIDTH);
            src_4[2].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 1));
            src_4[3].data = SIMD_LOAD(src + i + SIMD_WIDTH * 3);

            SIMD_STORE(dest + i, src_4[0].data);
            SIMD_STORE(dest + i + SIMD_WIDTH, src_4[1].data);
            SIMD_STORE(dest + i + (SIMD_WIDTH << 1), src_4[2].data);
            SIMD_STORE(dest + i + SIMD_WIDTH * 3, src_4[3].data);
        }
    }
#endif
    if (param_size > rounded_size)
        helper_memcpy_1((dest + rounded_size), (src + rounded_size), (param_size - rounded_size));
}

static void helper_mempcy_8(float* dest, float* src, size_t param_size)
{
    size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

    rounded_size = ROUND_DOWN(param_size, (SIMD_WIDTH << 2));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3)) {
            AVX_Data src_4[8];
            src_4[0].data = SIMD_LOAD(src + i);
            src_4[1].data = SIMD_LOAD(src + i + SIMD_WIDTH);
            src_4[2].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 1));
            src_4[3].data = SIMD_LOAD(src + i + SIMD_WIDTH * 3);
            src_4[4].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 2));
            src_4[5].data = SIMD_LOAD(src + i + SIMD_WIDTH * 5);
            src_4[6].data = SIMD_LOAD(src + i + SIMD_WIDTH * 6);
            src_4[7].data = SIMD_LOAD(src + i + SIMD_WIDTH * 7);

            SIMD_STORE(dest + i, src_4[0].data);
            SIMD_STORE(dest + i + SIMD_WIDTH, src_4[1].data);
            SIMD_STORE(dest + i + (SIMD_WIDTH << 1), src_4[2].data);
            SIMD_STORE(dest + i + SIMD_WIDTH * 3, src_4[3].data);
            SIMD_STORE(dest + i + (SIMD_WIDTH << 2), src_4[4].data);
            SIMD_STORE(dest + i + SIMD_WIDTH * 5, src_4[5].data);
            SIMD_STORE(dest + i + SIMD_WIDTH * 6, src_4[6].data);
            SIMD_STORE(dest + i + SIMD_WIDTH * 7, src_4[7].data);
        }
    }
#endif
    if (param_size > rounded_size)
        helper_memcpy_4((dest + rounded_size), (src + rounded_size), (param_size - rounded_size));
}

int deepspeed_py_memcpy(torch::Tensor& dest, const torch::Tensor& src)
{
    auto dest_c = dest.contiguous();
    auto src_c = src.contiguous();

    float* dest_ptr = (float*)dest_c.data_ptr();
    float* src_ptr = (float*)src_c.data_ptr();

    helper_mempcy_8(dest_ptr, src_ptr, dest_c.size(0));

    return 0;
}
