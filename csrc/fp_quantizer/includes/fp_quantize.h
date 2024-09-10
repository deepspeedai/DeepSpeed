// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cuda.h>
#include <stdint.h>

#include <cuda_fp16.h>

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif
#include <cuda_runtime_api.h>
#include <stdio.h>

#define QUANT_SWITCH(Q_BITS, ...)                        \
    [&] {                                                \
        if (12 == Q_BITS) {                              \
            constexpr int CONST_STOCHASTIC_ROUNDING = 0; \
            constexpr int CONST_Q_BITS = 8;              \
            constexpr int CONST_Q_MANTISA_BITS = 3;      \
            __VA_ARGS__();                               \
        } else if (13 == Q_BITS) {                       \
            constexpr int CONST_STOCHASTIC_ROUNDING = 1; \
            constexpr int CONST_Q_BITS = 8;              \
            constexpr int CONST_Q_MANTISA_BITS = 3;      \
            __VA_ARGS__();                               \
        } else if (10 == Q_BITS) {                       \
            constexpr int CONST_STOCHASTIC_ROUNDING = 0; \
            constexpr int CONST_Q_BITS = 8;              \
            constexpr int CONST_Q_MANTISA_BITS = 2;      \
            __VA_ARGS__();                               \
        } else if (11 == Q_BITS) {                       \
            constexpr int CONST_STOCHASTIC_ROUNDING = 1; \
            constexpr int CONST_Q_BITS = 8;              \
            constexpr int CONST_Q_MANTISA_BITS = 2;      \
            __VA_ARGS__();                               \
        } else if (28 == Q_BITS) {                       \
            constexpr int CONST_STOCHASTIC_ROUNDING = 0; \
            constexpr int CONST_Q_BITS = 12;             \
            constexpr int CONST_Q_MANTISA_BITS = 7;      \
            __VA_ARGS__();                               \
        } else if (29 == Q_BITS) {                       \
            constexpr int CONST_STOCHASTIC_ROUNDING = 1; \
            constexpr int CONST_Q_BITS = 12;             \
            constexpr int CONST_Q_MANTISA_BITS = 7;      \
            __VA_ARGS__();                               \
        } else if (6 == Q_BITS) {                        \
            constexpr int CONST_STOCHASTIC_ROUNDING = 0; \
            constexpr int CONST_Q_BITS = 6;              \
            constexpr int CONST_Q_MANTISA_BITS = 2;      \
            __VA_ARGS__();                               \
        } else if (7 == Q_BITS) {                        \
            constexpr int CONST_STOCHASTIC_ROUNDING = 1; \
            constexpr int CONST_Q_BITS = 6;              \
            constexpr int CONST_Q_MANTISA_BITS = 2;      \
            __VA_ARGS__();                               \
        } else if (2 == Q_BITS) {                        \
            constexpr int CONST_STOCHASTIC_ROUNDING = 0; \
            constexpr int CONST_Q_BITS = 4;              \
            constexpr int CONST_Q_MANTISA_BITS = 1;      \
            __VA_ARGS__();                               \
        } else {                                         \
            constexpr int CONST_STOCHASTIC_ROUNDING = 1; \
            constexpr int CONST_Q_BITS = 4;              \
            constexpr int CONST_Q_MANTISA_BITS = 1;      \
            __VA_ARGS__();                               \
        }                                                \
    }()

#define DEQUANT_SWITCH(Q_MANTISA_EXPONENT_BITS, ...) \
    [&] {                                            \
        if (12 == Q_MANTISA_EXPONENT_BITS) {         \
            constexpr int CONST_Q_MANTISA_BITS = 3;  \
            constexpr int CONST_Q_EXPONENT_BITS = 4; \
            __VA_ARGS__();                           \
        } else if (10 == Q_MANTISA_EXPONENT_BITS) {  \
            constexpr int CONST_Q_MANTISA_BITS = 2;  \
            constexpr int CONST_Q_EXPONENT_BITS = 5; \
            __VA_ARGS__();                           \
        } else if (28 == Q_MANTISA_EXPONENT_BITS) {  \
            constexpr int CONST_Q_MANTISA_BITS = 7;  \
            constexpr int CONST_Q_EXPONENT_BITS = 4; \
            __VA_ARGS__();                           \
        } else if (6 == Q_MANTISA_EXPONENT_BITS) {   \
            constexpr int CONST_Q_MANTISA_BITS = 2;  \
            constexpr int CONST_Q_EXPONENT_BITS = 3; \
            __VA_ARGS__();                           \
        } else {                                     \
            constexpr int CONST_Q_MANTISA_BITS = 1;  \
            constexpr int CONST_Q_EXPONENT_BITS = 2; \
            __VA_ARGS__();                           \
        }                                            \
    }()

template <typename T, int mantisa, int exponent>
void launch_quantization(T* val,
                         uint8_t* q_val,
                         int num_groups,
                         int group_size,
                         cudaStream_t stream,
                         float q_range,
                         int q_bits,
                         int q_mantisa_bits,
                         int stochastic_rounding);

template <typename T, int mantisa>
void launch_dequantization(uint8_t* val,
                           T* q_val,
                           int num_groups,
                           int group_size,
                           int q_mantisa_bits,
                           int q_exponent_bits,
                           cudaStream_t stream);

template <typename T, int mantisa>
void launch_selective_dequantization(uint8_t* val,
                                     T* q_val,
                                     int32_t* indexes,
                                     int num_groups,
                                     int group_size,
                                     int num_indexes,
                                     int q_mantisa_bits,
                                     int q_exponent_bits,
                                     cudaStream_t stream);
