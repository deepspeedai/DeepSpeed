// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <cmath>
#include "shm.h"

using float16_t = _Float16;
using float32_t = float_t;

inline vfloat32m2_t cvt_bf16_to_fp32(vuint16m1_t src, size_t vl)
{
    vuint32m2_t widened = __riscv_vwcvtu_x_x_v_u32m2(src, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(widened, 16, vl));
}

inline vuint16m1_t cvt_fp32_to_bf16(vfloat32m2_t src, size_t vl)
{
    vuint32m2_t value = __riscv_vreinterpret_v_f32m2_u32m2(src);
    vuint32m2_t nan = __riscv_vmv_v_x_u32m2(0xFFFF, vl);
    vbool16_t mask_value = __riscv_vmfne_vv_f32m2_b16(src, src, vl);
    vuint32m2_t ones = __riscv_vmv_v_x_u32m2(0x1, vl);
    vuint32m2_t vec_bias = __riscv_vmv_v_x_u32m2(0x7FFF, vl);
    // uint32_t lsb = (input >> 16) & 1;
    vuint32m2_t t_value = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(value, 16, vl), 0x1, vl);
    // uint32_t rounding_bias = 0x7fff + lsb;
    t_value = __riscv_vadd_vv_u32m2(t_value, vec_bias, vl);
    // input += rounding_bias;
    t_value = __riscv_vadd_vv_u32m2(t_value, value, vl);
    // input = input >> 16;
    t_value = __riscv_vsrl_vx_u32m2(t_value, 16, vl);
    // Check NaN before converting back to bf16
    t_value = __riscv_vmerge_vvm_u32m2(t_value, nan, mask_value, vl);

    return __riscv_vncvt_x_x_w_u16m1(t_value, vl);
}

inline vfloat32m2_t cvt_fp16_to_fp32(vfloat16m1_t src, size_t vl)
{
    return __riscv_vfwcvt_f_f_v_f32m2(src, vl);
}

inline vfloat16m1_t cvt_fp32_to_fp16(vfloat32m2_t src, size_t vl)
{
    return __riscv_vfncvt_rod_f_f_w_f16m1(src, vl);
}

#define CVT_ADD_BF16(x)                                                                   \
    do {                                                                                  \
        auto in##x##_val =                                                                \
            cvt_bf16_to_fp32(__riscv_vle16_v_u16m1((uint16_t*)(buffers[x] + i), vl), vl); \
        inout_val = __riscv_vfadd_vv_f32m2(inout_val, in##x##_val, vl);                   \
    } while (0)

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length. 128bit vector ==> 16 bytes, 256bit vector ==> 32 bytes
void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 2;
    size_t vl = __riscv_vsetvl_e16m1(num_elements);
    int main_elements = num_elements - (num_elements % vl);
    int remain_elements = num_elements % vl;
    int vector_length_in_bytes = vl * element_size;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += vector_length_in_bytes) {
        auto inout_val =
            cvt_bf16_to_fp32(__riscv_vle16_v_u16m1((uint16_t*)(buffers[0] + i), vl), vl);
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
                    auto in_val = cvt_bf16_to_fp32(
                        __riscv_vle16_v_u16m1((uint16_t*)(buffers[j] + i), vl), vl);
                    inout_val = __riscv_vfadd_vv_f32m2(inout_val, in_val, vl);
                }
        }
        __riscv_vse16_v_u16m1((uint16_t*)(to_buffer + i), cvt_fp32_to_bf16(inout_val, vl), vl);
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

#define CVT_ADD_FP16(x)                                                                    \
    do {                                                                                   \
        auto in##x##_val =                                                                 \
            cvt_fp16_to_fp32(__riscv_vle16_v_f16m1((float16_t*)(buffers[x] + i), vl), vl); \
        inout_val = __riscv_vfadd_vv_f32m2(inout_val, in##x##_val, vl);                    \
    } while (0)

void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 2;
    size_t vl = __riscv_vsetvl_e16m1(num_elements);
    int main_elements = num_elements - (num_elements % vl);
    int remain_elements = num_elements % vl;
    int vector_length_in_bytes = vl * element_size;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += vector_length_in_bytes) {
        auto inout_val =
            cvt_fp16_to_fp32(__riscv_vle16_v_f16m1((float16_t*)(buffers[0] + i), vl), vl);
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
                    auto in_val = cvt_fp16_to_fp32(
                        __riscv_vle16_v_f16m1((float16_t*)(buffers[j] + i), vl), vl);
                    inout_val = __riscv_vfadd_vv_f32m2(inout_val, in_val, vl);
                }
        }
        __riscv_vse16_v_f16m1((float16_t*)(to_buffer + i), cvt_fp32_to_fp16(inout_val, vl), vl);
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

#define CVT_ADD_F32(x)                                                          \
    do {                                                                        \
        auto in##x##_val = __riscv_vle32_v_f32m1((float*)(buffers[x] + i), vl); \
        inout_val = __riscv_vfadd_vv_f32m1(inout_val, in##x##_val, vl);         \
    } while (0)

void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 2;
    size_t vl = __riscv_vsetvl_e32m1(num_elements);
    int main_elements = num_elements - (num_elements % vl);
    int remain_elements = num_elements % vl;
    int vector_length_in_bytes = vl * element_size;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += vector_length_in_bytes) {
        auto inout_val = __riscv_vle32_v_f32m1((float*)(buffers[0] + i), vl);
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
                    auto in_val = __riscv_vle32_v_f32m1((float*)(buffers[j] + i), vl);
                    inout_val = __riscv_vfadd_vv_f32m1(inout_val, in_val, vl);
                }
        }
        __riscv_vse32_v_f32m1((float*)(to_buffer + i), inout_val, vl);
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
    size_t vl = __riscv_vsetvl_e8m1(n_bytes);
    auto aligned_bytes = n_bytes - (n_bytes % vl);
    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < aligned_bytes; i += vl) {
        auto val = __riscv_vle8_v_u8m1((uint8_t*)((char*)from + i), vl);
        __riscv_vse8_v_u8m1((uint8_t*)((char*)to + i), val, vl);
    }
    // process remaining part
    for (int i = aligned_bytes; i < n_bytes; i++) { *((char*)to + i) = *((char*)from + i); }
}
