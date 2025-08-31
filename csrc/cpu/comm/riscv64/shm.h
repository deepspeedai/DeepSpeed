// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <riscv_vector.h>
#include <cmath>
using float16_t = _Float16;

inline vfloat32m2_t cvt_bf16_to_fp32(vuint16m1_t src, size_t vl) __attribute__((target("arch=+v")));
inline vfloat32m2_t cvt_bf16_to_fp32(vuint16m1_t src, size_t vl)
{
    vuint32m2_t widened = __riscv_vwcvtu_x_x_v_u32m2(src, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(widened, 16, vl));
}

inline vuint16m1_t cvt_fp32_to_bf16(vfloat32m2_t src, size_t vl) __attribute__((target("arch=+v")));
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
    __attribute__((target("arch=+v,+zvfh")));
inline vfloat32m2_t cvt_fp16_to_fp32(vfloat16m1_t src, size_t vl)
{
    return __riscv_vfwcvt_f_f_v_f32m2(src, vl);
}

inline vfloat16m1_t cvt_fp32_to_fp16(vfloat32m2_t src, size_t vl)
    __attribute__((target("arch=+v,+zvfh")));
inline vfloat16m1_t cvt_fp32_to_fp16(vfloat32m2_t src, size_t vl)
{
    return __riscv_vfncvt_rod_f_f_w_f16m1(src, vl);
}

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length. Dynamically acquired via the vsetvl instruction to
// compatible with different vector length.
static int vector_length_in_bytes = -1;

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("arch=+v")));
void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("arch=+v,+zvfh")));
void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("arch=+v")));

void parallel_memcpy(void* to, void* from, size_t n_bytes) __attribute__((target("arch=+v")));

#define VLOAD_U8(X) __riscv_vle8_v_u8m1((uint8_t*)(X), vl)
#define VLOAD_U16(X) __riscv_vle16_v_u16m1((uint16_t*)(X), vl)
#define VLOAD_F16(X) __riscv_vle16_v_f16m1((float16_t*)(X), vl)
#define VLOAD_F32(X) __riscv_vle32_v_f32m1((float*)(X), vl)

#define VSTORE_U8(A, B) __riscv_vse8_v_u8m1((uint8_t*)(A), B, vl)
#define VSTORE_U16(A, B) __riscv_vse16_v_u16m1((uint16_t*)(A), B, vl)
#define VSTORE_F16(A, B) __riscv_vse16_v_f16m1((float16_t*)(A), B, vl)
#define VSTORE_F32(A, B) __riscv_vse32_v_f32m1((float*)(A), B, vl)

#define VADD_F32(A, B) __riscv_vfadd_vv_f32m1(A, B, vl)
#define VADD_F32_2VL(A, B) __riscv_vfadd_vv_f32m2(A, B, vl)

#define CVT_BF16_TO_FP32(X) cvt_bf16_to_fp32(X, vl)
#define CVT_FP16_TO_FP32(X) cvt_fp16_to_fp32(X, vl)
#define CVT_FP32_TO_BF16(X) cvt_fp32_to_bf16(X, vl)
#define CVT_FP32_TO_FP16(X) cvt_fp32_to_fp16(X, vl)
