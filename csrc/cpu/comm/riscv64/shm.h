// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <riscv_vector.h>

inline vfloat32m2_t cvt_bf16_to_fp32(vuint16m1_t src, size_t vl) __attribute__((target("arch=+v")));
inline vuint16m1_t cvt_fp32_to_bf16(vfloat32m2_t src, size_t vl)
    __attribute__((target("arch=+v,+zvfh")));
inline vfloat32m2_t cvt_fp16_to_fp32(vfloat16m1_t src, size_t vl)
    __attribute__((target("arch=+v,+zvfh")));
inline vfloat16m1_t cvt_fp32_to_fp16(vfloat32m2_t src, size_t vl)
    __attribute__((target("arch=+v,+zvfh")));

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("arch=+v")));
void reduce_fp16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("arch=+v,+zvfh")));
void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("arch=+v")));

void parallel_memcpy(void* to, void* from, size_t n_bytes) __attribute__((target("arch=+v")));

extern int world_size;
