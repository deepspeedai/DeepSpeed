// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cstddef>

#if defined(__RVV__) && defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000
#include <riscv_vector.h>

// Adam update for one m4 block (vl lanes) of the four FP32 streams. The operation order
// matches the scalar Step_1 tail and the SVE kernel so every arithmetic tier stays within
// the same test tolerance. Scalar hyperparameters ride the vf-form intrinsics, so of all
// the loop constants only epsilon occupies a vector register.
inline void rvv_adam_block(float* params,
                           float* exp_avg,
                           float* exp_avg_sq,
                           vfloat32m4_t param,
                           vfloat32m4_t grad,
                           vfloat32m4_t momentum,
                           vfloat32m4_t variance,
                           size_t vl,
                           vfloat32m4_t epsilon_vec,
                           float beta1,
                           float beta2,
                           float bias_correction2,
                           float step_size,
                           float decay,
                           float weight_decay,
                           bool adamw_mode)
{
    const float beta1_complement = 1.0f - beta1;
    const float beta2_complement = 1.0f - beta2;

    if (weight_decay > 0 && !adamw_mode) {
        grad = __riscv_vfmacc_vf_f32m4(grad, weight_decay, param, vl);
    }
    momentum = __riscv_vfmacc_vf_f32m4(
        __riscv_vfmul_vf_f32m4(momentum, beta1, vl), beta1_complement, grad, vl);
    variance = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(variance, beta2, vl),
                                       beta2_complement,
                                       __riscv_vfmul_vv_f32m4(grad, grad, vl),
                                       vl);
    // bias_correction2 already holds 1/sqrt(1 - beta2^t), so the denominator is a single
    // fma instead of a second division.
    const vfloat32m4_t denominator = __riscv_vfmacc_vf_f32m4(
        epsilon_vec, bias_correction2, __riscv_vfsqrt_v_f32m4(variance, vl), vl);
    const vfloat32m4_t update = __riscv_vfdiv_vv_f32m4(momentum, denominator, vl);

    if (weight_decay > 0 && adamw_mode) {
        param = __riscv_vfmacc_vf_f32m4(param, decay, param, vl);
    }
    param = __riscv_vfmacc_vf_f32m4(param, step_size, update, vl);

    __riscv_vse32_v_f32m4(params, param, vl);
    __riscv_vse32_v_f32m4(exp_avg, momentum, vl);
    __riscv_vse32_v_f32m4(exp_avg_sq, variance, vl);
}

template <int span>
inline void rvv_adam_update(float* params,
                            const float* grads,
                            float* exp_avg,
                            float* exp_avg_sq,
                            size_t param_size,
                            float beta1,
                            float beta2,
                            float bias_correction1,
                            float bias_correction2,
                            float epsilon,
                            float learning_rate,
                            float weight_decay,
                            bool adamw_mode,
                            bool parallel)
{
    constexpr size_t tile_size = 128ULL * 1024 * 1024;
    // One m4 block per loop iteration. A wider pass body - two blocks with any
    // internal ordering of loads, math, and stores - collapses ~7x on in-order
    // cores: on an in-order SpacemiT core whose VLSU moves 128 bit/cycle, every
    // two-block organization of this same math measured ~44 cycles/element
    // against 6.2 for one block per iteration, the second block's memory ops
    // colliding with the first block's stores still in flight on the shared
    // port. The single-block loop keeps exactly four loads in flight and rides
    // the memory pipe at its sustained rate. (span only exists so Step_RVV
    // matches the Step_AVX/Step_SVE dispatch.)
    // 16 elements at VLEN=128: one 64-byte cache line per stream
    const size_t vector_width = __riscv_vsetvlmax_e32m4();

    const float step_size = -learning_rate / bias_correction1;
    const float decay = adamw_mode ? -learning_rate * weight_decay : weight_decay;

    for (size_t tile_start = 0; tile_start < param_size; tile_start += tile_size) {
        const size_t remaining = param_size - tile_start;
        const size_t tile_end = tile_start + (remaining < tile_size ? remaining : tile_size);

#pragma omp parallel if (parallel)
        {
            const vfloat32m4_t epsilon_vec = __riscv_vfmv_v_f_f32m4(epsilon, vector_width);

            // The hot loop runs entirely at the loop-invariant vlmax; vsetvl
            // executes only for the trailing short block below, which truncates
            // the final block to the exact tensor length so no scalar remainder
            // is ever needed.
            const size_t full_end =
                tile_start + (tile_end - tile_start) / vector_width * vector_width;

#pragma omp for
            for (size_t i = tile_start; i < full_end; i += vector_width) {
                vfloat32m4_t param = __riscv_vle32_v_f32m4(params + i, vector_width);
                vfloat32m4_t grad = __riscv_vle32_v_f32m4(grads + i, vector_width);
                vfloat32m4_t momentum = __riscv_vle32_v_f32m4(exp_avg + i, vector_width);
                vfloat32m4_t variance = __riscv_vle32_v_f32m4(exp_avg_sq + i, vector_width);

                rvv_adam_block(params + i,
                               exp_avg + i,
                               exp_avg_sq + i,
                               param,
                               grad,
                               momentum,
                               variance,
                               vector_width,
                               epsilon_vec,
                               beta1,
                               beta2,
                               bias_correction2,
                               step_size,
                               decay,
                               weight_decay,
                               adamw_mode);
            }

            // Tail block: fewer than vector_width elements remain.
#pragma omp single
            {
                const size_t tail_len = tile_end - full_end;
                if (tail_len > 0) {
                    const size_t vl = __riscv_vsetvl_e32m4(tail_len);
                    vfloat32m4_t param = __riscv_vle32_v_f32m4(params + full_end, vl);
                    vfloat32m4_t grad = __riscv_vle32_v_f32m4(grads + full_end, vl);
                    vfloat32m4_t momentum = __riscv_vle32_v_f32m4(exp_avg + full_end, vl);
                    vfloat32m4_t variance = __riscv_vle32_v_f32m4(exp_avg_sq + full_end, vl);

                    rvv_adam_block(params + full_end,
                                   exp_avg + full_end,
                                   exp_avg_sq + full_end,
                                   param,
                                   grad,
                                   momentum,
                                   variance,
                                   vl,
                                   epsilon_vec,
                                   beta1,
                                   beta2,
                                   bias_correction2,
                                   step_size,
                                   decay,
                                   weight_decay,
                                   adamw_mode);
                }
            }
        }
    }
}
#endif
