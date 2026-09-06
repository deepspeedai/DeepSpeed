// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cstddef>

#if defined(__SVE__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

template <int span>
inline void sve_adam_update(float* params,
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
    const size_t vector_width = svcntw();
    const size_t block_width = vector_width * span;

    const float decay = adamw_mode ? -learning_rate * weight_decay : weight_decay;

    for (size_t tile_start = 0; tile_start < param_size; tile_start += tile_size) {
        const size_t remaining = param_size - tile_start;
        const size_t tile_end = tile_start + (remaining < tile_size ? remaining : tile_size);

#pragma omp parallel if (parallel)
        {
            const svfloat32_t beta1_vec = svdup_n_f32(beta1);
            const svfloat32_t beta2_vec = svdup_n_f32(beta2);
            const svfloat32_t beta1_complement_vec = svdup_n_f32(1.0f - beta1);
            const svfloat32_t beta2_complement_vec = svdup_n_f32(1.0f - beta2);
            const svfloat32_t bias_correction2_vec = svdup_n_f32(bias_correction2);
            const svfloat32_t epsilon_vec = svdup_n_f32(epsilon);
            const svfloat32_t step_size_vec = svdup_n_f32(-learning_rate / bias_correction1);
            const svfloat32_t weight_decay_vec = svdup_n_f32(decay);

#pragma omp for
            for (size_t i = tile_start; i < tile_end; i += block_width) {
#pragma unroll
                for (int j = 0; j < span; ++j) {
                    const size_t offset = i + j * vector_width;
                    if (offset >= tile_end) break;

                    const svbool_t predicate = svwhilelt_b32(offset, tile_end);
                    svfloat32_t param = svldnt1_f32(predicate, params + offset);
                    svfloat32_t grad = svldnt1_f32(predicate, grads + offset);
                    svfloat32_t momentum = svldnt1_f32(predicate, exp_avg + offset);
                    svfloat32_t variance = svldnt1_f32(predicate, exp_avg_sq + offset);

                    if (weight_decay > 0 && !adamw_mode) {
                        grad = svmad_f32_x(predicate, param, weight_decay_vec, grad);
                    }

                    momentum = svmad_f32_x(predicate,
                                           momentum,
                                           beta1_vec,
                                           svmul_f32_x(predicate, grad, beta1_complement_vec));
                    const svfloat32_t grad_squared = svmul_f32_x(predicate, grad, grad);
                    variance =
                        svmad_f32_x(predicate,
                                    variance,
                                    beta2_vec,
                                    svmul_f32_x(predicate, grad_squared, beta2_complement_vec));

                    const svfloat32_t denominator = svmad_f32_x(predicate,
                                                                svsqrt_f32_x(predicate, variance),
                                                                bias_correction2_vec,
                                                                epsilon_vec);
                    const svfloat32_t update = svdiv_f32_x(predicate, momentum, denominator);

                    if (weight_decay > 0 && adamw_mode) {
                        param = svmad_f32_x(predicate, param, weight_decay_vec, param);
                    }
                    param = svmad_f32_x(predicate, update, step_size_vec, param);

                    svstnt1_f32(predicate, params + offset, param);
                    svstnt1_f32(predicate, exp_avg + offset, momentum);
                    svstnt1_f32(predicate, exp_avg_sq + offset, variance);
                }
            }
        }
    }
}
#endif
