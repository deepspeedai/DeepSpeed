// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <torch/extension.h>
#include <cassert>
#include "simd.h"

#if defined(__ENABLE_CUDA__)
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "custom_cuda_layers.h"
typedef __half ds_half_precision_t;
#elif defined(__ENABLE_CANN__)
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
typedef c10::Half ds_half_precision_t;
#else
#include <cmath>
typedef unsigned short ds_half_precision_t;
#endif

#define STEP(SPAN)                                             \
    void Step_##SPAN(float* _params,                           \
                     float* grads,                             \
                     float* _exp_avg,                          \
                     size_t _param_size,                       \
                     ds_half_precision_t* dev_param = nullptr, \
                     bool half_precision = false);

class Lion_Optimizer {
public:
    Lion_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float weight_decay = 0)
        : _alpha(alpha), _betta1(betta1), _betta2(betta2), _weight_decay(weight_decay), _step(0)
    {
#if defined(__ENABLE_CUDA__)
        cudaMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        cudaMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));

        _streams[0] = TrainingContext::Instance().GetCurrentStream();
        _streams[1] = TrainingContext::Instance().GetNewStream();
        _buf_index = false;
#elif defined(__ENABLE_CANN__)
        aclrtMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        aclrtMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));

        _buf_index = false;
#endif
    }
    ~Lion_Optimizer()
    {
#if defined(__ENABLE_CUDA__)
        cudaFreeHost(_doubled_buffer[0]);
        cudaFreeHost(_doubled_buffer[1]);
#elif defined(__ENABLE_CANN__)
        aclrtFreeHost(_doubled_buffer[0]);
        aclrtFreeHost(_doubled_buffer[1]);
#endif
    }

#if defined(__AVX512__) or defined(__AVX256__)
    template <int span>
    void Step_AVX(size_t* rounded_size,
                  float* _params,
                  float* grads,
                  float* _exp_avg,
                  size_t param_size,
                  ds_half_precision_t* dev_param = nullptr,
                  bool half_precision = false);
#endif
    STEP(1)
    STEP(4)
    STEP(8)
#if defined(__ENABLE_CUDA__)
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) cudaStreamSynchronize(_streams[i]);
    }
#elif defined(__ENABLE_CANN__)
    inline void SynchronizeStreams()
    {
        for (int i = 0; i < 2; i++) aclrtSynchronizeStream(_streams[i].stream());
    }
#endif
    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        _step++;
        if (_step != step || beta1 != _betta1 || beta2 != _betta2) {
            _step = step;
            _betta1 = beta1;
            _betta2 = beta2;
        }
    }
    inline void update_state(float lr, float weight_decay)
    {
        _alpha = lr;
        _weight_decay = weight_decay;
    }

private:
    float _alpha;
    float _betta1;
    float _betta2;
    float _weight_decay;
    size_t _step;

#if defined(__ENABLE_CUDA__)
    float* _doubled_buffer[2];
    cudaStream_t _streams[2];
    bool _buf_index;
#elif defined(__ENABLE_CANN__)
    float* _doubled_buffer[2];
    c10_npu::NPUStream _streams[2] = {c10_npu::getCurrentNPUStream(),
                                      c10_npu::getNPUStreamFromPool()};
    bool _buf_index;
#endif
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void Lion_Optimizer::Step_AVX(size_t* rounded_size,
                              float* _params,
                              float* grads,
                              float* _exp_avg,
                              size_t _param_size,
                              ds_half_precision_t* dev_params,
                              bool half_precision)
{
    size_t new_rounded_size = 0;
    int rshft = half_precision ? 1 : 0;

    constexpr float neg1 = -1.0f;
    AVX_Data neg1_4;
    neg1_4.data = SIMD_SET(neg1);

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    float step_size = -_alpha;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float after_decay = 1.0f - _alpha * _weight_decay;
    AVX_Data after_decay_4;
    if (_weight_decay > 0) after_decay_4.data = SIMD_SET(after_decay);

    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#elif defined(__ENABLE_CANN__)
        if ((t / TILE) >= 2) { aclrtSynchronizeStream(_streams[_buf_index].stream()); }
#endif
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + (i >> rshft), half_precision);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, _exp_avg + i, false);

            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + (i >> rshft), half_precision);

            AVX_Data tmp_4[span];

            simd_mul<span>(tmp_4, momentum_4, betta1_4);
            simd_fma<span>(tmp_4, grad_4, betta1_minus1_4, tmp_4);
            // We already used intrinsics, so consider the machine representation fixed.
            simd_and<span>(tmp_4, tmp_4, neg1_4);
            simd_xor<span>(tmp_4, tmp_4, step_size_4);
            if (_weight_decay > 0) {
                simd_fma<span>(param_4, param_4, after_decay_4, tmp_4);
            } else {
                simd_add<span>(param_4, param_4, tmp_4);
            }

            simd_mul<span>(momentum_4, momentum_4, betta2_4);
            simd_fma<span>(momentum_4, grad_4, betta2_minus1_4, momentum_4);

            simd_store<span>(_params + (i >> rshft), param_4, half_precision);
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
            if (dev_params) {
                simd_store<span>(_doubled_buffer[_buf_index] + (i - t), param_4, half_precision);
            }
#endif
            simd_store<span>(_exp_avg + i, momentum_4, false);
        }
#if defined(__ENABLE_CUDA__)
        if (dev_params) {
            if (half_precision)
                launch_param_update_half(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            else
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);

            _buf_index = !_buf_index;
        }
#elif defined(__ENABLE_CANN__)
        if (dev_params) {
            size_t memcpy_size = copy_size * sizeof(_doubled_buffer[_buf_index][0]);
            if (half_precision) memcpy_size /= 2;
            aclrtMemcpy(dev_params + t,
                        memcpy_size,
                        _doubled_buffer[_buf_index],
                        memcpy_size,
                        aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE);

            _buf_index = !_buf_index;
        }
#endif
    }
    *rounded_size = new_rounded_size;
}
#endif

int create_lion_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float weight_decay = 0,
                          bool should_log = false);

int ds_lion_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float weight_decay,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg);

int ds_lion_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float weight_decay,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& gpu_params);

int destroy_lion_optimizer(int optimizer_id);
