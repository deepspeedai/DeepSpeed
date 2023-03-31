// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <fstream>
#include "custom_cuda_layers.h"

using namespace std;

template <typename T>
class Normalize_Layer {
public:
    struct Config {
        uint32_t batchSize;
        uint32_t seqLength;
        uint32_t hiddenDim;
        float epsilon;
        bool training;
        bool useMean;
        Config(uint32_t batch,
               uint32_t seq,
               uint32_t h,
               float epsilon = 1e-12,
               bool training = true,
               bool useMean = true)
            : batchSize(batch),
              seqLength(seq),
              hiddenDim(h),
              epsilon(epsilon),
              training(training),
              useMean(useMean)
        {
        }
    };

    Normalize_Layer(Config config)
        : config_(config), vars(nullptr), means(nullptr), vals_hat(nullptr)
    {
    }

    ~Normalize_Layer() {}

    void ForwardCheckpoint(int bsz,  // batch * seq
                           T* vals,
                           const T* residual,
                           const T* gamma,
                           const T* betta,
                           cudaStream_t& stream,
                           bool preLayerNorm = false)
    {
        launch_bias_residual_layer_norm(vals,
                                        residual,
                                        gamma,
                                        betta,
                                        config_.epsilon,
                                        bsz,
                                        config_.hiddenDim,
                                        stream,
                                        preLayerNorm,
                                        config_.training,
                                        vars,
                                        means);
    }

    void Forward(int bsz,
                 T* vals,
                 const T* residual,
                 const T* gamma,
                 const T* betta,
                 cudaStream_t& stream,
                 bool preLayerNorm = false)
    {
        launch_bias_residual_layer_norm(vals,
                                        residual,
                                        gamma,
                                        betta,
                                        config_.epsilon,
                                        bsz,
                                        config_.hiddenDim,
                                        stream,
                                        preLayerNorm,
                                        config_.training,
                                        vars);
    }

    void Backward(int bsz,
                  const T* out_grad,
                  const T* gamma,
                  T* gamma_grad,
                  T* betta_grad,
                  cudaStream_t stream[2],
                  T* inp_grad_out,
                  const T* norm_in = nullptr)
    {
        launch_layerNorm_backward(out_grad,
                                  norm_in,
                                  vars,
                                  means,
                                  gamma,
                                  gamma_grad,
                                  betta_grad,
                                  inp_grad_out,
                                  bsz,
                                  config_.hiddenDim,
                                  stream);
    }

    void Backward(int bsz,
                  const T* out_grad,
                  const T* gamma,
                  const T* betta,
                  T* gamma_grad,
                  T* betta_grad,
                  cudaStream_t stream[2],
                  T* inp_grad_out,
                  const T* norm_out)
    {
        launch_layerNorm_backward(out_grad,
                                  norm_out,
                                  vars,
                                  gamma,
                                  gamma_grad,
                                  betta_grad,
                                  inp_grad_out,
                                  bsz,
                                  config_.hiddenDim,
                                  stream,
                                  !config_.useMean,
                                  betta);
    }

    void BackwardFusedAdd(int bsz,
                          const T* out_grad1,
                          const T* out_grad2,
                          const T* gamma,
                          T* gamma_grad,
                          T* betta_grad,
                          cudaStream_t stream[2],
                          T* inp_grad_out,
                          const T* norm_in = nullptr)
    {
        launch_layerNorm_backward_fused_add(out_grad1,
                                            out_grad2,
                                            norm_in,
                                            vars,
                                            means,
                                            gamma,
                                            gamma_grad,
                                            betta_grad,
                                            inp_grad_out,
                                            bsz,
                                            config_.hiddenDim,
                                            stream);
    }

    void BackwardFusedAdd(int bsz,
                          const T* out_grad1,
                          const T* out_grad2,
                          const T* gamma,
                          const T* betta,
                          T* gamma_grad,
                          T* betta_grad,
                          cudaStream_t stream[2],
                          T* inp_grad_out,
                          const T* norm_out)
    {
        launch_layerNorm_backward_fused_add(out_grad1,
                                            out_grad2,
                                            norm_out,
                                            vars,
                                            gamma,
                                            gamma_grad,
                                            betta_grad,
                                            inp_grad_out,
                                            bsz,
                                            config_.hiddenDim,
                                            stream,
                                            !config_.useMean,
                                            betta);
    }

    inline bool UseMean() const { return config_.useMean; }

    inline void SetVar(T* variance)
    {
        if (!variance) { throw std::runtime_error("Normalize variance is null."); }
        vars = variance;
    }

    inline void SetMean(T* mean)
    {
        if (!mean) { throw std::runtime_error("Normalize mean is null."); }
        means = mean;
    }

private:
    Config config_;
    T* vars;
    T* means;
    T* vals_hat;
};
