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
        bool training, save_vals;
        bool allocateGrad;
        bool useMean;
        Config(uint32_t batch,
               uint32_t seq,
               uint32_t h,
               bool training,
               bool save_vals = true,
               bool allocateGrad = true,
               bool useMean = true)
            : batchSize(batch),
              seqLength(seq),
              hiddenDim(h),
              epsilon(1e-12),
              training(training),
              save_vals(save_vals),
              allocateGrad(allocateGrad),
              useMean(useMean)
        {
        }
    };

    Normalize_Layer(Config config) : config_(config), vars(nullptr), vals_hat(nullptr)
    {
        if (config_.training) {
            cudaMalloc((void**)&vars, config_.batchSize * config_.seqLength * sizeof(T));

            if (config_.useMean)
                cudaMalloc((void**)&means, config_.batchSize * config_.seqLength * sizeof(T));

            if (config_.save_vals)
                cudaMalloc((void**)&vals_hat,
                           config_.batchSize * config_.seqLength * config_.hiddenDim * sizeof(T));

            if (config_.allocateGrad)
                cudaMalloc((void**)&inp_grad,
                           config_.batchSize * config_.seqLength * config_.hiddenDim * sizeof(T));
        }
    }

    ~Normalize_Layer()
    {
        if (config_.training) {
            cudaFree(vars);
            if (config_.useMean) cudaFree(means);
            if (config_.save_vals) cudaFree(vals_hat);
            if (config_.allocateGrad) cudaFree(inp_grad);
        }
    }

    void ForwardCheckpoint(int bsz,
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
                                        config_.seqLength,
                                        config_.hiddenDim,
                                        stream,
                                        preLayerNorm,
                                        config_.training,
                                        vars,
                                        means,
                                        vals_hat);
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
                                        config_.seqLength,
                                        config_.hiddenDim,
                                        stream,
                                        preLayerNorm,
                                        config_.training,
                                        vars,
                                        vals_hat,
                                        config_.save_vals);
    }

    void Backward(int bsz,
                  const T* out_grad,
                  const T* gamma,
                  T* gamma_grad,
                  T* betta_grad,
                  cudaStream_t stream[2],
                  T* inp_grad_out = nullptr,
                  const T* norm_in = nullptr)
    {
        launch_layerNorm_backward(out_grad,
                                  norm_in,
                                  vars,
                                  means,
                                  gamma,
                                  gamma_grad,
                                  betta_grad,
                                  (config_.allocateGrad ? inp_grad : inp_grad_out),
                                  bsz,
                                  config_.seqLength,
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
                  T* inp_grad_out = nullptr,
                  const T* norm_out = nullptr)
    {
        launch_layerNorm_backward(out_grad,
                                  (config_.save_vals ? vals_hat : norm_out),
                                  vars,
                                  gamma,
                                  gamma_grad,
                                  betta_grad,
                                  (config_.allocateGrad ? inp_grad : inp_grad_out),
                                  bsz,
                                  config_.seqLength,
                                  config_.hiddenDim,
                                  stream,
                                  config_.save_vals,
                                  betta);
    }

    void BackwardFusedAdd(int bsz,
                          const T* out_grad1,
                          const T* out_grad2,
                          const T* gamma,
                          T* gamma_grad,
                          T* betta_grad,
                          cudaStream_t stream[2],
                          T* inp_grad_out = nullptr,
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
                                            (config_.allocateGrad ? inp_grad : inp_grad_out),
                                            bsz,
                                            config_.seqLength,
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
                          T* inp_grad_out = nullptr,
                          const T* norm_out = nullptr)
    {
        launch_layerNorm_backward_fused_add(out_grad1,
                                            out_grad2,
                                            (config_.save_vals ? vals_hat : norm_out),
                                            vars,
                                            gamma,
                                            gamma_grad,
                                            betta_grad,
                                            (config_.allocateGrad ? inp_grad : inp_grad_out),
                                            bsz,
                                            config_.seqLength,
                                            config_.hiddenDim,
                                            stream,
                                            config_.save_vals,
                                            betta);
    }

    inline T* GetInputGrad() const { return inp_grad; }

    inline bool UseMean() const { return config_.useMean; }

private:
    Config config_;
    T* vars;
    T* means;
    T* vals_hat;
    T* inp_grad;
};
