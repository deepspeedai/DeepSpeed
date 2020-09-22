#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "custom_cuda_layers.h"

template <typename T>
class Gelu {
public:
    struct Config {
        uint32_t intermediate_size;
        Config(uint32_t inter_size) : intermediate_size(inter_size) {}
    };

    Gelu(const Config& config) : _config(config) {}

    virtual ~Gelu() {}

    void ForwardWithBiasAdd(int bsz,
                            const T* input_buf,
                            const T* bias,
                            T* output,
                            cudaStream_t stream)
    {
        launch_bias_gelu<T>(input_buf, bias, output, _config.intermediate_size, bsz, stream);
    }

    void Backward(int bsz, T* d_output, const T* input_buf, const T* bias, cudaStream_t stream)
    {
        launch_d_gelu<T>(d_output, input_buf, bias, _config.intermediate_size, bsz, stream);
    }

private:
    Config _config;
};
