#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

template <typename T>
class Dropout {
public:
    struct Config {
        float ratio;
        uint32_t batch, dim;
        bool training;

        Config(float r, uint32_t batch, uint32_t dim)
            : ratio(r), batch(batch), dim(dim), training(true)
        {
        }

        float RATIO() const { return training ? ratio : 0.0; }
    };

    Dropout(const Config& config) : _config(config), _mask(nullptr) {}

    virtual ~Dropout() {}

    void Forward(int bsz, T* out, const T* vals, cudaStream_t stream, bool bwd = false)
    {
        launch_dropout<T>(
            out, vals, _mask, bsz * _config.dim, _config.dim, _config.RATIO(), stream, bwd);
    }

    void ForwardWithBias(int bsz, T* vals, const T* bias, cudaStream_t stream)
    {
        launch_dropout<T>(vals, bias, _mask, bsz, _config.dim, _config.RATIO(), stream);
    }

    void ForwardWithBias(int bsz,
                         T* out,
                         const T* vals,
                         const T* residual,
                         const T* bias,
                         cudaStream_t stream)
    {
        launch_dropout<T>(
            out, vals, residual, bias, _mask, bsz, _config.dim, _config.RATIO(), stream);
    }

    void Backward(int bsz, T* d_vals, cudaStream_t stream)
    {
        launch_dropout_grad<T>(d_vals, _mask, bsz * _config.dim, _config.RATIO(), stream);
    }

    void Backward(int bsz, T* d_vals_out, const T* d_vals, cudaStream_t stream)
    {
        launch_dropout_grad<T>(
            d_vals_out, d_vals, _mask, bsz * _config.dim, _config.RATIO(), stream);
    }

    bool HasDropout() const { return _config.RATIO() > 0.0; }

    void SetTrainingMode(bool training) { _config.training = training; }

    void SetMask(uint8_t* mask)
    {
        if (!mask) { throw std::runtime_error("Dropout mask is null."); }

        _mask = mask;
    }

    Config GetConfig() const { return _config; }

private:
    uint8_t* _mask;
    Config _config;
};
