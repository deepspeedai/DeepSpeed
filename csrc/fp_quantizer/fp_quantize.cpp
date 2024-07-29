// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "fp_quantize.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <vector>

#define DISPATCH_QUANTIZE(T_TYPE, C_TYPE, mantisa, exponent)                             \
    if (val.options().dtype() == torch::T_TYPE) {                                        \
        launch_quantization<C_TYPE, mantisa, exponent>((C_TYPE*)val.data_ptr(),          \
                                                       (uint8_t*)out.data_ptr(),         \
                                                       num_groups,                       \
                                                       group_size,                       \
                                                       at::cuda::getCurrentCUDAStream(), \
                                                       q_range,                          \
                                                       q_bits,                           \
                                                       q_mantisa_bits,                   \
                                                       stochastic_rounding);             \
    }

at::Tensor quantize(torch::Tensor& out,
                    torch::Tensor& val,
                    int group_size,
                    int stochastic_rounding,
                    int q_bits,
                    int q_mantisa_bits)
{
    int total_elems = at::numel(val);
    float q_range = q_bits == 8 ? (q_mantisa_bits == 3 ? 480.0 : 114688.0) :  // fp8 ranges
                        (q_bits == 12 ? 510.0 :                               // fp12 range
                             (q_bits == 6 ? 28.0 :                            // fp6 range
                                  6.0));  // fp4 range (using power 2); TODO (Reza): add the power-4
                                          // in case accuracy is not matching!
    int num_groups = total_elems / group_size;

    DISPATCH_QUANTIZE(kHalf, __half, 23, 8);
#ifdef BF16_AVAILABLE
    DISPATCH_QUANTIZE(kBFloat16, __nv_bfloat16, 23, 8);
#endif

    return out;
}

#define DISPATCH_DEQUANTIZE(T_TYPE, C_TYPE, mantisa)                              \
    if (val.options().dtype() == torch::T_TYPE) {                                 \
        launch_dequantization<C_TYPE, mantisa>((uint8_t*)val_q.data_ptr(),        \
                                               (C_TYPE*)val.data_ptr(),           \
                                               num_groups,                        \
                                               group_size,                        \
                                               q_mantisa_bits,                    \
                                               q_exponent_bits,                   \
                                               at::cuda::getCurrentCUDAStream()); \
        return;                                                                   \
    }

void dequantize(torch::Tensor& val,
                torch::Tensor& val_q,
                int group_size,
                int q_mantisa_bits,
                int q_exponent_bits)
{
    int total_elems = at::numel(val);

    int num_groups = total_elems / group_size;

    DISPATCH_DEQUANTIZE(kHalf, __half, 10);
#ifdef BF16_AVAILABLE
    DISPATCH_DEQUANTIZE(kBFloat16, __nv_bfloat16, 7);
#endif
}

#define DISPATCH_DEQUANTIZE_INDEX(T_TYPE, C_TYPE, mantisa)                                  \
    if (val.options().dtype() == torch::T_TYPE) {                                           \
        launch_selective_dequantization<C_TYPE, mantisa>((uint8_t*)val_q.data_ptr(),        \
                                                         (C_TYPE*)val.data_ptr(),           \
                                                         (int32_t*)indexes.data_ptr(),      \
                                                         num_groups,                        \
                                                         group_size,                        \
                                                         num_indexes,                       \
                                                         q_mantisa_bits,                    \
                                                         q_exponent_bits,                   \
                                                         at::cuda::getCurrentCUDAStream()); \
        return;                                                                             \
    }
void selective_dequantize(torch::Tensor& val,
                          torch::Tensor& val_q,
                          torch::Tensor& indexes,
                          int group_size,
                          int q_mantisa_bits,
                          int q_exponent_bits)
{
    int total_elems = at::numel(val);
    int num_indexes = indexes.size(0);
    int num_groups = total_elems / group_size;

    DISPATCH_DEQUANTIZE_INDEX(kHalf, __half, 10);
#ifdef BF16_AVAILABLE
    DISPATCH_DEQUANTIZE_INDEX(kBFloat16, __nv_bfloat16, 7);
#endif
}

at::Tensor get_scales(torch::Tensor& out, int num_groups)
{
    auto options = at::TensorOptions()
                       .dtype(torch::kFloat)
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto scales =
        torch::from_blob(out.data_ptr(), {num_groups, 1}, {out.stride(0) / 4, 1}, options);
    return scales;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize", &quantize, "quantize function");
    m.def("dequantize", &dequantize, "dequantize function");
    m.def("get_scales", &get_scales, "get scales function");
    m.def("selective_dequantize", &selective_dequantize, "selective dequantize function");
}
