// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cassert>
#include <vector>
#include "quantization.h"

template <typename T>
at::Tensor ds_quantize(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if ((((size / groups) - 1) / 4096 + 1) <= 256) {
        launch_fake_quantize_kernel(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_sr_quantize(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if (((size / groups) / 4 / 1024) <= 256) {
        launch_sr_fake_quantize_kernel(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_quantize_asym(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if ((((size / groups) - 1) / 4096 + 1) <= 256) {
        launch_fake_quantize_kernel_asym(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_sr_quantize_asym(at::Tensor& vals, int groups, int bits)
{
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if (((size / groups) / 4 / 1024) <= 256) {
        launch_sr_fake_quantize_kernel_asym(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

std::vector<at::Tensor> quantize_kernel(at::Tensor& input_vals,
                                        int groups,
                                        int numBits,
                                        quantize::Type quantType)
{
    auto dtype = at::kFloat;
    auto params_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int param_elems = (quantize::requires_offset(quantType)) ? 2 : 1;
    auto params = torch::empty({groups, param_elems}, params_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    auto output_sizes = input_vals.sizes().vec();
    output_sizes[output_sizes.size() - 1] /= numBits == 8 ? 1 : 2;
    auto output = torch::empty(output_sizes, output_options);

    const int elems_per_group = at::numel(input_vals) / groups;

    launch_quant((int8_t*)output.data_ptr(),
                 (float*)params.data_ptr(),
                 (__half*)input_vals.data_ptr(),
                 groups,
                 elems_per_group,
                 numBits,
                 quantType,
                 at::cuda::getCurrentCUDAStream());

    return {output, params};
}

template <typename T>
at::Tensor dequantize(at::Tensor& quantized_data,
                      at::Tensor& params,
                      int groups,
                      int num_bits,
                      quantize::Type quant_type)
{
    auto dtype = (std::is_same<T, float>::value) ? torch::kFloat32 : torch::kFloat16;
    auto output_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    auto output_sizes = quantized_data.sizes().vec();
    output_sizes[output_sizes.size() - 1] *= num_bits == 8 ? 1 : 2;
    auto output = torch::empty(output_sizes, output_options);

    const int total_elems = at::numel(output);
    const int elems_per_group = total_elems / groups;

    launch_dequantize_kernel((T*)output.data_ptr(),
                             (const int8_t*)quantized_data.data_ptr(),
                             (const float*)params.data_ptr(),
                             quant_type,
                             num_bits,
                             elems_per_group,
                             total_elems,
                             at::cuda::getCurrentCUDAStream());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ds_quantize_fp32", &ds_quantize<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_quantize_fp16", &ds_quantize<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_sr_quantize_fp32", &ds_sr_quantize<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_sr_quantize_fp16", &ds_sr_quantize<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_quantize_asym_fp32", &ds_quantize_asym<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def(
        "ds_quantize_asym_fp16", &ds_quantize_asym<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_sr_quantize_asym_fp32",
          &ds_sr_quantize_asym<float>,
          "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_sr_quantize_asym_fp16",
          &ds_sr_quantize_asym<__half>,
          "DeepSpeed Quantize with fp16 (CUDA)");
    pybind11::enum_<quantize::Type>(m, "QuantizationType")
        .value("Symmetric", quantize::Type::Symmetric)
        .value("Asymmetric", quantize::Type::Asymmetric)
        .export_values();
    m.def("quantize", &quantize_kernel);
    m.def("dequantize", &dequantize<__half>);
    m.def("dequantize_fp32", &dequantize<float>);
}
