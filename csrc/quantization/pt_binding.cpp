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

at::Tensor dequantize_int4_to_half_experimental(at::Tensor& data_in,
                                                at::Tensor& scale_buffer,
                                                at::Tensor& min_val_buffer,
                                                int num_group,
                                                int group_size)
{
    auto output_options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
    auto output = torch::empty({num_group, group_size}, output_options);

    launch_dequantize_int4_to_half_experimental((uint8_t*)data_in.data_ptr(),
                                                (half*)output.data_ptr(),
                                                (half*)scale_buffer.data_ptr(),
                                                (half*)min_val_buffer.data_ptr(),
                                                num_group,
                                                group_size,
                                                at::cuda::getCurrentCUDAStream());

    return output;
}

std::vector<at::Tensor> ds_swizzle_quant(at::Tensor& input_vals,
                                         int groups,
                                         int num_bits,
                                         quantize::Type quant_type,
                                         int pipeline_size,
                                         int nodes,
                                         int devices_per_node)
{
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
    auto scales = torch::empty({groups, scales_elems}, scales_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    const int quantization_scalar = 8 / num_bits;
    const int compressed_vals = at::numel(input_vals) / quantization_scalar;

    auto output = torch::empty({compressed_vals}, output_options);
    const int elems_per_group = at::numel(input_vals) / groups;

    launch_swizzled_quant((int8_t*)output.data_ptr(),
                          (float*)scales.data_ptr(),
                          (__half*)input_vals.data_ptr(),
                          num_bits,
                          quant_type,
                          groups,
                          elems_per_group,
                          pipeline_size,
                          nodes,
                          devices_per_node,
                          at::cuda::getCurrentCUDAStream());

    return {output, scales};
}

std::vector<at::Tensor> quantized_reduction(at::Tensor& input_vals,
                                            at::Tensor& input_scales,
                                            int in_groups,
                                            int out_groups,
                                            int num_bits,
                                            quantize::Type quant_type,
                                            int devices_per_node)
{
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
    auto scales = torch::empty({out_groups, scales_elems}, scales_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    std::vector<long int> sz(input_vals.sizes().begin(), input_vals.sizes().end());
    sz[sz.size() - 1] = sz.back() / devices_per_node;  // num of GPU per nodes
    const int elems_per_in_tensor = at::numel(input_vals) / devices_per_node;
    auto output = torch::empty(sz, output_options);

    const int elems_per_in_group = elems_per_in_tensor / (in_groups / devices_per_node);
    const int elems_per_out_group = elems_per_in_tensor / out_groups;

    launch_dequant_reduce((int8_t*)output.data_ptr(),
                          (float*)scales.data_ptr(),
                          (const int8_t*)input_vals.data_ptr(),
                          (const float*)input_scales.data_ptr(),
                          devices_per_node,
                          num_bits,
                          quant_type,
                          out_groups,
                          elems_per_out_group,
                          elems_per_in_tensor,
                          in_groups / devices_per_node,
                          elems_per_in_group,
                          at::cuda::getCurrentCUDAStream());
    return {output, scales};
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
    m.def("dequantize_int4_to_half_experimental",
          &dequantize_int4_to_half_experimental,
          "Dequantize int4 to half (experimental)");
    m.def("swizzle_quant", &ds_swizzle_quant);
    m.def("quantized_reduction", &quantized_reduction);
}
