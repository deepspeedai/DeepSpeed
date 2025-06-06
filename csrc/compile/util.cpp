// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#include <ATen/ATen.h>

namespace dc {

std::string tensorToString(const at::Tensor& t, size_t max_elem, size_t max_str_len)
{
    auto t_cpu = t.flatten()
                     .slice(0, 0, std::min((int64_t)max_elem, t.numel()))
                     .to(c10::Device(c10::kCPU), false, true);

    size_t size = std::min(max_elem, productDim(t.sizes()));

    if (t.scalar_type() == c10::ScalarType::Half || t.scalar_type() == c10::ScalarType::BFloat16) {
        auto float_ten = t_cpu.to(c10::ScalarType::Float, false, true).contiguous();
        return tensorPtrToString((float*)float_ten.data_ptr(), size, max_str_len);
    } else if (t.scalar_type() == c10::ScalarType::Float) {
        return tensorPtrToString((float*)t_cpu.data_ptr(), size, max_str_len);
    } else if (t.scalar_type() == c10::ScalarType::Double) {
        return tensorPtrToString((double*)t_cpu.data_ptr(), size, max_str_len);
    } else if (t.scalar_type() == c10::ScalarType::Int) {
        int* ptr = static_cast<int*>(t_cpu.data_ptr());
        return tensorPtrToString(ptr, size, max_str_len);
    } else if (t.scalar_type() == c10::ScalarType::Long) {
        long* ptr = static_cast<long*>(t_cpu.data_ptr());
        return tensorPtrToString(ptr, size, max_str_len);
    } else if (t.scalar_type() == c10::ScalarType::Byte) {
        unsigned char* ptr = static_cast<unsigned char*>(t_cpu.data_ptr());
        std::vector<unsigned short> vec;
        vec.reserve(size);
        for (size_t i = 0; i < size; i++) {
            vec.push_back(*ptr);
            ptr++;
        }
        return tensorPtrToString(&vec[0], size, max_str_len);
    } else if (t.scalar_type() == c10::ScalarType::Bool) {
        bool* ptr = static_cast<bool*>(t_cpu.data_ptr());
        std::vector<int> vec;
        vec.reserve(size);
        for (size_t i = 0; i < size; i++) {
            vec.push_back(*ptr);
            ptr++;
        }
        return tensorPtrToString(&vec[0], size, max_str_len);
    }
    std::stringstream ss;
    ss << "Failed to convert tensor to string. Invalid type of tensor: "
       << toString(t.scalar_type());
    throw std::invalid_argument(ss.str());
}

std::string tensorPtrToString(void* ptr,
                              size_t size,
                              c10::ScalarType datatype,
                              size_t max_elem,
                              size_t max_str_len)
{
    int64_t elem_size = std::min((size_t)max_elem, size);

    if (datatype == c10::ScalarType::Long) {
        return tensorPtrToString(static_cast<long*>(ptr), elem_size, max_str_len);
    } else if (datatype == c10::ScalarType::Int) {
        return tensorPtrToString(static_cast<int*>(ptr), elem_size, max_str_len);
    } else if (datatype == c10::ScalarType::Double) {
        return tensorPtrToString(static_cast<double*>(ptr), elem_size, max_str_len);
    } else if (datatype == c10::ScalarType::Float) {
        return tensorPtrToString(static_cast<float*>(ptr), elem_size, max_str_len);
    } else if (datatype == c10::ScalarType::Half || datatype == c10::ScalarType::BFloat16) {
        const auto ten = torch::from_blob(ptr, {(int64_t)elem_size}, datatype);
        auto float_ten = ten.to(c10::ScalarType::Float, false, true).contiguous();
        return tensorPtrToString((float*)float_ten.data_ptr(), elem_size, max_str_len);
    }
    std::stringstream ss;
    ss << "Failed to convert tensor ptr to string. Invalid type of tensor: " << toString(datatype);
    throw std::invalid_argument(ss.str());
}

std::string tensorDimToString(const at::Tensor& t)
{
    const auto dim = t.sizes();
    return join_as_str(dim);
}
}  // namespace dc
