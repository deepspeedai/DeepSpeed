// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "tensor_cast.h"

at::Tensor cast_to_byte_tensor(at::Tensor& src_tensor)
{
    if (src_tensor.nbytes() <= 1) return src_tensor;

    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .layout(src_tensor.layout())
                       .device(src_tensor.device());
    return at::from_blob(
        src_tensor.data_ptr(), static_cast<long int>(src_tensor.nbytes()), options);
}

std::vector<at::Tensor> cast_to_byte_tensor(std::vector<at::Tensor>& tensor_list)
{
    std::vector<at::Tensor> byte_tensors;
    for (auto src_tensor : tensor_list) { byte_tensors.push_back(cast_to_byte_tensor(src_tensor)); }

    return byte_tensors;
}
