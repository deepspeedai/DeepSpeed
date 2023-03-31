// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Copyright NVIDIA/apex
This file is adapted from fused adam in NVIDIA/apex, commit a109f85
*/

#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/extension.h>
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_flatten.h

at::Tensor flatten(std::vector<at::Tensor> tensors)
{
    return torch::utils::flatten_dense_tensors(tensors);
}

std::vector<at::Tensor> unflatten(at::Tensor flat, std::vector<at::Tensor> tensors)
{
    return torch::utils::unflatten_dense_tensors(flat, tensors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flatten", &flatten, "Flatten dense tensors");
    m.def("unflatten", &unflatten, "Unflatten dense tensors");
}
