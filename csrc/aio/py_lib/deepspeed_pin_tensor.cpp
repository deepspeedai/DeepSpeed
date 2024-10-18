// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for managing CPU tensors occupying page-locked memory.
*/

#include "deepspeed_pin_tensor.h"

using namespace std;

deepspeed_pin_tensor_t::~deepspeed_pin_tensor_t()
{
    for (auto iter = _locked_tensors.begin(); iter != _locked_tensors.end(); ++iter) {
        munlock(iter->first, iter->second);
        std::free((void*)iter->first);
    }
    _locked_tensors.clear();
}

torch::Tensor deepspeed_pin_tensor_t::alloc(const int64_t num_elem,
                                            const torch::TensorOptions& options)
{
    const auto scalar_dtype = torch::typeMetaToScalarType(options.dtype());
    const auto num_bytes = num_elem * torch::elementSize(scalar_dtype);
    auto pinned_buffer = ds_page_aligned_alloc(num_bytes, true);
    assert(nullptr != pinned_buffer);

    _locked_tensors[pinned_buffer] = num_bytes;

    return at::from_blob(pinned_buffer, static_cast<int64_t>(num_elem), options);
}

torch::Tensor deepspeed_pin_tensor_t::alloc(const int64_t num_elem, const at::ScalarType& elem_type)
{
    auto options = torch::TensorOptions().dtype(elem_type).device(torch::kCPU).requires_grad(false);
    return alloc(num_elem, options);
}

bool deepspeed_pin_tensor_t::free(torch::Tensor& locked_tensor)
{
    auto addr = locked_tensor.data_ptr();
    if (_locked_tensors.find(addr) != _locked_tensors.end()) {
        munlock(addr, _locked_tensors[addr]);
        std::free(addr);
        _locked_tensors.erase(addr);
        return true;
    }

    return false;
}

bool deepspeed_pin_tensor_t::is_managed(const torch::Tensor& buffer)
{
    if (!buffer.is_cpu()) { return false; }
    auto addr = buffer.data_ptr();
    if (_locked_tensors.find(addr) != _locked_tensors.end()) { return true; }
    return false;
};
