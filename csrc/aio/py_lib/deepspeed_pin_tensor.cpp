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
    }
    _locked_tensors.clear();
}

torch::Tensor deepspeed_pin_tensor_t::alloc(const size_t num_elem, const at::ScalarType& elem_type)
{
    const auto num_bytes = num_elem * elementSize(elem_type);
    auto pinned_buffer = ds_page_aligned_alloc(num_bytes, true);
    assert(nullptr != pinned_buffer);

    _locked_tensors[pinned_buffer] = num_bytes;

    auto options = torch::TensorOptions().dtype(elem_type).device(torch::kCPU);

    return at::from_blob(pinned_buffer, static_cast<long int>(num_bytes), options);
}

bool deepspeed_pin_tensor_t::free(torch::Tensor& locked_tensor)
{
    auto addr = locked_tensor.data_ptr();
    if (_locked_tensors.find(addr) != _locked_tensors.end()) {
        munlock(addr, _locked_tensors[addr]);
        _locked_tensors.erase(addr);
        return true;
    }

    return false;
}
