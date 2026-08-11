// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for managing CPU tensors occupying page-locked memory.
TODO: Implement a full-featured manager that
1. Avoid page-locked memory leaks
2. Minimize page-locked memory usage by reducing internal fragmentation
*/

#pragma once

#include <torch/extension.h>
#include <map>
#include <memory>
#include <mutex>

struct deepspeed_pin_tensor_t {
    std::map<void*, int64_t> _locked_tensors;
    std::mutex _mutex;

    deepspeed_pin_tensor_t() = default;

    ~deepspeed_pin_tensor_t();

    // Process-wide shared manager so that pinned-buffer recognition is consistent
    // across every io handle (each handle references this single instance).
    // Canonical instance lives in the pin_memory extension; other ops resolve it
    // via deepspeed_pin_tensor_mgr_holder() after that extension is loaded.
    static std::shared_ptr<deepspeed_pin_tensor_t> shared();

    torch::Tensor alloc(const int64_t num_elem, const at::ScalarType& elem_type);
    torch::Tensor alloc(const int64_t num_elem, const torch::TensorOptions& options);

    bool free(torch::Tensor& locked_tensor);

    bool is_managed(const torch::Tensor& buffer);
};

// Exported so async_io/gds can resolve the same manager across .so boundaries.
extern "C" void* deepspeed_pin_tensor_mgr_holder();
