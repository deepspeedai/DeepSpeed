/*
Copyright 2023 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for managing CPU tensors occupying page-locked memory.
TODO: Implement a full-featured manager that
 1. Avoid page-locked memory leaks
 2. Minimize page-locked memory usage by reducing internal fragmentation
*/

#include <map>
#include "deepspeed_py_aio.h"

struct deepspeed_pin_tensor_t {
    std::map<void*, size_t> _locked_tensors;

    deepspeed_pin_tensor_t() = default;

    ~deepspeed_pin_tensor_t();

    torch::Tensor alloc(const size_t num_elem, const at::ScalarType& elem_type);

    bool free(torch::Tensor& locked_tensor);
};
