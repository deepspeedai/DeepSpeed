// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
    Collection of system utilities.
*/

#include <torch/extension.h>
#include "tensor_cast.h"
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cast_to_byte_tensor",
          py::overload_cast<at::Tensor&>(&cast_to_byte_tensor),
          "Cast a 1-dimensional tensor of any type to byte tensor.",
          "src_tensor"_a);

    m.def("cast_to_byte_tensor",
          py::overload_cast<std::vector<at::Tensor>&>(&cast_to_byte_tensor),
          "Cast a multi-dimensional tensor of any type to byte tensor.",
          "src_tensor"_a);
}
