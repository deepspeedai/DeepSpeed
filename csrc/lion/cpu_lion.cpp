// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_lion.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lion_update", &ds_lion_step, "DeepSpeed CPU Lion update (C++)");
    m.def("lion_update_copy",
          &ds_lion_step_plus_copy,
          "DeepSpeed CPU Lion update and param copy (C++)");
    m.def("create_lion", &create_lion_optimizer, "DeepSpeed CPU Lion (C++)");
    m.def("destroy_lion", &destroy_lion_optimizer, "DeepSpeed CPU Lion destroy (C++)");
}
