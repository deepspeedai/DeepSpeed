// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adam.h"

// C++ interface

void multi_tensor_adam(int chunk_size,
                       at::Tensor noop_flag,
                       std::vector<std::vector<at::Tensor>> tensor_lists, /*gpmv*/
                       const float lr,
                       const float beta1,
                       const float beta2,
                       const float epsilon,
                       const int step,
                       const int mode,
                       const int bias_correction,
                       const float weight_decay)
{
    static bool initialized = false;
    if (!initialized) {
        create_adam_optimizer(0);
        initialized = true;
    }
    for (int i = 0; i < tensor_lists[0].size(); i++) {
        ds_adam_step(0,
                     step,
                     lr,
                     beta1,
                     beta2,
                     epsilon,
                     weight_decay,
                     bias_correction,
                     tensor_lists[1][i],
                     tensor_lists[0][i],
                     tensor_lists[2][i],
                     tensor_lists[3][i]);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("multi_tensor_adam",
          &multi_tensor_adam,
          "Compute and apply gradient update to parameters for Adam optimizer");
}
