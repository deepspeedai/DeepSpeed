// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

void multi_tensor_lion_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const int step,
                            const float weight_decay);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("multi_tensor_lion",
          &multi_tensor_lion_cuda,
          "Compute and apply gradient update to parameters for Lion optimizer");
}
