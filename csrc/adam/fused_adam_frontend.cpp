// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

void multi_tensor_adam_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const float epsilon,
                            const int step,
                            const int mode,
                            const int bias_correction,
                            const float weight_decay);

void multi_tensor_adam_mixed_precision_cuda(int chunk_size,
                                            at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr,
                                            const float beta1,
                                            const float beta2,
                                            const float epsilon,
                                            const int step,
                                            const int mode,
                                            const int bias_correction,
                                            const float weight_decay,
                                            const float grad_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("multi_tensor_adam",
          &multi_tensor_adam_cuda,
          "Compute and apply gradient update to parameters for Adam optimizer");
    m.def("multi_tensor_adam_mixed_precision",
          &multi_tensor_adam_mixed_precision_cuda,
          "Update fp32 Adam state from low-precision gradients and write low-precision parameters");
}
