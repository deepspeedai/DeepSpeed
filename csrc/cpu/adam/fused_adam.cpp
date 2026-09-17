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
    // ds_adam_step reads lr/betas/eps/weight_decay per call; only AdamW-vs-L2 is fixed at
    // construction, so keep one optimizer instance per mode (mode 1 == AdamW, as in CUDA).
    // The constructor hyperparameters are placeholders: update_state overwrites them on every
    // step, so any value works here.
    constexpr float kPlaceholderHyperparam = 0.0f;
    static bool initialized[2] = {false, false};
    const int optimizer_id = mode;
    if (!initialized[mode]) {
        create_adam_optimizer(optimizer_id,
                              kPlaceholderHyperparam,
                              kPlaceholderHyperparam,
                              kPlaceholderHyperparam,
                              kPlaceholderHyperparam,
                              kPlaceholderHyperparam,
                              mode == 1);
        initialized[mode] = true;
    }
    for (int i = 0; i < tensor_lists[0].size(); i++) {
        ds_adam_step(optimizer_id,
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
