// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace pybind11::literals;
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    // `parallel` defaults to true (OpenMP across elements, as before). ZenFlow's native
    // pinned thread pool sets it false so each pool thread runs its slice serially.
    m.def("adam_update_multi",
          &ds_adam_step_multi,
          "DeepSpeed CPU Adam fused multi-tensor update (C++)",
          "optimizer_id"_a,
          "step"_a,
          "lr"_a,
          "beta1"_a,
          "beta2"_a,
          "epsilon"_a,
          "weight_decay"_a,
          "bias_correction"_a,
          "params"_a,
          "grads"_a,
          "exp_avgs"_a,
          "exp_avg_sqs"_a,
          "stale_params"_a,
          "parallel"_a = true);
    m.def("adam_rollback", &ds_adam_rollback, "DeepSpeed CPU Adam rollback (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
}
