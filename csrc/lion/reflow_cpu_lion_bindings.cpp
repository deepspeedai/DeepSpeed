// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

#include "reflow_bindings.h"
#include "reflow_cpu_lion.h"

// Memory fence to enforce store ordering after non-temporal (streaming) stores.
static void reflow_lion_memory_fence()
{
#if defined(__AVX512__) or defined(__AVX256__)
    _mm_mfence();
#endif
}

static inline int reflow_ds_lion_step_params_halfgrad_gilfree(int optimizer_id,
                                                              size_t step,
                                                              float lr,
                                                              float beta1,
                                                              float beta2,
                                                              float weight_decay,
                                                              torch::Tensor& params_fp32,
                                                              torch::Tensor& grads,
                                                              torch::Tensor& exp_avg,
                                                              torch::Tensor& half_params,
                                                              float combined_scale,
                                                              bool skip_increment_step = false)
{
    pybind11::gil_scoped_release release;
    return reflow_ds_lion_step_params_halfgrad(optimizer_id,
                                               step,
                                               lr,
                                               beta1,
                                               beta2,
                                               weight_decay,
                                               params_fp32,
                                               grads,
                                               exp_avg,
                                               half_params,
                                               combined_scale,
                                               skip_increment_step);
}

static inline int reflow_ds_lion_state_step_halfgrad_gilfree(int optimizer_id,
                                                             size_t step,
                                                             float lr,
                                                             float beta1,
                                                             float beta2,
                                                             float weight_decay,
                                                             torch::Tensor& params_fp32,
                                                             torch::Tensor& grads,
                                                             torch::Tensor& exp_avg,
                                                             float combined_scale,
                                                             bool skip_increment_step = false)
{
    pybind11::gil_scoped_release release;
    return reflow_ds_lion_state_step_halfgrad(optimizer_id,
                                              step,
                                              lr,
                                              beta1,
                                              beta2,
                                              weight_decay,
                                              params_fp32,
                                              grads,
                                              exp_avg,
                                              combined_scale,
                                              skip_increment_step);
}

void bind_reflow_lion(pybind11::module_& m)
{
    m.def("reflow_create_lion",
          &reflow_create_lion_optimizer,
          "Reflow CPU Lion create (C++)",
          pybind11::arg("optimizer_id"),
          pybind11::arg("alpha") = 1e-3,
          pybind11::arg("betta1") = 0.9,
          pybind11::arg("betta2") = 0.999,
          pybind11::arg("weight_decay") = 0,
          pybind11::arg("should_log") = false,
          pybind11::arg("num_threads") = -1);
    m.def("reflow_destroy_lion", &reflow_destroy_lion_optimizer, "Reflow CPU Lion destroy (C++)");
    m.def("reflow_lion_update_params_halfgrad",
          &reflow_ds_lion_step_params_halfgrad_gilfree,
          "Reflow CPU Lion update FP16 with half gradients, GIL-released (C++)",
          pybind11::arg("optimizer_id"),
          pybind11::arg("step"),
          pybind11::arg("lr"),
          pybind11::arg("beta1"),
          pybind11::arg("beta2"),
          pybind11::arg("weight_decay"),
          pybind11::arg("params_fp32"),
          pybind11::arg("grads"),
          pybind11::arg("exp_avg"),
          pybind11::arg("half_params"),
          pybind11::arg("combined_scale"),
          pybind11::arg("skip_increment_step") = false);
    m.def("reflow_lion_update_state_halfgrad",
          &reflow_ds_lion_state_step_halfgrad_gilfree,
          "Reflow CPU Lion update state stream with half gradients, GIL-released (C++)",
          pybind11::arg("optimizer_id"),
          pybind11::arg("step"),
          pybind11::arg("lr"),
          pybind11::arg("beta1"),
          pybind11::arg("beta2"),
          pybind11::arg("weight_decay"),
          pybind11::arg("params_fp32"),
          pybind11::arg("grads"),
          pybind11::arg("exp_avg"),
          pybind11::arg("combined_scale"),
          pybind11::arg("skip_increment_step") = false);
    m.def("reflow_lion_memory_fence",
          &reflow_lion_memory_fence,
          "Reflow CPU Lion memory fence (C++)");
}
