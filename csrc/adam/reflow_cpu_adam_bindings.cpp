// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

#include "reflow_bindings.h"
#include "reflow_cpu_adam.h"

// Memory fence to enforce store ordering after non-temporal (streaming) stores.
static void reflow_adam_memory_fence()
{
#if defined(__AVX512__) or defined(__AVX256__)
    _mm_mfence();
#endif
}

static inline int reflow_ds_adam_step_params_halfgrad_gilfree(int optimizer_id,
                                                              size_t step,
                                                              float lr,
                                                              float beta1,
                                                              float beta2,
                                                              float epsilon,
                                                              float weight_decay,
                                                              bool bias_correction,
                                                              torch::Tensor& params_fp32,
                                                              torch::Tensor& grads,
                                                              torch::Tensor& exp_avg,
                                                              torch::Tensor& exp_avg_sq,
                                                              torch::Tensor& half_params,
                                                              float combined_scale,
                                                              bool skip_increment_step = false,
                                                              bool maximize = false)
{
    pybind11::gil_scoped_release release;
    return reflow_ds_adam_step_params_halfgrad(optimizer_id,
                                               step,
                                               lr,
                                               beta1,
                                               beta2,
                                               epsilon,
                                               weight_decay,
                                               bias_correction,
                                               params_fp32,
                                               grads,
                                               exp_avg,
                                               exp_avg_sq,
                                               half_params,
                                               combined_scale,
                                               skip_increment_step,
                                               maximize);
}

static inline int reflow_ds_adam_state_step_halfgrad_gilfree(int optimizer_id,
                                                             size_t step,
                                                             float lr,
                                                             float beta1,
                                                             float beta2,
                                                             float epsilon,
                                                             float weight_decay,
                                                             bool bias_correction,
                                                             torch::Tensor& params_fp32,
                                                             torch::Tensor& grads,
                                                             torch::Tensor& exp_avg,
                                                             torch::Tensor& exp_avg_sq,
                                                             float combined_scale,
                                                             bool skip_increment_step = false,
                                                             bool maximize = false)
{
    pybind11::gil_scoped_release release;
    return reflow_ds_adam_state_step_halfgrad(optimizer_id,
                                              step,
                                              lr,
                                              beta1,
                                              beta2,
                                              epsilon,
                                              weight_decay,
                                              bias_correction,
                                              params_fp32,
                                              grads,
                                              exp_avg,
                                              exp_avg_sq,
                                              combined_scale,
                                              skip_increment_step,
                                              maximize);
}

static inline int reflow_ds_bf16_accumulate_gilfree(torch::Tensor& dst, torch::Tensor& src)
{
    pybind11::gil_scoped_release release;
    return reflow_ds_bf16_accumulate(dst, src);
}

void bind_reflow_adam(pybind11::module_& m)
{
    m.def("reflow_create_adam",
          &reflow_create_adam_optimizer,
          "Reflow CPU Adam create (C++)",
          pybind11::arg("optimizer_id"),
          pybind11::arg("alpha") = 1e-3,
          pybind11::arg("betta1") = 0.9,
          pybind11::arg("betta2") = 0.999,
          pybind11::arg("eps") = 1e-8,
          pybind11::arg("weight_decay") = 0,
          pybind11::arg("adamw_mode") = true,
          pybind11::arg("should_log") = false,
          pybind11::arg("num_threads") = -1);
    m.def("reflow_destroy_adam", &reflow_destroy_adam_optimizer, "Reflow CPU Adam destroy (C++)");
    m.def("reflow_adam_update_params_halfgrad",
          &reflow_ds_adam_step_params_halfgrad_gilfree,
          "Reflow CPU Adam update FP16 with half gradients, GIL-released (C++)",
          pybind11::arg("optimizer_id"),
          pybind11::arg("step"),
          pybind11::arg("lr"),
          pybind11::arg("beta1"),
          pybind11::arg("beta2"),
          pybind11::arg("epsilon"),
          pybind11::arg("weight_decay"),
          pybind11::arg("bias_correction"),
          pybind11::arg("params_fp32"),
          pybind11::arg("grads"),
          pybind11::arg("exp_avg"),
          pybind11::arg("exp_avg_sq"),
          pybind11::arg("half_params"),
          pybind11::arg("combined_scale"),
          pybind11::arg("skip_increment_step") = false,
          pybind11::arg("maximize") = false);
    m.def("reflow_adam_update_state_halfgrad",
          &reflow_ds_adam_state_step_halfgrad_gilfree,
          "Reflow CPU Adam update state stream with half gradients, GIL-released (C++)",
          pybind11::arg("optimizer_id"),
          pybind11::arg("step"),
          pybind11::arg("lr"),
          pybind11::arg("beta1"),
          pybind11::arg("beta2"),
          pybind11::arg("epsilon"),
          pybind11::arg("weight_decay"),
          pybind11::arg("bias_correction"),
          pybind11::arg("params_fp32"),
          pybind11::arg("grads"),
          pybind11::arg("exp_avg"),
          pybind11::arg("exp_avg_sq"),
          pybind11::arg("combined_scale"),
          pybind11::arg("skip_increment_step") = false,
          pybind11::arg("maximize") = false);
    m.def("reflow_adam_memory_fence",
          &reflow_adam_memory_fence,
          "Reflow CPU Adam memory fence (C++)");
    m.def("reflow_bf16_accumulate",
          &reflow_ds_bf16_accumulate_gilfree,
          "Reflow CPU BF16/FP16 gradient accumulate using AVX, GIL-released (C++)",
          pybind11::arg("dst"),
          pybind11::arg("src"));
}
