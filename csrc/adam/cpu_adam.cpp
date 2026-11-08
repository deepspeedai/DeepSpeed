// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace pybind11::literals;
    // The update, the rescale and the rollback release the GIL, so that a CPU Adam step in another
    // thread lets the training thread run.
    m.def("adam_update",
          &ds_adam_step,
          "DeepSpeed CPU Adam update (C++); grad_scale multiplies each gradient as it is read",
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
          "exp_avg"_a,
          "exp_avg_sq"_a,
          "grad_scale"_a = 1.0f,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("adam_rescale_step",
          &ds_adam_rescale_step,
          "DeepSpeed CPU Adam: turn the last step, taken on grads * old_scale, into the step on "
          "grads * new_scale (C++)",
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("adam_rollback",
          &ds_adam_rollback,
          "DeepSpeed CPU Adam rollback (C++)",
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");

    // ZenFlowAdam: the native CPU Adam backing ZenFlow's overlapped optimizer step. create /
    // register_group / destroy set up the handle-indexed pinned pool (used by the worker process).
    m.def("zenflow_adam_create", &zenflow_adam_create, "ZenFlowAdam create (C++)");
    m.def("zenflow_adam_register_group",
          &zenflow_adam_register_group,
          "ZenFlowAdam register a parameter group (C++)");
    m.def("zenflow_adam_destroy",
          &zenflow_adam_destroy,
          "ZenFlowAdam destroy (C++)",
          pybind11::call_guard<pybind11::gil_scoped_release>());

#if defined(__linux__)
    // The optimizer runs in a separate process, coordinated through a shared-memory semaphore
    // control block. submit/wait/run_worker release the GIL so the optimizer process overlaps
    // the Python training thread.
    m.def(
        "zenflow_adam_ctrl_size", &zenflow_adam_ctrl_size, "ZenFlowAdam control block size (C++)");
    m.def("zenflow_adam_ctrl_init", &zenflow_adam_ctrl_init, "ZenFlowAdam control init (C++)");
    m.def("zenflow_adam_run_worker",
          &zenflow_adam_run_worker,
          "ZenFlowAdam optimizer-process worker loop (C++)",
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("zenflow_adam_submit",
          &zenflow_adam_submit,
          "ZenFlowAdam submit an overlapped step (C++)",
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("zenflow_adam_wait",
          &zenflow_adam_wait,
          "ZenFlowAdam wait for a submitted step (C++)",
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("zenflow_adam_ctrl_exit",
          &zenflow_adam_ctrl_exit,
          "ZenFlowAdam cross-process exit (C++)",
          pybind11::call_guard<pybind11::gil_scoped_release>());
#endif
}
