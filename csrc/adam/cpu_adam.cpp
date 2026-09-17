// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace pybind11::literals;
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("adam_rollback", &ds_adam_rollback, "DeepSpeed CPU Adam rollback (C++)");
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
