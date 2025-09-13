// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"
#include "z1.h"
#include "z2.h"
#include "z3.h"

TORCH_LIBRARY(dc, m)
{
    m.def("allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor");
    m.def(
        "prefetch_params_fused(int graph_id, Tensor[] params, int[] ids,"
        "                      ScalarType[]? dtypes = None) -> ()");
    m.def("wait_allgather(Tensor(a) a, int graph_id, int id) -> Tensor(a)");
    m.def("release_param(Tensor(a) a, int graph_id, int id, int n_users) -> Tensor(a)");
    m.def("reduce_grad(Tensor a, int graph_id, int id) -> Tensor");
    m.def("free_tensors(Tensor[] a) -> ()");
    m.def("offload_tensor(Tensor a, int id, int id) -> Tensor");
    m.def("reload_tensor(Tensor a, int id, int id) -> Tensor");
    m.def("wait_offload(Tensor a, int id, int id) -> Tensor");
    m.def("wait_reload(Tensor a, int id, int id) -> Tensor");
    m.def("offload_parameter(Tensor a, int id, int id) -> ()");
    m.def("reload_parameter(Tensor a, int id, int id) -> ()");
    m.def("end_backward(int graph_id) -> ()");

    m.def("test_call(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(dc, CPU, m)
{
    m.impl("allgather_param", &dc::allgather_param);
    m.impl("prefetch_params_fused", &dc::prefetch_params_fused);
    m.impl("wait_allgather", &dc::wait_allgather);
    m.impl("release_param", &dc::release_param);
    m.impl("reduce_grad", &dc::reduce_grad);
    m.impl("free_tensors", &dc::free_tensors);
    m.impl("offload_tensor", &dc::offload_tensor);
    m.impl("reload_tensor", &dc::reload_tensor);
    m.impl("wait_offload", &dc::wait_offload);
    m.impl("wait_reload", &dc::wait_reload);
    m.impl("offload_parameter", &dc::offload_parameter);
    m.impl("reload_parameter", &dc::reload_parameter);

    m.impl("test_call", &dc::test_call);
}

TORCH_LIBRARY_IMPL(dc, CUDA, m)
{
    m.impl("allgather_param", &dc::allgather_param);
    m.impl("prefetch_params_fused", &dc::prefetch_params_fused);
    m.impl("wait_allgather", &dc::wait_allgather);
    m.impl("release_param", &dc::release_param);
    m.impl("reduce_grad", &dc::reduce_grad);
    m.impl("free_tensors", &dc::free_tensors);
    m.impl("offload_tensor", &dc::offload_tensor);
    m.impl("reload_tensor", &dc::reload_tensor);
    m.impl("wait_offload", &dc::wait_offload);
    m.impl("wait_reload", &dc::wait_reload);
    m.impl("offload_parameter", &dc::offload_parameter);
    m.impl("reload_parameter", &dc::reload_parameter);

    m.impl("test_call", &dc::test_call);
}

TORCH_LIBRARY_IMPL(dc, Meta, m)
{
    m.impl("allgather_param", &dc::allgather_param_meta);
    m.impl("prefetch_params_fused", &dc::prefetch_params_fused_meta);
    m.impl("release_param", &dc::release_param_meta);
    m.impl("wait_allgather", &dc::wait_allgather_meta);
    m.impl("reduce_grad", &dc::reduce_grad_meta);
    m.impl("free_tensors", &dc::free_tensors_meta);
    m.impl("reload_parameter", &dc::reload_parameter_meta);
    m.impl("offload_parameter", &dc::offload_parameter_meta);
}

// The "Undefined" dispatch key is for operations whose arguments do not contain
// a tensor.
TORCH_LIBRARY_IMPL(dc, Undefined, m) { m.impl("end_backward", &dc::end_backward); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("set_persistent", &dc::set_persistent, "Set persistent flag for a parameter");
    m.def("enable_profiling", &dc::enable_profiling, "Enable profiling");
    m.def("is_profiling", &dc::is_profiling, "Check if profiling is enabled");
    m.def("init", &dc::init, "Set the process group");
    m.def("cleanup", &dc::cleanup, "Cleanup the process group");
    m.def("register_param", &dc::register_param, "Register a parameter");
    m.def("register_graph_z1",
          &dc::register_graph_z1,
          "Register graph with a list of ds parameter ids");
    m.def("register_graph_z2",
          &dc::register_graph_z2,
          "Register graph with a list of ds parameter ids");
    m.def("register_z3_param", &dc::register_z3_param, "Register a parameter");
    m.def("register_graph_z3",
          &dc::register_graph_z3,
          "Register graph with a list of ds parameter ids");
    m.def("start_forward", &dc::start_forward, "Start forward pass");
    m.def("end_forward", &dc::end_forward, "End forward pass");
    m.def("start_backward", &dc::start_backward, "Start backward pass");
    m.def("cleanup", &dc::cleanup, "Clean up DeepCompile");
    m.def("reset", &dc::reset, "Reset the state");
    m.def("invalidate_gathered_param", &dc::invalidate_gathered_param, "Invalidate gathered param");
    m.def("clear_all_gathered_params", &dc::clear_all_gathered_params, "Clear all gathered params");
}
