// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include "shm.h"

// #define DO_PROFILE
#ifdef DO_PROFILE
#include <cfloat>
#include <chrono>
#endif

// Communication settings
static int world_rank = -1;
static int world_size = -1;

static bool is_initialized = 0;

static bool all_ranks_local_p = false;

void initialize(int size, int rank)
{
    if (is_initialized) return;

    // Check whether all ranks is on the same physical machine.
    // If true, we will use an SHM based low latency allreduce

    auto ls_string = std::getenv("LOCAL_SIZE");
    int ls = 0;
    if (ls_string != NULL) { ls = std::stoi(std::getenv("LOCAL_SIZE")); }

    if (size >= 1 && size == ls) { all_ranks_local_p = true; }

    world_size = size;
    world_rank = rank;
    is_initialized = 1;

    auto addr_string = std::getenv("MASTER_ADDR");
    if (addr_string == NULL) { addr_string = ""; }
    auto port_string = std::getenv("MASTER_PORT");
    if (port_string == NULL) { port_string = ""; }

    if (all_ranks_local_p) { shm_initialize(size, rank, addr_string, port_string); }
}

void inference_all_reduce_(torch::Tensor& data, int op);

// Success - return 0
// Fail (cannot hornor the request and need to fall back) - return -1
void inference_all_reduce_(torch::Tensor& data, int op)
{
    assert(op == 0);
#ifdef DO_PROFILE
    static double total_time = 0.0;
    static double total_time_sq = 0.0;
    static int count = -16;  // warmup
    static double max_time = 0.0;
    static double min_time = DBL_MAX;
    // make sure all rank reach this point before measuring time
    // turn on this if you suspect each rank didn't reach here at the same time (stragger)
    // if (all_ranks_local_p) { barrier_wait(0, world_size); }
    auto start = std::chrono::system_clock::now();
#endif

    auto numel = data.numel();

    int data_size = 0;
    bool data_type_fallback = false;

    switch (data.scalar_type()) {
        case c10::ScalarType::BFloat16: data_size = numel * 2; break;
        case c10::ScalarType::Half: data_size = numel * 2; break;
        case c10::ScalarType::Float: data_size = numel * 4; break;
        default: data_type_fallback = true;
    }

    if (data_type_fallback) return;

    all_reduce_outer_loop(data, numel, data_size);

#ifdef DO_PROFILE
    auto end = std::chrono::system_clock::now();
    count++;
    if (count > 0) {
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (elapsed > max_time) { max_time = elapsed; }
        if (elapsed < min_time) { min_time = elapsed; }
        total_time += elapsed;
        total_time_sq += elapsed * elapsed;
        if (world_rank == 0 && count == 1000) {
            auto avg = total_time / count;
            auto sd =
                sqrt(total_time_sq / count - total_time * total_time / (count * count)) / avg * 100;
            printf("      C++ kernel\t\t    %.2f\t  %.2f\t%.2f\t      %.2f\n",
                   min_time,
                   max_time,
                   total_time / count,
                   sd);
        }
    }
#endif
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("initialize", &initialize, "shm initialize"); }

TORCH_LIBRARY(deepspeed, m)
{
    m.def("inference_all_reduce(Tensor self) -> Tensor");
    m.def("inference_all_reduce_(Tensor(a!) self) -> Tensor(a!)");
}

torch::Tensor inference_all_reduce_meta(const torch::Tensor& self_)
{
    torch::Tensor result_ = torch::empty_like(self_);
    return result_;
}

torch::Tensor& inference_all_reduce__meta(torch::Tensor& self_) { return self_; }

torch::Tensor& inference_all_reduce__cpu(torch::Tensor& self_)
{
    TORCH_INTERNAL_ASSERT(self_.device().type() == torch::DeviceType::CPU);
    torch::Tensor self_tensor = self_.contiguous();
    inference_all_reduce_(self_tensor, 0);
    return self_;
}

torch::Tensor inference_all_reduce_cpu(const torch::Tensor& self_)
{
    torch::Tensor result = self_.clone();
    inference_all_reduce__cpu(result);
    return result;
}

#include <ATen/FunctionalTensorWrapper.h>
// The boilerplate functionalization logic, that teaches functionalization
// how to map x_() calls into x() calls.
// Long term, we'd like to not require users to write this logic.
// HOWEVER, if you have a custom op that is mutable,
// You will still need to write an out-of-place version of that op!
at::Tensor& inference_all_reduce__functionalization_glue(at::Tensor& x)
{
    // We expect all tensor inputs to our op to be "functional tensors"
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(x));
    // First, sync and unwrap and functional tensors
    at::functionalization::impl::sync(x);
    auto x_ = at::functionalization::impl::from_functional_tensor(x);
    // Grab the dispatcher entry corresponding to the out-of-place op, "x"
    static auto op_handle = c10::Dispatcher::singleton()
                                // specify namespace::op_name, op_overload_name
                                .findSchemaOrThrow("deepspeed::inference_all_reduce", "")
                                // Specify the C++ schema of the out-of-place op.
                                .typed<at::Tensor(const at::Tensor&)>();
    // Next, redispatch to the out-of-place op, x() (user called x_, we call x)
    at::Tensor tmp_output;
    {
        at::AutoDispatchSkipFunctionalize guard;
        tmp_output = op_handle.call(x_);
    }
    // Finally, tell functionalization about this mutation.
    at::functionalization::impl::replace_(x, tmp_output);
    at::functionalization::impl::commit_update(x);
    at::functionalization::impl::sync(x);
    return x;
}

TORCH_LIBRARY_IMPL(deepspeed, CPU, m)
{
    m.impl("inference_all_reduce", inference_all_reduce_cpu);
    m.impl("inference_all_reduce_", inference_all_reduce__cpu);
}

TORCH_LIBRARY_IMPL(deepspeed, Meta, m)
{
    m.impl("inference_all_reduce", inference_all_reduce_meta);
    m.impl("inference_all_reduce_", inference_all_reduce__meta);
}

TORCH_LIBRARY_IMPL(deepspeed, Functionalize, m)
{
    m.impl("inference_all_reduce_", inference_all_reduce__functionalization_glue);
}
