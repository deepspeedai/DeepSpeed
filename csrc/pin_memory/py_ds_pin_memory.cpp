// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <cstdint>
#include "deepspeed_pin_tensor.h"

using namespace pybind11::literals;

struct deepspeed_pin_handle_t {
    std::shared_ptr<deepspeed_pin_tensor_t> _pinned_tensor_mgr;

    deepspeed_pin_handle_t() : _pinned_tensor_mgr(deepspeed_pin_tensor_t::shared()) {}

    torch::Tensor new_cpu_locked_tensor(const int64_t num_elem, const torch::Tensor& example_tensor)
    {
        return _pinned_tensor_mgr->alloc(num_elem, example_tensor.options());
    }

    bool free_cpu_locked_tensor(torch::Tensor& locked_tensor)
    {
        return _pinned_tensor_mgr->free(locked_tensor);
    }

    bool free_cpu_locked_tensor_by_ptr(const uintptr_t address)
    {
        return _pinned_tensor_mgr->free(reinterpret_cast<void*>(address));
    }

    bool is_pinned(const torch::Tensor& buffer)
    {
        return buffer.is_pinned() || _pinned_tensor_mgr->is_managed(buffer);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<deepspeed_pin_handle_t>(m, "pin_handle")
        .def(py::init<>(), "Pin-memory handle constructor")
        .def("new_cpu_locked_tensor",
             &deepspeed_pin_handle_t::new_cpu_locked_tensor,
             "Allocate a page-locked CPU tensor",
             "num_elem"_a,
             "example_tensor"_a)
        .def("free_cpu_locked_tensor",
             &deepspeed_pin_handle_t::free_cpu_locked_tensor,
             "Free a page-locked CPU tensor",
             "locked_tensor"_a)
        .def("free_cpu_locked_tensor_by_ptr",
             &deepspeed_pin_handle_t::free_cpu_locked_tensor_by_ptr,
             "Free a page-locked CPU tensor by its allocation base address",
             "address"_a)
        .def("is_pinned",
             &deepspeed_pin_handle_t::is_pinned,
             "Return whether buffer is torch-pinned or managed by DeepSpeed pin_memory",
             "buffer"_a);
}
