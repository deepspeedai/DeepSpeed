// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_cpu_op.h"
#include "deepspeed_pin_tensor.h"

using namespace std;

cpu_op_desc_t::cpu_op_desc_t(
    const bool read_op,
    const torch::Tensor& buffer,
    const std::unique_ptr<struct deepspeed_pin_tensor_t>& pinned_tensor_mgr,
    const int fd,
    const char* filename,
    const int64_t file_num_bytes,
    const int intra_op_parallelism,
    const bool validate,
    const int64_t file_offset)
    : io_op_desc_t(read_op,
                   buffer,
                   fd,
                   filename,
                   file_num_bytes,
                   intra_op_parallelism,
                   validate,
                   file_offset),
      _cpu_buffer(buffer),
      _pinned_tensor_mgr(pinned_tensor_mgr),
      _is_managed_bounce_buffer(false)
{
    // Need to use CPU bounce buffer if buffer is not a page-locked DRAM memory.
    _use_bounce_buffer =
        !(_buffer.is_cpu() && (_buffer.is_pinned() || _pinned_tensor_mgr->is_managed(_buffer)));
    if (_use_bounce_buffer) {
        _alloc_bounce_buffer();
        if (!_read_op) { _cpu_buffer.copy_(_buffer); }
    }
    _contiguous_buffer = _cpu_buffer.contiguous();
}

char* cpu_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void cpu_op_desc_t::finish()
{
    if (_use_bounce_buffer) {
        if (_read_op) {
            if (_buffer.is_cuda()) {
                _buffer.copy_(_cpu_buffer.to(torch::Device(torch::kCUDA, _buffer.get_device()),
                                             /*non_blocking=*/true));
            }
            if (_buffer.is_xpu()) { _buffer.copy_(_cpu_buffer.to(torch::kXPU)); }
            if (_buffer.is_cpu()) { _buffer.copy_(_cpu_buffer); }
#if defined(__ENABLE_CANN__)
            if (torch_npu::utils::is_npu(_buffer)) {
                auto device = at::Device("npu:0");
                _buffer.copy_(_cpu_buffer.to(device));
            }
#endif
        }

        _free_bounce_buffer();
    }
}

void cpu_op_desc_t::validate()
{
    validate_aio_operation(_read_op, _filename.c_str(), data_ptr(), _file_num_bytes);
}

void cpu_op_desc_t::run(const int tid,
                        std::unique_ptr<aio_context>& aio_ctxt,
                        deepspeed_aio_config_t* aio_config)
{
    assert(tid < _intra_op_parallelism);
    const auto buffer_base_offset = _num_bytes_per_thread * tid;
    const auto file_base_offset = _file_offset + (_num_bytes_per_thread * tid);

    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
        _fd, file_base_offset, buffer_base_offset, _num_bytes_per_thread, data_ptr()));

    if (aio_config->_overlap_events) {
        do_aio_operation_overlap(_read_op, aio_ctxt, xfer_ctxt, aio_config, nullptr);
    } else {
        do_aio_operation_sequential(_read_op, aio_ctxt, xfer_ctxt, aio_config, nullptr);
    }
}

void cpu_op_desc_t::_alloc_bounce_buffer()
{
    auto options = torch::TensorOptions()
                       .dtype(_buffer.dtype())
                       .layout(_buffer.layout())
                       .device(torch::kCPU)
                       .requires_grad(false);

#if defined(__CUDA_ARCH__)
    _cpu_buffer = torch::empty(_buffer.numel(), options).pin_memory();
#else
    _is_managed_bounce_buffer = true;
    _cpu_buffer = _pinned_tensor_mgr->alloc(_buffer.numel(), options);
#endif
}

void cpu_op_desc_t::_free_bounce_buffer()
{
    if (_is_managed_bounce_buffer) { _pinned_tensor_mgr->free(_cpu_buffer); }
}
