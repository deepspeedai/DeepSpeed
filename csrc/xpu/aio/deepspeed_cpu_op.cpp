// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_cpu_op.h"

using namespace std;

cpu_op_desc_t::cpu_op_desc_t(const bool read_op,
                             const torch::Tensor& buffer,
                             const int fd,
                             const char* filename,
                             const long long int file_num_bytes,
                             const int num_threads,
                             const bool validate)
    : io_op_desc_t(read_op, buffer, fd, filename, file_num_bytes, num_threads, validate),
      _cpu_buffer(buffer)
{
    // XPU don't handle buffer here. See XPU Accelerator pin_memory.
    _contiguous_buffer = _cpu_buffer.contiguous();
}

char* cpu_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void cpu_op_desc_t::finish()
{
    if (_read_op && _buffer.is_xpu()) { _buffer.copy_(_cpu_buffer.to(torch::kXPU)); }
}

void cpu_op_desc_t::validate()
{
    validate_aio_operation(_read_op, _filename.c_str(), data_ptr(), _file_num_bytes);
}

void cpu_op_desc_t::run(const int tid,
                        std::unique_ptr<aio_context>& aio_ctxt,
                        deepspeed_aio_config_t* aio_config)
{
    assert(tid < _num_threads);
    const auto base_offset = _num_bytes_per_thread * tid;

    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(
        new io_xfer_ctxt(_fd, base_offset, _num_bytes_per_thread, data_ptr()));

    if (aio_config->_overlap_events) {
        do_aio_operation_overlap(_read_op, aio_ctxt, xfer_ctxt, aio_config, nullptr);
    } else {
        do_aio_operation_sequential(_read_op, aio_ctxt, xfer_ctxt, aio_config, nullptr);
    }
}
