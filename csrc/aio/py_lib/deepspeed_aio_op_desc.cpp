// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_aio_op_desc.h"

#include <mutex>

using namespace std;

namespace {
// Warn home users only once per process: repeatedly offloading data (e.g. hidden
// states) to a consumer SSD can exhaust its rated write endurance (TBW).
std::once_flag write_warning_flag;

void warn_consumer_ssd_writes()
{
    const char* msg =
        "Offloading to NVMe can generate heavy write traffic. On consumer/home SSDs, repeatedly "
        "offloading data (e.g. optimizer states, hidden states) can blow through the drive's "
        "rated write endurance (TBW) and shorten its lifespan. Prefer enterprise/datacenter SSDs "
        "for sustained offloading workloads.";

    // Route through DeepSpeed's Python logger. Acquire the GIL since this may be
    // invoked from a non-Python thread.
    pybind11::gil_scoped_acquire acquire;
    pybind11::module_::import("deepspeed.utils").attr("logger").attr("warning")(msg);
}
}  // namespace

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const int intra_op_parallelism,
                           const bool validate,
                           const int64_t file_offset)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename((filename == nullptr) ? std::string() : filename),
      _file_offset(file_offset),
      _intra_op_parallelism(intra_op_parallelism),
      _num_bytes_per_thread(static_cast<int64_t>(buffer.nbytes()) / intra_op_parallelism),
      _validate(validate)
{
    if (validate) { assert(nullptr != filename); }
    if (!read_op) { std::call_once(write_warning_flag, warn_consumer_ssd_writes); }
}

char* io_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void io_op_desc_t::finish() {}

void io_op_desc_t::validate() {}

void io_op_desc_t::run(const int tid,
                       std::unique_ptr<aio_context>& aio_ctxt,
                       deepspeed_aio_config_t* aio_config)
{
}
