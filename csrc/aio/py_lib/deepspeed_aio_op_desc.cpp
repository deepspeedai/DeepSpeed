// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepspeed_aio_op_desc.h"

using namespace std;

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const long long int file_num_bytes,
                           const int num_threads,
                           const bool validate)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename(filename),
      _file_num_bytes(file_num_bytes),
      _num_threads(num_threads),
      _num_bytes_per_thread(file_num_bytes / num_threads),
      _validate(validate)
{
}

char* io_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void io_op_desc_t::finish() {}

void io_op_desc_t::validate() {}

void io_op_desc_t::run(const int tid,
                       std::unique_ptr<aio_context>& aio_ctxt,
                       deepspeed_aio_config_t* aio_config)
{
}
