// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#ifndef _IO_OP_DESC_T_
#define _IO_OP_DESC_T_
#include <atomic>
#include <memory>
#include <queue>
#include "deepspeed_py_aio.h"

void warn_consumer_ssd_writes() noexcept;

struct io_op_desc_t {
    const bool _read_op;
    torch::Tensor _buffer;
    int _fd;
    std::string _filename;
    const int _intra_op_parallelism;
    const int64_t _num_bytes_per_thread;
    torch::Tensor _contiguous_buffer;
    const bool _validate;
    const int64_t _file_offset;
    // The first error of the threads that run this op, as a negative errno; 0 if none failed.
    std::atomic<int> _error_code;

    io_op_desc_t(const bool read_op,
                 const torch::Tensor& buffer,
                 const int fd,
                 const char* filename,
                 const int intra_op_parallelism,
                 const bool validate,
                 const int64_t file_offset);

    virtual void run(const int tid,
                     std::unique_ptr<aio_context>& aio_ctxt,
                     deepspeed_aio_config_t* aio_config);

    virtual char* data_ptr() const;

    virtual void validate();

    virtual void finish();

    void set_error(const int error_code);
};
#endif  // _IO_OP_DESC_T_
