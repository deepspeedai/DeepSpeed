// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <memory>
#include <queue>
#include "deepspeed_aio_op_desc.h"

struct cpu_op_desc_t : io_op_desc_t {
    torch::Tensor _cpu_buffer;
    bool _use_bounce_buffer;

    cpu_op_desc_t(const bool read_op,
                  const torch::Tensor& buffer,
                  const int fd,
                  const char* filename,
                  const long long int file_num_bytes,
                  const int num_threads,
                  const bool validate);

    void run(const int tid,
             std::unique_ptr<aio_context>& aio_ctxt,
             deepspeed_aio_config_t* aio_config);

    char* data_ptr() const;

    void validate();

    void finish();
};
