// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <condition_variable>
#include <memory>
#include "deepspeed_py_io_handle.h"

struct deepspeed_aio_handle_t : deepspeed_io_handle_t {
    deepspeed_aio_handle_t(const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const int num_threads);

    ~deepspeed_aio_handle_t();
};
