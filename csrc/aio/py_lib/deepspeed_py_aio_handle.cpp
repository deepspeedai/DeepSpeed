// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_aio_handle.h"
#include <cstdlib>

using namespace std;

deepspeed_aio_handle_t::deepspeed_aio_handle_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const int intra_op_parallelism)
    : deepspeed_io_handle_t(block_size,
                            queue_depth,
                            single_submit,
                            overlap_events,
                            intra_op_parallelism)
{
}

deepspeed_aio_handle_t::~deepspeed_aio_handle_t() {}
