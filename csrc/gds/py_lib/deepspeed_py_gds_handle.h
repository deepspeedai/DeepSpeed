// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <condition_variable>
#include <memory>
#include "deepspeed_py_io_handle.h"

struct deepspeed_gds_handle_t : deepspeed_io_handle_t {
    deepspeed_gds_handle_t(const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const int num_threads);

    ~deepspeed_gds_handle_t();

    torch::Tensor new_pinned_device_tensor(const size_t num_elem,
                                           const torch::Tensor& example_tensor);

    bool free_pinned_device_tensor(torch::Tensor&);

    bool pin_device_tensor(const torch::Tensor& buffer);

    bool unpin_device_tensor(const torch::Tensor& buffer);

    void _init_cuFile(const int block_size, const int queue_length, const int num_threads);

    void _close_cuFile();

    std::shared_ptr<struct io_op_desc_t> _create_io_op_desc(const bool read_op,
                                                            const torch::Tensor& buffer,
                                                            const int fd,
                                                            const char* filename,
                                                            const long long int file_num_bytes,
                                                            const bool validate);

    static int s_cuFile_init;
};
