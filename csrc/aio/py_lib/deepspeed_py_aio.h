// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <deepspeed_aio_common.h>
#include <stdlib.h>
#include <torch/extension.h>

int deepspeed_py_aio_write(const torch::Tensor& buffer,
                           const char* filename,
                           const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const bool validate);

int deepspeed_py_aio_read(torch::Tensor& buffer,
                          const char* filename,
                          const int block_size,
                          const int queue_depth,
                          const bool single_submit,
                          const bool overlap_events,
                          const bool validate);
