// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"

/*
Wrapper around cudaHostAlloc with some specific flags. Returns a pointer to the
memory region of `size` bytes.
*/
void* get_cuda_fast_buffer(int64_t size);
