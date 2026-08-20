// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Page-aligned host allocation with optional mlock, independent of libaio.
*/

#pragma once

#include <cstdint>

void* ds_page_aligned_alloc(const int64_t size, const bool lock = false);
