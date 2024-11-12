// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <libaio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <deepspeed_aio_types.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct io_xfer_ctxt {
    const int _fd;
    const int64_t _file_base_offset;
    const int64_t _buffer_base_offset;
    const void* _mem_buffer;
    const int64_t _num_bytes;

    io_xfer_ctxt(const int fd,
                 const int64_t file_offset,
                 const int64_t buffer_offset,
                 const int64_t num_bytes,
                 const void* buffer);
};

struct io_prep_context {
    const bool _read_op;
    const std::unique_ptr<io_xfer_ctxt>& _xfer_ctxt;
    const size_t _block_size;
    const std::vector<struct iocb*>* _iocbs;

    io_prep_context(const bool read_op,
                    const std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                    const size_t block_size,
                    const std::vector<struct iocb*>* iocbs);

    void prep_iocbs(const int n_iocbs,
                    const size_t num_bytes,
                    const void* start_buffer,
                    const int64_t start_offset);
};

struct io_prep_generator {
    const bool _read_op;
    const std::unique_ptr<io_xfer_ctxt>& _xfer_ctxt;
    const size_t _block_size;

    int64_t _remaining_bytes;
    int64_t _num_io_blocks;
    int64_t _remaining_io_blocks;
    int64_t _next_iocb_index;

    io_prep_generator(const bool read_op,
                      const std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                      const size_t block_size);

    int prep_iocbs(const int n_iocbs, std::vector<struct iocb*>* iocbs);
};

void* ds_page_aligned_alloc(const int64_t size, const bool lock = false);

int get_file_size(const char* filename, int64_t& size);
