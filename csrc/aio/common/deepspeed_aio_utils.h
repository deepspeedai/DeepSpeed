/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

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
    const long long int _base_offset;
    const void* _mem_buffer;
    const long long int _num_bytes;

    io_xfer_ctxt(const int fd,
                 const long long int file_offset,
                 const long long int num_bytes,
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
                    const long long int start_offset);
};

struct io_prep_generator {
    const bool _read_op;
    const std::unique_ptr<io_xfer_ctxt>& _xfer_ctxt;
    const size_t _block_size;

    long long int _remaining_bytes;
    long long int _num_io_blocks;
    long long int _remaining_io_blocks;
    long long int _next_iocb_index;

    io_prep_generator(const bool read_op,
                      const std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                      const size_t block_size);

    int prep_iocbs(const int n_iocbs, std::vector<struct iocb*>* iocbs);
};

void* ds_page_aligned_alloc(const size_t size, const bool lock = false);

int get_file_size(const char* filename, long long int& size);
