// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <cmath>
#include <iostream>

#include "deepspeed_aio_utils.h"

using namespace std;

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

io_xfer_ctxt::io_xfer_ctxt(const aio_fd_t fd,
                           const int64_t file_offset,
                           const int64_t buffer_offset,
                           const int64_t num_bytes,
                           const void* buffer)
    : _fd(fd),
      _file_base_offset(file_offset),
      _buffer_base_offset(buffer_offset),
      _mem_buffer(buffer),
      _num_bytes(num_bytes)
{
}

io_prep_context::io_prep_context(const bool read_op,
                                 const std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                                 const size_t block_size,
                                 const std::vector<io_request_t*>* iocbs)
    : _read_op(read_op), _xfer_ctxt(xfer_ctxt), _block_size(block_size), _iocbs(iocbs)
{
}

void io_prep_context::prep_iocbs(const int n_iocbs,
                                 const size_t num_bytes,
                                 const void* start_buffer,
                                 const int64_t start_offset)
{
    assert(static_cast<size_t>(n_iocbs) <= _iocbs->size());
    for (auto i = 0; i < n_iocbs; ++i) {
        const auto shift = i * _block_size;
        const auto xfer_buffer = (char*)start_buffer + _xfer_ctxt->_buffer_base_offset + shift;
        const auto xfer_offset = _xfer_ctxt->_file_base_offset + start_offset + shift;
        auto byte_count = _block_size;

        if ((shift + _block_size) > num_bytes) { byte_count = num_bytes - shift; }

#if defined(_WIN32)
        auto* req = _iocbs->at(i);
        req->_fd = _xfer_ctxt->_fd;
        req->_buf = xfer_buffer;
        req->_nbytes = byte_count;
        req->_offset = xfer_offset;
        req->_read_op = _read_op;
#else
        if (_read_op) {
            io_prep_pread(_iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, byte_count, xfer_offset);
        } else {
            io_prep_pwrite(_iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, byte_count, xfer_offset);
        }
#endif
    }
}

io_prep_generator::io_prep_generator(const bool read_op,
                                     const std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                                     const size_t block_size)
    : _read_op(read_op),
      _xfer_ctxt(xfer_ctxt),
      _block_size(block_size),
      _remaining_bytes(xfer_ctxt->_num_bytes),
      _next_iocb_index(0)
{
    _num_io_blocks =
        static_cast<int64_t>(ceil(static_cast<double>(xfer_ctxt->_num_bytes) / block_size));
    _remaining_io_blocks = _num_io_blocks;
}

int io_prep_generator::prep_iocbs(const int n_iocbs, std::vector<io_request_t*>* iocbs)
{
    if ((_remaining_bytes) == 0 || (_remaining_io_blocks == 0)) {
        assert(static_cast<int64_t>(_remaining_bytes) == _remaining_io_blocks);
        return 0;
    }

    assert(static_cast<size_t>(n_iocbs) <= iocbs->size());

    auto actual_n_iocbs = min(static_cast<int64_t>(n_iocbs), _remaining_io_blocks);
    for (auto i = 0; i < actual_n_iocbs; ++i, ++_next_iocb_index) {
        const auto xfer_buffer = (char*)_xfer_ctxt->_mem_buffer + _xfer_ctxt->_buffer_base_offset +
                                 (_next_iocb_index * _block_size);
        const auto xfer_offset = _xfer_ctxt->_file_base_offset + (_next_iocb_index * _block_size);
        const auto num_bytes = min(static_cast<int64_t>(_block_size), _remaining_bytes);
#if defined(_WIN32)
        auto* req = iocbs->at(i);
        req->_fd = _xfer_ctxt->_fd;
        req->_buf = xfer_buffer;
        req->_nbytes = num_bytes;
        req->_offset = xfer_offset;
        req->_read_op = _read_op;
#else
        if (_read_op) {
            io_prep_pread(iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, num_bytes, xfer_offset);
        } else {
            io_prep_pwrite(iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, num_bytes, xfer_offset);
        }
#endif
        _remaining_bytes -= num_bytes;
    }
    _remaining_io_blocks -= actual_n_iocbs;

    return actual_n_iocbs;
}

int64_t get_file_size(const char* filename, int64_t& size)
{
#if defined(_WIN32)
    WIN32_FILE_ATTRIBUTE_DATA attrs;
    if (!GetFileAttributesExA(filename, GetFileExInfoStandard, &attrs)) { return -1; }
    size = (static_cast<int64_t>(attrs.nFileSizeHigh) << 32) | attrs.nFileSizeLow;
    return 0;
#else
    struct stat st;
    if (stat(filename, &st) == -1) { return -1; }
    size = st.st_size;
    return 0;
#endif
}

int64_t get_fd_file_size(const aio_fd_t fd, int64_t& size)
{
#if defined(_WIN32)
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(fd, &file_size)) { return -1; }
    size = file_size.QuadPart;
    return 0;
#else
    struct stat st;
    if (fstat(fd, &st) == -1) { return -1; }
    size = st.st_size;
    return 0;
#endif
}
