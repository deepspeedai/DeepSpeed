/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <cmath>

#include "deepspeed_aio_utils.h"

using namespace std;

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

io_xfer_ctxt::io_xfer_ctxt(const int fd,
                           const long long int file_offset,
                           const long long int num_bytes,
                           const void* buffer)
    : _fd(fd), _base_offset(file_offset), _mem_buffer(buffer), _num_bytes(num_bytes)
{
}

io_prep_context::io_prep_context(const bool read_op,
                                 const std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                                 const size_t block_size,
                                 const std::vector<struct iocb*>* iocbs)
    : _read_op(read_op), _xfer_ctxt(xfer_ctxt), _block_size(block_size), _iocbs(iocbs)
{
}

void io_prep_context::prep_iocbs(const int n_iocbs,
                                 const size_t num_bytes,
                                 const void* start_buffer,
                                 const long long int start_offset)
{
    assert(static_cast<size_t>(n_iocbs) <= _iocbs->size());
    for (auto i = 0; i < n_iocbs; ++i) {
        const auto shift = i * _block_size;
        const auto xfer_buffer = (char*)start_buffer + _xfer_ctxt->_base_offset + shift;
        const auto xfer_offset = _xfer_ctxt->_base_offset + start_offset + shift;
        auto byte_count = _block_size;
        if ((shift + _block_size) > num_bytes) { byte_count = num_bytes - shift; }

        if (_read_op) {
            io_prep_pread(_iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, byte_count, xfer_offset);
        } else {
            io_prep_pwrite(_iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, byte_count, xfer_offset);
        }
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
        static_cast<long long int>(ceil(static_cast<double>(xfer_ctxt->_num_bytes) / block_size));
    _remaining_io_blocks = _num_io_blocks;
}

int io_prep_generator::prep_iocbs(const int n_iocbs, std::vector<struct iocb*>* iocbs)
{
    if ((_remaining_bytes) == 0 || (_remaining_io_blocks == 0)) {
        assert(static_cast<long long int>(_remaining_bytes) == _remaining_io_blocks);
        return 0;
    }

    assert(static_cast<size_t>(n_iocbs) <= iocbs->size());

    auto actual_n_iocbs = min(static_cast<long long int>(n_iocbs), _remaining_io_blocks);
    for (auto i = 0; i < actual_n_iocbs; ++i, ++_next_iocb_index) {
        const auto xfer_offset = _xfer_ctxt->_base_offset + (_next_iocb_index * _block_size);
        const auto xfer_buffer = (char*)_xfer_ctxt->_mem_buffer + xfer_offset;
        const auto num_bytes = min(static_cast<long long int>(_block_size), _remaining_bytes);

        if (_read_op) {
            io_prep_pread(iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, num_bytes, xfer_offset);
        } else {
            io_prep_pwrite(iocbs->at(i), _xfer_ctxt->_fd, xfer_buffer, num_bytes, xfer_offset);
        }
        _remaining_bytes -= num_bytes;
    }
    _remaining_io_blocks -= actual_n_iocbs;

    return actual_n_iocbs;
}

int get_file_size(const char* filename, long long int& size)
{
    struct stat st;
    if (stat(filename, &st) == -1) { return -1; }
    size = st.st_size;
    return 0;
}

void* ds_page_aligned_alloc(const size_t size, const bool lock)
{
    void* ptr;
    int retval;

    retval = posix_memalign(&ptr, (size_t)sysconf(_SC_PAGESIZE), size);
    if (retval) { return nullptr; }

    if (lock == false) { return ptr; }

    auto mlock_ret = mlock(ptr, size);
    if (mlock_ret != 0) {
        auto mlock_error = errno;
        printf("mlock failed with %d %s\n", mlock_error, strerror(mlock_error));

        free(ptr);
        return nullptr;
    }

    return ptr;
}
