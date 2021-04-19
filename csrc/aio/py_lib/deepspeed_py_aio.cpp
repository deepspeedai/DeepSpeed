
/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "deepspeed_py_aio.h"

using namespace std;
using namespace std::chrono;

#define DEBUG_DS_AIO_READ 0
#define DEBUG_DS_AIO_WRITE 0

static const std::string c_library_name = "deepspeed_aio";

int deepspeed_py_aio_write(const torch::Tensor& buffer,
                           const char* filename,
                           const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const bool validate)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    deepspeed_aio_config_t config(block_size, queue_depth, single_submit, overlap_events, false);

    const auto fd = open_file(filename, false);
    if (fd == -1) { return -1; }

    auto write_buffer = (char*)buffer.data_ptr();
    const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_write_bytes, write_buffer));
    std::unique_ptr<aio_context> aio_ctxt(new aio_context(config._block_size, config._queue_depth));

    if (config._overlap_events) {
        do_aio_operation_overlap(false, aio_ctxt, xfer_ctxt, &config, nullptr);
    } else {
        do_aio_operation_sequential(false, aio_ctxt, xfer_ctxt, &config, nullptr);
    }
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    close(fd);

    if (validate) { validate_aio_operation(false, filename, write_buffer, num_write_bytes); }

    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): "
              << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
              << std::endl;
    return 0;
}

int deepspeed_py_aio_read(torch::Tensor& buffer,
                          const char* filename,
                          const int block_size,
                          const int queue_depth,
                          const bool single_submit,
                          const bool overlap_events,
                          const bool validate)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    long long num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes)) {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }

    deepspeed_aio_config_t config(block_size, queue_depth, single_submit, overlap_events, false);
    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    auto read_buffer = (char*)buffer.data_ptr();
    assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);

    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_file_bytes, read_buffer));
    std::unique_ptr<aio_context> aio_ctxt(new aio_context(config._block_size, config._queue_depth));

    if (config._overlap_events) {
        do_aio_operation_overlap(true, aio_ctxt, xfer_ctxt, &config, nullptr);
    } else {
        do_aio_operation_sequential(true, aio_ctxt, xfer_ctxt, &config, nullptr);
    }
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    close(fd);

    if (validate) { validate_aio_operation(true, filename, read_buffer, num_file_bytes); }

    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): "
              << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
              << std::endl;
    return 0;
}
