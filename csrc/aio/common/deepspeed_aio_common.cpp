// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <libaio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "deepspeed_aio_common.h"

using namespace std;
using namespace std::chrono;

#define DEBUG_DS_AIO_PERF 0
#define DEBUG_DS_AIO_SUBMIT_PERF 0

static const std::string c_library_name = "deepspeed_aio";

#if !defined(_WIN32)
static void _report_aio_statistics(const char* tag,
                                   const std::vector<std::chrono::duration<double>>& latencies)
    __attribute__((unused));
#endif

static void _report_aio_statistics(const char* tag,
                                   const std::vector<std::chrono::duration<double>>& latencies)
{
    std::vector<double> lat_usec;
    for (auto& lat : latencies) { lat_usec.push_back(lat.count() * 1e6); }
    const auto min_lat = *(std::min_element(lat_usec.begin(), lat_usec.end()));
    const auto max_lat = *(std::max_element(lat_usec.begin(), lat_usec.end()));
    const auto avg_lat = std::accumulate(lat_usec.begin(), lat_usec.end(), 0) / lat_usec.size();

    std::cout << c_library_name << ": latency statistics(usec) " << tag
              << " min/max/avg = " << min_lat << " " << max_lat << " " << avg_lat << std::endl;
}

static void _get_aio_latencies(std::vector<std::chrono::duration<double>>& raw_latencies,
                               struct deepspeed_aio_latency_t& summary_latencies)
{
    std::vector<double> lat_usec;
    for (auto& lat : raw_latencies) { lat_usec.push_back(lat.count() * 1e6); }
    summary_latencies._min_usec = *(std::min_element(lat_usec.begin(), lat_usec.end()));
    summary_latencies._max_usec = *(std::max_element(lat_usec.begin(), lat_usec.end()));
    summary_latencies._avg_usec =
        std::accumulate(lat_usec.begin(), lat_usec.end(), 0) / lat_usec.size();
}

#if defined(_WIN32)
// Windows has no io_submit/io_pgetevents equivalent, so "submit" here actually performs a
// blocking ReadFile/WriteFile immediately; completion is then a no-op (see _do_io_complete).
static void _win_submit_one(io_request_t* req)
{
    char* buf = static_cast<char*>(req->_buf);
    int64_t offset = req->_offset;
    int64_t remaining = req->_nbytes;

    while (remaining > 0) {
        OVERLAPPED ov = {};
        ov.Offset = static_cast<DWORD>(offset & 0xffffffff);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD bytes_transferred = 0;
        const BOOL ok =
            req->_read_op
                ? ReadFile(req->_fd, buf, static_cast<DWORD>(remaining), &bytes_transferred, &ov)
                : WriteFile(req->_fd, buf, static_cast<DWORD>(remaining), &bytes_transferred, &ov);
        if (!ok || bytes_transferred == 0) {
            const auto error_code = GetLastError();
            report_file_error(
                "<aio>", req->_read_op ? "ReadFile" : "WriteFile", static_cast<int>(error_code));
            assert(ok && bytes_transferred > 0);
            return;
        }
        buf += bytes_transferred;
        offset += bytes_transferred;
        remaining -= bytes_transferred;
    }
}
#endif

static void _do_io_submit_singles(const int64_t n_iocbs,
                                  const int64_t iocb_index,
                                  std::unique_ptr<aio_context>& aio_ctxt,
                                  std::vector<std::chrono::duration<double>>& submit_times)
{
    for (auto i = 0; i < n_iocbs; ++i) {
        const auto st = std::chrono::high_resolution_clock::now();
#if defined(_WIN32)
        _win_submit_one(aio_ctxt->_iocbs[i]);
#else
        const auto submit_ret = io_submit(aio_ctxt->_io_ctxt, 1, aio_ctxt->_iocbs.data() + i);
        assert(submit_ret > 0);
#endif
        submit_times.push_back(std::chrono::high_resolution_clock::now() - st);
#if DEBUG_DS_AIO_SUBMIT_PERF && !defined(_WIN32)
        printf("submit(usec) %f io_index=%lld buf=%p len=%lu off=%llu \n",
               submit_times.back().count() * 1e6,
               iocb_index,
               aio_ctxt->_iocbs[i]->u.c.buf,
               aio_ctxt->_iocbs[i]->u.c.nbytes,
               aio_ctxt->_iocbs[i]->u.c.offset);
#endif
    }
}

static void _do_io_submit_block(const int64_t n_iocbs,
                                const int64_t iocb_index,
                                std::unique_ptr<aio_context>& aio_ctxt,
                                std::vector<std::chrono::duration<double>>& submit_times)
{
    const auto st = std::chrono::high_resolution_clock::now();
#if defined(_WIN32)
    for (auto i = 0; i < n_iocbs; ++i) { _win_submit_one(aio_ctxt->_iocbs[i]); }
#else
    const auto submit_ret = io_submit(aio_ctxt->_io_ctxt, n_iocbs, aio_ctxt->_iocbs.data());
    assert(submit_ret > 0);
#endif
    submit_times.push_back(std::chrono::high_resolution_clock::now() - st);
#if DEBUG_DS_AIO_SUBMIT_PERF && !defined(_WIN32)
    printf("submit(usec) %f io_index=%lld nr=%lld buf=%p len=%lu off=%llu \n",
           submit_times.back().count() * 1e6,
           iocb_index,
           n_iocbs,
           aio_ctxt->_iocbs[0]->u.c.buf,
           aio_ctxt->_iocbs[0]->u.c.nbytes,
           aio_ctxt->_iocbs[0]->u.c.offset);
#endif
}

static int _do_io_complete(const int64_t min_completes,
                           const int64_t max_completes,
                           std::unique_ptr<aio_context>& aio_ctxt,
                           std::vector<std::chrono::duration<double>>& reap_times)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
#if defined(_WIN32)
    // _win_submit_one already blocked until each request finished, so every
    // request handed in was completed by the time we get here.
    const int64_t n_completes = max_completes;
#else
    int64_t n_completes = io_pgetevents(aio_ctxt->_io_ctxt,
                                        min_completes,
                                        max_completes,
                                        aio_ctxt->_io_events.data(),
                                        nullptr,
                                        nullptr);
#endif
    reap_times.push_back(std::chrono::high_resolution_clock::now() - start_time);
    assert(n_completes >= min_completes);
    return n_completes;
}

void do_aio_operation_sequential(const bool read_op,
                                 std::unique_ptr<aio_context>& aio_ctxt,
                                 std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                                 deepspeed_aio_config_t* config,
                                 deepspeed_aio_perf_t* perf)
{
    struct io_prep_context prep_ctxt(read_op, xfer_ctxt, aio_ctxt->_block_size, &aio_ctxt->_iocbs);

    const auto num_io_blocks = static_cast<int64_t>(
        ceil(static_cast<double>(xfer_ctxt->_num_bytes) / aio_ctxt->_block_size));
#if DEBUG_DS_AIO_PERF
    const auto io_op_name = std::string(read_op ? "read" : "write");
    std::cout << c_library_name << ": start " << io_op_name << " " << xfer_ctxt->_num_bytes
              << " bytes with " << num_io_blocks << " io blocks" << std::endl;
#endif

    std::vector<std::chrono::duration<double>> submit_times;
    std::vector<std::chrono::duration<double>> reap_times;
    const auto max_queue_bytes =
        static_cast<int64_t>(aio_ctxt->_queue_depth * aio_ctxt->_block_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int64_t iocb_index = 0; iocb_index < num_io_blocks; iocb_index += aio_ctxt->_queue_depth) {
        const auto start_offset = iocb_index * aio_ctxt->_block_size;
        const auto start_buffer = (char*)xfer_ctxt->_mem_buffer + start_offset;
        const auto n_iocbs =
            min(static_cast<int64_t>(aio_ctxt->_queue_depth), (num_io_blocks - iocb_index));
        const auto num_bytes = min(max_queue_bytes, (xfer_ctxt->_num_bytes - start_offset));
        prep_ctxt.prep_iocbs(n_iocbs, num_bytes, start_buffer, start_offset);

        if (config->_single_submit) {
            _do_io_submit_singles(n_iocbs, iocb_index, aio_ctxt, submit_times);
        } else {
            _do_io_submit_block(n_iocbs, iocb_index, aio_ctxt, submit_times);
        }

        _do_io_complete(n_iocbs, n_iocbs, aio_ctxt, reap_times);
    }
    const std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    if (perf) {
        _get_aio_latencies(submit_times, perf->_submit);
        _get_aio_latencies(reap_times, perf->_complete);
        perf->_e2e_usec = elapsed.count() * 1e6;
        perf->_e2e_rate_GB = (xfer_ctxt->_num_bytes / elapsed.count() / 1e9);
    }

#if DEBUG_DS_AIO_PERF
    _report_aio_statistics("submit", submit_times);
    _report_aio_statistics("complete", reap_times);
#endif

#if DEBUG_DS_AIO_PERF
    std::cout << c_library_name << ": runtime(usec) " << elapsed.count() * 1e6
              << " rate(GB/sec) = " << (xfer_ctxt->_num_bytes / elapsed.count() / 1e9) << std::endl;
#endif

#if DEBUG_DS_AIO_PERF
    std::cout << c_library_name << ": finish " << io_op_name << " " << xfer_ctxt->_num_bytes
              << " bytes " << std::endl;
#endif
}

void do_aio_operation_overlap(const bool read_op,
                              std::unique_ptr<aio_context>& aio_ctxt,
                              std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                              deepspeed_aio_config_t* config,
                              deepspeed_aio_perf_t* perf)
{
    struct io_prep_generator io_gen(read_op, xfer_ctxt, aio_ctxt->_block_size);

#if DEBUG_DS_AIO_PERF
    const auto io_op_name = std::string(read_op ? "read" : "write");
    std::cout << c_library_name << ": start " << io_op_name << " " << xfer_ctxt->_num_bytes
              << " bytes with " << io_gen._num_io_blocks << " io blocks" << std::endl;
#endif

    std::vector<std::chrono::duration<double>> submit_times;
    std::vector<std::chrono::duration<double>> reap_times;

    auto request_iocbs = aio_ctxt->_queue_depth;
    auto n_pending_iocbs = 0;
    const auto min_completes = 1;
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        const auto n_iocbs = io_gen.prep_iocbs(request_iocbs - n_pending_iocbs, &aio_ctxt->_iocbs);
        if (n_iocbs > 0) {
            if (config->_single_submit) {
                _do_io_submit_singles(
                    n_iocbs, (io_gen._next_iocb_index - n_iocbs), aio_ctxt, submit_times);
            } else {
                _do_io_submit_block(
                    n_iocbs, (io_gen._next_iocb_index - n_iocbs), aio_ctxt, submit_times);
            }
        }

        n_pending_iocbs += n_iocbs;
        assert(n_pending_iocbs <= aio_ctxt->_queue_depth);

        if (n_pending_iocbs == 0) { break; }

        const auto n_complete =
            _do_io_complete(min_completes, n_pending_iocbs, aio_ctxt, reap_times);
        n_pending_iocbs -= n_complete;
    }

    const std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    if (perf) {
        _get_aio_latencies(submit_times, perf->_submit);
        _get_aio_latencies(reap_times, perf->_complete);
        perf->_e2e_usec = elapsed.count() * 1e6;
        perf->_e2e_rate_GB = (xfer_ctxt->_num_bytes / elapsed.count() / 1e9);
    }

#if DEBUG_DS_AIO_PERF
    _report_aio_statistics("submit", submit_times);
    _report_aio_statistics("complete", reap_times);
#endif

#if DEBUG_DS_AIO_PERF
    std::cout << c_library_name << ": runtime(usec) " << elapsed.count() * 1e6
              << " rate(GB/sec) = " << (xfer_ctxt->_num_bytes / elapsed.count() / 1e9) << std::endl;
#endif

#if DEBUG_DS_AIO_PERF
    std::cout << c_library_name << ": finish " << io_op_name << " " << xfer_ctxt->_num_bytes
              << " bytes " << std::endl;
#endif
}

void report_file_error(const char* filename, const std::string file_op, const int error_code)
{
    std::string err_msg = file_op + std::string(" failed on ") + std::string(filename) +
                          " error = " + std::to_string(error_code);
    std::cerr << c_library_name << ":  " << err_msg << std::endl;
}

aio_fd_t open_file(const char* filename, const bool read_op)
{
#if defined(_WIN32)
    // No FILE_FLAG_NO_BUFFERING (Windows' O_DIRECT analogue): it would force sector-aligned
    // offsets/sizes/buffers, which the synchronous path in _win_submit_one doesn't guarantee.
    const DWORD access = read_op ? GENERIC_READ : GENERIC_WRITE;
    const DWORD disposition = read_op ? OPEN_EXISTING : OPEN_ALWAYS;
    const auto fd = CreateFileA(
        filename, access, FILE_SHARE_READ, nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fd == INVALID_HANDLE_VALUE) {
        const auto error_code = GetLastError();
        const auto error_msg = read_op ? " open for read " : " open for write ";
        report_file_error(filename, error_msg, static_cast<int>(error_code));
        return AIO_INVALID_FD;
    }
    return fd;
#else
    const int flags = read_op ? (O_RDONLY | O_DIRECT) : (O_WRONLY | O_CREAT | O_DIRECT);
#if defined(__ENABLE_CANN__)
    int* flags_ptr = (int*)&flags;
    *flags_ptr = read_op ? (O_RDONLY) : (O_WRONLY | O_CREAT);
#endif
    const int mode = 0600;
    const auto fd = open(filename, flags, mode);
    if (fd == -1) {
        const auto error_code = errno;
        const auto error_msg = read_op ? " open for read " : " open for write ";
        report_file_error(filename, error_msg, error_code);
        return -1;
    }
    return fd;
#endif
}

void close_file(const aio_fd_t fd)
{
#if defined(_WIN32)
    CloseHandle(fd);
#else
    close(fd);
#endif
}

int regular_read(const char* filename, std::vector<char>& buffer)
{
    // Uses the C stdio API (rather than POSIX open/read) so this helper works unchanged on
    // Windows, which has no open()/read() pair with the same semantics.
    auto* file = fopen(filename, "rb");
    assert(file != nullptr);
    assert(fseek(file, 0, SEEK_END) == 0);
    const int64_t num_bytes = ftell(file);
    assert(num_bytes >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    buffer.resize(num_bytes);
    const int64_t read_bytes =
        num_bytes == 0 ? 0 : static_cast<int64_t>(fread(buffer.data(), 1, num_bytes, file));

    if (read_bytes != num_bytes) {
        std::cerr << "read error " << " read_bytes (fread) = " << read_bytes
                  << " num_bytes (ftell) = " << num_bytes << std::endl;
    }
    assert(read_bytes == num_bytes);
    fclose(file);
    return 0;
}

static bool _validate_buffer(const char* filename, void* aio_buffer, const int64_t num_bytes)
{
    std::vector<char> regular_buffer;
    const auto reg_ret = regular_read(filename, regular_buffer);
    assert(0 == reg_ret);
    std::cout << "regular read of " << filename << " returned " << regular_buffer.size() << " bytes"
              << std::endl;

    if (static_cast<int64_t>(regular_buffer.size()) != num_bytes) { return false; }

    return (0 == memcmp(aio_buffer, regular_buffer.data(), regular_buffer.size()));
}

bool validate_aio_operation(const bool read_op,
                            const char* filename,
                            void* aio_buffer,
                            const int64_t num_bytes)
{
    const auto msg_suffix = std::string("deepspeed_aio_") +
                            std::string(read_op ? "read()" : "write()") +
                            std::string("using read()");

    if (false == _validate_buffer(filename, aio_buffer, num_bytes)) {
        std::cout << "Fail: correctness of " << msg_suffix << std::endl;
        return false;
    }

    std::cout << "Pass: correctness of  " << msg_suffix << std::endl;
    return true;
}
