// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <deepspeed_aio_utils.h>
#include <stdlib.h>
#include <memory>
#include <string>

using namespace std;

void do_aio_operation_sequential(const bool read_op,
                                 std::unique_ptr<aio_context>& aio_ctxt,
                                 std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                                 deepspeed_aio_config_t* config,
                                 deepspeed_aio_perf_t* perf);

void do_aio_operation_overlap(const bool read_op,
                              std::unique_ptr<aio_context>& aio_ctxt,
                              std::unique_ptr<io_xfer_ctxt>& xfer_ctxt,
                              deepspeed_aio_config_t* config,
                              deepspeed_aio_perf_t* perf);

int open_file(const char* filename, const bool read_op);

void report_file_error(const char* filename, const std::string file_op, const int error_code);

int regular_read(const char* filename, std::vector<char>& buffer);

bool validate_aio_operation(const bool read_op,
                            const char* filename,
                            void* aio_buffer,
                            const long long int num_bytes);
