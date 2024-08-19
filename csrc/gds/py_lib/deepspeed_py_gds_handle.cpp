// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
    GPUDirect Storage functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_gds_handle.h"
#include <cstdlib>
#include "deepspeed_gds_op.h"

using namespace std;

int deepspeed_gds_handle_t::s_cuFile_init = 0;

deepspeed_gds_handle_t::deepspeed_gds_handle_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const int num_threads)
    : deepspeed_io_handle_t(block_size, queue_depth, single_submit, overlap_events, num_threads)
{
    _init_cuFile(block_size, queue_depth, num_threads);
}

deepspeed_gds_handle_t::~deepspeed_gds_handle_t() { _close_cuFile(); }

void deepspeed_gds_handle_t::_init_cuFile(const int block_size,
                                          const int queue_depth,
                                          const int num_threads)
{
    if (deepspeed_gds_handle_t::s_cuFile_init == 0) {
        std::string depthStr = std::to_string(queue_depth);
        std::string threadsStr = std::to_string(num_threads);
        std::string json1 = R"({"execution": {"max_io_queue_depth": )" + depthStr + ", ";
        std::string json2 = R"("max_request_parallelism": )" + threadsStr + ", ";
        std::string json3 = R"("max_io_threads": )" + threadsStr + ", ";
        std::string json4 = R"("parallel_io": true, "min_io_threshold_size_kb": 8192}})";
        std::ofstream outFile("local_cufile.json");
        if (outFile.is_open()) {
            outFile << json1 + json2 + json3 + json4;
            outFile.close();
        } else {
            std::cerr << "Can't open local cufile" << std::endl;
            exit(EXIT_FAILURE);
        }
        // TODO: Address the following issues with this code
        // (1) Fix C++14 warning
        // (2) Create file in a different location than PWD
        // (3) Handle multi-GPU/multi-rank scenarios: should cufile be shared, is per-rank cufile
        // safe?
        putenv("CUFILE_ENV_PATH_JSON=$PWD/local_cufile.json");
        cuFileDriverOpen();
        cudaCheckError();
        size_t direct_io_size = (size_t)block_size / 1024;
        CUfileError_t status = cuFileDriverSetMaxDirectIOSize(direct_io_size);
        if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "file register error:" << cuFileGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    deepspeed_gds_handle_t::s_cuFile_init++;
}

void deepspeed_gds_handle_t::_close_cuFile()
{
    deepspeed_gds_handle_t::s_cuFile_init--;
    if (deepspeed_gds_handle_t::s_cuFile_init == 0) { cuFileDriverClose(); }
}

torch::Tensor deepspeed_gds_handle_t::new_pinned_device_tensor(const size_t num_elem,
                                                               const torch::Tensor& example_tensor)
{
    auto options = torch::TensorOptions().dtype(example_tensor.scalar_type()).device(torch::kCUDA);
    auto dev_tensor = torch::empty(num_elem, options);
    pin_device_tensor(dev_tensor);
    return dev_tensor;
}

bool deepspeed_gds_handle_t::free_pinned_device_tensor(torch::Tensor& buffer)
{
    unpin_device_tensor(buffer);
    return true;
}

bool deepspeed_gds_handle_t::pin_device_tensor(const torch::Tensor& buffer)
{
    gds_op_desc_t::add_buffer_to_registry(buffer);
    return true;
}

bool deepspeed_gds_handle_t::unpin_device_tensor(const torch::Tensor& buffer)
{
    gds_op_desc_t::remove_buffer_from_registry(buffer);
    return true;
}

std::shared_ptr<struct io_op_desc_t> deepspeed_gds_handle_t::_create_io_op_desc(
    const bool read_op,
    const torch::Tensor& buffer,
    const int fd,
    const char* filename,
    const long long int file_num_bytes,
    const bool validate)
{
    if (buffer.is_cuda()) {
        return std::make_shared<gds_op_desc_t>(
            read_op, buffer, fd, filename, file_num_bytes, _num_threads, validate);
    }
    return deepspeed_io_handle_t::_create_io_op_desc(
        read_op, buffer, fd, filename, file_num_bytes, validate);
}
