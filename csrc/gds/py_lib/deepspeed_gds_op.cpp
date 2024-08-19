// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_gds_op.h"

using namespace std;

// For when there is more than 1 device
static std::map<const int64_t, std::set<void*>> base_ptr_registry;

static void _safe_handle_register(const int fd, CUfileDescr_t& cf_descr, CUfileHandle_t& cf_handle)
{
    memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "file register error:" << cuFileGetErrorString(status) << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }
}

static void* _find_base_ptr(const int64_t device, char* buf_ptr)
{
    void* base_ptr = nullptr;
    int64_t last = -1;
    int64_t ptr_diff;
    for (const auto& value : base_ptr_registry[device]) {
        ptr_diff = buf_ptr - (char*)value;
        if (last == -1 && ptr_diff >= 0) {
            last = ptr_diff;
            base_ptr = value;
        } else if (ptr_diff < last && ptr_diff >= 0) {
            last = ptr_diff;
            base_ptr = value;
        }
    }
    if (!base_ptr || buf_ptr < base_ptr) {
        std::cerr << "BASE PTR ERROR :" << base_ptr << " BUF PTR " << (void*)buf_ptr << std::endl;
        for (const auto& value : base_ptr_registry[device]) {
            std::cerr << "BASE PTR AVAIL :" << value << std::endl;
        }
        exit(EXIT_FAILURE);
    }

    return base_ptr;
}

void gds_op_desc_t::add_buffer_to_registry(const torch::Tensor& buffer)
{
    const int64_t device = buffer.get_device();
    void* reg_ptr = buffer.data_ptr();

    // std::cout << "REG PTR " <<  reg_ptr << std::endl;
    // TODO: add checking to make sure pointer isn't already in set
    const auto it = base_ptr_registry.find(device);
    if (it == base_ptr_registry.end()) {
        std::set<void*> new_ptr_set;
        new_ptr_set.insert(reg_ptr);
        base_ptr_registry.insert(std::pair<const int64_t, std::set<void*>>(device, new_ptr_set));
    } else {
        base_ptr_registry[device].insert(reg_ptr);
    }

    check_cudaruntimecall(cudaSetDevice(device));
    CUfileError_t status = cuFileBufRegister(reg_ptr, buffer.nbytes(), 0);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "buffer register failed:" << cuFileGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void gds_op_desc_t::remove_buffer_from_registry(const torch::Tensor& buffer)
{
    const int64_t device = buffer.get_device();
    void* reg_ptr = buffer.data_ptr();

    // std::cout << "DEREG PTR " <<  reg_ptr << std::endl;
    check_cudaruntimecall(cudaSetDevice(device));
    cuFileBufDeregister(reg_ptr);

    // Remove from tracked registry
    base_ptr_registry[device].erase(reg_ptr);
}

gds_op_desc_t::gds_op_desc_t(const bool read_op,
                             const torch::Tensor& buffer,
                             const int fd,
                             const char* filename,
                             const long long int file_num_bytes,
                             const int num_threads,
                             const bool validate)
    : io_op_desc_t(read_op, buffer, fd, filename, file_num_bytes, num_threads, validate)
{
    _contiguous_buffer = _buffer.contiguous();
    const int64_t device = _buffer.get_device();
    check_cudaruntimecall(cudaSetDevice(device));
    _base_ptr = _find_base_ptr(device, (char*)_contiguous_buffer.data_ptr());

    _safe_handle_register(fd, _cf_descr, _cf_handle);
}

char* gds_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void gds_op_desc_t::finish() { cuFileHandleDeregister(_cf_handle); }

void gds_op_desc_t::validate()
{
    check_cudaruntimecall(cudaSetDevice(_buffer.get_device()));
    const auto cpu_buffer = _buffer.to(torch::kCPU);
    validate_aio_operation(
        _read_op, _filename.c_str(), (char*)(cpu_buffer.data_ptr()), _file_num_bytes);
}

void gds_op_desc_t::run(const int tid,
                        std::unique_ptr<aio_context>& aio_ctxt,
                        deepspeed_aio_config_t* aio_config)
{
    assert(tid < _num_threads);
    check_cudaruntimecall(cudaSetDevice(_buffer.get_device()));
    int64_t buf_offset = data_ptr() + (_num_bytes_per_thread * tid) - (char*)_base_ptr;
    const auto file_offset = _num_bytes_per_thread * tid;

    if (_read_op) {
        auto ret =
            cuFileRead(_cf_handle, _base_ptr, _num_bytes_per_thread, file_offset, buf_offset);
        if (ret < 0) { _report_error(ret, errno, buf_offset); }
    } else {
        auto ret =
            cuFileWrite(_cf_handle, _base_ptr, _num_bytes_per_thread, file_offset, buf_offset);
        if (ret < 0) { _report_error(ret, errno, buf_offset); }
    }
}

void gds_op_desc_t::_report_error(const ssize_t return_code,
                                  const int error_num,
                                  const off_t offset)
{
    const auto op_string = _read_op ? "read failed with " : "write failed with ";
    const auto error_string = IS_CUFILE_ERR(return_code) ? "cuFile error: " : "posix error: ";
    const auto error_code = IS_CUFILE_ERR(return_code) ? cuFileGetErrorString(return_code)
                                                       : cuFileGetErrorString(error_num);
    std::cerr << op_string << error_string << error_code << " return code = " << return_code
              << " filename = " << _filename.c_str() << " num bytes = " << _num_bytes_per_thread
              << " offset = " << offset << std::endl;
    exit(EXIT_FAILURE);
}
