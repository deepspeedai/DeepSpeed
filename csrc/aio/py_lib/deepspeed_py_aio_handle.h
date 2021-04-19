/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <condition_variable>
#include <memory>
#include "deepspeed_aio_thread.h"

struct deepspeed_aio_handle_t {
    std::unique_ptr<struct aio_context> _aio_ctxt;
    const bool _single_submit;
    const bool _overlap_events;
    const int _num_threads;
    deepspeed_aio_config_t _aio_config;

    std::vector<std::shared_ptr<struct deepspeed_aio_thread_t>> _thread_contexts;
    std::vector<std::thread> _threads;
    int _num_pending_ops;

    deepspeed_aio_handle_t(const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const int num_threads);

    ~deepspeed_aio_handle_t();

    const int get_block_size() const;
    const int get_queue_depth() const;
    const bool get_single_submit() const;
    const bool get_overlap_events() const;
    const int get_thread_count() const;

    int read(torch::Tensor& buffer, const char* filename, const bool validate);

    int write(const torch::Tensor& buffer, const char* filename, const bool validate);

    int pread(const torch::Tensor& buffer,
              const char* filename,
              const bool validate,
              const bool async);

    int pwrite(const torch::Tensor& buffer,
               const char* filename,
               const bool validate,
               const bool async);

    int sync_pread(torch::Tensor& buffer, const char* filename);

    int sync_pwrite(const torch::Tensor& buffer, const char* filename);

    int async_pread(torch::Tensor& buffer, const char* filename);

    int async_pwrite(const torch::Tensor& buffer, const char* filename);

    int wait();

    void _stop_threads();

    void _schedule_aio_work(std::shared_ptr<struct io_op_desc_t> scheduled_op);

    std::shared_ptr<struct io_op_desc_t> _wait_for_aio_work();

    bool _is_valid_parallel_aio_op(const bool read_op, const long long int num_bytes);
};
