// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <condition_variable>
#include <memory>
#include <queue>
#include "deepspeed_cpu_op.h"

struct thread_sync_t {
    std::mutex _mutex;
    std::condition_variable _cond_var;
};

struct deepspeed_aio_thread_t {
    const int _tid;
    deepspeed_aio_config_t& _aio_config;

    std::unique_ptr<struct aio_context> _aio_ctxt;
    std::queue<std::shared_ptr<struct io_op_desc_t>> _work_queue;
    std::queue<std::shared_ptr<struct io_op_desc_t>> _complete_queue;

    bool _time_to_exit;

    struct thread_sync_t _work_sync;
    struct thread_sync_t _complete_sync;

    deepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config);

    ~deepspeed_aio_thread_t();

    void run();
};
