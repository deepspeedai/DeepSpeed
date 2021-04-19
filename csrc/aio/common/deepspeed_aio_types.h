/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <libaio.h>
#include <stdlib.h>

#include <string>
#include <vector>

using namespace std;

struct deepspeed_aio_latency_t {
    double _min_usec;
    double _max_usec;
    double _avg_usec;

    void dump(const std::string tag);
    void accumulate(const deepspeed_aio_latency_t&);
    void scale(const float value);
};

struct deepspeed_aio_perf_t {
    deepspeed_aio_latency_t _submit;
    deepspeed_aio_latency_t _complete;
    double _e2e_usec;
    double _e2e_rate_GB;
};

struct deepspeed_aio_config_t {
    const int _block_size;
    const int _queue_depth;
    const bool _single_submit;
    const bool _overlap_events;
    const bool _lock_memory;

    deepspeed_aio_config_t();
    deepspeed_aio_config_t(const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const bool lock_memory);
};

struct aio_context {
    io_context_t _io_ctxt;
    std::vector<struct io_event> _io_events;
    std::vector<struct iocb*> _iocbs;
    int _block_size;
    int _queue_depth;

    aio_context(const int block_size, const int queue_depth);
    ~aio_context();
};
