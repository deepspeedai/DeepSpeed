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

deepspeed_aio_config_t::deepspeed_aio_config_t()
    : _block_size(c_block_size),
      _queue_depth(c_io_queue_depth),
      _single_submit(false),
      _overlap_events(false),
      _lock_memory(false)
{
}

deepspeed_aio_config_t::deepspeed_aio_config_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const bool lock_memory)
    : _block_size(block_size),
      _queue_depth(queue_depth),
      _single_submit(single_submit),
      _overlap_events(overlap_events),
      _lock_memory(lock_memory)
{
}

void deepspeed_aio_latency_t::dump(const std::string tag)
{
    std::cout << tag << _min_usec << " " << _max_usec << " " << _avg_usec << " " << std::endl;
}

void deepspeed_aio_latency_t::accumulate(const struct deepspeed_aio_latency_t& other)
{
    _min_usec += other._min_usec;
    _max_usec += other._max_usec;
    _avg_usec += other._avg_usec;
}

void deepspeed_aio_latency_t::scale(const float scaler)
{
    _min_usec *= scaler;
    _max_usec *= scaler;
    _avg_usec *= scaler;
}

aio_context::aio_context(const int block_size, const int queue_depth)
{
    _block_size = block_size;
    _queue_depth = queue_depth;
    for (auto i = 0; i < queue_depth; ++i) {
        _iocbs.push_back((struct iocb*)calloc(1, sizeof(struct iocb)));
    }
    _io_events.resize(queue_depth);
    io_queue_init(queue_depth, &_io_ctxt);
}

aio_context::~aio_context()
{
    for (auto& iocb : _iocbs) { free(iocb); }
    _io_events.resize(0);
    io_queue_release(_io_ctxt);
}
