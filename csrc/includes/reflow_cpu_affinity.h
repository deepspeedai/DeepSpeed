// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

#pragma once

#if defined(__linux__) && defined(_OPENMP)
#include <omp.h>
#include <sched.h>
#endif

// OpenMP retains its helper threads when the calling worker changes CPU masks.
// Apply the caller's current mask inside the existing team, before it does any work.
class ReflowCPUAffinity {
public:
    ReflowCPUAffinity()
    {
#if defined(__linux__) && defined(_OPENMP)
        CPU_ZERO(&_mask);
        _valid = sched_getaffinity(0, sizeof(_mask), &_mask) == 0;
#endif
    }

    void apply() const
    {
#if defined(__linux__) && defined(_OPENMP)
        if (_valid && omp_get_thread_num() != 0) {
            cpu_set_t current;
            // Most kernels reuse the same team and mask; avoid the scheduler update in that case.
            if (sched_getaffinity(0, sizeof(current), &current) != 0 ||
                !CPU_EQUAL(&current, &_mask)) {
                sched_setaffinity(0, sizeof(_mask), &_mask);
            }
        }
#endif
    }

private:
#if defined(__linux__) && defined(_OPENMP)
    cpu_set_t _mask;
    bool _valid;
#endif
};
