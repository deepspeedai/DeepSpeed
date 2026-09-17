// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "page_alloc.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>

void* ds_page_aligned_alloc(const int64_t size, const bool lock)
{
    void* ptr;
    int retval;

    retval = posix_memalign(&ptr, (size_t)sysconf(_SC_PAGESIZE), size);
    if (retval) { return nullptr; }

    if (lock == false) { return ptr; }

    auto mlock_ret = mlock(ptr, size);
    if (mlock_ret != 0) {
        auto mlock_error = errno;
        std::cerr << "mlock failed to allocate " << size << " bytes with error no " << mlock_error
                  << " msg " << strerror(mlock_error) << std::endl;
        free(ptr);
        return nullptr;
    }

    return ptr;
}
