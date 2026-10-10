// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "page_alloc.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>

void* ds_page_aligned_alloc(const int64_t size, const bool lock)
{
#if defined(_WIN32)
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    void* ptr = _aligned_malloc(static_cast<size_t>(size), sys_info.dwPageSize);
    if (ptr == nullptr) { return nullptr; }

    if (lock == false) { return ptr; }

    // VirtualLock only succeeds up to the process's working-set quota, so that quota must be
    // raised by the allocation size first -- unlike POSIX mlock, which has no such per-process cap.
    SIZE_T min_ws = 0, max_ws = 0;
    GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws, &max_ws);
    SetProcessWorkingSetSize(GetCurrentProcess(),
                             min_ws + static_cast<SIZE_T>(size),
                             max_ws + static_cast<SIZE_T>(size));

    if (!VirtualLock(ptr, static_cast<size_t>(size))) {
        std::cerr << "VirtualLock failed to allocate " << size << " bytes with error no "
                  << GetLastError() << std::endl;
        _aligned_free(ptr);
        return nullptr;
    }

    return ptr;
#else
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
#endif
}
