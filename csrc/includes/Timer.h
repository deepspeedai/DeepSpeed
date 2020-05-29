
#ifndef __TIMER_H__
#define __TIMER_H__

#include <cuda_runtime.h>
#include <chrono>
#include "cuda.h"

class GPUTimer {
    cudaEvent_t start, stop;

public:
    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GPUTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    inline void Record() { cudaEventRecord(start); }
    inline void Elapsed(float& time_elapsed)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed, start, stop);
    }
};

class CPUTimer {
    std::chrono::high_resolution_clock::time_point start;

public:
    CPUTimer() : start(std::chrono::high_resolution_clock::now()) {}
    inline void Reset() { start = std::chrono::high_resolution_clock::now(); }
    inline float Elapsed()
    {
        auto temp = start;
        start = std::chrono::high_resolution_clock::now();
        return (float)(std::chrono::duration_cast<std::chrono::microseconds>(start - temp).count() /
                       1e3);
    }
};

#endif
