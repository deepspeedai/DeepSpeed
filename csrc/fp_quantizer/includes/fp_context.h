// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <sys/time.h>
#include <map>
#include <memory>
#include <stack>
#include <string>
#define WARP_SIZE 32

class FPContext {
public:
    FPContext() : _seed(42)
    {
        curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(_gen, 123);
    }

    virtual ~FPContext() {}

    static FPContext& Instance()
    {
        static FPContext _ctx;
        return _ctx;
    }

    curandGenerator_t& GetRandGenerator() { return _gen; }

    cudaStream_t GetCurrentStream()
    {
        // get current pytorch stream.
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        return stream;
    }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

private:
    curandGenerator_t _gen;
    cublasHandle_t _cublasHandle;
    uint64_t _seed;
    uint64_t _curr_offset;
};
