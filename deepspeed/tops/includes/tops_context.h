#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include "cuda.h"
#include "curand.h"
#include <cuda_runtime_api.h>

class TOPSContext {
public:
    TOPSContext() : _seed(42), _curr_offset(0)
    {
        curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(_gen, 123);
    }

    virtual ~TOPSContext()
    {
    }

    static TOPSContext& Instance()
    {
        static TOPSContext _ctx;
        return _ctx;
    }
    
    curandGenerator_t& GetRandGenerator() { return _gen; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }


private:
    curandGenerator_t _gen;
    uint64_t _seed;
    uint64_t _curr_offset;
};