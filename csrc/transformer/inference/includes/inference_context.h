/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "cublas_v2.h"
#include "cuda.h"

#define MEGABYTE (1024 * 1024)
#define GIGABYTE (1024 * 1024 * 1024)

#define MAX_OUT_TOKENS 8192
#define WARP_SIZE 32

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                                          \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N)
{
    return std::max(
        std::min((N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS, DS_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
}

class Context {
public:
    Context()
        : _workspace(nullptr),
          _seed(42),
          _curr_offset(0),
          _stream(0),
          _free_memory_size(0),
          _num_tokens(1)
    {
        if (cublasCreate(&_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            auto message = std::string("Fail to create cublas handle.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
#ifndef __HIP_PLATFORM_HCC__
        cublasSetMathMode(_cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
        cudaEventCreate(&_comp1_event);
        cudaEventCreate(&_comp2_event);
        cudaEventCreate(&_comp_event);
        cudaEventCreate(&_comm_event);
    }

    virtual ~Context()
    {
        cublasDestroy(_cublasHandle);
        cudaFree(_workspace);
        cudaEventDestroy(_comp1_event);
        cudaEventDestroy(_comp2_event);
        cudaEventDestroy(_comp_event);
        cudaEventDestroy(_comm_event);
    }

    static Context& Instance()
    {
        static Context _ctx;
        return _ctx;
    }

    void GenWorkSpace(const unsigned& num_layers,
                      const size_t& batch_size,
                      const size_t& hidden_dim,
                      const unsigned& mp_size,
                      const bool& external_cache,
                      const size_t& elem_size,
                      const unsigned& rank)
    {
        size_t total_size;
        if (!_free_memory_size) { cudaMemGetInfo(&_free_memory_size, &total_size); }

        size_t activation_size = 16 * hidden_dim * batch_size;
        size_t cache_size = num_layers * batch_size * (hidden_dim / mp_size) * 2;
        _max_seq_len =
            (((_free_memory_size - (_free_memory_size > GIGABYTE ? 500 : 100) * MEGABYTE) /
              elem_size)) /
            (activation_size + cache_size);
        size_t workSpaceSize = (external_cache ? activation_size : (activation_size + cache_size)) *
                               _max_seq_len * elem_size;
        _max_seq_len = std::min((size_t)MAX_OUT_TOKENS, _max_seq_len);
        if (rank == 0 && !_workspace)
            printf(
                "Free memory : %lu (Bytes)  Total memory: %lu (Bytes)  Setting maximum total "
                "tokens (input + output) to %lu \n",
                _free_memory_size,
                total_size,
                _max_seq_len);
        if (!_workspace) {
            assert(_workspace == nullptr);
            cudaMalloc(&_workspace, workSpaceSize);
        } else if (_workSpaceSize < workSpaceSize) {
            cudaFree(_workspace);
            cudaMalloc(&_workspace, workSpaceSize);
        }

        if (!_workspace) {
            printf("Requested:\t%lu\nFree:\t%lu\nTotal:\t%lu\n",
                   workSpaceSize,
                   _free_memory_size,
                   total_size);
            throw std::runtime_error("Workspace is null.");
        }
        _workSpaceSize = workSpaceSize;
    }
    inline size_t GetMaxTokenLenght() const { return _max_seq_len; }

    cudaEvent_t GetCompEvent(int id) { return id == 1 ? _comp1_event : _comp2_event; }

    size_t get_workspace_size() const { return _workSpaceSize; }
    void* GetWorkSpace() { return _workspace; }

    inline unsigned new_token(unsigned layer_id)
    {
        if (layer_id == 0) _token_length++;
        return _token_length;
    }

    inline void reset_tokens(unsigned initial_tokens = 1)
    {
        _num_tokens = initial_tokens;
    }  //_token_length = 0; }

    inline unsigned current_tokens() const { return _num_tokens; }

    inline void advance_tokens() { _num_tokens++; }

    cudaStream_t GetCommStream(bool async_op = false)
    {
        if (!_comm_stream)
            _comm_stream = async_op ? at::cuda::getStreamFromPool(true)
                                    : at::cuda::getCurrentCUDAStream();
        return _comm_stream;
    }
    cudaStream_t GetCurrentStream(bool other_stream = false)
    {
        // get current pytorch stream.
        if (other_stream) {
            if (!_stream) _stream = at::cuda::getStreamFromPool(true);
            return _stream;
        }
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        return stream;
    }

    cublasHandle_t GetCublasHandle() { return _cublasHandle; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

    const std::vector<std::array<int, 3>>& GetGemmAlgos() const { return _gemm_algos; }

    inline void SynchComp()
    {
        cudaEventRecord(_comp_event, _comp_stream);
        cudaStreamWaitEvent(_comm_stream, _comp_event, 0);
    }
    inline void SynchComm()
    {
        cudaEventRecord(_comm_event, _comm_stream);
        cudaStreamWaitEvent(_comp_stream, _comm_event, 0);
    }

private:
    cublasHandle_t _cublasHandle;

    cudaEvent_t _comp_event;
    cudaEvent_t _comm_event;

    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;

    size_t _workSpaceSize;
    size_t _free_memory_size;

    size_t _max_seq_len;

    cudaEvent_t _comp1_event;
    cudaEvent_t _comp2_event;

    cudaStream_t _stream;

    unsigned _token_length;
    unsigned _num_tokens;
    std::vector<std::array<int, 3>> _gemm_algos;

    cudaStream_t _comp_stream;
    cudaStream_t _comm_stream;

    std::unordered_map<int, int> _world_sizes;
};
