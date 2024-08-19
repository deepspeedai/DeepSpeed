// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cstring>

// CUDA/cuFile includes
#include <cuda.h>
#include <cuda_runtime.h>
#include "cufile.h"

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess) {                                                              \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    }

#define check_cudadrivercall(fn)                                                           \
    do {                                                                                   \
        CUresult res = fn;                                                                 \
        if (res != CUDA_SUCCESS) {                                                         \
            const char* str = nullptr;                                                     \
            cuGetErrorName(res, &str);                                                     \
            std::cerr << "cuda driver api call failed " << #fn << " res : " << res << ", " \
                      << __LINE__ << ":" << str << std::endl;                              \
            std::cerr << "EXITING program!!!" << std::endl;                                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define check_cudaruntimecall(fn)                                                         \
    do {                                                                                  \
        cudaError_t res = fn;                                                             \
        if (res != cudaSuccess) {                                                         \
            const char* str = cudaGetErrorName(res);                                      \
            std::cerr << "cuda runtime api call failed " << #fn << __LINE__ << ":" << str \
                      << std::endl;                                                       \
            std::cerr << "EXITING program!!!" << std::endl;                               \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

#define check_cuFileCall(fn, api_msg)                                                  \
    do {                                                                               \
        CUfileError_t status = fn;                                                     \
        if (status.err != CU_FILE_SUCCESS) {                                           \
            std::cout << api_msg << " failed with error " << CUFILE_ERRSTR(status.err) \
                      << std::endl;                                                    \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

//
// cuda driver error description
//
static inline const char* GetCuErrorString(CUresult curesult)
{
    const char* descp;
    if (cuGetErrorName(curesult, &descp) != CUDA_SUCCESS) descp = "unknown cuda error";
    return descp;
}

//
// cuFile APIs return both cuFile specific error codes as well as POSIX error codes
// for ease, the below template can be used for getting the error description depending
// on its type.

// POSIX
template <class T,
          typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status)
{
    status = std::abs(status);
    return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                                 : std::string(std::strerror(status));
}

// CUfileError_t
template <class T,
          typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status)
{
    std::string errStr = cuFileGetErrorString(static_cast<int>(status.err));
    if (IS_CUDA_ERR(status)) errStr.append(".").append(GetCuErrorString(status.cu_err));
    return errStr;
}
