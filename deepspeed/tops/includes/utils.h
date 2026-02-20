// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Centralized header file for preprocessor macros and constants
used throughout the codebase.
*/

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

#include <cuda_bf16.h>

#define DS_HD_INLINE __host__ __device__ __forceinline__
#define DS_D_INLINE __device__ __forceinline__

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 32;

#define HALF_PRECISION_AVAILABLE = 1
#define PTX_AVAILABLE

#define ASYNC_COPY_AVAILABLE

#include <cooperative_groups.h>

#include <stdint.h>

inline int next_pow2(const int val)
{
    int rounded_val = val - 1;
    rounded_val |= rounded_val >> 1;
    rounded_val |= rounded_val >> 2;
    rounded_val |= rounded_val >> 4;
    rounded_val |= rounded_val >> 8;
    return rounded_val + 1;
}



namespace conversion {

// Basic primitive for constructing conversions
template <typename TO, typename FROM>
DS_D_INLINE TO to(FROM val)
{
    return to(val);
}

// Specializations

/********************* Identity Conversions *********************/
/*
Identity conversions are useful in templated functions where we might have
a fixed destination type. For example, I might have a kernel that accepts
__half, __nv_bfloat16, and float but always want to do the core computation
at floating point:

T mem_value = input[idx];
float compute_value = conversion::to<float, T>(mem_value);

In practice, we should be able to elide the second template parameter:
float compute_val = conversion::to<float>(mem_value);

In this case, we need an implementation to handle the T = float case

NOTE: The type inferencing system appears to be unable to handle inferring the first
template parameter, even in the trivial case.
*/

// Floating point types
template <>
DS_D_INLINE double to(double val)
{
    return val;
}
template <>
DS_D_INLINE float to(float val)
{
    return val;
}
template <>
DS_D_INLINE __half to(__half val)
{
    return val;
}
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE __nv_bfloat16 to(__nv_bfloat16 val)
{
    return val;
}
#endif

// Integer types
template <>
DS_D_INLINE int8_t to(int8_t val)
{
    return val;
}
template <>
DS_D_INLINE uint8_t to(uint8_t val)
{
    return val;
}
template <>
DS_D_INLINE int16_t to(int16_t val)
{
    return val;
}
template <>
DS_D_INLINE uint16_t to(uint16_t val)
{
    return val;
}
template <>
DS_D_INLINE int32_t to(int32_t val)
{
    return val;
}
template <>
DS_D_INLINE uint32_t to(uint32_t val)
{
    return val;
}
template <>
DS_D_INLINE int64_t to(int64_t val)
{
    return val;
}
template <>
DS_D_INLINE uint64_t to(uint64_t val)
{
    return val;
}

// TODO: evaluate if we want bools

/*********************  To Double Conversions *********************/

// * to double variants

// Would normally like to not use C cast, but this is an important enough conversion
// to keep
template <>
DS_D_INLINE double to(float val)
{
#ifdef PTX_AVAILABLE
    double ret_val;
    asm("ctv.rn.f64.f32 %0, %1;\n" : "=d"(ret_val) : "f"(val));
    return ret_val;
#else
    return double(val);
#endif
}
// Note: there is a CVT instruction for __half -> double, but there's no inline interface
// for passing a single half value
template <>
DS_D_INLINE double to(__half val)
{
    return to<double>(__half2float(val));
}
//template <>
//DS_D_INLINE double to(int64_t val)
//{
//    return __ll2double_rn(val);
//}
template <>
DS_D_INLINE double to(int32_t val)
{
    return __int2double_rn(val);
}
template <>
DS_D_INLINE double to(int16_t val)
{
    return __int2double_rn(val);
}
template <>
DS_D_INLINE double to(int8_t val)
{
    return __int2double_rn(val);
}
template <>
DS_D_INLINE double to(uint64_t val)
{
    return __ull2double_rn(val);
}
template <>
DS_D_INLINE double to(uint32_t val)
{
    return __uint2double_rn(val);
}
template <>
DS_D_INLINE double to(uint16_t val)
{
    return __uint2double_rn(val);
}
template <>
DS_D_INLINE double to(uint8_t val)
{
    return __uint2double_rn(val);
}

// Same applies here
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE double to(__nv_bfloat16 val)
{
    return to<double>(__bfloat162float(val));
}
#endif

/*********************  To Float Conversions *********************/

template <>
DS_D_INLINE float to(double val)
{
    return __double2float_rn(val);
}
template <>
DS_D_INLINE float to(__half val)
{
    return __half2float(val);
}
template <>
DS_D_INLINE float to(int64_t val)
{
    return __ll2float_rn(val);
}
template <>
DS_D_INLINE float to(int32_t val)
{
    return __int2float_rn(val);
}
template <>
DS_D_INLINE float to(int16_t val)
{
    return __int2float_rn(val);
}
template <>
DS_D_INLINE float to(int8_t val)
{
    return __int2float_rn(val);
}
template <>
DS_D_INLINE float to(uint64_t val)
{
    return __ull2float_rn(val);
}
template <>
DS_D_INLINE float to(uint32_t val)
{
    return __uint2float_rn(val);
}
template <>
DS_D_INLINE float to(uint16_t val)
{
    return __uint2float_rn(val);
}
template <>
DS_D_INLINE float to(uint8_t val)
{
    return __uint2float_rn(val);
}

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE float to(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}
#endif

/*********************  To Float2 Conversions *********************/
template <>
DS_D_INLINE float2 to(__half2 val)
{
    return __half22float2(val);
}

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE float2 to(__nv_bfloat162 val)
{
    return __bfloat1622float2(val);
}
#endif

/*********************  To Half Conversions *********************/
template <>
DS_D_INLINE __half to(double val)
{
#ifdef __HIP_PLATFORM_AMD__
    float val_f = __double2float_rn(val);
    return __float2half(val_f);
#else
    return __double2half(val);
#endif
}
template <>
DS_D_INLINE __half to(float val)
{
    return __float2half(val);
}
template <>
DS_D_INLINE __half to(int64_t val)
{
    return __ll2half_rn(val);
}
template <>
DS_D_INLINE __half to(int32_t val)
{
    return __int2half_rn(val);
}
template <>
DS_D_INLINE __half to(int16_t val)
{
    return __short2half_rn(val);
}
template <>
DS_D_INLINE __half to(int8_t val)
{
    return __int2half_rn(val);
}
template <>
DS_D_INLINE __half to(uint64_t val)
{
    return __ull2half_rn(val);
}
template <>
DS_D_INLINE __half to(uint32_t val)
{
    return __uint2half_rn(val);
}
template <>
DS_D_INLINE __half to(uint16_t val)
{
    return __ushort2half_rn(val);
}
template <>
DS_D_INLINE __half to(uint8_t val)
{
    return __uint2half_rn(val);
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
DS_D_INLINE __half to(__nv_bfloat16 val)
{
    return to<__half>(to<float>(val));
}
#endif

/*********************  To Half2 Conversions *********************/
template <>
DS_D_INLINE __half2 to(float2 val)
{
    return __float22half2_rn(val);
}
template <>
DS_D_INLINE __half2 to(float val)
{
    return __float2half2_rn(val);
}

#ifdef BF16_AVAILABLE
// No direct conversion
template <>
DS_D_INLINE __half2 to(__nv_bfloat162 val)
{
    return to<__half2>(to<float2>(val));
}
#endif

/*********************  To BF16 Conversions *********************/
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE __nv_bfloat16 to(double val)
{
    return __double2bfloat16(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(float val)
{
    return __float2bfloat16(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int64_t val)
{
    return __ll2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int32_t val)
{
    return __int2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int16_t val)
{
    return __short2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(int8_t val)
{
    return __int2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint64_t val)
{
    return __ull2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint32_t val)
{
    return __uint2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint16_t val)
{
    return __ushort2bfloat16_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat16 to(uint8_t val)
{
    return __uint2bfloat16_rn(val);
}
#endif

/*********************  To BF162 Conversions *********************/
#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE __nv_bfloat162 to(float2 val)
{
    return __float22bfloat162_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat162 to(float val)
{
    return __float2bfloat162_rn(val);
}
template <>
DS_D_INLINE __nv_bfloat162 to(__half2 val)
{
    return to<__nv_bfloat162>(to<float2>(val));
}
#endif

/*********************  To INT64_T Conversions *********************/
template <> 
DS_D_INLINE int64_t to(double val)
{
    return __double2ll_rn(val);
}
template <>
DS_D_INLINE int64_t to(float val)
{
    return __float2ll_rn(val);
}
template <>
DS_D_INLINE int64_t to(__half val)
{
    return __half2ll_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int64_t to(__nv_bfloat16 val)
{
    return __bfloat162ll_rn(val);
}
#endif

/*********************  To INT32_T Conversions *********************/
template <>
DS_D_INLINE int32_t to(double val)
{
    return __double2int_rn(val);
}
template <>
DS_D_INLINE int32_t to(float val)
{
    return __float2int_rn(val);
}
template <>
DS_D_INLINE int32_t to(__half val)
{
    return __half2int_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int32_t to(__nv_bfloat16 val)
{
    return __bfloat162int_rn(val);
}
#endif

/*********************  To INT16_T Conversions *********************/
template <>
DS_D_INLINE int16_t to(double val)
{
    return __double2int_rn(val);
}
template <>
DS_D_INLINE int16_t to(float val)
{
    return __float2int_rn(val);
}
template <>
DS_D_INLINE int16_t to(__half val)
{
    return __half2int_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int16_t to(__nv_bfloat16 val)
{
    return __bfloat162int_rn(val);
}
#endif

/*********************  To INT8_T Conversions *********************/
template <>
DS_D_INLINE int8_t to(double val)
{
    return __double2int_rn(val);
}
template <>
DS_D_INLINE int8_t to(float val)
{
    return __float2int_rn(val);
}
template <>
DS_D_INLINE int8_t to(__half val)
{
    return __half2int_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE int8_t to(__nv_bfloat16 val)
{
    return __bfloat162int_rn(val);
}
#endif

/*********************  To UINT64_T Conversions *********************/
template <>
DS_D_INLINE uint64_t to(double val)
{
    return __double2ull_rn(val);
}
template <>
DS_D_INLINE uint64_t to(float val)
{
    return __float2ull_rn(val);
}
template <>
DS_D_INLINE uint64_t to(__half val)
{
    return __half2ull_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint64_t to(__nv_bfloat16 val)
{
    return __bfloat162ull_rn(val);
}
#endif

/*********************  To UINT32_T Conversions *********************/
template <>
DS_D_INLINE uint32_t to(double val)
{
    return __double2uint_rn(val);
}
template <>
DS_D_INLINE uint32_t to(float val)
{
    return __float2uint_rn(val);
}
template <>
DS_D_INLINE uint32_t to(__half val)
{
    return __half2uint_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint32_t to(__nv_bfloat16 val)
{
    return __bfloat162uint_rn(val);
}
#endif

/*********************  To UINT16_T Conversions *********************/
template <>
DS_D_INLINE uint16_t to(double val)
{
    return __double2uint_rn(val);
}
template <>
DS_D_INLINE uint16_t to(float val)
{
    return __float2uint_rn(val);
}
template <>
DS_D_INLINE uint16_t to(__half val)
{
    return __half2uint_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint16_t to(__nv_bfloat16 val)
{
    return __bfloat162uint_rn(val);
}
#endif

/*********************  To UINT8_T Conversions *********************/
template <>
DS_D_INLINE uint8_t to(double val)
{
    return __double2uint_rn(val);
}
template <>
DS_D_INLINE uint8_t to(float val)
{
    return __float2uint_rn(val);
}
template <>
DS_D_INLINE uint8_t to(__half val)
{
    return __half2uint_rn(val);
}
// No direct support for integer casts at the C++ level and I don't feel they're so important
// to demand an PTX at this time

#ifdef BF16_AVAILABLE
template <>
DS_D_INLINE uint8_t to(__nv_bfloat16 val)
{
    return __bfloat162uint_rn(val);
}
#endif

}  // namespace conversion

/////////////////////////////// Memory Access Utils ///////////////////////////////
namespace mem_access {

enum class LoadPolicy {
    CacheAll,       // Cache at all levels
    CacheGlobal,    // Cache at L2 only
    CacheStreaming  // Cache with evict first policy
};

enum class StorePolicy {
    Writeback,      // Cache in L1, write-back on eviction
    CacheGlobal,    // Bypass L1, write-back on eviction
    CacheStreaming  // Allocate cache line with evict first policy
};

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
__device__ __forceinline__ void load_global(void* dst, const void* src);

template <int AccessSize, LoadPolicy policy = LoadPolicy::CacheAll>
__device__ __forceinline__ void load_global(void* dst, const void* src, bool do_access);

// Shared accesses have no cache policy
template <int AccessSize>
__device__ __forceinline__ void load_shared(void* dst, const void* src);

template <int AccessSize>
__device__ __forceinline__ void load_shared(void* dst, const void* src, bool do_access);

template <int AccessSize, StorePolicy policy = StorePolicy::Writeback>
__device__ __forceinline__ void store_global(void* dst, const void* src);

// Shared accesses have no cache policy
template <int AccessSize>
__device__ __forceinline__ void store_shared(void* dst, const void* src);

#ifdef ASYNC_COPY_AVAILABLE
template <int AccessSize>
__device__ __forceinline__ void memcpy_async(void* shr, const void* gbl);

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_nop(void* shr, const void* gbl, bool predicate);

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_zero(void* shr, const void* gbl, bool predicate);

__device__ __forceinline__ void memcpy_async_fence();

template <int stages>
__device__ __forceinline__ void memcpy_async_wait();

template <int stages>
__device__ __forceinline__ void tail_complete_wait(int remaining_stages);
#endif

// Util for tracking pipeline buffers
// TODO: Evaluate whether this should also be guarded by ASYNC_COPY_AVAILABLE
template <int max>
class BufferTracker {
public:
    int current_state;

    __device__ __forceinline__ BufferTracker() : current_state(0) {}

    __device__ __forceinline__ int get()
    {
        int return_val = current_state++;
        current_state = (current_state == max ? 0 : current_state);
        return return_val;
    }
};

__device__ __forceinline__ uint32_t lane_id()
{
#ifdef PTX_AVAILABLE
    unsigned int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
#else
    return threadIdx.x & (warpSize - 1);  // Portable
#endif
}

/////////// Load Global ///////////
template <>
__device__ __forceinline__ void load_global<16>(void* dst, const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
                 : "l"(src));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<16>(void* dst, const void* src, bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
        : "l"(src), "r"((int)do_access));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
        data[0].z = 0;
        data[0].w = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<16, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
                 : "l"(src));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<16, LoadPolicy::CacheGlobal>(void* dst,
                                                                         const void* src,
                                                                         bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
        : "l"(src), "r"((int)do_access));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
        data[0].z = 0;
        data[0].w = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<16, LoadPolicy::CacheStreaming>(void* dst,
                                                                            const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
                 : "l"(src));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<16, LoadPolicy::CacheStreaming>(void* dst,
                                                                            const void* src,
                                                                            bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
        : "l"(src), "r"((int)do_access));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
        data[0].z = 0;
        data[0].w = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<8>(void* dst, const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x), "=r"(data[0].y)
                 : "l"(src));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<8>(void* dst, const void* src, bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.global.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y)
        : "l"(src), "r"((int)do_access));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<8, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x), "=r"(data[0].y)
                 : "l"(src));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<8, LoadPolicy::CacheGlobal>(void* dst,
                                                                        const void* src,
                                                                        bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y)
        : "l"(src), "r"((int)do_access));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<8, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x), "=r"(data[0].y)
                 : "l"(src));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<8, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src,
                                                                           bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y)
        : "l"(src), "r"((int)do_access));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<4>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.ca.u32 {%0}, [%1];\n" : "=r"(*data) : "l"(src));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<4>(void* dst, const void* src, bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.global.u32 {%0}, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "l"(src), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<4, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cg.u32 {%0}, [%1];\n" : "=r"(*data) : "l"(src));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<4, LoadPolicy::CacheGlobal>(void* dst,
                                                                        const void* src,
                                                                        bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.global.cg.u32 {%0}, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "l"(src), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<4, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cs.u32 {%0}, [%1];\n" : "=r"(*data) : "l"(src));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<4, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src,
                                                                           bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.global.cs.u32 {%0}, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "l"(src), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<2>(void* dst, const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.ca.u16 {%0}, [%1];\n" : "=h"(*data) : "l"(src));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<2>(void* dst, const void* src, bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.u16 %0, 0;\n"
        "\t@p ld.global.u16 {%0}, [%1];\n"
        "}\n"
        : "=h"(*data)
        : "l"(src), "r"((int)do_access));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<2, LoadPolicy::CacheGlobal>(void* dst, const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cg.u16 {%0}, [%1];\n" : "=h"(*data) : "l"(src));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<2, LoadPolicy::CacheGlobal>(void* dst,
                                                                        const void* src,
                                                                        bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.u16 %0, 0;\n"
        "\t@p ld.global.cg.u16 {%0}, [%1];\n"
        "}\n"
        : "=h"(*data)
        : "l"(src), "r"((int)do_access));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_global<2, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile("ld.global.cs.u16 {%0}, [%1];\n" : "=h"(*data) : "l"(src));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_global<2, LoadPolicy::CacheStreaming>(void* dst,
                                                                           const void* src,
                                                                           bool do_access)
{
    int16_t* data = reinterpret_cast<int16_t*>(dst);
#ifdef PTX_AVAILABLE
    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.u16 %0, 0;\n"
        "\t@p ld.global.cs.u16 {%0}, [%1];\n"
        "}\n"
        : "=h"(*data)
        : "l"(src), "r"((int)do_access));
#else
    const int16_t* src_cast = reinterpret_cast<const int16_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

/////////// Load Shared ///////////
namespace internal {

#ifdef PTX_AVAILABLE
__device__ __forceinline__ unsigned convert_to_shared(const void* ptr)
{
#if __CUDACC_VER_MAJOR__ >= 11
    // In CUDA 11 we have a builtin intrinsic
    return __cvta_generic_to_shared(ptr);
#else
    unsigned ret_val;
    asm volatile(
        "{\n"
        "\t.reg .u64 p1;\n"
        "\tcvta.to.shared.u64 p1, %1\n"
        "\tcvt.u32.u64 %0, p1;\n"
        "}\n"
        : "=r"(ret_val)
        : "l"(ptr));
    return ret_val;
#endif
}
#endif

}  // namespace internal

template <>
__device__ __forceinline__ void load_shared<16>(void* dst, const void* src)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
                 : "r"(src_shr));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_shared<16>(void* dst, const void* src, bool do_access)
{
    uint4* data = reinterpret_cast<uint4*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %5, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\tmov.b32 %2, 0;\n"
        "\tmov.b32 %3, 0;\n"
        "\t@p ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w)
        : "r"(src_shr), "r"((int)do_access));
#else
    const uint4* src_cast = reinterpret_cast<const uint4*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
        data[0].z = 0;
        data[0].w = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_shared<8>(void* dst, const void* src)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(data[0].x), "=r"(data[0].y)
                 : "r"(src_shr));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_shared<8>(void* dst, const void* src, bool do_access)
{
    uint2* data = reinterpret_cast<uint2*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %3, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\tmov.b32 %1, 0;\n"
        "\t@p ld.shared.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y)
        : "r"(src_shr), "r"((int)do_access));
#else
    const uint2* src_cast = reinterpret_cast<const uint2*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0].x = 0;
        data[0].y = 0;
    }
#endif
}

template <>
__device__ __forceinline__ void load_shared<4>(void* dst, const void* src)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    asm volatile("ld.shared.u32 {%0}, [%1];\n" : "=r"(*data) : "r"(src_shr));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    data[0] = src_cast[0];
#endif
}

template <>
__device__ __forceinline__ void load_shared<4>(void* dst, const void* src, bool do_access)
{
    int32_t* data = reinterpret_cast<int32_t*>(dst);
#ifdef PTX_AVAILABLE
    unsigned src_shr = internal::convert_to_shared(src);

    asm volatile(
        "{\n"
        "\t.reg .pred p;\n"
        "\tsetp.ne.b32 p, %2, 0;\n"
        "\tmov.b32 %0, 0;\n"
        "\t@p ld.shared.u32 %0, [%1];\n"
        "}\n"
        : "=r"(data[0])
        : "r"(src_shr), "r"((int)do_access));
#else
    const int32_t* src_cast = reinterpret_cast<const int32_t*>(src);
    if (do_access) {
        data[0] = src_cast[0];
    } else {
        data[0] = 0;
    }
#endif
}

/////////// Store Global ///////////

template <>
__device__ __forceinline__ void store_global<16>(void* dst, const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w)
                 : "memory");
#else
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<16, StorePolicy::CacheGlobal>(void* dst,
                                                                           const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w)
                 : "memory");
#else
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<16, StorePolicy::CacheStreaming>(void* dst,
                                                                              const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w)
                 : "memory");
#else
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<8>(void* dst, const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "l"(dst), "r"(data[0].x), "r"(data[0].y));
#else
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<8, StorePolicy::CacheGlobal>(void* dst,
                                                                          const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.cg.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "l"(dst), "r"(data[0].x), "r"(data[0].y));
#else
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<8, StorePolicy::CacheStreaming>(void* dst,
                                                                             const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "l"(dst), "r"(data[0].x), "r"(data[0].y));
#else
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<4>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.wb.u32 [%0], %1;\n" : : "l"(dst), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<4, StorePolicy::CacheGlobal>(void* dst,
                                                                          const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.cg.u32 [%0], %1;\n" : : "l"(dst), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<4, StorePolicy::CacheStreaming>(void* dst,
                                                                             const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    asm volatile("st.global.cs.u32 [%0], %1;\n" : : "l"(dst), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_global<2>(void* dst, const void* src)
{
    const int16_t* data = reinterpret_cast<const int16_t*>(src);
    
    int16_t* dst_cast = reinterpret_cast<int16_t*>(dst);
    dst_cast[0] = data[0];
}

template <>
__device__ __forceinline__ void store_global<2, StorePolicy::CacheGlobal>(void* dst,
                                                                          const void* src)
{
    const int16_t* data = reinterpret_cast<const int16_t*>(src);
    
    int16_t* dst_cast = reinterpret_cast<int16_t*>(dst);
    dst_cast[0] = data[0];
}

template <>
__device__ __forceinline__ void store_global<2, StorePolicy::CacheStreaming>(void* dst,
                                                                             const void* src)
{
    const int16_t* data = reinterpret_cast<const int16_t*>(src);
    
    int16_t* dst_cast = reinterpret_cast<int16_t*>(dst);
    dst_cast[0] = data[0];
}
/////////// Store Shared ///////////

template <>
__device__ __forceinline__ void store_shared<16>(void* dst, const void* src)
{
    const uint4* data = reinterpret_cast<const uint4*>(src);
#ifdef PTX_AVAILABLE
    unsigned dst_int = internal::convert_to_shared(dst);

    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "r"(dst_int), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w));
#else
    uint4* dst_cast = reinterpret_cast<uint4*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_shared<8>(void* dst, const void* src)
{
    const uint2* data = reinterpret_cast<const uint2*>(src);
#ifdef PTX_AVAILABLE
    unsigned dst_int = internal::convert_to_shared(dst);

    asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                 :
                 : "r"(dst_int), "r"(data[0].x), "r"(data[0].y));
#else
    uint2* dst_cast = reinterpret_cast<uint2*>(dst);
    dst_cast[0] = data[0];
#endif
}

template <>
__device__ __forceinline__ void store_shared<4>(void* dst, const void* src)
{
    const int32_t* data = reinterpret_cast<const int32_t*>(src);
#ifdef PTX_AVAILABLE
    unsigned dst_int = internal::convert_to_shared(dst);

    asm volatile("st.shared.u32 [%0], %1;\n" : : "r"(dst_int), "r"(*data));
#else
    int32_t* dst_cast = reinterpret_cast<int32_t*>(dst);
    dst_cast[0] = data[0];
#endif
}

/////////// Asynchronous Memory Copy ///////////

#ifdef ASYNC_COPY_AVAILABLE
template <int AccessSize>
__device__ __forceinline__ void memcpy_async(void* shr, const void* gbl)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                 :
                 : "r"(shr_int), "l"(gbl), "n"(AccessSize));
}

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_nop(void* shr, const void* gbl, bool predicate)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n"
        :
        : "r"((int)predicate), "r"(shr_int), "l"(gbl), "n"(AccessSize));
}

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_zero(void* shr, const void* gbl, bool predicate)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (predicate ? AccessSize : 0);

    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                 :
                 : "r"(shr_int), "l"(gbl), "n"(AccessSize), "r"(bytes_to_copy));
}

template <int AccessSize>
__device__ __forceinline__ void memcpy_async_zero_nop(void* shr,
                                                      const void* gbl,
                                                      bool zero_predicate,
                                                      bool nop_predicate)
{
    static_assert((AccessSize == 4 || AccessSize == 8 || AccessSize == 16));
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (zero_predicate ? AccessSize : 0);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
        "}\n"
        :
        : "r"((int)nop_predicate), "r"(shr_int), "l"(gbl), "n"(AccessSize), "r"(bytes_to_copy));
}

// Cache global variants. Separate interface to require deliberate use of them.
__device__ __forceinline__ void memcpy_async_cg(void* shr, const void* gbl)
{
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(shr_int), "l"(gbl));
}

__device__ __forceinline__ void memcpy_async_nop_cg(void* shr, const void* gbl, bool predicate)
{
    unsigned shr_int = internal::convert_to_shared(shr);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], 16;\n"
        "}\n"
        :
        : "r"((int)predicate), "r"(shr_int), "l"(gbl));
}

__device__ __forceinline__ void memcpy_async_zero_cg(void* shr, const void* gbl, bool predicate)
{
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (predicate ? 16 : 0);

    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(shr_int), "l"(gbl), "r"(bytes_to_copy));
}

__device__ __forceinline__ void memcpy_async_zero_nop_cg(void* shr,
                                                         const void* gbl,
                                                         bool zero_predicate,
                                                         bool nop_predicate)
{
    unsigned shr_int = internal::convert_to_shared(shr);
    int bytes_to_copy = (zero_predicate ? 16 : 0);

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], 16, %3;\n"
        "}\n"
        :
        : "r"((int)nop_predicate), "r"(shr_int), "l"(gbl), "r"(bytes_to_copy));
}

__device__ __forceinline__ void memcpy_async_fence() { asm volatile("cp.async.commit_group;\n"); }

template <int stages>
__device__ __forceinline__ void memcpy_async_wait()
{
    static_assert(stages <= 8);

    asm volatile("cp.async.wait_group %0;\n" : : "n"(stages));
}

// TODO: The tail complete should be a known compile time artifact, should try and induce this
// without all of the branches from the call-site. This is a hacky solution.
template <>
__device__ __forceinline__ void tail_complete_wait<1>(int remaining_stages)
{
    if (remaining_stages == 0) memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<2>(int remaining_stages)
{
    if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<3>(int remaining_stages)
{
    if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<4>(int remaining_stages)
{
    if (remaining_stages == 3)
        memcpy_async_wait<3>();
    else if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<5>(int remaining_stages)
{
    if (remaining_stages == 4)
        memcpy_async_wait<4>();
    else if (remaining_stages == 3)
        memcpy_async_wait<3>();
    else if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}

template <>
__device__ __forceinline__ void tail_complete_wait<6>(int remaining_stages)
{
    if (remaining_stages == 5)
        memcpy_async_wait<5>();
    else if (remaining_stages == 4)
        memcpy_async_wait<4>();
    else if (remaining_stages == 3)
        memcpy_async_wait<3>();
    else if (remaining_stages == 2)
        memcpy_async_wait<2>();
    else if (remaining_stages == 1)
        memcpy_async_wait<1>();
    else if (remaining_stages == 0)
        memcpy_async_wait<0>();
}
#endif

}  // namespace mem_access
