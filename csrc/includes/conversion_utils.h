// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"

#include <stdint.h>

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif

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
template <>
DS_D_INLINE double to(int64_t val)
{
    return __ll2double_rn(val);
}
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
#ifdef __HIP_PLATFORM_HCC__
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
