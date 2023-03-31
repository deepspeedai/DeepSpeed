// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

namespace reduce {

enum class ROpType {
    // Addition
    Add,

    // Maximum reduction
    Max,

    // Minimum reduction
    Min,
};

constexpr int max_threads = 1024;
constexpr int max_warps = max_threads / hw_warp_size;

/*
High level API. The API takes in a set of operations and variables
and performs that reduction operation on that variable. The reductions
of each of the arguments are completely independent of each other (
i.e., the val1-op1 combination has no impact on val2-op2).

Example usage:
``` cpp
float max_val;
float min_val;
reduce::block<rop::Max, rop::Min>(tb, warp, max_val, min_val);
```

TODO(cmikeh2): In theory, we might be able to do this sequentially with
device functions and rely on the assembler correctly behaving. My initial
instinct is this won't work, but if it does it would reduce implementation
cost significantly.

TODO(cmikeh2): We need to support sub-block reductions. The warp intrinsic
currently supports this (more incidentally than anything else). It is not
uncommon in something like softmax or a fused attention kernel to map multiple
reductions to a thread block, but each reduction itself is only scoped
to part of the threads (i.e block size = 512, 128 threads per reduction).
*/
template <ROpType Op, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp, float& val);

template <ROpType Op1, ROpType Op2, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound = max_warps>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4);

/*
The partitioned block is a special case of the above where in the warps of a threadblock are
partitioned into separate independent reductions. For example, I might have an 8 warp thread block
in which each pair of warps is processing an independent piece of data. I would then reduce that
data with the something like the following:
``` cpp
float max_val;
reduce::partitioned_block<rop::Max, 2>(tb, warp, max_val);
```
After which, each pair of warps would have coherent data with each other. Note, this API will not
provide correct results if the number of warps per partition is not a power of 2.
*/
template <ROpType Op, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val);

template <ROpType Op1, ROpType Op2, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val1,
                                   float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3,
                                   float& val4);

/*
Single element reduction primitives. Used inside serial collection
loops.

Example usage:
using rop = reduce::OpType;
float min = init<rop::Min>();
for (int i = 0; i < 4; i++) {
    min = reduce::element<rop::Min>(min, data[i]);
}
*/

template <ROpType Op, typename T>
DS_D_INLINE T element(const T lhs, const T rhs);

template <ROpType OType, typename T = float>
DS_D_INLINE T init();

/********************** Internal reduction APIs **********************/

/*
Single element "reductions". TODO(cmikeh2): this sort of "op" concept
should be refactored into its own implementation at some point. This interface
may be easily expanded for new types/operations, but the typical reductions
we need are covered with min/max/add on float.

NOTE: there is no mean reduction because that relies on knowledge of how
many values were already reduced into each scalar. Implementing this on top
of reduce should be straightforward (can just wrap the sum reduction) and
would be a good extension of the header.
*/

/* Float element reduce implementations */
template <>
DS_D_INLINE float element<ROpType::Add>(const float lhs, const float rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE float element<ROpType::Max>(const float lhs, const float rhs)
{
    return fmaxf(lhs, rhs);
}

template <>
DS_D_INLINE float element<ROpType::Min>(const float lhs, const float rhs)
{
    return fminf(lhs, rhs);
}

/* __half element reduce implementation */
template <>
DS_D_INLINE __half element<ROpType::Add>(const __half lhs, const __half rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE __half element<ROpType::Max>(const __half lhs, const __half rhs)
{
#if __CUDA_ARCH__ >= 800
    // Intrinsic limited to Ampere + newer
    return __hmax(lhs, rhs);
#else
    return (lhs > rhs) ? lhs : rhs;
#endif
}

template <>
DS_D_INLINE __half element<ROpType::Min>(const __half lhs, const __half rhs)
{
#if __CUDA_ARCH__ >= 800
    // Intrinsic limited to Ampere + newer
    return __hmin(lhs, rhs);
#else
    return (lhs < rhs) ? lhs : rhs;
#endif
}

/* __half2 element reduce implementation */
template <>
DS_D_INLINE __half2 element<ROpType::Add>(const __half2 lhs, const __half2 rhs)
{
    return lhs + rhs;
}

template <>
DS_D_INLINE __half2 element<ROpType::Max>(const __half2 lhs, const __half2 rhs)
{
#if __CUDA_ARCH__ >= 800
    return __hmax2(lhs, rhs);
#else
    __half2 ret_val;
    ret_val.x = (lhs.x > rhs.x) ? lhs.x : rhs.x;
    ret_val.y = (lhs.y > rhs.y) ? lhs.y : rhs.y;
    return ret_val;
#endif
}

template <>
DS_D_INLINE __half2 element<ROpType::Min>(const __half2 lhs, const __half2 rhs)
{
#if __CUDA_ARCH__ >= 800
    return __hmin2(lhs, rhs);
#else
    __half2 ret_val;
    ret_val.x = (lhs.x < rhs.x) ? lhs.x : rhs.x;
    ret_val.y = (lhs.y < rhs.y) ? lhs.y : rhs.y;
    return ret_val;
#endif
}

/*
Reduction initialization primitives
*/
template <>
DS_D_INLINE float init<ROpType::Add>()
{
    return 0.0f;
}

template <>
DS_D_INLINE float init<ROpType::Min>()
{
    // Positive infinity
    return INFINITY;
}

template <>
DS_D_INLINE float init<ROpType::Max>()
{
    // Negative infinity
    return -INFINITY;
}

template <>
DS_D_INLINE __half init<ROpType::Add>()
{
    constexpr __half_raw zero = {0x0000};
    return __half(zero);
}

template <>
DS_D_INLINE __half init<ROpType::Min>()
{
    constexpr __half_raw inf = {0x7C00};
    return __half(inf);
}

template <>
DS_D_INLINE __half init<ROpType::Max>()
{
    constexpr __half_raw neg_inf = {0xFC00};
    return __half(neg_inf);
}

template <>
DS_D_INLINE __half2 init<ROpType::Add>()
{
    constexpr __half2_raw zero = {0x0000, 0x0000};
    return __half2(zero);
}

template <>
DS_D_INLINE __half2 init<ROpType::Min>()
{
    constexpr __half2_raw inf = {0x7C00, 0x7C00};
    return __half2(inf);
}

template <>
DS_D_INLINE __half2 init<ROpType::Max>()
{
    constexpr __half2_raw neg_inf = {0xFC00, 0xFC00};
    return __half2(neg_inf);
}

template <ROpType Op, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op, T>();
}

template <ROpType Op1, ROpType Op2, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
}

template <ROpType Op1, ROpType Op2, ROpType Op3, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
    data[2] = init<Op3, T>();
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, typename T>
DS_D_INLINE void init(T* data)
{
    data[0] = init<Op1, T>();
    data[1] = init<Op2, T>();
    data[2] = init<Op3, T>();
    data[3] = init<Op4, T>();
}

/*
Warp reduction primitives

`reduction_width` is an unsafe template parameter, that is that
when using `reduction_width` < hw_warp_size the warp is partitioned
into `hw_warp_size` / `reduction_width` groups of partial sums.

If someone can figure out how to use variadic templates in a reasonable way
here (fold is C++17 only and I don't think helps and recursion feels like
huge overkill that harms readability) that would be wonderful.
*/

template <ROpType Op, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op>(data[0], warp.shfl_xor(data[0], i));
    }
}

template <ROpType Op1, ROpType Op2, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shfl_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shfl_xor(data[1], i));
    }
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shfl_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shfl_xor(data[1], i));
        data[2] = element<Op3>(data[2], warp.shfl_xor(data[2], i));
    }
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int reduce_width = hw_warp_size>
DS_D_INLINE void _warp(cg::thread_block_tile<hw_warp_size>& warp, float* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op1>(data[0], warp.shfl_xor(data[0], i));
        data[1] = element<Op2>(data[1], warp.shfl_xor(data[1], i));
        data[2] = element<Op3>(data[2], warp.shfl_xor(data[2], i));
        data[3] = element<Op4>(data[3], warp.shfl_xor(data[3], i));
    }
}

/*
Implementation for primary block reduction that serves both `block` and
`partitioned_block`.

`local_warp_rank` refers to the warp's location within the partition, so
for an unpartitioned threadblock this will be equivalent to
`warp_arg.meta_group_rank()`.

Similarly, the warp offset is the `local_warp_rank` of the warp with the
lowest rank in the partition. In the case of an 8 warp block with a
4 warp reduction, this would map to [0, 0, 0, 0, 4, 4, 4, 4].

Partition size is the number of warps per partition (equal to the thread
block in the default case). This enables us to only perform the warp reduction
when able to.
*/
template <int total_warps, ROpType... Ops>
DS_D_INLINE void _block(cg::thread_block& tb,
                        cg::thread_block_tile<hw_warp_size>& warp_arg,
                        float* data,
                        int warp_offset)
{
    constexpr int elems = sizeof...(Ops);
    // Separated for now in case this no longer is true
    constexpr int bytes = sizeof(float);
    // Unused when `partition_size == 1` or total_warps == 1
    __shared__ float reduce_buffer[max_warps * elems];

    // Always perform warp-scope reduction
    _warp<Ops...>(warp_arg, data);

    // If max_warps == 1 let's skip the runtime check
    if (warp_arg.meta_group_size() > 1 && total_warps != 1) {
        if (warp_arg.thread_rank() == 0) {
#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::store_shared<bytes>(
                    reduce_buffer + elems * warp_arg.meta_group_rank() + i, data + i);
            }
        }

        // Synchronization inside block-uniform conditional is safe
        tb.sync();

        if (warp_arg.meta_group_rank() == 0) {
            if (warp_arg.thread_rank() < warp_arg.meta_group_size()) {
#pragma unroll
                for (int i = 0; i < elems; i++) {
                    mem_access::load_shared<bytes>(
                        data + i, reduce_buffer + elems * warp_arg.thread_rank() + i);
                }
            } else {
                init<Ops...>(data);
            }

            _warp<Ops..., total_warps>(warp_arg, data);

#pragma unroll
            for (int i = 0; i < elems; i++) {
                mem_access::store_shared<bytes>(reduce_buffer + elems * warp_arg.thread_rank() + i,
                                                data + i);
            }
        }

        // Synchronization inside block-uniform conditional is safe
        tb.sync();

#pragma unroll
        for (int i = 0; i < elems; i++) {
            mem_access::load_shared<bytes>(data + i,
                                           reduce_buffer + warp_arg.meta_group_rank() * elems + i);
        }
    }
}

/*
Main API implementations. For the most part, they just convert the individual
variables into arrays, which makes working with them easier with a single
implementation. In theory, we could use the `_block` implementation as another
option, but the nature of using a pointer is a little less safe and this allows
us to obfuscate the details of the partitioned implementation.
*/
template <ROpType Op, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp, float& val)
{
    _block<warp_bound, Op>(tb, warp, &val, 0);
}

template <ROpType Op1, ROpType Op2, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2)
{
    float data[2] = {val1, val2};
    _block<warp_bound, Op1, Op2>(tb, warp, data, 0);
    val1 = data[0];
    val2 = data[1];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3)
{
    float data[3] = {val1, val2, val3};
    _block<warp_bound, Op1, Op2, Op3>(tb, warp, data, 0);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound>
DS_D_INLINE void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4)
{
    float data[4] = {val1, val2, val3, val4};
    _block<warp_bound, Op1, Op2, Op3, Op4>(tb, warp, data, 0);
    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
    val4 = data[3];
}

/*
Note: for the partitioned blocks, the implementation does not support non-power of 2 blocks in order
to shorten block scale reduction length.
*/
template <ROpType Op, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val)
{
    if (num_threads <= hw_warp_size) {
        _warp<Op, num_threads>(warp, &val);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.meta_group_rank() & ~(num_warps - 1);
        _block<num_warps, Op>(tb, warp, &val, warp_offset);
    }
}

template <ROpType Op1, ROpType Op2, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val1,
                                   float& val2)
{
    float data[2] = {val1, val2};

    if (num_threads <= hw_warp_size) {
        _warp<Op1, Op2, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.meta_group_rank() & ~(num_warps - 1);
        _block<num_warps, Op1, Op2>(tb, warp, data, warp_offset);
    }

    val1 = data[0];
    val2 = data[1];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3)
{
    float data[3] = {val1, val2, val3};

    if (num_threads <= hw_warp_size) {
        _warp<Op1, Op2, Op3, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.meta_group_rank() & ~(num_warps - 1);
        _block<num_warps, Op1, Op2, Op3>(tb, warp, data, warp_offset);
    }

    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
}

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int num_threads>
DS_D_INLINE void partitioned_block(cg::thread_block& tb,
                                   cg::thread_block_tile<hw_warp_size>& warp,
                                   float& val1,
                                   float& val2,
                                   float& val3,
                                   float& val4)
{
    float data[4] = {val1, val2, val3, val4};

    if (num_threads <= hw_warp_size) {
        _warp<Op1, Op2, Op3, Op4, num_threads>(warp, data);
    } else {
        constexpr int num_warps = num_threads / hw_warp_size;
        const int warp_offset = warp.meta_group_rank() & ~(num_warps - 1);
        _block<num_warps, Op1, Op2, Op3, Op4>(tb, warp, data, warp_offset);
    }

    val1 = data[0];
    val2 = data[1];
    val3 = data[2];
    val4 = data[3];
}

}  // namespace reduce
