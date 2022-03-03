/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  hcc_detail/hip_cooperative_groups_helper.h
 *
 *  @brief Device side implementation of cooperative group feature.
 *
 *  Defines helper constructs and APIs which aid the types and device API
 *  wrappers defined within `hcc_detail/hip_cooperative_groups.h`.
 */
#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H

#if __cplusplus

#if ROCM_VERSION_MAJOR < 5 and ROCM_VERSION_MINOR < 4
#include <hip/hcc_detail/device_functions.h>
#include <hip/hcc_detail/hip_runtime_api.h>
#else
#include <hip/hcc_detail/amd_device_functions.h>
#endif

#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif

#if !defined(__CG_QUALIFIER__)
#define __CG_QUALIFIER__ __device__ __forceinline__
#endif

#if !defined(__CG_STATIC_QUALIFIER__)
#define __CG_STATIC_QUALIFIER__ __device__ static __forceinline__
#endif

#if !defined(WAVEFRONT_SIZE)
#define WAVEFRONT_SIZE 64
#endif

namespace cooperative_groups {

namespace internal {

/** \brief Enums representing different cooperative group types
 */
typedef enum { cg_invalid, cg_multi_grid, cg_grid, cg_workgroup, cg_coalesced_tile } group_type;

/**
 *  Functionalities related to multi-grid cooperative group type
 */
namespace multi_grid {

__CG_STATIC_QUALIFIER__ uint32_t num_grids() { return (uint32_t)__ockl_multi_grid_num_grids(); }

__CG_STATIC_QUALIFIER__ uint32_t grid_rank() { return (uint32_t)__ockl_multi_grid_grid_rank(); }

__CG_STATIC_QUALIFIER__ uint32_t size() { return (uint32_t)__ockl_multi_grid_size(); }

__CG_STATIC_QUALIFIER__ uint32_t thread_rank() { return (uint32_t)__ockl_multi_grid_thread_rank(); }

__CG_STATIC_QUALIFIER__ bool is_valid() { return (bool)__ockl_multi_grid_is_valid(); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_multi_grid_sync(); }

}  // namespace multi_grid

/**
 *  Functionalities related to grid cooperative group type
 */
namespace grid {

__CG_STATIC_QUALIFIER__ uint32_t size()
{
    return (uint32_t)((hipBlockDim_z * hipGridDim_z) * (hipBlockDim_y * hipGridDim_y) *
                      (hipBlockDim_x * hipGridDim_x));
}

__CG_STATIC_QUALIFIER__ uint32_t thread_rank()
{
    // Compute global id of the workgroup to which the current thread belongs to
    uint32_t blkIdx = (uint32_t)((hipBlockIdx_z * hipGridDim_y * hipGridDim_x) +
                                 (hipBlockIdx_y * hipGridDim_x) + (hipBlockIdx_x));

    // Compute total number of threads being passed to reach current workgroup
    // within grid
    uint32_t num_threads_till_current_workgroup =
        (uint32_t)(blkIdx * (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z));

    // Compute thread local rank within current workgroup
    uint32_t local_thread_rank = (uint32_t)((hipThreadIdx_z * hipBlockDim_y * hipBlockDim_x) +
                                            (hipThreadIdx_y * hipBlockDim_x) + (hipThreadIdx_x));

    return (num_threads_till_current_workgroup + local_thread_rank);
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return (bool)__ockl_grid_is_valid(); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_grid_sync(); }

}  // namespace grid

/**
 *  Functionalities related to `workgroup` (thread_block in CUDA terminology)
 *  cooperative group type
 */
namespace workgroup {

__CG_STATIC_QUALIFIER__ dim3 group_index()
{
    return (dim3((uint32_t)hipBlockIdx_x, (uint32_t)hipBlockIdx_y, (uint32_t)hipBlockIdx_z));
}

__CG_STATIC_QUALIFIER__ dim3 thread_index()
{
    return (dim3((uint32_t)hipThreadIdx_x, (uint32_t)hipThreadIdx_y, (uint32_t)hipThreadIdx_z));
}

__CG_STATIC_QUALIFIER__ uint32_t size()
{
    return ((uint32_t)(hipBlockDim_x * hipBlockDim_y * hipBlockDim_z));
}

__CG_STATIC_QUALIFIER__ uint32_t thread_rank()
{
    return ((uint32_t)((hipThreadIdx_z * hipBlockDim_y * hipBlockDim_x) +
                       (hipThreadIdx_y * hipBlockDim_x) + (hipThreadIdx_x)));
}

__CG_STATIC_QUALIFIER__ bool is_valid()
{
    // TODO(mahesha) any functionality need to be added here? I believe not
    return true;
}

__CG_STATIC_QUALIFIER__ void sync() { __syncthreads(); }

}  // namespace workgroup

}  // namespace internal

}  // namespace cooperative_groups

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
