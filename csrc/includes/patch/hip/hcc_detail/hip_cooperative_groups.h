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
 *  @file  hcc_detail/hip_cooperative_groups.h
 *
 *  @brief Device side implementation of `Cooperative Group` feature.
 *
 *  Defines new types and device API wrappers related to `Cooperative Group`
 *  feature, which the programmer can directly use in his kernel(s) in order to
 *  make use of this feature.
 */
#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_H

//#if __cplusplus
#if __cplusplus && defined(__clang__) && defined(__HIP__)
#include <hip/hcc_detail/hip_cooperative_groups_helper.h>
#if ROCM_VERSION_MAJOR < 5 and ROCM_VERSION_MINOR < 4
#include <hip/hcc_detail/device_functions.h>
#endif
namespace cooperative_groups {

/** \brief The base type of all cooperative group types
 *
 *  \details Holds the key properties of a constructed cooperative group type
 *           object, like the group type, its size, etc
 */
/*
class thread_group {
 protected:
  uint32_t _type; // thread_group type
  uint32_t _size; // total number of threads in the tread_group
  uint64_t _mask; // Lanemask for coalesced and tiled partitioned group types,
                  // LSB represents lane 0, and MSB represents lane 63

  // Construct a thread group, and set thread group type and other essential
  // thread group properties. This generic thread group is directly constructed
  // only when the group is supposed to contain only the calling the thread
  // (throurh the API - `this_thread()`), and in all other cases, this thread
  // group object is a sub-object of some other derived thread group object
  __CG_QUALIFIER__ thread_group(internal::group_type type, uint32_t size,
                                uint64_t mask = (uint64_t)0) {
    _type = type;
    _size = size;
    _mask = mask;
  }

 public:
  // Total number of threads in the thread group, and this serves the purpose
  // for all derived cooperative group types since their `size` is directly
  // saved during the construction
  __CG_QUALIFIER__ uint32_t size() const {
    return _size;
  }
  // Rank of the calling thread within [0, size())
  __CG_QUALIFIER__ uint32_t thread_rank() const;
  // Is this cooperative group type valid?
  __CG_QUALIFIER__ bool is_valid() const;
  // synchronize the threads in the thread group
  __CG_QUALIFIER__ void sync() const;
};
*/

class thread_group {
protected:
    bool _tiled_partition;  // this_thread_block() constructor sets to false
    uint32_t _size;         // this_thread_block() constructor sets to size()
    uint32_t local_rank;    // this_thread_block() constructor sets to thread_rank()
    uint32_t _mask;
    uint32_t _type;

public:
    __CG_QUALIFIER__ thread_group(internal::group_type type,
                                  uint32_t group_size,
                                  uint64_t mask = (uint64_t)0)
    {
        _type = type;
        _size = group_size;
        _mask = mask;
        local_rank = internal::workgroup::thread_rank();
    }

    __CG_QUALIFIER__ void tiled_partition(const thread_group& parent, unsigned int tile_size)
    {
        if ((ceil(log2(tile_size)) == floor(log2(tile_size))) || tile_size == 0 || tile_size > 64 ||
            parent.size() < tile_size)
            _tiled_partition = false;
        // xxx : abort
        _tiled_partition = true;
        _size = tile_size;
        local_rank = parent.thread_rank() % tile_size;
    }
    __CG_QUALIFIER__ void sync() const;
    __CG_QUALIFIER__ uint32_t size() const { return _size; }
    __CG_QUALIFIER__ uint32_t thread_rank() const;
    __CG_QUALIFIER__ float shfl_down(float var, unsigned int delta) const
    {
        return (__shfl_down(var, delta, _size));
    }
    __CG_QUALIFIER__ float shfl_xor(float var, int mask) const
    {
        return (__shfl_xor(var, mask, _size));
    }
    __CG_QUALIFIER__ float shfl(float var, unsigned int src_lane) const
    {
        return (__shfl(var, src_lane, _size));
    }
    __CG_QUALIFIER__ bool is_valid() const;
};

/** \brief The multi-grid cooperative group type
 *
 *  \details Represents an inter-device cooperative group type where the
 *           participating threads within the group spans across multiple
 *           devices, running the (same) kernel on these devices
 */
class multi_grid_group : public thread_group {
    // Only these friend functions are allowed to construct an object of this class
    // and access its resources
    friend __CG_QUALIFIER__ multi_grid_group this_multi_grid();

protected:
    // Construct mutli-grid thread group (through the API this_multi_grid())
    explicit __CG_QUALIFIER__ multi_grid_group(uint32_t size)
        : thread_group(internal::cg_multi_grid, size)
    {
    }

public:
    // Number of invocations participating in this multi-grid group. In other
    // words, the number of GPUs
    __CG_QUALIFIER__ uint32_t num_grids() { return internal::multi_grid::num_grids(); }
    // Rank of this invocation. In other words, an ID number within the range
    // [0, num_grids()) of the GPU, this kernel is running on
    __CG_QUALIFIER__ uint32_t grid_rank() { return internal::multi_grid::grid_rank(); }
    __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::multi_grid::thread_rank(); }
    __CG_QUALIFIER__ bool is_valid() const { return internal::multi_grid::is_valid(); }
    __CG_QUALIFIER__ void sync() const { internal::multi_grid::sync(); }
};

/** \brief User exposed API interface to construct multi-grid cooperative
 *         group type object - `multi_grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 */
__CG_QUALIFIER__ multi_grid_group this_multi_grid()
{
    return multi_grid_group(internal::multi_grid::size());
}

/** \brief The grid cooperative group type
 *
 *  \details Represents an inter-workgroup cooperative group type where the
 *           participating threads within the group spans across multiple
 *           workgroups running the (same) kernel on the same device
 */
class grid_group : public thread_group {
    // Only these friend functions are allowed to construct an object of this class
    // and access its resources
    friend __CG_QUALIFIER__ grid_group this_grid();

protected:
    // Construct grid thread group (through the API this_grid())
    explicit __CG_QUALIFIER__ grid_group(uint32_t size) : thread_group(internal::cg_grid, size) {}

public:
    __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::grid::thread_rank(); }
    __CG_QUALIFIER__ bool is_valid() const { return internal::grid::is_valid(); }
    __CG_QUALIFIER__ void sync() const { internal::grid::sync(); }
};

/** \brief User exposed API interface to construct grid cooperative group type
 *         object - `grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 */
__CG_QUALIFIER__ grid_group this_grid() { return grid_group(internal::grid::size()); }

/** \brief The workgroup (thread-block in CUDA terminology) cooperative group
 *         type
 *
 *  \details Represents an intra-workgroup cooperative group type where the
 *           participating threads within the group are exctly the same threads
 *           which are participated in the currently executing `workgroup`
 */
class thread_block : public thread_group {
    // Only these friend functions are allowed to construct an object of this
    // class and access its resources
    friend __CG_QUALIFIER__ thread_block this_thread_block();

protected:
    // Construct a workgroup thread group (through the API this_thread_block())
    explicit __CG_QUALIFIER__ thread_block(uint32_t size)
        : thread_group(internal::cg_workgroup, size)
    {
    }

public:
    // 3-dimensional block index within the grid
    __CG_QUALIFIER__ dim3 group_index() { return internal::workgroup::group_index(); }
    // 3-dimensional thread index within the block
    __CG_QUALIFIER__ dim3 thread_index() { return internal::workgroup::thread_index(); }
    __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::workgroup::thread_rank(); }
    __CG_QUALIFIER__ bool is_valid() const { return internal::workgroup::is_valid(); }
    __CG_QUALIFIER__ void sync() const { internal::workgroup::sync(); }
};

/** \brief User exposed API interface to construct workgroup cooperative
 *         group type object - `thread_block`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `thread_block`. Instead, he should construct it through this API
 *           function
 */
__CG_QUALIFIER__ thread_block this_thread_block()
{
    return thread_block(internal::workgroup::size());
}

/**
 *  Implementation of all publicly exposed base class APIs
 */
__CG_QUALIFIER__ uint32_t thread_group::thread_rank() const
{
    switch (this->_type) {
        case internal::cg_multi_grid: {
            return (static_cast<const multi_grid_group*>(this)->thread_rank());
        }
        case internal::cg_grid: {
            return (static_cast<const grid_group*>(this)->thread_rank());
        }
        case internal::cg_workgroup: {
            return (static_cast<const thread_block*>(this)->thread_rank());
        }
        case internal::cg_coalesced_tile: {
            return local_rank;
        }
        default: {
            assert(false && "invalid cooperative group type");
            return -1;
        }
    }
}

__CG_QUALIFIER__ bool thread_group::is_valid() const
{
    switch (this->_type) {
        case internal::cg_multi_grid: {
            return (static_cast<const multi_grid_group*>(this)->is_valid());
        }
        case internal::cg_grid: {
            return (static_cast<const grid_group*>(this)->is_valid());
        }
        case internal::cg_workgroup: {
            return (static_cast<const thread_block*>(this)->is_valid());
        }
        case internal::cg_coalesced_tile: {
            return _tiled_partition;
        }
        default: {
            assert(false && "invalid cooperative group type");
            return false;
        }
    }
}

__CG_QUALIFIER__ void thread_group::sync() const
{
    switch (this->_type) {
        case internal::cg_multi_grid: {
            static_cast<const multi_grid_group*>(this)->sync();
            break;
        }
        case internal::cg_grid: {
            static_cast<const grid_group*>(this)->sync();
            break;
        }
        case internal::cg_workgroup: {
            static_cast<const thread_block*>(this)->sync();
            break;
        }
        case internal::cg_coalesced_tile: {
            if (!_tiled_partition)  // If in a tiled partition, this is a no-op
                __syncthreads();
            break;
        }
        default: {
            assert(false && "invalid cooperative group type");
        }
    }
}

/**
 *  Implementation of publicly exposed `wrapper` APIs on top of basic cooperative
 *  group type APIs
 */
template <class CGTy>
__CG_QUALIFIER__ uint32_t group_size(CGTy const& g)
{
    return g.size();
}

template <class CGTy>
__CG_QUALIFIER__ uint32_t thread_rank(CGTy const& g)
{
    return g.thread_rank();
}

template <class CGTy>
__CG_QUALIFIER__ bool is_valid(CGTy const& g)
{
    return g.is_valid();
}

template <class CGTy>
__CG_QUALIFIER__ void sync(CGTy const& g)
{
    g.sync();
}

}  // namespace cooperative_groups

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COOPERATIVE_GROUPS_H
