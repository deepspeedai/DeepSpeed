// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <sycl/sycl.hpp>

inline void has_capability_or_fail(const sycl::device& dev,
                                   const std::initializer_list<sycl::aspect>& props)
{
    for (const auto& it : props) {
        if (dev.has(it)) continue;
        switch (it) {
            case sycl::aspect::fp64:
                throw std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() + "' device");
                break;
            case sycl::aspect::fp16:
                throw std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() + "' device");
                break;
            default:
#define __SYCL_ASPECT(ASPECT, ID) \
    case sycl::aspect::ASPECT: return #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect AspectNum) -> std::string {
                    switch (AspectNum) {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
                        default: return "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                throw std::runtime_error("'" + getAspectNameStr(it) + "' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() + "' device");
        }
        break;
    }
}

void param_update_kernel(const float* input, sycl::half* output, int size)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if (id < size) { output[id] = (sycl::half)input[id]; }
}

void launch_param_update(const float* input, sycl::half* output, int size, sycl::queue* stream)
{
    int threads = 1024;

    sycl::range<3> grid_dim(1, 1, (size - 1) / threads + 1);
    sycl::range<3> block_dim(1, 1, threads);

    {
        has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(grid_dim * block_dim, block_dim),
            [=](sycl::nd_item<3> item_ct1) { param_update_kernel(input, output, size); });
    }
}

void param_update_kernel_half(const float* input, sycl::half* output, int size)
{
    auto item_ct1 = sycl::ext::oneapi::experimental::this_nd_item<3>();
    int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    sycl::half2* output_cast = reinterpret_cast<sycl::half2*>(output);
    if (id < size) {
        float input_f = input[id];
        sycl::half2* input_h = reinterpret_cast<sycl::half2*>(&input_f);
        output_cast[id] = *input_h;
    }
}

void launch_param_update_half(const float* input, sycl::half* output, int size, sycl::queue* stream)
{
    int threads = 1024;
    size /= 2;
    sycl::range<3> grid_dim(1, 1, (size - 1) / threads + 1);
    sycl::range<3> block_dim(1, 1, threads);

    {
        has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(grid_dim * block_dim, block_dim),
            [=](sycl::nd_item<3> item_ct1) { param_update_kernel_half(input, output, size); });
    }
}
