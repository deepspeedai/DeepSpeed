// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdio>
#include <vector>
#include "spatial_cuda_layers.h"

ChannelsLastProblem dimension_problem(at::Tensor& input)
{
    ChannelsLastProblem dims;

    if (input.dim() == 4) {
        // In some sense this is unsafe (and a reflection of the assumptions made inside
        // the C10 options checker). Basically, there's no great way to be sure that
        // a tensor is in channels last because a 1x1 image will appear to be in channels
        // last even when it isn't.
        assert(input.is_contiguous(at::MemoryFormat::ChannelsLast));
        dims.batch_size = input.size(0);
        dims.seq_len = input.size(2) * input.size(3);
        dims.channels = input.size(1);
    } else {
        assert(input.is_contiguous());
        dims.batch_size = input.size(0);
        dims.seq_len = input.size(1);
        dims.channels = input.size(2);
    }

    return dims;
}

at::Tensor seq_unroll_bias_add(at::Tensor& input, at::Tensor& bias)
{
    assert(input.dtype() == at::kHalf);

    // TODO(cmikeh2): Should probably refactor this into a more portable
    // description, since it does generalize for channels-last
    ChannelsLastProblem problem = dimension_problem(input);

    auto output = at::empty_like(input);

    launch_opt_bias_add((__half*)output.data_ptr(),
                        (const __half*)input.data_ptr(),
                        (const __half*)bias.data_ptr(),
                        nullptr,
                        nullptr,
                        problem.batch_size,
                        problem.seq_len,
                        problem.channels,
                        at::cuda::getCurrentCUDAStream());

    return output;
}

at::Tensor seq_bias_add_add(at::Tensor& input, at::Tensor& bias, at::Tensor& other)
{
    assert(input.dtype() == at::kHalf);

    // TODO(cmikeh2): Should probably refactor this into a more portable
    // description, since it does generalize for channels-last
    ChannelsLastProblem problem = dimension_problem(input);

    auto output = at::empty_like(input);

    launch_opt_bias_add((__half*)output.data_ptr(),
                        (const __half*)input.data_ptr(),
                        (const __half*)bias.data_ptr(),
                        (const __half*)other.data_ptr(),
                        nullptr,
                        problem.batch_size,
                        problem.seq_len,
                        problem.channels,
                        at::cuda::getCurrentCUDAStream());

    return output;
}

at::Tensor seq_bias_add_bias_add(at::Tensor& input,
                                 at::Tensor& bias,
                                 at::Tensor& other,
                                 at::Tensor& other_bias)
{
    assert(input.dtype() == at::kHalf);

    // TODO(cmikeh2): Should probably refactor this into a more portable
    // description, since it does generalize for channels-last
    ChannelsLastProblem problem = dimension_problem(input);

    auto output = at::empty_like(input);

    launch_opt_bias_add((__half*)output.data_ptr(),
                        (const __half*)input.data_ptr(),
                        (const __half*)bias.data_ptr(),
                        (const __half*)other.data_ptr(),
                        (const __half*)other_bias.data_ptr(),
                        problem.batch_size,
                        problem.seq_len,
                        problem.channels,
                        at::cuda::getCurrentCUDAStream());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("nhwc_bias_add", &seq_unroll_bias_add);
    m.def("nhwc_bias_add_add", &seq_bias_add_add);
    m.def("nhwc_bias_add_bias_add", &seq_bias_add_bias_add);
}
