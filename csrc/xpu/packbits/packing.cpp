// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ipex.h>
#include <torch/extension.h>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace xpu;

void packbitskernel(const float* input, uint8_t* output, const int input_size, id<1> item_ct1)
{
    // get the sign bit of each float and pack them into byte
    int i = item_ct1;
    for (int j = 0; j < 8; ++j) {
        int k = i * 8 + j;
        int bit = k < input_size && (!sycl::signbit(input[k]));
        output[i] |= bit << (7 - j);
    }
}

void unpackbitskernel(const uint8_t* input, float* output, id<1> item_ct1)
{
    // use the bit value to set float, bit 0 -> float -1, bit 1 -> float 1
    int i = item_ct1;
    output[i] = (float((input[i / 8] >> (7 - i % 8)) & 1) - 0.5) * 2;
}

sycl::queue get_current_queue(at::Device device)
{
    c10::impl::VirtualGuardImpl impl(device.type());
    c10::Stream _stream = impl.getStreamFromGlobalPool(device, /*isHighPriority=*/false);
    sycl::queue queue = xpu::get_queue_from_stream(_stream);
    return queue;
}

/*
pack float tensor into uint8 tensor. Every eight float elements get packed into one uint8
if float x >= 0, will be packed as a '1' bit, or will be packed as '0'
Arguments:
    tensor: A bool tensor that get packed.
    input_size: numel of input tensor
    rank: device id in order to get corresponding stream
*/
at::Tensor packbits(at::Tensor tensor, int input_size, int rank)
{
    at::Device device = "xpu:" + std::to_string(rank);
    sycl::queue q = get_current_queue(device);

    int packed_size = (input_size + 7) / 8;
    auto unit8_options = at::TensorOptions().dtype(at::kByte).device(at::kXPU);
    at::Tensor packed = torch::zeros({packed_size}, unit8_options);

    float* input = (float*)tensor.data_ptr();
    uint8_t* output = (uint8_t*)packed.data_ptr();

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<>(range(packed_size), [=](id<1> item_ct1) {
            packbitskernel(input, output, input_size, item_ct1);
        });
    });

    return packed;
}

/*
unpack uint8 tensor into float tensor. Every uint8 element get unpacked into eight float
a '1' bit will be converted to a float(1), a '0' bit will be converted to a float(-1).
Arguments:
    tensor: A uint8 tensor that get unpacked.
    input_size: numel of input tensor
    rank: device id in order to get corresponding stream
*/
at::Tensor unpackbits(at::Tensor tensor, int input_size, int rank)
{
    at::Device device = "xpu:" + std::to_string(rank);
    sycl::queue q = get_current_queue(device);

    auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kXPU);
    at::Tensor unpacked = torch::empty({input_size * 8}, float_options);

    uint8_t* input = (uint8_t*)tensor.data_ptr();
    float* output = (float*)unpacked.data_ptr();

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<>(range(input_size * 8),
                           [=](id<1> item_ct1) { unpackbitskernel(input, output, item_ct1); });
    });

    return unpacked;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packbits", &packbits, "DeepSpeed XPU packbits (C++)");
    m.def("unpackbits", &unpackbits, "DeepSpeed XPU unpackbits (C++)");
}
