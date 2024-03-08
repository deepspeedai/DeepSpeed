// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <stdexcept>
#include "activation_type.h"
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

namespace gated_act {

constexpr int access_size = 16;
constexpr int threads = 1024;

template <ActivationType ActType>
float gated_act_fn(float x, float y);

template <>
DS_D_INLINE float gated_act_fn<ActivationType::GEGLU>(float x, float y)
{
    constexpr float sqrt_param = 0.79788456080286535587989211986876f;
    constexpr float mul_param = 0.044715;
    return y * x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

template <>
DS_D_INLINE float gated_act_fn<ActivationType::ReGLU>(float x, float y)
{
    return y * (x > 0.0f ? x : 0.0f);
}

template <>
DS_D_INLINE float gated_act_fn<ActivationType::SiGLU>(float x, float y)
{
    return y * (x / (1.0f + expf(-x)));
}

}  // namespace gated_act

template <typename T, ActivationType ActType, int loopUnroll>
__global__ void gated_activation_kernel(T* output,
                                        const T* input,
                                        const T* bias,
                                        int rows,
                                        int cols)
{
    constexpr int read_vector = gated_act::access_size / sizeof(T);
    constexpr int write_vector = read_vector / 2;

    const int row = blockIdx.x;
    const int col = threadIdx.x * read_vector;

    const T* input_row = input + row * cols;
    T* output_row = output + row * cols / 2;

#pragma unroll
    for (int i = 0; i < loopUnroll; i++) {
        T read[read_vector];
        T bias_read[read_vector];
        T store[write_vector];

        const int read_offset = col + gated_act::threads * read_vector * i;
        const int write_offset = col / 2 + gated_act::threads * write_vector * i;

        if (i != loopUnroll - 1 || read_offset < cols) {
            mem_access::load_global<gated_act::access_size>(read, input_row + read_offset);
            mem_access::load_global<gated_act::access_size>(
                bias_read, bias + read_offset, bias != nullptr);

            for (int j = 0; j < write_vector; j++) {
                float g_val =
                    conversion::to<float>(read[j * 2]) + conversion::to<float>(bias_read[j * 2]);
                float a_val = conversion::to<float>(read[j * 2 + 1]) +
                              conversion::to<float>(bias_read[j * 2 + 1]);

                float act_val = gated_act::gated_act_fn<ActType>(g_val, a_val);
                store[j] = conversion::to<T>(act_val);
            }

            mem_access::store_global<gated_act::access_size / 2>(output_row + write_offset, store);
        }
    }
}

#define DISPATCH_UNROLL(unroll_val)                 \
    gated_activation_kernel<T, ActType, unroll_val> \
        <<<grid, block, 0, stream>>>(output, input, bias, rows, cols);

template <typename T, ActivationType ActType>
void launch_gated_activation_impl(T* output,
                                  const T* input,
                                  const T* bias,
                                  int rows,
                                  int cols,
                                  cudaStream_t stream)
{
    constexpr int read_vector = gated_act::access_size / sizeof(T);
    constexpr int cols_per_unroll = gated_act::threads * read_vector;
    const int req_threads = (cols + read_vector - 1) / read_vector;
    const int threads = std::min(req_threads, gated_act::threads);

    const dim3 grid(rows);
    const dim3 block(threads);
    const int unroll = (cols + cols_per_unroll - 1) / cols_per_unroll;

    if (unroll == 1) {
        DISPATCH_UNROLL(1);
    } else if (unroll == 2) {
        DISPATCH_UNROLL(2);
    } else if (unroll == 3) {
        DISPATCH_UNROLL(3);
    } else if (unroll == 4) {
        DISPATCH_UNROLL(4);
    } else if (unroll == 5) {
        DISPATCH_UNROLL(5);
    } else if (unroll == 6) {
        DISPATCH_UNROLL(6);
    } else if (unroll == 7) {
        DISPATCH_UNROLL(7);
    } else {
        // TODO: provide a kernel with an outer loop to handle larger columns.
        throw std::runtime_error(
            "Called with more columns than supported, please report this bug and this limit will "
            "be increased.");
    }
}

template <typename T>
void launch_gated_activation(T* output,
                             const T* input,
                             const T* bias,
                             int rows,
                             int cols,
                             ActivationType act_type,
                             cudaStream_t stream)
{
    switch (act_type) {
        case ActivationType::GEGLU:
            launch_gated_activation_impl<T, ActivationType::GEGLU>(
                output, input, bias, rows, cols, stream);
            break;
        case ActivationType::ReGLU:
            launch_gated_activation_impl<T, ActivationType::ReGLU>(
                output, input, bias, rows, cols, stream);
            break;
        case ActivationType::SiGLU:
            launch_gated_activation_impl<T, ActivationType::SiGLU>(
                output, input, bias, rows, cols, stream);
            break;
        default: throw std::runtime_error("Unsupported activation type");
    }
}

#define INSTANTIATE_FOR_TYPE(T)                                       \
    template void launch_gated_activation<T>(T * output,              \
                                             const T* input,          \
                                             const T* bias,           \
                                             int rows,                \
                                             int cols,                \
                                             ActivationType act_type, \
                                             cudaStream_t stream);

INSTANTIATE_FOR_TYPE(float)
INSTANTIATE_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_FOR_TYPE(__nv_bfloat16)
#endif
