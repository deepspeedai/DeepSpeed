// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "dequantization_utils.h"
#include "memory_access_utils.h"
#include "quantization_utils.h"
#include "reduction_utils.h"

using rop = reduce::ROpType;

namespace swiz_quant {
constexpr int max_threads = 512;
constexpr int min_threads = 32;

constexpr int step_granularity = 2;
constexpr int h_per_step = step_granularity * quantize::h_per_load;
}  // namespace swiz_quant

template <int numBits, int totalChunks, int threads, quantize::Type quantType>
__global__ void swizzled_quant_kernel(int8_t* quantized_data,
                                      float* quantized_scales,
                                      const __half* uncompressed_data,
                                      int elems_per_group,
                                      int nodes,
                                      int devices_per_node)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets, same as normal quantization for in-case
    const int block_rank = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    const int block_offset = block_rank * elems_per_group;
    const int elem_offset = tb.thread_index().x * quantize::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.size() * quantize::h_per_load;
    const __half* input_base = uncompressed_data + base_offset;

    // Local buffer
    __half2 local_buffer[totalChunks * quantize::h2_per_load];

    quantize::GroupStats<quantType> stats;
#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        __half2* iteration_buffer = local_buffer + i * quantize::h2_per_load;

        mem_access::load_global<quantize::granularity>(
            iteration_buffer, input_base + i * stride, elem_offset + i * stride < elems_per_group);

#pragma unroll
        for (int j = 0; j < quantize::h2_per_load; j++) { stats.update(iteration_buffer[j]); }
    }

    auto params = stats.template get_params<numBits, threads>(tb, warp);

    const int partition_id = blockIdx.z;
    const int partition_offset = partition_id / devices_per_node;
    const int partition_base = (partition_id % devices_per_node) * nodes;
    const int pipelining_offset = blockIdx.y * (devices_per_node * nodes);
    const int output_partition = (pipelining_offset + partition_base + partition_offset);

    constexpr int out_scalar_effect = 8 / numBits;
    const int out_block_rank = output_partition * gridDim.x + blockIdx.x;
    const int out_block_offset = out_block_rank * elems_per_group / out_scalar_effect;
    const int out_base_offset = out_block_offset + elem_offset / out_scalar_effect;
    int8_t* out_base = quantized_data + out_base_offset;

    const int out_stride = stride / out_scalar_effect;
    constexpr int num_int8_out = quantize::h_per_load / out_scalar_effect;

    if (tb.thread_index().x == 0) { params.store(quantized_scales, out_block_rank); }

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        if (i * stride + elem_offset < elems_per_group) {
            int8_t local_output[quantize::h_per_load / out_scalar_effect];
            quantize::_chunk<numBits, quantType>(
                local_output, local_buffer + i * quantize::h2_per_load, params);
            mem_access::store_global<num_int8_out>(out_base + i * out_stride, local_output);
        }
    }
}

#define LAUNCH_SWIZZLE_QUANT(total_chunks, threads)                                           \
    swizzled_quant_kernel<numBits, total_chunks, threads, qType><<<grid, block, 0, stream>>>( \
        q_data, q_scales, input_data, elems_per_group, nodes, devices_per_node);

/*
Swizzled quantization reorganizes the quantized groups in order to better facilitate
communication. As an example of the partitioning scheme we have the following example
of 2 node, 4 device swizzling:

 --- --- --- --- --- --- --- ---
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
 --- --- --- --- --- --- --- ---
becomes
 --- --- --- --- --- --- --- ---
| 0 | 4 | 1 | 5 | 2 | 6 | 3 | 7 |
 --- --- --- --- --- --- --- ---

Multiple quantization groups may be mapped into a single partition. In order to better support
later pipelining, we may also perform an additional slicing. In two-way slicing, for instance,
the first halves of each partition are concatenated.
*/

template <int numBits, quantize::Type qType>
void launch_swizzled_quant_impl(int8_t* q_data,
                                float* q_scales,
                                const __half* input_data,
                                int groups,
                                int elems_per_group,
                                int pipelining,
                                int nodes,
                                int devices_per_node,
                                cudaStream_t stream)
{
    const int one_step_threads =
        next_pow2((elems_per_group + swiz_quant::h_per_step - 1) / swiz_quant::h_per_step);
    const int max_threads = (one_step_threads < swiz_quant::max_threads) ? one_step_threads
                                                                         : swiz_quant::max_threads;
    const int threads = (max_threads < swiz_quant::min_threads) ? swiz_quant::min_threads
                                                                : max_threads;

    dim3 block(threads);
    const int groups_per_partition = groups / (nodes * devices_per_node);
    assert(groups_per_partition % pipelining == 0);
    const int contiguous_groups = groups_per_partition / pipelining;
    const int partitions = nodes * devices_per_node;
    dim3 grid(contiguous_groups, pipelining, partitions);

    const int elems_per_step = threads * swiz_quant::h_per_step;
    const int external_unroll = ((elems_per_group + elems_per_step - 1) / elems_per_step);
    const int total_unroll = external_unroll * swiz_quant::step_granularity;

    assert(total_unroll % 2 == 0);

    if (threads == 32) {
        LAUNCH_SWIZZLE_QUANT(2, 32);
    } else if (threads == 64) {
        LAUNCH_SWIZZLE_QUANT(2, 64);
    } else if (threads == 128) {
        LAUNCH_SWIZZLE_QUANT(2, 128);
    } else if (threads == 256) {
        LAUNCH_SWIZZLE_QUANT(2, 256);
    } else if (threads == 512) {
        if (total_unroll == 2) {
            LAUNCH_SWIZZLE_QUANT(2, 512);
        } else if (total_unroll == 4) {
            LAUNCH_SWIZZLE_QUANT(4, 512);
        } else if (total_unroll == 6) {
            LAUNCH_SWIZZLE_QUANT(6, 512);
        } else if (total_unroll == 8) {
            LAUNCH_SWIZZLE_QUANT(8, 512);
        } else if (total_unroll == 10) {
            LAUNCH_SWIZZLE_QUANT(10, 512);
        }
    }
}

#define DISPATCH_SWIZZLE_QUANT(num_bits, qtype)                   \
    launch_swizzled_quant_impl<num_bits, qtype>(q_data,           \
                                                q_scales,         \
                                                input_data,       \
                                                groups,           \
                                                elems_per_group,  \
                                                pipelining,       \
                                                nodes,            \
                                                devices_per_node, \
                                                stream);

void launch_swizzled_quant(int8_t* q_data,
                           float* q_scales,
                           const __half* input_data,
                           int num_bits,
                           quantize::Type q_type,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           cudaStream_t stream)
{
    if (num_bits == 4) {
        if (q_type == quantize::Type::Asymmetric) {
            DISPATCH_SWIZZLE_QUANT(4, quantize::Type::Asymmetric);
        } else if (q_type == quantize::Type::Symmetric) {
            DISPATCH_SWIZZLE_QUANT(4, quantize::Type::Symmetric);
        }
    } else if (num_bits == 8) {
        if (q_type == quantize::Type::Asymmetric) {
            DISPATCH_SWIZZLE_QUANT(8, quantize::Type::Asymmetric);
        } else if (q_type == quantize::Type::Symmetric) {
            DISPATCH_SWIZZLE_QUANT(8, quantize::Type::Symmetric);
        }
    }
}

template <int numBits, int totalChunks, int threads, quantize::Type quantType>
__global__ void loco_swizzled_quant_kernel(int8_t* quantized_data,
                                           float* quantized_scales,
                                           const __half* uncompressed_data,
                                           __half* error_feedback,
                                           const float err_beta,
                                           int groups,
                                           int elems_per_group,
                                           int pipelining,
                                           int nodes,
                                           int devices_per_node)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets, same as normal quantization for in-case
    const int block_rank_data =
        blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    const int block_offset_data = block_rank_data * elems_per_group;
    const int elem_offset = tb.thread_index().x * quantize::h_per_load;
    const int base_offset_data = block_offset_data + elem_offset;
    const int stride = tb.size() * quantize::h_per_load;
    const __half* uncompressed_data_base = uncompressed_data + base_offset_data;

    const int partition_id = blockIdx.z;
    const int partition_offset = partition_id / devices_per_node;
    const int partition_base = (partition_id % devices_per_node) * nodes;
    const int pipelining_offset = blockIdx.y * (devices_per_node * nodes);
    const int output_partition = (pipelining_offset + partition_base + partition_offset);
    const int block_rank_err = output_partition * gridDim.x + blockIdx.x;

    const int block_offset_err = block_rank_err * elems_per_group;
    const int base_offset_err = block_offset_err + elem_offset;
    __half* error_feedback_base = error_feedback + base_offset_err;

    __half2 local_buffer[totalChunks * quantize::h2_per_load];
    __half2 err_buffer[totalChunks * quantize::h2_per_load];

    quantize::GroupStats<quantType> stats;

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        __half2* iteration_buffer = local_buffer + i * quantize::h2_per_load;
        __half2* iter_err_buffer = err_buffer + i * quantize::h2_per_load;
        const int i_stride = i * stride;
        bool do_loads = (elem_offset + i_stride) < elems_per_group;

        mem_access::load_global<quantize::granularity>(
            iteration_buffer, uncompressed_data_base + i_stride, do_loads);

        mem_access::load_global<quantize::granularity>(
            iter_err_buffer, error_feedback_base + i_stride, do_loads);

#pragma unroll
        for (int j = 0; j < quantize::h2_per_load; j++) {
            iteration_buffer[j] = __hadd2(iteration_buffer[j], iter_err_buffer[j]);
            stats.update(iteration_buffer[j]);
        }
    }

    auto params = stats.template get_params<numBits, threads>(tb, warp);

    // Initialize dequantization parameters based on params
    auto de_params = params;
    de_params.scale = 1.0f / params.scale;
    if constexpr (quantType == quantize::Type::Asymmetric) { de_params.offset = params.offset; }

    if (threadIdx.x == 0) { params.store(quantized_scales, block_rank_err); }

    constexpr int out_scalar_effect = 8 / numBits;
    const int out_block_offset = block_rank_err * elems_per_group / out_scalar_effect;
    const int out_base_offset = out_block_offset + elem_offset / out_scalar_effect;
    int8_t* out_base = quantized_data + out_base_offset;

    const int out_stride = stride / out_scalar_effect;
    constexpr int num_int8_out = quantize::h_per_load / out_scalar_effect;

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        const int i_stride = i * stride;
        __half2* iteration_buffer = local_buffer + i * quantize::h2_per_load;
        __half2* iter_err_buffer = err_buffer + i * quantize::h2_per_load;

        if (i_stride + elem_offset < elems_per_group) {
            int8_t local_output[quantize::h_per_load / out_scalar_effect];
            quantize::_chunk<numBits, quantType>(local_output, iteration_buffer, params);
            mem_access::store_global<num_int8_out>(out_base + i * out_stride, local_output);

            // Dequantize the quantized output to compute the dequantized value
            __half2 dequant_buffer[quantize::h2_per_load];
            dequantize::chunk<numBits, quantType>(dequant_buffer, local_output, de_params);

// Compute new error: sum - dequant_buffer
#pragma unroll
            for (int k = 0; k < quantize::h2_per_load; k++) {
                // __half2 to float2
                float2 iter_buf_f = __half22float2(iteration_buffer[k]);
                float2 dequant_buf_f = __half22float2(dequant_buffer[k]);

                // Update within float precision
                float2 new_error_f;
                new_error_f.x = iter_buf_f.x - dequant_buf_f.x;
                new_error_f.y = iter_buf_f.y - dequant_buf_f.y;

                float2 iter_err_buf_f = __half22float2(iter_err_buffer[k]);

                iter_err_buf_f.x = err_beta * iter_err_buf_f.x + (1.0f - err_beta) * new_error_f.x;
                iter_err_buf_f.y = err_beta * iter_err_buf_f.y + (1.0f - err_beta) * new_error_f.y;

                // float2 back to __half2
                iter_err_buffer[k] = __float22half2_rn(iter_err_buf_f);
            }
            __half2* error_feedback_base_h2 = reinterpret_cast<__half2*>(error_feedback_base);
            mem_access::store_global<quantize::granularity>(error_feedback_base_h2 + i_stride / 2,
                                                            iter_err_buffer);
        }
    }
}

#define LAUNCH_LOCO_SWIZZLE_QUANT(total_chunks, threads)              \
    loco_swizzled_quant_kernel<numBits, total_chunks, threads, qType> \
        <<<grid, block, 0, stream>>>(output_data,                     \
                                     params,                          \
                                     input_data,                      \
                                     error_feedback,                  \
                                     err_beta,                        \
                                     groups,                          \
                                     elems_per_group,                 \
                                     pipelining,                      \
                                     nodes,                           \
                                     devices_per_node);

template <int numBits, quantize::Type qType>
void launch_loco_swizzled_quant_impl(int8_t* output_data,
                                     float* params,
                                     const __half* input_data,
                                     __half* error_feedback,
                                     const float err_beta,
                                     int groups,
                                     int elems_per_group,
                                     int pipelining,
                                     int nodes,
                                     int devices_per_node,
                                     cudaStream_t stream)
{
    const int one_step_threads =
        next_pow2((elems_per_group + swiz_quant::h_per_step - 1) / swiz_quant::h_per_step);
    const int max_threads = (one_step_threads < swiz_quant::max_threads) ? one_step_threads
                                                                         : swiz_quant::max_threads;
    const int threads = (max_threads < swiz_quant::min_threads) ? swiz_quant::min_threads
                                                                : max_threads;

    dim3 block(threads);
    const int groups_per_partition = groups / (nodes * devices_per_node);
    assert(groups_per_partition % pipelining == 0);
    const int contiguous_groups = groups_per_partition / pipelining;
    const int partitions = nodes * devices_per_node;
    dim3 grid(contiguous_groups, pipelining, partitions);

    const int elems_per_step = threads * swiz_quant::h_per_step;
    const int external_unroll = ((elems_per_group + elems_per_step - 1) / elems_per_step);
    const int total_unroll = external_unroll * swiz_quant::step_granularity;

    assert(total_unroll % 2 == 0);

    if (threads == 32) {
        LAUNCH_LOCO_SWIZZLE_QUANT(2, 32);
    } else if (threads == 64) {
        LAUNCH_LOCO_SWIZZLE_QUANT(2, 64);
    } else if (threads == 128) {
        LAUNCH_LOCO_SWIZZLE_QUANT(2, 128);
    } else if (threads == 256) {
        LAUNCH_LOCO_SWIZZLE_QUANT(2, 256);
    } else if (threads == 512) {
        if (total_unroll == 2) {
            LAUNCH_LOCO_SWIZZLE_QUANT(2, 512);
        } else if (total_unroll == 4) {
            LAUNCH_LOCO_SWIZZLE_QUANT(4, 512);
        } else if (total_unroll == 6) {
            LAUNCH_LOCO_SWIZZLE_QUANT(6, 512);
        } else if (total_unroll == 8) {
            LAUNCH_LOCO_SWIZZLE_QUANT(8, 512);
        } else if (total_unroll == 10) {
            LAUNCH_LOCO_SWIZZLE_QUANT(10, 512);
        }
    }
}

#define DISPATCH_LOCO_SWIZZLE_QUANT(num_bits, qtype)                   \
    launch_loco_swizzled_quant_impl<num_bits, qtype>(output_data,      \
                                                     params,           \
                                                     input_data,       \
                                                     error_feedback,   \
                                                     err_beta,         \
                                                     groups,           \
                                                     elems_per_group,  \
                                                     pipelining,       \
                                                     nodes,            \
                                                     devices_per_node, \
                                                     stream);

void launch_loco_swizzled_quant(int8_t* output_data,
                                float* params,
                                const __half* input_data,
                                __half* error_feedback,
                                const float err_beta,
                                int num_bits,
                                quantize::Type q_type,
                                int groups,
                                int elems_per_group,
                                int pipelining,
                                int nodes,
                                int devices_per_node,
                                cudaStream_t stream)
{
    if (num_bits == 4) {
        if (q_type == quantize::Type::Asymmetric) {
            DISPATCH_LOCO_SWIZZLE_QUANT(4, quantize::Type::Asymmetric);
        } else if (q_type == quantize::Type::Symmetric) {
            DISPATCH_LOCO_SWIZZLE_QUANT(4, quantize::Type::Symmetric);
        }
    } else if (num_bits == 8) {
        if (q_type == quantize::Type::Asymmetric) {
            DISPATCH_LOCO_SWIZZLE_QUANT(8, quantize::Type::Asymmetric);
        } else if (q_type == quantize::Type::Symmetric) {
            DISPATCH_LOCO_SWIZZLE_QUANT(8, quantize::Type::Symmetric);
        }
    }
}
