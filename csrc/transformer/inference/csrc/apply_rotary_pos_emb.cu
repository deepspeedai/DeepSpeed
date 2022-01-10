#include "custom_cuda_layers.h"

#include <cuda_profiler_api.h>

namespace cg = cooperative_groups;

__global__ void apply_rotary_pos_emb(float* mixed_query,
                                     float* key_layer,
                                     unsigned rotary_dim,
                                     unsigned seq_len,
                                     unsigned seq_offset,
                                     unsigned num_heads,
                                     unsigned head_size,
                                     unsigned total_count)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = mixed_query[offset + lane];
            float k = key_layer[offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = g.shfl_xor(q_rot, 1);
            k_rot = g.shfl_xor(k_rot, 1);
            q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

            mixed_query[offset + lane] = q;
            key_layer[offset + lane] = k;

            lane += WARP_SIZE;
        }
    }
}

__global__ void apply_rotary_pos_emb(__half* mixed_query,
                                     __half* key_layer,
                                     unsigned rotary_dim,
                                     unsigned seq_len,
                                     unsigned seq_offset,
                                     unsigned num_heads,
                                     unsigned head_size,
                                     unsigned total_count)
{
#if __CUDA_ARCH__ >= 700
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = (float)mixed_query[offset + lane];
            float k = (float)key_layer[offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = g.shfl_xor(q_rot, 1);
            k_rot = g.shfl_xor(k_rot, 1);
            q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

            mixed_query[offset + lane] = (__half)q;
            key_layer[offset + lane] = (__half)k;

            lane += WARP_SIZE;
        }
    }
#endif
}

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 cudaStream_t stream)
{
    int total_count = batch * num_heads * seq_len;
    dim3 block_dims(1024);
    dim3 grid_dims((total_count - 1) / MAX_WARP_NUM + 1);  // (batch_size);

    apply_rotary_pos_emb<<<grid_dims, block_dims, 0, stream>>>(
        mixed_query, key_layer, rotary_dim, seq_len, offset, num_heads, head_size, total_count);
}

template void launch_apply_rotary_pos_emb<float>(float*,
                                                 float*,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 cudaStream_t);
template void launch_apply_rotary_pos_emb<__half>(__half*,
                                                  __half*,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  cudaStream_t);
