/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "inference_cuda_layers.h"

#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#endif

namespace cg = cooperative_groups;
namespace cg = cooperative_groups;

__global__ void apply_rotary_pos_emb(float* mixed_query,
                                     float* key_layer,
                                     unsigned rotary_dim,
                                     unsigned seq_len,
                                     unsigned seq_offset,
                                     unsigned num_heads,
                                     unsigned head_size,
                                     unsigned total_count,
                                     int max_out_tokens)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;
    unsigned seq_index = head_id % seq_len;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = mixed_query[offset + lane];
            float k = key_layer[k_offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = g.shfl_xor(q_rot, 1);
            k_rot = g.shfl_xor(k_rot, 1);
            q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

            mixed_query[offset + lane] = q;
            key_layer[k_offset + lane] = k;

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
                                     unsigned total_count,
                                     int max_out_tokens)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;
    unsigned seq_index = head_id % seq_len;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = (float)mixed_query[offset + lane];
            float k = (float)key_layer[k_offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = g.shfl_xor(q_rot, 1);
            k_rot = g.shfl_xor(k_rot, 1);
            q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

            mixed_query[offset + lane] = (__half)q;
            key_layer[k_offset + lane] = (__half)k;

            lane += WARP_SIZE;
        }
    }
}
__global__ void apply_rotary_pos_emb1(float* mixed_query,
                                      float* key_layer,
                                      unsigned rotary_dim,
                                      unsigned seq_len,
                                      unsigned seq_offset,
                                      unsigned num_heads,
                                      unsigned head_size,
                                      unsigned total_count,
                                      int max_out_tokens)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned offset = head_id * head_size;

    unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;
    unsigned seq_index = head_id % seq_len;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = mixed_query[offset + lane];
            float k = key_layer[k_offset + lane];
            float rotary_sign = (lane % 2 == 1 ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            q_rot = g.shfl_xor(q_rot, 1);
            k_rot = g.shfl_xor(k_rot, 1);
            q = q * cosf(inv_freq) + q_rot * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot * sinf(inv_freq);

            mixed_query[offset + lane] = q;
            key_layer[k_offset + lane] = k;

            lane += WARP_SIZE;
        }
    }
}
__global__ void apply_rotary_pos_emb1(__half* mixed_query,
                                      __half* key_layer,
                                      unsigned rotary_dim,
                                      unsigned seq_len,
                                      unsigned seq_offset,
                                      unsigned num_heads,
                                      unsigned head_size,
                                      unsigned total_count,
                                      int max_out_tokens)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int lane = id & 0x1f;

    unsigned head_id = blockIdx.x * MAX_WARP_NUM + gid;
    unsigned seq_index = head_id % seq_len;
    unsigned offset = head_id * head_size;
    unsigned k_offset = (seq_index + (head_id / seq_len) * max_out_tokens) * head_size;

    constexpr unsigned mask[32] = {
        0x1 | 0x1000,     0x2 | 0x2000,     0x4 | 0x4000,     0x8 | 0x8000,     0x10 | 0x10000,
        0x20 | 0x20000,   0x40 | 0x40000,   0x80 | 0x80000,   0x100 | 0x100000, 0x200 | 0x200000,
        0x400 | 0x400000, 0x800 | 0x800000, 0x1000 | 0x1,     0x2000 | 0x2,     0x4000 | 0x4,
        0x8000 | 0x8,     0x10000 | 0x10,   0x20000 | 0x20,   0x40000 | 0x40,   0x80000 | 0x80,
        0x100000 | 0x100, 0x200000 | 0x200, 0x400000 | 0x400, 0x800000 | 0x800, 0x1000000,
        0x2000000,        0x4000000,        0x8000000,        0x10000000,       0x20000000,
        0x40000000,       0x80000000};

    unsigned seq_id = (head_id % seq_len) + seq_offset;
    unsigned half_dim = rotary_dim >> 1;
    if (head_id < total_count) {
        while (lane < rotary_dim) {
            float inv_freq = (float)((lane % half_dim) * 2) / (float)rotary_dim;
            inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
            float q = (float)mixed_query[offset + lane];
            float k = (float)key_layer[k_offset + lane];
            float rotary_sign = (lane > (half_dim - 1) ? -1.0 : 1.0);
            float q_rot = (q * rotary_sign);
            float k_rot = (k * rotary_sign);
            auto q_rot_tmp = lane < half_dim ? __shfl_sync(mask[lane], q_rot, lane + half_dim)
                                             : __shfl_sync(mask[lane], q_rot, lane - half_dim);
            auto k_rot_tmp = lane < half_dim ? __shfl_sync(mask[lane], k_rot, lane + half_dim)
                                             : __shfl_sync(mask[lane], k_rot, lane - half_dim);
            q = q * cosf(inv_freq) + q_rot_tmp * sinf(inv_freq);
            k = k * cosf(inv_freq) + k_rot_tmp * sinf(inv_freq);

            mixed_query[offset + lane] = (__half)q;
            key_layer[k_offset + lane] = (__half)k;

            lane += WARP_SIZE;
        }
    }
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
                                 bool rotate_half,
                                 bool rotate_every_two,
                                 cudaStream_t stream,
                                 int max_out_tokens)
{
    int total_count = batch * num_heads * seq_len;
    dim3 block_dims(1024);
    dim3 grid_dims((total_count - 1) / MAX_WARP_NUM + 1);  // (batch_size);
    if (rotate_every_two)
        apply_rotary_pos_emb<<<grid_dims, block_dims, 0, stream>>>(mixed_query,
                                                                   key_layer,
                                                                   rotary_dim,
                                                                   seq_len,
                                                                   offset,
                                                                   num_heads,
                                                                   head_size,
                                                                   total_count,
                                                                   max_out_tokens);
    else if (rotate_half)
        apply_rotary_pos_emb1<<<grid_dims, block_dims, 0, stream>>>(mixed_query,
                                                                    key_layer,
                                                                    rotary_dim,
                                                                    seq_len,
                                                                    offset,
                                                                    num_heads,
                                                                    head_size,
                                                                    total_count,
                                                                    max_out_tokens);
}

template void launch_apply_rotary_pos_emb<float>(float*,
                                                 float*,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 unsigned,
                                                 bool,
                                                 bool,
                                                 cudaStream_t,
                                                 int);
template void launch_apply_rotary_pos_emb<__half>(__half*,
                                                  __half*,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  unsigned,
                                                  bool,
                                                  bool,
                                                  cudaStream_t,
                                                  int);

/*
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
constexpr unsigned mask[32] = {0x1 | 0x1000, 0x2 | 0x2000, 0x4 | 0x4000, 0x8 | 0x8000,
0x10 | 0x10000, 0x20 | 0x20000, 0x40 | 0x40000, 0x80 | 0x80000,
0x100 | 0x100000, 0x200 | 0x200000, 0x400 | 0x400000, 0x800 | 0x800000,
0x1000 | 0x1, 0x2000 | 0x2, 0x4000 | 0x4, 0x8000 | 0x8,
0x10000 | 0x10, 0x20000 | 0x20, 0x40000 | 0x40, 0x80000 | 0x80,
0x100000 | 0x100, 0x200000 | 0x200, 0x400000 | 0x400, 0x800000 | 0x800,
0x1000000, 0x2000000, 0x4000000, 0x8000000,
0x10000000, 0x20000000, 0x40000000, 0x80000000};
unsigned seq_id = (head_id / num_heads) % seq_len + seq_offset;

if (head_id < total_count) {
while (lane < rotary_dim) {
//float inv_freq = (float)((lane / 2) * 2) / (float)rotary_dim;
float inv_freq = (float)((lane % (rotary_dim >> 1)) * 2) / (float)rotary_dim;
inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
float q = (float)mixed_query[offset + lane];
float k = (float)key_layer[offset + lane];
float rotary_sign = (lane > 11 ? -1.0 : 1.0);
float q_rot = (q * rotary_sign);
float k_rot = (k * rotary_sign);
auto q_rot_tmp = lane < 12 ? __shfl_sync(mask[lane], q_rot, lane + 12) : __shfl_sync(mask[lane],
q_rot, lane - 12);//g.shfl_xor(q_rot, 12); auto k_rot_tmp = lane < 12 ? __shfl_sync(mask[lane],
k_rot, lane + 12) : __shfl_sync(mask[lane], k_rot, lane - 12);//g.shfl_xor(k_rot, 12); q = q *
cosf(inv_freq) + q_rot_tmp * sinf(inv_freq); k = k * cosf(inv_freq) + k_rot_tmp * sinf(inv_freq);

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
*/
