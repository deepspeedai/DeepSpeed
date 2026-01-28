#include "moe_gating.cuh"
#include "reduction_utils.h"
#include "stdio.h"
#include "tops_context.h"

using ROp = reduce::ROpType;



__global__ void top_1_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    float* logits,
                                    float* logits_out,
                                    const int capacity,
                                    const int32_t n_experts,
                                    const int32_t n_tokens)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    float* token_logits = logits + token_idx * n_experts;
    float* token_logits_out = logits_out + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) 
        logit_val = token_logits[expert_idx];
    else {
        reduce::init<ROp::Max>(&logit_val);
    }

    int32_t inverted_expert = n_experts - expert_idx - 1;
    // Perform softmax
    const reduce::IdxReduceResult res =
        reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, logit_val, inverted_expert);
    // Recover the original expert index
    const int32_t assigned_expert = n_experts - res.idx - 1;
    const float max_logit = res.val;

    float softlogit = __expf(logit_val - max_logit);
    float softmax_sum = softlogit;
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    // Compute the score
    const float score = 1.0 / softmax_sum;
    if (expert_idx < n_experts) token_logits_out[expert_idx] = softlogit / softmax_sum;
    
    if (threadIdx.x == 0)
    {
        atomicAdd(expert_counts + assigned_expert, 1);
        scores[token_idx] = score;
        assignments[token_idx] = assigned_expert;
    }
}

template <typename t>
__global__ void refine_expert_mapping(int32_t* expert_counts,
                                    int32_t* mapped_expert_counts,
                                    int32_t* expert_count_cumsums,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    int32_t* backup_offsets,
                                    const int capacity,
                                    const int32_t n_tokens,
                                    const int32_t n_experts,
                                    std::pair<uint64_t, uint64_t> seed){

    const int32_t bidx = blockIdx.x;
    const int32_t tidx = threadIdx.x;
    int32_t token_idx = bidx * blockDim.x + tidx;
    if (token_idx >= n_tokens) {
        return;
    }

    int32_t assignment = assignments[token_idx];

    int idx = token_idx << 2;
    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    int32_t total_ec = expert_counts[assignment];
    float ratio = 1.0 - (float)(capacity) / (float)total_ec;

    float4 rand_val = curand_uniform4(&state);
    

    {
        if ((rand_val.z > ratio && rand_val.x > ratio && rand_val.y > ratio) || 
            (rand_val.z <= ratio && (rand_val.x > ratio || rand_val.y > ratio))) {
        // if (true) {
            offsets[token_idx] = atomicAdd(mapped_expert_counts + assignment, 1);
            //backup_offsets[token_idx] = atomicAdd(expert_count_cumsums + assignment, 1);// gating::unassigned;

        }
        else{
            offsets[token_idx] = gating::unassigned;
            backup_offsets[token_idx] = atomicAdd(expert_count_cumsums + assignment, 1);// gating::unassigned;
            
            // assignments[token_idx] = n_experts; //gating::unassigned;
            // scores[token_idx] = 0.f;
        } // need to set these tokens to Zero!
    }
}


__global__ void gate_logits_bwd_kernel(float* logits_grad,
                                        float* scores_grad,
                                        const int32_t* assignment,
                                        float* logits,
                                        const int32_t n_experts,
                                        const int32_t n_tokens)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    int32_t assigned_expert = assignment[token_idx];
    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= n_tokens) {
        return;
    }

    float* token_logits = logits + token_idx * n_experts;
    float* token_logits_grad = logits_grad + token_idx * n_experts;

    float logit_val;
    float logit_grad_val;

    if (expert_idx < n_experts) {
        logit_val = token_logits[expert_idx];
        logit_grad_val = token_logits_grad[expert_idx];
    } else {
        reduce::init<ROp::Add>(&logit_val);
        reduce::init<ROp::Add>(&logit_grad_val);
    }

    if (assigned_expert == expert_idx) {
        logit_grad_val += scores_grad[token_idx];
    }
    float softmax_grad_sum = logit_val * logit_grad_val;
    reduce::block<ROp::Add>(tb, warp, softmax_grad_sum);
    logit_grad_val = logit_val * (logit_grad_val - softmax_grad_sum);
    
    if (expert_idx < n_experts) 
        token_logits_grad[expert_idx] = logit_grad_val;
}

template <typename T, int copyUnroll>
__global__ void moe_gather_kernel(T* layer_output,
                                  const T* moe_output,
                                  const float* scores,
                                  const int32_t* mapped_slots,
                                  const int32_t n_channels,
                                  const int32_t num_tokens)
{
    constexpr int32_t vector_size = scatter_gather::access_granularity / sizeof(T);
    constexpr int32_t stride = vector_size * scatter_gather::threads;

    const int32_t token_idx = blockIdx.x;
    const int32_t mapped_slot = mapped_slots[token_idx];

    const float score = scores[token_idx];
    const int32_t channel_offset = threadIdx.x * vector_size;

    const T* moe_output_base = moe_output + mapped_slot * n_channels + channel_offset;
    T* layer_output_base = layer_output + token_idx * n_channels + channel_offset;

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T reg_buffer[vector_size];

        if (i * stride + channel_offset < n_channels) {
            if (mapped_slot != gating::unassigned && mapped_slot < num_tokens)
            {
                mem_access::load_global<scatter_gather::access_granularity>(
                    reg_buffer,
                    moe_output_base + i * stride
                );
#pragma unroll
                for (int j = 0; j < vector_size; j++) {
                    float up_cast = conversion::to<float>(reg_buffer[j]);
                    reg_buffer[j] = conversion::to<T>(up_cast * score);
                }
            }
            else{
#pragma unroll
                for (int j = 0; j < vector_size; j++) {
                    reg_buffer[j] = conversion::to<T>(0.f);
                }
            }

            mem_access::store_global<scatter_gather::access_granularity>(
                layer_output_base + i * stride,
                reg_buffer
            );
        }
    }
}

template <typename T, int copyUnroll>
__global__ void moe_gather_bwd_kernel(T* layer_output_grad,
                                  float* scores_grad,
                                  T* moe_output_grad,
                                  T* moe_output,
                                  const float* scores,
                                  const int32_t* mapped_slots,
                                  const int32_t n_channels,
                                  const int32_t num_tokens)
{
    constexpr int32_t vector_size = scatter_gather::access_granularity / sizeof(T);
    constexpr int32_t stride = vector_size * scatter_gather::threads;

    const int32_t token_idx = blockIdx.x;
    const int32_t mapped_slot = mapped_slots[token_idx];


    const float score = scores[token_idx];
    const int32_t channel_offset = threadIdx.x * vector_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    T* layer_output_grad_base = layer_output_grad + token_idx * n_channels + channel_offset;
    T* moe_output_grad_base = moe_output_grad + mapped_slot * n_channels + channel_offset;
    T* moe_output_base = moe_output + mapped_slot * n_channels + channel_offset;
    float score_grad = 0.f;
#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T reg_buffer[vector_size];
        T out_buffer[vector_size];

        if (i * stride + channel_offset < n_channels) {
            if (mapped_slot != gating::unassigned && mapped_slot < num_tokens)
            {
                mem_access::load_global<scatter_gather::access_granularity>(reg_buffer,
                                                                layer_output_grad_base + i * stride);

                mem_access::load_global<scatter_gather::access_granularity>(out_buffer,
                                                                moe_output_base + i * stride);

#pragma unroll
                for (int j = 0; j < vector_size; j++) {
                    float up_cast = conversion::to<float>(reg_buffer[j]);
                    float out_up_cast = conversion::to<float>(out_buffer[j]);
                    reg_buffer[j] = conversion::to<T>(up_cast * score);
                    score_grad += (up_cast * out_up_cast);
                }
                mem_access::store_global<scatter_gather::access_granularity>(moe_output_grad_base + i * stride,
                                                                    reg_buffer);
            }

        }
    }

    reduce::_block<float, scatter_gather::warps, ROp::Add>(tb, warp, &score_grad);
    if (threadIdx.x == 0) scores_grad[token_idx] = score_grad;
}

template <typename T, int copyUnroll>
__global__ void moe_scatter_bwd_kernel(T* moe_input_grad,
                                   T* activations_grad,
                                   const int32_t* assignments,
                                   const int32_t* offsets,
                                   const int32_t n_channels,
                                   const int capacity,
                                   const int32_t num_tokens,
                                   const int32_t n_experts)
{
    constexpr int32_t vector_size = scatter_gather::access_granularity / sizeof(T);
    constexpr int32_t load_stride = vector_size * scatter_gather::threads;

    const int32_t token_idx = blockIdx.x;
    const int32_t tidx = threadIdx.x;
    const int32_t warp_rank = tidx / hw_warp_size;

    // Bank aligned and sufficient
    __shared__ int32_t red_buffer[32];
    __shared__ int32_t token_0_row;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    int assigned_expert = assignments[token_idx];

    // For the different codepaths, we'll converge on this variable for doing
    // the token copy.
    int32_t token_base_row;

    
    token_base_row = capacity * assigned_expert; 

    // Data copy to appropriate location
    const int32_t thread_offset = tidx * vector_size;

    const int32_t base_load_offset = token_idx * n_channels + thread_offset;
    T* load_base_ptr = activations_grad + base_load_offset;
    int32_t offset = offsets[token_idx];
    const int32_t store_row = token_base_row + offset;
    const int32_t base_store_offset = store_row * n_channels + thread_offset;
    T* store_base_ptr = moe_input_grad + base_store_offset;

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T tmp_buf[vector_size];

        if (i * load_stride + thread_offset < n_channels) {
            if (assigned_expert < n_experts && offset != gating::unassigned && offset < capacity && store_row < num_tokens)
                mem_access::load_global<scatter_gather::access_granularity>(tmp_buf, store_base_ptr + i * load_stride);
            else
            {
                #pragma unroll
                for (int k = 0; k < vector_size; k++)
                   tmp_buf[k] = conversion::to<T>(0.f);
            }

            mem_access::store_global<scatter_gather::access_granularity>(load_base_ptr + i * load_stride, tmp_buf);
        }
    }
}

template <typename T, int copyUnroll>
__global__ void moe_scatter_kernel(T* moe_input,
                                   int32_t* expert_count_cumsums,
                                   int32_t* mapped_slots,
                                   float* scores,
                                   const T* activations,
                                   int32_t* assignments,
                                   const int32_t* expert_counts,
                                   const int32_t* mapped_expert_counts,
                                   int32_t* offsets,
                                   const int32_t* backup_offsets,
                                   const int32_t n_channels,
                                   const int32_t n_experts,
                                   const int capacity,
                                   const int32_t num_tokens)
{
    constexpr int32_t vector_size = scatter_gather::access_granularity / sizeof(T);
    constexpr int32_t load_stride = vector_size * scatter_gather::threads;

    const int32_t token_idx = blockIdx.x;
    const int32_t tidx = threadIdx.x;
    const int32_t warp_rank = tidx / hw_warp_size;

    // Bank aligned and sufficient
    __shared__ int32_t red_buffer[32];
    __shared__ int32_t token_0_row;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    int assigned_expert = assignments[token_idx];
    if (assigned_expert >= n_experts || assigned_expert == gating::unassigned) {
        // For whatever reason, don't need to perform the copy, so we'll early return
        // and signal this wasn't mapped with a negative 1.
        
        if (tidx == 0) mapped_slots[token_idx] = gating::unassigned;
        return;
    } 
    int expert_count = expert_counts[assigned_expert];
    int mapped_expert_count = mapped_expert_counts[assigned_expert];

    // For the different codepaths, we'll converge on this variable for doing
    // the token copy.
    int32_t token_base_row;
    int32_t offset = offsets[token_idx];
    int32_t other_offset = backup_offsets[token_idx];
    //if (offset == gating::unassigned && expert_count <= capacity) {
    //    if (tidx == 0) {
    //        assignments[token_idx] = n_experts;
    //        mapped_slots[token_idx] = gating::unassigned;
    //    }
    //    return;
    //}
    //else if (mapped_expert_count != capacity){
    //    offset = backup_offsets[token_idx] + mapped_expert_count;        
    //    if (tidx == 0) printf("Coming here: E(%d), T(%d), MEC(%d), O(%d)\n", assigned_expert, token_idx, mapped_expert_count, offset);
    //}
//
    if (offset == gating::unassigned && expert_count > capacity && mapped_expert_count < capacity)
    {
        offset = backup_offsets[token_idx] + mapped_expert_count;
        if (tidx == 0) offsets[token_idx] = offset;
    }
    // if (other_offset == 0)//>= capacity || offset == gating::unassigned)
    if (offset >= capacity || offset == gating::unassigned)
    {
       if (tidx == 0) {
           mapped_slots[token_idx] = gating::unassigned;
           assignments[token_idx] = n_experts; //gating::unassigned;
           scores[token_idx] = 0.f;
       }
       return;
    }
    //else{
    //    if (tidx == 0) offsets[token_idx] = offset;
    //}


    token_base_row = capacity * assigned_expert; 

    // Data copy to appropriate location
    const int32_t thread_offset = tidx * vector_size;

    const int32_t base_load_offset = token_idx * n_channels + thread_offset;
    const T* load_base_ptr = activations + base_load_offset;
    const int32_t store_row = token_base_row + offset;
    const int32_t base_store_offset = store_row * n_channels + thread_offset;
    T* store_base_ptr = moe_input + base_store_offset;
#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T tmp_buf[vector_size];

        if (i * load_stride + thread_offset < n_channels && store_row < num_tokens) {
            mem_access::load_global<scatter_gather::access_granularity>(tmp_buf,
                                                                 load_base_ptr + i * load_stride);
            mem_access::store_global<scatter_gather::access_granularity>(store_base_ptr + i * load_stride,
                                                                  tmp_buf);
        }
    }

    if (threadIdx.x == 0) { 
        mapped_slots[token_idx] = (store_row < num_tokens) ? store_row : gating::unassigned;
    }
}

#define LAUNCH_FOR_UNROLL(COUNT)                                                       \
    case COUNT:                                                                        \
        moe_scatter_kernel<T, COUNT><<<grid1, block1, 0, stream>>>(moe_input,            \
                                                                 expert_count_cumsums, \
                                                                 mapped_slots,         \
                                                                 scores,            \
                                                                 activations,          \
                                                                 assignments,          \
                                                                 expert_counts,        \
                                                                 mapped_expert_counts,  \
                                                                 offsets,              \
                                                                 backup_offsets,              \
                                                                 n_channels,           \
                                                                 n_experts,           \
                                                                 capacity,          \
                                                                 n_tokens);           \
        break;


template <typename T>
void launch_moe_gating(T* moe_input,
                        int32_t* expert_count_cumsums,
                        int32_t* mapped_slots,
                        const T* activations,
                        int32_t* expert_counts,
                        int32_t* mapped_expert_counts,
                        float* scores,
                        int32_t* assignments,
                        int32_t* offsets,
                        int32_t* backup_offsets,
                        float* logits,
                        float* logits_out,
                        const int capacity,
                        const int32_t n_tokens,
                        const int32_t n_channels,
                        const int32_t n_experts,
                        cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts - 1) / hw_warp_size + 1) * hw_warp_size);

    std::pair<uint64_t, uint64_t> seed = TOPSContext::Instance().IncrementOffset(16);

    top_1_gating_kernel<<<grid, block, 0, stream>>>(
        expert_counts, 
        scores, 
        assignments, 
        offsets, 
        logits, 
        logits_out,
        capacity, 
        n_experts, 
        n_tokens
    );
    const dim3 block2(scatter_gather::threads);
    const dim3 grid2((n_tokens - 1) / scatter_gather::threads + 1);

    refine_expert_mapping<T><<<grid2, block2, 0, stream>>>(
        expert_counts,
        mapped_expert_counts,
        expert_count_cumsums,
        scores,
        assignments,
        offsets,
        backup_offsets,
        capacity,
        n_tokens,
        n_experts,
        seed
    );

    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block1(scatter_gather::threads);
    const dim3 grid1(n_tokens);

    switch (copy_unroll) {
        LAUNCH_FOR_UNROLL(1);
        LAUNCH_FOR_UNROLL(2);
        LAUNCH_FOR_UNROLL(3);
        LAUNCH_FOR_UNROLL(4);
        LAUNCH_FOR_UNROLL(5);
        LAUNCH_FOR_UNROLL(6);
    }
}

#define INSTANTIATE_MoE_Gating_KERNEL(T)                                                   \
    template void launch_moe_gating<T>(T* moe_input,                                  \
                                        int32_t* expert_count_cumsums,                  \
                                        int32_t* mapped_slots,                          \
                                        const T* activations,                        \
                                        int32_t * expert_counts,                     \
                                        int32_t * mapped_expert_counts,                     \
                                        float* scores,                               \
                                        int32_t* assignments,                        \
                                        int32_t* offsets,                            \
                                        int32_t* backup_offsets,                            \
                                        float* logits,                             \
                                        float* logits_out,                             \
                                        const int capacity,                          \
                                        const int32_t n_tokens,                      \
                                        const int32_t n_channels,                    \
                                        const int32_t n_experts,                     \
                                        cudaStream_t stream);

INSTANTIATE_MoE_Gating_KERNEL(float)
INSTANTIATE_MoE_Gating_KERNEL(__half)
#ifdef BF16_AVAILABLE
INSTANTIATE_MoE_Gating_KERNEL(__nv_bfloat16)
#endif


template <typename T>
void launch_moe_scatter(T* moe_input,
                        int32_t* expert_count_cumsums,
                        int32_t* mapped_slots,
                        const T* activations,
                        int32_t* expert_counts,
                        int32_t* mapped_expert_counts,
                        float* scores,
                        int32_t* assignments,
                        int32_t* offsets,
                        int32_t* backup_offsets,
                        const int capacity,
                        const int32_t n_tokens,
                        const int32_t n_channels,
                        const int32_t n_experts,
                        cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block1(scatter_gather::threads);
    const dim3 grid1(n_tokens);

    switch (copy_unroll) {
        LAUNCH_FOR_UNROLL(1);
        LAUNCH_FOR_UNROLL(2);
        LAUNCH_FOR_UNROLL(3);
        LAUNCH_FOR_UNROLL(4);
        LAUNCH_FOR_UNROLL(5);
        LAUNCH_FOR_UNROLL(6);
    }
}

#define INSTANTIATE_MoE_SCATTER_KERNEL(T)                                                   \
    template void launch_moe_scatter<T>(T* moe_input,                                  \
                                        int32_t* expert_count_cumsums,                  \
                                        int32_t* mapped_slots,                          \
                                        const T* activations,                        \
                                        int32_t * expert_counts,                     \
                                        int32_t * mapped_expert_counts,                     \
                                        float* scores,                               \
                                        int32_t* assignments,                        \
                                        int32_t* offsets,                            \
                                        int32_t* backup_offsets,                     \
                                        const int capacity,                          \
                                        const int32_t n_tokens,                      \
                                        const int32_t n_channels,                    \
                                        const int32_t n_experts,                     \
                                        cudaStream_t stream);

INSTANTIATE_MoE_SCATTER_KERNEL(float)
INSTANTIATE_MoE_SCATTER_KERNEL(__half)
#ifdef BF16_AVAILABLE
INSTANTIATE_MoE_SCATTER_KERNEL(__nv_bfloat16)
#endif


#define LAUNCH_FOR_UNROLL_GATHER(COUNT)                                                                   \
    case COUNT:                                                                               \
            moe_gather_kernel<T, COUNT><<<grid, block, 0, stream>>>(layer_output,      \
                                                                       moe_output,        \
                                                                       scores,            \
                                                                       mapped_slots,      \
                                                                       n_channels,        \
                                                                       n_tokens);  \
        break;

template <typename T>
void launch_moe_gather(T* layer_output,
                       const T* moe_output,
                       const float* scores,
                       const int32_t* mapped_slots,
                       const int32_t n_channels,
                       const int32_t n_tokens,
                       cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(scatter_gather::threads);
    const dim3 grid(n_tokens);

    switch (copy_unroll) {
        LAUNCH_FOR_UNROLL_GATHER(1)
        LAUNCH_FOR_UNROLL_GATHER(2)
        LAUNCH_FOR_UNROLL_GATHER(3)
        LAUNCH_FOR_UNROLL_GATHER(4)
        LAUNCH_FOR_UNROLL_GATHER(5)
        LAUNCH_FOR_UNROLL_GATHER(6)
    }
}

#define INSTANTIATE_GATHER_FOR_TYPE(TYPE)                              \
    template void launch_moe_gather<TYPE>(TYPE * layer_output,         \
                                          const TYPE* moe_output,      \
                                          const float* scores,         \
                                          const int32_t* mapped_slots, \
                                          const int32_t n_channels,    \
                                          const int32_t n_tokens,      \
                                          cudaStream_t stream);         \

INSTANTIATE_GATHER_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_GATHER_FOR_TYPE(__nv_bfloat16)
#endif


#define LAUNCH_FOR_UNROLL_GATHER_BWD(COUNT)                                                                   \
    case COUNT:                                                                               \
            moe_gather_bwd_kernel<T, COUNT><<<grid, block, 0, stream>>>(layer_output_grad,      \
                                                                       scores_grad,        \
                                                                       moe_output_grad,        \
                                                                       moe_output,        \
                                                                       scores,            \
                                                                       mapped_slots,      \
                                                                       n_channels,      \
                                                                       n_tokens);  \
        break;

template <typename T>
void launch_moe_gather_bwd(T* layer_output_grad,
                        float* scores_grad,
                        T* moe_output_grad,
                        T* moe_output,
                        const float* scores,
                        const int32_t* mapped_slots,
                        const int32_t n_channels,
                        const int32_t n_tokens,
                        cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(scatter_gather::threads);
    const dim3 grid(n_tokens);

    switch (copy_unroll) {
        LAUNCH_FOR_UNROLL_GATHER_BWD(1)
        LAUNCH_FOR_UNROLL_GATHER_BWD(2)
        LAUNCH_FOR_UNROLL_GATHER_BWD(3)
        LAUNCH_FOR_UNROLL_GATHER_BWD(4)
        LAUNCH_FOR_UNROLL_GATHER_BWD(5)
        LAUNCH_FOR_UNROLL_GATHER_BWD(6)
    }
}

#define INSTANTIATE_GATHER_BWD_FOR_TYPE(TYPE)                              \
    template void launch_moe_gather_bwd<TYPE>(TYPE * layer_output_grad,         \
                                          float* scores_grad,      \
                                          TYPE* moe_output_grad,      \
                                          TYPE* moe_output,      \
                                          const float* scores,         \
                                          const int32_t* mapped_slots, \
                                          const int32_t n_channels,    \
                                          const int32_t n_tokens,      \
                                          cudaStream_t stream);

INSTANTIATE_GATHER_BWD_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_GATHER_BWD_FOR_TYPE(__nv_bfloat16)
#endif


#define LAUNCH_FOR_UNROLL_MOE_BWD(COUNT)                                                                   \
    case COUNT:                                                                               \
            moe_scatter_bwd_kernel<T, COUNT><<<grid1, block1, 0, stream>>>(moe_input_grad,      \
                                                                       activations_grad,        \
                                                                       assignments,            \
                                                                       offsets,      \
                                                                       n_channels,  \
                                                                       capacity,    \
                                                                       n_tokens,    \
                                                                       n_experts);  \
        break;

template <typename T>
void launch_moe_gating_bwd(T* moe_input_grad,
                        float* scores_grad,
                        T* activations_grad,
                        float* logits_grad,
                        float* logits,
                        const int32_t* assignments,
                        const int32_t* offsets,
                        const int32_t n_channels,
                        const int32_t n_experts,
                        const int32_t n_tokens,
                        int capacity,
                        cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts - 1) / hw_warp_size + 1) * hw_warp_size);

    gate_logits_bwd_kernel<<<grid, block, 0, stream>>> (logits_grad, scores_grad, assignments, logits, n_experts, n_tokens);

    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block1(scatter_gather::threads);
    const dim3 grid1(n_tokens);

    switch (copy_unroll) {
       LAUNCH_FOR_UNROLL_MOE_BWD(1);
       LAUNCH_FOR_UNROLL_MOE_BWD(2);
       LAUNCH_FOR_UNROLL_MOE_BWD(3);
       LAUNCH_FOR_UNROLL_MOE_BWD(4);
       LAUNCH_FOR_UNROLL_MOE_BWD(5);
       LAUNCH_FOR_UNROLL_MOE_BWD(6);
    }
}

#define INSTANTIATE_MOE_GATING_BWD_FOR_TYPE(TYPE)                              \
    template void launch_moe_gating_bwd<TYPE>(TYPE * moe_input_grad,         \
                                          float* scores_grad,           \
                                          TYPE* activations_grad,      \
                                          float* logits_grad,      \
                                          float* logits,      \
                                          const int32_t* assignments,      \
                                          const int32_t* offsets,      \
                                          const int32_t n_channels,    \
                                          const int32_t n_experts,     \
                                          const int32_t n_tokens,       \
                                          int capacity,                 \
                                          cudaStream_t stream);         \

INSTANTIATE_MOE_GATING_BWD_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_MOE_GATING_BWD_FOR_TYPE(__nv_bfloat16)
#endif
