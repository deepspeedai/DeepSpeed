#include "moe_gating.cuh"
#include "reduction_utils.h"
#include "stdio.h"
#include "tops_context.h"

using ROp = reduce::ROpType;


template<int TOP_K>
__global__ void top_2_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const float* logits,
                                    float* logits_out,
                                    const int32_t n_experts,
                                    const int32_t n_tokens)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    const float* token_logits = logits + token_idx * n_experts;
    float* token_logits_out = logits_out + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = token_logits[expert_idx];
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;

    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val); }
    }

    const float max_logit = local_assigned_logits[0];
    float softlogit = __expf(logit_val - max_logit);
    float softmax_sum = softlogit;

    reduce::block<ROp::Add>(tb, warp, softmax_sum);
    if (expert_idx < n_experts)
        token_logits_out[expert_idx] = softlogit / softmax_sum;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < TOP_K; ++i) {
            scores[token_idx * TOP_K + i] = __expf(local_assigned_logits[i] - max_logit) / softmax_sum;
            assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
            atomicAdd(expert_counts + n_experts * i + local_assigned_experts[i], 1);
        }
    }
}

template <int TOP_K>
__global__ void refine_expert_mapping_for_top2(int32_t* expert_counts,
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

    int32_t assignment[TOP_K];
    #pragma unroll
    for (int i = 0; i < TOP_K; ++i)
        assignment[i] = assignments[(token_idx * TOP_K) + i];

    int idx = token_idx << (1 + TOP_K);
    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);


#pragma unroll
    for (int i = 0; i < TOP_K; i++) {    
        float4 rand_val = curand_uniform4(&state);

        int32_t total_ec = expert_counts[assignment[i] + n_experts * i];
        float ratio = 1.0 - (float)(capacity * (TOP_K - i)) / (float)total_ec;        
        {
            if ((rand_val.z > ratio && rand_val.x > ratio && rand_val.y > ratio) || 
                (rand_val.z <= ratio && (rand_val.x > ratio || rand_val.y > ratio))) {
                offsets[token_idx * TOP_K + i] = atomicAdd(mapped_expert_counts + n_experts * i + assignment[i], 1);
            }
            else{
                offsets[token_idx * TOP_K + i] = gating::unassigned;
                backup_offsets[token_idx * TOP_K + i] = atomicAdd(expert_count_cumsums + n_experts * i + assignment[i], 1);
            } 
        }
    }
}

template<int TOP_K>
__global__ void top2_gate_logits_bwd_kernel(float* logits_grad,
                                        float* scores_grad,
                                        const int32_t* assignment,
                                        float* logits,
                                        const int32_t n_experts,
                                        const int32_t n_tokens)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;

    int32_t assigned_expert[TOP_K];

#pragma unroll
    for (int i = 0; i < TOP_K; i++) 
        assigned_expert[i] = assignment[token_idx * TOP_K + i];
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
    

#pragma unroll
    for (int i = 0; i < TOP_K; i++) 
    {
        if (assigned_expert[i] == expert_idx) {
            logit_grad_val += scores_grad[token_idx * TOP_K + i];
        }
    }
    float softmax_grad_sum = logit_val * logit_grad_val;
    reduce::block<ROp::Add>(tb, warp, softmax_grad_sum);
    logit_grad_val = logit_val * (logit_grad_val - softmax_grad_sum);
    if (expert_idx < n_experts) 
        token_logits_grad[expert_idx] = logit_grad_val;
}

template <typename T, int copyUnroll, int TOP_K>
__global__ void moe_top2_gather_kernel(T* layer_output,
                                  const T* moe_output,
                                  const float* scores,
                                  const int32_t* mapped_slots,
                                  const int32_t n_channels,
                                  const int32_t num_tokens)
{
    constexpr int32_t vector_size = scatter_gather::access_granularity / sizeof(T);
    constexpr int32_t stride = vector_size * scatter_gather::threads;

    const int32_t token_idx = blockIdx.x;

    int32_t mapped_slot[TOP_K];
    float score[TOP_K];
        
        
    float sum = 0.0f;

#pragma unroll
    for (int i = 0; i < TOP_K; i++) mapped_slot[i] = mapped_slots[token_idx * TOP_K + i];
   
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        if (mapped_slot[i] != gating::unassigned && mapped_slot[i] < (num_tokens * TOP_K))
        {
            score[i] = scores[token_idx * TOP_K + i];
            sum += score[i];
        }
    }
    sum += 1.192092895e-07;

    const int32_t channel_offset = threadIdx.x * vector_size;

    T* layer_output_base = layer_output + token_idx * n_channels + channel_offset;

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        if (i * stride + channel_offset < n_channels) 
        {
            float accumulator[vector_size];
            if (mapped_slot[0] != gating::unassigned && mapped_slot[0] < (num_tokens * TOP_K))
            {
                T read_buf[vector_size];
                const T* moe_output_base = moe_output + mapped_slot[0] * n_channels + channel_offset;
                mem_access::load_global<scatter_gather::access_granularity>(
                    read_buf,
                    moe_output_base + i * stride
                );
#pragma unroll
                for (int j = 0; j < vector_size; j++) {
                    float up_cast = conversion::to<float>(read_buf[j]);
                    accumulator[j] = up_cast * (score[0] / sum);
                }
            }
            else{
#pragma unroll
                for (int j = 0; j < vector_size; j++) {
                    accumulator[j] = 0.f;
                }
            }
#pragma unroll
            for (int k = 1; k < TOP_K; k++) {
                const T* moe_output_base = moe_output + mapped_slot[k] * n_channels + channel_offset;
            
                if (mapped_slot[k] != gating::unassigned && mapped_slot[k] < (num_tokens * TOP_K))
                {
                    T read_buf[vector_size];
                    mem_access::load_global<scatter_gather::access_granularity>(
                        read_buf,
                        moe_output_base + i * stride
                    );
#pragma unroll
                    for (int j = 0; j < vector_size; j++) {
                        float up_cast = conversion::to<float>(read_buf[j]);
                        accumulator[j] += up_cast * (score[k] / sum);
                    }
                }
            }

            T reg_buffer[vector_size];
#pragma unroll
            for (int j = 0; j < vector_size; j++) reg_buffer[j] = conversion::to<T>(accumulator[j]);

            mem_access::store_global<scatter_gather::access_granularity>(
                layer_output_base + i * stride,
                reg_buffer
            );
        }
    }
}

template <typename T, int copyUnroll, int TOP_K>
__global__ void moe_top2_gather_bwd_kernel(T* layer_output_grad,
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

    int32_t mapped_slot[TOP_K];
    float score[TOP_K];

      
    float sum = 0.0f;

#pragma unroll
    for (int i = 0; i < TOP_K; i++) mapped_slot[i] = mapped_slots[token_idx * TOP_K + i];
   
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = (mapped_slot[i] != gating::unassigned && mapped_slot[i] < (num_tokens * TOP_K)) ? scores[token_idx * TOP_K + i] : 0.f;
        sum += score[i];
    }
    sum += 1.192092895e-07;

    const int32_t channel_offset = threadIdx.x * vector_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    T* layer_output_grad_base = layer_output_grad + token_idx * n_channels + channel_offset;
    // float score_grad[TOP_K];
    float score_out_grad[TOP_K];

#pragma unroll
    for (int j = 0; j < TOP_K; j++) {
        // score_grad[j] = 0.f;
        score_out_grad[j] = 0.f;
    }
    
#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {

        if (i * stride + channel_offset < n_channels) 
        {
            float reg_buffer[vector_size];
            {
                T read_buf[vector_size];
                mem_access::load_global<scatter_gather::access_granularity>(
                    read_buf, layer_output_grad_base + i * stride);
#pragma unroll
                for (int j = 0; j < vector_size; j++) reg_buffer[j] = conversion::to<float>(read_buf[j]);
            }

#pragma unroll
            for (int k = 0; k < TOP_K; k++) {
                T store_buffer[vector_size];
                if (mapped_slot[k] != gating::unassigned && mapped_slot[k] < (num_tokens * TOP_K))
                { 
                    T out_buffer[vector_size];
                    T* moe_output_base = moe_output + mapped_slot[k] * n_channels + channel_offset;
                    T* moe_output_grad_base = moe_output_grad + mapped_slot[k] * n_channels + channel_offset;
                    mem_access::load_global<scatter_gather::access_granularity>(
                        out_buffer, moe_output_base + i * stride
                    );

#pragma unroll
                    for (int j = 0; j < vector_size; j++) {
                        float out_up_cast = conversion::to<float>(out_buffer[j]);
                        store_buffer[j] = conversion::to<T>(reg_buffer[j] * (score[k] / sum));
                        for (int m = 0;m < TOP_K;m++)
                            score_out_grad[m] += (float)((double)(reg_buffer[j] * out_up_cast * 
                                                                    (m == k ? (sum - score[k]) : (-score[m]))) / (double)(sum * sum));
                    }
                    mem_access::store_global<scatter_gather::access_granularity>(
                        moe_output_grad_base + i * stride, store_buffer
                    );
                }
            }
        }
    }

    for (int j = 0; j < TOP_K; j++) 
        reduce::_block<float, scatter_gather::warps, ROp::Add>(tb, warp, score_out_grad + j);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < TOP_K; j++) 
        {
            scores_grad[token_idx * TOP_K + j] = (float)score_out_grad[j];
        }
    }
}

template <typename T, int copyUnroll, int TOP_K>
__global__ void moe_top2_scatter_bwd_kernel(T* moe_input_grad,
                                   T* activations_grad,
                                   const int32_t* assignments,
                                   const int32_t* mapped_slots,
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

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    int assigned_expert[TOP_K];

#pragma unroll
    for (int i = 0; i < TOP_K; i++)
        assigned_expert[i] = assignments[token_idx * TOP_K + i];


    // Data copy to appropriate location
    const int32_t thread_offset = tidx * vector_size;

    const int32_t base_load_offset = token_idx * n_channels + thread_offset;
    T* load_base_ptr = activations_grad + base_load_offset;
    int32_t store_row[TOP_K];
    
#pragma unroll
    for (int i = 0; i < TOP_K; i++)
        store_row[i] = mapped_slots[token_idx * TOP_K + i];

#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {

        float tmp_buf[vector_size];

        if (i * load_stride + thread_offset < n_channels) {
            if (assigned_expert[0] < n_experts && store_row[0] != gating::unassigned)
            {
                int32_t base_store_offset = store_row[0] * n_channels + thread_offset;
                T* store_base_ptr = moe_input_grad + base_store_offset;
                T reg_buffer[vector_size];
                mem_access::load_global<scatter_gather::access_granularity>(reg_buffer, store_base_ptr + i * load_stride);
#pragma unroll
                for (int k = 0; k < vector_size; k++)
                    tmp_buf[k] = conversion::to<float>(reg_buffer[k]);
            }
            else
            {
#pragma unroll
                for (int k = 0; k < vector_size; k++)
                    tmp_buf[k] = 0.f;
            }
#pragma unroll
            for (int k = 1; k < TOP_K; k++)
            {
                if (assigned_expert[k] < n_experts && store_row[k] != gating::unassigned){
                    T reg_buffer[vector_size];
                    const int32_t base_store_offset = store_row[k] * n_channels + thread_offset;
                    T* store_base_ptr = moe_input_grad + base_store_offset;
                    mem_access::load_global<scatter_gather::access_granularity>(reg_buffer, store_base_ptr + i * load_stride);

                    #pragma unroll
                    for (int j = 0; j < vector_size; j++)
                        tmp_buf[j] += conversion::to<float>(reg_buffer[j]);
                }
            }
            T store_buf[vector_size];

#pragma unroll
            for (int k = 0; k < vector_size; k++)
                store_buf[k] = conversion::to<T>(tmp_buf[k]);
            mem_access::store_global<scatter_gather::access_granularity>(load_base_ptr + i * load_stride, store_buf);
        }
    }
}

template <typename T, int copyUnroll, int TOP_K>
__global__ void moe_top2_scatter_kernel(T* moe_input,
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

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    int assigned_expert[TOP_K];
    int32_t offset[TOP_K];
    int mapped_expert_count[TOP_K];
    int expert_count[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {

        assigned_expert[i] = assignments[token_idx * TOP_K + i];

        // expert_count[i] = expert_counts[assigned_expert[i] + n_experts * i];
        // mapped_expert_count[i] = mapped_expert_counts[assigned_expert[i] + n_experts * i];

        offset[i] = offsets[token_idx + i * num_tokens];
        
        // if (offset[i] == gating::unassigned && expert_count[i] > (capacity * TOP_K) && mapped_expert_count[i] < (capacity * TOP_K))
        // {
        //     offset[i] = backup_offsets[token_idx * TOP_K + i] + mapped_expert_count[i];
        //     // if (tidx == 0) offsets[token_idx * TOP_K + i] = offset[i] + offsets_offset;
        // }
        
        // if (offset[i] != gating::unassigned) {
        //     int32_t offsets_offset = 0;
        //     for (int j = 0; j < i; j++) offsets_offset += expert_count[j];
        //     offset[i] += offsets_offset;
        // }
        int32_t other_offset = backup_offsets[token_idx + i * num_tokens];
        
        if (other_offset == 0) //if (offset[i] >= (capacity * TOP_K) || offset[i] == gating::unassigned)
        {
            if (tidx == 0) {
                mapped_slots[token_idx * TOP_K + i] = gating::unassigned;
                assignments[token_idx * TOP_K + i] = n_experts; //gating::unassigned;
                offset[i] = gating::unassigned;
                scores[token_idx * TOP_K + i] = 0.f;
            }
            assigned_expert[i] = n_experts;
        }
    }

    // Data copy to appropriate location
    const int32_t thread_offset = tidx * vector_size;

    const int32_t base_load_offset = token_idx * n_channels + thread_offset;
    const T* load_base_ptr = activations + base_load_offset;
    T* store_base_ptr[TOP_K];

#pragma unroll
    for (int j = 0; j < TOP_K; j++) {
        if (assigned_expert[j] < n_experts && assigned_expert[j] != gating::unassigned && offset[j] != gating::unassigned && offset[j] < (capacity * TOP_K))
        {
            int32_t store_row = capacity * TOP_K * assigned_expert[j] + offset[j];
            int32_t base_store_offset = store_row * n_channels + thread_offset;
            store_base_ptr[j] = moe_input + base_store_offset;
            if (threadIdx.x == 0) { 
                mapped_slots[token_idx * TOP_K + j] = store_row; 
            }
        }
        else{
            store_base_ptr[j] = nullptr;
        }
    }
#pragma unroll
    for (int i = 0; i < copyUnroll; i++) {
        T tmp_buf[vector_size];
        if ((i * load_stride + thread_offset) < n_channels) 
        {
            mem_access::load_global<scatter_gather::access_granularity>(
                tmp_buf, load_base_ptr + i * load_stride);
#pragma unroll
            for (int j = 0; j < TOP_K; j++) {
                if (store_base_ptr[j] != nullptr)
                   mem_access::store_global<scatter_gather::access_granularity>(
                       store_base_ptr[j] + i * load_stride, tmp_buf);
        
            }
        }
    }
}

#define LAUNCH_TOP2_FOR_UNROLL(COUNT)                                                       \
    case COUNT:                                                                        \
        moe_top2_scatter_kernel<T, COUNT, CONST_TOP_K><<<grid1, block1, 0, stream>>>(moe_input,            \
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
void launch_top2_moe_gating(T* moe_input,
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
                        const int32_t n_top_k,
                        cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts - 1) / hw_warp_size + 1) * hw_warp_size);

    std::pair<uint64_t, uint64_t> seed = TOPSContext::Instance().IncrementOffset(16);

    TOP_K_SWITCH(n_top_k, [&] {    
        top_2_gating_kernel<CONST_TOP_K><<<grid, block, 0, stream>>>(
            expert_counts, 
            scores, 
            assignments, 
            offsets, 
            logits, 
            logits_out,
            n_experts, 
            n_tokens
        );
    });
    const dim3 block2(scatter_gather::threads);
    const dim3 grid2((n_tokens - 1) / scatter_gather::threads + 1);

    // TOP_K_SWITCH(n_top_k, [&] {    
    //     refine_expert_mapping_for_top2<CONST_TOP_K><<<grid2, block2, 0, stream>>>(
    //         expert_counts,
    //         mapped_expert_counts,
    //         expert_count_cumsums,
    //         scores,
    //         assignments,
    //         offsets,
    //         backup_offsets,
    //         capacity,
    //         n_tokens,
    //         n_experts,
    //         seed
    //     );
    // });
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block1(scatter_gather::threads);
    const dim3 grid1(n_tokens);

    TOP_K_SWITCH(n_top_k, [&] {    
        switch (copy_unroll) {
            LAUNCH_TOP2_FOR_UNROLL(1);
            LAUNCH_TOP2_FOR_UNROLL(2);
            LAUNCH_TOP2_FOR_UNROLL(3);
            LAUNCH_TOP2_FOR_UNROLL(4);
            LAUNCH_TOP2_FOR_UNROLL(5);
            LAUNCH_TOP2_FOR_UNROLL(6);
        }
    });
}

#define INSTANTIATE_TOP2_MoE_Gating_KERNEL(T)                                                   \
    template void launch_top2_moe_gating<T>(T* moe_input,                                  \
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
                                        const int32_t n_top_k,                      \
                                        cudaStream_t stream);

INSTANTIATE_TOP2_MoE_Gating_KERNEL(float)
INSTANTIATE_TOP2_MoE_Gating_KERNEL(__half)
#ifdef BF16_AVAILABLE
INSTANTIATE_TOP2_MoE_Gating_KERNEL(__nv_bfloat16)
#endif


template <typename T>
void launch_top2_moe_scatter(T* moe_input,
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
                        const int32_t n_top_k,
                        cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block1(scatter_gather::threads);
    const dim3 grid1(n_tokens);

    TOP_K_SWITCH(n_top_k, [&] {    
        switch (copy_unroll) {
            LAUNCH_TOP2_FOR_UNROLL(1);
            LAUNCH_TOP2_FOR_UNROLL(2);
            LAUNCH_TOP2_FOR_UNROLL(3);
            LAUNCH_TOP2_FOR_UNROLL(4);
            LAUNCH_TOP2_FOR_UNROLL(5);
            LAUNCH_TOP2_FOR_UNROLL(6);
        }
    });
}

#define INSTANTIATE_TOP2_MoE_SCATTER_KERNEL(T)                                                   \
    template void launch_top2_moe_scatter<T>(T* moe_input,                                  \
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
                                        const int32_t n_top_k,                      \
                                        cudaStream_t stream);

INSTANTIATE_TOP2_MoE_SCATTER_KERNEL(float)
INSTANTIATE_TOP2_MoE_SCATTER_KERNEL(__half)
#ifdef BF16_AVAILABLE
INSTANTIATE_TOP2_MoE_SCATTER_KERNEL(__nv_bfloat16)
#endif



#define LAUNCH_FOR_UNROLL_GATHER_TOP2(COUNT)                                                                   \
    case COUNT:                                                                               \
            moe_top2_gather_kernel<T, COUNT, CONST_TOP_K><<<grid, block, 0, stream>>>(layer_output,      \
                                                                       moe_output,        \
                                                                       scores,            \
                                                                       mapped_slots,      \
                                                                       n_channels,        \
                                                                       n_tokens);  \
        break;

template <typename T>
void launch_top2_moe_gather(T* layer_output,
                       const T* moe_output,
                       const float* scores,
                       const int32_t* mapped_slots,
                       const int32_t n_channels,
                       const int32_t n_tokens,
                        const int32_t n_top_k,
                       cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(scatter_gather::threads);
    const dim3 grid(n_tokens);

    TOP_K_SWITCH(n_top_k, [&] {    
        switch (copy_unroll) {
            LAUNCH_FOR_UNROLL_GATHER_TOP2(1)
            LAUNCH_FOR_UNROLL_GATHER_TOP2(2)
            LAUNCH_FOR_UNROLL_GATHER_TOP2(3)
            LAUNCH_FOR_UNROLL_GATHER_TOP2(4)
            LAUNCH_FOR_UNROLL_GATHER_TOP2(5)
            LAUNCH_FOR_UNROLL_GATHER_TOP2(6)
        }
    });
}

#define INSTANTIATE_TOP2_GATHER_FOR_TYPE(TYPE)                              \
    template void launch_top2_moe_gather<TYPE>(TYPE * layer_output,         \
                                          const TYPE* moe_output,      \
                                          const float* scores,         \
                                          const int32_t* mapped_slots, \
                                          const int32_t n_channels,    \
                                          const int32_t n_tokens,      \
                                          const int32_t n_top_k,        \
                                          cudaStream_t stream);         \

INSTANTIATE_TOP2_GATHER_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_TOP2_GATHER_FOR_TYPE(__nv_bfloat16)
#endif


#define LAUNCH_FOR_UNROLL_GATHER_TOP2_BWD(COUNT)                                                                   \
    case COUNT:                                                                               \
            moe_top2_gather_bwd_kernel<T, COUNT, CONST_TOP_K><<<grid, block, 0, stream>>>(layer_output_grad,      \
                                                                       scores_grad,        \
                                                                       moe_output_grad,        \
                                                                       moe_output,        \
                                                                       scores,            \
                                                                       mapped_slots,      \
                                                                       n_channels,      \
                                                                       n_tokens);  \
        break;

template <typename T>
void launch_top2_moe_gather_bwd(T* layer_output_grad,
                        float* scores_grad,
                        T* moe_output_grad,
                        T* moe_output,
                        const float* scores,
                        const int32_t* mapped_slots,
                        const int32_t n_channels,
                        const int32_t n_tokens,
                        const int32_t n_top_k,
                        cudaStream_t stream)
{
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block(scatter_gather::threads);
    const dim3 grid(n_tokens);

    TOP_K_SWITCH(n_top_k, [&] {    
        switch (copy_unroll) {
            LAUNCH_FOR_UNROLL_GATHER_TOP2_BWD(1)
            LAUNCH_FOR_UNROLL_GATHER_TOP2_BWD(2)
            LAUNCH_FOR_UNROLL_GATHER_TOP2_BWD(3)
            LAUNCH_FOR_UNROLL_GATHER_TOP2_BWD(4)
            LAUNCH_FOR_UNROLL_GATHER_TOP2_BWD(5)
            LAUNCH_FOR_UNROLL_GATHER_TOP2_BWD(6)
        }
    });
}

#define INSTANTIATE_TOP2_GATHER_BWD_FOR_TYPE(TYPE)                              \
    template void launch_top2_moe_gather_bwd<TYPE>(TYPE * layer_output_grad,         \
                                          float* scores_grad,      \
                                          TYPE* moe_output_grad,      \
                                          TYPE* moe_output,      \
                                          const float* scores,         \
                                          const int32_t* mapped_slots, \
                                          const int32_t n_channels,    \
                                          const int32_t n_tokens,      \
                                          const int32_t n_top_k,        \
                                          cudaStream_t stream);

INSTANTIATE_TOP2_GATHER_BWD_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_TOP2_GATHER_BWD_FOR_TYPE(__nv_bfloat16)
#endif


#define LAUNCH_FOR_UNROLL_TOP2_MOE_BWD(COUNT)                                                                   \
    case COUNT:                                                                               \
            moe_top2_scatter_bwd_kernel<T, COUNT, CONST_TOP_K><<<grid1, block1, 0, stream>>>(moe_input_grad,      \
                                                                       activations_grad,        \
                                                                       assignments,            \
                                                                       mapped_slots,      \
                                                                       n_channels,  \
                                                                       capacity,    \
                                                                       n_tokens,    \
                                                                       n_experts);  \
        break;

template <typename T>
void launch_top2_moe_gating_bwd(T* moe_input_grad,
                        float* scores_grad,
                        T* activations_grad,
                        float* logits_grad,
                        float* logits,
                        const int32_t* assignments,
                        const int32_t* mapped_slots,
                        const int32_t n_channels,
                        const int32_t n_experts,
                        const int32_t n_tokens,
                        int capacity,
                        const int32_t n_top_k,
                        cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts - 1) / hw_warp_size + 1) * hw_warp_size);

    TOP_K_SWITCH(n_top_k, [&] {    
        top2_gate_logits_bwd_kernel<CONST_TOP_K><<<grid, block, 0, stream>>> 
            (logits_grad, scores_grad, assignments, logits, n_experts, n_tokens);
    });
    constexpr int vals_per_unroll = scatter_gather::threads * scatter_gather::access_granularity / sizeof(T);
    const int copy_unroll = (n_channels + vals_per_unroll - 1) / vals_per_unroll;

    const dim3 block1(scatter_gather::threads);
    const dim3 grid1(n_tokens);
    
    TOP_K_SWITCH(n_top_k, [&] {    
        switch (copy_unroll) {
            LAUNCH_FOR_UNROLL_TOP2_MOE_BWD(1);
            LAUNCH_FOR_UNROLL_TOP2_MOE_BWD(2);
            LAUNCH_FOR_UNROLL_TOP2_MOE_BWD(3);
            LAUNCH_FOR_UNROLL_TOP2_MOE_BWD(4);
            LAUNCH_FOR_UNROLL_TOP2_MOE_BWD(5);
            LAUNCH_FOR_UNROLL_TOP2_MOE_BWD(6);
        }
    });
}

#define INSTANTIATE_TOP2_MOE_GATING_BWD_FOR_TYPE(TYPE)                              \
    template void launch_top2_moe_gating_bwd<TYPE>(TYPE * moe_input_grad,         \
                                          float* scores_grad,           \
                                          TYPE* activations_grad,      \
                                          float* logits_grad,      \
                                          float* logits,      \
                                          const int32_t* assignments,      \
                                          const int32_t* mapped_slots,      \
                                          const int32_t n_channels,    \
                                          const int32_t n_experts,     \
                                          const int32_t n_tokens,       \
                                          int capacity,                 \
                                          const int32_t n_top_k,        \
                                          cudaStream_t stream);         \

INSTANTIATE_TOP2_MOE_GATING_BWD_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_TOP2_MOE_GATING_BWD_FOR_TYPE(__nv_bfloat16)
#endif
