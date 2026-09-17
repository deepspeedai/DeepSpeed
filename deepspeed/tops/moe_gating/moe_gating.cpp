#include "moe_gating.h"

#include <c10/cuda/CUDAStream.h>

#define DISPATCH_MOE_GATING(T_TYPE, C_TYPE)                          \
    if (activations.options().dtype() == torch::T_TYPE) {             \
        if (top_k == 1)                                                 \
            launch_moe_gating((C_TYPE*)moe_input.data_ptr(),             \
                           (int32_t*)expert_count_cumsums.data_ptr(), \
                           (int32_t*)mapped_slots.data_ptr(),         \
                           (const C_TYPE*)activations.data_ptr(),     \
                           (int32_t*)expert_counts.data_ptr(),  \
                           (int32_t*)mapped_expert_counts.data_ptr(),  \
                           (float*)scores.data_ptr(),                 \
                           (int32_t*)assignments.data_ptr(),    \
                           (int32_t*)offsets.data_ptr(),        \
                           (int32_t*)backup_offsets.data_ptr(),        \
                           (float*)logits.data_ptr(),              \
                           (float*)logits_out.data_ptr(),              \
                           capacity,                                  \
                           n_tokens,                                  \
                           n_channels,                                \
                           n_experts,                                 \
                           at::cuda::getCurrentCUDAStream());         \
        else                                                            \
            launch_top2_moe_gating((C_TYPE*)moe_input.data_ptr(),             \
                           (int32_t*)expert_count_cumsums.data_ptr(), \
                           (int32_t*)mapped_slots.data_ptr(),         \
                           (const C_TYPE*)activations.data_ptr(),     \
                           (int32_t*)expert_counts.data_ptr(),  \
                           (int32_t*)mapped_expert_counts.data_ptr(),  \
                           (float*)scores.data_ptr(),                 \
                           (int32_t*)assignments.data_ptr(),    \
                           (int32_t*)offsets.data_ptr(),        \
                           (int32_t*)backup_offsets.data_ptr(),        \
                           (float*)logits.data_ptr(),              \
                           (float*)logits_out.data_ptr(),              \
                           capacity,                                  \
                           n_tokens,                                  \
                           n_channels,                                \
                           n_experts,                                 \
                           top_k,                                     \
                           at::cuda::getCurrentCUDAStream());         \
        return;                                                       \
    }


void gate_fwd(torch::Tensor& moe_input,
                torch::Tensor& expert_count_cumsums,
                torch::Tensor& mapped_slots,
                torch::Tensor& activations,
                torch::Tensor& expert_counts,
                torch::Tensor& mapped_expert_counts,
                torch::Tensor& scores,
                torch::Tensor& assignments,
                torch::Tensor& offsets,
                torch::Tensor& backup_offsets,
                torch::Tensor& logits,
                torch::Tensor& logits_out,
                int top_k,
                int capacity,
                bool use_rts)
{
    const int32_t n_tokens = activations.size(0);
    const int32_t n_channels = activations.size(1);

    const int32_t n_experts = expert_count_cumsums.size(0) / top_k;

    DISPATCH_MOE_GATING(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_MOE_GATING(kBFloat16, __nv_bfloat16);
#endif

}


#define DISPATCH_MOE_SCATTER(T_TYPE, C_TYPE)                          \
    if (activations.options().dtype() == torch::T_TYPE) {             \
        if (top_k == 1)                                                 \
            launch_moe_scatter((C_TYPE*)moe_input.data_ptr(),             \
                           (int32_t*)expert_count_cumsums.data_ptr(), \
                           (int32_t*)mapped_slots.data_ptr(),         \
                           (const C_TYPE*)activations.data_ptr(),     \
                           (int32_t*)expert_counts.data_ptr(),  \
                           (int32_t*)mapped_expert_counts.data_ptr(),  \
                           (float*)scores.data_ptr(),                 \
                           (int32_t*)assignments.data_ptr(),    \
                           (int32_t*)offsets.data_ptr(),        \
                           (int32_t*)backup_offsets.data_ptr(),        \
                           capacity,                                  \
                           n_tokens,                                  \
                           n_channels,                                \
                           n_experts,                                 \
                           at::cuda::getCurrentCUDAStream());         \
        else                                                            \
            launch_top2_moe_scatter((C_TYPE*)moe_input.data_ptr(),             \
                           (int32_t*)expert_count_cumsums.data_ptr(), \
                           (int32_t*)mapped_slots.data_ptr(),         \
                           (const C_TYPE*)activations.data_ptr(),     \
                           (int32_t*)expert_counts.data_ptr(),  \
                           (int32_t*)mapped_expert_counts.data_ptr(),  \
                           (float*)scores.data_ptr(),                 \
                           (int32_t*)assignments.data_ptr(),    \
                           (int32_t*)offsets.data_ptr(),        \
                           (int32_t*)backup_offsets.data_ptr(),        \
                           capacity,                                  \
                           n_tokens,                                  \
                           n_channels,                                \
                           n_experts,                                 \
                           top_k,                                     \
                           at::cuda::getCurrentCUDAStream());         \
        return;                                                       \
    }


void gate_scatter(torch::Tensor& moe_input,
                torch::Tensor& expert_count_cumsums,
                torch::Tensor& mapped_slots,
                torch::Tensor& activations,
                torch::Tensor& expert_counts,
                torch::Tensor& mapped_expert_counts,
                torch::Tensor& scores,
                torch::Tensor& assignments,
                torch::Tensor& offsets,
                torch::Tensor& backup_offsets,
                int top_k,
                int capacity,
                bool use_rts)
{
    const int32_t n_tokens = activations.size(0);
    const int32_t n_channels = activations.size(1);

    const int32_t n_experts = expert_count_cumsums.size(0) / top_k;

    DISPATCH_MOE_SCATTER(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_MOE_SCATTER(kBFloat16, __nv_bfloat16);
#endif

}

#define DISPATCH_MOE_GATING_BWD(T_TYPE, C_TYPE)                          \
    if (moe_input_grad.options().dtype() == torch::T_TYPE) {             \
        if (top_k == 1)                                                 \
            launch_moe_gating_bwd((C_TYPE*)moe_input_grad.data_ptr(),             \
                           (float*)scores_grad.data_ptr(),     \
                           (C_TYPE*)activations_grad.data_ptr(),     \
                           (float*)logits_grad.data_ptr(),     \
                           (float*)logits.data_ptr(),     \
                           (int32_t*)assignments.data_ptr(),    \
                           (int32_t*)offsets.data_ptr(),        \
                           n_channels,                                \
                           n_experts,                                 \
                           n_tokens,                                  \
                           capacity,                                  \
                           at::cuda::getCurrentCUDAStream());         \
        else                                                           \
            launch_top2_moe_gating_bwd((C_TYPE*)moe_input_grad.data_ptr(),             \
                           (float*)scores_grad.data_ptr(),     \
                           (C_TYPE*)activations_grad.data_ptr(),     \
                           (float*)logits_grad.data_ptr(),     \
                           (float*)logits.data_ptr(),     \
                           (int32_t*)assignments.data_ptr(),    \
                           (int32_t*)mapped_slots.data_ptr(),        \
                           n_channels,                                \
                           n_experts,                                 \
                           n_tokens,                                  \
                           capacity,                                  \
                           top_k,                                     \
                           at::cuda::getCurrentCUDAStream());         \
        return;                                                       \
    }

void gate_bwd(torch::Tensor& moe_input_grad,
                torch::Tensor& scores_grad,
                torch::Tensor& activations_grad,
                torch::Tensor& logits_grad,
                torch::Tensor& logits,
                torch::Tensor& assignments,
                torch::Tensor& offsets,
                torch::Tensor& mapped_slots,
                int top_k,
                int capacity,
                bool use_rts)
{
    const int32_t n_tokens = scores_grad.size(0) / top_k;
    const int32_t n_channels = moe_input_grad.size(1);

    const int32_t n_experts = logits.size(1);
    DISPATCH_MOE_GATING_BWD(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_MOE_GATING_BWD(kBFloat16, __nv_bfloat16);
#endif

}



#define DISPATCH_GATHER(T_TYPE, C_TYPE)                          \
    if (layer_output.options().dtype() == torch::T_TYPE) {             \
        if (top_k == 1)                                             \
            launch_moe_gather((C_TYPE*)layer_output.data_ptr(),             \
                           (const C_TYPE*)moe_output.data_ptr(),             \
                           (const float*)scores.data_ptr(),                 \
                           (const int32_t*)mapped_slots.data_ptr(),         \
                           n_channels,                                \
                           n_tokens,                                  \
                           at::cuda::getCurrentCUDAStream());         \
        else                                                            \
            launch_top2_moe_gather((C_TYPE*)layer_output.data_ptr(),             \
                           (const C_TYPE*)moe_output.data_ptr(),             \
                           (const float*)scores.data_ptr(),                 \
                           (const int32_t*)mapped_slots.data_ptr(),         \
                           n_channels,                                \
                           n_tokens,                                  \
                           top_k,                                     \
                           at::cuda::getCurrentCUDAStream());         \
        return;                                                       \
    }

void gather_fwd(torch::Tensor& layer_output,
                torch::Tensor& moe_output,
                torch::Tensor& scores,
                torch::Tensor& mapped_slots,
                int top_k)
{
    const int32_t n_tokens = layer_output.size(0);
    const int32_t n_channels = layer_output.size(1);

    DISPATCH_GATHER(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_GATHER(kBFloat16, __nv_bfloat16);
#endif

}

#define DISPATCH_GATHER_BWD(T_TYPE, C_TYPE)                          \
    if (layer_output_grad.options().dtype() == torch::T_TYPE) {             \
        if (top_k == 1)                                                     \
            launch_moe_gather_bwd((C_TYPE*)layer_output_grad.data_ptr(),             \
                           (float*)scores_grad.data_ptr(),     \
                           (C_TYPE*)moe_output_grad.data_ptr(),     \
                           (C_TYPE*)moe_output.data_ptr(),     \
                           (const float*)scores.data_ptr(),    \
                           (const int32_t*)mapped_slots.data_ptr(),        \
                           n_channels,                                \
                           n_tokens,                                  \
                           at::cuda::getCurrentCUDAStream());         \
        else                                                            \
            launch_top2_moe_gather_bwd((C_TYPE*)layer_output_grad.data_ptr(),             \
                           (float*)scores_grad.data_ptr(),     \
                           (C_TYPE*)moe_output_grad.data_ptr(),     \
                           (C_TYPE*)moe_output.data_ptr(),     \
                           (const float*)scores.data_ptr(),    \
                           (const int32_t*)mapped_slots.data_ptr(),        \
                           n_channels,                                \
                           n_tokens,                                  \
                           top_k,                                     \
                           at::cuda::getCurrentCUDAStream());         \
        return;                                                       \
    }


void gather_bwd(torch::Tensor& layer_output_grad,
                torch::Tensor& scores_grad,
                torch::Tensor& moe_output_grad,
                torch::Tensor& moe_output,
                torch::Tensor& scores,
                torch::Tensor& mapped_slots,
                int top_k)
{
    const int32_t n_tokens = layer_output_grad.size(0);
    const int32_t n_channels = layer_output_grad.size(1);

    DISPATCH_GATHER_BWD(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_GATHER_BWD(kBFloat16, __nv_bfloat16);
#endif

}
