#include "rope.h"
#include <c10/cuda/CUDAStream.h>

#define DISPATCH_ROPE(T_TYPE, C_TYPE)                \
    if (query.options().dtype() == torch::T_TYPE) {          \
        launch_apply_rotary_pos_emb((C_TYPE*)query.data_ptr(),             \
                      (C_TYPE*)key.data_ptr(),             \
                       head_size,                                \
                       seq_len,                        \
                       rotary_dim,                      \
                       offset,                      \
                       num_heads,                           \
                       batch,                           \
                       rope_theta,                          \
                       at::cuda::getCurrentCUDAStream());  \
        return;                                        \
    }

void rope_fwd(torch::Tensor& query, torch::Tensor& key, int rotary_dim, float rope_theta)
{
    int seq_len = query.size(0);
    int batch = query.size(1);
    int num_heads = query.size(2);
    int head_size = query.size(3);
    int offset = 0;

    DISPATCH_ROPE(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_ROPE(kBFloat16, __nv_bfloat16);
#endif

}

#define DISPATCH_ROPE_BWD(T_TYPE, C_TYPE)                \
    if (query_grad.options().dtype() == torch::T_TYPE) {          \
        launch_apply_rotary_pos_bwd_emb((C_TYPE*)query_grad.data_ptr(),             \
                      (C_TYPE*)key_grad.data_ptr(),             \
                       head_size,                                \
                       seq_len,                        \
                       rotary_dim,                      \
                       offset,                      \
                       num_heads,                           \
                       batch,                           \
                       rope_theta,                          \
                       at::cuda::getCurrentCUDAStream());  \
        return;                                        \
    }

void rope_bwd(torch::Tensor& query_grad, torch::Tensor& key_grad, int rotary_dim, float rope_theta)
{
    
    int seq_len = query_grad.size(0);
    int batch = query_grad.size(1);
    int num_heads = query_grad.size(2);
    int head_size = query_grad.size(3);
    int offset = 0;

    DISPATCH_ROPE_BWD(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_ROPE_BWD(kBFloat16, __nv_bfloat16);
#endif

}
