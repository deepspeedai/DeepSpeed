// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include "inference_context.h"
#include "inference_cublas_wrappers.h"
#include "inference_cuda_layers.h"

std::array<int, 3> gemm_algos = std::array<int, 3>({99, 99, 99});

// NOTE: This activation function type enum should be always in sync
// with the python counterpart, otherwise the casting from python binding
// will be incorrect.
enum class ActivationFuncType { UNKNOWN = 0, GELU = 1, ReLU = 2, GATED_GELU = 3, GATED_SILU = 4 };

enum class NormType { UNKNOWN = 0, LayerNorm = 1, GroupNorm = 2, RMSNorm = 3 };

enum class TransformerType : uint8_t { UNKNOWN = 0, GPTType = 1, BERTType = 2 };

// NOTE: this is a temporary and dodgy solution to distinguish GPT and BERT style models
// based on the dimensions of the corresponding attention mask.
inline auto infer_transformer_type(at::Tensor& attn_mask) -> TransformerType
{
    auto attn_mask_num_dims = attn_mask.sizes().size();

    if (attn_mask_num_dims > 2) {
        return TransformerType::GPTType;
    } else if (attn_mask_num_dims == 2) {
        return TransformerType::BERTType;
    } else {
        return TransformerType::UNKNOWN;
    }
}

// infer stride of attention mask memory layout based on the model type.
inline auto get_attn_mask_stride(at::Tensor& attn_mask) -> int
{
    auto trnsfrmr_type = infer_transformer_type(attn_mask);

    if (trnsfrmr_type == TransformerType::GPTType) {
        return attn_mask.size(2);
    } else if (trnsfrmr_type == TransformerType::BERTType) {
        // Bert style models have always a mask stride of 1.
        return 1;
    } else if (trnsfrmr_type == TransformerType::UNKNOWN) {
        return 0;
    }

    // this is just to make the compiler happy.
    return 0;
}

template <typename T>
at::Tensor ds_softmax(at::Tensor& attn_scores,
                      at::Tensor& attn_mask,
                      at::Tensor& alibi,
                      bool triangular,
                      bool recompute,
                      bool local_attention,
                      int window_size,
                      bool async_op,
                      float layer_scale,
                      int head_offset,
                      int mp_size)
{
    auto attn_scores_c = attn_scores.contiguous();
    int bsz = attn_scores_c.size(0);

    int seq_len = attn_scores_c.size(1);
    int len = attn_scores_c.sizes().size();
    if (len > 2) seq_len = attn_scores_c.size(2);

    int soft_len = attn_scores_c.size(2);
    if (len > 3) soft_len = attn_scores_c.size(3);

    int heads = 1;
    if (len > 1) heads = attn_scores_c.size(1);

    auto mask_stride = get_attn_mask_stride(attn_mask);

    launch_attn_softmax_v2((T*)attn_scores_c.data_ptr(),
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           (alibi.sizes().size() > 1 ? (T*)alibi.data_ptr() : nullptr),
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           head_offset,
                           mask_stride,
                           mp_size,
                           InferenceContext::Instance().GetCurrentStream(async_op));

    return attn_scores_c;
}

template <typename T>
void allocate_workspace(unsigned hidden_dim,
                        unsigned num_heads,
                        unsigned prompt_length,
                        unsigned batch_size,
                        unsigned num_layers,
                        unsigned mp_size = 1,
                        bool external_cache = false,
                        unsigned rank = 0,
                        unsigned max_out_tokens = 1024,
                        unsigned min_out_tokens = 1)
{
    InferenceContext::Instance().GenWorkSpace(num_layers,
                                              num_heads,
                                              batch_size,
                                              prompt_length,
                                              hidden_dim,
                                              mp_size,
                                              external_cache,
                                              sizeof(T),
                                              rank,
                                              max_out_tokens,
                                              min_out_tokens);
}

template <typename T>
at::Tensor einsum_sec_sm_ecm(at::Tensor& Q, at::Tensor& W)
{
    auto options = at::TensorOptions()
                       .dtype(Q.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    float alpha = 1;
    float gemm_beta = 0.0;

    /*
    // Reallocate memory if we received a new prompt
    if (!workspace || input.size(1) != 1) {
        allocate_workspace<T>(W.size(1), InferenceContext::Instance().GetMaxTokenLength(),
    Q.size(0), 1, head_size); workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    }
    */

    auto O = at::from_blob(workspace, {Q.size(1), Q.size(2), W.size(1)}, options);
    unsigned m = W.size(1);
    unsigned n = Q.size(1) * Q.size(2);
    unsigned k = Q.size(0);
    cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   m,
                   n,
                   k,
                   &alpha,
                   &gemm_beta,
                   (T*)W.data_ptr(),
                   (T*)Q.data_ptr(),
                   (T*)O.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    return O;
}

template <typename T>
void attention_unfused(at::Tensor& prev_key_cont,
                       at::Tensor& query_cont,
                       at::Tensor& attn_mask,
                       at::Tensor& prev_value_cont,
                       at::Tensor& output,
                       int& bsz,
                       int& seq_len,
                       int& soft_len,
                       int& heads,
                       float& norm_factor,
                       bool triangular,
                       bool recompute,
                       bool local_attention,
                       int window_size)
{
    auto options = at::TensorOptions()
                       .dtype(query_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    float alpha = norm_factor;
    float gemm_beta = 0.0;
    auto attn_score = at::empty({bsz, heads, seq_len, soft_len}, options);
    int k = prev_value_cont.size(2) / heads;

    auto mask_stride = get_attn_mask_stride(attn_mask);

    cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                    InferenceContext::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(InferenceContext::Instance().GetCublasHandle(),
                                soft_len,
                                seq_len,
                                k,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_key_cont.data_ptr(),
                                (T*)query_cont.data_ptr(),
                                (T*)attn_score.data_ptr(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * k,
                                seq_len * soft_len,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    launch_attn_softmax_v2((T*)attn_score.data_ptr(),
                           (T*)(attn_mask.sizes().size() > 1 ? attn_mask.data_ptr() : nullptr),
                           (T*)nullptr,
                           1.0,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           0,
                           mask_stride,
                           1,
                           InferenceContext::Instance().GetCurrentStream(false));
    alpha = 1.0;
    cublas_strided_batched_gemm(InferenceContext::Instance().GetCublasHandle(),
                                k,
                                seq_len,
                                soft_len,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_value_cont.data_ptr(),
                                (T*)attn_score.data_ptr(),
                                (T*)output.data_ptr(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                soft_len * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
std::vector<at::Tensor> ds_softmax_context1(at::Tensor& query,
                                            at::Tensor& prev_key,
                                            at::Tensor& new_key,
                                            at::Tensor& attn_mask,
                                            at::Tensor& prev_value,
                                            at::Tensor& new_value,
                                            int heads,
                                            float norm_factor,
                                            bool merging,
                                            bool triangular,
                                            bool local_attention,
                                            int window_size,
                                            bool no_masking)
{
    auto query_cont = query.contiguous();
    auto prev_key_cont = prev_key.contiguous();
    auto prev_value_cont = prev_value.contiguous();

    int new_size = (new_value.sizes().size() > 1 ? new_value.size(1) : 0);

    // Attn_Score [ batch Head Sequence-length Softmax-length]

    int bsz = query_cont.size(0);
    int seq_len = query_cont.size(1);
    int soft_len = prev_value.size(1);

    auto options = at::TensorOptions()
                       .dtype(query_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output =
        at::empty({prev_value.size(0), heads, seq_len, prev_value.size(2) / heads}, options);
    attention_unfused<T>(prev_key_cont,
                         query_cont,
                         attn_mask,  //(no_masking ? nullptr : (T*)attn_mask.data_ptr()),
                         prev_value_cont,
                         output,
                         bsz,
                         seq_len,
                         soft_len,
                         heads,
                         norm_factor,
                         (triangular && (new_size == 0)),
                         (new_size == 0),
                         local_attention,
                         window_size);

    return {output, prev_key, prev_value};
}

template <typename T>
void ds_softmax_internal(T* attn_scores,
                         at::Tensor& attn_mask,
                         at::Tensor& alibi,
                         float& layer_scale,
                         bool triangular,
                         bool recompute,
                         bool local_attention,
                         int window_size,
                         int bsz,
                         int seq_len,
                         int soft_len,
                         int heads)
{
    auto mask_stride = get_attn_mask_stride(attn_mask);

    launch_attn_softmax_v2((T*)attn_scores,
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           (alibi.sizes().size() > 1 ? (T*)alibi.data_ptr() : nullptr),
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           0,
                           mask_stride,
                           1,
                           at::cuda::getCurrentCUDAStream());
}

template <typename T>
void attention_unfused(T* prev_key_cont,
                       T* query_cont,
                       at::Tensor& attn_mask,
                       T* prev_value_cont,
                       T* output,
                       unsigned& bsz,
                       int& k,
                       unsigned& seq_len,
                       unsigned& soft_len,
                       int& heads,
                       float& norm_factor,
                       bool triangular,
                       bool recompute,
                       bool local_attention,
                       int window_size,
                       at::Tensor& alibi,
                       int layer_id)
{
    float layer_scale = alibi.sizes().size() > 1 ? std::max(1, layer_id) : 1.0;
    float alpha = norm_factor * norm_factor / layer_scale;
    float gemm_beta = 0.0;
    T* workspace = (T*)InferenceContext::Instance().GetAttentionUnfusedWorkspace();

    cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                    InferenceContext::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(InferenceContext::Instance().GetCublasHandle(),
                                soft_len,
                                seq_len,
                                k,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_key_cont,
                                (T*)query_cont,
                                workspace,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                InferenceContext::Instance().GetMaxTokenLength() * k,
                                seq_len * k,
                                seq_len * soft_len,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    ds_softmax_internal<T>(workspace,
                           attn_mask,
                           alibi,
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           seq_len,
                           soft_len,
                           heads);
    alpha = 1.0;
    cublas_strided_batched_gemm(InferenceContext::Instance().GetCublasHandle(),
                                k,
                                seq_len,
                                soft_len,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_value_cont,
                                workspace,
                                (T*)output,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                InferenceContext::Instance().GetMaxTokenLength() * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

void reset_cache() { InferenceContext::Instance().reset_tokens(); }

template <typename T>
std::vector<at::Tensor> ds_softmax_context(at::Tensor& query_key_value,
                                           at::Tensor& attn_mask,
                                           int rotary_dim,
                                           bool rotate_half,
                                           bool rotate_every_two,
                                           int heads,
                                           int num_kv,
                                           float norm_factor,
                                           bool triangular,
                                           bool local_attention,
                                           int window_size,
                                           bool no_masking,
                                           unsigned layer_id,
                                           unsigned num_layers,
                                           at::Tensor& alibi)
{
    unsigned bsz = query_key_value.size(0);
    unsigned seq_len = query_key_value.size(1);
    int k = query_key_value.size(2) / (heads + 2 * (num_kv > 0 ? num_kv : heads));
    unsigned hidden_dim = heads * k;

    bool is_prompt = (seq_len > 1);

    if (is_prompt) InferenceContext::Instance().reset_tokens(seq_len);
    unsigned soft_len = InferenceContext::Instance().current_tokens();

    auto options = at::TensorOptions()
                       .dtype(query_key_value.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    size_t buf_size = bsz * seq_len * hidden_dim;
    auto output = torch::from_blob(workspace + 4 * buf_size, {bsz, seq_len, hidden_dim}, options);

    auto query_cont = workspace + 5 * buf_size;
    size_t offset =
        10 * (hidden_dim * bsz * InferenceContext::Instance().GetMaxTokenLength()) +
        layer_id * 2 * bsz * InferenceContext::Instance().GetMaxTokenLength() * hidden_dim;
    unsigned all_tokens = soft_len;
    auto kv_cache = workspace + offset + (hidden_dim / heads) * (is_prompt ? 0 : soft_len - 1);
    size_t value_offset = bsz * InferenceContext::Instance().GetMaxTokenLength() * hidden_dim;

    T* temp_buf = (T*)output.data_ptr() + at::numel(output);
    launch_bias_add_transform_0213<T>((T*)query_cont,
                                      kv_cache,
                                      kv_cache + value_offset,
                                      (T*)query_key_value.data_ptr(),
                                      nullptr,
                                      bsz,
                                      seq_len,
                                      (is_prompt ? 0 : soft_len - 1),
                                      soft_len,
                                      hidden_dim,
                                      heads,
                                      (num_kv > 0 ? num_kv : heads),
                                      rotary_dim,
                                      rotate_half,
                                      rotate_every_two,
                                      InferenceContext::Instance().GetCurrentStream(),
                                      3,
                                      InferenceContext::Instance().GetMaxTokenLength());
    if (rotary_dim > 0 && rotate_half)
        launch_apply_rotary_pos_emb(query_cont,
                                    kv_cache,
                                    k,
                                    seq_len,
                                    rotary_dim,
                                    (is_prompt ? 0 : soft_len - 1),
                                    heads,
                                    bsz,
                                    InferenceContext::Instance().GetCurrentStream(),
                                    InferenceContext::Instance().GetMaxTokenLength());

    attention_unfused<T>(workspace + offset,
                         (T*)query_cont,
                         attn_mask,
                         workspace + offset + value_offset,
                         temp_buf,
                         bsz,
                         k,
                         seq_len,
                         all_tokens,
                         heads,
                         norm_factor,
                         (triangular && is_prompt),
                         is_prompt,
                         local_attention,
                         window_size,
                         alibi,
                         layer_id);
    launch_transform4d_0213<T>((T*)output.data_ptr(),
                               temp_buf,
                               bsz,
                               heads,
                               seq_len,
                               output.size(2),
                               InferenceContext::Instance().GetCurrentStream(false),
                               1);

    if (layer_id == num_layers - 1) InferenceContext::Instance().advance_tokens();
    auto prev_key = torch::from_blob(workspace + offset,
                                     {bsz, heads, all_tokens, k},
                                     {hidden_dim * InferenceContext::Instance().GetMaxTokenLength(),
                                      k * InferenceContext::Instance().GetMaxTokenLength(),
                                      k,
                                      1},
                                     options);

    auto prev_value =
        torch::from_blob(workspace + offset + value_offset,
                         {bsz, heads, all_tokens, k},
                         {hidden_dim * InferenceContext::Instance().GetMaxTokenLength(),
                          k * InferenceContext::Instance().GetMaxTokenLength(),
                          k,
                          1},
                         options);

    return {output, prev_key, prev_value};
}

template <typename T>
at::Tensor ds_bias_gelu(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int intermediate_size = input_cont.size(2);

    launch_bias_gelu((T*)input_cont.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     InferenceContext::Instance().GetCurrentStream());
    return input_cont;
}

#define DISPATCH_GATED_ACT(T_TYPE, C_TYPE)                                         \
    if (activation.options().dtype() == torch::T_TYPE) {                           \
        launch_gated_activation((C_TYPE*)output.data_ptr(),                        \
                                (const C_TYPE*)activation.data_ptr(),              \
                                (const C_TYPE*)bias.data_ptr(),                    \
                                rows,                                              \
                                out_channels,                                      \
                                channels,                                          \
                                activation_type == ActivationFuncType::GATED_GELU, \
                                InferenceContext::Instance().GetCurrentStream());  \
    }

at::Tensor ds_gated_activation(at::Tensor& activation, at::Tensor& bias, int actFun)
{
    /*
    Used in FF of Stable diffusion
    */

    const ActivationFuncType activation_type = static_cast<ActivationFuncType>(actFun);

    assert(activation_type == ActivationFuncType::GATED_GELU ||
           activation_type == ActivationFuncType::GATED_SILU);

    const int batch_size = activation.size(0);
    const int seq_len = activation.size(1);
    const int channels = activation.size(2);

    const int rows = batch_size * seq_len;
    // Dimensionality is cut in half
    const int out_channels = channels / 2;

    auto output = at::empty({batch_size, seq_len, out_channels}, activation.options());

    DISPATCH_GATED_ACT(kFloat, float);
    DISPATCH_GATED_ACT(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_GATED_ACT(kBFloat16, __nv_bfloat16);
#endif

    return output;
}

template <typename T>
at::Tensor ds_bias_relu(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int intermediate_size = input_cont.size(2);

    launch_bias_relu((T*)input_cont.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     InferenceContext::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_bias_add(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int hidden_size = input_cont.size(2);

    launch_bias_add((T*)input_cont.data_ptr(),
                    (T*)bias.data_ptr(),
                    hidden_size,
                    bsz,
                    InferenceContext::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_bias_residual(at::Tensor& input, at::Tensor& residual, at::Tensor& bias)
{
    auto input_cont = input.contiguous();
    auto residual_cont = residual.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    // launch_bias_residual((T*)input_cont.data_ptr(),
    //                      (T*)residual_cont.data_ptr(),
    //                      (T*)bias.data_ptr(),
    //                      bsz,
    //                      input_cont.size(2),
    //                      (bias.size(0) > 1),
    //                      InferenceContext::Instance().GetCurrentStream());
    return input_cont;
}

#define DISPATCH_LAYER_NORM(T_TYPE, C_TYPE)                               \
    if (input.options().dtype() == torch::T_TYPE) {                       \
        launch_fused_ln((C_TYPE*)output.data_ptr(),                       \
                        (const C_TYPE*)input.data_ptr(),                  \
                        (const C_TYPE*)gamma.data_ptr(),                  \
                        (const C_TYPE*)beta.data_ptr(),                   \
                        epsilon,                                          \
                        rows,                                             \
                        elems_per_row,                                    \
                        InferenceContext::Instance().GetCurrentStream()); \
    }

at::Tensor ds_layer_norm(at::Tensor& input, at::Tensor& gamma, at::Tensor& beta, float epsilon)
{
    const int rows = input.size(0) * input.size(1);
    const int elems_per_row = input.size(2);
    auto output = at::empty_like(input);

    DISPATCH_LAYER_NORM(kFloat, float);
    DISPATCH_LAYER_NORM(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_LAYER_NORM(kBFloat16, __nv_bfloat16);
#endif

    return output;
}

#define DISPATCH_RMS_NORM(T_TYPE, C_TYPE)                                 \
    if (input.options().dtype() == torch::T_TYPE) {                       \
        launch_rms_norm((C_TYPE*)output.data_ptr(),                       \
                        (C_TYPE*)nullptr,                                 \
                        (const C_TYPE*)input.data_ptr(),                  \
                        (const C_TYPE*)nullptr,                           \
                        (const C_TYPE*)gamma.data_ptr(),                  \
                        epsilon,                                          \
                        rows,                                             \
                        elems_per_row,                                    \
                        InferenceContext::Instance().GetCurrentStream()); \
    }

at::Tensor ds_rms_norm(at::Tensor& input, at::Tensor& gamma, float epsilon)
{
    // Get number of dims of tensor
    int num_dims = input.dim();
    const int rows = (num_dims == 2) ? input.size(0) : input.size(0) * input.size(1);
    const int elems_per_row = (num_dims == 2) ? input.size(1) : input.size(2);

    auto output = at::empty_like(input);

    DISPATCH_RMS_NORM(kFloat, float);
    DISPATCH_RMS_NORM(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_RMS_NORM(kBFloat16, __nv_bfloat16);
#endif

    return output;
}

#define DISPATCH_PRE_RMS_NORM(T_TYPE, C_TYPE)                             \
    if (input.options().dtype() == torch::T_TYPE) {                       \
        launch_rms_norm((C_TYPE*)output.data_ptr(),                       \
                        (C_TYPE*)res_out.data_ptr(),                      \
                        (const C_TYPE*)input.data_ptr(),                  \
                        (const C_TYPE*)residual.data_ptr(),               \
                        (const C_TYPE*)gamma.data_ptr(),                  \
                        epsilon,                                          \
                        rows,                                             \
                        elems_per_row,                                    \
                        InferenceContext::Instance().GetCurrentStream()); \
    }

std::vector<at::Tensor> ds_pre_rms_norm(at::Tensor& input,
                                        at::Tensor& residual,
                                        at::Tensor& gamma,
                                        float epsilon)
{
    // Get number of dims of tensor
    int num_dims = input.dim();
    const int rows = (num_dims == 2) ? input.size(0) : input.size(0) * input.size(1);
    const int elems_per_row = (num_dims == 2) ? input.size(1) : input.size(2);

    auto output = at::empty_like(input);
    auto res_out = at::empty_like(residual);

    DISPATCH_PRE_RMS_NORM(kFloat, float);
    DISPATCH_PRE_RMS_NORM(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_PRE_RMS_NORM(kBFloat16, __nv_bfloat16);
#endif

    return {output, res_out};
}

template <typename T>
void ds_layer_norm_internal(T* workspace,
                            at::Tensor& input,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            float epsilon)
{
    int bsz = input.size(0) * input.size(1);
    launch_fused_ln(workspace,
                    (const T*)input.data_ptr(),
                    (const T*)gamma.data_ptr(),
                    (const T*)beta.data_ptr(),
                    epsilon,
                    bsz,
                    input.size(2),
                    InferenceContext::Instance().GetCurrentStream());
}

#define DISPATCH_LAYER_NORM_RESIDUAL(T_TYPE, C_TYPE)                               \
    if (input.options().dtype() == torch::T_TYPE) {                                \
        launch_fused_residual_ln((C_TYPE*)output.data_ptr(),                       \
                                 (const C_TYPE*)input.data_ptr(),                  \
                                 (const C_TYPE*)residual.data_ptr(),               \
                                 (const C_TYPE*)bias.data_ptr(),                   \
                                 (const C_TYPE*)gamma.data_ptr(),                  \
                                 (const C_TYPE*)beta.data_ptr(),                   \
                                 epsilon,                                          \
                                 rows,                                             \
                                 elems_per_row,                                    \
                                 InferenceContext::Instance().GetCurrentStream()); \
    }

/* Currently only used in unit testing */
at::Tensor ds_layer_norm_residual(at::Tensor& input,
                                  at::Tensor& bias,
                                  at::Tensor& residual,
                                  at::Tensor& gamma,
                                  at::Tensor& beta,
                                  float epsilon)
{
    const int rows = input.size(0) * input.size(1);
    const int elems_per_row = input.size(2);
    auto output = at::empty_like(input);

    DISPATCH_LAYER_NORM_RESIDUAL(kFloat, float);
    DISPATCH_LAYER_NORM_RESIDUAL(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_LAYER_NORM_RESIDUAL(kBFloat16, __nv_bfloat16);
#endif

    return output;
}

#define DISPATCH_PRE_LAYER_NORM_RESIDUAL(T_TYPE, C_TYPE)      \
    if (input.options().dtype() == torch::T_TYPE) {           \
        launch_fused_residual_ln_store_pre_ln_res(            \
            (C_TYPE*)norm_output.data_ptr(),                  \
            (C_TYPE*)res_output.data_ptr(),                   \
            (const C_TYPE*)input.data_ptr(),                  \
            (const C_TYPE*)residual.data_ptr(),               \
            (const C_TYPE*)bias.data_ptr(),                   \
            (const C_TYPE*)gamma.data_ptr(),                  \
            (const C_TYPE*)beta.data_ptr(),                   \
            epsilon,                                          \
            rows,                                             \
            elems_per_row,                                    \
            InferenceContext::Instance().GetCurrentStream()); \
    }

/* Currently only used in unit testing */
std::vector<at::Tensor> ds_layer_norm_residual_store_pre_ln_res(at::Tensor& input,
                                                                at::Tensor& bias,
                                                                at::Tensor& residual,
                                                                at::Tensor& gamma,
                                                                at::Tensor& beta,
                                                                float epsilon)
{
    const int rows = input.size(0) * input.size(1);
    const int elems_per_row = input.size(2);
    auto norm_output = at::empty_like(input);
    auto res_output = at::empty_like(input);

    DISPATCH_PRE_LAYER_NORM_RESIDUAL(kFloat, float);
    DISPATCH_PRE_LAYER_NORM_RESIDUAL(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_PRE_LAYER_NORM_RESIDUAL(kBFloat16, __nv_bfloat16);
#endif

    return {norm_output, res_output};
}

template <typename T>
void quantized_gemm(void* output,
                    T* input,
                    at::Tensor& weight,
                    at::Tensor& qscale,
                    int groups,
                    int bsz,
                    int hidden_size)
{
    // T* weight16 = (T*)InferenceContext::Instance().GetWorkSpace() + 12 * hidden_size * bsz;

    auto options = at::TensorOptions()
                       .dtype(at::kHalf)
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto tmp = torch::empty(weight.sizes(), options);
    T* weight16 = (T*)tmp.data_ptr();
    launch_dequantize(weight16,
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(0),
                      weight.size(1),
                      groups,
                      InferenceContext::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   weight.size(0),
                   bsz,
                   weight.size(1),
                   &alpha,
                   &gemm_beta,
                   weight16,
                   (T*)input,
                   (T*)output,
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
at::Tensor qkv_unfused_cublas(at::Tensor& output,
                              at::Tensor& input,
                              at::Tensor& weight,
                              at::Tensor& q_scale,
                              at::Tensor& bias,
                              at::Tensor& gamma,
                              at::Tensor& beta,
                              const float epsilon,
                              bool add_bias,
                              bool q_int8,
                              bool transposed_mode)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    workspace += (3 * bsz * input.size(2));
    ds_layer_norm_internal<T>(workspace, input, gamma, beta, epsilon);

    if (q_int8) {
        quantized_gemm<T>(
            output.data_ptr(), workspace, weight, q_scale, q_scale.size(0), bsz, input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;

        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight.size(transposed_mode ? 0 : 1),
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       workspace,
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                        bsz,
                        InferenceContext::Instance().GetCurrentStream());
    return torch::from_blob(workspace, input.sizes(), input.options());
}

template <typename T>
std::vector<at::Tensor> ds_rms_qkv(at::Tensor& input,
                                   at::Tensor& weight,
                                   at::Tensor& q_scale,
                                   at::Tensor& gamma,
                                   const float epsilon,
                                   bool q_int8,
                                   bool transposed_mode)
{
    const int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    T* rms_norm_ptr = workspace + (3 * bsz * input.size(2));
    int out_size = (transposed_mode || q_int8) ? weight.size(0) : weight.size(1);

    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto rms_norm = at::from_blob(rms_norm_ptr, input.sizes(), options);
    auto output = at::from_blob(workspace, {input.size(0), input.size(1), out_size}, options);

    launch_rms_norm((T*)rms_norm.data_ptr(),
                    (T*)nullptr,
                    (const T*)input.data_ptr(),
                    (const T*)nullptr,
                    (const T*)gamma.data_ptr(),
                    epsilon,
                    bsz,
                    input.size(2),
                    InferenceContext::Instance().GetCurrentStream());

    if (q_int8) {
        quantized_gemm<T>((T*)output.data_ptr(),
                          (T*)rms_norm.data_ptr(),
                          weight,
                          q_scale,
                          q_scale.size(0),
                          bsz,
                          input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;

        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight.size(transposed_mode ? 0 : 1),
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       (T*)rms_norm.data_ptr(),
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }

    return {output, rms_norm};
}

template <typename T>
std::vector<at::Tensor> ds_qkv_gemm(at::Tensor& input,
                                    at::Tensor& weight,
                                    at::Tensor& q_scale,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool add_bias,
                                    bool q_int8,
                                    bool transposed_mode)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    int out_size = (transposed_mode || q_int8) ? weight.size(0) : weight.size(1);

    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::from_blob(workspace, {input.size(0), input.size(1), out_size}, options);
    auto inp_norm = qkv_unfused_cublas<T>(output,
                                          input,
                                          weight,
                                          q_scale,
                                          bias,
                                          gamma,
                                          beta,
                                          epsilon,
                                          add_bias,
                                          q_int8,
                                          transposed_mode);

    return {output, inp_norm};
}

template <typename T>
void quantized_gemm(at::Tensor& output,
                    at::Tensor& input,
                    at::Tensor& weight,
                    at::Tensor& qscale,
                    int groups,
                    int merge_count)
{
    int bsz = input.size(0) * input.size(1);
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto weight16 = at::empty({weight.size(0), weight.size(1)}, options);

    launch_dequantize((T*)weight16.data_ptr(),
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(0),
                      weight.size(1),
                      groups,
                      merge_count,
                      InferenceContext::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   weight.size(0),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight16.data_ptr(),
                   (T*)input.data_ptr(),
                   (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
at::Tensor ds_linear_layer(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& bias,
                           bool add_bias,
                           bool do_flash_attn,
                           int num_heads,
                           bool transposed_mode)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int head_size = input_cont.size(2) / num_heads;
    int bsz = input.size(0) * input.size(1);
    int out_size = transposed_mode ? weight.size(0) : weight.size(1);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    auto output = at::from_blob(workspace, {input.size(0), input.size(1), out_size}, options);

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                    InferenceContext::Instance().GetCurrentStream());

    cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                   (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                   CUBLAS_OP_N,
                   weight.size(transposed_mode ? 0 : 1),
                   bsz,
                   input_cont.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight.data_ptr(),
                   (T*)input_cont.data_ptr(),
                   (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        weight.size(transposed_mode ? 0 : 1),
                        bsz,
                        InferenceContext::Instance().GetCurrentStream());
    bool add_padding = (head_size % 32 != 0 && head_size < 64) || (head_size % 64 != 0);
    if (do_flash_attn) {
        if (add_padding) {
            int padded_head_size = head_size < 32 ? 32 : (head_size < 64 ? 64 : 128);
            auto padded_output = workspace + output.numel();
            auto final_output =
                padded_output + (input.size(0) * input.size(1) * 3 * num_heads * padded_head_size);
            pad_data(padded_output,
                     workspace,
                     3 * bsz * num_heads,
                     head_size,
                     padded_head_size,
                     InferenceContext::Instance().GetCurrentStream());

            launch_bias_add_transform_0213<T>(
                final_output,
                final_output + (input.size(0) * input.size(1) * num_heads * padded_head_size),
                final_output + (input.size(0) * input.size(1) * 2 * num_heads * padded_head_size),
                padded_output,
                nullptr,
                input.size(0),
                input.size(1),
                0,
                input.size(1),
                (num_heads * padded_head_size),
                num_heads,
                -1,
                -1,
                false,
                false,
                InferenceContext::Instance().GetCurrentStream(),
                3,
                input.size(1));
            return at::from_blob(final_output,
                                 {3, input.size(0), num_heads, input.size(1), padded_head_size},
                                 options);
            // return at::from_blob(padded_output, {input.size(0) * input.size(1), 3, num_heads,
            // padded_head_size}, options);
        } else {
            auto final_output = workspace + output.numel();
            launch_bias_add_transform_0213<T>(
                final_output,
                final_output + (input.size(0) * input.size(1) * input_cont.size(2)),
                final_output + (input.size(0) * input.size(1) * 2 * input_cont.size(2)),
                workspace,
                nullptr,
                input.size(0),
                input.size(1),
                0,
                input.size(1),
                input_cont.size(2),
                num_heads,
                -1,
                -1,
                false,
                false,
                InferenceContext::Instance().GetCurrentStream(),
                3,
                input.size(1));
            return at::from_blob(
                final_output, {3, input.size(0), num_heads, input.size(1), head_size}, options);
            // return at::from_blob(workspace, {input.size(0) * input.size(1), 3, num_heads,
            // head_size}, options);
        }

    } else
        return output;
}

template <typename T>
std::vector<at::Tensor> add_padding(at::Tensor& query, at::Tensor& key, at::Tensor& value)
{
    int head_size = query.size(3);
    int padded_head_size = head_size < 32 ? 32 : (head_size < 64 ? 64 : 128);
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    T* key_pad_ptr = workspace + padded_head_size * query.size(0) * query.size(1) * query.size(2);
    T* value_pad_ptr = key_pad_ptr + padded_head_size * query.size(0) * query.size(1) * 128;
    pad_head_seq(workspace,
                 (T*)query.data_ptr(),
                 query.size(0) * query.size(1),
                 query.size(2),
                 query.size(2),
                 head_size,
                 padded_head_size,
                 InferenceContext::Instance().GetCurrentStream());
    pad_head_seq(key_pad_ptr,
                 (T*)key.data_ptr(),
                 query.size(0) * query.size(1),
                 key.size(2),
                 128,
                 head_size,
                 padded_head_size,
                 InferenceContext::Instance().GetCurrentStream());
    pad_head_seq(value_pad_ptr,
                 (T*)value.data_ptr(),
                 query.size(0) * query.size(1),
                 key.size(2),
                 128,
                 head_size,
                 padded_head_size,
                 InferenceContext::Instance().GetCurrentStream());
    return {
        at::from_blob(workspace,
                      {query.size(0), query.size(1), query.size(2), padded_head_size},
                      query.options()),
        at::from_blob(
            key_pad_ptr, {query.size(0), query.size(1), 128, padded_head_size}, query.options()),
        at::from_blob(
            value_pad_ptr, {query.size(0), query.size(1), 128, padded_head_size}, query.options())};
}

template <typename T>
std::vector<at::Tensor> padd_add_transform(at::Tensor& query,
                                           at::Tensor& key,
                                           at::Tensor& value,
                                           int heads,
                                           bool add_padding)
{
    int head_size = query.size(2) / heads;
    int key_value_length = add_padding ? 128 : key.size(1);
    int padded_head_size = add_padding ? (head_size < 32 ? 32 : (head_size < 64 ? 64 : 128))
                                       : head_size;
    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    T* key_pad_ptr = workspace + padded_head_size * query.size(0) * heads * query.size(1);
    T* value_pad_ptr = key_pad_ptr + padded_head_size * query.size(0) * heads * key_value_length;
    launch_pad_add_transform_0213(workspace,
                                  (T*)query.data_ptr(),
                                  query.size(0),
                                  query.size(2),
                                  query.size(1),
                                  query.size(1),
                                  heads,
                                  padded_head_size,
                                  InferenceContext::Instance().GetCurrentStream());
    launch_pad_add_transform_0213(key_pad_ptr,
                                  (T*)key.data_ptr(),
                                  key.size(0),
                                  key.size(2),
                                  key.size(1),
                                  key_value_length,
                                  heads,
                                  padded_head_size,
                                  InferenceContext::Instance().GetCurrentStream());
    launch_pad_add_transform_0213(value_pad_ptr,
                                  (T*)value.data_ptr(),
                                  value.size(0),
                                  value.size(2),
                                  value.size(1),
                                  key_value_length,
                                  heads,
                                  padded_head_size,
                                  InferenceContext::Instance().GetCurrentStream());
    return {
        at::from_blob(
            workspace, {query.size(0), heads, query.size(1), padded_head_size}, query.options()),
        at::from_blob(key_pad_ptr,
                      {query.size(0), heads, key_value_length, padded_head_size},
                      query.options()),
        at::from_blob(value_pad_ptr,
                      {query.size(0), heads, key_value_length, padded_head_size},
                      query.options())};
}

template <typename T>
at::Tensor ds_vector_matmul(at::Tensor& input,
                            at::Tensor& weight,
                            bool async_op,
                            at::Tensor& q_scale,
                            bool q_int8,
                            bool transposed_mode)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    int out_size = (q_int8 || transposed_mode) ? weight.size(0) : weight.size(1);
    int bsz = input.size(0) * input.size(1);

    T* workspace = (T*)InferenceContext::Instance().GetWorkSpace();
    auto output = at::from_blob(workspace, {input.size(0), input.size(1), out_size}, options);
    if (q_int8) {
        quantized_gemm<T>(output.data_ptr(),
                          (T*)input.data_ptr(),
                          weight,
                          q_scale,
                          q_scale.size(0),
                          bsz,
                          input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream(async_op));
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight.size(transposed_mode ? 0 : 1),
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       (T*)input.data_ptr(),
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    return output;
}

template <typename T>
at::Tensor ds_vector_matmul_int8(at::Tensor& input,
                                 at::Tensor& weight,
                                 at::Tensor& q_scale,
                                 int groups,
                                 int merge_count)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    quantized_gemm<T>(output, input_cont, weight, q_scale, groups, merge_count);
    return output;
}

template <typename T>
at::Tensor mlp_unfused_cublas(at::Tensor& output,
                              at::Tensor& input,
                              at::Tensor& residual,
                              at::Tensor& input_bias,
                              at::Tensor& weight,
                              at::Tensor& weight1,
                              at::Tensor& bias,
                              at::Tensor& gamma,
                              at::Tensor& beta,
                              const float epsilon,
                              bool preLayerNorm,
                              bool mlp_after_attn,
                              at::Tensor& q_scale,
                              at::Tensor& q_scale1,
                              bool q_int8,
                              ActivationFuncType act_func_type,
                              bool transposed_mode)
{
    int bsz = input.size(0) * input.size(1);
    T* inp_norm = (T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input) +
                  torch::numel(output);
    T* intermediate = inp_norm + torch::numel(input);

    if (mlp_after_attn) {
        launch_fused_residual_ln((T*)inp_norm,
                                 (const T*)input.data_ptr(),
                                 (const T*)residual.data_ptr(),
                                 (const T*)input_bias.data_ptr(),
                                 (const T*)gamma.data_ptr(),
                                 (const T*)beta.data_ptr(),
                                 epsilon,
                                 bsz,
                                 input.size(2),
                                 InferenceContext::Instance().GetCurrentStream());
    } else {
        ds_layer_norm_internal(inp_norm, input, gamma, beta, epsilon);
    }
    if (q_int8) {
        quantized_gemm<T>(
            intermediate, inp_norm, weight, q_scale, q_scale.size(0), bsz, input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight.size(transposed_mode ? 0 : 1),
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       inp_norm,
                       intermediate,
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    if (act_func_type == ActivationFuncType::GELU) {
        launch_bias_gelu(intermediate,
                         (T*)bias.data_ptr(),
                         (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::ReLU) {
        launch_bias_relu(intermediate,
                         (T*)bias.data_ptr(),
                         (transposed_mode || q_int8) ? weight.size(0) : weight.size(1),
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    }

    if (q_int8) {
        quantized_gemm<T>(output.data_ptr(),
                          intermediate,
                          weight1,
                          q_scale1,
                          q_scale1.size(0),
                          bsz,
                          input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       weight1.size(transposed_mode ? 0 : 1),
                       bsz,
                       weight1.size(transposed_mode ? 1 : 0),
                       &alpha,
                       &gemm_beta,
                       (T*)weight1.data_ptr(),
                       intermediate,
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }

    return torch::from_blob(inp_norm, input.sizes(), input.options());
}

template <typename T>
std::vector<at::Tensor> ds_mlp_gemm(at::Tensor& input,
                                    at::Tensor& residual,
                                    at::Tensor& input_bias,
                                    at::Tensor& weight_interm,
                                    at::Tensor& weight_out,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool preLayerNorm,
                                    bool mlp_after_attn,
                                    at::Tensor& q_scale,
                                    at::Tensor& q_scale1,
                                    bool q_int8,
                                    int activation_type,
                                    bool transposed_mode)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int out_size = (q_int8 || transposed_mode) ? weight_out.size(0) : weight_out.size(1);
    auto output =
        at::from_blob((T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input),
                      {input.size(0), input.size(1), out_size},
                      options);
    int bsz = input.size(0) * input.size(1);

    auto act_func_type = static_cast<ActivationFuncType>(activation_type);
    auto res_add = mlp_unfused_cublas<T>(output,
                                         mlp_after_attn ? input : residual,
                                         residual,
                                         input_bias,
                                         weight_interm,
                                         weight_out,
                                         bias,
                                         gamma,
                                         beta,
                                         epsilon,
                                         preLayerNorm,
                                         mlp_after_attn,
                                         q_scale,
                                         q_scale1,
                                         q_int8,
                                         act_func_type,
                                         transposed_mode);

    return {output, res_add};
}

template <typename T>
std::vector<at::Tensor> ds_rms_mlp_gemm(at::Tensor& input,
                                        at::Tensor& residual,
                                        at::Tensor& weight_interm,
                                        at::Tensor& weight_out,
                                        at::Tensor& gamma,
                                        const float epsilon,
                                        at::Tensor& q_scale,
                                        at::Tensor& q_scale1,
                                        bool q_int8,
                                        int activation_type,
                                        bool transposed_mode)
{
    const int bsz = input.size(0) * input.size(1);
    const size_t input_neurons = input.size(2);
    const size_t mlp_1_out_neurons = transposed_mode ? weight_interm.size(0)
                                                     : weight_interm.size(1);
    const size_t mlp_2_in_neurons = transposed_mode ? weight_out.size(1) : weight_out.size(0);

    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    T* output_ptr = (T*)InferenceContext::Instance().GetWorkSpace() + torch::numel(input);
    T* inp_norm_ptr = output_ptr + torch::numel(input);
    T* intermediate_ptr = inp_norm_ptr + torch::numel(input);

    auto output = at::from_blob(output_ptr, input.sizes(), options);
    auto inp_norm = at::from_blob(inp_norm_ptr, input.sizes(), options);
    auto intermediate_gemm =
        at::from_blob(intermediate_ptr, {input.size(0), input.size(1), mlp_1_out_neurons}, options);

    auto act_func_type = static_cast<ActivationFuncType>(activation_type);

    // RMS Norm, we'll update the residual in-place
    launch_rms_norm((T*)inp_norm.data_ptr(),
                    (T*)residual.data_ptr(),
                    (const T*)input.data_ptr(),
                    (const T*)residual.data_ptr(),
                    (const T*)gamma.data_ptr(),
                    epsilon,
                    bsz,
                    input_neurons,
                    InferenceContext::Instance().GetCurrentStream());

    if (q_int8) {
        quantized_gemm<T>(intermediate_ptr,
                          (T*)inp_norm.data_ptr(),
                          weight_interm,
                          q_scale,
                          q_scale.size(0),
                          bsz,
                          input_neurons);
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       mlp_1_out_neurons,
                       bsz,
                       input_neurons,
                       &alpha,
                       &gemm_beta,
                       (T*)weight_interm.data_ptr(),
                       (T*)inp_norm.data_ptr(),
                       intermediate_ptr,
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }

    if (act_func_type == ActivationFuncType::GELU) {
        launch_bias_gelu(intermediate_ptr,
                         (T*)nullptr,
                         mlp_1_out_neurons,
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::ReLU) {
        launch_bias_relu(intermediate_ptr,
                         (T*)nullptr,
                         mlp_1_out_neurons,
                         bsz,
                         InferenceContext::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::GATED_GELU) {
        launch_gated_activation(intermediate_ptr,
                                (const T*)intermediate_ptr,
                                (const T*)nullptr,
                                bsz,
                                mlp_1_out_neurons,
                                mlp_1_out_neurons,
                                true,
                                InferenceContext::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::GATED_SILU) {
        launch_gated_activation(intermediate_ptr,
                                (const T*)intermediate_ptr,
                                (const T*)nullptr,
                                bsz,
                                mlp_1_out_neurons,
                                mlp_1_out_neurons,
                                false,
                                InferenceContext::Instance().GetCurrentStream());
    }

    if (q_int8) {
        quantized_gemm<T>(output.data_ptr(),
                          intermediate_ptr,
                          weight_out,
                          q_scale1,
                          q_scale1.size(0),
                          bsz,
                          input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       input_neurons,
                       bsz,
                       mlp_2_in_neurons,
                       &alpha,
                       &gemm_beta,
                       (T*)weight_out.data_ptr(),
                       intermediate_ptr,
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard,
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP,
#endif
                       mlp_1_out_neurons);
    }

    return {output, residual};
}

template <typename T>
at::Tensor fused_gemm_gelu(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& weight_scale,
                           at::Tensor& bias,
                           at::Tensor& weight_out,
                           at::Tensor& weight_out_scale,
                           bool q_int8,
                           bool transposed_mode)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int intm_dim = (transposed_mode || q_int8) ? weight.size(0) : weight.size(1);

    // auto output = at::from_blob((T*)InferenceContext::Instance().GetWorkSpace() +
    // torch::numel(input),
    //                            {input.size(0), input.size(1), out_size},
    //                            options);
    // T* intermediate = (T*)input.data_ptr() + torch::numel(input);
    auto intermediate = at::empty({input.size(0), input.size(1), intm_dim}, options);

    int bsz = input.size(0) * input.size(1);

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    if (q_int8) {
        quantized_gemm<T>(intermediate.data_ptr(),
                          (T*)input.data_ptr(),
                          weight,
                          weight_scale,
                          weight_scale.size(0),
                          bsz,
                          input.size(2));
    } else {
        cublasSetStream(InferenceContext::Instance().GetCublasHandle(),
                        InferenceContext::Instance().GetCurrentStream());
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       intm_dim,
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       (T*)input.data_ptr(),
                       (T*)intermediate.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    launch_bias_gelu((T*)intermediate.data_ptr(),
                     (T*)bias.data_ptr(),
                     intm_dim,
                     bsz,
                     InferenceContext::Instance().GetCurrentStream());

    int out_size = (transposed_mode || q_int8) ? weight_out.size(0) : weight_out.size(1);
    auto output = at::empty({input.size(0), input.size(1), out_size}, options);
    if (q_int8) {
        quantized_gemm<T>(output.data_ptr(),
                          (T*)intermediate.data_ptr(),
                          weight_out,
                          weight_out_scale,
                          weight_out_scale.size(0),
                          bsz,
                          input.size(2));
    } else {
        cublas_gemm_ex(InferenceContext::Instance().GetCublasHandle(),
                       (transposed_mode ? CUBLAS_OP_T : CUBLAS_OP_N),
                       CUBLAS_OP_N,
                       out_size,
                       bsz,
                       intm_dim,
                       &alpha,
                       &gemm_beta,
                       (T*)weight_out.data_ptr(),
                       (T*)intermediate.data_ptr(),
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    }
    // cudaEventRecord(InferenceContext::Instance().GetCompEvent(2),
    //                InferenceContext::Instance().GetCurrentStream(true));
    return output;
}

template <typename T>
at::Tensor& residual_add_bias(at::Tensor& hidden_state,
                              at::Tensor& residual,
                              const at::Tensor& attention_output,
                              const at::Tensor& attention_bias,
                              const at::Tensor& final_bias,
                              const int mp_size,
                              const bool mlp_after_attn,
                              const bool add_bias,
                              const bool preln)
{
    int bsz = residual.size(0) * residual.size(1);
    int hidden_size = residual.size(2);
    if (mlp_after_attn)
        launch_bias_residual(static_cast<T*>(residual.data_ptr()),
                             static_cast<T*>(hidden_state.data_ptr()),
                             static_cast<T*>(attention_output.data_ptr()),
                             static_cast<T*>(final_bias.data_ptr()),
                             static_cast<T*>(attention_bias.data_ptr()),
                             bsz,
                             hidden_size,
                             mp_size,
                             preln,
                             InferenceContext::Instance().GetCurrentStream());
    else
        launch_gptj_residual_add<T>(
            static_cast<T*>(residual.data_ptr()),
            static_cast<T*>(hidden_state.data_ptr()),
            static_cast<T*>(attention_output.data_ptr()),
            static_cast<T*>(final_bias.data_ptr()),
            static_cast<T*>((add_bias ? attention_bias.data_ptr() : nullptr)),
            hidden_size,
            bsz,
            mp_size,
            InferenceContext::Instance().GetCurrentStream());
    return residual;
}

#define DISPATCH_VECTOR_ADD(T_TYPE, C_TYPE)                                         \
    if (a.scalar_type() == at::k##T_TYPE) {                                         \
        launch_vector_add<C_TYPE>((C_TYPE*)(a.data_ptr()),                          \
                                  (const C_TYPE*)(a.data_ptr()),                    \
                                  (const C_TYPE*)(b.data_ptr()),                    \
                                  gamma,                                            \
                                  total_elems,                                      \
                                  InferenceContext::Instance().GetCurrentStream()); \
    }

at::Tensor& _vector_add(at::Tensor& a, at::Tensor& b, float gamma)
{
    const int total_elems = a.numel();

    DISPATCH_VECTOR_ADD(Float, float)
    DISPATCH_VECTOR_ADD(Half, __half)
#ifdef BF16_AVAILABLE
    DISPATCH_VECTOR_ADD(BFloat16, __nv_bfloat16)
#endif

    return a;
}

std::vector<at::Tensor> apply_rotary_pos_emb(at::Tensor& mixed_query,
                                             at::Tensor& key_layer,
                                             unsigned rotary_dim,
                                             unsigned offset,
                                             unsigned num_heads,
                                             bool rotate_half)
{
    auto query_cont = mixed_query.contiguous();
    auto key_cont = key_layer.contiguous();

    unsigned bsz = mixed_query.size(0);
    unsigned head_size = mixed_query.size(2) / num_heads;
    unsigned seq_len = mixed_query.size(1);

    if (mixed_query.scalar_type() == at::kFloat)
        launch_apply_rotary_pos_emb<float>((float*)query_cont.data_ptr(),
                                           (float*)key_cont.data_ptr(),
                                           head_size,
                                           seq_len,
                                           rotary_dim,
                                           offset,
                                           num_heads,
                                           bsz,
                                           InferenceContext::Instance().GetCurrentStream(),
                                           InferenceContext::Instance().GetMaxTokenLength());
    else
        launch_apply_rotary_pos_emb<__half>((__half*)query_cont.data_ptr(),
                                            (__half*)key_cont.data_ptr(),
                                            head_size,
                                            seq_len,
                                            rotary_dim,
                                            offset,
                                            num_heads,
                                            bsz,
                                            InferenceContext::Instance().GetCurrentStream(),
                                            InferenceContext::Instance().GetMaxTokenLength());
    return {query_cont, key_cont};
}

#define DISPATCH_MOE_RESIDUAL(T_TYPE, C_TYPE)                                           \
    if (moe_res.scalar_type() == torch::T_TYPE) {                                       \
        launch_moe_res_matmul<C_TYPE>((C_TYPE*)moe_res.data_ptr(),                      \
                                      (C_TYPE*)coef.data_ptr(),                         \
                                      (C_TYPE*)output.data_ptr(),                       \
                                      M,                                                \
                                      N,                                                \
                                      InferenceContext::Instance().GetCurrentStream()); \
    }

at::Tensor moe_res_matmul(at::Tensor& moe_res, at::Tensor& coef, at::Tensor& output)
{
    int M = moe_res.size(0) * moe_res.size(1);
    int N = moe_res.size(2);
    InferenceContext::Instance().SynchComm();

    DISPATCH_MOE_RESIDUAL(kFloat, float)
    DISPATCH_MOE_RESIDUAL(kHalf, __half)
#ifdef BF16_AVAILABLE
    DISPATCH_MOE_RESIDUAL(kBFloat16, __nv_bfloat16)
#endif

    return output;
}

void ds_release_workspace() { InferenceContext::Instance().release_workspace(); }

bool ds_retake_workspace() { return InferenceContext::Instance().retake_workspace(); }

template <typename T>
at::Tensor ds_dequantize(at::Tensor& weight, at::Tensor& qscale, int groups)
{
    auto options = at::TensorOptions()
                       .dtype(torch::kFloat16)
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto weight16 = at::empty({weight.size(0), weight.size(1)}, options);

    launch_dequantize((T*)weight16.data_ptr(),
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(0),
                      weight.size(1),
                      groups,
                      InferenceContext::Instance().GetCurrentStream());

    return weight16;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_context_int8",
          &ds_softmax_context1<__half>,
          "DeepSpeed attention with int8 (CUDA)");

    // The following functions handle type dispatching internally
    m.def("gated_activation", &ds_gated_activation, "DeepSpeed Bias GEGLU (CUDA)");
    m.def("layer_norm", &ds_layer_norm, "DeepSpeed layer norm (CUDA)");
    m.def(
        "_layer_norm_residual", &ds_layer_norm_residual, "DeepSpeed layer norm + residual (CUDA)");
    m.def("layer_norm_residual_store_pre_ln_res",
          &ds_layer_norm_residual_store_pre_ln_res,
          "DeepSpeed layer norm + store pre Layernorm residual (CUDA)");
    m.def("rms_norm", &ds_rms_norm, "DeepSpeed rms norm (CUDA)");
    m.def("pre_rms_norm", &ds_pre_rms_norm, "DeepSpeed pre rms norm (CUDA)");
    m.def("_vector_add", &_vector_add, "DeepSpeed vector add (CUDA)");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("moe_res_matmul", &moe_res_matmul, "DeepSpeed moe residual matmul (CUDA)");
    m.def("reset_cache", &reset_cache, "Reset Cache for generation tasks");
    m.def("release_workspace", &ds_release_workspace, "DeepSpeed Release Workspace");
    m.def("retake_workspace", &ds_retake_workspace, "DeepSpeed Retake Workspace");

    // The following functions are templated and need to be explicitly instantiated and bound
    // to different python methods
#define DEF_OPS(_name, _dtype)                                                                    \
    m.def("softmax_" #_name, &ds_softmax<_dtype>, "DeepSpeed SoftMax with " #_name " (CUDA)");    \
    m.def("softmax_context_" #_name,                                                              \
          &ds_softmax_context<_dtype>,                                                            \
          "DeepSpeed attention with " #_name " (CUDA)");                                          \
    m.def("bias_gelu_" #_name, &ds_bias_gelu<_dtype>, "DeepSpeed Gelu with " #_name " (CUDA)");   \
    m.def("bias_add_" #_name, &ds_bias_add<_dtype>, "DeepSpeed Bias Add with " #_name " (CUDA)"); \
    m.def("bias_relu_" #_name, &ds_bias_relu<_dtype>, "DeepSpeed ReLU with " #_name " (CUDA)");   \
    m.def("bias_residual_" #_name,                                                                \
          &ds_bias_residual<_dtype>,                                                              \
          "DeepSpeed residual-bias add with " #_name " (CUDA)");                                  \
    m.def("qkv_gemm_" #_name, &ds_qkv_gemm<_dtype>, "DeepSpeed qkv gemm with " #_name " (CUDA)"); \
    m.def("rms_qkv_gemm_" #_name,                                                                 \
          &ds_rms_qkv<_dtype>,                                                                    \
          "DeepSpeed rms qkv gemm with " #_name " (CUDA)");                                       \
    m.def("mlp_gemm_" #_name, &ds_mlp_gemm<_dtype>, "DeepSpeed mlp with " #_name " (CUDA)");      \
    m.def("rms_mlp_gemm_" #_name,                                                                 \
          &ds_rms_mlp_gemm<_dtype>,                                                               \
          "DeepSpeed rms mlp gemm with " #_name " (CUDA)");                                       \
    m.def("vector_matmul_" #_name,                                                                \
          &ds_vector_matmul<_dtype>,                                                              \
          "DeepSpeed vector-MM with " #_name " (CUDA)");                                          \
    m.def("linear_layer_" #_name,                                                                 \
          &ds_linear_layer<_dtype>,                                                               \
          "DeepSpeed linear_layer with " #_name " (CUDA)");                                       \
    m.def("fused_gemm_gelu_" #_name,                                                              \
          &fused_gemm_gelu<_dtype>,                                                               \
          "DeepSpeed mlp with " #_name " (CUDA)");                                                \
    m.def("residual_add_bias_" #_name,                                                            \
          &residual_add_bias<_dtype>,                                                             \
          "DeepSpeed residual add with " #_name " (CUDA)");                                       \
    m.def("einsum_sec_sm_ecm_" #_name,                                                            \
          &einsum_sec_sm_ecm<_dtype>,                                                             \
          "DeepSpeed vector-MM with " #_name " (CUDA)");                                          \
    m.def("add_padding_" #_name,                                                                  \
          &add_padding<_dtype>,                                                                   \
          "DeepSpeed residual add with " #_name " (CUDA)");                                       \
    m.def("pad_transform_" #_name,                                                                \
          &padd_add_transform<_dtype>,                                                            \
          "DeepSpeed residual add with " #_name " (CUDA)");                                       \
    m.def("allocate_workspace_" #_name,                                                           \
          &allocate_workspace<_dtype>,                                                            \
          "DeepSpeed memory allocation for GPT inference with " #_name " (CUDA)");                \
    m.def("dequantize_" #_name,                                                                   \
          &ds_dequantize<_dtype>,                                                                 \
          "DeepSpeed dequantize with " #_name " (CUDA)")

    DEF_OPS(fp32, float);
    DEF_OPS(fp16, __half);
#ifdef BF16_AVAILABLE
    DEF_OPS(bf16, __nv_bfloat16);
#endif
}
