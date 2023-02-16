/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

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
enum class ActivationFuncType { UNKNOWN = 0, GELU = 1, ReLU = 2 };

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
                           Context::Instance().GetCurrentStream(async_op));

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
                        unsigned max_out_tokens = 1024)
{
    Context::Instance().GenWorkSpace(num_layers,
                                     num_heads,
                                     batch_size,
                                     prompt_length,
                                     hidden_dim,
                                     mp_size,
                                     external_cache,
                                     sizeof(T),
                                     rank,
                                     max_out_tokens);
}

template <typename T>
at::Tensor einsum_sec_sm_ecm(at::Tensor& Q, at::Tensor& W)
{
    auto options = at::TensorOptions()
                       .dtype(Q.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    float alpha = 1;
    float gemm_beta = 0.0;

    /*
    // Reallocate memory if we received a new prompt
    if (!workspace || input.size(1) != 1) {
        allocate_workspace<T>(W.size(1), Context::Instance().GetMaxTokenLenght(), Q.size(0), 1,
    head_size); workspace = (T*)Context::Instance().GetWorkSpace();
    }
    */

    auto O = at::from_blob(workspace, {Q.size(1), Q.size(2), W.size(1)}, options);
    unsigned m = W.size(1);
    unsigned n = Q.size(1) * Q.size(2);
    unsigned k = Q.size(0);
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
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

    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
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
                           Context::Instance().GetCurrentStream(false));
    alpha = 1.0;
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
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
    T* workspace = (T*)Context::Instance().GetAttentionUnfusedWorkspace();

    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
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
                                Context::Instance().GetMaxTokenLenght() * k,
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
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
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
                                Context::Instance().GetMaxTokenLenght() * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

void reset_cache() { Context::Instance().reset_tokens(); }

template <typename T>
std::vector<at::Tensor> ds_softmax_context(at::Tensor& query_key_value,
                                           at::Tensor& attn_mask,
                                           int rotary_dim,
                                           bool rotate_half,
                                           bool rotate_every_two,
                                           int heads,
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
    unsigned hidden_dim = query_key_value.size(2) / 3;

    bool is_prompt = (seq_len > 1);

    if (is_prompt) Context::Instance().reset_tokens(seq_len);
    unsigned soft_len = Context::Instance().current_tokens();

    int k = hidden_dim / heads;
    auto options = at::TensorOptions()
                       .dtype(query_key_value.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    T* workspace = (T*)Context::Instance().GetWorkSpace();
    size_t buf_size = bsz * seq_len * hidden_dim;
    auto output = torch::from_blob(workspace + 4 * buf_size, {bsz, seq_len, hidden_dim}, options);

    auto query_cont = workspace + 8 * buf_size;
    size_t offset = 16 * (hidden_dim * bsz * Context::Instance().GetMaxTokenLenght()) +
                    layer_id * 2 * bsz * Context::Instance().GetMaxTokenLenght() * hidden_dim;
    unsigned all_tokens = soft_len;
    auto kv_cache = workspace + offset + (hidden_dim / heads) * (is_prompt ? 0 : soft_len - 1);
    size_t value_offset = bsz * Context::Instance().GetMaxTokenLenght() * hidden_dim;

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
                                      rotary_dim,
                                      rotate_half,
                                      rotate_every_two,
                                      Context::Instance().GetCurrentStream(),
                                      3,
                                      Context::Instance().GetMaxTokenLenght());
    if (rotary_dim > 0 && rotate_half)
        launch_apply_rotary_pos_emb(query_cont,
                                    kv_cache,
                                    k,
                                    seq_len,
                                    rotary_dim,
                                    (is_prompt ? 0 : soft_len - 1),
                                    heads,
                                    bsz,
                                    rotate_half,
                                    rotate_every_two,
                                    Context::Instance().GetCurrentStream(),
                                    Context::Instance().GetMaxTokenLenght());

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
                               Context::Instance().GetCurrentStream(false),
                               1);

    if (layer_id == num_layers - 1) Context::Instance().advance_tokens();
    auto prev_key = torch::from_blob(workspace + offset, {bsz, heads, all_tokens, k}, options);
    auto prev_value =
        torch::from_blob(workspace + offset + value_offset, {bsz, heads, all_tokens, k}, options);
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
                     Context::Instance().GetCurrentStream());
    return input_cont;
}

at::Tensor ds_bias_geglu(at::Tensor& activation, at::Tensor& bias)
{
    /*
    Used in FF of Stable diffusion
    */

    const int batch_size = activation.size(0);
    const int seq_len = activation.size(1);
    const int channels = activation.size(2);

    const int rows = batch_size * seq_len;
    // Dimensionality is cut in half
    const int out_channels = channels / 2;

    auto output = at::empty({batch_size, seq_len, out_channels}, activation.options());

    if (activation.options().dtype() == torch::kFloat32) {
        launch_fused_bias_geglu((float*)output.data_ptr(),
                                (const float*)activation.data_ptr(),
                                (const float*)bias.data_ptr(),
                                rows,
                                channels,
                                Context::Instance().GetCurrentStream());
    } else {
        launch_fused_bias_geglu((__half*)output.data_ptr(),
                                (const __half*)activation.data_ptr(),
                                (const __half*)bias.data_ptr(),
                                rows,
                                channels,
                                Context::Instance().GetCurrentStream());
    }

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
                     Context::Instance().GetCurrentStream());
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
                    Context::Instance().GetCurrentStream());
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
    //                      Context::Instance().GetCurrentStream());
    return input_cont;
}

at::Tensor ds_layer_norm(at::Tensor& input, at::Tensor& gamma, at::Tensor& beta, float epsilon)
{
    const int rows = input.size(0) * input.size(1);
    const int elems_per_row = input.size(2);
    auto output = at::empty_like(input);

    if (input.options().dtype() == torch::kFloat16) {
        launch_fused_ln((__half*)output.data_ptr(),
                        (const __half*)input.data_ptr(),
                        (const __half*)gamma.data_ptr(),
                        (const __half*)beta.data_ptr(),
                        epsilon,
                        rows,
                        elems_per_row,
                        Context::Instance().GetCurrentStream());
    } else {
        launch_fused_ln((float*)output.data_ptr(),
                        (const float*)input.data_ptr(),
                        (const float*)gamma.data_ptr(),
                        (const float*)beta.data_ptr(),
                        epsilon,
                        rows,
                        elems_per_row,
                        Context::Instance().GetCurrentStream());
    }

    return output;
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
                    Context::Instance().GetCurrentStream());
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

    if (input.options().dtype() == torch::kFloat16) {
        launch_fused_residual_ln((__half*)output.data_ptr(),
                                 (const __half*)input.data_ptr(),
                                 (const __half*)residual.data_ptr(),
                                 (const __half*)bias.data_ptr(),
                                 (const __half*)gamma.data_ptr(),
                                 (const __half*)beta.data_ptr(),
                                 epsilon,
                                 rows,
                                 elems_per_row,
                                 Context::Instance().GetCurrentStream());
    } else {
        launch_fused_residual_ln((float*)output.data_ptr(),
                                 (const float*)input.data_ptr(),
                                 (const float*)residual.data_ptr(),
                                 (const float*)bias.data_ptr(),
                                 (const float*)gamma.data_ptr(),
                                 (const float*)beta.data_ptr(),
                                 epsilon,
                                 rows,
                                 elems_per_row,
                                 Context::Instance().GetCurrentStream());
    }

    return output;
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

    if (input.options().dtype() == torch::kFloat16) {
        launch_fused_residual_ln_store_pre_ln_res((__half*)norm_output.data_ptr(),
                                                  (__half*)res_output.data_ptr(),
                                                  (const __half*)input.data_ptr(),
                                                  (const __half*)residual.data_ptr(),
                                                  (const __half*)bias.data_ptr(),
                                                  (const __half*)gamma.data_ptr(),
                                                  (const __half*)beta.data_ptr(),
                                                  epsilon,
                                                  rows,
                                                  elems_per_row,
                                                  Context::Instance().GetCurrentStream());
    } else {
        launch_fused_residual_ln_store_pre_ln_res((float*)norm_output.data_ptr(),
                                                  (float*)res_output.data_ptr(),
                                                  (const float*)input.data_ptr(),
                                                  (const float*)residual.data_ptr(),
                                                  (const float*)bias.data_ptr(),
                                                  (const float*)gamma.data_ptr(),
                                                  (const float*)beta.data_ptr(),
                                                  epsilon,
                                                  rows,
                                                  elems_per_row,
                                                  Context::Instance().GetCurrentStream());
    }

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
    // T* weight16 = (T*)Context::Instance().GetWorkSpace() + 12 * hidden_size * bsz;

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
                      Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
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
                              bool q_int8)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    workspace += (3 * bsz * input.size(2));
    ds_layer_norm_internal<T>(workspace, input, gamma, beta, epsilon);

    if (q_int8) {
        quantized_gemm<T>(
            output.data_ptr(), workspace, weight, q_scale, q_scale.size(0), bsz, input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;

        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
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
                        q_int8 ? weight.size(0) : weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());
    return torch::from_blob(workspace, input.sizes(), input.options());
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
                                    unsigned num_layers,
                                    bool external_cache,
                                    unsigned mp_size,
                                    unsigned rank,
                                    bool q_int8)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    int out_size = q_int8 ? weight.size(0) : weight.size(1);

    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::from_blob(workspace, {input.size(0), input.size(1), out_size}, options);
    auto inp_norm = qkv_unfused_cublas<T>(
        output, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8);

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
                      Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
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
at::Tensor ds_qkv_gemm_int8(at::Tensor& input,
                            at::Tensor& weight,
                            at::Tensor& bias,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            const float epsilon,
                            at::Tensor& q_scale,
                            int groups,
                            bool add_bias)
{
    int bsz = input.size(0) * input.size(1);
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    auto inp_norm = ds_layer_norm(input_cont, gamma, beta, epsilon);

    quantized_gemm<T>(output, inp_norm, weight, q_scale, groups, 0);
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());

    return output;
}

template <typename T>
at::Tensor ds_linear_layer(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& bias,
                           bool add_bias,
                           bool do_flash_attn,
                           int num_heads)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int head_size = input_cont.size(2) / num_heads;
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    auto output = at::from_blob(workspace, {input.size(0), input.size(1), weight.size(1)}, options);

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());

    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   weight.size(1),
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
                        weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());
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
                     Context::Instance().GetCurrentStream());

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
                false,
                false,
                Context::Instance().GetCurrentStream(),
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
                false,
                false,
                Context::Instance().GetCurrentStream(),
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
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    T* key_pad_ptr = workspace + padded_head_size * query.size(0) * query.size(1) * query.size(2);
    T* value_pad_ptr = key_pad_ptr + padded_head_size * query.size(0) * query.size(1) * 128;
    pad_head_seq(workspace,
                 (T*)query.data_ptr(),
                 query.size(0) * query.size(1),
                 query.size(2),
                 query.size(2),
                 head_size,
                 padded_head_size,
                 Context::Instance().GetCurrentStream());
    pad_head_seq(key_pad_ptr,
                 (T*)key.data_ptr(),
                 query.size(0) * query.size(1),
                 key.size(2),
                 128,
                 head_size,
                 padded_head_size,
                 Context::Instance().GetCurrentStream());
    pad_head_seq(value_pad_ptr,
                 (T*)value.data_ptr(),
                 query.size(0) * query.size(1),
                 key.size(2),
                 128,
                 head_size,
                 padded_head_size,
                 Context::Instance().GetCurrentStream());
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
    T* workspace = (T*)Context::Instance().GetWorkSpace();
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
                                  Context::Instance().GetCurrentStream());
    launch_pad_add_transform_0213(key_pad_ptr,
                                  (T*)key.data_ptr(),
                                  key.size(0),
                                  key.size(2),
                                  key.size(1),
                                  key_value_length,
                                  heads,
                                  padded_head_size,
                                  Context::Instance().GetCurrentStream());
    launch_pad_add_transform_0213(value_pad_ptr,
                                  (T*)value.data_ptr(),
                                  value.size(0),
                                  value.size(2),
                                  value.size(1),
                                  key_value_length,
                                  heads,
                                  padded_head_size,
                                  Context::Instance().GetCurrentStream());
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
at::Tensor ds_linear_layer_int8(at::Tensor& input,
                                at::Tensor& weight,
                                at::Tensor& bias,
                                at::Tensor& q_scale,
                                int groups)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    int bsz = input_cont.size(0) * input_cont.size(1);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    quantized_gemm<T>(output, input_cont, weight, q_scale, groups, 0);
    launch_bias_add((T*)output.data_ptr(),
                    (T*)bias.data_ptr(),
                    weight.size(1),
                    bsz,
                    Context::Instance().GetCurrentStream());
    return output;
}

template <typename T>
at::Tensor ds_vector_matmul(at::Tensor& input,
                            at::Tensor& weight,
                            bool async_op,
                            at::Tensor& q_scale,
                            bool q_int8)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    int out_size = q_int8 ? weight.size(0) : weight.size(1);
    int bsz = input.size(0) * input.size(1);

    T* workspace = (T*)Context::Instance().GetWorkSpace();
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
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream(async_op));
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
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
                              ActivationFuncType act_func_type)
{
    int bsz = input.size(0) * input.size(1);
    T* inp_norm =
        (T*)Context::Instance().GetWorkSpace() + torch::numel(input) + torch::numel(output);
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
                                 Context::Instance().GetCurrentStream());
    } else {
        ds_layer_norm_internal(inp_norm, input, gamma, beta, epsilon);
    }
    if (q_int8) {
        quantized_gemm<T>(
            intermediate, inp_norm, weight, q_scale, q_scale.size(0), bsz, input.size(2));
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
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
                         q_int8 ? weight.size(0) : weight.size(1),
                         bsz,
                         Context::Instance().GetCurrentStream());
    } else if (act_func_type == ActivationFuncType::ReLU) {
        launch_bias_relu(intermediate,
                         (T*)bias.data_ptr(),
                         q_int8 ? weight.size(0) : weight.size(1),
                         bsz,
                         Context::Instance().GetCurrentStream());
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
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight1.size(1),
                       bsz,
                       weight1.size(0),
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
                                    int activation_type)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int out_size = q_int8 ? weight_out.size(0) : weight_out.size(1);
    auto output = at::from_blob((T*)Context::Instance().GetWorkSpace() + torch::numel(input),
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
                                         act_func_type);

    return {output, res_add};
}

template <typename T>
std::vector<at::Tensor> ds_mlp_gemm_int8(at::Tensor& input,
                                         at::Tensor& residual,
                                         at::Tensor& input_bias,
                                         at::Tensor& weight,
                                         at::Tensor& bias,
                                         at::Tensor& gamma,
                                         at::Tensor& beta,
                                         const float epsilon,
                                         at::Tensor& q_scale,
                                         int groups,
                                         bool preLayerNorm)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    int bsz = input_cont.size(0) * input_cont.size(1);
    auto inp_norm = at::empty_like(input_cont);

    auto residual_add = (preLayerNorm ? at::empty_like(input_cont) : inp_norm);
    quantized_gemm<T>(output, inp_norm, weight, q_scale, groups, 0);
    launch_bias_gelu((T*)output.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());

    return {output, residual_add};
}

template <typename T>
at::Tensor fused_gemm_gelu(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& weight_scale,
                           at::Tensor& bias,
                           at::Tensor& weight_out,
                           at::Tensor& weight_out_scale,
                           const float epsilon,
                           bool preLayerNorm,
                           bool q_int8,
                           bool async_op)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    int intm_dim = q_int8 ? weight.size(0) : weight.size(1);

    // auto output = at::from_blob((T*)Context::Instance().GetWorkSpace() + torch::numel(input),
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
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
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
                     Context::Instance().GetCurrentStream());

    int out_size = q_int8 ? weight_out.size(0) : weight_out.size(1);
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
        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
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
    // cudaEventRecord(Context::Instance().GetCompEvent(2),
    //                Context::Instance().GetCurrentStream(true));
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
                             Context::Instance().GetCurrentStream());
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
            Context::Instance().GetCurrentStream());
    return residual;
}

std::vector<at::Tensor> apply_rotary_pos_emb(at::Tensor& mixed_query,
                                             at::Tensor& key_layer,
                                             unsigned rotary_dim,
                                             unsigned offset,
                                             unsigned num_heads,
                                             bool rotate_half,
                                             bool rotate_every_two)
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
                                           rotate_half,
                                           rotate_every_two,
                                           Context::Instance().GetCurrentStream(),
                                           Context::Instance().GetMaxTokenLenght());
    else
        launch_apply_rotary_pos_emb<__half>((__half*)query_cont.data_ptr(),
                                            (__half*)key_cont.data_ptr(),
                                            head_size,
                                            seq_len,
                                            rotary_dim,
                                            offset,
                                            num_heads,
                                            bsz,
                                            rotate_half,
                                            rotate_every_two,
                                            Context::Instance().GetCurrentStream(),
                                            Context::Instance().GetMaxTokenLenght());
    return {query_cont, key_cont};
}

template <typename T>
at::Tensor fused_gemm_gelu_int8(at::Tensor& input,
                                at::Tensor& weight,
                                at::Tensor& bias,
                                const float epsilon,
                                at::Tensor& q_scale,
                                int groups,
                                bool preLayerNorm)
{
    auto input_cont = input.contiguous();
    auto options = at::TensorOptions()
                       .dtype(input_cont.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output = at::empty({input_cont.size(0), input_cont.size(1), weight.size(1)}, options);

    int bsz = input_cont.size(0) * input_cont.size(1);

    quantized_gemm<T>(output, input_cont, weight, q_scale, groups, 0);
    launch_bias_gelu((T*)output.data_ptr(),
                     (T*)bias.data_ptr(),
                     weight.size(1),
                     bsz,
                     Context::Instance().GetCurrentStream());

    return output;
}

at::Tensor moe_res_matmul(at::Tensor& moe_res, at::Tensor& coef, at::Tensor& output)
{
    int M = moe_res.size(0) * moe_res.size(1);
    int N = moe_res.size(2);
    Context::Instance().SynchComm();
    if (moe_res.scalar_type() == at::kFloat) {
        launch_moe_res_matmul<float>((float*)moe_res.data_ptr(),
                                     (float*)coef.data_ptr(),
                                     (float*)output.data_ptr(),
                                     M,
                                     N,
                                     at::cuda::getCurrentCUDAStream());
    } else {
        launch_moe_res_matmul<__half>((__half*)moe_res.data_ptr(),
                                      (__half*)coef.data_ptr(),
                                      (__half*)output.data_ptr(),
                                      M,
                                      N,
                                      at::cuda::getCurrentCUDAStream());
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_fp32", &ds_softmax<float>, "DeepSpeed SoftMax with fp32 (CUDA)");
    m.def("softmax_fp16", &ds_softmax<__half>, "DeepSpeed SoftMax with fp16 (CUDA)");
    m.def(
        "softmax_context_fp32", &ds_softmax_context<float>, "DeepSpeed attention with fp32 (CUDA)");
    m.def("softmax_context_fp16",
          &ds_softmax_context<__half>,
          "DeepSpeed attention with fp16 (CUDA)");
    m.def("softmax_context_int8",
          &ds_softmax_context1<__half>,
          "DeepSpeed attention with int8 (CUDA)");
    m.def("bias_gelu_fp32", &ds_bias_gelu<float>, "DeepSpeed Gelu with fp32 (CUDA)");
    m.def("bias_gelu_fp16", &ds_bias_gelu<__half>, "DeepSpeed Gelu with fp16 (CUDA)");
    m.def("bias_geglu", &ds_bias_geglu, "DeepSpeed Bias GEGLU (CUDA)");
    m.def("bias_add_fp32", &ds_bias_add<float>, "DeepSpeed Bias Add with fp32 (CUDA)");
    m.def("bias_add_fp16", &ds_bias_add<__half>, "DeepSpeed Gelu with fp16 (CUDA)");
    m.def("bias_relu_fp32", &ds_bias_relu<float>, "DeepSpeed ReLU with fp32 (CUDA)");
    m.def("bias_relu_fp16", &ds_bias_relu<__half>, "DeepSpeed ReLU with fp16 (CUDA)");
    m.def("bias_residual_fp32",
          &ds_bias_residual<float>,
          "DeepSpeed residual-bias add with fp32 (CUDA)");
    m.def("bias_residual_fp16",
          &ds_bias_residual<__half>,
          "DeepSpeed residual-bias add with fp16 (CUDA)");
    m.def("layer_norm", &ds_layer_norm, "DeepSpeed layer norm (CUDA)");
    m.def(
        "_layer_norm_residual", &ds_layer_norm_residual, "DeepSpeed layer norm + residual (CUDA)");
    m.def("layer_norm_residual_store_pre_ln_res",
          &ds_layer_norm_residual_store_pre_ln_res,
          "DeepSpeed layer norm + store pre Layernorm residual (CUDA)");
    m.def("qkv_gemm_fp32", &ds_qkv_gemm<float>, "DeepSpeed qkv gemm with fp32 (CUDA)");
    m.def("qkv_gemm_fp16", &ds_qkv_gemm<__half>, "DeepSpeed qkv gemm with fp16 (CUDA)");
    m.def("qkv_gemm_int8", &ds_qkv_gemm_int8<__half>, "DeepSpeed qkv gemm with int8 (CUDA)");
    m.def("mlp_gemm_fp32", &ds_mlp_gemm<float>, "DeepSpeed mlp with fp32 (CUDA)");
    m.def("mlp_gemm_fp16", &ds_mlp_gemm<__half>, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("mlp_gemm_int8", &ds_mlp_gemm_int8<__half>, "DeepSpeed mlp with int8 (CUDA)");
    m.def("vector_matmul_fp32", &ds_vector_matmul<float>, "DeepSpeed vector-MM with fp32 (CUDA)");
    m.def("vector_matmul_fp16", &ds_vector_matmul<__half>, "DeepSpeed vector-MM with fp16 (CUDA)");
    m.def("vector_matmul_int8",
          &ds_vector_matmul_int8<__half>,
          "DeepSpeed vector-MM with int8 (CUDA)");
    m.def("linear_layer_fp32", &ds_linear_layer<float>, "DeepSpeed linear_layer with fp32 (CUDA)");
    m.def("linear_layer_fp16", &ds_linear_layer<__half>, "DeepSpeed linear_layer with fp16 (CUDA)");
    m.def("linear_layer_int8",
          &ds_linear_layer_int8<__half>,
          "DeepSpeed linear_layer with int8 (CUDA)");
    m.def("fused_gemm_gelu_fp32", &fused_gemm_gelu<float>, "DeepSpeed mlp with fp32 (CUDA)");
    m.def("fused_gemm_gelu_fp16", &fused_gemm_gelu<__half>, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("residual_add_bias_fp32",
          &residual_add_bias<float>,
          "DeepSpeed residual add with fp32 (CUDA)");
    m.def("residual_add_bias_fp16",
          &residual_add_bias<__half>,
          "DeepSpeed residual add with fp16 (CUDA)");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("einsum_sec_sm_ecm_fp32",
          &einsum_sec_sm_ecm<float>,
          "DeepSpeed vector-MM with fp32 (CUDA)");

    m.def("einsum_sec_sm_ecm_fp16",
          &einsum_sec_sm_ecm<__half>,
          "DeepSpeed vector-MM with fp16 (CUDA)");
    m.def("moe_res_matmul", &moe_res_matmul, "DeepSpeed moe residual matmul (CUDA)");
    m.def("add_padding_fp32", &add_padding<float>, "DeepSpeed residual add with fp32 (CUDA)");
    m.def("add_padding_fp16", &add_padding<__half>, "DeepSpeed residual add with fp16 (CUDA)");
    m.def("pad_transform_fp32",
          &padd_add_transform<float>,
          "DeepSpeed residual add with fp32 (CUDA)");
    m.def("pad_transform_fp16",
          &padd_add_transform<__half>,
          "DeepSpeed residual add with fp16 (CUDA)");
    m.def("allocate_workspace_fp32",
          &allocate_workspace<float>,
          "DeepSpeed memory allocation for GPT inference with fp32 (CUDA)");
    m.def("allocate_workspace_fp16",
          &allocate_workspace<__half>,
          "DeepSpeed memory allocation for GPT inference with fp16 (CUDA)");
    m.def("reset_cache", &reset_cache, "Reset Cache for generation tasks");
}
