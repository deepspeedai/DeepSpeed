// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cuda_runtime_api.h>
#include <curand.h>
#include <memory>
#include <vector>
#include "cublas_v2.h"
#include "cuda.h"
#include "dropout.h"
#include "feed_forward.h"
#include "gelu.h"
#include "general_kernels.h"
#include "normalize_layer.h"
#include "softmax.h"
#include "strided_batch_gemm.h"

struct BertGemmAlgos {
    int m_gemm_qkv_algo;
    int m_gemm_inter_algo;
    int m_gemm_output_algo;
    int m_gemm_batch1_algo;
    int m_gemm_batch2_algo;

    BertGemmAlgos()
        : m_gemm_qkv_algo(-1),
          m_gemm_inter_algo(-1),
          m_gemm_output_algo(-1),
          m_gemm_batch1_algo(-1),
          m_gemm_batch2_algo(-1)
    {
    }
};

template <typename T>
class BertTransformerLayer {
public:
    BertTransformerLayer(unsigned layer_id,
                         unsigned batch_size,
                         unsigned hidden_size,
                         unsigned num_heads,
                         unsigned intermediate_size,
                         unsigned seq_length,
                         float attn_dropout_ratio,
                         float hidden_output_dropout_ratio,
                         float layer_norm_eps,
                         bool pre_or_postLayerNorm,
                         const std::vector<std::array<int, 3>>& gemm_algos,
                         bool attn_dropout_checkpoint,
                         bool normalize_invertible,
                         bool gelu_checkpoint,
                         bool stochastic_mode);

    virtual ~BertTransformerLayer();

    void Forward(unsigned bsz,
                 const T* input_ptr,
                 const T* input_mask_ptr,
                 const T* attn_qkvw_ptr,
                 const T* attn_qkvb_ptr,
                 const T* attn_ow_ptr,
                 const T* attn_ob_ptr,
                 const T* attn_nw_ptr,
                 const T* attn_nb_ptr,
                 const T* inter_w_ptr,
                 const T* inter_b_ptr,
                 const T* output_w_ptr,
                 const T* output_b_ptr,
                 const T* norm_w_ptr,
                 const T* norm_b_ptr,
                 T* out_ptr,
                 T* inp_norm_ptr,
                 T* q_tf_ptr,
                 T* k_tf_ptr,
                 T* v_tf_ptr,
                 T* softmax_output_ptr,
                 T* ctx_bufB_ptr,
                 T* attn_o_inp_ptr,
                 T* add_res_ptr,
                 T* ff1_inp_ptr,
                 T* gelu_inp_ptr,
                 T* ff2_inp_ptr);

    void Backward(unsigned bsz,
                  const T* grad_output_ptr,
                  const T* input_ptr,
                  const T* output_ptr,
                  const T* inp_norm_ptr,
                  const T* q_tf_ptr,
                  const T* k_tf_ptr,
                  const T* v_tf_ptr,
                  const T* softmax_output_ptr,
                  const T* ctx_bufB_ptr,
                  const T* attn_o_inp_ptr,
                  const T* add_res_ptr,
                  const T* ff1_inp_ptr,
                  const T* gelu_inp_ptr,
                  const T* ff2_inp_ptr,
                  const T* input_mask_ptr,
                  const T* attn_qkvw_ptr,
                  const T* attn_ow_ptr,
                  const T* attn_nw_ptr,
                  const T* attn_nb_ptr,
                  const T* inter_w_ptr,
                  const T* inter_b_ptr,
                  const T* output_w_ptr,
                  const T* norm_w_ptr,
                  const T* norm_b_ptr,

                  T* grad_input_ptr,
                  T* grad_attn_qkvw_ptr,
                  T* grad_attn_qkvb_ptr,
                  T* grad_attn_ow_ptr,
                  T* grad_attn_ob_ptr,
                  T* grad_attn_nw_ptr,
                  T* grad_attn_nb_ptr,
                  T* grad_inter_w_ptr,
                  T* grad_inter_b_ptr,
                  T* grad_output_w_ptr,
                  T* grad_output_b_ptr,
                  T* grad_norm_w_ptr,
                  T* grad_norm_b_ptr);

    void SetIntermediateBuffers(uint8_t* attn_prob_dropout_mask_ptr,
                                uint8_t* attn_output_dropout_mask_ptr,
                                uint8_t* layer_output_dropout_mask_ptr,
                                T* layer_norm_var,
                                T* layer_norm_mean,
                                T* attn_layer_norm_var,
                                T* attn_layer_norm_mean);

    inline unsigned GetBatchSize() const { return _batch_size; }
    inline unsigned GetNumHeads() const { return _heads; }
    inline unsigned GetSeqLength() const { return _seq_length; }
    inline unsigned GetIntermediateSize() const { return _intermediate_size; }

    void SetSeqLength(unsigned seq_len);
    inline unsigned GetHiddenSize() const { return _hidden_size; }
    void SetTrainingMode(bool training);
    inline bool IsTrainingMode() const { return _training; }
    inline bool GeluCheckpoint() const { return _gelu_checkpoint; }

private:
    void Initialize();
    size_t getWorkspaceSize(int maxBatchSize) const;

    // Params
    unsigned _layer_id;
    unsigned _batch_size;
    unsigned _hidden_size;
    unsigned _heads;
    unsigned _size_per_head;
    unsigned _intermediate_size;
    unsigned _seq_length;

    bool _pre_or_postLayerNorm;

    cublasHandle_t _cublasHandle;
    cudaStream_t _stream;

    // layers
    FeedForward<T> _qkv_linear;
    FeedForward<T> _attn_out_linear;
    Normalize_Layer<T> _attn_layer_norm;
    Normalize_Layer<T> _layer_norm;
    Normalize_Layer<T>* _last_normalize;
    FeedForward<T> _ff1, _ff2;
    Softmax<T> _softmax;
    Gelu<T> _gelu;
    Dropout<T> _attn_prob_dropout;
    Dropout<T> _attn_output_dropout;
    Dropout<T> _layer_output_dropout;
    StridedBatchGemm<T> _attn_scores;
    StridedBatchGemm<T> _attn_context;

    bool _training;

    // Memory saving flags
    bool _attn_dropout_checkpoint;
    bool _normalize_invertible;
    bool _gelu_checkpoint;

    // High Performance flags
    bool _stochastic_mode;
};
