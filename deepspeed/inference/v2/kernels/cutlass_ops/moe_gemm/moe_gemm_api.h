// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "activation_type.h"
#include "weight_variant.h"

namespace fastertransformer {

template <typename T, /*The type used for activations/scales/compute*/
          WeightVariant V /* The type for the MoE weights */>
class MoeGemmRunner {
public:
    MoeGemmRunner();

    void moe_gemm_bias_act(const T* A,
                           const char* B,
                           const T* weight_scales,
                           const T* biases,
                           T* C,
                           int64_t* total_rows_before_expert,
                           int64_t total_rows,
                           int64_t gemm_n,
                           int64_t gemm_k,
                           int num_experts,
                           ActivationType activation_type,
                           cudaStream_t stream);

    void moe_gemm(const T* A,
                  const char* B,
                  const T* weight_scales,
                  T* C,
                  int64_t* total_rows_before_expert,
                  int64_t total_rows,
                  int64_t gemm_n,
                  int64_t gemm_k,
                  int num_experts,
                  cudaStream_t stream);

private:
    int sm_;
    int multi_processor_count_;
};

}  // namespace fastertransformer

template <typename T, WeightVariant V>
class MoeGemmContext {
public:
    MoeGemmContext() { _runner = new fastertransformer::MoeGemmRunner<T, V>(); }

    virtual ~MoeGemmContext() { delete _runner; }

    static MoeGemmContext& Instance()
    {
        static MoeGemmContext _ctx;
        return _ctx;
    }

    fastertransformer::MoeGemmRunner<T, V>* GeMM_Runner() const { return _runner; }

    fastertransformer::MoeGemmRunner<T, V>* _runner;
};
