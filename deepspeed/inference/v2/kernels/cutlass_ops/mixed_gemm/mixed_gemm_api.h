// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "activation_type.h"
#include "weight_variant.h"

namespace fastertransformer {

template <typename T, WeightVariant V>
class CutlassFpAIntBGemmRunner {
public:
    void gemm(const T* A,
              const char* B,
              const T* weight_scales,
              T* C,
              int m,
              int n,
              int k,
              char* workspace_ptr,
              const size_t workspace_bytes,
              cudaStream_t stream);

    void gemm_bias_act(const T* A,
                       const char* B,
                       const T* weight_scales,
                       const T* biases,
                       T* C,
                       int m,
                       int n,
                       int k,
                       ActivationType activation_type,
                       char* workspace_ptr,
                       const size_t workspace_bytes,
                       cudaStream_t stream);
};

}  // namespace fastertransformer

template <typename T, WeightVariant V>
class MixedGemmContext {
public:
    MixedGemmContext() { _runner = new fastertransformer::CutlassFpAIntBGemmRunner<T, V>(); }

    virtual ~MixedGemmContext() { delete _runner; }

    static MixedGemmContext& Instance()
    {
        static MixedGemmContext _ctx;
        return _ctx;
    }

    fastertransformer::CutlassFpAIntBGemmRunner<T, V>* GeMM_Runner() const { return _runner; }

    fastertransformer::CutlassFpAIntBGemmRunner<T, V>* _runner;
};
