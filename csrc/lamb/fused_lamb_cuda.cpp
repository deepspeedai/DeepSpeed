// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

// CUDA forward declaration
void fused_lamb_cuda(at::Tensor& p,
                     at::Tensor& p_copy,
                     at::Tensor& m,
                     at::Tensor& v,
                     at::Tensor& g,
                     float lr,
                     float beta1,
                     float beta2,
                     float max_coeff,
                     float min_coeff,
                     float eps,
                     float grad_scale,
                     int step,
                     int mode,
                     int bias_correction,
                     float decay,
                     at::Tensor& w_l2_i,
                     at::Tensor& u_l2_i,
                     at::Tensor& lamb_coeff_val);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// C++ interface
at::Tensor lamb(at::Tensor& p,
                at::Tensor& p_copy,
                at::Tensor& m,
                at::Tensor& v,
                at::Tensor& g,
                float lr,
                float beta1,
                float beta2,
                float max_coeff,
                float min_coeff,
                float eps,
                float grad_scale,
                int step,
                int mode,
                int bias_correction,
                float decay)
{
    CHECK_INPUT(p);
    if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
    CHECK_INPUT(m);
    CHECK_INPUT(v);
    CHECK_INPUT(g);
    int64_t num_elem = p.numel();
    AT_ASSERTM(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
    AT_ASSERTM(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
    AT_ASSERTM(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
    AT_ASSERTM(
        p_copy.numel() == num_elem || p_copy.numel() == 0,
        "number of elements in p_copy and p tensors should be equal, or p_copy should be empty");

    // intermediate for weight L2 reduction
    // make sure that the threads per block is at least 512 during the kernel launch otherwise the
    // behaviour is unexpected
    at::Tensor w_l2_i = at::empty(
        {512},
        p.options().dtype(p.type().scalarType() == at::ScalarType::Half ? at::ScalarType::Float
                                                                        : p.type().scalarType()));

    // intermediate for update L2 reduction
    // make sure that the threads per block is at least 512 during the kernel launch otherwise the
    // behaviour is unexpected
    at::Tensor u_l2_i = at::empty(
        {512},
        p.options().dtype(p.type().scalarType() == at::ScalarType::Half ? at::ScalarType::Float
                                                                        : p.type().scalarType()));

    at::Tensor lamb_coeff_val = at::empty(
        {1},
        p.options().dtype(p.type().scalarType() == at::ScalarType::Half ? at::ScalarType::Float
                                                                        : p.type().scalarType()));

    fused_lamb_cuda(p,
                    p_copy,
                    m,
                    v,
                    g,
                    lr,
                    beta1,
                    beta2,
                    max_coeff,
                    min_coeff,
                    eps,
                    grad_scale,
                    step,
                    mode,
                    bias_correction,
                    decay,
                    w_l2_i,
                    u_l2_i,
                    lamb_coeff_val);

    return lamb_coeff_val;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lamb", &lamb, "Adam optimized CUDA implementation with LAMB.");
}
