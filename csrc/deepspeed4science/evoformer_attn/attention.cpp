// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

void attention_impl(torch::Tensor& q,
                    torch::Tensor& k,
                    torch::Tensor& v,
                    torch::Tensor& bias1,
                    torch::Tensor& bias2,
                    torch::Tensor& o,
                    torch::Tensor& lse);
void attention(torch::Tensor& q,
               torch::Tensor& k,
               torch::Tensor& v,
               torch::Tensor& bias1,
               torch::Tensor& bias2,
               torch::Tensor& o,
               torch::Tensor& lse)
{
    attention_impl(q, k, v, bias1, bias2, o, lse);
}

void attention_back_impl(torch::Tensor& go,
                         torch::Tensor& q,
                         torch::Tensor& k,
                         torch::Tensor& v,
                         torch::Tensor& o,
                         torch::Tensor& lse,
                         torch::Tensor& delta,
                         torch::Tensor& bias1,
                         torch::Tensor& bias2,
                         torch::Tensor& gq,
                         torch::Tensor& gk,
                         torch::Tensor& gv,
                         torch::Tensor& gb1,
                         torch::Tensor& gb2);
void attention_bwd(torch::Tensor& go,
                   torch::Tensor& q,
                   torch::Tensor& k,
                   torch::Tensor& v,
                   torch::Tensor& o,
                   torch::Tensor& lse,
                   torch::Tensor& delta,
                   torch::Tensor& bias1,
                   torch::Tensor& bias2,
                   torch::Tensor& gq,
                   torch::Tensor& gk,
                   torch::Tensor& gv,
                   torch::Tensor& gb1,
                   torch::Tensor& gb2)
{
    attention_back_impl(go, q, k, v, o, lse, delta, bias1, bias2, gq, gk, gv, gb1, gb2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention, "");
    m.def("attention_bwd", &attention_bwd, "");
}
