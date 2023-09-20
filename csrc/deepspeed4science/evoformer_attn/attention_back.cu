// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <type_traits>
#include "gemm_kernel_utils.h"
#include "kernel_backward.h"
#include "transform/bias_broadcast.h"

constexpr auto kBlockSizeI = 64;
constexpr auto kBlockSizeJ = 64;

template <typename arch,
          typename scalar_t,
          typename torch_scalar_t,
          template <typename, typename, typename>
          class Broadcast1_,
          template <typename, typename, typename>
          class Broadcast2_>
typename std::enable_if<!CheckArch<arch, scalar_t>::value>::type attention_back_impl_template(
    torch::Tensor& go,
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
    EVOFORMER_CHECK(false, "Unsupported GPU and data type combination")
}

template <typename arch,
          typename scalar_t,
          typename torch_scalar_t,
          template <typename, typename, typename>
          class Broadcast1_,
          template <typename, typename, typename>
          class Broadcast2_>
typename std::enable_if<CheckArch<arch, scalar_t>::value>::type attention_back_impl_template(
    torch::Tensor& go,
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
    constexpr bool kPreload_ = arch::kMinComputeCapability >= 80;
    using Kernel = AttentionBackwardKernel<arch,
                                           scalar_t,     // scalar_t
                                           true,         // kIsAligned_
                                           false,        // kApplyDropout_
                                           kPreload_,    // kPreload_
                                           kBlockSizeI,  // kBlockSizeI_,
                                           kBlockSizeJ,  // kBlockSizeJ_,
                                           64,           // kMaxK
                                           Broadcast1_,
                                           Broadcast2_>;
    int head_size = q.size(-1);
    int head_number = q.size(-2);
    int seq_length = q.size(-3);
    auto q_view = q.view({-1, seq_length, head_number, head_size});
    auto k_view = k.view({-1, seq_length, head_number, head_size});
    auto v_view = v.view({-1, seq_length, head_number, head_size});
    auto o_view = o.view({-1, seq_length, head_number, head_size});
    auto do_view = go.view({-1, seq_length, head_number, head_size});
    auto dk_view = gk.view({-1, seq_length, head_number, head_size});
    auto dv_view = gv.view({-1, seq_length, head_number, head_size});
    auto dq_view = gq.view({-1, seq_length, head_number, head_size});
    auto q_ptr = reinterpret_cast<scalar_t*>(q.data_ptr<torch_scalar_t>());
    auto k_ptr = reinterpret_cast<scalar_t*>(k.data_ptr<torch_scalar_t>());
    auto v_ptr = reinterpret_cast<scalar_t*>(v.data_ptr<torch_scalar_t>());
    auto o_ptr = reinterpret_cast<scalar_t*>(o.data_ptr<torch_scalar_t>());
    auto do_ptr = reinterpret_cast<scalar_t*>(go.data_ptr<torch_scalar_t>());
    auto dk_ptr = reinterpret_cast<scalar_t*>(gk.data_ptr<torch_scalar_t>());
    auto dv_ptr = reinterpret_cast<scalar_t*>(gv.data_ptr<torch_scalar_t>());
    auto dq_ptr = reinterpret_cast<scalar_t*>(gq.data_ptr<torch_scalar_t>());
    auto db1_ptr = gb1.size(0) > 0 ? reinterpret_cast<float*>(gb1.data_ptr<float>()) : nullptr;
    auto db2_ptr = gb2.size(0) > 0 ? reinterpret_cast<float*>(gb2.data_ptr<float>()) : nullptr;
    auto lse_ptr = reinterpret_cast<float*>(lse.data_ptr<float>());
    auto delta_ptr = reinterpret_cast<float*>(delta.data_ptr<float>());
    auto bias1_ptr = reinterpret_cast<scalar_t*>(bias1.data_ptr<torch_scalar_t>());
    auto bias2_ptr = reinterpret_cast<scalar_t*>(bias2.data_ptr<torch_scalar_t>());
    static_assert(Kernel::kKernelComputesDelta, "Kernel must compute delta");

    typename Kernel::Params p;
    p.query_ptr = q_ptr;
    p.key_ptr = k_ptr;
    p.value_ptr = v_ptr;
    p.logsumexp_ptr = lse_ptr;
    p.output_ptr = o_ptr;
    p.grad_output_ptr = do_ptr;
    p.delta_ptr = delta_ptr;
    p.grad_query_ptr = dq_ptr;
    p.grad_key_ptr = dk_ptr;
    p.grad_value_ptr = dv_ptr;

    p.grad_bias1_ptr = db1_ptr;
    p.grad_bias2_ptr = db2_ptr;
    p.B = q.size(0);
    p.N = q.size(1);
    p.bias1_ptr = bias1.size(0) ? bias1_ptr : nullptr;
    p.bias2_ptr = bias2.size(0) ? bias2_ptr : nullptr;

    p.scale = 1.0f / sqrtf(head_size);

    p.head_dim = head_size;
    p.head_dim_value = head_size;
    p.num_queries = seq_length;
    p.num_keys = seq_length;
    p.num_heads = head_number;

    p.q_strideM = q_view.stride(-3);
    p.k_strideM = k_view.stride(-3);
    p.v_strideM = v_view.stride(-3);
    p.gO_strideM = do_view.stride(-3);
    p.o_strideH = o_view.stride(-2);
    p.q_strideH = q_view.stride(-2);
    p.k_strideH = k_view.stride(-2);
    p.v_strideH = v_view.stride(-2);
    p.o_strideB = o_view.stride(-4);
    p.q_strideB = q_view.stride(-4);
    p.k_strideB = k_view.stride(-4);
    p.v_strideB = v_view.stride(-4);
    p.lse_strideB = lse.stride(-3);
    p.lse_strideH = lse.stride(-2);
    p.delta_strideB = delta.stride(-3);
    p.delta_strideH = delta.stride(-2);
    p.num_batches = q_view.size(-4);

    p.gO_strideB = do_view.stride(-4);
    p.gQ_strideB = dq_view.stride(-4);
    p.gK_strideB = dk_view.stride(-4);
    p.gV_strideB = dv_view.stride(-4);
    p.gO_strideH = do_view.stride(-2);
    p.gQ_strideH = dq_view.stride(-2);
    p.gK_strideH = dk_view.stride(-2);
    p.gV_strideH = dv_view.stride(-2);

    torch::Tensor workspace = torch::empty(p.workspace_size() / 4, lse.options());
    p.workspace = workspace.data_ptr<float>();

    auto kernel_fn = attention_kernel_backward_batched_impl<Kernel>;
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));
    if (!Kernel::check_supported(p)) { throw std::runtime_error("Unsupported parameters"); }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
}

#define CODE(scalar_t, torch_scalar_t)                                           \
    do {                                                                         \
        if (bias1.size(0) == 0 && bias2.size(0) == 0) {                          \
            attention_back_impl_template<ArchTag,                                \
                                         scalar_t,                               \
                                         torch_scalar_t,                         \
                                         BroadcastNoLoad,                        \
                                         BroadcastNoLoad>(                       \
                go, q, k, v, o, lse, delta, bias1, bias2, gq, gk, gv, gb1, gb2); \
        } else if (bias1.size(0) > 0 && bias2.size(0) > 0) {                     \
            attention_back_impl_template<ArchTag,                                \
                                         scalar_t,                               \
                                         torch_scalar_t,                         \
                                         BroadcastA,                             \
                                         BroadcastB>(                            \
                go, q, k, v, o, lse, delta, bias1, bias2, gq, gk, gv, gb1, gb2); \
        } else if (bias1.size(0) > 0) {                                          \
            attention_back_impl_template<ArchTag,                                \
                                         scalar_t,                               \
                                         torch_scalar_t,                         \
                                         BroadcastA,                             \
                                         BroadcastNoLoad>(                       \
                go, q, k, v, o, lse, delta, bias1, bias2, gq, gk, gv, gb1, gb2); \
        } else {                                                                 \
            attention_back_impl_template<ArchTag,                                \
                                         scalar_t,                               \
                                         torch_scalar_t,                         \
                                         BroadcastNoLoad,                        \
                                         BroadcastB>(                            \
                go, q, k, v, o, lse, delta, bias1, bias2, gq, gk, gv, gb1, gb2); \
        }                                                                        \
    } while (0)

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
                         torch::Tensor& gb2)
{
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    DISPATCH_ARCHTAG(prop->major * 10 + prop->minor,
                     DISPATCH_TYPES(q, { CODE(scalar_t, torch_scalar_t); }));
}
