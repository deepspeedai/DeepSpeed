// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// Fused Adam / AdamW step for one tensor. Mirrors csrc/adam/multi_tensor_adam.cu: all math is done in
// fp32 regardless of the storage dtype, so fp16/bf16 parameters see the same numerics as on CUDA.

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void fused_adam(device T* param [[buffer(0)]],
                       device const T* grad [[buffer(1)]],
                       device T* exp_avg [[buffer(2)]],
                       device T* exp_avg_sq [[buffer(3)]],
                       constant float& lr [[buffer(4)]],
                       constant float& beta1 [[buffer(5)]],
                       constant float& beta2 [[buffer(6)]],
                       constant float& epsilon [[buffer(7)]],
                       constant float& weight_decay [[buffer(8)]],
                       constant float& bias_correction1 [[buffer(9)]],
                       constant float& bias_correction2 [[buffer(10)]],
                       constant uint& adam_w_mode [[buffer(11)]],
                       uint i [[thread_position_in_grid]])
{
    float g = float(grad[i]);
    float p = float(param[i]);
    float m = float(exp_avg[i]);
    float v = float(exp_avg_sq[i]);

    // L2 mode folds weight decay into the gradient; AdamW mode applies it to the parameter.
    if (adam_w_mode == 0) { g += weight_decay * p; }
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    float denom = sqrt(v / bias_correction2) + epsilon;
    float update = (m / bias_correction1) / denom;
    if (adam_w_mode != 0) { update += weight_decay * p; }

    param[i] = T(p - lr * update);
    exp_avg[i] = T(m);
    exp_avg_sq[i] = T(v);
}

template [[host_name("fused_adam_float")]] kernel void fused_adam<float>(
    device float*, device const float*, device float*, device float*,
    constant float&, constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant uint&, uint);
template [[host_name("fused_adam_half")]] kernel void fused_adam<half>(
    device half*, device const half*, device half*, device half*,
    constant float&, constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant uint&, uint);
// bfloat is a Metal 3.1 type (macOS 14); older toolchains still get the float and half kernels.
#if __METAL_VERSION__ >= 310
template [[host_name("fused_adam_bfloat")]] kernel void fused_adam<bfloat>(
    device bfloat*, device const bfloat*, device bfloat*, device bfloat*,
    constant float&, constant float&, constant float&, constant float&, constant float&,
    constant float&, constant float&, constant uint&, uint);
#endif
