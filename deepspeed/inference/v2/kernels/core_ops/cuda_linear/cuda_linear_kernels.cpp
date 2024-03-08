// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>

#include "cuda_linear_kernels.h"

namespace {

// For bit-level debugging.
template <typename T>
void print_bits(T num)
{
    char bits[sizeof(T) * 8 + 1] = {'\0'};
    for (int bit = 0; bit < (sizeof(T) * 8); bit++) {
        bits[sizeof(T) * 8 - 1 - bit] = '0' + (num & 0x01);
        num = num >> 1;
    }
    printf("%s\n", bits);
}

void print_bits(half num)
{
    char bits[sizeof(half) * 8 + 1] = {'\0'};
    auto int_num = *reinterpret_cast<uint16_t*>(&num);
    for (int bit = 0; bit < (sizeof(half) * 8); bit++) {
        bits[sizeof(half) * 8 - 1 - bit] = '0' + (int_num & 0x01);
        int_num = int_num >> 1;
    }
    printf("%s\n", bits);
}

/*
 * Function to pack 4 fake quantized FP16 value into continuously stored 4 FP6 values.
 */
void cast_fp16_fp6(uint16_t* FP16x4, uint8_t* FP6x4)
{
    // Constants for FP6
    constexpr int exponent_nbits_fp6 = 3;
    constexpr int mantissa_nbits_fp6 = 2;
    constexpr int exp_bias_fp6 = (1 << (exponent_nbits_fp6 - 1)) - 1;
    // Constants for FP16
    constexpr int exponent_nbits_fp16 = 5;
    constexpr int mantissa_nbits_fp16 = 10;
    constexpr int exp_bias_fp16 = (1 << (exponent_nbits_fp16 - 1)) - 1;

    int fp6_temp[4];

    float absmin_nonzero_fp6 = 0.0625;
    // Note that we regard the exponent of '111' as a regular value rather than NaN or inf. This is
    // the same with that in qtorch.
    float absmax_fp6 = 28;

    for (int i = 0; i < 4; ++i) {
        uint16_t source = FP16x4[i];
        float fp6_value_abs = std::abs(__half2float(*((half*)(&source))));
        if ((fp6_value_abs != 0 && fp6_value_abs < absmin_nonzero_fp6) ||
            fp6_value_abs > absmax_fp6) {
            // TODO(zhen): a better way may be rounding it to the nearest FP6 value.
            throw std::invalid_argument("Input value out of range for FP6.");
        }

        // It is not safe to do shift operation on uint16_t. So we promote it to int.
        int source_promote = int(source);

        int sign_bit = (source_promote >> 15);
        // Extracting exponent represented in FP16. The sign mask 0x7FFF is '0111 1111 1111 1111'
        int exp_bit = (source_promote & 0x7FFF) >> mantissa_nbits_fp16;
        // Extracting mantissa represented in FP16
        int mant_bit = source_promote & ((1 << mantissa_nbits_fp16) - 1);

        int new_exp_bit;
        int new_mant_bit;

        if (exp_bit == 0) {
            // Subnormal FP16 number. Too small for FP6.
            new_exp_bit = 0;
            new_mant_bit = 0;
        } else {
            new_mant_bit = mant_bit >> (mantissa_nbits_fp16 - mantissa_nbits_fp6);
            new_exp_bit = exp_bit - exp_bias_fp16 + exp_bias_fp6;

            // Deal with subnormal FP6 values.
            int target_exp_val = exp_bit - exp_bias_fp16;
            int min_fp6_exp_val = -exp_bias_fp6 + 1;
            bool subnormal_fp6 = target_exp_val < min_fp6_exp_val;
            if (subnormal_fp6) {
                // TODO(zhen): add the rounding logic.
                new_exp_bit = 0;
                // The implicit 1 in the mantissa of FP16 is not present in subnormal FP6. Thus we
                // need to add it
                new_mant_bit = (new_mant_bit | (1 << mantissa_nbits_fp6)) >>
                               (min_fp6_exp_val - target_exp_val);
            }
        }

        fp6_temp[i] = (sign_bit << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                      (new_exp_bit << mantissa_nbits_fp6) | new_mant_bit;
    }
    // Pack the values
    FP6x4[0] = fp6_temp[0] << 2 | (fp6_temp[1] >> 4);
    FP6x4[1] = (fp6_temp[1] & 0x0F) << 4 | (fp6_temp[2] >> 2);
    FP6x4[2] = (fp6_temp[2] & 0x03) << 6 | fp6_temp[3];
}

/*
 *  Function to prepack FP16 weights into continuous FP6 values.
 *
 *  Parameters:
 *     weight_16bit: input weight in FP16, size M*K
 *     weight_6bit: output weight in packed FP6, continuously stored, size M*K*6/8
 *     M, K: the shape of the weight
 */
void weight_prepacking_fp16_to_fp6(uint16_t* weight_16bit,
                                   uint8_t* weight_6bit_packed,
                                   size_t M,
                                   size_t K)
{
    // Every four 16-bit elements are packed into three 6-bit values (4*6bit == 3*8bit).
    if (K * 6 % 8 != 0) { throw std::invalid_argument("(K * 6 % 8) should be 0"); }
    size_t K_fp6_packed = K * 6 / 8;
    // #pragma omp parallel for
    for (auto m = 0; m < M; m++) {
        uint8_t* ptr_6bit = weight_6bit_packed + m * K_fp6_packed;
        uint16_t* ptr_16bit = weight_16bit + m * K;
        for (auto k = 0; k < K; k += 4) {
            cast_fp16_fp6(ptr_16bit, ptr_6bit);
            ptr_16bit += 4;
            ptr_6bit += 3;
        }
    }
}

}  // namespace

/*
 * Function to execute the FP6 linear kernel.
 *
 * Parameters:
 *    output: output tensor, size M*N
 *    hidden_states: input activation tensor, size N*K
 *    weights_2bit: packed 2bit weights, size M*K*2/8
 *    weights_4bit: packed 4bit weights, size M*K*4/8
 *    scales: scale tensor, size M
 *    workspace: workspace tensor, size M*N*split_k
 *    M: the output channel number of the weight
 *    N: the token number of the activation
 *    K: the input channel number of the weight
 *    split_k: the split size of the GEMM calculation
 */
void cuda_wf6af16_linear(torch::Tensor& output,
                         torch::Tensor& hidden_states,
                         torch::Tensor& weights_2bit,
                         torch::Tensor& weights_4bit,
                         torch::Tensor& scales,
                         torch::Tensor& workspace,
                         int M,
                         int N,
                         int K,
                         int split_k)
{
    TORCH_CHECK(weights_2bit.device().type() == torch::kCUDA, "weight_2bit must be on CUDA");
    TORCH_CHECK(weights_4bit.device().type() == torch::kCUDA, "weight_4bit must be on CUDA");
    TORCH_CHECK(hidden_states.device().type() == torch::kCUDA, "X must be on CUDA");
    TORCH_CHECK(scales.device().type() == torch::kCUDA, "scales must be on CUDA");

    auto status = fp6_linear_kernel(at::cuda::getCurrentCUDAStream(),
                                    (uint4*)(weights_2bit.data_ptr<uint8_t>()),
                                    (uint4*)(weights_4bit.data_ptr<uint8_t>()),
                                    (half*)(scales.data_ptr<at::Half>()),
                                    (half*)(hidden_states.data_ptr<at::Half>()),
                                    (half*)(output.data_ptr<at::Half>()),
                                    M,
                                    N,
                                    K,
                                    workspace.data_ptr<float>(),
                                    split_k);
    if (status != cudaSuccess) {
        AT_ERROR("fp6_linear_kernel failed with error: ", cudaGetErrorString(status));
    }
}

/*
 * Function to prepack the fake 6-bit-quantized FP16 weights into 2bit and 4bit.
 *
 * Parameters:
 *    weight: input weight in FP16 (containing the quantized FP6-ranged value), size M*K
 * Returns:
 *   weight_2bit: output weight in 2bit, size M*K*2/8
 *   weight_4bit: output weight in 4bit, size M*K*4/8
 */
std::vector<torch::Tensor> preprocess_weight(torch::Tensor& weight)
{
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-dimensional");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be FP16");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.device().type() == torch::kCPU, "weight must be on CPU");
    auto M = weight.size(0);
    auto K = weight.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4");

    // Pack weight from FP16 to FP6.
    uint16_t* weight_16bit_ptr = reinterpret_cast<uint16_t*>(weight.data_ptr<at::Half>());
    std::vector<uint8_t> weight_6bit_packed(M * K * 6 / 8);
    uint8_t* weight_6bit_ptr = weight_6bit_packed.data();
    weight_prepacking_fp16_to_fp6(weight_16bit_ptr, weight_6bit_ptr, M, K);

    // Split weight into 2bit and 4bit.
    weight_matrix_prepacking(reinterpret_cast<int*>(weight_6bit_ptr), M, K);
    uint8_t* weight_2bit_ptr = weight_6bit_ptr;

    // Make sure that the new split tensor does not share the underlying memory with the original
    // one. Otherwise it will incur some problems when the original tensor is deleted. It also
    // makes the memory flattern risky.
    auto weight_2bit =
        torch::from_blob(weight_2bit_ptr, {M * K * 2 / 8}, torch::kUInt8).clone().detach();
    uint8_t* weight_4bit_ptr = weight_2bit_ptr + M * K * 2 / 8;
    auto weight_4bit =
        torch::from_blob(weight_4bit_ptr, {M * K * 4 / 8}, torch::kUInt8).clone().detach();

    return {weight_2bit, weight_4bit};
}
