#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "moe_gating.cuh"

void gate_scatter(torch::Tensor& moe_input,
                torch::Tensor& expert_count_cumsums,
                torch::Tensor& mapped_slots,
                torch::Tensor& activations,
                torch::Tensor& expert_counts,
                torch::Tensor& mapped_expert_counts,
                torch::Tensor& scores,
                torch::Tensor& assignments,
                torch::Tensor& offsets,
                torch::Tensor& backup_offsets,
                int top_k,
                int capacity,
                bool use_rts);

void gate_fwd(torch::Tensor& moe_input,
                torch::Tensor& expert_count_cumsums,
                torch::Tensor& mapped_slots,
                torch::Tensor& activations,
                torch::Tensor& expert_counts,
                torch::Tensor& mapped_expert_counts,
                torch::Tensor& scores,
                torch::Tensor& assignments,
                torch::Tensor& offsets,
                torch::Tensor& backup_offsets,
                torch::Tensor& logits,
                torch::Tensor& logits_out,
                int top_k,
                int capacity,
                bool use_rts);

void gate_bwd(torch::Tensor& moe_input_grad,
                torch::Tensor& scores_grad,
                torch::Tensor& activations_grad,
                torch::Tensor& logits_grad,
                torch::Tensor& logits,
                torch::Tensor& assignments,
                torch::Tensor& offsets,
                torch::Tensor& mapped_slots,
                int top_k,
                int capacity,
                bool use_rts);


void gather_fwd(torch::Tensor& layer_output,
                torch::Tensor& moe_output,
                torch::Tensor& scores,
                torch::Tensor& mapped_slots,
                int top_k);

void gather_bwd(torch::Tensor& layer_output_grad,
                torch::Tensor& scores_grad,
                torch::Tensor& moe_output_grad,
                torch::Tensor& moe_output,
                torch::Tensor& scores,
                torch::Tensor& mapped_slots,
                int top_k);
