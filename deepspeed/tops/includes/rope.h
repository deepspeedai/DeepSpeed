
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "rope.cuh"

void rope_fwd(torch::Tensor& query, torch::Tensor& key, int rotary_dim, float rope_theta);
void rope_bwd(torch::Tensor& query_grad, torch::Tensor& key_grad, int rotary_dim, float rope_theta);
