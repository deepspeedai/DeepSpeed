#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "swiglu.cuh"

void swiglu_fwd(torch::Tensor& inp, torch::Tensor& out);
void swiglu_bwd(torch::Tensor& inp, torch::Tensor& out_grad, torch::Tensor& inp_grad);
