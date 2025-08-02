#include "swiglu.h"
#include <c10/cuda/CUDAStream.h>

#define DISPATCH_SWIGLU(T_TYPE, C_TYPE)                \
    if (inp.options().dtype() == torch::T_TYPE) {          \
        launch_swiglu((C_TYPE*)out.data_ptr(),             \
                      (C_TYPE*)inp.data_ptr(),             \
                       bsz,                                \
                       hidden_size,                        \
                       at::cuda::getCurrentCUDAStream());  \
        return;                                        \
    }


void swiglu_fwd(torch::Tensor& inp, torch::Tensor& out)
{
    int inp_dims = inp.sizes().size();
    int hidden_size = inp.size(inp_dims - 1);
    int bsz = inp.size(0);
    for (int i = 1;i < inp_dims - 1;i++)
        bsz *= inp.size(i);
    // printf("bsz = %d, hidden_size = %d \n", bsz, hidden_size);

    DISPATCH_SWIGLU(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_SWIGLU(kBFloat16, __nv_bfloat16);
#endif

}

#define DISPATCH_SWIGLU_BWD(T_TYPE, C_TYPE)                \
    if (inp.options().dtype() == torch::T_TYPE) {          \
        launch_swiglu_bwd((C_TYPE*)inp_grad.data_ptr(),             \
                      (C_TYPE*)out_grad.data_ptr(),             \
                      (C_TYPE*)inp.data_ptr(),             \
                       bsz,                                \
                       hidden_size,                        \
                       at::cuda::getCurrentCUDAStream());  \
        return;                                        \
    }

void swiglu_bwd(torch::Tensor& inp, torch::Tensor& out_grad, torch::Tensor& inp_grad)
{
    int inp_dims = inp.sizes().size();
    int hidden_size = inp.size(inp_dims - 1);
    int bsz = inp.size(0);
    for (int i = 1;i < inp_dims-1;i++)
        bsz *= inp.size(i);
    
    DISPATCH_SWIGLU_BWD(kHalf, __half);
#ifdef BF16_AVAILABLE
    DISPATCH_SWIGLU_BWD(kBFloat16, __nv_bfloat16);
#endif

}
