#include "swiglu.cuh"
#include "utils.h"

#include <stdexcept>

DS_D_INLINE float gated_act_fn(float x, float y)
{
    return y * (x / (1.0f + expf(-x)));
}

template <typename T, int loopUnroll>
__global__ void swiglu_kernel(T* out, T* inp, int hidden_size)
{
    
    constexpr int read_vector = 16 / sizeof(T);
    constexpr int write_vector = read_vector; // / 2;

    const int row = blockIdx.x;
    const int col = threadIdx.x * read_vector;



    T* input_row = inp + row * hidden_size;
    T* output_row = out + row * (hidden_size >> 1);

#pragma unroll
    for (int i = 0; i < loopUnroll; i++) {
        T read1[read_vector];
        T read2[read_vector];
        T store[write_vector];

        const int read_offset = col + ((read_vector * i) << 10);
        const int write_offset = col + ((write_vector * i) << 10);

        if (i != loopUnroll - 1 || read_offset < (hidden_size >> 1)) {
            mem_access::load_global<16>(read1, input_row + read_offset);
            mem_access::load_global<16>(read2, input_row + read_offset + (hidden_size >> 1));

            for (int j = 0; j < write_vector; j++) {
                float g_val = conversion::to<float>(read1[j]);
                float a_val = conversion::to<float>(read2[j]) ;

                float act_val = gated_act_fn(g_val, a_val);
                store[j] = conversion::to<T>(act_val);
                // if (threadIdx.x == 0 && blockIdx.x == 0) printf("I am here! %f %p %p %d\n", act_val, out, output_row, write_offset);
            }

            mem_access::store_global<16>(output_row + write_offset, store);
        }
    }
}


#define DISPATCH_UNROLL(unroll_val)                 \
    swiglu_kernel<T, unroll_val> \
        <<<grid, block, 0, stream>>>(out, inp, hidden_size);


template <typename T>
void launch_swiglu(T* out, 
                       T* inp,
                       int bsz, 
                       int hidden_size,
                       cudaStream_t stream)
{
    const int threads = 1024;
    const dim3 grid(bsz);
    const dim3 block(threads);
    constexpr int cols_per_unroll = threads * 16 / sizeof(T);
    const int unroll = ((hidden_size >> 1) - 1) / cols_per_unroll + 1;
    // printf("bsz = %d, cols_per_unroll = %d, unroll = %d, hidden_size = %d \n", bsz, cols_per_unroll, unroll, hidden_size);
    
    if (unroll == 1) {
        DISPATCH_UNROLL(1);
    } else if (unroll == 2) {
        DISPATCH_UNROLL(2);
    } else if (unroll == 3) {
        DISPATCH_UNROLL(3);
    } else if (unroll == 4) {
        DISPATCH_UNROLL(4);
    } else if (unroll == 5) {
        DISPATCH_UNROLL(5);
    } else if (unroll == 6) {
        DISPATCH_UNROLL(6);
    } else {
        throw std::runtime_error(
            "[RuntimeError]: SwiGlu kernel limit surpassed");
    }
}

#define INSTANTIATE_FOR_TYPE(T)                                       \
    template void launch_swiglu<T>(T * out,              \
                                       T* inp,          \
                                       int bsz,                \
                                       int hidden_size,                \
                                       cudaStream_t stream);

INSTANTIATE_FOR_TYPE(float)
INSTANTIATE_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_FOR_TYPE(__nv_bfloat16)
#endif



DS_D_INLINE float gated_act_bwd_fn(float &x, float &y, float &grad)
{
    float sigmoid = 1.0 / (1.0 + expf(-x));
    return  y * grad * sigmoid * (1.0 + x * (1.0 - sigmoid));
}

DS_D_INLINE float sig_fn(float x, float grad)
{
    return grad * (x / (1.0f + expf(-x)));
}


template <typename T, int loopUnroll>
__global__ void swiglu_bwd_kernel(T* inp_grad, T* out_grad, T* inp, int hidden_size)
{
    
    constexpr int read_vector = 16 / sizeof(T);
    constexpr int write_vector = read_vector; /// 2;

    const int row = blockIdx.x;
    const int col = threadIdx.x * read_vector;

    T* input_row = inp + row * hidden_size;
    T* inp_grad_row = inp_grad + row * hidden_size;
    T* out_grad_row = out_grad + row * (hidden_size >> 1);

#pragma unroll
    for (int i = 0; i < loopUnroll; i++) {
        T read1[read_vector];
        T read2[read_vector];
        T read_grad[write_vector];
        T store1[read_vector];
        T store2[read_vector];

        const int read_offset = col + ((read_vector * i) << 10);
        const int write_offset = col + ((write_vector * i) << 10);

        if (i != loopUnroll - 1 || read_offset < (hidden_size >> 1)) {
            mem_access::load_global<16>(read1, input_row + read_offset);
            mem_access::load_global<16>(read2, input_row + read_offset + (hidden_size >> 1));
            mem_access::load_global<16>(read_grad, out_grad_row + write_offset);

            for (int j = 0; j < write_vector; j++) {
                float g_val = conversion::to<float>(read1[j]);
                float a_val = conversion::to<float>(read2[j]) ;
                float grad_val = conversion::to<float>(read_grad[j]) ;

                float grad_y = sig_fn(g_val, grad_val);
                float grad_x = gated_act_bwd_fn(g_val, a_val, grad_val);

                store1[j] = conversion::to<T>(grad_x);
                store2[j] = conversion::to<T>(grad_y);
            }

            mem_access::store_global<16>(inp_grad_row + read_offset, store1);
            mem_access::store_global<16>(inp_grad_row + read_offset + (hidden_size >> 1), store2);
        }
    }
}


#define BWD_DISPATCH_UNROLL(unroll_val)                 \
    swiglu_bwd_kernel<T, unroll_val> \
        <<<grid, block, 0, stream>>>(inp_grad, out_grad, inp, hidden_size);


template <typename T>
void launch_swiglu_bwd(T* inp_grad, T* out_grad, T* inp,
                        int bsz, int hidden_size,
                        cudaStream_t stream)
{
    const int threads = 1024;
    const dim3 grid(bsz);
    const dim3 block(threads);
    constexpr int cols_per_unroll = threads * 16 / sizeof(T);
    const int unroll = ((hidden_size >> 1) - 1) / cols_per_unroll + 1;
    if (unroll == 1) {
        BWD_DISPATCH_UNROLL(1);
    } else if (unroll == 2) {
        BWD_DISPATCH_UNROLL(2);
    } else if (unroll == 3) {
        BWD_DISPATCH_UNROLL(3);
    } else if (unroll == 4) {
        BWD_DISPATCH_UNROLL(4);
    } else if (unroll == 5) {
        BWD_DISPATCH_UNROLL(5);
    } else if (unroll == 6) {
        BWD_DISPATCH_UNROLL(6);
    } else {
        throw std::runtime_error(
            "[RuntimeError]: SwiGlu BWD kernel limit surpassed");
    }
}

#define INSTANTIATE_BWD_FOR_TYPE(T)                                       \
    template void launch_swiglu_bwd<T>(T * inp_grad,              \
                                       T * out_grad,              \
                                       T* inp,          \
                                       int bsz,                \
                                       int hidden_size,                \
                                       cudaStream_t stream);

INSTANTIATE_BWD_FOR_TYPE(float)
INSTANTIATE_BWD_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_BWD_FOR_TYPE(__nv_bfloat16)
#endif