#include "custom_cuda_layers.h"

__global__ void dropout_kernel(const int N,
                               const float ratio,
                               float* out,
                               const float* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    CUDA_1D_KERNEL_LOOP(j, N / 4)
    {
        float4 rand = curand_uniform4(&state);
        uint8_t m[4];

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        int i = j * 4;

        mask[i] = (uint8_t)m[0];
        mask[i + 1] = (uint8_t)m[1];
        mask[i + 2] = (uint8_t)m[2];
        mask[i + 3] = (uint8_t)m[3];

        out[i] = Xdata[i] * scale * m[0];
        out[i + 1] = Xdata[i + 1] * scale * m[1];
        out[i + 2] = Xdata[i + 2] * scale * m[2];
        out[i + 3] = Xdata[i + 3] * scale * m[3];
    }
}

__global__ void dropout_kernel(const int N,
                               const float ratio,
                               __half* out,
                               const __half* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

#ifdef __STOCHASTIC_MODE__

    const __half2 h_scale = __float2half2_rn(scale);
    const float2* x_cast = reinterpret_cast<const float2*>(Xdata);
    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    uint32_t m_32;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m_32);

    float2 result_f;
    __half2* result_h = reinterpret_cast<__half2*>(&result_f);
    __half2 mask_h[2];
    float2 mask_f[2];

    CUDA_1D_KERNEL_LOOP(j, N / 4)
    {
        float2 x_f = x_cast[j];
        __half2* x_h = reinterpret_cast<__half2*>(&x_f);

        float4 rand = curand_uniform4(&state);

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < 4; i++) mask_f_data[i] = (float)(m[i]);

        mask_h[0] = __float22half2_rn(mask_f[0]);
        mask_h[1] = __float22half2_rn(mask_f[1]);

        result_h[0] = x_h[0] * h_scale * mask_h[0];
        result_h[1] = x_h[1] * h_scale * mask_h[1];

        out_cast[j] = result_f;

        mask_cast[j] = m_32;
    }

#else

    CUDA_1D_KERNEL_LOOP(j, N / 4)
    {
        int i = j * 4;

        const __half2* vals_half = reinterpret_cast<const __half2*>(Xdata + i);
        float2 vals_half_f[2];
        vals_half_f[0] = __half22float2(vals_half[0]);
        vals_half_f[1] = __half22float2(vals_half[1]);

        uint8_t m[4];
        float4 rand = curand_uniform4(&state);
        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        out[i] = __float2half(vals_half_f[0].x * scale * m[0]);
        out[i + 1] = __float2half(vals_half_f[0].y * scale * m[1]);
        out[i + 2] = __float2half(vals_half_f[1].x * scale * m[2]);
        out[i + 3] = __float2half(vals_half_f[1].y * scale * m[3]);

        mask[i] = m[0];
        mask[i + 1] = m[1];
        mask[i + 2] = m[2];
        mask[i + 3] = m[3];
    }

#endif
}

__global__ void dropout_kernel_bwd(const int N,
                                   const float ratio,
                                   const float* Xdata,
                                   float* out,
                                   uint8_t* mask,
                                   std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    CUDA_1D_KERNEL_LOOP(j, N / 4)
    {
        int i = j * 4;

        out[i] = mask[i] ? Xdata[i] * scale : 0.0;
        out[i + 1] = mask[i + 1] ? Xdata[i + 1] * scale : 0.0;
        out[i + 2] = mask[i + 2] ? Xdata[i + 2] * scale : 0.0;
        out[i + 3] = mask[i + 3] ? Xdata[i + 3] * scale : 0.0;
    }
}

__global__ void dropout_kernel_bwd(const int N,
                                   const float ratio,
                                   const __half* Xdata,
                                   __half* out,
                                   uint8_t* mask,
                                   std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);

#ifdef __STOCHASTIC_MODE__

    const __half2 h_scale = __float2half2_rn(scale);

    const float2* x_cast = reinterpret_cast<const float2*>(Xdata);
    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    CUDA_1D_KERNEL_LOOP(j, N / 4)
    {
        float2 x_f = x_cast[j];
        __half2* x_h = reinterpret_cast<__half2*>(&x_f);

        uint8_t* m = reinterpret_cast<uint8_t*>(mask_cast + j);
        __half2 mask_h[2];
        float2 mask_f[2];

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < 4; i++) mask_f_data[i] = (float)(m[i]);

#pragma unroll
        for (int i = 0; i < 2; i++) mask_h[i] = __float22half2_rn(mask_f[i]);

        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);

        result_h[0] = x_h[0] * h_scale * mask_h[0];
        result_h[1] = x_h[1] * h_scale * mask_h[1];

        out_cast[j] = result_f;
    }

#else

    const __half h_scale = __float2half(scale);
    const __half h_zero = __float2half(0.0);

    CUDA_1D_KERNEL_LOOP(j, N / 4)
    {
        int i = j * 4;

        const __half2* vals_half = reinterpret_cast<const __half2*>(Xdata + i);

        uint8_t* m = mask + i;

        float2 vals_half_f[2];

        vals_half_f[0] = __half22float2(vals_half[0]);
        vals_half_f[1] = __half22float2(vals_half[1]);

        out[i] = __float2half(vals_half_f[0].x * scale * m[0]);
        out[i + 1] = __float2half(vals_half_f[0].y * scale * m[1]);
        out[i + 2] = __float2half(vals_half_f[1].x * scale * m[2]);
        out[i + 3] = __float2half(vals_half_f[1].y * scale * m[3]);
    }

#endif
}

template <typename T>
void launch_dropout(T* out,
                    const T* vals,
                    uint8_t* mask,
                    int total_count,
                    int dim,
                    float ratio,
                    cudaStream_t stream,
                    bool bwd)
{
    dim3 grid_dim = DS_GET_BLOCKS(total_count / 4);
    dim3 block_dim = DS_CUDA_NUM_THREADS;

    if (dim > 512) {
        block_dim.x >>= 1;
        grid_dim.x <<= 1;
    }
    uint64_t inc = total_count / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc);
    if (bwd)
        dropout_kernel_bwd<<<grid_dim, block_dim, 0, stream>>>(
            total_count, ratio, vals, out, mask, seed);
    else
        dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(
            total_count, ratio, out, vals, mask, seed);
}

template void launch_dropout(float* out,
                             const float* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             cudaStream_t stream,
                             bool);
template void launch_dropout(__half* out,
                             const __half* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             cudaStream_t stream,
                             bool);

__global__ void dropout_grad_kernel(const int N, const float scale, float* Xdata, uint8_t* mask)
{
    CUDA_1D_KERNEL_LOOP(i, N) { Xdata[i] *= scale * mask[i]; }
}

__global__ void dropout_grad_kernel(const int N, const float scale, __half* Xdata, uint8_t* mask)
{
#ifdef __STOCHASTIC_MODE__

    const __half2 h_scale = __float2half2_rn(scale);
    float2* x_cast = reinterpret_cast<float2*>(Xdata);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    CUDA_1D_KERNEL_LOOP(j, N / 4)
    {
        uint8_t* m = reinterpret_cast<uint8_t*>(mask_cast + j);
        __half2 mask_h[2];
        float2 mask_f[2];

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < 4; i++) *(mask_f_data++) = (float)(m[i]);

        mask_h[0] = __float22half2_rn(mask_f[0]);
        mask_h[1] = __float22half2_rn(mask_f[1]);

        float2 x_data = x_cast[j];
        __half2* x_data_h = reinterpret_cast<__half2*>(&x_data);

        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);

        result_h[0] = x_data_h[0] * h_scale * mask_h[0];
        result_h[1] = x_data_h[1] * h_scale * mask_h[1];

        x_cast[j] = result_f;
    }

#else

    CUDA_1D_KERNEL_LOOP(j, N / 2)
    {
        int i = j * 2;
        Xdata[i] = (__half)((float)Xdata[i] * scale * mask[i]);
        Xdata[i + 1] = (__half)((float)Xdata[i + 1] * scale * mask[i + 1]);
    }

#endif
}

template <typename T>
void launch_dropout_grad(T* vals, uint8_t* mask, int total_count, float ratio, cudaStream_t stream)
{
    const float scale = 1. / (1. - ratio);
    dropout_grad_kernel<<<DS_GET_BLOCKS(total_count / 2), DS_CUDA_NUM_THREADS, 0, stream>>>(
        total_count, scale, vals, mask);
}

template void launch_dropout_grad(float* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  cudaStream_t stream);
template void launch_dropout_grad(__half* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  cudaStream_t stream);

__global__ void dropout_grad_kernel(const int N,
                                    const float scale,
                                    const float* Xdata,
                                    float* out,
                                    uint8_t* mask)
{
    CUDA_1D_KERNEL_LOOP(i, N) { out[i] = Xdata[i] * scale * mask[i]; }
}

__global__ void dropout_grad_kernel(const int N,
                                    const float scale,
                                    const __half* Xdata,
                                    __half* out,
                                    uint8_t* mask)
{
    CUDA_1D_KERNEL_LOOP(j, N / 2)
    {
        int i = j * 2;
        out[i] = (__half)((float)Xdata[i] * scale * mask[i]);
        out[i + 1] = (__half)((float)Xdata[i + 1] * scale * mask[i + 1]);
    }
}

template <typename T>
void launch_dropout_grad(T* vals_out,
                         const T* vals,
                         uint8_t* mask,
                         int total_count,
                         float ratio,
                         cudaStream_t stream)
{
    const float scale = 1. / (1. - ratio);
    dropout_grad_kernel<<<DS_GET_BLOCKS(total_count / 2), DS_CUDA_NUM_THREADS, 0, stream>>>(
        total_count, scale, vals, vals_out, mask);
}
template void launch_dropout_grad(float*,
                                  const float* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  cudaStream_t stream);
template void launch_dropout_grad(__half*,
                                  const __half* vals,
                                  uint8_t* mask,
                                  int total_count,
                                  float ratio,
                                  cudaStream_t stream);

__global__ void dropout_kernel(const int dim,
                               const float ratio,
                               const float* bias,
                               float* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float4* Xdata_cast = reinterpret_cast<float4*>(Xdata);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);

    {
        float4 rand = curand_uniform4(&state);
        uint8_t m[4];

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        int i = blockIdx.x * dim + tid * 4;

        float4 x_data = Xdata_cast[idx];
        float4 b_data = bias_cast[tid];

        x_data.x += b_data.x;
        x_data.y += b_data.y;
        x_data.z += b_data.z;
        x_data.w += b_data.w;

        x_data.x = x_data.x * scale * m[0];
        x_data.y = x_data.y * scale * m[1];
        x_data.z = x_data.z * scale * m[2];
        x_data.w = x_data.w * scale * m[3];

        mask[i] = (uint8_t)m[0];
        mask[i + 1] = (uint8_t)m[1];
        mask[i + 2] = (uint8_t)m[2];
        mask[i + 3] = (uint8_t)m[3];

        Xdata_cast[idx] = x_data;
    }
}

__global__ void dropout_kernel(const int dim,
                               const float ratio,
                               const __half* bias,
                               __half* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float2* Xdata_cast = reinterpret_cast<float2*>(Xdata);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    {
        int i = blockIdx.x * dim + tid * 4;

        float4 rand = curand_uniform4(&state);

        float2 data_f;
        __half2* data_h = reinterpret_cast<__half2*>(&data_f);

        float2 bias_f;
        __half2* bias_h = reinterpret_cast<__half2*>(&bias_f);

        data_f = Xdata_cast[idx];
        bias_f = bias_cast[tid];

        float2 data_h_0 = __half22float2(data_h[0]);
        float2 data_h_1 = __half22float2(data_h[1]);

        float2 bias_h_0 = __half22float2(bias_h[0]);
        float2 bias_h_1 = __half22float2(bias_h[1]);

        data_h_0.x += bias_h_0.x;
        data_h_0.y += bias_h_0.y;
        data_h_1.x += bias_h_1.x;
        data_h_1.y += bias_h_1.y;

        uint8_t m[4];  // = mask + i;

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        data_h_0.x = __float2half(data_h_0.x * scale * m[0]);
        data_h_0.y = __float2half(data_h_0.y * scale * m[1]);
        data_h_1.x = __float2half(data_h_1.x * scale * m[2]);
        data_h_1.y = __float2half(data_h_1.y * scale * m[3]);

        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);

        result_h[0] = __float22half2_rn(data_h_0);
        result_h[1] = __float22half2_rn(data_h_1);

        Xdata_cast[idx] = result_f;

        mask[i] = m[0];
        mask[i + 1] = m[1];
        mask[i + 2] = m[2];
        mask[i + 3] = m[3];
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    cudaStream_t stream)
{
    dim3 grid_dim(batch);     // DS_GET_BLOCKS(total_count/4);
    dim3 block_dim(dim / 4);  // DS_CUDA_NUM_THREADS;

    uint64_t inc = (batch * dim) / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc);

    dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(dim, ratio, bias, out, mask, seed);
}

template void launch_dropout(float*,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);
template void launch_dropout(__half*,
                             const __half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);

__global__ void dropout_kernel(const int dim,
                               const float ratio,
                               const float* input,
                               const float* residual,
                               const float* bias,
                               float* out,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float4* out_cast = reinterpret_cast<float4*>(out);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);
    const float4* residual_cast = reinterpret_cast<const float4*>(residual);
    const float4* input_cast = reinterpret_cast<const float4*>(input);

    {
        float4 rand = curand_uniform4(&state);
        uint8_t m[4];
        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        // int bid = k * blockDim.x + tid;
        int i = blockIdx.x * dim + tid * 4;

        float4 out_data = out_cast[idx];
        float4 b_data = bias_cast[tid];
        float4 res_data = residual_cast[idx];
        float4 inp_data = input_cast[idx];

        out_data.x = (b_data.x + inp_data.x);
        out_data.y = (b_data.y + inp_data.y);
        out_data.z = (b_data.z + inp_data.z);
        out_data.w = (b_data.w + inp_data.w);

        out_data.x = out_data.x * scale * m[0];
        out_data.y = out_data.y * scale * m[1];
        out_data.z = out_data.z * scale * m[2];
        out_data.w = out_data.w * scale * m[3];

        out_data.x += res_data.x;
        out_data.y += res_data.y;
        out_data.z += res_data.z;
        out_data.w += res_data.w;

        mask[i] = m[0];
        mask[i + 1] = m[1];
        mask[i + 2] = m[2];
        mask[i + 3] = m[3];

        out_cast[idx] = out_data;
    }
}

__global__ void dropout_kernel(const int dim,
                               const float ratio,
                               const __half* input,
                               const __half* residual,
                               const __half* bias,
                               __half* out,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float2* out_cast = reinterpret_cast<float2*>(out);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);
    const float2* residual_cast = reinterpret_cast<const float2*>(residual);
    const float2* input_cast = reinterpret_cast<const float2*>(input);

    {
        int i = blockIdx.x * dim + tid * 4;

        float4 rand = curand_uniform4(&state);

        float2 data_f;
        __half2* data_h = reinterpret_cast<__half2*>(&data_f);

        float2 bias_f;
        __half2* bias_h = reinterpret_cast<__half2*>(&bias_f);

        float2 residual_f;
        __half2* residual_h = reinterpret_cast<__half2*>(&residual_f);

        float2 input_f;
        __half2* input_h = reinterpret_cast<__half2*>(&input_f);

        data_f = out_cast[idx];
        bias_f = bias_cast[tid];
        residual_f = residual_cast[idx];
        input_f = input_cast[idx];

        float2 data_h_0 = __half22float2(data_h[0]);
        float2 data_h_1 = __half22float2(data_h[1]);

        float2 bias_h_0 = __half22float2(bias_h[0]);
        float2 bias_h_1 = __half22float2(bias_h[1]);

        float2 residual_h_0 = __half22float2(residual_h[0]);
        float2 residual_h_1 = __half22float2(residual_h[1]);

        float2 input_h_0 = __half22float2(input_h[0]);
        float2 input_h_1 = __half22float2(input_h[1]);

        data_h_0.x = (bias_h_0.x + input_h_0.x);
        data_h_0.y = (bias_h_0.y + input_h_0.y);
        data_h_1.x = (bias_h_1.x + input_h_1.x);
        data_h_1.y = (bias_h_1.y + input_h_1.y);

        uint8_t m[4];  // = mask + i;

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        data_h_0.x = __float2half(data_h_0.x * scale * m[0]);
        data_h_0.y = __float2half(data_h_0.y * scale * m[1]);
        data_h_1.x = __float2half(data_h_1.x * scale * m[2]);
        data_h_1.y = __float2half(data_h_1.y * scale * m[3]);

        data_h_0.x += residual_h_0.x;
        data_h_0.y += residual_h_0.y;
        data_h_1.x += residual_h_1.x;
        data_h_1.y += residual_h_1.y;

        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);

        result_h[0] = __float22half2_rn(data_h_0);
        result_h[1] = __float22half2_rn(data_h_1);

        out_cast[idx] = result_f;

        mask[i] = m[0];
        mask[i + 1] = m[1];
        mask[i + 2] = m[2];
        mask[i + 3] = m[3];
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* input,
                    const T* residual,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    cudaStream_t stream)
{
    dim3 grid_dim(batch);     // DS_GET_BLOCKS(total_count/4);
    dim3 block_dim(dim / 4);  // DS_CUDA_NUM_THREADS;

    uint64_t inc = (batch * dim) / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(inc);

    dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dim, ratio, input, residual, bias, out, mask, seed);
}

template void launch_dropout(float*,
                             const float*,
                             const float* residual,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);
template void launch_dropout(__half*,
                             const __half*,
                             const __half* residual,
                             const __half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);
