#include "custom_cuda_layers.h"

namespace cg = cooperative_groups;

/*
Fused bias add, residual (elementwise) add, and normalization layer.

Unlike the GELU, which doesn't require template parameters, this layer does since it
does rely fairly heavily on unrolling loops. Currently, I exclude bounds checks and
assume that the number of elements is a multiple of a power of 2. Default behavior
for our purposes uses 256 threads for floats, and 128 threads for __half. This restriction
is a result of using the shift parameter to perform the minimum number of register file
shuffles necessary, which requires the number of threads in the secondary reduction to
be 1, 2, 4, 8, 16, or 32. The number of threads here corresponds to the number of complete
warps in the threadblock.

For FP16, this kernel does not promote to FP32 in order to utilize the 2x throughput for
__half2 instructions, and avoid the conversion overhead (1/8 of __hal2 arithmetic).

For specific launch constraints, see the launch functions.
*/

template <int row_stride, int iterations>
__global__ void fused_bias_residual_layer_norm(float* vals,
                                               const float* residual,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               float* vars,
                                               float* means)
{
    constexpr int iteration_stride = row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id / 32;

    float vals_arr[iterations];
    __shared__ float shr[iteration_stride >> 5];

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = residual[row * row_stride + i * iteration_stride + id];
        sum += vals_arr[i];
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

#if !defined(__STOCHASTIC_MODE__) || __CUDA_ARCH__ < 700
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }

    sum = g.shfl(sum, 0);
    float mean = sum / row_stride;
    if (training)
        if (g.thread_rank() == 0) means[row] = mean;
    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        variance += (vals_arr[i] - mean) * (vals_arr[i] - mean);
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= row_stride;
    variance += epsilon;
    if (training)
        if (g.thread_rank() == 0) vars[row] = variance;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = (vals_arr[i] - mean) * rsqrtf(variance);
        vals_arr[i] =
            vals_arr[i] * gamma[i * iteration_stride + id] + beta[i * iteration_stride + id];
        vals[row * row_stride + i * iteration_stride + id] = vals_arr[i];
    }
}

template <int row_stride, int iterations>
__global__ void fused_bias_residual_layer_norm(__half* vals,
                                               const __half* residual,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               __half* vars,
                                               __half* means)
{
#if __CUDA_ARCH__ >= 700
    constexpr int iteration_stride = row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;

    __half2 vals_arr[iterations];
    float2 vals_f[iterations];
    __shared__ float shr[iteration_stride >> 5];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_f[i] = __half22float2(residual_cast[row * row_stride + i * iteration_stride + id]);
        sum += vals_f[i].x;
        sum += vals_f[i].y;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride * 2);

    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        variance += (vals_f[i].x - mean) * (vals_f[i].x - mean);
        variance += (vals_f[i].y - mean) * (vals_f[i].y - mean);
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= (row_stride * 2);
    variance += epsilon;

    __half2 mean_h = __float2half2_rn(mean);
    __half2 variance_h = __float2half2_rn(variance);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);

    if (training && g.thread_rank() == 0) {
        vars[row] = __float2half(variance);
        means[row] = __float2half(mean);
    }

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = __float22half2_rn(vals_f[i]);
        vals_arr[i] = (vals_arr[i] - mean_h) * h2rsqrt(variance_h);
        vals_arr[i] = vals_arr[i] * gamma_cast[i * iteration_stride + id] +
                      beta_cast[i * iteration_stride + id];
        vals_cast[row * row_stride + i * iteration_stride + id] = vals_arr[i];
    }
#endif
}

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means);

template <>
void launch_bias_residual_layer_norm<float>(float* vals,
                                            const float* residual,
                                            const float* gamma,
                                            const float* beta,
                                            float epsilon,
                                            int batch_size,
                                            int hidden_dim,
                                            cudaStream_t stream,
                                            bool preLayerNorm,
                                            bool training,
                                            float* vars,
                                            float* means)
{
    constexpr int threads = THREADS;

    dim3 grid_dim(batch_size);

    dim3 block_dim(threads);

    // There are some limitations to call below functions, now just enumerate the situations.
    if (hidden_dim == 768)
        fused_bias_residual_layer_norm<768, 3><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 512)
        fused_bias_residual_layer_norm<512, 2><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 1024)
        fused_bias_residual_layer_norm<1024, 4><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 1536)
        fused_bias_residual_layer_norm<1536, 6><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 2048)
        fused_bias_residual_layer_norm<2048, 8><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 2560)
        fused_bias_residual_layer_norm<2560, 10><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <>
void launch_bias_residual_layer_norm<__half>(__half* vals,
                                             const __half* residual,
                                             const __half* gamma,
                                             const __half* beta,
                                             float epsilon,
                                             int batch_size,
                                             int hidden_dim,
                                             cudaStream_t stream,
                                             bool preLayerNorm,
                                             bool training,
                                             __half* vars,
                                             __half* means)
{
    constexpr int threads = 128;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    // There are some limitations to call below functions, now just enumerate the situations.
    if (hidden_dim == 768)
        fused_bias_residual_layer_norm<384, 3><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 512)
        fused_bias_residual_layer_norm<256, 2><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 1024)
        fused_bias_residual_layer_norm<512, 4><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 1536)
        fused_bias_residual_layer_norm<768, 6><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 2048)
        fused_bias_residual_layer_norm<1024, 8><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else if (hidden_dim == 2560)
        fused_bias_residual_layer_norm<1280, 10><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <int row_stride, int iterations>
__global__ void fused_bias_residual_layer_norm(float* vals,
                                               const float* residual,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               float* vars)
{
    constexpr int iteration_stride = row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id / 32;

    float vals_arr[iterations];
    __shared__ float shr[iteration_stride >> 5];

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = residual[row * row_stride + i * iteration_stride + id];
        sum += vals_arr[i];
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

#if !defined(__STOCHASTIC_MODE__) || __CUDA_ARCH__ < 700
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }

    sum = g.shfl(sum, 0);
    float mean = sum / row_stride;
    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        variance += (vals_arr[i] - mean) * (vals_arr[i] - mean);
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= row_stride;
    variance += epsilon;
    if (training)
        if (g.thread_rank() == 0) vars[row] = variance;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = (vals_arr[i] - mean) * rsqrtf(variance);
        vals_arr[i] =
            vals_arr[i] * gamma[i * iteration_stride + id] + beta[i * iteration_stride + id];
        vals[row * row_stride + i * iteration_stride + id] = vals_arr[i];
    }
}

template <int row_stride, int iterations>
__global__ void fused_bias_residual_layer_norm(__half* vals,
                                               const __half* residual,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               __half* vars)
{
#if __CUDA_ARCH__ >= 700
    constexpr int iteration_stride = row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;

    __half2 vals_arr[iterations];
    float2 vals_f[iterations];
    __shared__ float shr[iteration_stride >> 5];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_f[i] = __half22float2(residual_cast[row * row_stride + i * iteration_stride + id]);
        sum += vals_f[i].x;
        sum += vals_f[i].y;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride * 2);

    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        variance += (vals_f[i].x - mean) * (vals_f[i].x - mean);
        variance += (vals_f[i].y - mean) * (vals_f[i].y - mean);
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= (row_stride * 2);
    variance += epsilon;

    __half2 mean_h = __float2half2_rn(mean);
    __half2 variance_h = __float2half2_rn(variance);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);

    if (training && g.thread_rank() == 0) vars[row] = __float2half(variance);

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = __float22half2_rn(vals_f[i]);
        vals_arr[i] = (vals_arr[i] - mean_h) * h2rsqrt(variance_h);
        vals_arr[i] = vals_arr[i] * gamma_cast[i * iteration_stride + id] +
                      beta_cast[i * iteration_stride + id];
        vals_cast[row * row_stride + i * iteration_stride + id] = vals_arr[i];
    }
#endif
}

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars);

/*
To tune this launch the following restrictions must be met:

For float:
row_stride == hidden_size
threads * iterations == row_stride
threads is in [32, 64, 128, 256, 512, 1024]

For half:
row_stride == hidden_size / 2
threads * iterations == row_stride
threads is in [32, 64, 128, 256, 512, 1024]

*/

template <>
void launch_bias_residual_layer_norm<float>(float* vals,
                                            const float* residual,
                                            const float* gamma,
                                            const float* beta,
                                            float epsilon,
                                            int batch_size,
                                            int hidden_dim,
                                            cudaStream_t stream,
                                            bool preLayerNorm,
                                            bool training,
                                            float* vars)
{
    constexpr int threads = THREADS;

    dim3 grid_dim(batch_size);

    dim3 block_dim(threads);

    // There are some limitations to call below functions, now just enumerate the situations.
    if (hidden_dim == 768)
        fused_bias_residual_layer_norm<768, 3><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 512)
        fused_bias_residual_layer_norm<512, 2><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 1024)
        fused_bias_residual_layer_norm<1024, 4><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 1536)
        fused_bias_residual_layer_norm<1536, 6><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 2048)
        fused_bias_residual_layer_norm<2048, 8><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 2560)
        fused_bias_residual_layer_norm<2560, 10><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <>
void launch_bias_residual_layer_norm<__half>(__half* vals,
                                             const __half* residual,
                                             const __half* gamma,
                                             const __half* beta,
                                             float epsilon,
                                             int batch_size,
                                             int hidden_dim,
                                             cudaStream_t stream,
                                             bool preLayerNorm,
                                             bool training,
                                             __half* vars)
{
    constexpr int threads = 128;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    // There are some limitations to call below functions, now just enumerate the situations.
    if (hidden_dim == 768)
        fused_bias_residual_layer_norm<384, 3><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 512)
        fused_bias_residual_layer_norm<256, 2><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 1024)
        fused_bias_residual_layer_norm<512, 4><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 1536)
        fused_bias_residual_layer_norm<768, 6><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 2048)
        fused_bias_residual_layer_norm<1024, 8><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else if (hidden_dim == 2560)
        fused_bias_residual_layer_norm<1280, 10><<<grid_dim, block_dim, 0, stream>>>(
            vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using either X_hat or
 * normalize input (invertible).
 * Combine transpose with gradients computation.
 */

template <typename T>
__global__ void LayerNormBackward1(const T* __restrict__ out_grad,
                                   const T* __restrict__ vals_hat,
                                   const T* __restrict__ gamma,
                                   const T* __restrict__ betta,
                                   T* __restrict__ gamma_grad,
                                   T* __restrict__ betta_grad,
                                   int rows,
                                   int width,
                                   bool invertible)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = blockIdx.x * TILE_DIM + threadIdx.y;
    float betta_reg = (invertible ? (float)betta[pos] : 0.0f);
    float gamma_reg = (float)gamma[pos];

    // Loop across matrix height
    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (invertible ? ((float)vals_hat[offset] - betta_reg) / gamma_reg
                                : (float)vals_hat[offset]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using the input to
 * the normalize.
 * Combine transpose with gradients computation.
 */

template <typename T>
__global__ void LayerNormBackward1(const T* __restrict__ out_grad,
                                   const T* __restrict__ X_data,
                                   const T* __restrict__ vars,
                                   const T* __restrict__ means,
                                   T* __restrict__ gamma_grad,
                                   T* __restrict__ betta_grad,
                                   int rows,
                                   int width)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = blockIdx.x * TILE_DIM + threadIdx.y;
    // Loop across matrix height

    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (float)X_data[offset];
        val = (val - (float)means[r]) * rsqrtf((float)vars[r]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}
/*

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is invertible!
 * We do the backward using the X_hat (X - u) / sqrt(variance) or the output of Normalization.
 */

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2(const float* out_grad,
                                   const float* vals_hat,
                                   const float* gamma,
                                   const float* betta,
                                   const float* vars,
                                   float* inp_grad,
                                   bool invertible)
{
    constexpr int iterations = row_stride / THREADS;
    constexpr int iteration_stride = THREADS;  // row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[warp_num];

    float vals_arr[iterations];
    float vals_hat_arr[iterations];

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] = (invertible ? (vals_hat[row * row_stride + i * iteration_stride + id] -
                                         betta[i * iteration_stride + id]) /
                                            gamma_reg
                                      : vals_hat[row * row_stride + i * iteration_stride + id]);
    }

    float var_reg = vars[row];

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += vals_hat_arr[i] * vals_arr[i] *
               sqrtf(var_reg);           // dval_hat = gamma * (x - u) * out_grad
        vals_arr[i] *= rsqrtf(var_reg);  // dvar_inv = gamma * out_grad / sqrt(var)
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++)
        inp_grad[row * row_stride + i * iteration_stride + id] = (vals_arr[i] - sum);
}

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2(const __half* out_grad,
                                   const __half* vals_hat,
                                   const __half* gamma,
                                   const __half* betta,
                                   const __half* vars,
                                   __half* inp_grad,
                                   bool invertible)
{
    constexpr int iteration_stride = THREADS / 2;  // row_stride / iterations;
    constexpr int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num =
        (iteration_stride < row_stride ? iteration_stride : row_stride) / WARP_SIZE;
    __shared__ float partialSum[warp_num];

    __half2 vals_arr[iterations];
    float2 vals_arr_f[iterations];
    __half2 vals_hat_arr[iterations];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h = reinterpret_cast<const __half2*>(out_grad);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(vals_hat);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);
    const __half2* betta_h = (invertible ? reinterpret_cast<const __half2*>(betta) : nullptr);

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] = (invertible ? (vals_hat_h[row * row_stride + i * iteration_stride + id] -
                                         betta_h[i * iteration_stride + id]) /
                                            gamma_reg
                                      : vals_hat_h[row * row_stride + i * iteration_stride + id]);
    }
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        __half2 result_h = (vals_hat_arr[i] * vals_arr[i] * h2sqrt(var_reg));
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 temp = ((-sum_h * vals_hat_arr[i]) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 temp_f = __half22float2(temp);
        vals_arr_f[i].x += temp_f.x;
        vals_arr_f[i].y += temp_f.y;
    }
    sum = 0.f;

    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);

        inp_grad_h[row * row_stride + i * iteration_stride + id] = temp;
    }
}

template <>
void launch_layerNorm_backward<float>(const float* out_grad,
                                      const float* vals_hat,
                                      const float* vars,
                                      const float* gamma,
                                      float* gamma_grad,
                                      float* betta_grad,
                                      float* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2],
                                      bool invertible,
                                      const float* betta)
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads);

    if (hidden_dim == 768)
        LayerNormBackward2<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 512)
        LayerNormBackward2<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1024)
        LayerNormBackward2<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1536)
        LayerNormBackward2<1536><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2048)
        LayerNormBackward2<2048><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2560)
        LayerNormBackward2<2560><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <>
void launch_layerNorm_backward<__half>(const __half* out_grad,
                                       const __half* vals_hat,
                                       const __half* vars,
                                       const __half* gamma,
                                       __half* gamma_grad,
                                       __half* betta_grad,
                                       __half* inp_grad,
                                       int batch,
                                       int hidden_dim,
                                       cudaStream_t stream[2],
                                       bool invertible,
                                       const __half* betta)
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads / 2);

    if (hidden_dim == 768)
        LayerNormBackward2<384><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 512)
        LayerNormBackward2<256><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1024)
        LayerNormBackward2<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1536)
        LayerNormBackward2<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2048)
        LayerNormBackward2<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2560)
        LayerNormBackward2<1280><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is not invertible!
 * We do the backward using the input (X)
 */

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2(const float* out_grad,
                                   const float* X_vals,
                                   const float* gamma,
                                   const float* vars,
                                   const float* means,
                                   float* inp_grad)
{
    constexpr int iterations = row_stride / THREADS;
    constexpr int iteration_stride = THREADS;  // row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[warp_num];

    float vals_arr[iterations];

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
    }

    float var_reg = vars[row];
    float mean_reg = means[row];

    float sum = 0;
    float xu[iterations];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (X_vals[row * row_stride + i * iteration_stride + id] - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrtf(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++)
        inp_grad[row * row_stride + i * iteration_stride + id] = (vals_arr[i] - sum);
}

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2(const __half* out_grad,
                                   const __half* X_vals,
                                   const __half* gamma,
                                   const __half* vars,
                                   const __half* means,
                                   __half* inp_grad)
{
    constexpr int iteration_stride = THREADS / 2;  // row_stride / iterations;
    constexpr int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num =
        (iteration_stride < row_stride ? iteration_stride : row_stride) / WARP_SIZE;

    __shared__ float partialSum[warp_num];

    __half2 vals_arr[iterations];
    float2 vals_arr_f[iterations];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h = reinterpret_cast<const __half2*>(out_grad);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(X_vals);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;  // out_grad * gamma
    }
    __half mean_h = means[row];
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);
    __half2 mean_reg = __halves2half2(mean_h, mean_h);
    __half2 xu[iterations];

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        xu[i] = (vals_hat_h[row * row_stride + i * iteration_stride + id] - mean_reg);
        __half2 result_h = (xu[i] * vals_arr[i]);
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 xu_grad = ((-sum_h * xu[i] * h2rsqrt(var_reg)) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 xu_grad_f = __half22float2(xu_grad);
        vals_arr_f[i].x += xu_grad_f.x;
        vals_arr_f[i].y += xu_grad_f.y;
    }

    sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);
        inp_grad_h[row * row_stride + i * iteration_stride + id] = temp;
    }
}

template <>
void launch_layerNorm_backward<float>(const float* out_grad,
                                      const float* X_data,
                                      const float* vars,
                                      const float* means,
                                      const float* gamma,
                                      float* gamma_grad,
                                      float* betta_grad,
                                      float* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2])
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads);

    if (hidden_dim == 768)
        LayerNormBackward2<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 512)
        LayerNormBackward2<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1024)
        LayerNormBackward2<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1536)
        LayerNormBackward2<1536><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2048)
        LayerNormBackward2<2048><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2560)
        LayerNormBackward2<2560><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <>
void launch_layerNorm_backward<__half>(const __half* out_grad,
                                       const __half* X_data,
                                       const __half* vars,
                                       const __half* means,
                                       const __half* gamma,
                                       __half* gamma_grad,
                                       __half* betta_grad,
                                       __half* inp_grad,
                                       int batch,
                                       int hidden_dim,
                                       cudaStream_t stream[2])
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads / 2);

    if (hidden_dim == 768)
        LayerNormBackward2<384><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 512)
        LayerNormBackward2<256><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1024)
        LayerNormBackward2<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1536)
        LayerNormBackward2<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2048)
        LayerNormBackward2<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2560)
        LayerNormBackward2<1280><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad, X_data, gamma, vars, means, inp_grad);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <typename T>
__global__ void LayerNormBackward1_fused_add(const T* __restrict__ out_grad1,
                                             const T* __restrict__ out_grad2,
                                             const T* __restrict__ vals_hat,
                                             const T* __restrict__ gamma,
                                             const T* __restrict__ betta,
                                             T* __restrict__ gamma_grad,
                                             T* __restrict__ betta_grad,
                                             int rows,
                                             int width,
                                             bool invertible)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = blockIdx.x * TILE_DIM + threadIdx.y;
    float betta_reg = (invertible ? (float)betta[pos] : 0.0f);
    float gamma_reg = (float)gamma[pos];

    // Loop across matrix height
    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad1[offset] + (float)out_grad2[offset];
        float val = (invertible ? ((float)vals_hat[offset] - betta_reg) / gamma_reg
                                : (float)vals_hat[offset]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

template <typename T>
__global__ void LayerNormBackward1_fused_add(const T* __restrict__ out_grad1,
                                             const T* __restrict__ out_grad2,
                                             const T* __restrict__ X_data,
                                             const T* __restrict__ vars,
                                             const T* __restrict__ means,
                                             T* __restrict__ gamma_grad,
                                             T* __restrict__ betta_grad,
                                             int rows,
                                             int width)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = blockIdx.x * TILE_DIM + threadIdx.y;
    // Loop across matrix height

    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad1[offset] + (float)out_grad2[offset];
        float val = (float)X_data[offset];
        val = (val - (float)means[r]) * rsqrtf((float)vars[r]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2_fused_add(const float* out_grad1,
                                             const float* out_grad2,
                                             const float* vals_hat,
                                             const float* gamma,
                                             const float* betta,
                                             const float* vars,
                                             float* inp_grad,
                                             bool invertible)
{
    constexpr int iterations = row_stride / THREADS;
    constexpr int iteration_stride = THREADS;  // row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[warp_num];

    float vals_arr[iterations];
    float vals_hat_arr[iterations];

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad1[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] = (invertible ? (vals_hat[row * row_stride + i * iteration_stride + id] -
                                         betta[i * iteration_stride + id]) /
                                            gamma_reg
                                      : vals_hat[row * row_stride + i * iteration_stride + id]);
    }

    float var_reg = vars[row];

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += vals_hat_arr[i] * vals_arr[i] * sqrtf(var_reg);
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++)
        inp_grad[row * row_stride + i * iteration_stride + id] =
            (vals_arr[i] - sum) + out_grad2[row * row_stride + i * iteration_stride + id];
}

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2_fused_add(const __half* out_grad1,
                                             const __half* out_grad2,
                                             const __half* vals_hat,
                                             const __half* gamma,
                                             const __half* betta,
                                             const __half* vars,
                                             __half* inp_grad,
                                             bool invertible)
{
    constexpr int iteration_stride = THREADS / 2;  // row_stride / iterations;
    constexpr int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num =
        (iteration_stride < row_stride ? iteration_stride : row_stride) / WARP_SIZE;
    __shared__ float partialSum[warp_num];

    __half2 vals_arr[iterations];
    float2 vals_arr_f[iterations];
    __half2 vals_hat_arr[iterations];

    // float2 result[iterations];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h1 = reinterpret_cast<const __half2*>(out_grad1);
    const __half2* out_grad_h2 = reinterpret_cast<const __half2*>(out_grad2);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(vals_hat);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);
    const __half2* betta_h = (invertible ? reinterpret_cast<const __half2*>(betta) : nullptr);

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h1[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;  // out_grad * gamma
        vals_hat_arr[i] = (invertible ? (vals_hat_h[row * row_stride + i * iteration_stride + id] -
                                         betta_h[i * iteration_stride + id]) /
                                            gamma_reg
                                      : vals_hat_h[row * row_stride + i * iteration_stride + id]);
    }
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        __half2 result_h = (vals_hat_arr[i] * vals_arr[i] * h2sqrt(var_reg));
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 temp = ((-sum_h * vals_hat_arr[i]) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 temp_f = __half22float2(temp);
        vals_arr_f[i].x += temp_f.x;
        vals_arr_f[i].y += temp_f.y;
    }
    sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);

        inp_grad_h[row * row_stride + i * iteration_stride + id] =
            temp + out_grad_h2[row * row_stride + i * iteration_stride + id];
    }
}

template <>
void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                const float* out_grad2,
                                                const float* vals_hat,
                                                const float* vars,
                                                const float* gamma,
                                                float* gamma_grad,
                                                float* betta_grad,
                                                float* inp_grad,
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2],
                                                bool invertible,
                                                const float* betta)
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);
    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads);

    if (hidden_dim == 768)
        LayerNormBackward2_fused_add<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 512)
        LayerNormBackward2_fused_add<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1024)
        LayerNormBackward2_fused_add<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1536)
        LayerNormBackward2_fused_add<1536><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2048)
        LayerNormBackward2_fused_add<2048><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2560)
        LayerNormBackward2_fused_add<2560><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <>
void launch_layerNorm_backward_fused_add<__half>(const __half* out_grad1,
                                                 const __half* out_grad2,
                                                 const __half* vals_hat,
                                                 const __half* vars,
                                                 const __half* gamma,
                                                 __half* gamma_grad,
                                                 __half* betta_grad,
                                                 __half* inp_grad,
                                                 int batch,
                                                 int hidden_dim,
                                                 cudaStream_t stream[2],
                                                 bool invertible,
                                                 const __half* betta)
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads / 2);

    if (hidden_dim == 768)
        LayerNormBackward2_fused_add<384><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 512)
        LayerNormBackward2_fused_add<256><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1024)
        LayerNormBackward2_fused_add<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 1536)
        LayerNormBackward2_fused_add<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2048)
        LayerNormBackward2_fused_add<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else if (hidden_dim == 2560)
        LayerNormBackward2_fused_add<1280><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is not invertible!
 * We do the backward using the input (X)
 */

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2_fused_add(const float* out_grad1,
                                             const float* out_grad2,
                                             const float* X_vals,
                                             const float* gamma,
                                             const float* vars,
                                             const float* means,
                                             float* inp_grad)
{
    constexpr int iterations = row_stride / THREADS;
    constexpr int iteration_stride = THREADS;  // row_stride / iterations;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[warp_num];

    float vals_arr[iterations];
    float vals_hat_arr[iterations];

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad1[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] = X_vals[row * row_stride + i * iteration_stride + id];
    }

    float var_reg = vars[row];
    float mean_reg = means[row];

    float sum = 0;
    float xu[iterations];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (vals_hat_arr[i] - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrtf(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++)
        inp_grad[row * row_stride + i * iteration_stride + id] =
            (vals_arr[i] - sum) + out_grad2[row * row_stride + i * iteration_stride + id];
    ;
}

template <int row_stride>  // Hidden_Dim
__global__ void LayerNormBackward2_fused_add(const __half* out_grad1,
                                             const __half* out_grad2,
                                             const __half* X_vals,
                                             const __half* gamma,
                                             const __half* vars,
                                             const __half* means,
                                             __half* inp_grad)
{
    constexpr int iteration_stride = THREADS / 2;  // row_stride / iterations;
    constexpr int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    constexpr int warp_num =
        (iteration_stride < row_stride ? iteration_stride : row_stride) / WARP_SIZE;

    __shared__ float partialSum[warp_num];

    __half2 vals_arr[iterations];
    float2 vals_arr_f[iterations];
    __half2 vals_hat_arr[iterations];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h1 = reinterpret_cast<const __half2*>(out_grad1);
    const __half2* out_grad_h2 = reinterpret_cast<const __half2*>(out_grad2);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(X_vals);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h1[row * row_stride + i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;  // out_grad * gamma
        vals_hat_arr[i] = vals_hat_h[row * row_stride + i * iteration_stride + id];
    }

    __half mean_h = means[row];
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);
    __half2 mean_reg = __halves2half2(mean_h, mean_h);
    __half2 xu[iterations];

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        xu[i] = (vals_hat_arr[i] - mean_reg);
        __half2 result_h = (xu[i] * vals_arr[i]);
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 xu_grad = ((-sum_h * xu[i] * h2rsqrt(var_reg)) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 xu_grad_f = __half22float2(xu_grad);
        vals_arr_f[i].x += xu_grad_f.x;
        vals_arr_f[i].y += xu_grad_f.y;
    }

    sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);
        inp_grad_h[row * row_stride + i * iteration_stride + id] =
            temp + out_grad_h2[row * row_stride + i * iteration_stride + id];
    }
}

template <>
void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                const float* out_grad2,
                                                const float* X_data,
                                                const float* vars,
                                                const float* means,
                                                const float* gamma,
                                                float* gamma_grad,
                                                float* betta_grad,
                                                float* inp_grad,
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2])
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads);

    if (hidden_dim == 768)
        LayerNormBackward2_fused_add<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 512)
        LayerNormBackward2_fused_add<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1024)
        LayerNormBackward2_fused_add<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1536)
        LayerNormBackward2_fused_add<1536><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2048)
        LayerNormBackward2_fused_add<2048><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2560)
        LayerNormBackward2_fused_add<2560><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}

template <>
void launch_layerNorm_backward_fused_add<__half>(const __half* out_grad1,
                                                 const __half* out_grad2,
                                                 const __half* X_data,
                                                 const __half* vars,
                                                 const __half* means,
                                                 const __half* gamma,
                                                 __half* gamma_grad,
                                                 __half* betta_grad,
                                                 __half* inp_grad,
                                                 int batch,
                                                 int hidden_dim,
                                                 cudaStream_t stream[2])
{
    constexpr int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);
    dim3 block_dim2(threads / 2);

    if (hidden_dim == 768)
        LayerNormBackward2_fused_add<384><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 512)
        LayerNormBackward2_fused_add<256><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1024)
        LayerNormBackward2_fused_add<512><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 1536)
        LayerNormBackward2_fused_add<768><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2048)
        LayerNormBackward2_fused_add<1024><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else if (hidden_dim == 2560)
        LayerNormBackward2_fused_add<1280><<<grid_dim2, block_dim2, 0, stream[1]>>>(
            out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad);
    else
        throw std::runtime_error("Unsupport hidden_dim.");
}
