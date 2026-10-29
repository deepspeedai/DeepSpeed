// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

// Latency-optimized inference all-reduce for small messages on P2P-capable
// GPU pairs. Fills the deepspeed.comm.inference_all_reduce hook the same way
// the CPU backend fills it with shm+AVX: the op stays accelerator-owned and
// every backend can walk the same route.
//
// Stream-ordered 5-kernel protocol (graph-capturable, correctness by kernel
// boundaries instead of a grid-wide barrier):
//   k0 seq_d[0]++                       replay-safe sequence number
//   k1 copy input -> my IPC buf[p]      (all copies complete at boundary)
//   k2 system fence; my flag[p] = seq
//   k3 spin until peer flag[p] >= seq
//   k4 out[i] = in[i] + peer_buf[p][i]
// Double buffering (p = seq & 1) lets call N+1 publish while call N reduces.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CAR_MAX_BYTES (4 * 1024 * 1024)
#define CAR_THREADS 256

namespace {

struct CarState {
    bool ready = false;
    int world = 0;
    char* my_base = nullptr;   // python-owned: [flag0|flag1|pad][buf0][buf1]
    char* peer_base = nullptr;
    int* seq_d = nullptr;
    torch::Tensor buf_ref;  // keeps the python buffer alive
    torch::Tensor seq_ref;
};
CarState g_state;

constexpr size_t kFlagOff0 = 0;
constexpr size_t kFlagOff1 = 4;
constexpr size_t kBufBase = 256;
constexpr size_t kBufStride = CAR_MAX_BYTES;

__global__ void seq_bump_kernel(int* seq_d) { *seq_d = *seq_d + 1; }

__global__ void copy_kernel(const char* __restrict__ in, char* __restrict__ my_base, const int* __restrict__ seq_d,
                            size_t nbytes) {
    const int parity = (*seq_d) & 1;
    char* dst = my_base + kBufBase + (size_t)parity * kBufStride;
    const size_t vec = nbytes / 16;
    const uint4* in4 = (const uint4*)in;
    uint4* dst4 = (uint4*)dst;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < vec; v += (size_t)gridDim.x * blockDim.x)
        dst4[v] = in4[v];
    for (size_t b = vec * 16 + (size_t)blockIdx.x * blockDim.x + threadIdx.x; b < nbytes;
         b += (size_t)gridDim.x * blockDim.x)
        dst[b] = in[b];
}

__global__ void flag_kernel(char* __restrict__ my_base, const int* __restrict__ seq_d) {
    const int seq = *seq_d;
    const int parity = seq & 1;
    __threadfence_system();
    *(volatile int*)(my_base + (parity ? kFlagOff1 : kFlagOff0)) = seq;
}

__global__ void wait_kernel(const char* __restrict__ peer_base, const int* __restrict__ seq_d) {
    const int seq = *seq_d;
    const int parity = seq & 1;
    volatile const int* pflag = (volatile const int*)(peer_base + (parity ? kFlagOff1 : kFlagOff0));
    while (*pflag < seq) {
    }
    __threadfence_system();
}

__global__ void reduce_bf16_kernel(const char* __restrict__ in, char* __restrict__ out,
                                   const char* __restrict__ peer_base, const int* __restrict__ seq_d, size_t numel) {
    const int parity = (*seq_d) & 1;
    const char* psrc = peer_base + kBufBase + (size_t)parity * kBufStride;
    const __nv_bfloat16* pin = (const __nv_bfloat16*)in;
    const __nv_bfloat16* pp = (const __nv_bfloat16*)psrc;
    __nv_bfloat16* pout = (__nv_bfloat16*)out;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += (size_t)gridDim.x * blockDim.x)
        pout[i] = __float2bfloat16(__bfloat162float(pin[i]) + __bfloat162float(pp[i]));
}

__global__ void reduce_fp16_kernel(const char* __restrict__ in, char* __restrict__ out,
                                   const char* __restrict__ peer_base, const int* __restrict__ seq_d, size_t numel) {
    const int parity = (*seq_d) & 1;
    const char* psrc = peer_base + kBufBase + (size_t)parity * kBufStride;
    const __half* pin = (const __half*)in;
    const __half* pp = (const __half*)psrc;
    __half* pout = (__half*)out;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += (size_t)gridDim.x * blockDim.x)
        pout[i] = __float2half(__half2float(pin[i]) + __half2float(pp[i]));
}

__global__ void reduce_fp32_kernel(const char* __restrict__ in, char* __restrict__ out,
                                   const char* __restrict__ peer_base, const int* __restrict__ seq_d, size_t numel) {
    const int parity = (*seq_d) & 1;
    const char* psrc = peer_base + kBufBase + (size_t)parity * kBufStride;
    const float* pin = (const float*)in;
    const float* pp = (const float*)psrc;
    float* pout = (float*)out;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += (size_t)gridDim.x * blockDim.x)
        pout[i] = pin[i] + pp[i];
}

}  // namespace

void car_initialize(torch::Tensor buf, torch::Tensor seq)
{
    TORCH_CHECK(!g_state.ready, "custom AR already initialized");
    TORCH_CHECK(buf.is_cuda() && buf.dtype() == torch::kUInt8, "uint8 CUDA buffer required");
    g_state.buf_ref = buf;
    g_state.seq_ref = seq;
    g_state.my_base = (char*)buf.data_ptr();
    g_state.seq_d = (int*)seq.data_ptr();
    g_state.ready = true;
}

void car_connect(int64_t peer_ptr)
{
    TORCH_CHECK(g_state.ready, "call car_initialize first");
    g_state.peer_base = (char*)peer_ptr;
    g_state.world = 2;
}

bool car_ready(int64_t world) { return g_state.ready && g_state.world == world && g_state.peer_base != nullptr; }

int64_t car_max_bytes() { return CAR_MAX_BYTES; }

torch::Tensor car_all_reduce(torch::Tensor t)
{
    TORCH_CHECK(g_state.ready && g_state.peer_base, "custom AR not connected");
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "contiguous CUDA tensor required");
    const size_t nbytes = (size_t)t.numel() * t.element_size();
    TORCH_CHECK(nbytes <= CAR_MAX_BYTES, "message exceeds custom AR buffer");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int cblocks = (int)((nbytes / 16 + CAR_THREADS - 1) / CAR_THREADS) + 1;
    const int rblocks = (int)((t.numel() + CAR_THREADS - 1) / CAR_THREADS);
    seq_bump_kernel<<<1, 1, 0, stream.stream()>>>(g_state.seq_d);
    copy_kernel<<<cblocks, CAR_THREADS, 0, stream.stream()>>>((const char*)t.data_ptr(), g_state.my_base, g_state.seq_d,
                                                              nbytes);
    flag_kernel<<<1, 1, 0, stream.stream()>>>(g_state.my_base, g_state.seq_d);
    wait_kernel<<<1, 1, 0, stream.stream()>>>(g_state.peer_base, g_state.seq_d);
    if (t.scalar_type() == at::kBFloat16) {
        reduce_bf16_kernel<<<rblocks, CAR_THREADS, 0, stream.stream()>>>((const char*)t.data_ptr(),
                                                                         (char*)t.data_ptr(), g_state.peer_base,
                                                                         g_state.seq_d, (size_t)t.numel());
    } else if (t.scalar_type() == at::kHalf) {
        reduce_fp16_kernel<<<rblocks, CAR_THREADS, 0, stream.stream()>>>((const char*)t.data_ptr(),
                                                                         (char*)t.data_ptr(), g_state.peer_base,
                                                                         g_state.seq_d, (size_t)t.numel());
    } else if (t.scalar_type() == at::kFloat) {
        reduce_fp32_kernel<<<rblocks, CAR_THREADS, 0, stream.stream()>>>((const char*)t.data_ptr(),
                                                                         (char*)t.data_ptr(), g_state.peer_base,
                                                                         g_state.seq_d, (size_t)t.numel());
    } else {
        TORCH_CHECK(false, "unsupported dtype for custom AR");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("initialize", &car_initialize, "bind python-owned IPC buffers");
    m.def("connect", &car_connect, "map peer buffer (data pointer from CUDA IPC storage)");
    m.def("ready", &car_ready, "is the custom path usable");
    m.def("max_bytes", &car_max_bytes, "max message size");
    m.def("all_reduce", &car_all_reduce, "one-shot P2P all-reduce (in place)");
}
