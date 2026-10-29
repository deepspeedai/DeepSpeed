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
#include <pybind11/pytypes.h>
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
    char* my_base = nullptr;   // cudaMalloc'd: [flag0|flag1|pad][buf0][buf1]
    char* peer_base = nullptr;
    int* seq_d = nullptr;
    char* own_base = nullptr;  // our cudaMalloc (never freed; process exit)
    int* own_seq = nullptr;
};
CarState g_state;

constexpr size_t kFlagOff0 = 0;
constexpr size_t kFlagOff1 = 4;
constexpr size_t kBufBase = 256;
constexpr size_t kBufStride = CAR_MAX_BYTES;
constexpr size_t kTotalBytes = kBufBase + 2 * kBufStride;

#define CUDA_CHECK(expr) TORCH_CHECK((expr) == cudaSuccess, "CUDA: ", cudaGetErrorString(cudaGetLastError()))

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

py::bytes car_create_buffers()
{
    // Pure CUDA IPC (vLLM-style): our own cudaMalloc + GetMemHandle, no
    // torch allocator wrappers in the exchange path.
    TORCH_CHECK(!g_state.ready, "custom AR already initialized");
    void* buf = nullptr;
    int* seq = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, kTotalBytes));
    CUDA_CHECK(cudaMalloc((void**)&seq, sizeof(int)));
    CUDA_CHECK(cudaMemset(buf, 0, kTotalBytes));
    CUDA_CHECK(cudaMemset(seq, 0, sizeof(int)));
    cudaIpcMemHandle_t h;
    CUDA_CHECK(cudaIpcGetMemHandle(&h, buf));
    g_state.my_base = (char*)buf;
    g_state.own_base = (char*)buf;
    g_state.own_seq = seq;
    g_state.seq_d = seq;
    g_state.ready = true;
    return py::bytes((const char*)&h, sizeof(h));  // raw handle, not utf-8
}

void car_connect(const std::string& handle_bytes)
{
    TORCH_CHECK(g_state.ready, "call car_create_buffers first");
    TORCH_CHECK(handle_bytes.size() >= sizeof(cudaIpcMemHandle_t), "short IPC handle");
    cudaIpcMemHandle_t h;
    memcpy(&h, handle_bytes.data(), sizeof(h));
    void* mapped = nullptr;
    // Open on THIS device's context so our kernels can dereference the
    // mapping over P2P; the lazy flag enables peer access on first touch.
    cudaError_t err = cudaIpcOpenMemHandle(&mapped, h, cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(err == cudaSuccess, "cudaIpcOpenMemHandle failed: ", cudaGetErrorString(err));
    g_state.peer_base = (char*)mapped;
    g_state.world = 2;
}

__global__ void handshake_kernel(char* __restrict__ my_base, const char* __restrict__ peer_base, int* __restrict__ seq_d,
                                 int* __restrict__ result)
{
    const int seq = *seq_d + 1;  // handshake rides the next sequence number
    *seq_d = seq;
    const int parity = seq & 1;
    char* my_buf = my_base + kBufBase + (size_t)parity * kBufStride;
    const char* peer_buf = peer_base + kBufBase + (size_t)parity * kBufStride;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(int*)(my_buf) = 0x5A5A5A5A;
        __threadfence_system();
        *(volatile int*)(my_base + (parity ? kFlagOff1 : kFlagOff0)) = seq;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile const int* pflag = (volatile const int*)(peer_base + (parity ? kFlagOff1 : kFlagOff0));
        long long spins = 0;
        while (*pflag < seq && spins < 2000000000LL) spins++;  // bounded (~seconds)
        if (*pflag < seq) {
            *result = -1;  // peer flag never became visible
            return;
        }
        __threadfence_system();
        *result = (*(volatile const int*)(peer_buf) == 0x5A5A5A5A) ? 1 : -2;  // flag ok, data?
    }
}

bool car_handshake()
{
    TORCH_CHECK(g_state.ready && g_state.peer_base, "custom AR not connected");
    auto result = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto stream = at::cuda::getCurrentCUDAStream();
    handshake_kernel<<<1, 32, 0, stream.stream()>>>(g_state.my_base, g_state.peer_base, g_state.seq_d,
                                                    (int*)result.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // both ranks must publish before either inspects — a plain NCCL-free
    // sync is impossible, so the caller synchronizes the stream first
    cudaStreamSynchronize(stream.stream());
    return result.cpu().item<int>() == 1;
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
    m.def("create_buffers", &car_create_buffers, "allocate IPC buffers, return handle bytes");
    m.def("connect", &car_connect, "map peer buffer (data pointer from CUDA IPC storage)");
    m.def("handshake", &car_handshake, "bounded probe: is the peer flag visible over this fabric");
    m.def("ready", &car_ready, "is the custom path usable");
    m.def("max_bytes", &car_max_bytes, "max message size");
    m.def("all_reduce", &car_all_reduce, "one-shot P2P all-reduce (in place)");
}
