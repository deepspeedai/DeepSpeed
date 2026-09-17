// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

// C++ decode loop: runs the full generation loop in a single C++ call.
// The CUDA graph is accepted as a py::object; its replay() method is invoked
// from C++ via pybind11 (bypassing the Python for-loop while reusing the
// same graph object that was captured on the Python side).
//
// Per-step overhead: graph.attr("replay")() ≈ 2-5μs (pybind11 dispatch)
// vs Python graph.replay() ≈ 10-15μs + loop overhead ≈ 30μs.
// The fused step-update kernel launch is pure C++ (zero Python).

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void ds_loop_step_kernel(const __nv_bfloat16* __restrict__ logits,
                                    int64_t* __restrict__ token_out,
                                    int64_t* __restrict__ write_pos,
                                    bool* __restrict__ mask,
                                    int64_t* __restrict__ out_buf,
                                    int step,
                                    int vocab_size,
                                    int max_len)
{
    __shared__ int s_idx[1024];
    __shared__ float s_val[1024];

    int tid = threadIdx.x;
    int local_idx = 0;
    float local_val = -INFINITY;

    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = __bfloat162float(logits[v]);
        if (val > local_val) {
            local_val = val;
            local_idx = v;
        }
    }
    s_idx[tid] = local_idx;
    s_val[tid] = local_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t best = (int64_t)s_idx[0];
        token_out[0] = best;
        out_buf[step] = best;
        int64_t new_pos = write_pos[0] + 1;
        write_pos[0] = new_pos;
        if (new_pos + 1 < max_len) { mask[new_pos + 1] = true; }
    }
}

int64_t ds_decode_loop(py::object graph,
                       at::Tensor logits,
                       at::Tensor token_out,
                       at::Tensor write_pos,
                       at::Tensor mask,
                       at::Tensor out_buf,
                       int64_t max_steps,
                       int64_t eos_token_id,
                       int64_t pad_token_id,
                       int64_t eos_check_every)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::ScalarType::BFloat16,
                "logits must be CUDA bf16");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int vocab = (int)logits.numel();
    int max_len = (int)mask.size(-1);

    const __nv_bfloat16* log_ptr =
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>());
    int64_t* tok_ptr = token_out.data_ptr<int64_t>();
    int64_t* wp_ptr = write_pos.data_ptr<int64_t>();
    bool* mask_ptr = mask.data_ptr<bool>();
    int64_t* buf_ptr = out_buf.data_ptr<int64_t>();

    // Cache the replay method once (avoids repeated attribute lookup)
    py::object replay_fn = graph.attr("replay");

    int64_t steps_done = 0;

    for (int64_t step = 1; step < max_steps; step++) {
        // 1) Replay the captured forward pass via pybind11 method call
        replay_fn();

        // 2) Fused step-update kernel (pure C++, zero Python)
        ds_loop_step_kernel<<<1, 1024, 0, stream>>>(
            log_ptr, tok_ptr, wp_ptr, mask_ptr, buf_ptr, (int)step, vocab, max_len);

        steps_done = step + 1;

        // 3) Periodic EOS check (D2H sync, amortized over N steps)
        if (eos_token_id >= 0 && step % eos_check_every == 0) {
            int64_t token;
            cudaMemcpyAsync(&token, tok_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (token == eos_token_id) {
                for (int64_t i = step + 1; i < max_steps; i++) { buf_ptr[i] = pad_token_id; }
                break;
            }
        }
    }

    return steps_done;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decode_loop",
          &ds_decode_loop,
          "C++ decode loop: graph replay + step update, zero Python per step",
          py::arg("graph"),
          py::arg("logits"),
          py::arg("token_out"),
          py::arg("write_pos"),
          py::arg("mask"),
          py::arg("out_buf"),
          py::arg("max_steps"),
          py::arg("eos_token_id"),
          py::arg("pad_token_id"),
          py::arg("eos_check_every"));
}
