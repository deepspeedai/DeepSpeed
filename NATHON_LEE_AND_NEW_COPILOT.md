# PR #8602 — nathon-lee 的评论 + 新一批 Copilot comments

## nathon-lee（顶楼评论，2026-10-06）

> Hi @delock, thanks for submitting this PR. I have left a few comments and questions.

**⚠️ 他的 inline comments 在 API 中不存在**——大概率写在未提交（pending）的 review 里，对他可见、对你不可见。需要他点 "Submit review" 才会出现。

---

## 新一批 Copilot inline comments（2026-10-07，共 6 条）

### 1. `deepspeed/module_inject/segment_ki.py:376`

```python
     371 |     def conv_fn_with_weights(mixed_qkv, **kw):
     372 |         # causal_conv1d_fn does not accept HF generation kwargs; filter them
     373 |         # the same way the scan closure does.
     374 |         kw.pop("use_cache", None)
     375 |         kw.pop("cu_seqlens", None)
>>>  376 |         kw.pop("cu_seq_lens_q", None)
     377 |         return conv_fn(mixed_qkv,
     378 |                        parent.conv1d.weight.squeeze(1),
     379 |                        parent.conv1d.bias,
```

**Copilot:**

Dropping `cu_seq_lens_q` here loses packed-sequence boundaries before the causal convolution, allowing state to bleed from one packed sequence into the next. The upstream Qwen3.5 forward passes this kwarg through to `causal_conv1d_fn`; only unrelated kwargs should be filtered.

This issue also appears on line 389 of the same file.

---

### 2. `deepspeed/module_inject/segment_ki.py:563`

```python
     558 |             continue
     559 |         if getattr(module, "_ki_orig_attn_forward", None) is None:
     560 |             module._ki_orig_attn_forward = module.forward
     561 |         module._ki_attn_op = kernel_op
     562 |         module._ki_qkv_op = kernel_op if (hasattr(kernel_op, "triple_gemv")
>>>  563 |                                           and os.environ.get("DS_TIER2", "1") == "1") else None
     564 |         module._ki_write_pos = write_pos
     565 |         module._ki_rope = apply_rotary_pos_emb
     566 |         module._ki_num_q_heads = num_q_heads
```

**Copilot:**

`triple_gemv` consumes only weights, but this enables it for `nn.Linear` projections even when Q/K/V have biases. Qwen3.5 exposes `attention_bias` as a valid config option, so those models silently lose all three projection biases and produce different logits. Keep the fused QKV projection disabled unless all three biases are `None`.

---

### 3. `deepspeed/runtime/rollout/hybrid_engine_rollout.py:1039`

```python
    1034 |         prefill_out = module(
    1035 |             prompt_ids,
    1036 |             attention_mask={
    1037 |                 "full_attention": None,
    1038 |                 "linear_attention": None
>>> 1039 |             },
    1040 |             past_key_values=prefill_cache,
    1041 |             use_cache=True,
    1042 |         )
```

**Copilot:**

The graph path is not Qwen-specific, but this now passes an attention-mask dictionary unconditionally. Standard full-attention models such as LLaMA expect a tensor/`None`, so `use_graph_capture=True` regresses for them (even when `use_segki=False`). Build the per-type mapping only for hybrid `layer_types` models and retain the tensor mask for ordinary models at prefill, warmup, and capture.

---

### 4. `csrc/module_inject/fused_glu.cu:114`

```cuda
     109 |         if (tid < stride) {
     110 |             if (s_val[tid + stride] > s_val[tid]) {
     111 |                 s_val[tid] = s_val[tid + stride];
     112 |                 s_idx[tid] = s_idx[tid + stride];
     113 |             }
>>>  114 |         }
     115 |         __syncthreads();
     116 |     }
     117 | 
```

**Copilot:**

The reduction does not implement PyTorch `argmax`'s first-index tie rule. With vocabularies larger than 1024, equal bf16 maxima can land in different thread strides, and the lower thread ID is not necessarily the lower vocabulary index, so greedy decode can choose a different token from `torch.argmax`. Break equal-value ties using `s_idx`.

This issue also appears on line 182 of the same file.

---

### 5. `csrc/module_inject/fused_glu.cu:733`

```cuda
     728 | 
     729 | PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
     730 | {
     731 |     m.def("decode_attn", &decode_attn, "b=1 decode attention with GQA, graph-compatible (CUDA)");
     732 |     m.def("decode_step", &decode_step, "fused decode step update (CUDA)");
>>>  733 |     m.def("decode_step_graph", &decode_step_graph, "graph-capturable decode step (CUDA)");
     734 |     m.def("gdn_gates", &gdn_gates, "fused GDN beta/g gating (CUDA)");
     735 |     m.def("quad_gemv", &quad_gemv, "b=1 GDN 4-weight input GEMV (qkv|z|b|a), no concat (CUDA)");
     736 |     m.def("dual_gemv_silu_mul",
```

**Copilot:**

The runtime's primary b=1 path depends on `decode_step_graph`, but no new test invokes it—the kernel consistency class covers the other math kernels only. Add a direct CUDA contract test that verifies argmax (including tied logits), `write_pos`, mask revelation, and output-buffer updates across repeated calls; otherwise the full-step control path can regress while all listed tests pass.

---

### 6. `tests/unit/module_inject/test_segment_ki_kernels.py:277`

```python
     272 |         torch.optim.SGD(model.parameters(), lr=0.5).step()
     273 |         model.zero_grad()
     274 | 
     275 |         batch2 = rollout.generate(req, greedy)
     276 |         assert not torch.equal(batch1.input_ids, batch2.input_ids), \
>>>  277 |             "rollout output unchanged after training — kernels read stale weights"
```

**Copilot:**

A correct live-weight implementation can still generate the same greedy token sequence after one SGD step because changed logits need not change any argmax. This makes the integration test nondeterministic with respect to the contract it claims to verify. Use a deterministic live-weight observable or force a controlled weight change that guarantees a known output difference.

---
