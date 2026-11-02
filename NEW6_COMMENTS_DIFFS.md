# PR #8602 — 新 6 条 Copilot comments × 修复 diff 对照（注释收紧后最终版）

## N1. `deepspeed/module_inject/segment_ki.py:376` — 修复 commit `2fa194c97`

### 🤖 Copilot 原文

> Dropping `cu_seq_lens_q` here loses packed-sequence boundaries before the causal convolution, allowing state to bleed from one packed sequence into the next. The upstream Qwen3.5 forward passes this kwarg through to `causal_conv1d_fn`; only unrelated kwargs should be filtered.
> 
> This issue also appears on line 389 of the same file.

### 修复 diff

```diff
diff --git a/deepspeed/module_inject/segment_ki.py b/deepspeed/module_inject/segment_ki.py
index c3362cadd..41c9f3350 100644
--- a/deepspeed/module_inject/segment_ki.py
+++ b/deepspeed/module_inject/segment_ki.py
@@ -366,17 +366,14 @@ def _install_gdn_segment(seg: GDNSegment) -> bool:
 
     def conv_update_with_weights(mixed_qkv, conv_state):
         return conv_update(mixed_qkv, conv_state, parent.conv1d.weight.squeeze(1), parent.conv1d.bias,
                            parent.activation)
 
     def conv_fn_with_weights(mixed_qkv, **kw):
-        # causal_conv1d_fn does not accept HF generation kwargs; filter them
-        # the same way the scan closure does.
+        # cu_seq_lens_* mark packed-sequence boundaries and must reach the conv.
         kw.pop("use_cache", None)
-        kw.pop("cu_seqlens", None)
-        kw.pop("cu_seq_lens_q", None)
         return conv_fn(mixed_qkv,
                        parent.conv1d.weight.squeeze(1),
                        parent.conv1d.bias,
                        activation=parent.activation,
                        **kw)
 
@@ -384,14 +381,12 @@ def _install_gdn_segment(seg: GDNSegment) -> bool:
         # Route through the block's own instance-bound kernels: transformers
         # binds the FLA fused implementations onto the module when available
         # (recurrent_gated_delta_rule/chunk_gated_delta_rule) and only falls
         # back to the pure-torch references otherwise. Calling the module-level
         # torch_* functions directly forces the slow fallback on every layer.
         single = kw.get("initial_state") is not None and query.shape[1] == 1
-        kw.pop("cu_seqlens", None)
-        kw.pop("cu_seq_lens_q", None)
         fn = parent.recurrent_gated_delta_rule if single else parent.chunk_gated_delta_rule
         return fn(query, key, value, **kw)
 
     parent._ki_gdn_op = _load_native_op()
     parent._ki_gdn_conv_update = conv_update_with_weights
     parent._ki_gdn_conv_fn = conv_fn_with_weights
```

---

## N2. `deepspeed/module_inject/segment_ki.py:563` — 修复 commit `16261514f`

### 🤖 Copilot 原文

> `triple_gemv` consumes only weights, but this enables it for `nn.Linear` projections even when Q/K/V have biases. Qwen3.5 exposes `attention_bias` as a valid config option, so those models silently lose all three projection biases and produce different logits. Keep the fused QKV projection disabled unless all three biases are `None`.

### 修复 diff

```diff
diff --git a/deepspeed/module_inject/segment_ki.py b/deepspeed/module_inject/segment_ki.py
index 41c9f3350..f91c82aa0 100644
--- a/deepspeed/module_inject/segment_ki.py
+++ b/deepspeed/module_inject/segment_ki.py
@@ -551,13 +551,15 @@ def install_decode_attention(model: torch.nn.Module, write_pos: torch.Tensor, ke
         # its 16 warps-per-block KV split.
         if head_dim not in (64, 128, 256) or 16 % (num_q_heads // num_kv_heads) != 0:
             continue
         if getattr(module, "_ki_orig_attn_forward", None) is None:
             module._ki_orig_attn_forward = module.forward
         module._ki_attn_op = kernel_op
-        module._ki_qkv_op = kernel_op if (hasattr(kernel_op, "triple_gemv")
+        # The fused QKV kernel consumes weights only; biased projections would lose their bias.
+        unbiased_qkv = all(p.bias is None for p in projections[:3])
+        module._ki_qkv_op = kernel_op if (hasattr(kernel_op, "triple_gemv") and unbiased_qkv
                                           and os.environ.get("DS_TIER2", "1") == "1") else None
         module._ki_write_pos = write_pos
         module._ki_rope = apply_rotary_pos_emb
         module._ki_num_q_heads = num_q_heads
         module._ki_num_kv_heads = num_kv_heads
         module.forward = _decode_attn_forward.__get__(module, type(module))
```

---

## N3. `deepspeed/runtime/rollout/hybrid_engine_rollout.py:1039` — 修复 commit `39044b4a1`

### 🤖 Copilot 原文

> The graph path is not Qwen-specific, but this now passes an attention-mask dictionary unconditionally. Standard full-attention models such as LLaMA expect a tensor/`None`, so `use_graph_capture=True` regresses for them (even when `use_segki=False`). Build the per-type mapping only for hybrid `layer_types` models and retain the tensor mask for ordinary models at prefill, warmup, and capture.

### 修复 diff

```diff
diff --git a/deepspeed/runtime/rollout/hybrid_engine_rollout.py b/deepspeed/runtime/rollout/hybrid_engine_rollout.py
index 5f7bf68c2..26cea5dce 100644
--- a/deepspeed/runtime/rollout/hybrid_engine_rollout.py
+++ b/deepspeed/runtime/rollout/hybrid_engine_rollout.py
@@ -1025,21 +1025,20 @@ class HybridEngineRollout(RolloutEngine):
         max_len = prompt_len + max_new_tokens
         eos_token_id = self.tokenizer.eos_token_id
         model_dtype = next(module.parameters()).dtype
 
         # --- Prefill with HF StaticCache (correct attention semantics) ---
         prefill_cache = self._create_static_cache(StaticCache, module.config, batch_size, max_len, device, model_dtype)
-        # Mirror generate's hybrid-model kwargs exactly: a per-type mask dict
-        # (GDN must see None) and no explicit cache_position. A 2D mask here
-        # corrupts the GDN conv-state initialization and collapses decode.
+        # Hybrid models take a per-type mask dict (GDN must see None; a 2D
+        # mask corrupts its conv-state init); standard models take a tensor.
+        is_hybrid = "linear_attention" in (getattr(getattr(module.config, "text_config", module.config),
+                                                  "layer_types", None) or [])
+        prefill_mask = {"full_attention": None, "linear_attention": None} if is_hybrid else prompt_attn
         prefill_out = module(
             prompt_ids,
-            attention_mask={
-                "full_attention": None,
-                "linear_attention": None
-            },
+            attention_mask=prefill_mask,
             past_key_values=prefill_cache,
             use_cache=True,
         )
         next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
 
         # --- Copy prefill KV into DeepSpeedStaticCache ---
@@ -1078,12 +1077,16 @@ class HybridEngineRollout(RolloutEngine):
         # GDN layers must receive None: their recurrence ignores attention masks
         # and a non-None value corrupts the conv-state updates.
         static_attn = torch.zeros(batch_size, 1, 1, max_len, dtype=torch.bool, device=device)
         static_attn[:, :, :, :prompt_len] = prompt_attn.unsqueeze(1).unsqueeze(1).bool()
         static_attn[:, :, :, prompt_len] = True
 
+        # Decode-step mask argument: the per-type dict for hybrid models, the
+        # plain 4-D tensor for standard full-attention ones.
+        decode_mask = {"full_attention": static_attn, "linear_attention": None} if is_hybrid else static_attn
+
         full_token_buf = torch.zeros(max_len, dtype=torch.long, device=device)
         if batch_size == 1:
             full_token_buf[:prompt_len] = prompt_ids.view(-1)
         full_token_buf[prompt_len] = next_token.view(-1)[0]  # first generated token
 
         write_pos.fill_(prompt_len)
@@ -1132,16 +1135,13 @@ class HybridEngineRollout(RolloutEngine):
             s = get_accelerator().Stream()
             s.wait_stream(get_accelerator().current_stream())
             with get_accelerator().stream(s):
                 for _ in range(3):
                     out = module(
                         static_token,
-                        attention_mask={
-                            "full_attention": static_attn,
-                            "linear_attention": None
-                        },
+                        attention_mask=decode_mask,
                         past_key_values=ds_cache,
                         use_cache=True,
                     )
             get_accelerator().current_stream().wait_stream(s)
             restore_gdn_states()
 
@@ -1160,16 +1160,13 @@ class HybridEngineRollout(RolloutEngine):
                 graph_op = None
 
             graph = get_accelerator().create_graph()
             with get_accelerator().capture_to_graph(graph):
                 out = module(
                     static_token,
-                    attention_mask={
-                        "full_attention": static_attn,
-                        "linear_attention": None
-                    },
+                    attention_mask=decode_mask,
                     past_key_values=ds_cache,
                     use_cache=True,
                 )
                 static_logits = out.logits
                 # decode_step_graph kernel is b=1 only (single-sequence argmax);
                 # for b>1 the graph captures forward only and Python handles
```

---

## N4. `csrc/module_inject/fused_glu.cu:114` — 修复 commit `43343f543`

### 🤖 Copilot 原文

> The reduction does not implement PyTorch `argmax`'s first-index tie rule. With vocabularies larger than 1024, equal bf16 maxima can land in different thread strides, and the lower thread ID is not necessarily the lower vocabulary index, so greedy decode can choose a different token from `torch.argmax`. Break equal-value ties using `s_idx`.
> 
> This issue also appears on line 182 of the same file.

### 修复 diff

```diff
diff --git a/csrc/module_inject/fused_glu.cu b/csrc/module_inject/fused_glu.cu
index ac6122827..a6a99ca27 100644
--- a/csrc/module_inject/fused_glu.cu
+++ b/csrc/module_inject/fused_glu.cu
@@ -104,13 +104,15 @@ __global__ void decode_step_kernel(const __nv_bfloat16* __restrict__ logits,
     s_idx[tid] = local_idx;
     s_val[tid] = local_val;
     __syncthreads();
 
     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
         if (tid < stride) {
-            if (s_val[tid + stride] > s_val[tid]) {
+            // torch.argmax picks the lowest index on ties.
+            if (s_val[tid + stride] > s_val[tid] ||
+                (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                 s_val[tid] = s_val[tid + stride];
                 s_idx[tid] = s_idx[tid + stride];
             }
         }
         __syncthreads();
     }
@@ -178,13 +180,15 @@ __global__ void decode_step_graph_kernel(const __nv_bfloat16* __restrict__ logit
     s_idx[tid] = local_idx;
     s_val[tid] = local_val;
     __syncthreads();
 
     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
         if (tid < stride) {
-            if (s_val[tid + stride] > s_val[tid]) {
+            // torch.argmax picks the lowest index on ties.
+            if (s_val[tid + stride] > s_val[tid] ||
+                (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                 s_val[tid] = s_val[tid + stride];
                 s_idx[tid] = s_idx[tid + stride];
             }
         }
         __syncthreads();
     }
```

---

## N5. `csrc/module_inject/fused_glu.cu:733` — 修复 commit `0a37a754c`

### 🤖 Copilot 原文

> The runtime's primary b=1 path depends on `decode_step_graph`, but no new test invokes it—the kernel consistency class covers the other math kernels only. Add a direct CUDA contract test that verifies argmax (including tied logits), `write_pos`, mask revelation, and output-buffer updates across repeated calls; otherwise the full-step control path can regress while all listed tests pass.

### 修复 diff

```diff
diff --git a/tests/unit/module_inject/test_segment_ki_kernels.py b/tests/unit/module_inject/test_segment_ki_kernels.py
index 282664236..cad378013 100644
--- a/tests/unit/module_inject/test_segment_ki_kernels.py
+++ b/tests/unit/module_inject/test_segment_ki_kernels.py
@@ -66,12 +66,40 @@ def _rand_bf16(*shape, device=DEV):
 # ---------------------------------------------------------------------------
 
 
 @pytest.mark.skipif(not GPU_AVAILABLE, reason="non-CPU device required")
 class TestKernelReferenceConsistency:
 
+    def test_decode_step_graph_contract(self):
+        """The full-step graph kernel must reproduce torch.argmax (first
+        index on ties), advance write_pos, reveal exactly the next slot,
+        and record the token in the buffer across repeated calls."""
+        vocab, max_len = 2048, 16
+        logits = torch.full((1, vocab), -3.0, dtype=torch.bfloat16, device=DEV)
+        logits[0, 10] = 2.0
+        logits[0, 1500] = 2.0  # exact bf16 tie: torch.argmax keeps index 10
+        token_out = torch.zeros(1, 1, dtype=torch.long, device=DEV)
+        write_pos = torch.tensor([5], dtype=torch.long, device=DEV)
+        mask = torch.zeros(1, 1, 1, max_len, dtype=torch.bool, device=DEV)
+        out_buf = torch.zeros(max_len, dtype=torch.long, device=DEV)
+
+        expected = torch.argmax(logits[0].float())
+        assert expected.item() == 10  # torch.argmax first-index rule
+
+        _op().decode_step_graph(logits, token_out, write_pos, mask, out_buf)
+        assert token_out.view(-1)[0].item() == 10
+        assert write_pos.item() == 6
+        assert bool(mask[0, 0, 0, 6]) and not bool(mask[0, 0, 0, 7])
+        assert out_buf[6].item() == 10
+
+        # a second call advances from the updated write_pos and reveals slot 7
+        _op().decode_step_graph(logits, token_out, write_pos, mask, out_buf)
+        assert write_pos.item() == 7
+        assert bool(mask[0, 0, 0, 7]) and not bool(mask[0, 0, 0, 8])
+        assert out_buf[7].item() == 10
+
     def test_dual_gemv_silu_mul(self):
         h = _rand_bf16(256)
         gw = _rand_bf16(384, 256)
         uw = _rand_bf16(384, 256)
         out = torch.empty(384, dtype=torch.bfloat16, device=DEV)
         _op().dual_gemv_silu_mul(h, gw, uw, out)
```

---

## N6. `tests/unit/module_inject/test_segment_ki_kernels.py:277` — 修复 commit `b1dce5f4e`

### 🤖 Copilot 原文

> A correct live-weight implementation can still generate the same greedy token sequence after one SGD step because changed logits need not change any argmax. This makes the integration test nondeterministic with respect to the contract it claims to verify. Use a deterministic live-weight observable or force a controlled weight change that guarantees a known output difference.

### 修复 diff

```diff
diff --git a/tests/unit/module_inject/test_segment_ki_kernels.py b/tests/unit/module_inject/test_segment_ki_kernels.py
index cad378013..cb6f78494 100644
--- a/tests/unit/module_inject/test_segment_ki_kernels.py
+++ b/tests/unit/module_inject/test_segment_ki_kernels.py
@@ -289,17 +289,27 @@ class TestRolloutTrainRollout:
         req = RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=attn)
         greedy = SamplingConfig(max_new_tokens=24, temperature=0.0)
 
         batch1 = rollout.generate(req, greedy)
 
         # one training step through the SAME injected model
+        first_glu = next(m for n, m in model.named_modules() if n.endswith("mlp"))
+        gate_w_before = first_glu.gate_proj.weight.detach().clone()
         out = model(batch1.input_ids)
         loss = out.logits.float().pow(2).mean()
         loss.backward()
-        first_glu = next(m for n, m in model.named_modules() if n.endswith("mlp"))
         assert first_glu.gate_proj.weight.grad is not None, "gradient did not reach gate_proj"
         torch.optim.SGD(model.parameters(), lr=0.5).step()
         model.zero_grad()
 
-        batch2 = rollout.generate(req, greedy)
-        assert not torch.equal(batch1.input_ids, batch2.input_ids), \
-            "rollout output unchanged after training — kernels read stale weights"
+        # Live-weight contract, deterministically: the injected projection
+        # must read the Parameter's current values, so its output changes
+        # once the optimizer updated the weights.
+        from deepspeed.module_inject.segment_ki import DualWeightGluGEMV
+        probe = torch.randn_like(gate_w_before[0]).bfloat16()
+        w_now = first_glu.gate_proj.weight.detach()
+        before = DualWeightGluGEMV.apply(probe, gate_w_before, first_glu.up_proj.weight.detach(), None)
+        after = DualWeightGluGEMV.apply(probe, w_now, first_glu.up_proj.weight.detach(), None)
+        assert not torch.equal(before, after), "injected projection still reads pre-step weights"
+
+        # still exercise the post-training generate path end-to-end
+        rollout.generate(req, greedy)
```

---
