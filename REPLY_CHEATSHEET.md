# PR #8602 — Copilot comments 回复清单（rebase 后最终 sha）

> 用法：在对应 comment 下回复一行，然后 Resolve thread。
> 锚定已删文件（C1/C14/C18）的 thread push 后会自动 Outdated，也可直接 resolve。

| # | Comment 位置 | 回复文本 |
|---|---|---|
| C1 | decode_loop.cu:16 includes | Fixed in 5ae7bd73e (file later removed entirely in 3459bda4c). |
| C2 | segment_ki.py:308 gdn_gates autograd | Fixed in e5555393f — grad-enabled forwards now route through the composite ops; the kernel serves the no-grad decode path only. |
| C3 | segment_ki.py:327 cu_seq_lens or | Fixed in 6946a917c — uses dict.get's default instead of truth-testing the tensor. |
| C4 | segment_ki.py:492 decode attn mask | Fixed in c3c8c1fa9 — rollout flags padded prompts once per generate; the injected forward rides SDPA for them. Verified on GPU: flag set, decode falls back. |
| C5 | segment_ki.py:569 fused_add_norm autograd | Fixed in e5555393f (same commit as the gdn_gates gate) — grad-enabled calls take the original forward. |
| C6 | hybrid_engine_rollout.py:607 GDN slot bind | Addressed in 23405978f — batched GDN state (scatter on admission, compact on retirement) is real future work; for now _validate_continuous_inputs rejects hybrid models with a clear error instead of crashing mid-decode. |
| C7 | hybrid_engine_rollout.py:784 prefill mask | Fixed in c2f9b5c59 — padded prompts take module.generate, which owns the full mask semantics. GPU-verified: left-padded prompts generate correct continuations; graph path unchanged for unpadded prompts (0 perf impact). |
| C8 | hybrid_engine_rollout.py:935 restore condition | Fixed in 1121aaa01 — restore is now unconditional after capture. |
| C9 | hybrid_engine_rollout.py:970 loop selector | Fixed in 3b174956d — a step_kernel_mounted flag records actual mounting (b=1, CUDA bf16) and the loop selector branches on it. |
| C10 | hybrid_engine_rollout.py:975 EOS pad offset | Fixed in 229b5faba — pads from prompt_len + step + 2 (buffer-absolute, after the EOS slot). |
| C11 | segment_ki.py:213 dim()==2 | Fixed in 11d624124 — any single-hidden-vector input (incl. [1,1,hidden]) dispatches to the kernel. |
| C12 | segment_ki.py:411 op load fallback | Fixed in 623905238 (+51a791ab8) — load is best-effort keyed on builder registration; non-CUDA backends and unusable JIT toolchains fall back to composite. |
| C13 | segment_ki.py:453 fp16 decode attn | Fixed in 4f793163d — guard checks dtype/contiguity; fp16 falls back to the original attention. Also gated gdn_gates on bf16 in 77121e422 (same crash family, found during GPU verification). |
| C14 | decode_loop.py:11 builder outside op_builder | Fixed in 51a791ab8 (moved into op_builder/), then the layer was removed entirely in 3459bda4c — see C18. |
| C15 | fused_glu.py:10 builder outside op_builder | Fixed in 51a791ab8 — builder now lives in op_builder/, DS_BUILD_FUSED_GLU works at install time. |
| C16 | test gating | Fixed in b9872afd6 — tests gate on the accelerator's builder registration (NotImplemented backends skip). |
| C17 | docs Gemma | Fixed in efba6c815 — family list now says "with a SiLU activation"; Gemma named as passed through. |
| C18 | decode_loop.cu:84 no test | Addressed by removing the layer (3459bda4c): with fused_glu loaded, b=1 rides the full-step graph and b>1 uses graph+Python, so the standalone loop was unreachable (and crashed on fp16). It duplicated the step kernel and had no coverage; decode falls back to the Python loop. GPU-verified no perf change (67 tok/s before/after on Qwen3.5-4B, and +11% for segki at 2K context). |
