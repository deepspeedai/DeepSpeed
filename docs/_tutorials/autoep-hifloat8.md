---
title: "AutoEP HiFloat8 expert training on NPU (experimental)"
---

This opt-in integration extends the Dense HiFloat8 bridge with routed experts.
Use a torch-npu build exposing `hifloat8` and the training helpers
`hifloat8_grouped_mm` / `assert_hifloat8_grouped_training_available`.
A version string alone does not establish native kernel support. Quantization,
grouped forward/dX and split-K dW must be supported by the installed CANN/SoC.

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "expert_parallel": {
    "enabled": true,
    "autoep_size": 4,
    "preset_model": "qwen3_5_moe",
    "comm_backend": "comm"
  },
  "hifloat8": {
    "enabled": true,
    "module_name_patterns": [
      "*.experts",
      "*.shared_experts.gate_proj",
      "*.shared_experts.up_proj",
      "*.shared_experts.down_proj"
    ],
    "min_numel": 65536
  }
}
```

Patterns match the model **after AutoEP replacement**. For `GroupedExperts`,
`min_numel` applies to each expert projection matrix, excluding the expert-count
dimension. Router and shared-expert gates are deliberately not selected.
Expert parameters, trainability and checkpoint keys are unchanged by HiFloat8
selection; AutoEP's normal repacking and expert sharding still apply.

The Qwen3.5 preset covers both the text backbone and the language-model path
of its ConditionalGeneration wrapper. Vision and attention are not replaced.
Sorted top-k, FP32 softmax, BF16 routing weights and raw router-logit capture
follow the Qwen3.5 router; AutoEP retains its existing FP32 weighted combination.
ZeRO 0–2 are the initial scope; HiFloat8 rejects PipelineModule, AutoTP, custom
model-parallel units, ZeRO-3 and explicit `deepspeed.moe.layer.MoE` modules.

On NPU, grouped BF16 uses `npu_grouped_matmul` with explicit group0 dX and
group2 dW autograd; HiFloat8 uses the matching torch-npu autograd helper.
Token permutation currently uses the CPU reference
indices, which introduces synchronization and transfer overhead. Performance
must be measured; no speedup is promised. Communication remains at the model
dtype. HIF8 never silently falls back to BF16 or INT8.

On Ascend950PR with CANN 9.1, use a hardware-generated CANN v2 rank table via
`RANK_TABLE_FILE`. The root-info initialization route can fail for singleton
expert data-parallel groups even when the four-rank all-reduce works. This is
an environment setting, not a reason to change collective precision or routing.

This integration is experimental until EP>1, routing precision, actual expert
updates, empty ranks, checkpoint resume and native profiler evidence have been
validated on the exact configuration. Turning `hifloat8.enabled` off returns
the AutoEP BF16 path. Existing Dense training remains available separately.
