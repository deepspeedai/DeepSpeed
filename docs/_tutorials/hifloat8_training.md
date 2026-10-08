---
title: "HiFloat8 training on Ascend NPU"
---

DeepSpeed selects ordinary Linear and AutoEP GroupedExperts modules through
`hifloat8.module_name_patterns`. The `torchao_npu` backend owns quantization,
training GEMMs, ordinary module conversion, and parameter compatibility checks.
DeepSpeed retains initialization, module selection, routing/padding, EP
communication, and distributed lifecycle checks.

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "hifloat8": {
    "enabled": true,
    "backend": "torchao_npu",
    "module_name_patterns": ["*.experts", "*.shared_experts.*_proj"],
    "min_numel": 65536,
    "config": {
      "input_dst_type_max": 15.0,
      "weight_dst_type_max": 15.0,
      "grad_dst_type_max": 224.0,
      "scale_policy": "pertensor",
      "compute_dtype": "bfloat16"
    }
  }
}
```

The optional `config` object is parsed by `torchao_npu.hifloat8.HiFloat8Config`. One
immutable policy is shared by converted Linear modules and selected grouped
experts. Omitting it preserves DeepSpeed's existing 15/15/224 recipe.
`compute_dtype` accepts `null`, `"bfloat16"` or `"float16"` and controls the
high-precision working operands and GEMM output; it does not select the hardware
accumulator dtype. Parameter and optimizer precision remain unchanged.
Only `"pertensor"` scaling is currently supported. Custom policies are rejected
for the legacy `torch_npu` helper backend rather than silently ignored.

Install torchao_npu with PyTorch/torch_npu 2.11 and torchao 0.18.x. Native
capability probes run before model conversion. Unmatched selection patterns,
wrong expected module counts, unsupported policies or unavailable native kernels
fail before conversion. Existing restrictions remain: BF16 NPU inputs,
ZeRO stages 0–2, AutoEP expert containers; ZeRO-3, AutoTP, Pipeline parallelism
and custom model-parallel units are rejected. Single-node validation does not
establish multi-node compatibility.

The checkpoint stores the original high-precision parameters with unchanged
names. Configure the same HiFloat8 policy when reconstructing the engine to
resume; the policy is not a quantized weight checkpoint format.
