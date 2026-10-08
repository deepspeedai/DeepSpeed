---
title: "HiFloat8 training on Ascend NPU"
---

DeepSpeed selects ordinary Linear and AutoEP GroupedExperts modules through
`hifloat8.module_name_patterns`. The `torchao_npu` backend is the sole HiFloat8 implementation and owns quantization,
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
Only `"pertensor"` scaling is currently supported. `torchao_npu` is now the
default backend. The former `backend="torch_npu"` helper is removed; migrate old
JSON configurations to `backend="torchao_npu"` and install the current AO source.
No custom torch_npu training-helper patch is needed.

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


## NPU grouped GEMM ownership

AutoEP's NPU expert path calls `torchao_npu.ops.npu.grouped_mm` for BF16/FP16
and `torchao_npu.hifloat8.hifloat8_grouped_mm` for HiFloat8. Native kernel calls,
GEMM autograd, empty-group gradients and operand-layout handling belong to AO.
DeepSpeed retains token packing/padding, activation semantics and EP communication.
There is no `_NPUGroupedMatmul` implementation in DeepSpeed.

Both APIs accept `trans_b=True` for packed `[E,N,K]` parameter storage. Their
backward computes dW as `grad.T @ input` in that same layout, avoiding an external
transposed-gradient copy. Ordinary `[E,K,N]` callers retain the default
`trans_b=False`. Empty ranks now use AO's zero-gradient path without reducing
all expert parameters on the host-framework path.

The NPU grouped BF16 path also requires this AO API. With `use_grouped_mm=False`,
the existing sequential expert path remains available. CUDA/Triton and CPU
paths keep their existing implementation and do not eagerly import AO.
