---
# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
title: "TorchAO HiFloat8 training"
---

DeepSpeed selects ordinary Linear and AutoEP GroupedExperts modules and converts
both kinds through one public `torchao.quantization.quantize_` call. DeepSpeed owns selection, activation,
routing/padding, communication, optimizer and checkpoint lifecycle. TorchAO owns
numerical policy and the quantization/GEMM implementations.

## Quick start

Add the following section to your existing DeepSpeed training configuration.
AutoEP topology, batch size and optimizer settings remain in that configuration.

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "quantization": {
    "hifloat8": {
      "enabled": true,
      "module_name_patterns": ["*.experts", "*.shared_experts.*_proj"],
      "min_numel": 65536
    }
  }
}
```

Pass the configuration to the normal initialization call; DeepSpeed applies
HiFloat8 after AutoEP replacement and device placement, before optimizer setup:

```python
import deepspeed

engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)
```

For a Dense-only model, omit AutoEP and select the actual Linear module names.
Set `quantization.hifloat8.enabled=false` to return to the high-precision path.

AutoEP topology and trainable-parameter selection are configured separately.
Every pattern must match an eligible module. Optional `expected_module_count`
checks the complete selection before backend loading. The `quantization`
section gives each training format its own module selection. HiFloat8 is the
first supported format. Existing top-level `hifloat8` files and the
`backend="torchao_npu"` alias still work. Use one location for HiFloat8 options.

## Per-module recipes

`module_configs` assigns numerical overrides to individual logical modules.
For example, two projections can use different gradient ranges with one conversion
pass. Expert names refer to their AutoEP containers.

```json
{
  "bf16": {"enabled": true},
  "quantization": {
    "hifloat8": {
      "enabled": true,
      "config": {"weight_dst_type_max": 15},
      "module_configs": {
        "model.layers.0.self_attn.q_proj": {"grad_dst_type_max": 15},
        "re:model\\.layers\\.[0-9]+\\.self_attn\\.q_proj": {"grad_dst_type_max": 224},
        "re:model\\.layers\\.[0-9]+\\.mlp\\.experts": {"grad_dst_type_max": 224},
        "_default": null
      }
    }
  }
}
```

Exact names take precedence over regular expressions. Regex keys start with `re:`
and match the full module name. The first matching regex in JSON order wins,
followed by `_default`. A dictionary overrides fields in the common `config`;
`null` skips that module. Only eligible Linear and GroupedExperts modules
participate in selection. Each explicit selector must match an eligible module.

When `module_name_patterns` is also present, its globs restrict the scope and
modules without an override inherit the common recipe. When only `module_configs`
is supplied, unmatched modules are skipped. `expected_module_count` counts
converted logical modules after null exclusions. The empty module name `""`
selects a root Linear or GroupedExperts model.

## Conversion and device ownership

The NPU accelerator resolves the public TorchAO `HiFloat8Config` class.
DeepSpeed maps the selected Linear modules and AutoEP grouped GEMM leaves
to their configurations using `FqnToConfig`. AO parses every selected recipe before conversion and converts
the modules in one `quantize_` call. DeepSpeed keeps expert parameters,
activation, routing and padding on the expert container. The GEMM leaf
receives packed weights and their layout on each call, so quantization
preserves parameter identity and checkpoint keys.

The separate `get_grouped_mm()` accelerator hook supplies ordinary high-precision
GEMM. Its NPU implementation calls TorchNPU with explicit list-input autograd,
packed `[E,N,K]` weights and empty-group gradients. The operator implementation
lives in `deepspeed/ops/npu/grouped_mm.py`; the accelerator only resolves the
capability. HiFloat8 API discovery lives in `accelerator/npu/hifloat8.py`.
BF16/FP16 grouped training does not require TorchAO. CPU/CUDA dispatch retains
its existing implementation.

## Optional TorchAO policy

Install `torchao>=0.18,<0.19`, Triton and the official Ascend extension
with the composite `HiFloat8Config` and automatic activation enabled.
The accelerator resolves the dependency only when HiFloat8 is enabled.
DeepSpeed validates selected modules and calls `HiFloat8Config.from_dict`
before conversion. Verify native HiFloat8 forward, input-gradient and
weight-gradient kernels on the target hardware before training.

Missing, null or empty `config` uses the established input/weight/gradient scale
denominators `15/15/224`. An explicit null range uses the native AO default. `compute_dtype`
selects high-precision staging and GEMM output precision (`null`, `"bfloat16"`
or `"float16"`); operands remain HiFloat8 payloads and accumulation is
kernel-owned. Converted Linear and expert modules return their caller's dtype,
so FP16 working GEMMs compose with BF16 residuals. Raw tensor helpers retain
their working/output dtype contract. Scaling currently supports per-tensor only.

## Validation boundary

The current NPU HiFloat8 path requires BF16 model parameters and accepts ZeRO
stages 0–2. ZeRO-3, PipelineModule, AutoTP, custom MPU and explicit DeepSpeed MoE
layers are rejected, as is an explicit fused gate/up override on selected experts.

HiFloat8 and subchannel use separate conversion paths. Enable only one per engine
because both can replace the same modules. A selected HiFloat8 AutoEP expert
group cannot use `overlap_weight_grad` or `deferred_dw_flush_period` greater than
one because AO computes its weight gradients through its own autograd implementation.
Quantized checkpoint ingestion into ZeRO-3 is unsupported;
checkpoints retain high-precision weights and their original state names.
Reconstruct the same policy for resume. The opt-in DP2/GAS2 lifecycle entry point
is `tests/unit/runtime/hifloat8_resume_validation.py`, whose experts are replicated.
It does not establish AutoEP checkpoint recovery.

The integration includes Qwen3.5 text-wrapper routing compatibility and NPU
permutation index generation. These reuse AutoEP's existing preset and kernel
extension points without adding HiFloat8 configuration. Full-parameter convergence,
LoRA, offload, compile, EP8, multi-node and CUDA low-precision hardware validation
remain separate gates.
