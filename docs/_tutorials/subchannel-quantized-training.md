---
title: "Subchannel FP8/INT8 Quantized Training"
tags: training
---

Subchannel quantized training runs the matrix multiplications of `nn.Linear` layers and AutoEP grouped experts in FP8 or INT8 instead of BF16. It needs no code changes, only a `quantization` section in your DeepSpeed config.

This tutorial explains what the feature does, when it makes training faster and when it does not, and how to configure it. We recommend that you read [Getting Started](/getting-started/) first. The full parameter reference is in the [configuration documentation](/docs/config-json/#subchannel-quantized-training).

## What it does

Every linear layer runs three matrix multiplications (GEMMs) per training step:

| GEMM | Computes | Reduction dimension |
| ---- | -------- | ------------------- |
| `forward` | `y = x W^T` | input features |
| `dx` | `dx = dy W` | output features |
| `dw` | `dW = dy^T x` | tokens |

Each GEMM is configured on its own. Just before a GEMM runs, both of its operands are quantized along the reduction dimension, with FP32 scales that restore their magnitude in the product. The result comes back in the layer's usual dtype.

Parameters, gradients, optimizer state and activations all keep their usual precision. Only the GEMM operands are quantized, on the fly, at every call. This means the feature **does not reduce memory use**.

## When to use it

There are two modes, selected by `block_size`, and they serve different purposes.

**Row scaling (`block_size: 0`) is the mode for speed.** Every operand row gets one scale for the whole reduction dimension, so the GEMM is a plain FP8 tensor-core loop and the scales are applied once per output element. The GEMM itself runs about 1.6x faster than BF16, but both operands still have to be quantized in a separate pass at every call. Whether a training step gets faster depends on the layer sizes:

| Layer step (tokens x in -> out), one NVIDIA H200 | Speed vs BF16 |
| ------------------------------------------------- | ------------- |
| 32768 x 8192 -> 8192 | 1.39x |
| 16384 x 8192 -> 8192 | 1.34x |
| 8192 x 4096 -> 14336 | 1.16x |
| 8192 x 4096 -> 4096 | 0.78x |
| 8192 x 2048 -> 2048 | 0.23x |

In end-to-end training on 4 H200 GPUs with 8192 tokens per GPU per step, row scaling was 1.09x faster than BF16 on Qwen2.5-14B and 1.21x faster on Qwen2.5-32B, with validation loss 0.004 to 0.006 above BF16 after 100 steps. On the MoE models Qwen1.5-MoE-A2.7B and Qwen3-30B-A3B, whose layers are about 2048 wide, it was 1.5x slower.

Use row scaling when your model's linear layers are roughly 5000 or more wide and each GPU processes several thousand tokens per step. Measure your step time against BF16 before committing to it.

**Block scaling (`block_size` 32, 64, 128 or 256) is the mode for numerics.** Every block of the reduction dimension gets its own scale, which follows the data more closely: on the Qwen2.5 models it roughly halved the loss gap to BF16 compared with row scaling. The scales have to be applied between tensor-core dot products, which makes a block-scaled GEMM slower than BF16 on every model we measured (0.74x on the Qwen2.5 models). Use it to study low-precision training, for INT8, or with the Hadamard rotation, not for throughput.

## Enabling it

Add a `quantization` section to the config. This enables row-scaled FP8 for all three GEMMs:

```json
{
  "bf16": {"enabled": true},
  "quantization": {
    "subchannel": {
      "enabled": true,
      "forward": {"block_size": 0},
      "dx": {"block_size": 0},
      "dw": {"block_size": 0}
    }
  }
}
```

With only `"enabled": true`, the defaults apply: FP8 `e4m3`, block scaling with 128-element blocks, round-to-nearest in the forward GEMM, and stochastic rounding in `dx` and `dw`.

At `deepspeed.initialize()` the selected `nn.Linear` modules are converted in place and the selected AutoEP experts are routed through the quantized GEMMs. Module names, parameters, hooks and state-dict keys do not change, so checkpoints stay compatible with an unquantized model.

## Choosing which modules are quantized

`modules` and `exclude_modules` are lists of `fnmatch` patterns over qualified module names. A module is converted when it matches `modules` and does not match `exclude_modules`.

```json
"subchannel": {
  "enabled": true,
  "modules": ["*.mlp.*"],
  "exclude_modules": ["*lm_head*", "*router*", "*.gate"]
}
```

- By default every `nn.Linear` and every AutoEP expert module is selected, except the output head and MoE routers. Quantized router logits change which experts tokens reach, so routers stay in full precision.
- If you set `exclude_modules` yourself, repeat the defaults you still want; your list replaces them.
- Only modules whose type is exactly `nn.Linear` are converted. Subclasses are left alone.
- Initialization fails if the patterns select nothing, so a typo cannot silently train in full precision.

Since the speed of row scaling depends on layer width, a useful pattern is to quantize only the wide layers of a model, for example its MLP projections, and leave narrow ones in BF16.

## Per-GEMM settings

Each of `forward`, `dx` and `dw` accepts the same four settings.

**`format`**: `"e4m3"` (default), `"e5m2"`, `"int8"`, or `"none"` to leave that GEMM unquantized. `e4m3` has three mantissa bits and is the more accurate FP8 format; `e5m2` trades one of them for exponent range. For example, to quantize only the forward pass:

```json
"forward": {"format": "e4m3", "block_size": 0},
"dx": {"format": "none"},
"dw": {"format": "none"}
```

**`block_size`**: `0` for row scaling, or 32, 64, 128 (default) or 256 for block scaling. Row scaling requires an FP8 format. In the `dw` GEMM the reduction runs over tokens; for AutoEP experts, blocks restart at every expert and row scaling uses one scale per expert, so no scale mixes two experts' tokens.

**`stochastic_rounding`**: rounds up or down at random with probabilities that make the result unbiased, so rounding errors average out over steps instead of accumulating in one direction. It defaults to `false` for `forward` and `true` for `dx` and `dw`. In our 100 to 300 step runs it made no measurable difference to validation loss; the default follows common practice for low-precision gradients.

**`hadamard_size`**: rotates every `hadamard_size` chunk of a block by an orthonormal Hadamard matrix before quantizing. The rotation spreads an outlier over the whole chunk, and because both operands are rotated the same way it cancels in the product. It is a power of two up to `block_size`, and `0` (default) disables it. It is not available with row scaling.

The rotation mainly helps INT8, where one outlier inflates the scale of its whole block. On synthetic activations with outlier channels it reduced INT8 GEMM error by about 3.5x. It does not help FP8, which already gives every element the same relative precision, and it made no measurable difference to validation loss on the MoE models we trained.

```json
"forward": {"format": "int8", "hadamard_size": 128},
"dx": {"format": "int8", "hadamard_size": 128, "stochastic_rounding": true},
"dw": {"format": "int8", "hadamard_size": 128, "stochastic_rounding": true}
```

## Hardware requirements

The fused GEMMs are Triton kernels. They need Triton and a CUDA device of compute capability 8.9 or newer for FP8 (for example L40S, RTX 4090, H100, H200), or 8.0 or newer for INT8.

On any other device the GEMMs run on a PyTorch reference implementation that applies the same quantization and multiplies in FP32. DeepSpeed logs a warning when this happens. The reference reproduces the numerics, which is useful for experiments, but it is not meant for speed.

## Limitations

- Validated with ZeRO stages 1, 2 and 3 for `nn.Linear`, and with ZeRO-2 for AutoEP experts, all in BF16.
- AutoTP and `PipelineModule` are rejected at initialization.
- AutoEP's `gate_up_impl="fused"`, `overlap_weight_grad` and `deferred_dw_flush_period` above 1 are rejected for quantized experts.
- Not validated: FP16 and FP32 training, CPU or NVMe offload, ZeRO-3 with AutoEP, gradient accumulation and `torch.compile`.
- The accuracy results above come from short continued-pretraining runs. Check convergence on your own workload before relying on it for a full training run.
