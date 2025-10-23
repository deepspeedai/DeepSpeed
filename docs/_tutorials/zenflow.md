---
title: "ZenFlow"
tags: training
---

ZenFlow is an extension of ZeRO-Offload that decouples and asynchronously updates gradients during training. It reduces CPU-induced stalls when using offload optimizers, enabling smoother and faster training. Like ZeRO-Offload, ZenFlow requires no code changes, only configuration updates in your DeepSpeed JSON file.

We recommend that you read the tutorials on [Getting Started](/getting-started/) and [ZeRO](/tutorials/zero/) before stepping through this tutorial. ZenFlow builds on top of [ZeRO-Offload](/tutorials/zero-offload/), so shared setup details can be found there.

## Configuration Changes

To enable ZenFlow, simply add a `zenflow` section under the existing `zero_optimization` block in your DeepSpeed config:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "zenflow": {
      "topk_ratio": 0.05,
      "select_strategy": "auto",
      "select_interval": "auto",
      "update_interval": 4,
      "full_warm_up_rounds": 0,
      "overlap_step": true
    }
  }
}
```


Each field in the `zenflow` block controls selective gradient update behavior:

- `topk_ratio`: Fraction of the most important gradients to update on GPU (e.g., 0.05 means top 5% by importance).
- `select_strategy`: Strategy for selecting important gradients (`"auto"`, `"step"`, or custom).
- `select_interval`: How often to re-select important gradients (`"auto"` or integer like 1).
- `update_interval`: How often to update unimportant gradients (`"auto"` or an integer like 4, meaning every 4 steps).
- `full_warm_up_rounds`: Number of initial steps with full gradient updates before selection begins.
- `overlap_step`: Whether to overlap communication with computation (`true` enables it).

---

**Recommended**: Use `"auto"` for `select_strategy`, `select_interval`, and `update_interval` to enable adaptive behavior with minimal tuning.

You can continue using the same training setup and launch script as in the [ZeRO-Offload tutorial](/tutorials/zero-offload/), since ZenFlow builds directly on top of ZeRO Offload.

## Quick Start: Fine-tuning Example

A complete fine-tuning example using ZenFlow is available in [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) -- [ZenFlow Fine-Tuning on GLUE](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/DeepSpeed-ZenFlow)

This example shows how to fine-tune a GPT model on the GLUE benchmark with:
- CPU optimizer offload
- ZenFlow asynchronous updates

To run the example:

```bash
cd DeepSpeedExamples/training/DeepSpeed-ZenFlow
bash finetune_gpt_glue.sh
```

Refer to the `README.md` in the folder for setup instructions, dataset preparation, and configuration details.

---

Congratulations! You have successfully enabled ZenFlow for stall-free offloading.
