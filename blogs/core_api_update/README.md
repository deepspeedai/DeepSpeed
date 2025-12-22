# DeepSpeed Core API updates: PyTorch-style backward and low-precision master states

DeepSpeed is continuously evolving its core APIs to feel more natural to PyTorch users while giving them more control over performance and memory.

In this short blog, we highlight two recent core improvements:

  * **PyTorch-compatible backward API** – You can now use standard `tensor.backward(...)` patterns with DeepSpeed engines, including non-scalar outputs. ([\#7665](https://github.com/deepspeedai/DeepSpeed/pull/7665))
  * **Low-precision master params / grads / optimizer states** – You can keep more state in bf16/fp16 to reduce memory usage and work better with `torch.autocast`. ([\#7700](https://github.com/deepspeedai/DeepSpeed/pull/7700))

These changes enable more flexible training pipelines, such as [disaggregated hybrid parallelism](https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid) ([code](https://github.com/ray-project/multimodal-training)) for multimodal models using [Ray](https://github.com/ray-project/ray), and make DeepSpeed feel closer to “vanilla PyTorch”.

## 1\. PyTorch-compatible backward API

Traditionally, DeepSpeed’s training loop relied on the engine’s backward API:

```python
loss = model_engine(batch)
model_engine.backward(loss)
model_engine.step()
```

This API was sufficient for traditional pretraining and fine-tuning pipelines. However, recent complex training pipelines require more flexibility. There were two major constraints:

1.  It only accepted a **scalar loss**.
2.  You had to call **`model_engine.backward(loss)`**, rather than using the usual PyTorch `loss.backward()` style.

Due to these constraints, users could not simply implement patterns that plain PyTorch allows. Here are some examples:

```python
# 1. Combine multiple models and losses
output1 = model1(batch1)
output2 = model2(batch2)
loss = criterion(output1, output2)
loss.backward()

# 2. Define a loss function separately from the main model
output = model(batch)
loss = loss_fn(output)
loss.backward()

# 3. Call backward through non-scalar tensors with custom gradients
output = model(batch)
output.backward(grad)
```

The DeepSpeed Engine was able to handle these use cases using internal APIs; however, this required code changes. Additionally, if a user employed these patterns, the DeepSpeed engine might skip internal preprocessing/postprocessing (such as loss scaling and ZeRO-related logic), potentially leading to incorrect behavior.

With this API update, we can now use the same code as native PyTorch while keeping DeepSpeed's unique features, including ZeRO.

One example use case for this new API is [disaggregated hybrid parallelism](https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid) for multimodal models using [Ray](https://github.com/ray-project/ray). In this training pipeline, two Ray Actor groups handle the vision encoder and the LLM separately.

On a backward pass, the LLM passes a gradient to the vision encoder, and the vision encoder calls the backward function with that gradient. However, because the gradient is a non-scalar tensor, such a use case wasn't officially supported by DeepSpeed APIs.

Below is the pseudo-code for the two models running on different actors. Since they run in different processes, we pass gradients via Ray actor communication. As seen here, the gradient of the vision embedding is a non-scalar tensor. With this update, we can now simply call `self.vision_output.backward` while utilizing other DeepSpeed features, including ZeRO and highly efficient sequence parallelism (DeepSpeed-Ulysses).

```python
# Runs on LLM actors
def text_backward_step(self):
    # ...
    self.loss.backward()
    return self.vision_embeddings.grad.detach().clone()

# Runs on Vision actors
def vision_backward_step(self, vision_embedding_grad):
    self.vision_output.backward(gradient=vision_embedding_grad)
```

Check out the [repository](https://github.com/ray-project/multimodal-training) for the complete training pipeline.


## 2\. Low-precision master params, grads, and optimizer states

DeepSpeed supports mixed precision, which computes in bfloat16 or float16 while its optimizer maintains **FP32 master parameters, gradients, and optimizer states**.

On the other hand, PyTorch now offers `torch.autocast`, a different approach for mixed precision that casts data types for precision-sensitive operators on the fly. As this often requires less peak memory, many recent training pipelines use this approach.

DeepSpeed supports `torch.autocast` via configuration (see the [API documentation](https://deepspeed.readthedocs.io/en/rtd-staging/training.html#pytorch-automatic-mixed-precision-amp)). While it is technically safer to keep FP32 model states (master parameters/gradients and optimizer states) even with `torch.autocast`, there are many cases where training converges stably without them. Previously, the lack of an option to bypass creating FP32 states limited the trainablity of large models with constrained hardware resources.

To reduce memory usage in such cases, DeepSpeed now allows users to avoid creating FP32 states entirely.

### Enabling pure BF16/FP16 model states

For BF16 training, you can use the following settings under `bf16`:

  * `bf16_master_weights_and_grads`: Keep master parameters and gradients in bf16.
  * `bf16_optimizer_states`: Keep optimizer states (e.g., Adam moments) in bf16.

These configurations are compatible with ZeRO stages 1, 2, and 3. Note that there is also a supported mixed configuration where `bf16_master_weights_and_grads == true` and `bf16_optimizer_states == false`, but **only when using CPU offload**.

We offer similar support for FP16 training. You can use this setting under `fp16`:

  * `fp16_master_weights_and_gradients`: Keep master parameters and gradients in fp16.

We actually offered this option in previous versions, but it was undocumented and worked only for ZeRO 1 and 2. We now officially support it, and it works for all ZeRO stages. We intentionally excluded `fp16_optimizer_states` as it is generally impractical due to convergence instability.

A notable improvement is that we can combine these settings with `torch.autocast` support (via the [`torch_autocast` section](https://www.google.com/search?q=%5Bhttps://deepspeed.readthedocs.io/en/rtd-staging/training.html%23pytorch-automatic-mixed-precision-amp%5D\(https://deepspeed.readthedocs.io/en/rtd-staging/training.html%23pytorch-automatic-mixed-precision-amp\))). This combination drastically improves both memory efficiency and convergence.

### Example: Pure bf16 config with low-precision master state

Below is a simplified DeepSpeed config that keeps bf16 master weights, grads, and optimizer states, and uses `torch.autocast`:

```json
{
...
  "zero_optimization": {
    "stage": 3,
    ...
  },
  "bf16": {
    "enabled": true,
    "bf16_master_weights_and_grads": true,
    "bf16_optimizer_states": true
  },
  "torch_autocast": {
    "enabled": true,
    "dtype": "bfloat16"
  }
}
```

Our [example script](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/bf16_master_weight) demonstrates the significant memory savings:

| Configuration | Allocated Memory | Peak Memory | Avg Step Time |
|---------------|------------------|-------------|---------------|
| Baseline (fp32 master) | 25.74 GB | 31.38 GB | 0.6016s |
| BF16 low-precision (master + opt states) | **16.17 GB** | **18.93 GB** | 0.6427s |

To verify that BF16 low-precision training maintains numerical stability, we trained for 1000 steps on the Wikitext-103 dataset:

<div align="center">

<img src="assets/loss_comparison.png" width="800">

*Loss curve comparison*

</div>

| Configuration | Final Loss | Mean Loss |
|---------------|------------|-----------|
| Baseline (fp32 master) | 3.09 | 2.78 |
| BF16 Low-Precision | 3.12 | 2.90 |

Please check out our [example](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/bf16_master_weight) for more details.

## Closing thoughts

These core API improvements are incremental but important steps toward making DeepSpeed:

  * **More PyTorch-native** – Training loops can increasingly look like standard PyTorch code.
  * **More memory-efficient** – Especially when combined with bf16/fp16 and ZeRO on large models.
  * **Easier to compose** – Enabling multi-model and custom-gradient workflows without relying on DeepSpeed internal APIs.

We're excited to see how you use these APIs in your own training setups, and we welcome feedback and issues on GitHub as you try them out.

## Related Tests

For more usage examples, see the unit tests in the repository:

- [PyTorch-compatible backward API](https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/zero/test_zero_user_backward.py)
- [Low-precision master params/grads/optimizer states](https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/half_precision/test_bf16.py)
- [Combination with torch.autocast](https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/half_precision/test_with_autocast.py)
