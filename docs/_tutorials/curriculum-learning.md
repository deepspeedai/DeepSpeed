---
title: "Curriculum Learning: A Regularization Method for Efficient and Stable Billion-Scale GPT Model Pre-Training"
tags: training pre-training
---

**Watch out!**
On 12/12/2022, we released DeepSpeed Data Efficiency Library which provides a more general curriculum learning support. This legacy curriculum learning feature below is still supported but we recommend to use the Data Efficiency Library ([tutorial](/tutorials/data-efficiency/)).
{: .notice--warning}

**Note:**
This tutorial was updated on 10/29/2021. Changes include: 1) A more detailed tuning strategy. 2) Pipeline parallelism support. 3) Token-based learning rate decay. 4) A new GPT-2 example at [github.com/microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). See details below.
{: .notice--info}

In this tutorial, we introduce DeepSpeed's curriculum learning-based data pipeline, which presents easier or simpler examples earlier during training. By enabling stable training with 8x/4x larger batch size/learning rate (whereas the baseline approach struggles with training divergence), we observe that curriculum learning (based on sequence length) provides stable and 3.3x faster GPT-2 pre-training (tested on 117M and 1.5B parameters), together with better token-wise convergence speed and zero-shot WikiText-103/LAMBADA evaluation results. In addition, since curriculum learning only affects the data pipeline, its benefit is complementary to many DeepSpeed features and other system optimization techniques. For example, curriculum learning is compatible with DeepSpeed's [ZeRO Redundancy Optimizer](/tutorials/zero/), [ZeRO-Offload](/tutorials/zero-offload/), and [3D Parallelism](/tutorials/pipeline/).

To illustrate the benefits and usage of curriculum learning, we use the Megatron-LM GPT-2 pre-training task as example. For more details on this task, please refer to the [Megatron-LM GPT2 tutorial](/tutorials/megatron/). In addition, we also have a [paper](https://arxiv.org/abs/2108.06084) which provides the technical details including implementation and evaluations.

## 1. Configurations and tuning strategy
Curriculum learning can be used by setting the `curriculum_learning` key in the DeepSpeed configuration file:

```json
{
  "train_batch_size": 4096,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015,
      "max_grad_norm": 1.0,
      "betas": [0.9, 0.95]
    }
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
  },
  "curriculum_learning": {
    "enabled": true,
    "curriculum_type": "seqlen",
    "min_difficulty": 8,
    "max_difficulty": 1024,
    "schedule_type": "fixed_linear",
    "schedule_config": {
      "total_curriculum_step": 15000,
      "difficulty_step": 8
    }
  }
}
```
To support curriculum learning, we add the following new parameters:

`curriculum_type` is the type of curriculum difficulty metric. Currently we support the `seqlen` metric which presents shorter sequences earlier in training. We implement this type of curriculum learning by performing training data sequence truncation before the actual forward pass. We will describe how to implement this in the Megatron-LM GPT-2 pre-training example below.

`min_difficulty` is the starting difficulty level. For the `seqlen` metric it means we start with sequence length as `min_difficulty`. We observe that lower `min_difficulty` usually provides better stability/convergence speed but with two caveats: First, sometimes (especially for large models) starting with too small difficulty level may lead to severe overfitting (e.g., training loss divergence or validation perplexity fluctuations) thus hurting the convergence. Second, for `seqlen` metric we recommended setting `min_difficulty` to a multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable [NVIDIA GPU's Tensor Core acceleration](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/). To tune this hyperparameter for `seqlen` metric, we recommend starting with `min_difficulty` at 8 (million-scale models) or 64 (billion-scale models), and then increase it if you observe divergence or validation perplexity fluctuations at the very beginning.

`max_difficulty` is the ending difficulty level. For the `seqlen` metric it should be set to the full sequence length (e.g., 1024 for Megatron-LM GPT-2 pre-training).

`schedule_type` is the scheduling policy for curriculum learning (i.e., which difficulty level to use at certain step). Currently we support three schedules: `fixed_linear`, `fixed_root`, and `fixed_discrete`. We recommend to first try the `fixed_linear` schedule, which is easier to tune and provides great training stability/efficiency gain in our tests. Each schedule has its own configurations:


### 1.1 fixed_linear schedule
For `fixed_linear` schedule there are two configurations:

```json
"schedule_type": "fixed_linear",
"schedule_config": {
  "total_curriculum_step": 15000,
  "difficulty_step": 8
}
```

The `total_curriculum_step` is the total number of steps for the curriculum learning. For `fixed_linear` schedule the difficulty level will increase linearly from `min_difficulty` to `max_difficulty` during `total_curriculum_step` steps. This configuration must be tuned for each training task. We observe that too small and too large `total_curriculum_step` are both suboptimal: with too small `total_curriculum_step` curriculum learning might not be able to provide enough training stability benefit so the training might still diverge; with too large `total_curriculum_step` the model may overfit during curriculum learning on the easier/simpler training data thus hurt the overall convergence. To tune this hyperparameter, we recommend a binary search to find the largest `total_curriculum_step` that does not have significant validation perplexity fluctuation during the first few multiples of LR warmup steps. The underlying rationale can be found in our [paper](https://arxiv.org/abs/2108.06084) Appendix A.1.

The `difficulty_step` configuration ensures that at any time the difficulty level is a multiple of `difficulty_step`. A smaller value is preferable since it gives more smooth curriculum and better stability. We usually set it to 8 (for FP16 data) or 16 (for INT8 data) to enable [NVIDIA GPU's Tensor Core acceleration](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/). If this is unrelated to your hardware, you can set it to 1.

### 1.2 fixed_root schedule
For `fixed_root` schedule there are three configurations:

```json
"schedule_type": "fixed_root",
"schedule_config": {
  "total_curriculum_step": 15000,
  "difficulty_step": 8,
  "root_degree": 2
}
```

The `total_curriculum_step` and `difficulty_step` have the same meaning as for the `fixed_linear` schedule. The `root_degree` determines the root degree of the root function of the schedule. The difficulty level at certain step is determined as `((current step/total_curriculum_step)**(1/root_degree)) * (max_difficulty - min_difficulty) + min_difficulty`. Thus `fixed_linear` is basically a special case of `fixed_root` with `root_degree` as 1. In our (limited) study, we find the `fixed_root` schedule does not provide any clear advantage over `fixed_linear` schedule, while requiring one additional parameter.

### 1.3 fixed_discrete schedule
For `fixed_discrete` schedule there are two configurations:

```json
"schedule_type": "fixed_discrete",
"schedule_config": {
  "difficulty": [1,2,3],
  "max_step": [5,10]
}
```

The `difficulty` is a list of difficulty levels to be used during schedule. The `max_step` is a list of step timestamp to determine when to switch to next difficulty level. For example, the json config above means that at step 1-5 difficulty 1 is used, at step 6-10 difficulty 2 is used, from step 11 difficulty 3 is used. This `fixed_discrete` schedule provides the most flexible curriculum learning scheduling. However, we find that one risk of this kind of schedule is that if the model stays at certain difficulty level for too long, training divergence may happen when switching to next difficulty due to severe overfitting.

## 2. Curriculum learning for Megatron-LM GPT-2 pre-training

**Watch out!**
After the update on 10/29/2021, now there are two curriculum learning examples for Megatron-LM GPT-2 pre-training. Both of them have some unique features and limitations. See details below.
{: .notice--warning}

We provide two curriculum learning examples for Megatron-LM GPT-2 pre-training:

The first one is at [Megatron-DeepSpeed/tree/main/examples_deepspeed/curriculum_learning](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/curriculum_learning). This integration is based on a newer Megatron-LM fork, and only this curriculum learning example supports pipeline parallelism. However, as of 10/29/2021, we haven't verified ZeRO-2 and ZeRO-3 on this fork. Overall, we highly recommend you to use this example if your model does not require ZeRO-2/3.

The second one is at [DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/curriculum_learning/](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM-v1.1.5-ZeRO3/curriculum_learning). This integration is based on an older Megatron-LM hard copy that we will eventually deprecate and this curriculum learning example does not support pipeline parallelism. We recommend you to ONLY use this example if your model requires ZeRO-2/3.

Besides the DeepSpeed curriculum learning json configurations described above, there are some other necessary changes on the user side to integrate curriculum learning:

### 2.1 Training data truncation

To enable `seqlen`-based curriculum learning, we need to add the functionality of training data truncation based on the given curriculum sequence length. For the case without pipeline parallelism, it is necessary to add a `curriculum_seqlen` argument in the model's forward pass and use it to perform training data sequence length truncation. For Megatron-LM GPT-2 pre-training, we implement this in `forward()` in [megatron/model/gpt2_model.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-ZeRO3/megatron/model/gpt2_model.py) and in `forward_step()` in [pretrain_gpt2.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-ZeRO3/pretrain_gpt2.py).

For the case with pipeline parallelism, due to DeepSpeed engine limitations we cannot inject the `curriculum_seqlen` argument in the forward pass. Instead, we create a duplicate of `deepspeed.runtime.data_pipeline.curriculum_scheduler` on the user side, and use it to retrieve the `curriculum_seqlen`. This implementation can be found in [megatron/training.py](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/training.py).

### 2.2 Disable batch size warmup (`--rampup-batch-size`)
In our [paper](https://arxiv.org/abs/2108.06084) section 5.4 we demonstrate that curriculum learning (`seqlen`-based) provides much better training stability than the batch size warmup technique introduced by Open AI GPT-3. So when using curriculum learning you need to remove the `--rampup-batch-size` config in your training script. It's not recommended using both curriculum learning and batch size warmup, because both of them reduce the number of tokens in a batch. Another related change you might want is to increase your micro batch size, since without batch size warmup your batch size will be fixed now.

### 2.3 Token-based training termination

Because curriculum learning changes the length of each sequence/sample during training, it is very hard/impossible to use  a number of steps/samples to terminate the training exactly at the desired number of tokens. Thus, we add a `--train-tokens` config for accurate token-based termination. We recommend increasing your original `--train-samples` or `--train-iters` to a large enough number (e.g., 3X of what you used for baseline), and set `--train-tokens` at the exact desired number of training tokens.

### 2.4 Token-based LR decay

Again because curriculum learning changes the number of tokens per batch, in our [paper](https://arxiv.org/abs/2108.06084) Appendix A.2 we show that it is also necessary to change the LR decay to token-based (to avoid decaying LR too fast). Thus, we add a `--lr-decay-tokens` which will be the number of LR decay tokens. If previously you were using `--lr-decay-samples`, you can calculate your `--lr-decay-tokens` simply by multiplying the former by full `seqlen` (e.g., 1K for GPT-2 and 2K for GPT-3). If previously you were using `--lr-decay-iters`, you can calculate your `--lr-decay-tokens` by multiplying the former by full `seqlen` and the global batch size. Then you need to replace `--lr-decay-samples` or `--lr-decay-iters` with `--lr-decay-tokens` in your script.

### 2.5 LR warmup adjustment

For LR warmup we don't change it to token-based, because doing so for curriculum learning means slowing down the LR warmup, which is both unnecessary and harmful. However, to avoid too fast warmup you may need to adjust your `--lr-warmup-samples` or `--lr-warmup-iters` from non-CL cases for various reasons (e.g., if you used `--rampup-batch-size` in non-CL case, for CL we don't use it so the number of samples per batch will be different at beginning). Assuming you want to use `X` tokens to warmup the LR (for OpenAI GPT-3 this was 375M tokens), then for curriculum learning case you shall set `--lr-warmup-samples` as `X` divided by the `min_difficulty`, or set `--lr-warmup-iters` as `X` divided by `min_difficulty * --global-batch-size`. This is a rough estimation based on that curriculum learning starts from seqlen `min_difficulty` and it won't increase too much during LR warmup.
