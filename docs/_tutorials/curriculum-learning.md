---
title: "Curriculum Learning: A Regularization Method for Efficient and Stable Billion-Scale GPT Model Pre-Training"
---

In this tutorial, we introduce DeepSpeed's curriculum learning-based data pipeline, which presents easier or simpler examples earlier during training. By enabling stable training with 8x/4x larger batch size/learning rate (whereas the baseline approach struggles with training divergence), we observe that curriculum learning (based on sequence length) provides stable and 2.6x faster GPT-2 pre-training (tested on 117M and 1.5B parameters), together with better token-wise convergence speed and zero-shot WikiText-103/LAMBADA evaluation results. In addition, since curriculum learning only affect the data pipeline, its benefit is complementary to many DeepSpeed features and other system optimization techniques. For example, curriculum learning is compatible with DeepSpeed's [ZeRO Redundancy Optimizer](/tutorials/zero/) and [ZeRO-Offload](/tutorials/zero-offload/), and Megatron-LM's Model Parallelism.

To illustrate the benefits and usage of curriculum learning, we use the Megatron-LM GPT-2 pre-training task as example. For more details on this task, please refer to the [tutorial](/tutorials/megatron/). In addition, we also have a [paper](https://arxiv.org/abs/2108.06084) which provides the technical details including implementation and evaluations.

## 1. Configurations and tuning strategy
Curriculum learning can be used by setting the DeepSpeed configuration as the following example json config file:

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

`curriculum_type` is the type of curriculum difficulty metric. Currently we support the `seqlen` metric which presents shorter sequences earlier in training. We implement this type of curriculum learning by passing an additional `curriculum_seqlen` argument to the model's forward function, and performing training data sequence truncation before the actual forward pass. We will describe how to implement this in the Megatron-LM GPT-2 pre-training example below.

`min_difficulty` is the starting difficulty level. For `seqlen` metric it means we start with sequence length as `min_difficulty`. We observe that lower `min_difficulty` usually provides better convergence speedup but with two caveats: First, sometimes (especially for large models) starting with too small difficulty level may lead to severe overfitting (e.g., training loss divergence or validation loss keeps jumping up and down) thus hurt the convergence. In such case it is recommended to try increasing the `min_difficulty`. Second, for `seqlen` metric it is recommended to set `min_difficulty` as multiple of 8 (for FP16 data) or 16 (for INT8 data) in order to enable [NVIDIA GPU's Tensor Core acceleration](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/).

`max_difficulty` is the ending difficulty level. For `seqlen` metric it should be set as the full sequence length (e.g., 1024 for Megatron-LM GPT-2 pre-training).

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

The `total_curriculum_step` is the total number of steps for the curriculum learning. For `fixed_linear` schedule the difficulty level will linearly increase from `min_difficulty` to `max_difficulty` during the `total_curriculum_step` duration. This configuration needs to be tuned for each training task. We observe that too small and too large `total_curriculum_step` are both suboptimal: with too small `total_curriculum_step` curriculum learning might not be able to provide enough training stability benefit so the training might still diverge; with too large `total_curriculum_step` the model may overfit too much during curriculum learning on the easier/simpler training data thus hurt the overall convergence. We recommend to first set `total_curriculum_step` as 20% to 40% of the total training steps (note that if you increase the batch size for the curriculum learning-based training, you also need to reduce the total training steps correspondingly), then increase the `total_curriculum_step` if the training is not stable, or reduce the `total_curriculum_step` to test if convergence improves.

The `difficulty_step` configuration ensures that at anytime the difficulty level must be multiple of `difficulty_step`. We usually set it as 8 (for FP16 data) or 16 (for INT8 data) to enable [NVIDIA GPU's Tensor Core acceleration](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/). If this is unrelated to your training experiment, you can set it as 1.

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

The `total_curriculum_step` and `difficulty_step` have the same meaning as in the `fixed_linear` schedule case. The `root_degree` determines the root degree of the root function of the schedule. The difficulty level at certain step is determined as ((current step/`total_curriculum_step`)**(1/`root_degree`)) * (`max_difficulty` - `min_difficulty`) + `min_difficulty`. Thus `fixed_linear` is basically a special case of `fixed_root` with `root_degree` as 1. In our (limited) study, we find the `fixed_root` schedule does not provide any clear advantage over `fixed_linear` schedule, while requiring one additional parameter.

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

We provide example scripts under [DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/curriculum_learning/](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM-v1.1.5-ZeRO3/curriculum_learning). The `ds_train.sh` is the training script to run and it also includes the actual configurations we used for the experiments in our [paper](https://arxiv.org/abs/2108.06084).

Besides the additional DeepSpeed configurations, there are some other necessary changes on the user side to enable curriculum learning. First, it is necessary to add a `curriculum_seqlen` argument in the model's forward pass and use it to perform training data sequence length truncation. For Megatron-LM GPT-2 pre-training, we implement this in `forward()` in [DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/megatron/model/gpt2_model.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-ZeRO3/megatron/model/gpt2_model.py) and in `forward_step()` in [DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/pretrain_gpt2.py](https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-ZeRO3/pretrain_gpt2.py).

Second, since there will be less tokens per step during curriculum learning, for curriculum-learning based training it requires more steps in order to reach the same number of training tokens as baseline. Thus in Megatron-LM we add a `--train-tokens` argument to terminate the training based on number of tokens. Then we usually set a long enough `--train-iters` (e.g., two times of baseline's total training step), and set the `--train-tokens` the same for baseline and curriculum-learning based training.

Third, again due to the less tokens per step during curriculum learning, we find that for curriculum-learning based training it is beneficial to increase the learning rate decay steps (otherwise the curriculum learning case will have faster token-wise learning rate decay than baseline). For `fixed_linear` schedule because we start from very short sequence length, the total number of tokens during the curriculum learning is roughly halved. Thus we usually just add half of `fixed_linear` schedule's `total_curriculum_step` to the Megatron-LM's `--lr-decay-iters`.
