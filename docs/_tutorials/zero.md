---
title: "Zero Redundancy Optimizer (ZeRO)"
---
If you have not done so already, we advise that you read the DeepSpeed tutorials on [Getting Started](/getting-started/) and [Megatron-LM GPT-2](/tutorials/megatron/) before stepping through this tutorial.

In this tutorial, we will apply the ZeRO optimizer to the [Megatron-LM GPT-2](https://github.com/NVIDIA/Megatron-LM) model. ZeRO is a powerful set of memory optimization techniques that enable effective FP16 training of large models with trillions of parameters, such as [GPT-2](https://openai.com/blog/better-language-models/) and [Turing-NLG 17B](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/). Compared to the alternative model parallelism approaches for training large models, a key appeal of ZeRO is that no model code modifications are required. As this tutorial will demonstrate, *using ZeRO in a DeepSpeed model is quick and easy because all you need is to change a few configurations in the DeepSpeed configuration JSON*. No code changes are needed.

## ZeRO Overview
ZeRO leverages the aggregate computation and memory resources of data parallelism to reduce the memory and compute requirements of each device (GPU) used for model training. ZeRO reduces the memory consumption of each GPU by partitioning the various model training states (weights, gradients, and optimizer states) across the available devices (GPUs and CPUs) in the distributed training hardware. Concretely, ZeRO is being implemented as incremental stages of optimizations, where optimizations in earlier stages are available in the later stages. To deep dive into ZeRO, please see our [paper](https://arxiv.org/abs/1910.02054v3).

* **Stage 1**: The optimizer states (e.g., for [Adam optimizer](https://arxiv.org/abs/1412.6980), 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.

* **Stage 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.

* **Stage 3**: The 16-bit model parameters are partitioned across the processes. ZeRO will automatically collect and partition them during the forward and backward passes.

## Training environment
We use the DeepSpeed [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM) GPT-2 code for this exercise. You can step through the Megatron-LM [tutorial](/tutorials/megatron/) to familiarize yourself with the code. We will train the models in this tutorial on [NVIDIA Tesla V100-SXM3 Tensor Core GPUs](https://www.nvidia.com/en-us/data-center/v100/) with 32GB RAM.

## Enabling ZeRO Optimization
To enable ZeRO optimizations for a DeepSpeed model, we simply add the **_zero_optimization_** key to the DeepSpeed JSON configuration. A full description of configuration knobs of the **zero_optimization** key is available [here](/docs/config-json/#zero-optimizations-for-fp16-training).

### Training a 1.5B Parameter GPT-2 model
We demonstrate the benefits of ZeRO stage 1 by showing that it enables data parallel training of a 1.5 billion parameter GPT-2 model on eight V100 GPUs. We configure training to use a batch size of 1 per device to ensure that the memory consumption is primarily due to model parameters and optimizer states. We create this training scenario by applying the following modifications to the deepspeed launch script:

```bash
       --model-parallel-size 1 \
       --num-layers 48 \
       --hidden-size 1600 \
       --num-attention-heads 16 \
       --batch-size 1 \
       --deepspeed_config ds_zero_stage_1.config \
```

Training this model without ZeRO fails with an out-of-memory (OOM) error as shown below:

<a href="/assets/images/oom_dp8_1.5B_log.png">
<img src="/assets/images/oom_dp8_1.5B_log.png">
</a>

A key reason why this model does not fit in GPU memory is that the Adam optimizer states for the model consume 18GB; a significant portion of the 32GB RAM. By using ZeRO stage 1 to partition the optimizer state among eight data parallel ranks, the per-device memory consumption can be reduced to 2.25GB, thus making the model trainable. To enable ZeRO stage 1, we simply update the DeepSpeed JSON config file as below:

```json
{
    "zero_optimization": {
        "stage":1,
        "reduce_bucket_size": 5e8
    }
}
```
As seen above, we set two fields in the **zero_optimization** key. Specifically we set the _stage_ field to 1, and the optional _reduce_bucket_size_ for gradient reduction to 500M. With ZeRO stage 1 enabled, the model can now train smoothly on 8 GPUs without running out of memory.   Below we provide some screenshots of the model training:


<a href="/assets/images/zero1_dp8_1.5B_log.png">
<img src="/assets/images/zero1_dp8_1.5B_log.png">
</a>

<a href="/assets/images/zero1_dp8_1.5B_smi.png">
<img src="/assets/images/zero1_dp8_1.5B_smi.png">
</a>


From the nvidia-smi screenshot above we can see that only GPUs 6-7 are being used for training the model. With ZeRO stage 1 we can further reduce the per-device memory consumption by increasing the data parallelism degree. These memory savings can be leveraged to either increase model size and/or batch size. In contrast, such benefits are not possible with data parallelism alone.

### Training a 10B Parameter GPT-2 model
ZeRO stage 2 optimizations further increases the size of models that can be trained using data parallelism. We show this by training a model with 10B parameters using 32 V100 GPUs.

First, we need to configure a 10B parameter model with activation checkpointing enabled. This can be done by applying the following GPT-2 model configuration changes to the DeepSpeed launch script.

```bash
       --model-parallel-size 1 \
       --num-layers 50 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --batch-size 1 \
       --deepspeed_config ds_zero_stage_2.config \
       --checkpoint-activations
```

Next, we need to update the DeepSpeed JSON configuration, as shown below, to enable ZeRO stage 2 optimizations:

```json
{
    "zero_optimization": {
        "stage":2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

In the above changes, we have set the _stage_ field to 2, and configured other optimization knobs that are available in ZeRO stage 2. For example, we have enabled _contiguous_gradients_ to reduce memory fragmentation during backward pass. A full description of these optimization knobs is available [here](/docs/config-json/#zero-optimizations-for-fp16-training). With these changes, we can now launch the training run.

Here is a screenshot of the training log:

<a href="/assets/images/zero2_dp32_10B_log.png">
<img src="/assets/images/zero2_dp32_10B_log.png">
</a>

Here is a screenshot of nvidia-smi showing GPU activity during training:

<a href="/assets/images/zero2_dp32_10B_smi.png">
<img src="/assets/images/zero2_dp32_10B_smi.png">
</a>

### Training trillion-scale models with ZeRO-3 Offload

Stage 3 can be enabled in the JSON configuration. A full description of these
configurations is available [here](/docs/config-json/#zero-optimizations-for-fp16-training).

```json
{
  "zero_optimization": {
    "stage": 3,
    "cpu_offload": true,
    "cpu_offload_params": true,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 6000000,
    "stage3_max_reuse_distance": 100000000,
    "stage3_prefetch_bucket_size": 200000,
    "stage3_param_persitance_threshold": 100000,
    "reduce_bucket_size": 3000000,
    "sub_group_size": 1e6
  }
}
```


ZeRO-3 will automatically collect and partition the parameters as they are
needed during the forward and backward passes. However, in some cases a
parameter may be used outside of its module's forward pass. We call these
*external parameters*. ZeRO-3 can coordinate these parameters if they are
registered. Please see our [ZeRO-3 docs](https://deepspeed.readthedocs.io/en/latest/zero3.html) for more
information and examples of external parameters.

The Megatron-LM model has three external parameters that must be registered
with ZeRO-3. External parameters are those that are accessed outside of the
owning module's forward pass.

1. `megatron/model/gpt2_model.py:GPT2Model`: register the word embedding for both uses in forward.

```python
    class GPT2Model(MegatronModule):
    def __init__(self, num_tokentypes=0, parallel_output=True):
        ...
        deepspeed.zero.register_external_parameter(self,
                                                   self.language_model.embedding.word_embeddings.weight)


    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None):
        # self.embeddings will compute its forward pass here
        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past,
                                        get_key_value=get_key_value)
        ...

        # Accesses word_embeddings.weight outside of the embedding's forward pass.
        output = parallel_lm_logits(
            lm_output,
            self.language_model.embedding.word_embeddings.weight,
            parallel_output)
```

2. `megatron/model/transformer.py:ParallelMLP`: register a bias that is
returned from a submodule forward and used in this forward.

```python
class ParallelMLP(MegatronModule):
    def __init__(self, init_method, output_layer_init_method):
        ...
        if self.dense_h_to_4h.bias is not None:
            deepspeed.zero.register_external_parameter(self, self.dense_h_to_4h.bias)

    def forward(self, hidden_states):

        # bias_parallel is a parameter of dense_h_to_4h

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        ...
```

3. `megatron/model/transformer.py:ParallelTransformerLayer`: register two biases that
are returned from submodules and used in forward.

```python
class ParallelTransformerLayer(MegatronModule):
    ...
    def __init__(self, attention_mask_func, init_method,
                 output_layer_init_method, layer_number):
        ...
        if self.attention.dense.bias is not None:
            deepspeed.zero.register_external_parameter(self, self.attention.dense.bias)
        if self.mlp.dense_4h_to_h.bias is not None:
            deepspeed.zero.register_external_parameter(self, self.mlp.dense_4h_to_h.bias)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        ...
        # attention_bias is a parameter returned from attention

        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value)

        ...

        # mlp_bias is a parameter returned from mlp
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        ...
```



#### Allocating Massive Megatron-LM Models

We make two further changes to model initialization in order to support models
that exceed *local* system memory, but not *total* system memory.

1. Allocate the model in a memory-scalable fashion. The model parameters will
be allocated and immediately partitioned across the data parallel group. If
`remote_device="cpu"`, the model will also be allocated in CPU memory
instead of GPU memory. Please see the full
[ZeRO-3 Init docs](https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.zero.Init)
for more details.

    ```python
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=get_args().remote_device,
                             enabled=get_args().zero_stage==3):
        model = GPT2Model(num_tokentypes=0, parallel_output=True)
    ```

2. Gather the position embeddings weight for initialization. DeepSpeed will automatically
gather a module's parameters during its constructor and for its forward and backward pass.
However, additional accesses must coordinate with DeepSpeed to ensure that parameter data
is gathered and subsequently partitioned. If the tensor is modified, the `modifier_rank`
argument should also be used to ensure all ranks have a consistent view of
the data. Please see the full
[GatheredParameters docs](https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.zero.GatheredParameters)
for more details.

    ```python
    self.position_embeddings = torch.nn.Embedding(...)
    with deepspeed.zero.GatheredParameters(self.position_embeddings.weight,
                                           modifier_rank=0):
        # Initialize the position embeddings.
        self.init_method(self.position_embeddings.weight)
    ```


Congratulations! You have completed the ZeRO tutorial.
