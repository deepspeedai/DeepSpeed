---
title: "DeepSpeed Transformer Kernel"
tags: training
---

This tutorial shows how to enable the DeepSpeed transformer kernel and set its different configuration parameters.

## DeepSpeed Transformer Kernel
Transformer layers are ubiquitous in many recent sequence-processing models,
such as Natural-Language-Processing. Thus, training transformer-based networks
requires to be highly efficient in term of performance, in order to allow scientists to
explore different models across various application domains in a reasonable amount of time.
To this end, we have developed a new kernel for transformer networks which includes several
optimizations specific to these layers, which boost the training throughput on single GPU and scales
well as we increase the number of GPUs. For more information on the details
of transformer kernel, please visit our recent blog post on the [fastest BERT
training](https://www.deepspeed.ai/2020/05/27/fastest-bert-training.html).

## Prerequisites

To use transformer kernel for training a model, you should Integrate DeepSpeed into your training script using the [Getting Started](/getting-started/) guide.

**Note:** Currently DeepSpeed Transformer Kernels do not support Sparse Attention. To use Sparse Attention, you need to disable Transformer Kernels!
{: .notice--warning}

### **Integrate Transformer Kernel**

First of all, you need to integrate transformer kernel into the top-level model. Here, we show an example of instantiating the transformer kernel using the Pre-LN BERT-Large configuration settings. This configuration has 24 layers with 1024 hidden-dimension and uses the sequence length of 128 and batch size of 64. To add all these layers, we copy the same layer specification `num_hidden_layer`  times with different IDs inside a ModuleList.

```python
config = DeepSpeedTransformerConfig(batch_size = 64,
                                    max_seq_length = 128,
                                    hidden_size = 1024,
                                    heads = 16,
                                    attn_dropout_ratio = 0.1,
                                    hidden_dropout_ratio = 0.1,
                                    num_hidden_layers = 24,
                                    initializer_range = 0.02,
                                    local_rank = 0,
                                    seed = 1234,
                                    fp16 = True,
                                    pre_layer_norm=True,
                                    attn_dropout_checkpoint=False,
                                    normalize_invertible=False,
                                    gelu_checkpoint=False)
self.layer = nn.ModuleList([
    copy.deepcopy(DeepSpeedTransformerLayer(cuda_config))
    for _ in range(config.num_hidden_layers)
])
```
### Transformer kernel Parameters

The transformer kernel is configured by a number of parameters which allow users to
explore different settings. We partition these parameters into four categories:

1. General configuration, used by different types of transformer layers
2. Environment parameters, specifying the system's setting
3. High-performance flag, optimizing training with the stochastic computation
4. Memory optimization flags, trade off computing power for memory

The general parameters for configuring the transformer kernel are:

1. `batch_size`: The micro-batch size used for running the kernel on each GPU
2. `max_seq_length`: The sequence-length of the model being trained with DeepSpeed
3. `hidden_size`: The hidden size of the transformer layer
4. `heads`: The number of heads in the self-attention of the transformer layer
5. `attn_dropout_ratio`: The ratio of dropout for the attention's output
6. `hidden_dropout_ratio`: The ratio of dropout for the transformer's output
7. `num_hidden_layers`: The number of transformer layers
8. `pre_layer_norm`: Select between Pre-LN or Post-LN transformer architecture

The environment parameters of the transformer kernel includes:

1. `local_rank`: The rank of the current GPU running the transformer kernel
2. `seed`: The random seed for the dropout layer
3. `fp16`: Enable half-precision computation
4. `initializer_range`: BERT's initializer range

High-performance optimization flag:

1. `stochastic_mode`: By turning on this flag, the training can run faster by 2% on average. Note, that this flag has some level of non-determinism and can produce different results on different runs. However, we have seen that by enabling it, the pre-training tasks such as BERT are not affected and can obtain a high accuracy level. On the other hand, for the downstream tasks, such as fine-tuning, we recommend to turn it off in order to be able to reproduce the same result through the regular kernel execution.

The memory-optimization flags consist of:

1. `attn_dropout_checkpoint`: Enable checkpointing of attention dropout to save memory
2. `normalize_invertible`: Enable invertible LayerNorm execution (dropping the input activation)
3. `gelu_checkpoint`: Enable checkpointing of Gelu activation output to save memory

To illustrate the required model configuration changes to use transformer kernel in model training, we use a BERT model and go through the different configurations in order to support the different sequence lengths and batch sizes. Please see the instruction at [BERT training tutorial](/tutorials/bert-pretraining/).

### **Memory Optimization Flags**

We provide several techniques into the transformer kernel which saves the memory at different parts of a layer. We expose them as the configurable settings that can be enabled when calling the kernel. By turning on each of these optimization flags, we can support larger batch sizes. Even though we trade off performance for memory using some of these techniques, the end-to-end training efficiency increases by using the larger batch size.

By setting the `normalize_invertible` flag, we force the kernel to drop the input activations to the normalize layers of transformer. We can do this since the kernel includes an optimization to compute the gradients of the parameters and the input to this layer by only using the output activations.

The `attn_dropout_checkpoint` and `gelu_checkpoint` flags refer to the checkpointing approach, in which we drop the inputs to some parts of the transformer layer, attention dropout and Gelu, in order to save an important part of the activation memory. Based on our performance profiling, the performance cost of rematerializing these two are negligible and finally the performance benefit that we gain from running larger batch size compensate for that.

The following table shows which memory optimization flags need to be turned on when running BERT-Large on NVIDIA V100 GPU with 32GB of memory, considering different micro-batch sizes and sequence lengths. For the two sequence lengths, 128 and 512, used in our experiments, we have seen that larger batch size improves the overall training performance for both. Please see our [blog post](https://www.deepspeed.ai/2020/05/27/fastest-bert-training.html) for more information regarding the performance evaluation of these configurations.

| Micro-batch size |    128 sequence-length    |           512 sequence-length            |
| :--------------: | :-----------------------: | :--------------------------------------: |
|       > 12       |             -             |        `attn_dropout_checkpoint`         |
|       > 16       |             -             | `normalize_invertible`, `gelu_checkpoint`|
|       > 80       |  `normalize_invertible`   |                   OOM                    |
|      > 112       | `attn_dropout_checkpoint` |                   OOM                    |
|      > 128       |     `gelu_checkpoint`     |                   OOM                    |


### **Enable Transformer Kernel**

As mentioned earlier, in order to run the transformer network using the custom DeepSpeed kernel, we only need to pass the `deepspeed_transformer_kernel` option when running the training script. Below, we show an example of how we pass this parameter to the `deepspeed` launcher, besides the rest of parameters for the BERT pre-training task.

```bash
deepspeed deepspeed_train.py \
--cf bert_large_lamb.json \
--max_seq_length 512 \
--print_steps 100 \
--deepspeed \
--deepspeed_transformer_kernel \
--deepspeed_config deepspeed_bsz32K_lamb_config_seq512.json \
--rewarmup \
--lr_schedule "EE" \
--lr_offset 0.0 \
--attention_dropout_checkpoint \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_EPOCH150_NAME}

```
In addition to transformer kernel flag, we can specify the memory optimization settings as discussed earlier. As an example, we use the `attention_dropout_checkpoint` option here for running the sequence length 512, in order to run the micro-batch size of 16 at each GPU. If larger batch size is required, we can turn on the rest of memory optimization flags too.
