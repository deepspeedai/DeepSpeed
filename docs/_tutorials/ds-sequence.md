---
title: "Getting Started with DeepSpeed-Ulysses for Training Transformer Models with Extreme Long Sequences"
tags: training
---

In this tutorial we describe how to enable DeepSpeed-Ulysses. DeepSpeed-Ulysses is a simple but highly communication and memory efficient mechanism sequence parallelism approach for training of large transformer models with massive sequence lengths. It partitions input tensors along the sequence dimension and uses a communication-efficient all-2-all collective for distributed attention computations. Additionally, DeepSpeed-Ulysses incorporates advanced modeling and system optimizations, such as Flash attention, sparse attention, and ZeRO optimizer, to optimize both computational efficiency and memory usage. Training with DeepSpeed sequence parallelism allows both model size and sequence length to scale near indefinitely unbounded by single GPU memory limitation and at a high fraction of peak compute performance. Currently, DeepSpeed-Ulysses can handle sequences up to 1 million in length (10 times the size of a complete Harry Potter book!) on 64 A100 GPUs. Please read our [DeepSpeed-Ulysses blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses) to learn more!

## 1. Installation

You will need to install DeepSpeed v0.10.2 or higher to use the DeepSpeed Sequence feature. Installing DeepSpeed is as simple as `pip install deepspeed`, [see more details](/tutorials/getting-started/).


## 2. How to use DeepSpeed-Ulysses in your application?

Integrating DS-Seq into your training code is easy, and in this section we describe how to integrate DeepSpeed-Ulysses through our [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) code repo.


* **Replace attention module**: First, you need to update your attention module with DeepSpeed-Ulysses DistributedAttention. Here, we use the attention from [Megatron-DeepSpeed ](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/transformer.py) which is the causal attention used in GPT-3 like model training. Rewrite the attention block:

```python
def __init__():
    ...
    self.local_attn = CoreAttention(self.layer_number, config, self.attn_mask_type)
    self.core_attention = local_attn
    ...

def forward():
    ...
    context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask)
    ...
```

with:

```python
from deepspeed.sequence.layer import DistributedAttention

def __init__():
    ...
    self.local_attn = CoreAttention(self.layer_number, config, self.attn_mask_type)
    self.dist_attn = DistributedAttention(self.local_attn, parallel_state.get_sequence_parallel_group())
    ...

def forward():
    ...
    context_layer = self.dist_attn(query_layer, key_layer, value_layer, attention_mask)
    ...

```

* **Add sequence parallel communication group**:  Note that DistributedAttention takes `local_attn` and `sequence_parallel_group` as the parameters, where local_attn can be your original attention block. You also need to build the sequence parallel nication group and pass that the DistributedAttention. One way to do this is to build the sequence parallel group at the model initialization stage.


```python
def initialize_model_parallel(
    ...
    sequence_parallel_size,
    ...
):
    ...
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size
    ...
    global _SEQUENCE_PARALLEL_GROUP
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP

```

In the Megatron-DeepSpeed exampele, to enable sequence parallelism, set the degree of parallelism using the --ds-sequence-parallel-size argument. You also need to ensure that the number of attention heads is divisible by this value.
We have prepared scripts for you to quickly get some examples for training GPT-3 like models with very long sequences:

```shell
Megatron-DeepSpeed/examples_deepspeed/sequence_parallel$ bash ds_pretrain_gpt_1.3B_seq_parallel_32k.sh
Megatron-DeepSpeed/examples_deepspeed/sequence_parallel$ bash ds_pretrain_gpt_30B_seq_parallel_32k.sh
```

Please note that our sequence parallelism feature is currently incompatible with Megatron-LM's tensor or pipeline parallelism.

## 3. Enabling DeepSpeed-Ulysses with FlashAttention?

DeepSpeed's sequence parallelism can be combined with different types of attention implementations to further improve the memory and compute efficiency of long sequence training:

`Classic attention`: attention mechanism implemented via PyTorch.

`FlashAttention`: the implementation from [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). Enabled by `--use-flash-attn`.

`FlashAttention + Triton`: a of FlashAttention in Triton (tested with triton==2.0.0.dev20221202). Enabled by `--use-flash-attn-triton`.

For the best performance, we recommend using FlashAttention + Triton. Below are the installation steps. Note that FlashAttention is compatible only with NVIDIA Turing, Ampere, Ada, or Hopper GPUs.

```bash
# install triton
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake
pip install .
```

```bash
# install
cd ${WORK_DIR}
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python setup.py install
```

You may also want to ensure your model configuration is compliant with FlashAttention's requirements. For instance, to achieve optimal performance, the head size should be divisible by 8. Refer to the document of FlashAttention for more details.
