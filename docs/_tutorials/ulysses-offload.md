---
title: "DeepSpeed Ulysses-Offload"
tags: training ultra long context language model with fully pipelined distributed transformer
---

DeepSpeed Ulysses-Offload is a system of chunking and offloading long-context transformer model training scheme built on top of [ZeRO](/tutorials/zero/) and [DeepSpeed Ulysses](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md). It adopts Fully Pipeliend Distributed Transformer (FPDT) which enables 2M context size training on 8B models with only 4 GPUs, and 4M context size training on 70B models with 32 GPUs. Read our [Ulysses-Offload blog](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) and [paper](https://arxiv.org/pdf/2408.16978) to learn more!

We recommend that you read the tutorials on [Getting Started](/getting-started/), [ZeRO](/tutorials/zero/)  and [Megatron-DeepSpeed](/tutorials/megatron/) before stepping through this tutorial.


## Design of Ulysses-Offload
Ulysses-Offload is a chunking and offloading-based transformer implementation, which retain the full precision of the vanilla transformer, while significantly reduce the activation memory required during long-context model training. FPDT breaks long sequence input into smaller chunks, moving them among host and GPU memory to achieve the superior memory efficiency while reaching over 50% of MFU. FPDT adopts a double-buffer design, which overlaps the fetching/offloading with the attention computation. FPDT also allows uUsers to configure the chunk size to match the expected memory budget.

Ulysses-Offload supports ZeRO, which shards the model and tensors among GPU memory, further pushing the limit of long-context model training with state-of-the-art hardware efficiency.


## Training Environment

For this tutorial, Flash Attention (CUDA) is required. We will configure a 8 billion parameter LLaMA model using the DeepSpeed [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/master/) code. We will use 1 nodes of 4x [NVIDIA Tesla A100-SXM4 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/).


## Training a 6.7B parameter GPT with Ulysses-Offload
Users can set the context size at the beginning of the script, for this exercise, we will use 256K context and mini batch of one.
```
### Main configs
seq_len=262144 # need to be power of 2
```

For 6.7B model, we will enable ZeRO-3, Ulysses, activation checkpointing with CPU offloading first reach a decent GPU memory efficiency, then users can configure the following arguments:

 - ds_sequence_parallel_fpdt: Boolean indicating whether to use FPDT, default is false.
 - ds_sequence_parallel_fpdt_chunk_size: Integer indicating the chunk size in FPDT, default is 65536, meaning no matter how long the sequence is, FPDT will always process chunks of 65536 tokens until the entire sequence is all processed.
 - ds_sequence_parallel_fpdt_offloading: Boolean indicating whether to use host memory to offload chunks, default is false.


### Megatron-DeepSpeed Configuration Changes

1. An example snippet of megatron-deepspeed configurations with all Ulysses-Offload features enable is shown below:
    ```
    megatron_options="\
    --ds-sequence-parallel-fpdt \
    --ds-sequence-parallel-fpdt-chunk-size 65536 \
    --ds-sequence-parallel-fpdt-offloading \
    --ds-sequence-parallel-size 4"
    ```

2. FPDT requires Flash Attention, and also supports Rotary Position Embedding (RoPE):
    ```
    --use-flash-attn-v2 \
    --use-rotary-position-embeddings \
    --rotary-percent 0.25 \
    --rotary-position-embeddings-theta 100000000 \
    ```

3. We also enable CPU checkpointing to reduce activation memory footprints:
    ```
    if [ "${activation_checkpoint}" = "true" ]; then
    deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing \
        --checkpoint-in-cpu"
    fi
    ```

You can find the full script [here](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/main/examples_deepspeed/sequence_parallel/ds_pretrain_gpt_6.7B_fpdt_32k.sh).

See more details on Megatron-DeepSpeed [tutorial](/tutorials/megatron/) examples on how to launch a Megatron-DeepSpeed job.

Congratulations! You have completed the Ulysses-Offload tutorial.
