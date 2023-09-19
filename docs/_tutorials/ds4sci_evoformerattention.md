---
title: "DS4Sci_EvoformerAttention eliminates memory explosion problems for scaling Evoformer-centric structural biology models"
tags: training inference
---

## 1. What is DS4Sci_EvoformerAttention
`DS4Sci_EvoformerAttention` is a collection of kernels built to scale the [Evoformer](https://www.nature.com/articles/s41586-021-03819-2) computation to larger number of sequences and residuals by reducing the memory footprint and increasing the training speed.

## 2. When to use DS4Sci_EvoformerAttention
`DS4Sci_EvoformerAttention` is most beneficial when the number of sequences and residuals is large. The forward kernel is optimized to accelerate computation. It is beneficial to use the forward kernel during inference for various attention mechanisms. The associated backward kernel can be used during training to reduce the memory footprint at the cost of some computation. Therefore, it is beneficial to use `DS4Sci_EvoformerAttention` in training for memory-constrained operations such as MSA row-wise attention and MSA column-wise attention.

## 3. How to use DS4Sci_EvoformerAttention

### 3.1 Installation

`DS4Sci_EvoformerAttention` is released as part of DeepSpeed >= 0.10.3. `DS4Sci_EvoformerAttention` is implemented based on [CUTLASS](https://github.com/NVIDIA/cutlass). You need to clone the CUTLASS repository and specify the path to it in the environment variable `CUTLASS_PATH`.

```shell
git clone https://github.com/NVIDIA/cutlass
export CUTLASS_PATH=/path/to/cutlass
```
The kernels will be compiled when `DS4Sci_EvoformerAttention` is called for the first time.

`DS4Sci_EvoformerAttention` requires GPUs with compute capability 7.0 or higher (NVIDIA V100 or later GPUs) and the minimal CUDA version is 11.3. It is recommended to use CUDA 11.7 or later for better performance. Besides, the performance of backward kernel on V100 kernel is not as good as that on A100 for now.

### 3.2 Unit test and benchmark

The unit test and benchmark are available in the `tests` folder in DeepSpeed repo. You can use the following command to run the unit test and benchmark.

```shell
pytest -s tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py
python tests/benchmarks/DS4Sci_EvoformerAttention_bench.py
```

### 3.3 Applying DS4Sci_EvoformerAttention to your own model

To use `DS4Sci_EvoformerAttention` in user's own models, you need to import `DS4Sci_EvoformerAttention` from `deepspeed.ops.deepspeed4science`.

```python
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
```

`DS4Sci_EvoformerAttention` supports four attention mechanisms in Evoformer (MSA row-wise, MSA column-wise, and 2 kinds of Triangular) by using different inputs as shown in the following examples. In the examples, we denote the number of sequences as `N_seq` and the number of residuals as `N_res`. The dimension of the hidden states `Dim` and head number `Head` are different among different attention. Note that `DS4Sci_EvoformerAttention` requires the input tensors to be in `torch.float16` or `torch.bfloat16` data type.

(a) **MSA row-wise attention** builds attention weights for residue pairs and integrates the information from the pair representation as an additional bias term.
```python
# Q, K, V: [Batch, N_seq, N_res, Head, Dim]
# res_mask: [Batch, N_seq, 1, 1, N_res]
# pair_bias: [Batch, 1, Head, N_res, N_res]
out = DS4Sci_EvoformerAttention(Q, K, V, [res_mask, pair_bias])
```

(b) **MSA column-wise attention** lets the elements that belong to the same target residue exchange information.
```python
# Q, K, V: [Batch, N_res, N_seq, Head, Dim]
# res_mask: [Batch, N_seq, 1, 1, N_res]
out = DS4Sci_EvoformerAttention(Q, K, V, [res_mask])
```

(c) **Triangular self-attention** updates the pair representation. There are two kinds of Triangular self-attention: around starting and around ending node. Below is the example of triangular self-attention around starting node. The triangular self-attention around ending node is similar.
```python
# Q, K, V: [Batch, N_res, N_res, Head, Dim]
# res_mask: [Batch, N_res, 1, 1, N_res]
# right_edges: [Batch, 1, Head, N_res, N_res]
out = DS4Sci_EvoformerAttention(Q, K, V, [res_mask, right_edges])
```

## 4. DS4Sci_EvoformerAttention scientific application

### 4.1 DS4Sci_EvoformerAttention eliminates memory explosion problems for scaling Evoformer-centric structural biology models in OpenFold

[OpenFold](https://github.com/aqlaboratory/openfold) is a community reproduction of DeepMind's AlphaFold2 that makes it possible to train or finetune AlphaFold2 on new datasets. Training AlphaFold2 incurs a memory explosion problem because it contains several custom Evoformer attention variants that manifest unusually large activations. By leveraging DeepSpeed4Science's DS4Sci_EvoformerAttention kernels, OpenFold team is able to reduce the peak memory requirement by 13x without accuracy loss. Detailed information about the methodology can be found at [our website](https://deepspeed4science.ai/2023/09/18/model-showcase-openfold/).

<!-- OpenFold team also hosts an [example](https://github.com/aqlaboratory/openfold/blob/main/tests/test_deepspeed_evo_attention.py) about how to use DS4Sci_EvoformerAttention in the OpenFold repo. -->
