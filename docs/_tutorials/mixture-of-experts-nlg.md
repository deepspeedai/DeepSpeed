---
title: "Mixture of Experts for NLG models"
---

In this tutorial, we introduce how to apply DeepSpeed Mixture of Experts (MoE) to NLG models, which reduces the training cost by 5 times (details in our [Newsletter](https://www.deepspeed.ai/news/2021/12/09/deepspeed-moe-nlg.html)). We use the GPT-3 like models in Megatron-LM framework as the example. Before reading this tutorial, we recommend to first read the tutorials about [Mixture of Experts](/tutorials/mixture-of-experts/) and [Megatron-LM GPT pre-training](/tutorials/megatron/).

## 1. Installation

You would need to install DeepSpeed v0.5.8 or higher to use the MoE feature. The MoE for NLG model examples are in the [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) repo (currently under [the moe-training branch](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training) but later could be merged to main branch).

## 2. Training NLG+MoE models

### 2.1. Changes to the model
To apply MoE to the GPT-style model, we made several changes in Megatron framework, mostly in `megatron/model/` where we add the MoE layers into the model. Details of the code changes are at [this commit](https://github.com/microsoft/Megatron-DeepSpeed/commit/3c666e85b46ab26ef2dfadfdf7a18d186887856b).

### 2.2. Pre-training the model
We provide example training scripts under [examples/MoE](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training/examples/MoE) which we used to perform the experiments in our [Newsletter](https://www.deepspeed.ai/news/2021/12/09/deepspeed-moe-nlg.html). There are a few new hyperparameters for MoE model:

`--num-experts`: the number of experts per MoE layer. In our experiments we set it to 128. Larger number of experts tend to provide better convergence, but it's a diminishing return.

`--moe-expert-parallel-size`: degree of the MoE expert parallelism. In other words, there will be `num-experts/moe-expert-parallel-size` experts on each GPU. Thus `--moe-expert-parallel-size` should be no more than both number of GPUs, and `--num-experts`.

`--moe-loss-coeff`: scaling coefficient for adding MoE loss to model loss. In our experiments we find that 0.01 is a good setting.

`--moe-train-capacity-factor`, `--moe-eval-capacity-factor`, `--moe-min-capacity`: these configs determine how many tokens can a single expert handle. Larger numbers could lead to better convergence, but would also lead to slower training since the load would be more unbalanced on different experts.

`--disable-moe-token-dropping`: this will completely remove the limitation of how many tokens can a single expert handle. For the same reason as above, we only recommend using this during inference/eval.

In addition to the new hyperparameters above, for NLG+MoE models we found that it's helpful to lower the learning rate and increase the learning rate decay duration compared to the base dense model. Details of our tuning can be found in the example training scripts.

Regarding training data, we are not able to release our internal data but any public data for Megatron-LM pre-training (e.g., [The Pile dataset](https://the-eye.eu/public/AI/pile_neox/)) can be directly used to train MoE models (with the caveat that it might not provide the exact same model quality as in our experiments). We are evaluating public dataset for MoE pre-training and will post our findings here.
