---
title: "Mixture of Experts for NLG models"
---

In this tutorial, we introduce how to apply DeepSpeed Mixture of Experts (MoE) to NLG models, which reduces the training cost by 5 times and reduce the MoE model size by 3 times (details in our [Blog]({{ site.press_release_v6 }})). We use the GPT-3 like models in Megatron-LM framework as the example. Before reading this tutorial, we recommend to first read the tutorials about [Mixture of Experts](/tutorials/mixture-of-experts/) and [Megatron-LM GPT pre-training](/tutorials/megatron/).

## 1. Installation

You would need to install DeepSpeed v0.6.0 or higher to use the MoE feature. The MoE for NLG model examples are in the [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) repo (currently under [the moe branch](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe) but later could be merged to main branch).

## 2. Training NLG+MoE models

### 2.1. Changes to the model
To apply MoE to the GPT-style model, we made several changes in Megatron framework, mostly in `megatron/model/` where we add the MoE layers into the model.

### 2.2. Pre-training the Standard MoE model
We provide example training scripts under [examples/MoE](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe/examples/MoE) which we used to perform the experiments in our [Blog]({{ site.press_release_v6 }}). There are a few new hyperparameters for standard MoE model:

`--num-experts`: the number of experts per MoE layer. In our experiments we set it to 128. Larger number of experts tend to provide better convergence, but it's a diminishing return.

`--moe-expert-parallel-size`: degree of the MoE expert parallelism. In other words, there will be `num-experts/moe-expert-parallel-size` experts on each GPU. Thus `--moe-expert-parallel-size` should be no more than both number of GPUs, and `--num-experts`.

`--moe-loss-coeff`: scaling coefficient for adding MoE loss to model loss. In our experiments we find that 0.01 is a good setting.

`--moe-train-capacity-factor`, `--moe-eval-capacity-factor`, `--moe-min-capacity`: these configs determine how many tokens can a single expert handle. Larger numbers could lead to better convergence, but would also lead to slower training since the load would be more unbalanced on different experts.

`--disable-moe-token-dropping`: this will completely remove the limitation of how many tokens can a single expert handle. For the same reason as above, we only recommend using this during inference/eval.



### 2.3. Pre-training the PR-MoE model
PR-MoE is a new designed MoE models, standing for Pyramid-Residual-MoE, which improves the parameter efficiency up to 3x as compared to standard MoE. Please see our [Blog]({{ site.press_release_v6 }}) for more details. We provide example training scripts under [examples/MoE](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe/examples/MoE). There are a few different hyperparameters for PR-MoE model compared to standard MoE:

`--num-experts`: Instead of providing a single number, to enable Pyramid-MoE, you need to provide a list, whose length is the same as the number of MoE layers. We suggest to use more experts in the latter stage (close to output) of the model.

`--mlp-type`: chosen from `[standard, residual]`. When it is residual, Residual-MoE is enabled.

In addition to the new hyperparameters above for standard MoE and PR-MoE, for NLG+MoE models we found that it's helpful to lower the learning rate and increase the learning rate decay duration compared to the base dense model. Details of our tuning can be found in the example training scripts.

Regarding training data, we are not able to release our internal data but any public data for Megatron-LM pre-training can be directly used to train MoE models (with the caveat that it might not provide the exact same model quality as in our experiments). For example, we evaluated The Pile dataset ([pile.eleuther.ai](https://pile.eleuther.ai/), [github.com/EleutherAI/the-pile](https://github.com/EleutherAI/the-pile)) for both dense and MoE models. Table 1 below shows that this public data provides similar evaluation results as our internal data.

| Model size | LAMBADA: completion prediction | PIQA: commonsense reasoning | BoolQ: reading comprehension | RACE-h: reading comprehension | TriviaQA: question answering | WebQs: question answering |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Dense NLG:** | | | | | | |
| 350M, internal data | 0.5203 | 0.6931 | 0.5364 | 0.3177 | 0.0321 | 0.0157 |
| 350M, public Pile | 0.5106 | 0.6589 | 0.5933 | 0.3196 | 0.0257 | 0.0064 |
| **Standard MoE NLG:** | | | | | | |
| 350M+MoE-128, internal data | 0.6270 | 0.7459 | 0.6046 | 0.3560 | 0.1658 | 0.0517 |
| 350M+MoE-128, public Pile | 0.6128 | 0.7323 | 0.6040 | 0.3349 | 0.1111 | 0.0335 |
| **PR-MoE NLG:** | | | | | | |
| 350M+MoE-128, internal data | 0.6365 | 0.7399 | 0.5988 | 0.3569 | 0.1630 | 0.0473 |

Table 1: Zero-shot evaluation results (last six columns) for different dense and MoE NLG models. All zero-shot evaluation results use the accuracy metric.
