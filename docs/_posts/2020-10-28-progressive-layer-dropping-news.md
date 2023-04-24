---
title: "Progressive Layer Dropping"
excerpt: ""
date: 2020-10-29 00:00:00
tags: training English
toc: false
---

We introduce a new technology called progressive layer dropping (PLD) to speedup the pre-training of Transformer-based networks through efficient and robust compressed training. The pre-training step of Transformer networks often suffer from unbearable overall computational expenses. We analyze the training dynamics and stability of Transformer networks and propose PLD to sparsely update Transformer blocks following a progressive dropping schedule, which smoothly increases the layer dropping rate for each mini-batch as training evolves along both the temporal and the model depth dimension. PLD is able to allow the pre-training to be **2.5X faster** to get similar accuracy on downstream tasks and allows the training to be **24% faster** when training the same number of samples, not at the cost of excessive hardware resources.

  * For detailed technology deep dive, see our [technical report](https://arxiv.org/pdf/2010.13369.pdf).
  * For more information on how to use PLD, see our [Progressive layer dropping tutorial](https://www.deepspeed.ai/tutorials/progressive_layer_dropping/).
  * The source code for PLD is now available at the [DeepSpeed repo](https://github.com/microsoft/deepspeed).
