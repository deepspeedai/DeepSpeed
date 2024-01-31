---
title: "Inference Overview and Features"
layout: single
permalink: /inference/
toc: true
toc_label: "Contents"
---

>**DeepSpeed-Inference v2 is here and it's called DeepSpeed-FastGen! For the best performance, latest features, and newest model support please see our [DeepSpeed-FastGen release blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)!**

DeepSpeed-Inference introduces several features to efficiently serve transformer-based PyTorch models. It supports model parallelism (MP) to fit large models that would otherwise not fit in GPU memory. Even for smaller models, MP can be used to reduce latency for inference. To further reduce latency and cost, we introduce inference-customized kernels. Finally, we propose a novel approach to quantize models, called MoQ, to both shrink the model and reduce the inference cost at production. For more details on the inference related optimizations in DeepSpeed, please refer to our [blog post](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/).

DeepSpeed provides a seamless inference mode for compatible transformer based models trained using DeepSpeed, Megatron, and HuggingFace, meaning that we don’t require any change on the modeling side such as exporting the model or creating a different checkpoint from your trained checkpoints. To run inference on multi-GPU for compatible models, provide the model parallelism degree and the checkpoint information or the model which is already loaded from a checkpoint, and DeepSpeed will do the rest. It will automatically partition the model as necessary, inject compatible high performance kernels into your model and manage the inter-gpu communication. For list of compatible models please see [here](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py).

To get started with DeepSpeed-Inference, please checkout our [tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/).
