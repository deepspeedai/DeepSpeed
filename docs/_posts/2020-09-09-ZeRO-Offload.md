---
title: "10x bigger model training on a single GPU with ZeRO-Offload"
excerpt: ""
date: 2020-09-09 00:00:00
tags: training ZeRO English
toc: false
---

We introduce a new technology called ZeRO-Offload to enable **10X bigger model training on a single GPU**. ZeRO-Offload extends ZeRO-2 to leverage both CPU and GPU memory for training large models. Using a machine with **a single GPU**, our users now can run **models of up to 13 billion parameters** without running out of memory, 10x bigger than the existing approaches, while obtaining competitive throughput. This feature democratizes multi-billion-parameter model training and opens the window for many deep learning practitioners to explore bigger and better models.

* For more information on ZeRO-Offload, see our [press release]( {{ site.press_release_v3 }} ).
* For more information on how to use ZeRO-Offload, see our [ZeRO-Offload tutorial](https://www.deepspeed.ai/tutorials/ZeRO-offload/).
* The source code for ZeRO-Offload can be found in the [DeepSpeed repo](https://github.com/microsoft/deepspeed).
