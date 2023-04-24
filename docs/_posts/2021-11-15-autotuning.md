---
title: "Autotuning: Automatically discover the optimal DeepSpeed configuration that delivers good training speed"
excerpt: ""
date: 2021-11-16 10:00:00
tags: training English
toc: false
---

We introduce a new feature called Autotuning to automatically discover the optimal DeepSpeed configuration that delivers good training speed. One pain point in model training is to figure out good performance-relevant configurations such as micro-batch size to fully utilize the hardware and achieve a high throughput number. This configuration exploring process is commonly done manually but is important since model training is repeated many times and benefits from using a good configuration. Not only is the hand-tuning process time-consuming, but the outcome is hardware-dependent. This means that a good configuration on one hardware might not be the best on another different hardware. The user thus has to hand tune the configuration again. With DeepSpeed, there are more configuration parameters that could potentially affect the training speed, thus making it more tedious to manually tune the configuration.

The DeepSpeed Autotuner mitigates this pain point and automatically discovers the optimal DeepSpeed configuration that delivers good training speed. It not only reduces the time and resources users spend on tuning, but also can discover configurations better than hand-tuned methods. [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/autotuning) would demonstrate the effectiveness of autotuning across different models.

* For a brief overview, see the [Autotuning tutorial](https://www.deepspeed.ai/tutorials/autotuning/).
* For more information on how to use Autotuning, see the [Autotuning README](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning#deepspeed-autotuning).
* The source code can be found in the [DeepSpeed repo](https://github.com/microsoft/deepspeed).
