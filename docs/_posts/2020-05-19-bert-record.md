---
layout: single
title: "The Fastest and Most Efficient BERT Training through Optimized Transformer Kernels"
excerpt: ""
categories: news
new_post: true
date: 2020-05-19 00:00:00
---

We introduce new technology to accelerate single GPU performance via kernel
optimizations. These optimizations not only create a strong foundation for
scaling out large models, but also improve the single GPU performance of
highly tuned and moderately sized models like BERT by more than 30%, reaching
a staggering performance of 66 teraflops per V100 GPU, which is 52% of the
hardware peak. **Using optimized transformer kernels as the building block,
DeepSpeed achieves the fastest BERT training record: 44 minutes on 1,024
NVIDIA V100 GPUs**, compared with the best published result of 67 minutes on
the same number and generation of GPUs.

For a technical overview, see our [blog post](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/).

**Code and tutorials are coming soon!**
