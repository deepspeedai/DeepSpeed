<div align="center">

# DeepNVMe: Improving DL Applications through I/O Optimizations

</div>

# Introduction

Deep Learning (DL) continues to drive unprecedented advancements across important
Artificial Intelligence domains including language, speech, video, and multimodal applications.
A key factor to these advancements is dramatic scalability on multiple dimensions including model size,
sequence length, and hardware parallelism. From a system perspective, DL scalability puts significant
pressure on essential subsystems including computation, memory, communication, and storage. However,
existing DL optimization efforts have mostly neglected the storage subsystem, making I/O operations such
as data loading, model checkpointing, and offloading the main bottlenecks of large-scale DL. To address
this problem, DeepSpeed has created a suite of I/O optimizations collectively called DeepNVMe.

DeepNVMe improves the performance and efficiency of I/O-bound DL applications by accelerating I/O operations
and reducing hardware requirements. It achieves this by leveraging storage innovations such as Non-Volatile
Memory Express (NVMe) Solid Storage Devices (SSDs) and NVIDIA Magnum IO<sup>TM</sup> GPUDirect® Storage (GDS). In this
blog we show the benefits of DeepNVMe using microbenchmarks and an inference application. In experiments
conducted on an Azure NC96ads\_A100\_v4 VM, we observed that DeepNVMe saturates available NVMe bandwidth for
data transfers with GPU or CPU memory, achieving up to 10GB/sec reads and 5 GB/secs writes.

# Background
High-performance access to persistent storage is a common challenge in many computing domains, including DL. Thus, a significant number of hardware and software solutions have been proposed. DeepNVMe builds on three such solutions: (1) NVMe SSDs, (2) NVIDIA GDS, and (3) Linux Asynchronous I/O (libaio). We will briefly describe each of these technologies.

NVMe SSDs are Flash-based storage devices that are replacing much slower hard disk drives (HDD) as primary persistent storage in modern servers. For example, an Azure NC96ads\_A100\_v4 VM is equipped with four NVMe SSDs which are individually capable of 3.25 GB/sec reads and can be combined in a RAID-0 configuration for a theoretical aggregate read bandwidth of 13 GB/sec. NVIDIA GDS enables direct transfers between NVMe and GPU memory thus avoiding the inefficiencies of the traditional approach of using intermediate CPU memory (bounce buffer). NVIDIA GDS is generally available in CUDA versions 11.4 and above. Finally, libaio is an asynchronous I/O stack introduced in Linux to better extract raw performance of fast storage devices like NVMe SSDs compared to the traditional I/O stack.

# DeepNVMe: an Optimization Module for Deep Learning I/O

DeepNVMe is a Python module that we developed with two key design principles. First, it leverages the above discussed storage technologies to implement powerful optimizations such as non-blocking I/O operations, bulk submission of I/O operations, parallelization of an individual I/O operation, and a lightweight runtime. Second, it exposes these I/O optimizations through a simple POSIX-like interface to foster easy integration into DL applications while avoiding the complexities of the underlying technologies.

# Evaluation

Our experiments are conducted on an Azure NC96ads\_A100\_v4 VM with setup details summarized in Table 1. For multi-device experiments, the SSDs are combined in a RAID-0 configuration.

<img src="./media/table1.png" style="width:6.5in;height:3.42153in" />

<div align="center">
Table 1: Experimental setup details
</div>

## Microbenchmark Performance

We used three benchmarking tools for our evaluations. The first is fio, the popular I/O benchmarking tool written in C. The second is gdsio from NVIDIA for benchmarking GDS performance. The third is ds\_io, a Python tool that we created for easy integration with DeepNVMe and to be more representative of DL applications which are commonly Python-based.

## High-Performance I/O with CPU Buffers via NVMe Scaling

Our first set of microbenchmark evaluations used fio and ds\_io to measure the performance of transferring 1GB data between NVMe and CPU memory. We configure fio to use the libaio backend for these experiments1. The results are summarized in Figure 1, from which we make two observations. First, DeepNVMe demonstrates high performance as it roughly matches fio, despite being more representative of DL applications. Second, DeepNVMe scales I/O performance almost linearly with available NVMe bandwidth, achieving rates of 10GB/sec reads and 5GB/sec writes.

<img src="./media/figure1.png" style="width:6.5in;height:3.42153in" />

<div align="center">
Figure 1: Using DeepNVMe to scale data transfers between NVMe and CPU buffer
</div>

## High-Performance I/O with GPU Buffers via NVMe Scaling

Our second set of microbenchmark evaluations used gdsio and ds\_io to measure the performance of 1GB data transfer between NVMe and GPU memory. For this experiment, we configure ds\_io to use both the traditional bounce buffer approach and the more efficient GDS approach. The results are summarized in Figure 2, from which we make three observations. First, we see that GDS improves performance in DeepNVMe compared to the traditional bounce buffer approach, with up to 37% speedup. Second, DeepNVMe demonstrates high performance by matching (and sometimes surpassing) gdsio despite being more representative of DL applications. Third, we see that DeepNVMe, with and without GDS, scales I/O performance with available NVMe bandwidth. With GDS, DeepNVMe achieves a maximum of 9.6GB/sec reads and 5GB/sec writes, and without GDS achieves 7GB/sec reads and 4GB/sec writes.

<img src="./media/figure2.png" style="width:6.5in;height:3.42153in" />

<div align="center">
Figure 2: Using DeepNVMe to scale data transfers between NVMe and GPU memory
</div>

## ZeRO-Inference: Generative AI Performance

ZeRO-Inference is an AI democratization technology that reduces the hardware cost of inferencing massive models by using DeepNVMe to offload model weights to CPU or NVMe memory.  ZeRO-Inference is well suited for throughput-oriented applications, such as offline inferencing, and for scenarios with limited hardware budget. We use token generation workload to evaluate DeepNVMe performance for NVMe offloading.

## High-Performance Offloading via NVMe Scaling

We measure the generation throughput of inferencing a LLAMA3-70B model on a single NVIDIA A100-80GB with a prompt length of 512, generation length of 32, and batch size of 96. We scale the number of NVMe SSDs from 1 to 4 and present the results for ZeRO-Inference with and without GDS in Figure 3.  We make two observations from these results. First, GDS consistently provides better performance compared to the bounce buffer approach, achieving 10-18% faster token generation. Second, DeepNVMe, with and without GDS, scales generation performance with available NVMe bandwidth. With four NVMe SSDs, DeepNVMe achieves generation throughput rates of 7 tokens per second with GDS and 6 tokens per second without GDS. Our profiling results suggest that DeepNVMe will continue to scale with more NVMe bandwidth, making it an economic option for boosting generative application performance.

<img src="./media/figure3.png" style="width:6.5in;height:3.42153in" />

<div align="center">
Figure 3: Using DeepNVMe to scale LLAMA3-70B token generation performance with NVMe offloading.
</div>

# Summary

In this blog post, we introduced DeepNVMe, an I/O optimization technology created to tackle the emergence of I/O operations as key bottlenecks of Deep Learning scalability. DeepNVMe enables fast and efficient data transfers between persistent storage and DL application memory through optimizations built on popular storage technologies such as NVMe SSDs and NVIDIA GDS. We showed benefits of using DeepNVMe for LLAMA3-70B token generation on single A100-80GB GPU with NVMe offloading, for which it achieves up to 7 tokens per second in generation throughput on an Azure NC96ads\_A100\_v4 VM. DeepNVMe will be open-sourced and generally available in DeepSpeed versions >= [0.15.0](https://github.com/microsoft/DeepSpeed/releases/tag/v0.15.0).  In future blogs, we will report DeepNVMe improvements for other I/O bound DL applications such as model checkpointing and data loading.


# Acknowlegements
This work is the result of a deep collaboration between Microsoft and NVIDIA. The contributors include Joe Mayer, Martin Cai, and Olatunji Ruwase from Microsoft; Kiran Modukuri, Vahid Noormofidi, Sourab Gupta, and Sandeep Joshi from Nivida.
