# SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips

**Efficient full-parameter fine-tuning of GPT-OSS-20B & Qwen3-14B models on a single GPU and Llama3-70B on four GPUs, achieving up to 600 TFLOPS**

**Authors**  
[Xinyu Lian](https://xinyulian.tech/)<sup>1</sup>, [Masahiro Tanaka](https://tohtana.github.io/)<sup>2</sup>, [Olatunji Ruwase](https://www.snowflake.com/en/blog/authors/olatunji--tunji--ruwase/)<sup>3</sup>, [Minjia Zhang](https://minjiazhang.github.io/)<sup>1</sup>  

<sup>1</sup>University of Illinois Urbana-Champaign · <sup>2</sup>Anyscale · <sup>3</sup>Snowflake

---

## Table of Content

- [SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips](#superoffload-unleashing-the-power-of-large-scale-llm-training-on-superchips)
  - [Table of Content](#table-of-content)
  - [Introduction](#introduction)
  - [SuperOffload Highlights](#superoffload-highlights)
  - [How SuperOffload Works](#how-superoffload-works)
    - [1. Speculation-then-Validation (STV)](#1-speculation-then-validation-stv)
    - [2. Partial Offloading with Fine-Grained Bucketization](#2-partial-offloading-with-fine-grained-bucketization)
    - [3. Superchip-Aware Casting](#3-superchip-aware-casting)
  - [Experience and Insights](#experience-and-insights)
  - [Getting Started](#getting-started)
  - [Status \& Availability](#status--availability)
  - [Acknowledgements](#acknowledgements)
  - [BibTeX](#bibtex)

## Introduction

Recent models, especially MoE, at the scale of tens to hundreds of billions of parameters, make fine-tuning on limited GPUs difficult. Offloading to CPU memory helps reduce GPU demand but typically assumes GPU-CPU connections over PCIe, which is bandwidth-limited (e.g., 32 GB/s on PCIe-Gen4). Thus, prior work mainly optimizes data transfers to avoid PCIe becoming a major performance bottleneck. However, hardware vendors are introducing a new class of tightly coupled architectures—such as NVIDIA GH200, GB200, and AMD MI300A—that challenge these long-standing assumptions.

The open-source release of **SuperOffload** addresses this gap by providing a set of modular techniques for efficient large-model training. With SuperOffload, models such as **GPT-OSS-20B**, **Qwen3-14B**, and **Phi-4** can be fully fine-tuned on a single GH200, achieving **600 TFLOPS** under modest settings (sequence length 4k, batch size 4). This delivers up to **4×** higher throughput compared to ZeRO-Offload.

Built on top of ZeRO Stage 3, SuperOffload enables scaling to even larger models, including Qwen3-30B-A3B, Seed-OSS-36B on two GH200s and Llama-70B on four GH200s. All of this is supported natively through Hugging Face Transformers and DeepSpeed, with no need for custom modeling code.

![SuperOffload system overview](./images/superoffload_comparision.jpg)  
*Figure 1: SuperOffload delivers up to 4× higher throughput than ZeRO-Offload for large-model fine-tuning across varying sequence lengths and batch sizes, achieving a peak throughput of 600 TFLOPS.*

---

## SuperOffload Highlights

- **Single GH200:** Full fine-tuning of GPT-OSS-20B, Qwen3-14B, achieving ~600 TFLOPS (seq len 4K, batch size 4).
- **Scales Further:** Qwen3-30B-A3B & Seed-OSS-36B on 2× GH200; Llama-70B on 4× GH200.
- **Throughput Gains:** Up to 4× vs ZeRO-Offload under modest settings.
- **Built On:** DeepSpeed ZeRO Stage 3 + native Hugging Face Transformers integration.
- **No Custom Modeling Code:** Drop-in configuration driven.

---

## How SuperOffload Works

### 1. Speculation-then-Validation (STV)

Overlap CPU-Adam with backward propagation on the GPU.

- Traditional gradient clipping and NaN/INF checks block the optimizer step until all gradients arrive.
- **Speculation-then-validation** avoids this bottleneck by speculatively running CPU optimizer updates before full gradient checks finish.  
- If issues are detected later (NaN, INF, or gradient overflow), the update is rolled back and redone safely.

<img src="./images/superoffload_schedule.jpg" alt="Schedule comparison" width="80%">
<p align="center"><em>Figure 2: Previous offloading approach suffers from global gradient norm and global check of NAN and INF values, which expose the optimizer step to the critical path and prevent overlapping opportunities. In SuperOffload, we introduce a speculation-then-validation schedule to address this issue.</em></p>

<img src="./images/superoffload_rollback.jpg" alt="Gradient clipping data" width="80%">
*Figure 3: Red points indicate gradient clipping triggered during BLOOM pre-training — rare after warm-up, showing STV’s benefits.*

---

### 2. Partial Offloading with Fine-Grained Bucketization

- Instead of waiting for all updated parameters to return from CPU, keep the optimizer states and gradients of the last few buckets in GPU memory (if HBM allows).
- Reduces synchronization bubbles and idle time between iterations.
- Parameter \(n'\) controls how many tail buckets remain on GPU.

---

### 3. Superchip-Aware Casting

- Mixed precision training involves casting tensors between FP16/BF16 and FP32.
- On superchips with high CPU↔GPU bandwidth, casting cost matters.
- SuperOffload improves efficiency by performing casting on the GPU and sending **high-precision** tensors to the CPU.

<img src="./images/superoffload_cast_transfer.jpg" alt="Casting optimization" width="80%">
*Figure 4: Casting to higher precision first on GPU and then transferring tensors is more efficient on Superchips.*

---

## Experience and Insights

- **NUMA Binding:**  
  Pair each GPU with its directly associated CPU to maximize bandwidth. In DeepSpeed:  
  ```bash
  --bind_cores_to_rank
  ```

- **MPAM (Memory System Resource Partitioning and Monitoring):**  
  Reduces interference between CPU and GPU tasks.

  **How to enable MPAM on Nvidia Superchips:**
  1. Install the kernel from [NVIDIA NV-Kernels](https://github.com/NVIDIA/NV-Kernels/tree/24.04_linux-nvidia-adv-6.11).
  2. Check MPAM support:
     ```bash
     grep MPAM /boot/config-$(uname -r)
     ```
     Expected output:
     ```
     CONFIG_ARM64_MPAM=y
     CONFIG_ACPI_MPAM=y
     CONFIG_ARM64_MPAM_DRIVER=y
     CONFIG_ARM64_MPAM_RESCTRL_FS=y
     ```
     Verify resctrl filesystem:
     ```bash
     ls -ld /sys/fs/resctrl
     ```
  3. Mount resctrl:
     ```bash
     mount -t resctrl resctrl /sys/fs/resctrl
     ```
  4. Create partitions:
     ```bash
     mkdir /sys/fs/resctrl/p1 /sys/fs/resctrl/p2
     ```
  5. Set CPU cores & memory configs (example from experiments):
     ```
     /sys/fs/resctrl/p1/cpus_list:
     0-6
     /sys/fs/resctrl/p2/cpus_list:
     7-71
     /sys/fs/resctrl/p1/schemata:
     MB:1=100
     L3:1=ff0
     /sys/fs/resctrl/p2/schemata:
     MB:1=20
     L3:1=f
     ```

---

## Getting Started

SuperOffload is integrated into DeepSpeed as modular extensions on top of ZeRO Stage 3 and exposed via native configuration in Hugging Face Transformers—no model code changes required.

To enable SuperOffload, add the following switch to your DeepSpeed config:

<img src="./images/superoffload_enable.jpg" alt="Enable SuperOffload" width="60%">
*Figure 5: Enable SuperOffload with a single line in the DeepSpeed config.*

Tip: On superchip platforms (e.g., GH200/GB200/MI300A), combine NUMA binding and MPAM settings from "Experience and Insights" to stabilize bandwidth and improve end-to-end performance.

## Status & Availability

SuperOffload is open-sourced as modular extensions atop ZeRO Stage 3 in DeepSpeed and is exposed via native configuration in Hugging Face Transformers (no model code changes).

Community feedback and contributions are welcome. For enablement and examples, see "Getting Started" above.

---

## Acknowledgements

This work is a close collaboration between University of Illinois Urbana-Champaign (UIUC) and the DeepSpeed team.

We also gratefully acknowledge William Gropp, Brett Bode, and Gregory H. Bauer from the National Center for Supercomputing Applications (NCSA), as well as Dan Ernst, Ian Karlin, Giridhar Chukkapalli, Kurt Rago, and others from NVIDIA for their valuable discussions and guidance on MPAM support on Grace CPU.

---

## BibTeX

```bibtex
@inproceedings{superoffload,
    author = {Xinyu Lian and Masahiro Tanaka and Olatunji Ruwase and Minjia Zhang},
    title = "{SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips}",
    year = {2026},
    booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating System (ASPLOS'26)}
}
```
