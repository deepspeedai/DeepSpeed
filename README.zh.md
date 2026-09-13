[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/deepspeedai/DeepSpeed/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/deepspeed.svg)](https://pypi.org/project/deepspeed/)
[![Downloads](https://static.pepy.tech/badge/deepspeed)](https://pepy.tech/project/deepspeed)
[![Build](https://badgen.net/badge/build/check-status/blue)](#构建流水线状态)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9530/badge)](https://www.bestpractices.dev/projects/9530)
[![Twitter](https://img.shields.io/twitter/follow/DeepSpeedAI)](https://twitter.com/intent/follow?screen_name=DeepSpeedAI)
[![Japanese Twitter](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9ETwitter-%40DeepSpeedAI_JP-blue)](https://twitter.com/DeepSpeedAI_JP)
[![Chinese Zhihu](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%BE%AE%E8%BD%AFDeepSpeed-blue)](https://www.zhihu.com/people/deepspeed)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/deepspeedworkspace/shared_invite/zt-3a8pjd8dd-PCj2hMvR4Y2syPwVnjEoww)


<div align="center">
 <img src="docs/assets/images/DeepSpeed_light.svg#gh-light-mode-only" width="400px">
 <img src="docs/assets/images/DeepSpeed_dark_transparent.svg#gh-dark-mode-only" width="400px">
</div>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>


## 开发者交流会 (Office Hours)

DeepSpeed 于每月最后一个星期二美东时间 12:00（北京时间次日凌晨 00:00 / 01:00）定期举行线上 Office Hours，共同讨论开发规划、新特性与技术设计等。该会议对所有人公开，欢迎任何人加入并交流提问。
会议基于 Zoom 进行，可通过[此处链接](https://zoom-lfx.platform.linuxfoundation.org/meeting/93902569995?password=7d9c4fc9-3efa-4715-88f0-df8a6deb008b)直接参会。

## 最新动态

* [2026/05] [在 DeepSpeed 中使用 Muon 优化器](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/muon-optimizer/README.md)

* [2026/05] [针对 ZeRO-3 的系统 DMA (SDMA)：将通信算子从 AMD GPU 计算单元卸载，实现更优的通信与计算重叠](https://github.com/deepspeedai/DeepSpeed/blob/master/examples/sdma_allgather/README.md)

* [2026/03] DeepSpeed 团队在 ASPLOS 2026 上开展了题为 [“Building Efficient Large-Scale Model Systems with DeepSpeed: From Open-Source Foundations to Emerging Research”](https://supercomputing-system-ai-lab.github.io/events/asplos2026-llm-tutorial/index.html) 的技术专题研讨

* [2026/03] [我们的 SuperOffload 研究成果荣获 ASPLOS 2026 最佳论文提名 (Honorable Mention)](https://dl.acm.org/doi/10.1145/3760250.3762217)

* [2025/12] [DeepSpeed Core API 重大更新：支持原生 PyTorch 风格的反向传播与低精度主状态（Master States）管理](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/core_api_update/README.md)

* [2025/11] [DeepSpeed ZeRO++ 助力 LinkedIn 推荐系统超大规模大语言模型（LLM）的蒸馏训练](https://aclanthology.org/2025.emnlp-industry.119/)

* [2025/10] 我们在 Anyscale 举办了 [Ray x DeepSpeed 线下 Meetup](https://luma.com/3wctqteh)，分享了关于 SuperOffload、ZenFlow、Muon 优化器支持、Arctic 长序列训练（ALST）以及 DeepCompile 的最新研究进展。交流会演讲胶片见[此处](https://docs.google.com/presentation/d/1eM3mY6oW9GYkRy1Xz0iOnbbEr5T1t0JJXOM5BKtR-Ks/edit?slide=id.g38615d6b4c2_0_87#slide=id.g38615d6b4c2_0_87)。

* [2025/10] [SuperOffload：释放超级芯片（Superchips）上大规模大语言模型训练的强大潜能](https://pytorch.org/blog/superoffload-unleashing-the-power-of-large-scale-llm-training-on-superchips/)

* [2025/10] [结合 DeepSpeed CPU 核心绑定的 ZenFlow 与 ZeRO Offload 性能基准评测](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/zenflow-corebinding/README.md)

* [2025/08] [ZenFlow：面向大语言模型训练的无阻塞（Stall-Free）内存卸载引擎](https://pytorch.org/blog/zenflow-stall-free-offloading-engine-for-llm-training/)

* [2025/06] [基于 DeepSpeed 的 Arctic 长序列训练 (ALST)：支持数百万 Token 超长序列的高扩展与高效训练方案](https://www.snowflake.com/en/engineering-blog/arctic-long-sequence-training-multi-million-token-ai/)

* [2025/06] [DeepNVMe：面向深度学习应用的高性价比 I/O 扩展方案](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/06-2025/README.md)


<!-- NOTE: we must use html for news items otherwise links will be broken in the 'more news' section -->
<details>
<!-- NOTE: Maintain only three items in 'more news' section -->
 <summary>查看更多历史动态</summary>
 <ul>

   <li>[2025/04] <a href="https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md">DeepCompile：为分布式训练全面释放编译器级优化潜力</a></li>

   <li>[2025/03] <a href="https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/huggingface-tp/README.md">DeepSpeed AutoTP：Hugging Face 模型的自动化张量并行（Tensor Parallel）训练支持</a></li>

 <li>[2024/12] <a href="https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md">Ulysses-Offload：普惠超长上下文 LLM 的分布式训练</a></li>

 </ul>
</details>

---

# 极致速度与超大规模深度学习训练

***[DeepSpeed](https://www.deepspeed.ai/) 成功支撑了全球众多极具影响力的大规模语言模型训练，例如 [MT-530B](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) 以及 [BLOOM](https://huggingface.co/blog/bloom-megatron-deepspeed)***。DeepSpeed 汇聚了一系列前沿的[系统级创新](https://www.deepspeed.ai/training/)，使超大规模深度学习训练变得既可行又极其高效，大幅提升了开发易用性，并在模型可扩展性维度彻底重新定义了深度学习的训练边界。这些突破性技术包括 ZeRO、ZeRO-Infinity、3D 并行（3D-Parallelism）、Ulysses 序列并行、DeepSpeed-MoE 等。

---

# DeepSpeed 生态应用

DeepSpeed 是微软 [AI at Scale](https://www.microsoft.com/en-us/research/project/ai-at-scale/) 核心战略的重要支柱，旨在规模化赋能下一代人工智能前沿能力，更多背景可查阅[此处详情](https://innovation.microsoft.com/en-us/exploring-ai-at-scale)。

DeepSpeed 已被广泛应用于训练各类超大规模模型，以下为部分知名代表案例（若您希望在此列出您的模型，欢迎提交 PR）：

  * [Megatron-Turing NLG (530B)](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)
  * [Jurassic-1 (178B)](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)
  * [BLOOM (176B)](https://huggingface.co/blog/bloom-megatron-deepspeed)
  * [GLM (130B)](https://github.com/THUDM/GLM-130B)
  * [xTrimoPGLM (100B)](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v2)
  * [YaLM (100B)](https://github.com/yandex/YaLM-100B)
  * [GPT-NeoX (20B)](https://github.com/EleutherAI/gpt-neox)
  * [AlexaTM (20B)](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning)
  * [Turing NLG (17B)](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)
  * [METRO-LM (5.4B)](https://arxiv.org/pdf/2204.06644.pdf)

DeepSpeed 已无缝集成至众多主流开源深度学习框架与生态：

|                                                                                                | 文档指引                                |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------- |
<img src="docs/assets/images/transformers-light.png#gh-light-mode-only" width="250px"><img src="docs/assets/images/transformers-dark.png#gh-dark-mode-only" width="250px"> | [在 Transformers 中使用 DeepSpeed](https://huggingface.co/docs/transformers/deepspeed) |
| <img src="docs/assets/images/accelerate-light.png#gh-light-mode-only" width="250px"><img src="docs/assets/images/accelerate-dark.png#gh-dark-mode-only" width="250px"> | [在 Accelerate 中使用 DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) |
| <img src="docs/assets/images/lightning-light.svg#gh-light-mode-only" width="200px"><img src="docs/assets/images/lightning-dark.svg#gh-dark-mode-only" width="200px"> | [在 Lightning 中使用 DeepSpeed](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#deepspeed) |
| <img src="docs/assets/images/mosaicml.svg" width="200px"> | [在 MosaicML 中使用 DeepSpeed](https://docs.mosaicml.com/projects/composer/en/latest/trainer/using_the_trainer.html?highlight=deepspeed#deepspeed-integration) |
| <img src="docs/assets/images/determined.svg" width="225px"> | [在 Determined 中使用 DeepSpeed](https://docs.determined.ai/latest/training/apis-howto/deepspeed/overview.html) |
| <img src="https://user-images.githubusercontent.com/58739961/187154444-fce76639-ac8d-429b-9354-c6fac64b7ef8.jpg" width=150> | [在 MMEngine 中使用 DeepSpeed](https://mmengine.readthedocs.io/en/latest/common_usage/large_model_training.html#deepspeed) |

---

# 构建流水线状态

| 平台 / 运行环境 | 持续集成状态 |
| ----------- | ------ |
| NVIDIA | [![nv-pre-compile-ops](https://github.com/deepspeedai/DeepSpeed/actions/workflows/nv-pre-compile-ops.yml/badge.svg)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/nv-pre-compile-ops.yml) [![modal-torch-latest](https://github.com/deepspeedai/DeepSpeed/actions/workflows/modal-torch-latest.yml/badge.svg)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/modal-torch-latest.yml) |
| AMD | [![amd-mi200](https://github.com/deepspeedai/DeepSpeed/actions/workflows/amd-mi200.yml/badge.svg?branch=master)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/amd-mi200.yml) |
| CPU | [![torch-latest-cpu](https://github.com/deepspeedai/DeepSpeed/actions/workflows/cpu-torch-latest.yml/badge.svg?branch=master)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/cpu-torch-latest.yml) |
| Intel Gaudi | [![hpu-gaudi2](https://github.com/deepspeedai/DeepSpeed/actions/workflows/hpu-gaudi2.yml/badge.svg?branch=master)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/hpu-gaudi2.yml) |
| Intel XPU | [![xpu-max1100](https://github.com/deepspeedai/DeepSpeed/actions/workflows/xpu-max1100.yml/badge.svg?branch=master)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/xpu-max1100.yml) |
| 生态集成 | [![aws-accelerate](https://github.com/deepspeedai/DeepSpeed/actions/workflows/aws-accelerate.yml/badge.svg)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/aws-accelerate.yml) |
| 其他检查 | [![Formatting](https://github.com/deepspeedai/DeepSpeed/actions/workflows/formatting.yml/badge.svg?branch=master)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/formatting.yml) [![pages-build-deployment](https://github.com/deepspeedai/DeepSpeed/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/pages/pages-build-deployment) [![Documentation Status](https://readthedocs.org/projects/deepspeed/badge/?version=latest)](https://deepspeed.readthedocs.io/en/latest/?badge=latest)[![python](https://github.com/deepspeedai/DeepSpeed/actions/workflows/python.yml/badge.svg?branch=master)](https://github.com/deepspeedai/DeepSpeed/actions/workflows/python.yml) |
| 华为昇腾 NPU | [![Huawei Ascend NPU](https://github.com/Ascend/Ascend-CI/actions/workflows/deepspeed.yaml/badge.svg?branch=main)](https://github.com/Ascend/Ascend-CI/actions/workflows/deepspeed.yaml) |

# 安装指南

通过 pip 安装是快速上手 DeepSpeed 最便捷的方式，这将安装 DeepSpeed 的最新稳定发布版（且无需强制绑定特定的 PyTorch 或 CUDA 版本）。DeepSpeed 包含多个我们通常称为“算子 (Ops)”的 C++/CUDA 原生扩展。默认情况下，所有这些扩展/算子都会通过[依赖 ninja 的 PyTorch JIT C++ 扩展加载器](https://pytorch.org/docs/stable/cpp_extension.html)在运行时按需即时（JIT）编译并动态链接。

## 环境要求
* 在安装 DeepSpeed **之前**必须预先安装 [PyTorch](https://pytorch.org/)。
* 为获得完整的特性支持，建议使用 PyTorch >= 2.0 版本，最佳实践为使用最新的 PyTorch 稳定发布版。
* 需具备用于编译 C++/CUDA/HIP 扩展的 CUDA 或 ROCm 编译器，例如 [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#introduction) 或 [hipcc](https://github.com/ROCm-Developer-Tools/HIPCC)。
* 我们日常重点开发与测试的 GPU 架构如下。若您的 GPU 未列在其中并不意味着无法运行，而是指 DeepSpeed 在以下硬件平台上经过了最充分的测试验证：
  * NVIDIA：Pascal、Volta、Ampere 以及 Hopper 架构
  * AMD：MI100 与 MI200

## 社区贡献硬件支持
* DeepSpeed 目前已支持多种异构硬件加速器：

| 贡献方 | 硬件设备                            | 加速器标识 (Accelerator Name) | 贡献方已验证 | 上游官方已验证 |
|-------------|-------------------------------------|------------------| --------------------- |--------------------|
| 华为 (Huawei)      | 华为昇腾 NPU (Huawei Ascend NPU)                   | npu              | 是 (Yes) | 否 (No)                 |
| 英特尔 (Intel)       | Intel(R) Gaudi(R) 2 AI 加速器  | hpu              | 是 (Yes) | 是 (Yes)                |
| 英特尔 (Intel)       | Intel(R) 至强(R) 处理器 (Xeon Processors)         | cpu              | 是 (Yes) | 是 (Yes)                |
| 英特尔 (Intel)       | Intel(R) 数据中心 GPU Max 系列 (Data Center GPU Max) | xpu              | 是 (Yes) | 是 (Yes)                |
| 云天励飞/太行 (Tecorigin)   | 可扩展数据分析加速器 (SDAA) | sdaa             | 是 (Yes) | 否 (No)                 |

## PyPI 安装
我们定期向 [PyPI](https://pypi.org/project/deepspeed/) 发布最新版本，在绝大多数场景下推荐直接从 PyPI 安装：

```bash
pip install deepspeed
```

安装完成后，您可以通过运行 DeepSpeed 环境诊断报告来验证安装，并查看当前机器所兼容支持的扩展/算子列表：

```bash
ds_report
```

如果您希望预先编译安装任意 DeepSpeed 算子/扩展（而非在运行时 JIT 编译），或通过 PyPI 获取预编译好的算子，请参阅我们的[进阶安装指南](https://www.deepspeed.ai/tutorials/advanced-install/)。

## Windows 安装支持
DeepSpeed 在 Windows 平台上对训练和推理均已支持大量核心功能。关于此特性的详细背景可参阅[官方博客](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/windows/08-2024/README.md)。目前在 Windows 上尚未支持的特性包括异步 I/O (AIO) 与 GDS（其本身不支持 Windows）。
1. 安装 PyTorch（例如 pytorch 2.3+cu121）。
2. 安装 Visual C++ 生成工具（例如 VS2022 C++ x64/x86 生成工具）。
3. 以**管理员权限**打开 Cmd 控制台以创建所需的符号链接目录，并确保 MSVC 相关工具已添加至 PATH 环境变量中；或者以管理员权限启动 Visual Studio 2022 的 Developer Command Prompt。
4. 运行 `build_win.bat`，即可在 `dist` 目录下构建 Wheel 安装包。


# 延伸阅读

所有关于 DeepSpeed 的官方文档、教程指南与技术博客均可在我们的官方网站上找到：[deepspeed.ai](https://www.deepspeed.ai/)


|                                                                                                | 描述                                  |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [快速上手 (Getting Started)](https://www.deepspeed.ai/getting-started/)                                   |  DeepSpeed 初步入门指南                  |
| [JSON 配置详解 (DeepSpeed JSON Configuration)](https://www.deepspeed.ai/docs/config-json/)                     |  DeepSpeed 核心配置选项                       |
| [API 文档 (API Documentation)](https://deepspeed.readthedocs.io/en/latest/)                               |  DeepSpeed API 自动生成文档       |
| [教程指南 (Tutorials)](https://www.deepspeed.ai/tutorials/)                                               |  官方系列技术实战教程                                   |
| [技术博客 (Blogs)](https://www.deepspeed.ai/posts/)                                                       |  官方前沿技术深度博文                                   |


# 持续集成 (CI) 硬件赞助

作为一个开源项目，我们依赖合作伙伴为持续集成（CI）流水线提供硬件计算资源。目前，Modal 慷慨地为我们资助了 GPU CI 运行所需的全部硬件。Modal 是一个专注于模型推理、微调与批处理任务的一站式 AI 基础设施平台。现在访问 https://modal.com 即可获取每月 30 美元的免费额度开启体验。DeepSpeed 团队得到了来自 Modal 团队全方位的鼎力支持，在此向您的业务强烈推荐 Modal 服务。

# 参与贡献
DeepSpeed 非常欢迎开源社区的贡献！关于代码格式化规范、单元测试运行等详细信息，请参阅我们的[贡献指南](CONTRIBUTING.md)。<br/>
由衷感谢所有为 DeepSpeed 做出卓越贡献的开发者们！

<a href="https://github.com/deepspeedai/DeepSpeed/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/DeepSpeed&r="  width="800px"/>
</a>

## 开发者原创性承诺 (DCO)
本项目欢迎任何建议与代码贡献。对于绝大多数贡献，您需要同意[开发者原创性承诺 (DCO)](https://wiki.linuxfoundation.org/dco)，声明您认可发布于 https://developercertificate.org 的对应条款并拥有授予相关贡献权限的合法权利。

DCO 针对每次 Git Commit 生效，因此每个提交都需要附加签署声明（Sign-off）。您只需在提交代码时添加 `-s` 参数即可完成签署（例如 `git commit -s -m "Commit message"`）。在 Pull Request 界面中，也可以直接点击 DCO 检查项完成授权。

## 行为准则
本项目遵循 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。欲了解更多信息，请查阅[行为准则常见问题](https://opensource.microsoft.com/codeofconduct/faq/)，如有其他疑问或意见，亦可直接联系 [opencode@microsoft.com](mailto:opencode@microsoft.com)。

# 代表性学术论文
1. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. (2019) ZeRO: memory optimizations toward training trillion parameter models. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) and [In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20)](https://dl.acm.org/doi/10.5555/3433701.3433727).
2. Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. (2020) DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. [In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20, Tutorial)](https://dl.acm.org/doi/10.1145/3394486.3406703).
3. Minjia Zhang, Yuxiong He. (2020) Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping. [arXiv:2010.13369](https://arxiv.org/abs/2010.13369) and [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html).
4. Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He. (2021) ZeRO-Offload: Democratizing Billion-Scale Model Training. [arXiv:2101.06840](https://arxiv.org/abs/2101.06840) and [USENIX ATC 2021](https://www.usenix.org/conference/atc21/presentation/ren-jie). [[paper]](https://arxiv.org/abs/2101.06840) [[slides]](https://www.usenix.org/system/files/atc21_slides_ren-jie.pdf) [[blog]](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)
5. Hanlin Tang, Shaoduo Gan, Ammar Ahmad Awan, Samyam Rajbhandari, Conglong Li, Xiangru Lian, Ji Liu, Ce Zhang, Yuxiong He. (2021) 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed. [arXiv:2102.02888](https://arxiv.org/abs/2102.02888) and [ICML 2021](http://proceedings.mlr.press/v139/tang21a.html).
6. Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. (2021) ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857) and [SC 2021](https://dl.acm.org/doi/abs/10.1145/3458817.3476205). [[paper]](https://arxiv.org/abs/2104.07857) [[slides]](docs/assets/files/SC21-ZeRO-Infinity.pdf) [[blog]](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/)
7. Conglong Li, Ammar Ahmad Awan, Hanlin Tang, Samyam Rajbhandari, Yuxiong He. (2021) 1-bit LAMB: Communication Efficient Large-Scale Large-Batch Training with LAMB's Convergence Speed. [arXiv:2104.06069](https://arxiv.org/abs/2104.06069) and [HiPC 2022](https://hipc.org/advance-program/).
8. Conglong Li, Minjia Zhang, Yuxiong He. (2021) The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models. [arXiv:2108.06084](https://arxiv.org/abs/2108.06084) and [NeurIPS 2022](https://openreview.net/forum?id=JpZ5du_Kdh).
9. Yucheng Lu, Conglong Li, Minjia Zhang, Christopher De Sa, Yuxiong He. (2022) Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam. [arXiv:2202.06009](https://arxiv.org/abs/2202.06009).
10. Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He. (2022) DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale [arXiv:2201.05596](https://arxiv.org/abs/2201.05596) and [ICML 2022](https://proceedings.mlr.press/v162/rajbhandari22a.html). [[pdf]](https://arxiv.org/abs/2201.05596) [[slides]](docs/assets/files/ICML-5mins.pdf) [[blog]](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/)
11. Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, Rewon Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, Bryan Catanzaro. (2022) Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model [arXiv:2201.11990](https://arxiv.org/abs/2201.11990).
12. Xiaoxia Wu, Zhewei Yao, Minjia Zhang, Conglong Li, Yuxiong He. (2022) Extreme Compression for Pre-trained Transformers Made Simple and Efficient. [arXiv:2206.01859](https://arxiv.org/abs/2206.01859) and [NeurIPS 2022](https://openreview.net/forum?id=xNeAhc2CNAl).
13. Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He. (2022) ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. [arXiv:2206.01861](https://arxiv.org/abs/2206.01861) and [NeurIPS 2022](https://openreview.net/forum?id=f-fVCElZ-G1) [[slides]](docs/assets/files/zeroquant_series.pdf) [[blog]](https://www.microsoft.com/en-us/research/blog/deepspeed-compression-a-composable-library-for-extreme-compression-and-zero-cost-quantization/)
14. Reza Yazdani Aminabadi, Samyam Rajbhandari, Minjia Zhang, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Jeff Rasley, Shaden Smith, Olatunji Ruwase, Yuxiong He. (2022) DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale. [arXiv:2207.00032](https://arxiv.org/abs/2207.00032) and [SC 2022](https://dl.acm.org/doi/abs/10.5555/3571885.3571946). [[paper]](https://arxiv.org/abs/2207.00032) [[slides]](docs/assets/files/sc22-ds-inference.pdf) [[blog]](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/)
15. Zhewei Yao, Xiaoxia Wu, Conglong Li, Connor Holmes, Minjia Zhang, Cheng Li, Yuxiong He. (2022) Random-LTD: Random and Layerwise Token Dropping Brings Efficient Training for Large-scale Transformers. [arXiv:2211.11586](https://arxiv.org/abs/2211.11586).
16. Conglong Li, Zhewei Yao, Xiaoxia Wu, Minjia Zhang, Yuxiong He. (2022) DeepSpeed Data Efficiency: Improving Deep Learning Model Quality and Training Efficiency via Efficient Data Sampling and Routing. [arXiv:2212.03597](https://arxiv.org/abs/2212.03597) [ENLSP2023 Workshop at NeurIPS2023](https://neurips2023-enlsp.github.io/)
17. Xiaoxia Wu, Cheng Li, Reza Yazdani Aminabadi, Zhewei Yao, Yuxiong He. (2023) Understanding INT4 Quantization for Transformer Models: Latency Speedup, Composability, and Failure Cases. [arXiv:2301.12017](https://arxiv.org/abs/2301.12017) and [ICML2023](https://icml.cc/Conferences/2023).
18. Syed Zawad, Cheng Li, Zhewei Yao, Elton Zheng, Yuxiong He, Feng Yan. (2023) DySR: Adaptive Super-Resolution via Algorithm and System Co-design. [ICLR:2023](https://openreview.net/forum?id=Pgtn4l6eKjv).
19. Sheng Shen, Zhewei Yao, Chunyuan Li, Trevor Darrell, Kurt Keutzer, Yuxiong He. (2023) Scaling Vision-Language Models with Sparse Mixture of Experts. [arXiv:2303.07226](https://arxiv.org/abs/2303.07226) and [Finding at EMNLP2023](https://2023.emnlp.org/).
20. Quentin Anthony, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He, Aamir Shafi, Mustafa Abduljabbar, Hari Subramoni, Dhabaleswar Panda. (2023) MCR-DL: Mix-and-Match Communication Runtime for Deep Learning [arXiv:2303.08374](https://arxiv.org/abs/2303.08374) and will appear at IPDPS 2023.
21. Siddharth Singh, Olatunji Ruwase, Ammar Ahmad Awan, Samyam Rajbhandari, Yuxiong He, Abhinav Bhatele. (2023) A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize Mixture-of-Experts Training [arXiv:2303.06318](https://arxiv.org/abs/2303.06318) and [ICS 2023](https://dl.acm.org/doi/10.1145/3577193.3593704).
22. Guanhua Wang, Heyang Qin, Sam Ade Jacobs, Xiaoxia Wu, Connor Holmes, Zhewei Yao, Samyam Rajbhandari, Olatunji Ruwase, Feng Yan, Lei Yang, Yuxiong He. (2023) ZeRO++: Extremely Efficient Collective Communication for Giant Model Training [arXiv:2306.10209](https://arxiv.org/abs/2306.10209) and [ML for Sys Workshop at NeurIPS2023](http://mlforsystems.org/) [[blog]](https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/)
23. Zhewei Yao, Xiaoxia Wu, Cheng Li, Stephen Youn, Yuxiong He. (2023) ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation [arXiv:2303.08302](https://arxiv.org/abs/2303.08302) and [ENLSP2023 Workshop at NeurIPS2023](https://neurips2023-enlsp.github.io/) [[slides]](docs/assets/files/zeroquant_series.pdf)
24. Pareesa Ameneh Golnari, Zhewei Yao, Yuxiong He. (2023) Selective Guidance: Are All the Denoising Steps of Guided Diffusion Important? [arXiv:2305.09847](https://arxiv.org/abs/2305.09847)
25. Zhewei Yao, Reza Yazdani Aminabadi, Olatunji Ruwase, Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad Awan, Jeff Rasley, Minjia Zhang, Conglong Li, Connor Holmes, Zhongzhu Zhou, Michael Wyatt, Molly Smith, Lev Kurilenko, Heyang Qin, Masahiro Tanaka, Shuai Che, Shuaiwen Leon Song, Yuxiong He. (2023) DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales [arXiv:2308.01320](https://arxiv.org/abs/2308.01320).
26. Xiaoxia Wu, Zhewei Yao, Yuxiong He. (2023) ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [arXiv:2307.09782](https://arxiv.org/abs/2307.09782) and [ENLSP2023 Workshop at NeurIPS2023](https://neurips2023-enlsp.github.io/) [[slides]](docs/assets/files/zeroquant_series.pdf)
27. Zhewei Yao, Xiaoxia Wu, Conglong Li, Minjia Zhang, Heyang Qin, Olatunji Ruwase, Ammar Ahmad Awan, Samyam Rajbhandari, Yuxiong He. (2023) DeepSpeed-VisualChat: Multi-Round Multi-Image Interleave Chat via Multi-Modal Causal Attention [arXiv:2309.14327](https://arxiv.org/pdf/2309.14327.pdf)
28. Shuaiwen Leon Song, Bonnie Kruft, Minjia Zhang, Conglong Li, Shiyang Chen, Chengming Zhang, Masahiro Tanaka, Xiaoxia Wu, Jeff Rasley, Ammar Ahmad Awan, Connor Holmes, Martin Cai, Adam Ghanem, Zhongzhu Zhou, Yuxiong He, et al. (2023) DeepSpeed4Science Initiative: Enabling Large-Scale Scientific Discovery through Sophisticated AI System Technologies [arXiv:2310.04610](https://arxiv.org/abs/2310.04610) [[blog]](https://www.microsoft.com/en-us/research/blog/announcing-the-deepspeed4science-initiative-enabling-large-scale-scientific-discovery-through-sophisticated-ai-system-technologies/)
29. Zhewei Yao, Reza Yazdani Aminabadi, Stephen Youn, Xiaoxia Wu, Elton Zheng, Yuxiong He. (2023) ZeroQuant-HERO: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers [arXiv:2310.17723](https://arxiv.org/abs/2310.17723)

30. Xiaoxia Wu, Haojun Xia, Stephen Youn, Zhen Zheng, Shiyang Chen, Arash Bakhtiari, Michael Wyatt, Reza Yazdani Aminabadi, Yuxiong He, Olatunji Ruwase, Leon Song, Zhewei Yao (2023) ZeroQuant(4+2): Redefining LLMs Quantization with a New FP6-Centric Strategy for Diverse Generative Tasks [arXiv:2312.08583](https://arxiv.org/abs/2312.08583)

31. Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, Shuaiwen Leon Song. (2024) FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design  [arXiv:2401.14112](https://arxiv.org/abs/2401.14112)
32. Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Reza Yazdani Aminadabi, Shuaiwen Leon Song, Samyam Rajbhandari, Yuxiong He. (2024) [System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://dl.acm.org/doi/10.1145/3662158.3662806)
33. Xinyu Lian, Sam Ade Jacobs, Lev Kurilenko, Masahiro Tanaka, Stas Bekman, Olatunji Ruwase, Minjia Zhang. (2024) Universal Checkpointing: Efficient and Flexible Checkpointing for Large Scale Distributed Training [arXiv:2406.18820](https://arxiv.org/abs/2406.18820)
34. Stas Bekman, Samyam Rajbhandari, Michael Wyatt, Jeff Rasley, Tunji Ruwase, Zhewei Yao, Aurick Qiao, Yuxiong He. (2025) Arctic Long Sequence Training: Scalable And Efficient Training For Multi-Million Token Sequences [arXiv:2506.13996](https://arxiv.org/abs/2506.13996)
35. Tingfeng Lan, Yusen Wu, Bin Ma, Zhaoyuan Su, Rui Yang, Tekin Bicer, Masahiro Tanaka, Olatunji Ruwase, Dong Li, Yue Cheng. (2025) ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates [arXiv:2505.12242](https://arxiv.org/abs/2505.12242)
36. Kayhan Behdin, Ata Fatahibaarzi, Qingquan Song, Yun Dai, Aman Gupta, Zhipeng Wang, Hejian Sang, Shao Tang, Gregory Dexter, Sirou Zhu, Siyu Zhu, Tejas Dharamsi, Vignesh Kothapalli, Zhoutong Fu, Yihan Cao, Pin-Lun Hsu, Fedor Borisyuk, Natesh S. Pillai, Luke Simon, Rahul Mazumder.(2025) Scaling Down, Serving Fast: Compressing and Deploying Efficient LLMs for Recommendation Systems [EMNLP 2025](https://aclanthology.org/2025.emnlp-industry.119/)
37. Xinyu Lian, Masahiro Tanaka, Olatunji Ruwase, Minjia Zhang. (2026) SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips [arxiv](https://arxiv.org/abs/2509.21271), [ASPLOS 2026](https://www.asplos-conference.org/asplos2026)

# 视频讲座与教程
1. DeepSpeed KDD 2020 专题教程
    1. [概览与架构 (Overview)](https://www.youtube.com/watch?v=CaseqC45DNc&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=29)
    2. [ZeRO 优化器与超大模型训练 (ZeRO + large model training)](https://www.youtube.com/watch?v=y4_bCiAsIAk&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=28)
    3. [17B 参数 T-NLG 演示 (17B T-NLG demo)](https://www.youtube.com/watch?v=9V-ZbP92drg&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=27)
    4. [极速 BERT 训练与 RScan 调优 (Fastest BERT training + RScan tuning)](https://www.youtube.com/watch?v=o1K-ZG9F6u0&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=26)
    5. DeepSpeed 实战深度拆解：[第一部分 (part 1)](https://www.youtube.com/watch?v=_NOk-mBwDYg&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=92)、[第二部分 (part 2)](https://www.youtube.com/watch?v=sG6_c4VXLww&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=94)、[第三部分 (part 3)](https://www.youtube.com/watch?v=k9yPkBTayos&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=93)
    6. [常见问题解答 (FAQ)](https://www.youtube.com/watch?v=nsHu6vEgPew&list=PLa85ZdUjfWS21mgibJ2vCvLziprjpKoW0&index=24)
2. 微软研究院网络研讨会 (Microsoft Research Webinar)
    * 免费注册即可按需观看全部视频。
    * [ZeRO 与最快 BERT：在 DeepSpeed 中全面提升深度学习训练规模与速度](https://note.microsoft.com/MSR-Webinar-DeepSpeed-Registration-On-Demand.html)
3. [在 AzureML 上运行 DeepSpeed (DeepSpeed on AzureML)](https://youtu.be/yBVXR8G8Bg8)
4. [基于 DeepSpeed 的大模型训练与推理 // Samyam Rajbhandari // LLMs in Prod 大会演讲](https://www.youtube.com/watch?v=cntxC3g22oU) [[演示胶片]](docs/assets/files/presentation-mlops.pdf)
5. 社区优质教程
    * [DeepSpeed：扩展至超大模型的全套秘籍 (Mark Saroufim)](https://www.youtube.com/watch?v=pDGI668pNg0)
    * [Turing-NLG、DeepSpeed 与 ZeRO 优化器剖析 (Yannic Kilcher)](https://www.youtube.com/watch?v=tC01FRB0M7w)
    * [扩展机器学习模型规模的终极指南 (The AI Epiphany)](https://www.youtube.com/watch?v=hc0u4avAkuM)

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
