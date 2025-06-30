<div align="center">

# 在Windows系统上使用DeepSpeed

</div>

# 简介

DeepSpeed是一个广受欢迎的开源深度学习优化库，它使得分布式训练和推理变得简单、高效且有效。凭借其众多复杂的优化技术（如ZeRO、3D并行、MoE等），DeepSpeed已被成功应用于包括Phi-3、Megatron-Turing-530B、BLOOM-176B和Arctic在内的多种前沿模型的训练。然而，由于缺乏对主流操作系统微软 Windows的原生支持，许多AI开发者与用户无法充分利用DeepSpeed的创新。为此，我们致力于让DeepSpeed在Windows上实现原生全功能运行，并保持与Linux相同的易用性。

在这篇博客中，我们很高兴地宣布我们开发工作中的一些早期成果：DeepSpeed 现在可以在 Windows 上安装并原生支持单 GPU 的训练、微调和推理。重要的是，安装和使用体验与 Linux 上完全相同。此外，微调和推理工作展示了 DeepSpeed 的三个关键特性：HuggingFace Transformers 的集成、LoRA 的支持和 CPU offload。DeepSpeed 在 Windows 上的支持从 DeepSpeed 0.14.5 开始。接下来，我们将通过一些例子展示这些成就。

# 测试环境

我们在一台运行 Windows 11 23H2 版本号 22631.3880 的 Surface Laptop Studio 2 上进行了测试。该笔记本配备了一块 4GB 显存的 NVIDIA RTX A2000 GPU。我们使用了 Pytorch 2.3.0 和 HuggingFace Transformers 4.41.2。测试所用的脚本来自 [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples) 代码仓库，因此在运行以下任何示例前，你需要克隆该仓库。

# 安装指南
DeepSpeed可以通过两种方式在Windows系统上安装。较为简单的方式是使用pip包管理器安装，另一种方法是从源代码安装。两种安装方式的前提条件都是系统已经安装了Python 3.x 和支持CUDA的Pytorch.

## 通过pip安装
要安装 DeepSpeed，只需运行：`pip install deepspeed`。它将安装最新版本的 DeepSpeed（目前为 0.14.5）。与 Linux 版本不同的是，Windows 版本已经预先编译了内部的自定义算子，因此不需要安装 CUDA 或 C++ 编译器。

<div align="center">
    <img src="../media/win_pip_install_deepspeed.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    通过pip在Windows上安装Deepspeed.
</div>


##  通过源代码安装
克隆DeepSpeed代码仓库后，运行build_win.bat脚本进行编译安装。


## 验证安装
无论选择哪种安装方式，你都可以通过运行 ds_report 来检查安装是否成功。输出应该如下所示：


<div align="center">
    <img src="../media/ds_report.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    ds_report的输出结果，用于验证安装是否成功.
</div>

# 预训练(Pretraining)
我们使用图像分类模型 CIFAR10 和语言模型 BERT 来演示在 Windows 上使用 DeepSpeed 进行预训练。

## CIFAR10模型预训练
用于 CIFAR10 预训练的脚本和代码可以在以下路径找到：`DeepSpeedExamples\training\cifar`。你可以运行以下命令启动 CIFAR10 预训练实验：`deepspeed cifar10_deepspeed.py --deepspeed`。最终输出应类似于：
<div align="center">
    <img src="../media/cifar10_training.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    在 Windows 上使用 Deepspeed 进行 CIFAR10 模型预训练
</div>

## BERT模型预训练
用于 BERT 预训练的脚本和代码可以在以下路径找到：`DeepSpeedExamples\training\HelloDeepSpeed`。你可以使用以下命令启动 BERT 预训练实验：`deepspeed train_bert_ds.py --checkpoint_dir experiment_deepspeed`。最终输出应如下所示：

<div align="center">
    <img src="../media/bert_training.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    在 Windows 上使用 Deepspeed 进行 BERT 模型预训练
</div>

# 微调(Fine Tuning)
我们使用 DeepSpeed-Chat 应用的监督微调（SFT）步骤来演示微调能力。我们对 HuggingFace 的 facebook/opt-125m 模型进行监督微调，同时启用 LoRA 和 CPU offload进行内存优化。运行命令行如下：\
`deepspeed training\step1_supervised_finetuning\main.py --model_name_or_path facebook/opt-125m --gradient_accumulation_steps 8 --lora_dim 128 --only_optimize_lora --print_loss --zero_stage 2 --deepspeed --dtype bf16 --offload --output_dir output`\
输出应如下所示：

<div align="center">
    <img src="../media/opt125m_finetuning.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    在 Windows 上使用 DeepSpeed 对 facebook/opt-125m 监督微调
</div>

# 推理
我们使用 ZeRO-Inference 的token生成来演示推理能力。ZeRO-Inference 通过转移存储到 CPU 内存或 NVMe 硬盘内存来减少推理的硬件成本。我们使用以下脚本运行 HuggingFace 的 Llama-2-7B 模型来进行 token 生成。由于 4GB 显存无法容纳模型和生成所需的内存，我们将模型权重转移到 CPU 内存。我们使用以下命令行从 8个token的提示词中生成 32 个token：\
`deepspeed run_model.py --model meta-llama/Llama-2-7b-hf --batch-size 64 --prompt-len 8 --gen-len 32 --cpu-offload`\
输出应类似于：

<div align="center">
    <img src="../media/llama2-7b_inference.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    在 Windows 上使用 ZeRO-Inference 进行 LLAMA2-7B 模型的token生成
</div>

# 总结

让流行的深度学习框架 DeepSpeed 能够在最流行的操作系统 Windows 上原生运行，是让每个人和每个组织都能从正在进行的人工智能革命中受益的关键一步。在这篇博客中，我们分享了我们为实现这一目标所取得的早期成果。尽管 DeepSpeed 对 Windows 的支持仍在继续开发中，我们希望上述结果已经能够对我们的用户有实用价值，并且鼓舞他们。我们接下来的工作计划涵盖多GPU支持、权重量化以及性能优化。

# 致谢
这给项目的完成得益于现任和前任 DeepSpeed 成员的大力合作，包括 Costin Eseanu、Logan Adams、Elton Zheng、Reza Yazdani Aminabadi、Martin Cai 和 Olatunji Ruwase。我们还要感谢那些及时提出此项需求、提供关键的临时解决方法、部分解决方案和建设性反馈的 DeepSpeed 用户，最重要的是，他们始终与我们同行.
