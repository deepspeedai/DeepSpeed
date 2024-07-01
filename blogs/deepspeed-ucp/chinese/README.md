
<div align="center">

# DeepSpeed通用检查点：用于大规模分布式训练的高效灵活检查点系统

</div>

<img src="../media/image1.png" style="width:6.5in;height:3.65625in" />

要引用DeepSpeed通用检查点，请引用我们的[arxiv报告](https://arxiv.org/abs/2406.18820)：

```
@article{lian2024-ucp,
title={Universal Checkpointing: Efficient and Flexible Checkpointing for
Large Scale Distributed Training},
author={Xinyu Lian and Sam Ade Jacobs and Lev Kurilenko and Masahiro Tanaka
and Stas Bekman and Olatunji Ruwase and Minjia Zhang},
journal={arxiv preprint arxiv:406.18820},
year={2024},

}
```

# 引言

检查点是降低训练大型语言模型成本的关键技术，它使我们在训练过程中可以保存模型状态。这样，如果训练失败，训练可以从最后保存的点继续，而不是从头开始。此外，检查点还允许在训练的不同阶段评估模型性能，从而便于进行超参数调整以及针对不同和多样化下游任务的微调。

然而，在分布式训练和微调场景中设计、实施和使用检查点存在困难。ZeRO数据并行（ZeRO-DP）、流水线并行（PP）、张量并行（TP）和序列并行（SP）等方法是加速大型语言模型训练的出色技术，但与传统的默认（Torch）保存和加载检查点机制不兼容。此外，目前尚无技术支持将这些不同的并行拓扑与检查点灵活组合，部分原因是这些技术将模型和/或优化器状态分片，使得在不同GPU或加速器数量上创建的检查点难以用于恢复训练。

在此，我们很高兴地发布DeepSpeed通用检查点（*UCP*），这是解决分布式检查点问题的最全面的解决方案。*UCP*在高效创建检查点的同时，提供了在任意并行策略和硬件配置上恢复的灵活性。*UCP*还解锁了大规模训练的前所未有的能力，例如通过在剩余健康硬件上继续训练来提高对硬件故障的抵抗力，以及通过机会性利用弹性容量来减少训练时间。

简单来说，当前版本的*UCP*解锁了以下功能：

- 灵活的检查点可沿任何训练并行技术（即PP、TP、DP、ZeRO-DP、SP、MoE）重塑训练

- 弹性资源管理，在训练和微调中随意增加或减少硬件资源

- 支持多种商业规模模型的真实世界用例（例如BLOOM、Megatron GPT、LLAMA、Microsoft Phi）

# 核心设计

DeepSpeed *UCP*的关键洞察是在检查点生命周期的每个阶段选择最佳表示：分布式表示用于保存，合并表示用于加载。这通过两个关键机制实现。首先，通用检查点格式，它包括每个模型参数的合并表示和用于将参数片段映射到任意模型并行配置的训练级别的元数据。其次，通用检查点语言，这是一个简单但强大且健壮的规范语言，用于将分布式检查点转换为通用检查点格式。

## 通用检查点格式

<img src="../media/image2.png" style="width:6.5in;height:3.42153in" />

图1：UCP概述：顶部行和底部行分别为源并行配置和目标并行配置。中间行显示UCP作为从源到目标的转换中介块。

图1显示了*UCP*转换过程和格式的整体概念性描述。转换从任何并行策略格式的检查点顶部块开始。允许以训练的本地格式保存消除了可能因同步全局检查点保存而产生的任何开销。为确保保存的检查点（称为*源*）可以轻松转换并加载到任何并行策略以进行连续训练（称为*目标*），我们引入了作为中介块的原子检查点格式的概念。

原子检查点是*UCP*的核心概念。这些是包含每个模型参数的合并表示及其优化器状态的细粒度文件。原子检查点格式有三个用途。首先，原子检查点的表示解除了分布式检查点与特定并行技术和硬件配置的依赖。因此，无需为每个*源*到*目标*实现单独的转换器。相反，*UCP*可以充当不同分布式训练技术之间的通用交换格式，然后可以轻松地转换为其他分布式训练策略，如图2所示。通过保持每个模型参数的合并表示，*UCP*可以轻松地将模型状态或片段状态拆分并灵活地映射到不同GPU上，有效减少加载大型模型检查点所需的工作内存。其次，*UCP*转换是懒惰和按需进行的，例如，当训练过程检测到并行技术和硬件配置的变化时。换句话说，现有的分布式检查点保存逻辑不需要任何改变。第三，*UCP*的结构还易于处理分布式训练中的高级技术，例如混合精度训练。在实践中，研究人员和从业者可能在fp16和bfloat16混合精度训练之间切换。通过保持fp32的权重/优化器值，训练可以继续使用fp16或bfloat16恢复。

## 通用检查点语言

<img src="../media/flowchart.png" style="width:6.5in;height:2.22222in" />

图2：UCP语言帮助将分布式检查点转换为UCP格式，并根据目标并行技术和新硬件配置加载UCP检查点。


虽然*UCP*为不同的并行策略提供了一个公共接口，但从任意分布式检查点到*UCP*的转换仍然可能具有不菲的工程和实施成本。这是因为分布式训练中的每个GPU都调用一个持久方法（例如，在PyTorch中使用torch.save()）将其拥有的GPU模型状态保存到磁盘上的检查点文件中，而每个检查点的具体内容在不同技术之间会有所不同。

为了应对这一挑战，*UCP*提供了*UCP*语言，这是一个简单但强大的规范语言，用于将几种类型的分布式检查点转换为前一节中描述的通用格式。*UCP*以两种方式实现这一点。首先，它提供了一个具有预定义*参数模式*的声明式系统，这些模式涵盖了模型状态的广泛并行

策略。参数模式包含有关参数如何在GPU之间分区的运行时信息。例如，*nopattern*表示一个参数与某个GPU唯一相关，这是ZeRO-1/2和PP等技术中最常见的模式（参见我们的技术报告，以获得当前支持的参数模式完整列表）。其次，*UCP*语言提供了一组常见操作符，以便将分布式检查点转换为合并的原子检查点。从高层次来看，如图3所示，当需要新的*目标*并行技术或硬件配置发生变化时，将调用*UCP*语言。它首先将分布式检查点转换为*UCP*格式。然后根据*目标*并行技术和新硬件配置加载*UCP*检查点。

# 关键结果

我们通过一系列实验评估*UCP*，专注于仅解码器的Transformers架构，这是由于其最先进的性能。一些最大的模型也是基于解码器的，这使得灵活高效的检查点尤为重要。在本博客中，我们展示了在不同模型和并行策略下正确性验证的结果。有关并行效率分析、详细的系统和模型架构以及训练超参数的更多结果，请参阅上面引用的技术报告。

*UCP*提供了从一个*源*并行策略到不同的*目标*和不同硬件配置的灵活检查点。为验证这一能力，我们进行了正确性测试的两组实验。

## 单源到多目标

<img src="../media/image4.png" style="width:4.85477in;height:4in" />

图3：在第101次迭代时使用不同目标加载UCP检查点的训练曲线，具有不同GPU数量和并行策略

为测试UCP是否允许使用不同并行策略和硬件配置恢复训练，我们首先使用TP=2、PP=2、DP=2（ZeRO-1）和SP=1的配置训练GPT-3模型。由于时间和资源的限制，我们将实验限制在前200次迭代。我们将在第100次迭代保存的检查点转换为*UCP*检查点，并使用不同GPU数量和并行策略恢复训练。我们记录了每次迭代的LM损失（数据并行组的平均损失）。图3显示，训练可以使用不同的*目标*并行策略无缝地使用*UCP*检查点恢复，如果训练继续使用*源*策略，将实现一致的收敛。

## 多源到单目标

<img src="../media/image5.png" style="width:4.85477in;height:4in" />

图4：在第100次迭代将不同源并行策略转换为UCP并加载UCP的训练曲线，具有不同的目标。

图4显示了从多个*源*配置到单一*目标*的训练曲线。在固定随机种子的情况下，我们首先使用不同的*源*配置训练GPT-3模型。然后我们将它们在第100次迭代保存的分布式检查点转换为*UCP*检查点，并使用TP=2、PP=2、DP=1和SP=1的配置恢复训练。结果显示，无论不同的*源*配置如何，它们的检查点都可以转换为*UCP*并使用不同的配置恢复训练。最重要的是，恢复的训练曲线与第101--200次迭代的*源*曲线匹配。这些结果验证了*UCP*将任意配置转换为不同配置以恢复训练的有效性。

## 不同模型架构的变化

*UCP*与模型架构无关。因此，它不仅与GPT模型兼容，而且足够灵活，可以支持各种其他模型架构和大小。图5、6和7显示了使用新并行策略从*UCP*中恢复训练时的训练收敛情况。这些图表显示，训练可以使用*UCP*无缝恢复，实现与初始训练阶段一致的收敛，这与这些不同模型相符。这些结果表明，*UCP*对于各种模型架构和大小都非常灵活。

<img src="../media/image6.png" style="width:5in;height:4in"
alt="A graph of training step Description automatically generated" />

图5：使用LLaMA模型架构的训练曲线。源是TP=PP=DP=2。训练在第101次迭代时使用新目标TP=DP=2, PP=1和TP=PP=2, DP=1恢复

<img src="../media/image7.png" style="width:5in;height:4in"
alt="A graph with numbers and lines Description automatically generated" />

图6：使用BLOOM模型架构的训练曲线。源是TP=2, PP=24, DP=8。训练在第94767次迭代时使用新目标TP=2, DP=4, PP=24恢复。

<img src="../media/image8.png" style="width:5in;height:4in"
alt="A graph of training step Description automatically generated" />

图7：使用Mixtral-MoE模型架构变种的训练曲线。源是TP=1, PP=2, DP=4。训练在第501次迭代时使用新目标TP=PP=DP=2恢复。

# DeepSpeed通用检查点的普遍可用性

我们很高兴发布DeepSpeed通用检查点。DeepSpeed通用检查点已与Megatron-DeepSpeed的重构版本完全集成，并可通过DeepSpeed和Megatron-DeepSpeed的GitHub仓库访问。详细的使用教程可在[DeepSpeed教程页面](https://www.deepspeed.ai/tutorials/universal-checkpointing/)上找到。

我们欢迎来自更广泛开源社区的贡献和合作。DeepSpeed通用检查点是大规模AI训练和推理DeepSpeed生态系统的一部分。有关所有DeepSpeed技术和创新的更多详细信息，请访问我们的[网站](https://www.deepspeed.ai/)并在X（前Twitter）（[英文](https://twitter.com/MSFTDeepSpeed)，[日文](https://twitter.com/MSFTDeepSpeedJP)）和[中文知乎](https://www.zhihu.com/people/deepspeed)上关注我们。

# 致谢和贡献
我们感谢伊利诺伊大学厄巴纳-香槟分校、Statosphere和英特尔Habana的合作。

贡献者：
Xinyu Lian $^1$, Sam Ade Jacobs $^2$, Lev Kurilenko $^2$, Masahiro Tanaka $^2$,
Stas Bekman $^3$, Olatunji Ruwase $^2$, Minjia Zhang $^1$, Moshe Island $^4$

1: 伊利诺伊大学厄巴纳-香槟分校
2: 微软
3: Statosphere
4: 英特尔Habana
