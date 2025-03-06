<div align="center">

# DeepNVMe: 通过I/O优化提高深度学习应用性能

</div>

# 引言

深度学习（DL）在语言、语音、视频和多模态应用等重要人工智能领域不断推动着前所未有的进展。这些进展的关键因素是模型大小、序列长度和硬件并行性等多个维度上的显著可扩展性。从系统角度来看，深度学习的可扩展性给计算、内存、通信和存储等关键子系统带来了巨大的压力。然而，现有的深度学习优化工作大多忽略了存储子系统，使得数据加载、模型检查点和卸载等I/O操作成为大规模深度学习中的主要瓶颈。为了解决这个问题，DeepSpeed开发了一整套I/O优化技术，统称为DeepNVMe。

DeepNVMe通过加速I/O操作和减少硬件需求，提高了I/O受限的深度学习应用的性能和效率。它通过利用存储创新，如非易失性内存快速通道（NVMe）固态硬盘（SSD）和NVIDIA Magnum IO<sup>TM</sup> GPUDirect®存储（GDS）实现这一目标。在本文中，我们通过微基准测试和推理应用来展示DeepNVMe的优势。在对Azure NC96ads\_A100\_v4虚拟机进行的实验中，我们观察到DeepNVMe能充分利用可用的NVMe带宽进行GPU或CPU内存的数据传输，读取速度达到10GB/秒，写入速度达到5GB/秒。

# 背景

高性能访问持久存储是许多计算领域（包括深度学习）中的一个常见挑战。因此，已经提出了大量的硬件和软件解决方案。DeepNVMe基于三种解决方案： (1) NVMe SSDs，(2) NVIDIA GDS，(3) Linux异步I/O（libaio）。我们将简要介绍每项技术。

NVMe SSDs是基于闪存的存储设备，正在取代传统的硬盘驱动器（HDD），成为现代服务器的主要持久存储。例如，Azure NC96ads\_A100\_v4虚拟机配备了四个NVMe SSD，每个SSD可提供3.25GB/秒的读取速度，并且可以组合成RAID-0配置，理论上的总读取带宽为13GB/秒。NVIDIA GDS可以实现NVMe和GPU内存之间的直接数据传输，从而避免了传统使用中间CPU内存（缓冲区）方法的低效。NVIDIA GDS在CUDA 11.4及以上版本中可用。最后，libaio是Linux引入的异步I/O栈，它比传统的I/O栈更有效地提取NVMe SSD等高速存储设备的原始性能。

# DeepNVMe: 深度学习I/O优化模块

DeepNVMe是一个Python模块，我们开发时遵循了两个关键设计原则。首先，它利用上述存储技术，实现了强大的优化，如非阻塞I/O操作、批处理I/O操作提交、单个I/O操作的并行化以及轻量级运行时。其次，它通过一个简单的POSIX-like接口让用户使用I/O优化，便于深度学习应用集成，同时避免了底层技术的复杂性。

# 评估

我们的实验在Azure NC96ads\_A100\_v4虚拟机上进行，实验设置的详细信息见表1。对于多设备实验，SSD是以RAID-0配置组合使用的。

<img src="../media/table1.png" style="width:6.5in;height:3.42153in" />

<div align="center">
表1: 实验设置详细信息
</div>

## 微基准性能测试

我们使用了三种基准测试工具进行评估。第一个是fio，这是一个用C语言编写的流行I/O基准测试工具。第二个是来自NVIDIA的gdsio，用于基准测试GDS性能。第三个是ds\_io，这是我们创建的Python工具，便于与DeepNVMe集成，并且更能代表常见的基于Python的深度学习应用。

## 通过NVMe扩展CPU缓冲区，从而提高I/O性能

我们的第一组微基准评估使用fio和ds\_io，测量1GB数据在NVMe和CPU内存之间的传输性能。我们配置fio使用libaio后端进行这些实验。结果总结在图1中，我们可以得出两个结论。首先，DeepNVMe表现出高性能，尽管它更能代表深度学习应用，但其性能与fio大致相当。其次，DeepNVMe的I/O性能几乎与可用的NVMe带宽成线性扩展，达到了10GB/秒的读取速度和5GB/秒的写入速度。

<img src="../media/figure1.png" style="width:6.5in;height:3.42153in" />

<div align="center">
图1: 使用DeepNVMe扩展NVMe与CPU缓冲区之间的数据传输
</div>

## 通过NVMe扩展GPU缓冲区，从而提高I/O性能

我们的第二组微基准评估使用gdsio和ds\_io，测量1GB数据在NVMe和GPU内存之间的传输性能。在此实验中，我们配置ds\_io同时使用传统的缓冲区方法和更高效的GDS方法。结果总结在图2中，我们可以得出三个结论。首先，我们看到GDS提高了DeepNVMe的性能，相比传统缓冲区方法，速度提高了最多37%。其次，DeepNVMe表现出高性能，尽管它更能代表深度学习应用，但其性能与gdsio相匹配（有时甚至超过）。第三，我们看到DeepNVMe，无论是否使用GDS，都能根据可用的NVMe带宽扩展I/O性能。使用GDS时，DeepNVMe的读取速度最高达到9.6GB/秒，写入速度为5GB/秒；不使用GDS时，读取速度为7GB/秒，写入速度为4GB/秒。

<img src="../media/figure2.png" style="width:6.5in;height:3.42153in" />

<div align="center">
图2: 使用DeepNVMe扩展NVMe与GPU内存之间的数据传输
</div>

## ZeRO-Inference: 生成式AI性能

ZeRO-Inference是一项AI普及技术，通过使用DeepNVMe将模型权重卸载(Offload)到CPU或NVMe内存，降低了推理大规模模型的硬件成本。ZeRO-Inference非常适合于面向吞吐量的应用，如离线推理，和硬件预算有限的场景。我们使用token生成工作负载来评估DeepNVMe在NVMe卸载下的性能。

## 通过NVMe扩展的高性能卸载(Offload)

我们测量了在单个NVIDIA A100-80GB上推理LLAMA3-70B模型的生成吞吐量，使用512的提示长度、32的生成长度和96的批量大小。我们将NVMe SSD的数量从1扩展到4，并呈现了ZeRO-Inference在有GDS和没有GDS的情况下的结果，如图3所示。我们从这些结果中得出两个结论。首先，GDS始终提供比传统缓冲区方法更好的性能，token生成速度提高了10-18%。其次，DeepNVMe，无论是否使用GDS，都能根据可用的NVMe带宽扩展生成性能。在四个NVMe SSD的情况下，DeepNVMe的生成吞吐量分别为每秒7个token（使用GDS）和每秒6个token（不使用GDS）。我们的分析结果表明，DeepNVMe将在更多的NVMe带宽下继续扩展，是提升生成应用性能的经济选择。

<img src="../media/figure3.png" style="width:6.5in;height:3.42153in" />

<div align="center">
图3: 使用DeepNVMe通过NVMe卸载(offload)扩展LLAMA3-70B的token生成性能
</div>

# 总结

在本文中，我们介绍了DeepNVMe，一项为了解决I/O操作成为深度学习可扩展性关键瓶颈而创建的I/O优化技术。DeepNVMe通过基于流行存储技术（如NVMe SSD和NVIDIA GDS）的优化，实现了持久存储与深度学习应用内存之间的快速高效数据传输。我们展示了在Azure NC96ads\_A100\_v4虚拟机上，DeepNVMe通过NVMe卸载支持LLAMA3-70B的token生成，最高达到每秒7个token的生成吞吐量。DeepNVMe将在DeepSpeed版本>= 0.15.0中开源，并广泛发布。在未来的博客中，我们将报告DeepNVMe在其他I/O受限的深度学习应用中的改进，如模型检查点和数据加载。

# 致谢

这项工作是微软和NVIDIA之间深入合作的结果。贡献者包括微软的Joe Mayer、Martin Cai和Olatunji Ruwase；NVIDIA的Kiran Modukuri、Vahid Noormofidi、Sourab Gupta和Sandeep Joshi。
