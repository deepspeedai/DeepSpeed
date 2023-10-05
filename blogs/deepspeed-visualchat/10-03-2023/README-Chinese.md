
<div align="center">

# DeepSpeed-VisualChat：多轮图像+文字，为你展现不一样的AI聊天魅力

</div>

<div align="center">

<img src="../assets/images/hero-figure.png" width="1000px" alt="DeepSpeed-VisualChat!"/>

</div>

要引用 DeepSpeed-VisualChat，请引用我们的 [arxiv 报告](https://arxiv.org/abs/2309.14327)：


```
@article{yao2023deepspeed-visualchat,
  title={{DeepSpeed-VisualChat: Multi-Round Multi-Image Interleave Chat via Multi-Modal Causal Attention}},
  author={Zhewei Yao and Xiaoxia Wu and Conglong Li and Minjia Zhang and Heyang Qin and Olatunji Ruwase and Ammar Ahmad Awan and Samyam Rajbhandari and Yuxiong He},
  journal={arXiv preprint arXiv:2309.14327},
  year={2023}
}
```

# 1. 概述
大型语言模型 (LLMs)，如 GPT 和 LLaMa，在各种文本生成和理解任务中都展现出了卓越的能力，特别是在经过零次/少次学习（zero-/few-shot learning）或微调（instructed fine-tuning）后。然而，要让 AI 模型为多样化的任务做好准备，需要加入的一个关键特性是多模态能力；例如，AI 模型应该能够读取图像、听到声音、观看视频等。这种能力在纯文本基础的 LLMs 中基本上是不存在的。

最近，大量的研究项目开始探索将视觉能力引入到 LLMs 中，特别是通过插入图片输入使 LLMs 来理解图片（简称为大型视觉语言模型或 LVLMs）。

大多数现有工作的主要缺点是：
* 重点主要放在与单一图像相关的任务上，如视觉问题回答和字幕，或处理需要同时输入的多个图像。两种方法都不太擅长管理交错的图像和文本输入。
* 系统的可扩展性仅限于具有约 10B 参数的模型，这比最大的开源模型小了一个数量级。

然而，对于一个真正的 AI 聊天模型，输入内容可能是与文本交错的多个图像，这是目前的工作很少涉及的情况。此外，随着模型大小的增加，LLMs 的生成能力增长迅速。因此，将系统能力集中在约 10B 的模型上限制了对 LVLMs 潜力的进一步探索。

为了解决这些问题，我们推出了 DeepSpeed-VisualChat（请参阅 [arxiv 报告](https://arxiv.org/abs/2309.14327) 以获取更多详细信息），带有以下新特性：

* ***全开源多轮多图框架与前所未有的可扩展性***：DeepSpeed-VisualChat，作为开创性的全开源框架之一，支持多轮和多图对话，容纳交错的文本和图像输入。我们利用 DeepSpeed 提高我们的训练效果，使用一个 2B 的视觉编码器和一个 70B 的 LLaMA-2 解码器模型，展示了我们框架的显著可扩展性。
* ***多模态因果注意力 (MMCA)*** 我们为多模态模型设计了一个新的 MMCA 注意力机制，独立地计算各种模态的注意力权重。MMCA 达到了与传统交叉注意机制类似的目标，但为生成任务提供了增强的因果注意解释，消除了对额外模块或参数的需求。与标准的因果注意力相比，它还提供了更好的训练数据效率。
* ***交错输入的数据混合*** 为了促进交错模态的对话，DeepSpeed-VisualChat 在现有数据集上采用了各种数据混合技术，克服了大多数现有开源数据集中交错文本和图像输入的短缺。

# 2. 模型架构概述
<div align="center">
  <img src="../assets/images/model.png" alt="模型结构" width="400"/>

  *图 1：DeepSpeed-VisualChat 的模型架构示意图。*
</div>

如 *图 1* 所示，DeepSpeed-VisualChat 的模型架构由三个部分组成：一个视觉编码器，如 CLIP；一个语言解码器，如 LLaMa-7B；和一个特征对齐线性投影层。模型的大部分都是冻结的，只有语言模型的嵌入和线性投影层是可训练的。因此，可训练参数的总数大约在 O(10M) (LLaMa-2-13B) 到 O(100M) (LLaMa-2-70B) 之间。

# 3. DeepSpeed 多模态因果注意力

用于在多模态模型中连接视觉和文本组件的两种常见注意机制是：因果注意力，如在 MiniGPT 和 QWen-VL 中使用的，以及交叉注意力，如在 Otter 和 Flamingo 中使用的。

<div align="center">
  <img src="../assets/images/attention.png" alt="不同的注意机制" width="1000"/>

  *图 2：不同的注意机制：使用一个输入句子“用户：请描述这个图片。”和三个图像令牌（I-token1、I-token2、I-token3）来比较不同的注意机制。在左边，我们展示了标准的因果注意力，将图像令牌视为文本。在中间，我们展示了应用于图像的交叉注意力，同时保持文本令牌的标准因果注意力。在右边，我们展示了我们的创新 MMCA 注意力机制，其中图像令牌只执行自注意，文本令牌独立地注意文本/图像令牌，橙色为图像部分。这种机制由：softmax($`QK^T \odot M_1`$)+ softmax($`QK^T \odot M_2`$) 定义，其中 Q 和 K 分别为查询和密钥，$`M_1`$=[M==1]，和 $`M_2`$=[M==2]，其中 M $`\in`$ R<sup>10x10</sup>。*
</div>

<b>因果注意力 (CA)</b>：基于 CA 的方法简单地将视觉特征（即来自最终视觉编码器层输出的特征）投影到文本特征，并将它们与文本嵌入层后的正常文本特征组合，以送入 LLMs。CA 的好处是它是 LLMs 原始注意机制的自然扩展，因此，它不引入任何额外的模块或参数。但是，直觉上这种方法会带来一些问题：

* 每个视觉令牌会关注它之前的视觉和文本令牌。然而视觉令牌已经以双向方式完全编码，不需要进一步关注它之前的视觉和文本令牌。
* 对于一个文本令牌，模型需要学习如何在其之前的文本和图像令牌之间分配其注意权重。由于这些问题，我们发现 LVLMs 中 CA 的数据效率通常是有问题的。为了解决这个问题，LLaVA 和 QWen-VL 需要视觉-语言预训练来完全对齐视觉特征和文本特征。

<b>交叉注意力 (CrA)</b>：作为替代方案，交叉注意力 (CrA) 与 CA 的结合展示出更好的数据效率，但也带有一些缺点：

* 它为模型引入了新的参数。例如，具有交叉注意力引入的新参数的 Otter 拥有超过 15 亿的可训练参数。和 LLaVA 的百万级可训练参数相比，这大大增加了训练成本和内存需求。
* 如果在训练过程中中间引入了一个图像，需要仔细设计，因为先前的文本令牌不应该能够注意到图像。

<b>多模态因果注意机制 (MMCA)</b>：为了克服这些问题，我们提出了一种新的多模态因果注意机制 (MMCA)，它既有 CA 的参数效率，又有 CrA 的数据效率。总体思路如下：

* 对于视觉令牌，它们只关注自己，因为视觉令牌是由视觉编码器编码的。
* 对于文本令牌，它们关注所有以前的令牌。但是，对文本和图像令牌 MMCA 使用两个单独的注意权重矩阵。

MMCA 的第二点背后的直觉是，一个模态的注意权重可能会影响另一个模态。例如，文本令牌可能会比视觉信息更多地关注文本信息。因此，如果注意权重矩阵在两种模态之间进行归一化，那么视觉令牌的注意得分可能会非常小。请参考 *图 2* 以查看三种注意机制的可视化。

<b>演示结果。</b>我们首先通过几个例子展示在不同的注意机制下 DeepSpeed-VisualChat 的单图像视觉语言对话功能。在这些实验中，我们使用 LLaMA2-7B 语言模型和 QWen-VL 视觉编码器作为我们的视觉编码器。这两个模型通过一个简单的线性投影层连接在一起。这个模型在两个 LLaVa 数据集上进行了训练。正如 *图 3* 和 *图 4* 所示，当与 MMCA 配合使用时，DeepSpeed-VisualChat 有效地识别了图像中的视觉细节，对用户的问题提供了准确通顺的回答。
此外，与其他注意机制（如使用因果注意力和交叉注意力的组合）相比，MMCA 表现出更全面和精确的图像细节把握。与 CrA 和 CA 的组合以及 MMCA 相比，仅使用 CA 可能会显示出稍微多一些的错误（*图 3*）或导致较低的理解能力（*图 4*）。

<div align="center">
  <img src="../assets/images/cat-chat.png" alt="小猫咪" width="600"/>

  *图 3：示例视觉和语言输入，显示了（1）标准因果注意力 (CA) （2）与交叉注意力组合的标准因果注意力 (CA+ CrA) 和（3）DeepSpeed-VisualChat 中的特殊多模态因果注意力 (MMCA) 之间的输出比较。*
</div>

<div align="center">
  <img src="../assets/images/lake-chat.png" alt="美丽的湖泊" width="600"/>

  *图 4：DeepSpeed-VisualChat 准确地识别了场景是一个美丽的湖泊，并提供了一组合理的建议。相比之下，其他的注意力机制误解了图像认为其包含“带船坡的码头”。*
</div>

# 4. 数据混合
我们使用了 3 个来源的 9 个数据集，如我们的 [arxiv 报告](https://arxiv.org/abs/2309.14327) 所述。一个实现多轮和多图对话的关键缺失元素是没有足够的数据。我们找到的唯一的多轮多图数据来源是 SparklesDialogue 数据集，它只包含 6520 个样本。为了解决这个问题，我们采用了两种方法，从现有的单图或单轮数据中合成多轮多图数据：简单的数据连接和 LLaVA-Otter 数据混合。

## 4.1 简单数据连接
对于 LLaVA 模型使用的 "llava" 和 "llava_dial" 数据集，每个样本包括单图像的单轮/多轮对话。为了模拟用户依次询问多个图像的情况，我们对这两个数据集进行了简单的数据后处理。具体来说，我们随机将不同数量的样本连接成一个样本。在 "llava" 的情况下，我们连接了 1 到 3 个样本，而在 "llava_dial" 的情况下，我们连接了 1 到 2 个样本。

## 4.2 LLaVA-Otter 数据混合
我们注意到，LLaVA 模型使用的 llava 和 llava_dial 数据集以及 Otter 模型使用的 otter_mimicit_cgd 数据集都使用了 COCO train2017 图像。对于 llava 和 llava_dial 数据集，每个样本包括一个图像的单轮/多轮对话。对于 otter_mimicit_cgd 数据集，每个样本包括一对图像的单轮对话。这使我们能够构建一个合成的多轮多图数据 llava_otter_blend 作为更自然的混合：对于 otter_mimicit_cgd 数据集中的每个样本，我们寻找使用相同图像的 llava 和 llava_dial 样本，然后以 "llava/llava_dial 对话然后 otter_mimicit_cgd 对话" 的方式构建一个新样本。

<div align="center">
  <img src="../assets/images/data-blending.png" alt="朋友们" width="600"/>

  *图 5：经过 LLaVA-Otter 数据混合后的数据样本。灰色对话框来自 LLaVA 数据集，橙色对话框来自 Otter 数据集。*
</div>

# 5. 演示
我们在几个开源数据集上训练了我们的 DeepSpeed-VisualChat-13B 模型，该模型使用一个 2B 的视觉编码器和 13B 的 LLaMA 模型。DeepSpeed-VisualChat-13B 展示了图像字幕功能（*图 6--8*），计数和文本阅读（*图 6*），名人识别（*图 7*），讲故事（*图 8*）等。

<div align="center">
  <img src="../assets/images/friends.png" alt="朋友们" width="600"/>

  *图 6：DeepSpeed-VisualChat 可以计算图像中的人数，并读取第一张图像中的文本。它还展示了跨图像的理解。*
</div>

<div align="center">
  <img src="../assets/images/ceos.png" alt="CEO" width="600"/>

  *图 7：DeepSpeed-VisualChat 可以识别名人并将他们与其成就联系起来。*
</div>

<div align="center">
  <img src="../assets/images/zootopia.png" alt="疯狂动物城" width="600"/>

  *图 8：DeepSpeed-VisualChat 可以讲故事并识别电影。*
</div>

# 6. 如何开始使用 DeepSpeed-VisualChat
DeepSpeed-VisualChat 是一个易于使用的训练框架，具有很好的可扩展性，到目前为止已经在 LLaMa-2-70B 模型上进行了测试。我们为所有实验采用了统一的指令调优格式，模板如下所示。
```
<System Instruction>      % You are a powerful vision-language assistant.

### Image 1: <image>       % some image, e.g., cat-1.png
### Question: <question>   % please describe the image.
### Answer: <answer>       % It's a cute black cat.

### Image 2: <image>       % some image, e.g., cat-2.png
### Image 3: <image>       % some image, e.g., cat-3.png
### Question: <question>   % What's the difference between the three cats?
### Answer: <answer>       % The colors of the three cats are different.
...
```

使用 DeepSpeed-VisualChat 训练模型是简单和方便的。这里我们给出了基于 CLIP 视觉编码器和 LLaMa-7B 模型的一个例子：

```
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-VisualChat/
pip install -r requirements.txt
cd training
bash training_scripts/run_7b.sh
```

训练后的模型权重将自动保存为 Hugging Face 兼容版本，并且可以用于启动您自己的视觉聊天 API：
```
cd ../chat
bash chat_scripts/run.sh # You need to change necessary variables, e.g, ckpt path
```

为了支持更大的模型推理，我们已经将 Hugging Face 大模型推理集成到我们的 DeepSpeed-VisualChat API 中。因此，用户可以根据 GPU 内存容量和模型大小选择不同数量的 GPU。

请参考我们的 [GitHub 主页](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat) 了解更多细节。

# 7. 发布：今天尝试 DeepSpeed-VisualChat！

我们非常兴奋地分享 DeepSpeed-VisualChat 现已开源并供 AI 社区使用。

* 要开始使用，请访问我们的 DeepSpeed-VisualChat GitHub 页面：[GitHub 主页](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat)

* 我们将继续在您的反馈和支持下改进 DeepSpeed-VisualChat。我们的 [路线图](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat/README.md#-deepspeed-visualchats-roadmap-) 显示了目前支持的功能以及未来计划支持的功能。

DeepSpeed-VisualChat 是更大的 DeepSpeed 生态系统的一部分，其中包括一系列深度学习系统和建模技术。要了解更多信息，

* 请访问我们的 [网站](https://www.deepspeed.ai/) 了解详细的博客文章、教程和文档。
* 在我们的 [英文 X(Twitter)](https://twitter.com/MSFTDeepSpeed)、[日语 X(Twitter)](https://twitter.com/MSFTDeepSpeedJP) 和 [中文知乎](https://www.zhihu.com/people/deepspeed) 上关注我们，以获取 DeepSpeed 的最新消息。

我们欢迎您为 DeepSpeed 做出贡献！我们鼓励您报告问题、贡献 PRs、并在 [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) 页面上参加讨论。有关更多详细信息，请查看我们的 [贡献指南](https://github.com/microsoft/DeepSpeed/blob/master/CONTRIBUTING.md)。我们对与大学、研究实验室、公司等进行合作持开放态度，例如共同进行深度学习研究、应用 DeepSpeed 为现实世界的 AI 模型和应用提供支持等等。对于此类请求（以及其他不适合 GitHub 的请求），请直接发送电子邮件至 deepspeed-info@microsoft.com。

* 如果你喜欢我们的工作，请在 [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) 和 [DeepSpeedExamples GitHub](https://github.com/microsoft/DeepSpeedExamples/) 上为我们的仓库点“星”。
