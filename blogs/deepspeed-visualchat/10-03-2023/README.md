<div align="center">

# DeepSpeed-VisualChat: Improve Your Chat Experience with Multi-Round Multi-Image Inputs

</div>

<div align="center">

<img src="../assets/images/hero-figure.png" width="1000px" alt="DeepSpeed-VisualChat!"/>

</div>

To cite DeepSpeed-VisualChat, please cite our [arxiv report](https://arxiv.org/abs/2309.14327):

```
@article{yao2023deepspeed-visualchat,
  title={{DeepSpeed-VisualChat: Multi-Round Multi-Image Interleave Chat via Multi-Modal Causal Attention}},
  author={Zhewei Yao and Xiaoxia Wu and Conglong Li and Minjia Zhang and Heyang Qin and Olatunji Ruwase and Ammar Ahmad Awan and Samyam Rajbhandari and Yuxiong He},
  journal={arXiv preprint arXiv:2309.14327},
  year={2023}
}
```
# 1. Overview
Large Language models (LLMs), such as GPT and LLaMa, have showcased exceptional prowess in a myriad of text generation and comprehension tasks, especially when subjected to zero-/few-shot learning, particularly after instructed fine-tuning. However, to equip AI agents for diverse tasks, one critical feature that needs to be incorporated is multi-modal capability; for instance, the AI agent should be able to read images, hear voices, watch videos, etc. This capability is largely absent in solely text-based LLMs.

Recently, one of the research/practice mainstreams has begun exploring the incorporation of visual capability into LLMs, especially enabling LLMs to understand images by inserting raw pictures (referred to as large visual language models, or LVLMs in short).

The main caveats of the majority of existing works are:
* The focus is predominantly on tasks related to a single image, such as visual question answering and captioning, or on handling multiple images that require concurrent input. Neither approach adeptly manages interleaved image-and-text input.
* The scalability of the system is limited to models with ~10B parameters, which is about an order of magnitude smaller than largest open-sourced models.

However, for a genuine AI chat agent, the content of inputs could be multiple images interleaved with text, a situation rarely addressed by current works. Also, the generation capability of LLMs grows quickly as the model size increases. Therefore, focusing system capability on ~10B models limits further exploration of the potential of LVLMs.

To resolve these issues, we are introducing DeepSpeed-VisualChat (see [arxiv report](https://arxiv.org/abs/2309.14327) for more details) with the following new features:

* ***Fully Open-Sourced Multi-round Multi-image Framework with Unprecedented Scalability***: DeepSpeed-VisualChat, one of the pioneering fully open-sourced frameworks, enables multi-round and multi-image dialogues, accommodating interleaved text-and-image inputs. We leverage DeepSpeed to enhance our training with a 2B visual encoder and a 70B LLaMA-2 decoder model, illustrating the remarkable scalability of our framework.
* ***Multi-Modal Causal Attention (MMCA)***
We devise a novel MMCA for multi-modal models that computes attention weights independently across various modalities. MMCA achieves objectives analogous to conventional cross-attention mechanisms but offers enhanced causal attention interpretations for generative tasks, eliminating the need for additional modules or parameters. It also presents superior training data efficiency compared to standard causal attention.
* ***Data Blending for Interleaved Inputs*** To facilitate conversations with interleaved modalities, DeepSpeed-VisualChat employs assorted data blending techniques on existing datasets, overcoming the shortage of interleaved text-and-image inputs in most available open-source datasets.



# 2 Model architecture overview
<div align="center">
  <img src="../assets/images/model.png" alt="model arch" width="400"/>

  *Figure 1: Model architecture illustration.*

</div>

The model architecture of DeepSpeed-VisualChat, as depicted in *Figure 1*, is composed of three components: a visual encoder, such as CLIP; a language decoder, such as LLaMa-7B; and a feature alignment linear projection layer. Most parts of the model are frozen, with only the embedding of the language model and the linear projection layer being trainable. Consequently, the total number of trainable parameters ranges from approximately O(10M) (LLaMa-2-13B) to O(100M) (LLaMa-2-70B).

# 3. DeepSpeed multi-modal causal attention

There are two common attention mechanisms used to connect the visual and textual components in a multi-modal model: causal attention, as used in MiniGPT and QWen-VL, and cross attention, as used in Otter and Flamingo.

<div align="center">
  <img src="../assets/images/attention.png" alt="Different attention mechanisms" width="1000"/>

  *Figure 2: Different Attention Mechanisms: Examine the differing attention mechanisms using an input sentence "User: Please describe the image." coupled with three Image tokens (I-token1, I-token2, I-token3). On the left, we demonstrate standard causal attention, treating image tokens as text. In the middle, we present cross attention applied to images, while maintaining standard causal attention for text tokens. On the right, we illustrate our innovative multi-modal attention proposal where image tokens only perform self-attention, and text tokens attend to text/image tokens independently, highlighted with an orange mask. This mechanism is defined by: softmax($`QK^T \odot M_1`$)+ softmax($`QK^T \odot M_2`$) with Q and K as query and key, $`M_1`$=[M==1], and $`M_2`$=[M==2], with M $`\in`$ R<sup>10x10</sup> in this case.*
</div>


<b>Causal Attention (CA)</b>: The CA-based method simply projects visual features (i.e., the features from the output of the final visual encoder layer) into textual features and combines them with the normal textual features after the textual embedding layer to feed into LLMs. The benefit of CA is that it's a natural extension of the original attention mechanism in LLMs, and as such, it doesn't introduce any extra modules or parameters. However, this approach raises some intuitive problems:

* For a visual token, it attends to previous visual and textual tokens, even though visual tokens are already fully encoded in a bidirectional manner and do not need further attention to other visual tokens or previous textual tokens.
* For a textual token, the model needs to learn how to distribute its attention weights between its previous textual and image tokens. Due to these issues, we found that the data efficiency of CA in LVLMs is often problematic. To address this, LLaVA and QWen-VL require visual-language pretraining to fully align visual features with textual features.

<b>Cross Attention (CrA)</b>: The alternative, cross attention (CrA), along with CA, exhibits better data efficiency but also comes with a few drawbacks:

* It introduces new parameters to the model. For example, Otter has more than 1.5 billion trained parameters compared to the millions of trained parameters in LLaVA due to the new parameters introduced by cross attention. This significantly increases the training cost and memory requirements.
* It requires careful design if an image is introduced in the middle of a conversation during training, as previous text tokens should not be able to attend to the image.

<b>Multi-Modal Causal Attention Mechanism (MMCA)</b>: To overcome these issues, we propose a new multi-modal causal attention mechanism (MMCA), which has both benefits, i.e., similar parameter efficiency as CA and similar data efficiency as CrA. The overall idea is as follows:

* For visual tokens, they only attend to themselves, as visual tokens are  encoded by the visual encoder.
* For textual tokens, they attend to all their previous tokens. However, they have two separate attention weight matrices for their previous textual tokens and image tokens.

The intuition behind the second point of MMCA is that the attention weight for one modality may affect the other modality. For instance, a textual token may pay more attention to textual information than visual information. Therefore, if the attention weight matrix is normalized across both modalities, the attention score for visual tokens might be very small. Refer to *Figure 2* for a visualization of the three attention mechanisms.


<b>Demo Results.</b> We begin by showcasing various examples that highlight the capabilities of DeepSpeed-VisualChat in single-image visual language conversations, employing different attention mechanisms. In these experiments, we employ the LLaMA2-7B language model in conjunction with the QWen-VL visual-encoder as our visual encoder. These two models are connected via a straightforward linear projection layer. Our model underwent training on two LLaVa datasets. As demonstrated in *Figure 3* and *Figure 4*, DeepSpeed-VisualChat, when coupled with MMCA, effectively discerns visual details in images and furnishes coherent responses to user queries.
Furthermore, DeepSpeed-VisualChat exhibits a more comprehensive and precise grasp of image details compared to alternative attention mechanisms, such as the use of combined masks from both causal attention and cross attention. It is also evident that, in contrast to the combination of CrA and CA, as well as MMCA, CA alone may exhibit slightly more errors (*Figure 3*) and capture a lower degree of reasoning capability (*Figure 4*).

<div align="center">
  <img src="../assets/images/cat-chat.png" alt="Small kitten" width="600"/>

  *Figure 3: Example visual and language inputs that demonstrate the output comparison between (1) the standard causal attention (CA) (2)  the standard causal attention combined with cross-attention (CA+ CrA) and (3) the special multi-modal causal attention (MMCA) in DeepSpeed-VisualChat.*

</div>

<div align="center">
  <img src="../assets/images/lake-chat.png" alt="Beautiful lake" width="600"/>

  *Figure 4: DeepSpeed-VisualChat accurately identifies the scene as a beautiful lake and offers a set of plausible suggestions. In contrast, the baseline misinterprets the image as containing “dock with a boat ramp”.*

</div>

# 4. Data blending
We used 9 datasets from 3 sources as described in our [arxiv report](https://arxiv.org/abs/2309.14327). A critical missing element for enabling multi-round and multi-image conversations is the absence of adequate data. The sole source of multi-round multi-image data we located is the SparklesDialogue dataset, which contains a mere 6520 samples. To address this limitation, we employed two methods to synthesize multi-round multi-image data from existing single-image or single-round data: simple data concatenation and LLaVA-Otter data blending.

## 4.1 Simple data concatenation
For the "llava" and "llava_dial" datasets utilized by the LLaVA model, each sample comprises single/multi-round conversations for a single image. To simulate scenarios where a user sequentially asks questions about multiple images, we conducted straightforward data post-processing for these two datasets. Specifically, we randomly concatenated different numbers of samples into a single sample. In the case of "llava," we concatenated 1 to 3 samples, while for "llava_dial," we concatenated 1 to 2 samples.

## 4.2 LLaVA-Otter data blending
We noticed that the llava and llava_dial datasets used by LLaVA model and the otter_mimicit_cgd dataset used by the Otter model all use the COCO train2017 images. For the llava and llava_dial datasets, each sample includes a single/multi-round conversations for a single image. For the otter_mimicit_cgd dataset, each sample includes a single-round conversation for a pair of images. This enables us to build a synthesized multi-round multi-image data llava_otter_blend as a more natural blending: for each sample in the otter_mimicit_cgd dataset, we look for llava and llava_dial samples that use the same image, and then build a new sample in a "llava/llava_dial conversations then otter_mimicit_cgd conversation" fashion.

<div align="center">
  <img src="../assets/images/data-blending.png" alt="Friends" width="600"/>

  *Figure 5:  A data sample after LLaVA-Otter data blending. Gray dialog boxes are from LLaVA datasets, and orange ones are from Otter dataset.*
</div>

# 5. Demonstration
We trained our DeepSpeed-VisualChat-13B model with a 2B visual encoder and the 13B LLaMA model on several open-sourced datasets. DeepSpeed-VisualChat-13B shows image captioning capabilities (*Figure 6--8*), counting and text reading (*Figure 6*), celebrity recognition (*Figure 7*), storytelling (*Figure 8*), etc.

<div align="center">
  <img src="../assets/images/friends.png" alt="Friends" width="600"/>

  *Figure 6:  DeepSpeed-VisualChat can count the number of people in the image and read the text in the first image. It also demonstrates cross-image understanding.*
</div>


<div align="center">
  <img src="../assets/images/ceos.png" alt="CEO" width="600"/>

  *Figure 7:  DeepSpeed-VisualChat can recognize celebrities and associate them with their achievements.*
</div>


<div align="center">
  <img src="../assets/images/zootopia.png" alt="Zootopia" width="600"/>

  *Figure 8: DeepSpeed-VisualChat can tell stories and recognize movies.*
</div>


# 6. How to begin with DeepSpeed-VisualChat
DeepSpeed-VisualChat is an easy-to-use training framework with great scalability, having been tested up to LLaMa-2-70B models so far. We adopt a unified instruction tuning format for all experiments, and the template is shown below.
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

The training experience of DeepSpeed-VisualChat is straightforward and convenient. Here we give an example based on the CLIP visual encoder and the LLaMa-7B model:
```
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-VisualChat/
pip install -r requirements.txt
cd training
bash training_scripts/run_7b.sh
```

The trained checkpoint will be automatically saved in a Hugging Face-compatible version and can be used to launch your own visual chat API:
```
cd ../chat
bash chat_scripts/run.sh # You need to change necessary variables, e.g, ckpt path
```
To support larger model inference, we have incorporated Hugging Face large model inference into our DeepSpeed-VisualChat API. Therefore, users can choose a different number of GPUs based on the GPU memory capacity and the model size.

Please refer to our [GitHub Landing Page](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat) for more details.

# 7. Release: Try DeepSpeed-VisualChat today!

We are very excited to share that DeepSpeed-VisualChat is now open-sourced and available to the AI community.

* To get started, please visit our GitHub page for DeepSpeed-VisualChat: [GitHub Landing Page](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat)

* We will continue to improve DeepSpeed-VisualChat with your feedback and support. Our [roadmap](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat/README.md#-deepspeed-visualchats-roadmap-) shows currently supported features as well as ones that are planned for the future.


DeepSpeed-VisualChat is a component of the larger DeepSpeed ecosystem, which includes a range of Deep Learning systems and modeling technologies. To learn more,

* Please visit our [website](https://www.deepspeed.ai/) for detailed blog posts, tutorials, and helpful documentation.
* Follow us on our [English X(Twitter)](https://twitter.com/MSFTDeepSpeed), [Japanese X(Twitter)](https://twitter.com/MSFTDeepSpeedJP), and [Chinese Zhihu](https://www.zhihu.com/people/deepspeed) for latest news on DeepSpeed.

We welcome your contributions to DeepSpeed! We encourage you to report issues, contribute PRs, and join discussions on the [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) page. Please see our [contributing guide](https://github.com/microsoft/DeepSpeed/blob/master/CONTRIBUTING.md) for more details. We are open to collaborations with universities, research labs, companies, such as those working together on deep learning research, applying DeepSpeed to empower real-world AI models and applications, and so on. For such requests (and other requests unsuitable for GitHub), please directly email to deepspeed-info@microsoft.com.

* "Star" our [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/) and [DeepSpeedExamples GitHub](https://github.com/microsoft/DeepSpeedExamples/) repositories if you like our work!
