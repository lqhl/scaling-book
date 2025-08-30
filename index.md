---
layout: distill
title: "如何扩展你的模型"
subtitle: "从系统视角看TPU上的大语言模型"
# permalink: /main/
description: "训练大语言模型（LLMs，Large Language Models）常常感觉像炼金术，但理解和优化模型性能不必如此。本书旨在揭开扩展语言模型的科学：TPU（和GPU）如何工作以及它们如何相互通信，LLM如何在真实硬件上运行，以及如何在训练和推理期间并行化模型以实现大规模高效运行。如果你曾经想知道'训练这个LLM应该花费多少成本'或'我自己服务这个模型需要多少内存'或'什么是AllGather'，我们希望这本书对你有用。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "第0部分：介绍"

next_section_url: roofline
next_section_name: "第1部分：性能上限分析"

bibliography: main.bib

citation: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 高层概述
  - name: 章节链接

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习在很大程度上仍然像一种黑魔法，但优化模型性能不必如此——即使在大规模情况下也是如此！相对简单的原则适用于各个层面——从处理单个加速器到处理数万个加速器——理解这些原则可以让你做很多有用的事情：

- 大致估算模型各部分接近理论最优值的程度
- 在不同规模下做出明智的并行方案选择（如何在多个设备间分割计算）
- 估算训练和运行大型Transformer模型所需的成本和时间
- 设计利用[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[特性](https://arxiv.org/abs/2007.00072)的算法
- 基于对当前算法性能限制的明确理解来设计硬件

**预期背景：** 我们假设你对LLM（大语言模型）和Transformer架构有基本了解，但不一定了解它们在大规模下的运行方式。你应该了解LLM训练的基础知识，最好对JAX有一些基本熟悉。一些有用的背景阅读可能包括关于Transformer架构的[这篇博客文章](https://jalammar.github.io/illustrated-transformer/)和[原始Transformer论文](https://arxiv.org/abs/1706.03762)。还可以查看[这个列表](conclusion#further-reading)获取更多有用的并发和未来阅读材料。

**目标与反馈：** 到最后，你应该能够自信地估算在给定硬件平台上Transformer模型的最佳并行方案，以及训练和推理大致需要多长时间。如果不能，请给我们发邮件或留言！我们很想知道如何能让这些内容更清晰。

<p markdown=1 class="announce">你可能也会喜欢阅读新的[第12节](gpus)关于NVIDIA GPU的内容！</p>

### 为什么你应该关心？

三四年前，我认为大多数ML研究人员不需要理解本书中的任何内容。但如今即使是"小型"模型也运行得如此接近硬件极限，以至于进行新颖的研究需要你考虑大规模效率。<d-footnote>历史上，ML研究在系统创新和软件改进之间遵循着某种滴答周期。Alex Krizhevsky不得不编写复杂的CUDA代码来使CNN快速运行，但几年之内，像Theano和TensorFlow这样的库意味着你不需要这样做。也许这里也会发生同样的情况，几年后本书中的所有内容都会被抽象掉。但扩展定律已经将我们的模型永久推向了硬件的极限，在不久的将来，进行前沿研究似乎将不可避免地与理解如何有效地将模型扩展到大型硬件拓扑结构联系在一起。</d-footnote> **如果在基准测试中获得20%的胜利却以20%的性能上限效率为代价，那么这种胜利是无关紧要的。** 有前途的模型架构经常失败，要么是因为它们_无法_在大规模下高效运行，要么是因为没有人投入工作使它们做到这一点。

**"模型扩展"的目标是能够增加用于训练或推理的芯片数量，同时实现成比例的线性吞吐量增长。** 这被称为"*强扩展*"。虽然添加额外的芯片（"并行性"）通常会减少计算时间，但也会带来芯片间增加的通信成本。当通信时间超过计算时间时，我们变得"通信受限"，无法进行强扩展。<d-footnote>随着计算时间的减少，你通常还会在单个芯片层面遇到瓶颈。你闪亮的新TPU或GPU可能额定执行500万亿次操作/秒，但如果不小心，如果它被在内存中移动参数所拖累，它可能很容易只做到十分之一。每芯片计算、内存带宽和总内存的相互作用对于扩展故事至关重要。</d-footnote> 如果我们足够了解硬件以预测这些瓶颈将在何处出现，我们就可以设计或重新配置模型来避免它们。<d-footnote>硬件设计者面临相反的问题：构建提供足够计算、带宽和内存的硬件，同时最小化成本。你可以想象这个"协同设计"问题有多紧张：你必须赌注在第一批芯片实际可用时算法会是什么样子，通常是2到3年后。TPU的故事在这个游戏中是一个响亮的成功。矩阵乘法是一个独特的算法，因为它使用的每字节内存的FLOPs（浮点操作次数）比几乎所有其他算法都多（每字节N FLOPs），早期的TPU及其脉动阵列架构在当时比GPU实现了更好的性能/美元比。TPU是为ML工作负载设计的，带有TensorCore的GPU也在迅速改变以填补这个利基市场。但你可以想象，如果神经网络没有起飞，或者以某种TPU（本质上不如GPU灵活）无法处理的基本方式发生变化，那将是多么昂贵。</d-footnote>

*我们在本书中的目标是解释TPU（和GPU）硬件如何工作，以及Transformer架构如何演变为在当前硬件上表现良好。我们希望这对设计新架构的研究人员和使用当前一代LLM运行的工程师都有用。*

## 高层概述

本书的整体结构如下：

[第1节](roofline)解释了性能上限分析以及哪些因素可能限制我们的扩展能力（通信、计算和内存）。[第2节](tpus)和[第3节](sharding)详细讨论了TPU如何工作，既作为单个芯片，又作为具有有限带宽和延迟的互连芯片系统——这一点至关重要。我们将回答以下问题：

* 特定大小的矩阵乘法应该需要多长时间？在什么情况下它受计算、内存或通信带宽的限制？
* TPU如何连接形成训练集群？系统的每个部分有多少带宽？
* 在多个TPU之间收集、分散或重新分配数组需要多长时间？
* 我们如何高效地乘以在不同设备上分布不同的矩阵？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图：</b>来自<a href='tpus'>第2节</a>的图表，显示TPU如何执行逐元素乘积。根据数组大小和各种链路的带宽，我们可能会发现自己受计算限制（使用完整的硬件计算能力）或受通信限制（受内存加载瓶颈）。" %}

五年前，ML有着丰富多彩的架构景观——ConvNets、LSTMs、MLPs、Transformers——但现在我们主要只有Transformer<d-cite key="transformers"></d-cite>。我们坚信值得理解Transformer架构的每一个部分：每个矩阵的确切大小、归一化发生的位置、每个部分有多少参数和FLOPs<d-footnote>FLoating point OPs，基本上是所需的加法和乘法总数。虽然许多来源将FLOPs理解为"每秒操作数"，但我们使用FLOPs/s来明确表示这一点。</d-footnote>。[第4节](transformers)仔细介绍了这种"Transformer数学"，展示了如何计算训练和推理的参数数量和FLOPs。这告诉我们模型将使用多少内存，我们将花费多少时间在计算或通信上，以及注意力相对于前馈块何时变得重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b>标准Transformer层，每个矩阵乘法（matmul）显示为圆圈内的点。所有参数（不包括归一化）以紫色显示。<a href='transformers'>第4节</a>更详细地讲解了这个图表。" %}

[第5节：训练](training)和[第7节：推理](inference)是本文的核心，我们在这里讨论基本问题：给定某个大小的模型和一定数量的芯片，我如何并行化我的模型以保持在"强扩展"状态？这是一个简单的问题，但答案却出奇地复杂。在高层面上，有4种主要的并行技术用于在多个芯片上分割模型（**数据**、**张量**、**流水线**和**专家**并行），以及许多其他减少内存需求的技术（**重计算**、**优化器/模型分片（又名ZeRO）**、**主机卸载**、**梯度累积**）。我们在这里讨论其中的许多技术。

我们希望到这些章节结束时，你应该能够为自己选择新的架构或设置。[第6节](applied-training)和[第8节](applied-inference)是将这些概念应用于LLaMA-3（一个流行的开源模型）的实践教程。

最后，[第9节](profiling)和[第10节](jax-stuff)探讨了如何在JAX中实现其中一些想法，以及当出现问题时如何分析和调试代码。[第12节](gpus)是一个新章节，也深入探讨了GPU。

在整个过程中，我们尝试给你一些问题自己解决。请不必有压力阅读所有章节或按顺序阅读。请留下反馈。目前，这是一个草稿，将继续修订。谢谢！

*我们要感谢James Bradbury和Blake Hechtman，他们推导出了本文档中的许多想法。*

<h3 markdown=1 class="next-section">事不宜迟，[这里是第1节](roofline)关于TPU性能上限分析。</h3>

## 章节链接

*这个系列可能比需要的要长，但我们希望这不会阻止你。前三章是预备知识，如果熟悉可以跳过，尽管它们介绍了后面使用的符号。最后三个部分可能是最实用的，因为它们解释了如何与真实模型一起工作。*

**第一部分：预备知识**

* [**第1章：性能上限分析简介**](roofline)。算法受三个因素限制：计算、通信和内存。我们可以使用这些来近似估算算法的运行速度。

* [**第2章：如何理解TPU**](tpus)。TPU如何工作？这如何影响我们可以训练和服务的模型？

* [**第3章：分片矩阵及其乘法**](sharding)。在这里，我们通过我们最喜欢的操作：（分片）矩阵乘法，来解释模型分片和多TPU并行性。

**第二部分：Transformer**

* [**第4章：你需要知道的所有Transformer数学**](transformers)。Transformer在前向和后向传递中使用多少FLOPs？你能计算参数数量吗？其KV缓存（Key-Value缓存，用于存储注意力机制的键值对以减少重复计算）的大小？我们在这里详细讲解这些数学。

* [**第5章：如何为训练并行化Transformer**](training)。FSDP。Megatron分片。流水线并行。给定一定数量的芯片，如何尽可能高效地训练给定大小和批次的模型？

* [**第6章：在TPU上训练LLaMA 3**](applied-training)。我们如何在TPU上训练LLaMA 3？需要多长时间？成本是多少？

* [**第7章：Transformer推理全解析**](inference)。训练完模型后，我们需要部署它进行推理。推理增加了一个新的考虑因素——延迟——并改变了内存格局。我们将讨论解耦式服务如何工作以及如何思考KV缓存（Key-Value缓存，用于存储注意力机制的键值对以减少重复计算）。

* [**第8章：在TPU上服务LLaMA 3**](applied-inference)。在TPU v5e上服务LLaMA 3需要多少成本？延迟和吞吐量之间的权衡是什么？

**第三部分：实践教程**

* [**第9章：如何分析TPU代码性能**](profiling)。真实的LLM从来不像上面的理论那么简单。这里我们解释JAX + XLA堆栈以及如何使用JAX/TensorBoard性能分析器来调试和修复实际问题。

* [**第10章：在JAX中编程TPU**](jax-stuff). JAX提供了一系列神奇的API用于并行化计算，但你需要知道如何使用它们。有趣的示例和练习题。

**第四部分：结论和附加内容**

* [**第11章：结论和进一步阅读**](conclusion). 关于TPU和LLM的总结性思考和进一步阅读材料。

* [**第12章：如何理解GPU**](gpus). 关于GPU的附加章节，介绍GPU如何工作、如何联网，以及它们的性能上限分析与TPU有何不同。

