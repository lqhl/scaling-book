---
layout: distill
title: "如何扩展你的模型"
subtitle: "TPU 上的 LLM 系统视图"
# permalink: /main/
description: "训练 LLM (大型语言模型) 通常感觉像炼金术, 但理解和优化模型的性能并非必须如此. 本书旨在揭开语言模型扩展科学的神秘面纱: TPU (和 GPU) 如何工作以及它们之间如何通信, LLM 如何在真实硬件上运行, 以及如何在训练和推理过程中并行化你的模型, 使其在大规模上高效运行. 如果你曾想过“训练这个 LLM 应该有多昂贵”或“我自己需要多少内存来服务这个模型”或“什么是 AllGather”, 我们希望这本书对你有所帮助."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: roofline
next_section_name: "Part 1: Rooflines"

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
  - name: "内容大纲"
  - name: "各章节链接"

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

深度学习的很多方面仍然可以归结为一种黑魔法, 但优化模型性能并非必须如此 —— 即使是在巨大规模上! 相对简单的原则无处不在 —— 从处理单个加速器到数万个加速器 —— 理解它们可以让你做很多有用的事情:

-   粗略估算你的模型部分距离其理论最优值有多近.
-   在不同规模上就不同的并行方案做出明智的选择 (如何将计算分配到多个设备上).
-   估算训练和运行大型 Transformer 模型所需的成本和时间.
-   设计能够利用 [特定](https://arxiv.org/abs/2205.14135) [硬件](https://arxiv.org/abs/1911.02150) [特性](https://arxiv.org/abs/2007.00072) 的算法.
-   在明确了解当前算法性能限制的情况下设计硬件.

**背景要求:** 我们假设你对 LLM 和 Transformer 架构有基本的了解, 但不一定了解它们如何大规模运行. 你应该了解 LLM 训练的基础知识, 最好对 JAX 有一些基本的熟悉. 一些有用的背景阅读可能包括关于 Transformer 架构的 [这篇博客文章](https://jalammar.github.io/illustrated-transformer/) 和 [原始的 Transformer 论文](https://arxiv.org/abs/1706.03762). 另外, 请查看 [这个列表](conclusion#further-reading) 以获取更多有用的同步和未来阅读材料.

**目标与反馈:** 读完本书后, 你应该能够轻松地为给定硬件平台上的 Transformer 模型估算最佳并行方案, 以及大致的训练和推理时间. 如果你做不到, 请给我们发邮件或留言! 我们很想知道如何能让内容更清晰.

<p markdown=1 class="announce">你可能也会喜欢阅读关于 NVIDIA GPU 的新 [第 12 节](gpus)!</p>

### 你为什么应该关心?

三四年前, 我认为大多数机器学习研究人员不需要理解本书中的任何内容. 但如今, 即使是“小型”模型也运行得如此接近硬件极限, 以至于进行新颖的研究需要你考虑规模化的效率.<d-footnote>从历史上看, 机器学习研究遵循着系统创新和软件改进之间的某种“滴答”循环. Alex Krizhevsky 不得不编写复杂的 CUDA 代码来使 CNN 变快, 但在几年内, 像 Theano 和 TensorFlow 这样的库意味着你不再需要这样做了. 也许这里也会发生同样的事情, 本书中的所有内容在几年后都将被抽象掉. 但是, 扩展定律已将我们的模型永久地推向了硬件的最前沿, 而且在不久的将来, 进行前沿研究似乎将与理解如何有效地将模型扩展到大型硬件拓扑结构密不可分.</d-footnote> **在基准测试中取得 20% 的胜利, 如果以 20% 的屋顶线效率为代价, 那是无关紧要的.** 有前途的模型架构之所以经常失败, 要么是因为它们*无法*在规模上高效运行, 要么是因为没有人投入精力去实现它们.

**“模型扩展”的目标是能够在增加用于训练或推理的芯片数量的同时, 实现吞吐量的成比例线性增长.** 这被称为“*强扩展*”. 尽管增加额外的芯片 (“并行化”) 通常会减少计算时间, 但它也带来了芯片之间通信增加的代价. 当通信时间超过计算时间时, 我们就会变得“受通信限制”, 无法实现强扩展.<d-footnote>随着计算时间的减少, 你通常也会在单个芯片级别面临瓶颈. 你闪亮的新 TPU 或 GPU 可能额定每秒执行 500 万亿次操作, 但如果你不小心, 它同样很容易只做到十分之一, 如果它被在内存中移动参数所拖累. 单芯片计算, 内存带宽和总内存之间的相互作用对扩展至关重要.</d-footnote> 如果我们对硬件有足够的了解, 能够预测这些瓶颈将在何处出现, 我们就可以设计或重新配置我们的模型以避免它们.<d-footnote>硬件设计者面临着相反的问题: 构建能够为我们的算法提供恰到好处的计算, 带宽和内存, 同时最小化成本的硬件. 你可以想象这个“协同设计”问题有多么紧张: 你必须押注于当第一批芯片实际可用时算法会是什么样子, 这通常是 2 到 3 年之后的事情. TPU 的故事是这场博弈中一个响亮的成功. 矩阵乘法是一种独特的算法, 因为它每字节内存使用的 FLOPs (浮点运算次数) 比几乎任何其他算法都多 (每字节 N FLOPs), 早期的 TPU 及其脉动阵列架构在当时实现了比 GPU 好得多的性能/价格比. TPU 是为 ML 工作负载设计的, 而带有 TensorCores 的 GPU 也在迅速改变以填补这一空白. 但你可以想象, 如果神经网络没有兴起, 或者发生了某些根本性的变化, 那代价会有多大... [截断]</d-footnote>

*我们在本书中的目标是解释 TPU (和 GPU) 硬件如何工作, 以及 Transformer 架构如何演变为在当前硬件上表现良好. 我们希望这对于设计新架构的研究人员和致力于让当前一代 LLM 快速运行的工程师都有用.*

## 内容大纲

本书的总体结构如下:

[第 1 节](roofline) 解释了屋顶线分析以及哪些因素会限制我们的扩展能力 (通信, 计算和内存). [第 2 节](tpus) 和 [第 3 节](sharding) 详细讨论了 TPU 的工作原理, 既包括作为单个芯片, 也包括 —— 至关重要的 —— 作为一个具有有限带宽和延迟的互连芯片的互连系统. 我们将回答以下问题:

*   特定大小的矩阵乘法应该花费多长时间? 在什么时候它会受到计算, 内存或通信带宽的限制?
*   TPU 是如何连接在一起形成训练集群的? 系统的每个部分有多少带宽?
*   在多个 TPU 之间收集, 分散或重新分配数组需要多长时间?
*   我们如何有效地乘以分布在不同设备上的矩阵?

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图:</b> 来自 <a href='tpus'>第 2 节</a> 的图表, 显示了 TPU 如何执行逐元素乘积. 根据我们数组的大小和各种链接的带宽, 我们可能会发现自己受计算限制 (使用全部硬件计算能力) 或受通信限制 (受内存加载瓶颈)." %}

五年前, 机器学习拥有丰富多彩的架构格局 —— ConvNets, LSTMs, MLPs, Transformers —— 但现在我们主要只有 Transformer<d-cite key="transformers"></d-cite>. 我们坚信, 了解 Transformer 架构的每个部分都是值得的: 每个矩阵的确切大小, 归一化发生在哪里, 每个部分有多少参数和 FLOPs<d-footnote>浮点运算, 基本上是所需加法和乘法的总数. 虽然许多资料将 FLOPs 理解为“每秒操作数”, 但我们使用 FLOPs/s 来明确表示这一点.</d-footnote>. [第 4 节](transformers) 仔细地讲解了这种“Transformer 数学”, 展示了如何计算训练和推理的参数和 FLOPs. 这告诉我们模型将使用多少内存, 我们将在计算或通信上花费多少时间, 以及注意力何时会相对于前馈块变得重要.

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图:</b> 一个标准的 Transformer 层, 每个矩阵乘法 (matmul) 显示为一个圆圈内的点. 所有参数 (不包括范数) 都以紫色显示. <a href='transformers'>第 4 节</a> 更详细地介绍了这个图." %}

[第 5 节: 训练](training) 和 [第 7 节: 推理](inference) 是本文的核心, 我们在其中讨论了基本问题: 给定某个大小的模型和一定数量的芯片, 我该如何并行化我的模型以保持在“强扩展”状态? 这是一个简单的问题, 却有着惊人复杂的答案. 从高层次上讲, 有 4 种主要的并行技术用于在多个芯片上拆分模型 (**数据**, **张量**, **流水线** 和 **专家**), 以及许多其他技术来减少内存需求 (**重物质化**, **优化器/模型分片 (又名 ZeRO)**, **主机卸载**, **梯度累积**). 我们在这里讨论其中的许多技术.

我们希望在这些章节结束时, 你应该能够自己为新的架构或设置选择它们. [第 6 节](applied-training) 和 [第 8 节](applied-inference) 是将这些概念应用于 LLaMA-3 (一个流行的开源模型) 的实践教程.

最后, [第 9 节](profiling) 和 [第 10 节](jax-stuff) 介绍了如何在 JAX 中实现其中一些想法, 以及在出现问题时如何分析和调试代码. [第 12 节](gpus) 是一个深入探讨 GPU 的新章节.

在整本书中, 我们都试图给你一些问题让你自己解决. 请不要有压力阅读所有章节或按顺序阅读. 并请留下反馈. 目前, 这是一个草稿, 将继续修订. 谢谢!

*我们要感谢 James Bradbury 和 Blake Hechtman, 他们推导出了本文档中的许多想法.*

<h3 markdown=1 class="next-section">话不多说, [这里是关于 TPU 屋顶线的第 1 节](roofline).</h3>

## 各章节链接

*这个系列可能比它需要的要长, 但我们希望这不会阻止你. 前三章是预备知识, 如果熟悉可以跳过, 尽管它们介绍了后面使用的符号. 最后三个部分可能是最实用的, 因为它们解释了如何处理真实模型.*

**第一部分: 预备知识**

*   [**第 1 章: 屋顶线分析简介**](roofline). 算法受到三件事的限制: 计算, 通信和内存. 我们可以用这些来近似我们的算法将运行多快.

*   [**第 2 章: 如何看待 TPU**](tpus). TPU 是如何工作的? 这如何影响我们可以训练和服务的模型?

*   [**第 3 章: 分片矩阵以及如何乘以它们**](sharding). 在这里, 我们通过我们最喜欢的操作: (分片) 矩阵乘法来解释模型分片和多 TPU 并行.

**第二部分: Transformers**

*   [**第 4 章: 你需要知道的所有 Transformer 数学**](transformers). Transformer 在其前向和后向传播中使用了多少 FLOPs? 你能计算出参数的数量吗? 它的 KV cache (键值缓存) 的大小? 我们在这里详细讲解了这些数学.

*   [**第 5 章: 如何为训练并行化 Transformer**](training). FSDP. Megatron 分片. 流水线并行. 给定一定数量的芯片, 我如何以尽可能高效的方式用给定的批量大小训练给定大小的模型?

*   [**第 6 章: 在 TPU 上训练 LLaMA 3**](applied-training). 我们将如何在 TPU 上训练 LLaMA 3? 需要多长时间? 成本是多少?

*   [**第 7 章: 关于 Transformer 推理的一切**](inference). 一旦我们训练了一个模型, 我们就必须为它提供服务. 推理增加了一个新的考虑因素 —— 延迟 —— 并改变了内存格局. 我们将讨论分离式服务的工作原理以及如何看待 KV caches.

*   [**第 8 章: 在 TPU 上服务 LLaMA 3**](applied-inference). 在 TPU v5e 上服务 LLaMA 3 的成本是多少? 延迟/吞吐量的权衡是什么?

**第三部分: 实践教程**

*   [**第 9 章: 如何分析 TPU 代码**](profiling). 真实的 LLM 从不像上面的理论那么简单. 在这里, 我们解释了 JAX + XLA 堆栈以及如何使用 JAX/TensorBoard 分析器来调试和修复实际问题.

*   [**第 10 章: 在 JAX 中编程 TPU**](jax-stuff). JAX 提供了一系列用于并行化计算的神奇 API, 但你需要知道如何使用它们. 有趣的例子和已解决的问题.

**第四部分: 结论和附加内容**

*   [**第 11 章: 结论和进一步阅读**](conclusion). 关于 TPU 和 LLM 的总结性思考和进一步阅读.

*   [**第 12 章: 如何看待 GPU**](gpus). 一个关于 GPU 的附加章节, 它们如何工作, 如何联网, 以及它们的屋顶线与 TPU 有何不同.