---
layout: distill
title: "在TPU上训练LLaMA 3"
# permalink: /main/
description: "让我们仔细看看我们如何使用前几节学到的知识在TPU v5p上训练LLaMA 3模型。它们有多大？不同配置下的训练成本如何？它们如何分片？让我们通过一些粗略估计来分析前几节如何映射到真实模型。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "Part 5: Training"

next_section_url: ../inference
next_section_name: "Part 7: Inference"

bibliography: main.bib

giscus_comments: true

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
  - name: "LLaMA 3 是什么样的？"
  - name: "计算参数和 FLOPs"
  - name: "如何对 LLaMA 3-70B 进行分片训练"
  - name: "练习题"

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

_本节的目标是将前一节的结果应用到一个非常实际的问题：训练 LLaMA 3 系列（群组）模型。与前面的章节不同，我们希望您自己完成大部分工作。因此，我们隐藏了每个小节的答案，以便您可以先尝试回答。试着拿起笔手动计算吧！_

### LLaMA 3 是什么样的？

LLaMA-3 模型系列<d-cite key="llama3"></d-cite>包含 3 个主要模型：LLaMA 3 8B、70B 和 405B。我们将主要关注 70B，而 8B 和 405B 留给您在最后的问题部分中探索。以下是 LLaMA 3-70B 的架构，取自 LLaMA [HuggingFace 页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)。

| **hyperparam**              | **value** |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

为了突出显示找到这些配置的容易程度，这里是配置文件本身以及映射关系：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_为许多不同的开源 LLM 制作一个包含这些数字的大表格是很有用的，这样您就可以快速比较它们所做的设计决策。_

### 计算参数和 FLOPs

**问题：** 从这个表格中，我们能否计算出 LLaMA 3-70B 的参数数量？🤫 让我们应用[第 4 节](../transformers)的内容，看看能否得到 70B！

| 参数             | 公式                                                                                                                                              | 数量                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW 参数         | d_model * d_ff * 3 (用于 gelu + 输出投影) * n_layers                                                                                              | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                    |
| 词汇参数         | 2 (输入和输出嵌入) * n_embeddings * d_model                                                                                                      | 2 * 128,256 * 8,192 = **2.1e9**                              |
| 注意力参数       | n_layers * [ 2 (用于 q 嵌入和连接的输出投影) * d_model * n_heads * d_qkv + 2 (用于 k 和 v) * d_model * n_kv_heads * d_qkv]                       | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

太棒了！我们得到了预期的数字。您会注意到，正如预期的那样，FFW 参数在总体参数计数中占主导地位，尽管注意力机制也不容忽视。

<p markdown=1 class="takeaway">**要点**：MLP 块中的 3 个大型权重矩阵比 Transformer 中的所有其他数组大得多，因此在推理模型内存或 FLOPs 时，我们通常可以几乎忽略所有其他参数。对于 LLaMA 3-70B，它们代表了 700 亿参数中的 560 亿。</p>

现在让我们看看 FLOPs！*记住[第 4 节](../transformers)中关于训练的一般规则。*

**问题：** LLaMA-3 在每个 token 的每个训练步骤中执行多少 FLOPs？_这有助于我们确定整个训练过程的成本。_

{% details 想好答案后点击这里！ %}

**答案**：如[第 4 节](../transformers)所示，每个 token 大约执行 $$6 \cdot \text{参数数量}$$ FLOPs，所以这里大约是 `6 * 70e9 = 4.2e11` FLOPs / token。这大约是每个 token 每步半 TFLOP。假设我们受计算限制，在单个 TPU v5p 芯片上大约需要 `4.2e11 / 4.59E+14 = 1ms`，假设 FLOPs 利用率完美。

{% enddetails %}

**问题：** LLaMA 3 训练了大约 15 万亿个 token。总共执行了多少 FLOPs？

{% details 想好答案后点击这里！ %}

**答案**：这很简单，总共就是 `4.2e11 * 15e12 = 6.3e24 FLOPs`。6.3 yottaFLOPs。这是一个巨大的数字！在单个 TPU 上这将需要 `6.3e24 / 4.59E+14 = 435 年`。这也是一个很长的时间！

{% enddetails %}

**问题：** 假设我们要在完整的 TPU v5p pod（16x20x28 = 8960 个芯片）上训练。在 bfloat16 下以 40% MFU 训练需要多长时间，假设我们受计算限制？

{% details 想好答案后点击这里！ %}

**答案**：我们知道每个 TPU v5p 可以执行 4.59e14 FLOPs / 秒。在 40% MFU 下，这将需要大约 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 秒`。**这大约是 44 天！**这是相当合理的，假设我们真的能达到 40% MFU。

{% enddetails %}

**问题：** LLaMA 3-70B 使用大约 4M token 的批量大小进行预训练。我们需要至少多少个 TPU 来用这个批量大小进行训练？_您可以假设 bfloat16 参数和 float32 优化器状态，并且每层 checkpoint 梯度 4 次。_

{% details 想好答案后点击这里！ %}

**答案**：这个问题主要询问内存使用情况，因为这是对可用计算的唯一严格约束。在训练期间，我们有三个主要的 HBM 用途：模型参数、优化器状态和梯度 checkpoint。如果我们假设 bfloat16 权重、float32 优化器状态，以及一个_非常_保守的梯度 checkpoint 方案（每层 4 次），我们有：

| **参数** | 2 * 70GB | ~140GB |
| **优化器状态** | 8 * 70GB | ~560GB |
| **梯度 Checkpoint** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **总计**                |                         | ~21.6TB |

这里的总计大约是 21.6TB。您会注意到，即使使用非常保守的 checkpoint 方案，梯度 checkpoint 也强烈主导了内存情况。技术上我们可以做到每层 1 个 checkpoint，或者进行微批处理，但这是一个合理的图景。基于这些假设，由于每个 TPU v5p 有 96GB 的 HBM，我们需要 `21.6e12 / 96e9 = 225` 个 TPU。实际上这并不多！

*为什么我们不这样做？* 嗯，因为训练将需要 `44 天 * 8960 / 225 = 1752 天`。这将近四年。**这太长了。**尽管如此，这清楚地表明我们使用这些大型集群不是因为受内存限制，而是因为我们需要额外的 FLOPs。

{% enddetails %}

**问题：** 在与上述问题相同的假设下，如果我们使用 8960 个 TPU v5p 芯片，每个芯片将使用多少内存？

{% details 想好答案后点击这里！ %}

**答案**：我们的总内存仍然是大约 21.6TB，所以每个芯片我们将使用大约 2.4GB，这基本上没什么。如果我们进行更激进的 checkpoint，例如每层 12 个 checkpoint，我们每个芯片仍然只需要 8GB。在这种规模的训练中，我们远未达到内存限制。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：技术上即使在很小的拓扑结构上训练非常大的模型也是可能的，需要注意的是它们可能会花费很长时间。能够计算训练运行的总 FLOPs 允许我们通过假设适度的 MFU 和已知的拓扑结构来估算其训练时间。</p>

### 如何对 LLaMA 3-70B 进行分片训练

让我们坚持上面的设置，假设我们要在 8960 个芯片的 TPU v5p pod 上用 4M token 批量大小（每批 1024 个长度为 4096 的序列）训练 LLaMA 3-70B。让我们讨论这个模型的最佳分片策略是什么。

**问题：** 在上述假设下，我们能单独使用 FSDP 训练我们的模型吗？首先，假设我们不能进行任何序列/上下文并行。_这应该是您的第一个想法，因为它很简单，如果可行的话不会引入额外的通信。_

{% details Click here for the answer, once you've thought about it! %}

**答案**：这个答案会有点学究气。如上所述，LLaMA 3-70B 最初是用长度为 4K 的序列训练的，所以 4M token 的批量大小给我们一个*序列批量大小*为 1024。这意味着我们最多只能在 1024 个芯片上进行纯数据并行/FSDP，_因为这是我们进行数据并行的序列数量_。所以"完全数据并行且没有额外通信"这个简单意义上的答案是否定的。下一个问题将回答一个稍微不那么学究气的版本。

{% enddetails %}

**问题：** 让我们放宽不进行任何序列分片的要求。如果我们允许自己在批量_和_序列轴上都进行 FSDP，我们能在 8960 个芯片上仅使用 FSDP 训练 LLaMA 3-70B 吗？

{% details Click here for the answer, once you've thought about it! %}

**答案**：既然我们允许自己进行序列/上下文并行，我们可以扩展得更多。首先让我们计算每设备批量大小。如果我们进行 8960 路 FSDP，我们最终得到每 TPU 批量大小为 `4 * 1024 * 1024 / 8960 = 468 tokens`。我们从上一节知道，当 $$\text{每设备批量大小} < 2550 / M_X$$ 时，我们会因 FSDP 而受到 ICI 限制。由于我们可以在这里用完整的 3D pod 分配 3 个轴，这将给我们一个下限 850，我们远低于这个值。**所以答案是否定的，即使有 3 个轴。我们将严重受通信限制。**

{% enddetails %}

**问题：** 现在让我们看看混合的 tensor parallelism 和 FSDP。是否存在某种组合让我们保持计算限制？如果存在，我们应该进行多少 FSDP 和 tensor parallelism？

{% details Click here for the answer, once you've thought about it! %}

**答案**：首先让我们检查这是否合适。我们知道如果我们的每芯片批量大小小于 $2550^2 / 2F = 113$，我们将受到通信限制。如上所示，我们稍微高于这个值。所以这很好！现在选择最佳 FSDP 数量，我们可以使用公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

四舍五入到合理的 2 的倍数，这给我们大约 2048 路 FSDP 和 4 路 model parallelism。这应该效果很好！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们可以在完整的 TPU v5p pod 上用 4M token 批量大小训练 LLaMA-3，使用数据并行（1024 路）、序列并行（2 路）和 tensor parallelism（4 路）的混合，而不受通信限制。如果我们尝试进行纯 FSDP 或 FSDP + 序列并行，我们将受到通信限制。我们在前一节中推导的方程非常实用。</p>

## 练习题

**问题 1 [将 LLaMA 70B 扩展到更多芯片]：** 假设我们要在 4 个 pod 上用相同的批量大小训练 LLaMA 3-70B。我们会使用什么并行方案？我们会受到计算还是通信限制？训练大约需要多长时间？*确保使用正确的 roofline 限制。*

**问题 2 [LLaMA 405B]：**

(a) 使用 LLaMA 3-405B [配置](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)，写一个包含所有关键超参数的表格，如上所示。这个模型总共有多少参数？每个训练步骤有多少 FLOPs？如果我们训练 15T token，我们执行多少 FLOPs？

(b) 假设我们要在 8 个 TPU v5p pod 上训练。我们会使用什么并行方案？训练需要多长时间？会受到计算还是通信限制？

<h3 markdown=1 class="next-section">这就是第 6 节的全部内容。关于 Transformer 推理的第 7 节，请点击[这里](../inference)。</h3>