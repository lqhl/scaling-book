---
layout: distill
title: "在 TPU 上训练 LLaMA 3"
# permalink: /main/
description: "让我们仔细看看如何使用我们在上一节中学到的知识在 TPU v5p 上训练 LLaMA 3 模型. 它们有多大? 不同配置的训练有多昂贵? 它们是如何分片的? 让我们通过一些粗略的估算, 看看前面的章节如何映射到真实模型上."
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
  - name: "LLaMA 3 是什么样的?"
  - name: "计算参数和 FLOPs"
  - name: "如何为训练分片 LLaMA 3-70B"
  - name: "已解决的问题"

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

_本节的目标是将上一节的结果应用于一个非常实际的问题: 训练 LLaMA 3 模型家族 (群). 与前面的章节不同, 我们希望你自己完成大部分工作. 因此, 我们隐藏了每个部分的答案, 以便你可以先尝试回答. 试着用笔和纸手动计算一下!_

### LLaMA 3 是什么样的?

LLaMA-3 模型家族<d-cite key="llama3"></d-cite> 包括 3 个主要模型: LLaMA 3 8B, 70B 和 405B. 我们将主要关注 70B, 并将 8B 和 405B 留给你在末尾的问题部分进行探索. 这是 LLaMA 3-70B 的架构, 取自 LLaMA [HuggingFace 页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json).

| **超参数** | **值** |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

为了突出显示这有多容易找到, 这是配置本身, 以及一个映射:

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_为许多不同的开源 LLM 制作一个包含这些数字的大表格是很有用的, 这样你就可以快速比较它们所做的设计决策._

### 计算参数和 FLOPs

**问题:** 从这个表中, 我们能计算出 LLaMA 3-70B 的参数数量吗? 🤫 让我们应用[第 4 节](../transformers)的内容, 看看我们是否能得到 70B!

| 参数 | 公式 | 数量 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW 参数 | d_model * d_ff * 3 (用于 gelu + 输出投影) * n_layers | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9** |
| 词汇表参数 | 2 (输入和输出嵌入) * n_embeddings * d_model | 2 * 128,256 * 8,192 = **2.1e9** |
| 注意力参数 | n_layers * [ 2 (用于 q 嵌入和连接的输出投影) * d_model * n_heads * d_qkv + 2 (用于 k 和 v) * d_model * n_kv_heads * d_qkv] | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
| | | 56.3e9 + 2.1e9 + 12e9 = **70.4e9** |

太棒了! 我们得到了我们期望的数字. 你会注意到, 正如预期的那样, FFW 参数完全主导了总参数数量, 尽管注意力也不容忽视.

<p markdown=1 class="takeaway">**要点**: MLP 块中的 3 个大权重矩阵比 Transformer 中所有其他数组都大得多, 以至于我们在推理模型内存或 FLOPs 时通常几乎可以忽略所有其他参数. 对于 LLaMA 3-70B, 它们占 70B 参数中的 56B.</p>

现在让我们看看 FLOPs! *记住[第 4 节](../transformers)中训练的一般规则.*

**问题:** LLaMA-3 每个训练步骤每个 token 执行多少 FLOPs? _这有助于我们确定整个训练过程的成本._

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 正如[第 4 节](../transformers)所示, 我们每个 token 大约执行 $$6 \cdot \text{参数数量}$$ FLOPs, 所以这里大约是 `6 * 70e9 = 4.2e11` FLOPs / token. 这大约是每个 token 每步半个 TFLOP. 假设我们受计算限制, 在单个 TPU v5p 芯片上, 假设完美的 FLOPs 利用率, 这大约需要 `4.2e11 / 4.59E+14 = 1ms`.

{% enddetails %}

**问题:** LLaMA 3 训练了大约 15 万亿个 token. 这总共是多少 FLOPs?

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 这很简单, 就是 `4.2e11 * 15e12 = 6.3e24 FLOPs`. 6.3 yottaFLOPs. 这很多! 在单个 TPU 上, 这将需要 `6.3e24 / 4.59E+14 = 435 年`. 这也很多!

{% enddetails %}

**问题:** 假设我们想在一个完整的 TPU v5p pod 上训练, 该 pod 有 16x20x28 = 8960 个芯片. 在 bfloat16 中, 以 40% MFU 进行训练需要多长时间, 假设我们受计算限制?

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 我们知道每个 TPU v5p 每秒可以执行 4.59e14 FLOPs. 在 40% MFU 下, 这大约需要 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 秒`. **这大约是 44 天!** 这是相当合理的, 假设我们真的能达到 40% MFU.

{% enddetails %}

**问题:** LLaMA 3-70B 的预训练批量大小约为 4M token. 我们至少需要多少个 TPU 才能用这个批量大小进行训练? _你可以假设 bfloat16 参数和 float32 优化器状态, 并且你每层对梯度进行 4 次检查点._

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 这个问题主要是在问内存使用情况, 因为这是可用计算的唯一严格限制. 在训练期间, 我们有三个主要的 HBM 用途: 模型参数, 优化器状态和梯度检查点. 如果我们假设 bfloat16 权重, float32 优化器状态和一个*非常*保守的梯度检查点方案 (每层 4 次), 我们有:

| **参数** | 2 * 70GB | ~140GB |
| **优化器状态** | 8 * 70GB | ~560GB |
| **梯度检查点** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **总计** | | ~21.6TB |

这里的总数大约是 21.6TB. 你会注意到, 即使采用非常保守的检查点方案, 梯度检查点也严重主导了内存情况. 我们技术上可以每层只进行 1 次检查点, 或者进行微批处理, 但这是一个合理的画面. 在这些假设下, 由于每个 TPU v5p 有 96GB 的 HBM, 我们需要 `21.6e12 / 96e9 = 225` 个 TPU. 这实际上并不多!

*我们为什么不这样做?* 嗯, 因为训练需要 `44 天 * 8960 / 225 = 1752 天`. 这将近四年. **这很多.** 尽管如此, 这清楚地表明, 我们使用这些大型集群不是因为我们受内存限制, 而是因为我们需要额外的 FLOPs.

{% enddetails %}

**问题:** 在与上述问题相同的假设下, 如果我们使用 8960 个 TPU v5p 芯片, 我们每个芯片将使用多少内存?

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 我们的总内存仍然是大约 21.6TB, 所以每个芯片我们将使用大约 2.4GB, 这基本上不算什么. 如果我们进行更激进的检查点, 例如每层 12 个检查点, 我们每个芯片也只有 8GB. 在这些规模的训练中, 我们远未达到内存限制.

{% enddetails %}

<p markdown=1 class="takeaway">**要点**: 从技术上讲, 即使在非常小的拓扑结构上训练非常大的模型也是可能的, 但需要注意的是, 它们可能会花费很长时间. 能够计算训练运行的总 FLOPs, 让我们能够通过假设适度的 MFU 和已知的拓扑结构来粗略估算其训练时间.</p>

### 如何为训练分片 LLaMA 3-70B

让我们坚持上面的设置, 假设我们想在 8960 个芯片的 TPU v5p pod 上用 4M token 的批量大小 (每批 1024 个序列, 长度为 4096) 训练 LLaMA 3-70B. 让我们讨论一下这个模型的最佳分片策略是什么.

**问题:** 在上述假设下, 我们可以仅用 FSDP 训练我们的模型吗? 首先, 让我们假设我们不能进行任何序列/上下文并行. _这应该是你首先想到的想法, 因为它很简单, 如果可行, 不会引入额外的通信._

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 这个答案会有点迂腐. 如上所述, LLaMA 3-70B 最初是用 4K 长度的序列进行训练的, 所以 4M token 的批量大小给了我们一个 1024 的*序列批量大小*. 这意味着我们实际上只能进行纯数据并行/FSDP, 最多 1024 个芯片, *因为我们只有那么多序列可以进行数据并行*. 所以从“完全数据并行, 没有额外通信”的简单意义上来说, 答案是否定的. 下一个问题将回答一个稍微不那么迂腐的版本.

{% enddetails %}

**问题:** 让我们放宽不进行任何序列分片的要求. 如果我们允许自己在批处理*和*序列轴上都进行 FSDP, 我们能仅用 FSDP 在 8960 个芯片上训练 LLaMA 3-70B 吗?

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 现在我们允许自己也进行序列/上下文并行, 我们可以扩展得更多. 首先让我们计算我们每个设备的批量大小. 如果我们进行 8960 路 FSDP, 我们最终每个 TPU 的批量大小为 `4 * 1024 * 1024 / 8960 = 468` 个 token. 我们从上一节知道, 当 $$\text{每个设备的批量大小} < 2550 / M_X$$ 时, 我们会受到 FSDP 的 ICI 限制. 由于我们可以在这里用一个完整的 3D pod 投入 3 个轴, 这会给我们一个 850 的下限, 我们远低于这个值. **所以答案是否定的, 即使有 3 个轴. 我们将完全受通信限制.**

{% enddetails %}

**问题:** 现在让我们看看混合张量并行和 FSDP. 是否存在某种组合能让我们保持受计算限制? 如果是, 我们应该进行多少 FSDP 和张量并行?

{% details 点击这里查看答案, 在你思考之后! %}

**答案**: 首先让我们检查一下这是否可行. 我们知道, 如果我们每个芯片的批量大小小于 $2550^2 / 2F = 113$, 我们就会受通信限制. 正如我们上面看到的, 我们略高于这个值. 所以这很棒! 现在要选择最佳的 FSDP 数量, 我们可以使用公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

四舍五入到一个合理的 2 的倍数, 这给了我们大约 2048 路 FSDP 和 4 路模型并行. 这应该能很好地工作!

{% enddetails %}

<p markdown=1 class="takeaway">**要点**: 我们可以用 4M token 的批量大小, 在一个完整的 TPU v5p pod 上, 混合使用数据并行 (1024 路), 序列并行 (2 路) 和张量并行 (4 路) 来训练 LLaMA-3, 而不受通信限制. 如果我们尝试纯 FSDP 或 FSDP + 序列并行, 我们将受通信限制. 我们在上一节中提出的方程非常实用.</p>

## 已解决的问题

**问题 1 [扩展 LLaMA 70B 到更多芯片]:** 假设我们想用相同的批量大小在 4 个 pod 上训练 LLaMA 3-70B. 我们会使用什么并行方案? 我们会受计算限制还是通信限制? 训练大约需要多长时间? *确保使用正确的屋顶线限制.*

**问题 2 [LLaMA 405B]:**

(a) 使用 LLaMA 3-405B [配置](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json), 写一个包含所有关键超参数的表格, 如上所示. 这个模型总共有多少参数? 每个训练步骤有多少 FLOPs? 如果我们训练 15T token, 我们执行多少 FLOPs?

(b) 假设我们想在 8 个 TPU v5p pod 上进行训练. 我们会使用什么并行方案? 训练需要多长时间? 会受计算限制还是通信限制?

<h3 markdown=1 class="next-section">第 6 节到此结束. 第 7 节, 关于 Transformer 推理, 点击 [这里](../inference).</h3>
