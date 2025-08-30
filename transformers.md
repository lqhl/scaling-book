---
layout: distill
title: "您需要了解的所有Transformer数学"
# permalink: /main/
description: "在这里我们将快速回顾Transformer架构，特别是如何计算FLOPs、字节和其他感兴趣的量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding"
previous_section_name: "Part 3: Sharding"

next_section_url: ../training
next_section_name: "Part 5: Training"

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

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "Counting Dots"
  - subsections:
    - name: "Forward and reverse FLOPs"
  - name: "Transformer Accounting"
  - name: "Global FLOPs and Params Calculation"
  - name: "Miscellaneous Math"
  - subsections:
    - name: "Sparsity and Mixture-of-Experts"
    - name: "Gradient checkpointing"
    - name: "Key-Value (KV) caching"
  - name: "What Should You Take Away from this Section?"
  - name: "A Few Problems to Work"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: How does Flash Attention work?"

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

## 计算点积

让我们从以下形状的向量 $$x$$、$$y$$ 和矩阵 $$A$$、$$B$$ 开始：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{数组}  & \textrm{形状} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end {array}
$$

- 点积 $$x \cdot y$$ 需要 $$P$$ 次 _加法_ 和 _乘法_，总共是 $$2P$$ 次浮点运算。
- 矩阵-向量乘积 $$Ax$$ 沿着 $$A$$ 的行进行 $$N$$ 次点积，需要 $$2NP$$ FLOPs。
- 矩阵-矩阵乘积 $$AB$$ 对 $$B$$ 的 $$M$$ 列中的每一列进行矩阵-向量乘积，总共需要 $$2NPM$$ FLOPs。
- 一般来说，如果我们有两个高维数组 $$C$$ 和 $$D$$，其中一些维度是 <span style="color:red">收缩维度</span>，一些是 <span style="color:blue">批处理维度</span>。（例如 $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$）那么这个收缩的 FLOPs 成本是所有 $$C$$ 和 $$D$$ 维度乘积的两倍，其中批处理和收缩维度只计算一次，（例如 $$2\blue{GH}IJMN\red{KL}$$）。注意，只有当维度在两个乘数中都出现时才是批处理维度。（还要注意，如果没有收缩维度且这只是逐元素乘积，则不会应用因子 2。）

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{FLOPs} & \textrm{数据} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end {array}
$$

请注意，对于矩阵-矩阵乘法，*计算量* 以三次方 $$O(N^3)$$ 缩放，而数据传输仅以二次方 $$O(N^2)$$ 缩放 - 这意味着当我们扩大矩阵乘法规模时，达到计算饱和极限变得*更容易*。这是极其不寻常的，并且在很大程度上解释了为什么我们使用以矩阵乘法为主的架构 - 它们适合扩展！

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向和反向 FLOPs

在训练期间，我们并不特别关心给定矩阵乘法的结果；我们真正关心的是其导数。这意味着我们在反向传播期间执行显著更多的 FLOPs。

如果我们假设 **B** 只是更大网络中的一个矩阵，而 **A** 是我们的输入激活，且 **C = A B**，那么损失 **L** 对 **B** 的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

这是一个外积，需要 $2NPM$ FLOPs 来计算（因为它在 $N$ 维度上收缩）。同样，损失对 **A** 的导数是

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

再次需要 $2NPM$ FLOPs，因为 **dL/dC** 是大小为 $$[N, M]$$ 的（余）向量。虽然这个量不是对参数的导数，但它用于计算网络前层的导数（例如，就像上面的 dL/dC 用于计算 dL/dB）。

将它们相加，我们看到**在训练期间，我们总共有 6NPM FLOPs**，而在推理期间只有 2NPM：前向传播 2NPM，反向传播 4NPM。由于 PM 是矩阵中的参数数量，这是著名的 Transformer 训练期间 FLOPs 近似值 $$6 * \text{参数数量} * \text{令牌数量}$$ 的最简单形式：每个令牌需要 $$6 * \text{参数数量}$$ FLOPs。我们将在下面展示更正确的推导。

## Transformer 核算

Transformer 是未来。嗯，至少它们是现在。也许几年前，它们只是众多架构之一。但今天，了解架构的几乎所有细节都是值得的。我们不会重新介绍架构，但 [这篇博客](https://jalammar.github.io/illustrated-transformer/) 和 [原始 Transformer 论文](https://arxiv.org/abs/1706.03762) 可能是有帮助的参考资料。

这是 Transformer 解码器架构的基本图：

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b>此图显示了标准 Transformer 的一层，从上到下流动。我们使用单字母约定来描述 Transformer 中数组的形状和布局，再次用红色显示收缩维度，用蓝色显示批处理维度。在给定操作中，输入形状在左上角给出，参数形状在右上角给出，结果形状在下方给出，例如 BTD 是门控 einsum 的输入形状，DF 是权重形状。" %}

**注意 [门控 einsum]**：上图使用了"[门控 einsums](https://arxiv.org/abs/2002.05202)"<d-cite key="glu"></d-cite>，我们将上投影矩阵分成两个矩阵（上面的 $W_\text{In1}$ 和 $W_\text{In2}$），它们的输出逐元素相乘作为一种"门控函数"。并非所有 LLM 都使用这个，因此您有时会看到单个 $W_\text{In}$ 矩阵，总的 MLP 参数数量为 2DF 而不是 3DF。通常在这种情况下，D 和 F 会按比例放大以保持参数数量与 3 矩阵情况相同。话虽如此，LLAMA、DeepSeek 和许多其他模型都使用某种形式的门控 einsum。

**注意 2 [MHA 注意力]**：在自注意力中，T 和 S 是相同的，但在交叉注意力中它们可能不同。在标准多头注意力（MHA）中，N 和 K 相同，而对于 [多头查询注意力](https://arxiv.org/abs/1911.02150)（MQA）<d-cite key="mqa"></d-cite> K=1，对于 [分组 MQA](https://arxiv.org/abs/2305.13245)（GMQA）<d-cite key="gmqa"></d-cite> K 只需要能整除 N 即可。

## 全局 FLOPs 和参数计算

为了下面计算，我们将计算每层的 FLOPs，以避免到处都要加上 **L** 的因子。

### MLP

Transformer 的 MLP 通常由 2 个逐元素组合的输入矩阵乘法和一个单一的输出矩阵乘法组成：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练 FLOPs} & \textrm{参数} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### 注意力

对于具有不同 **Q** 和 **KV** 头数的通用分组查询注意力情况，让我们假设 **Q**、**K**、**V** 投影具有相等的头维度 H，并估算 **QKVO** 矩阵乘法的成本：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练 FLOPs} & \textrm{参数} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

点积注意力操作更加微妙，有效地是一个在 $$B$$、$$K$$ 维度上分批的 $$TH \cdot HS$$ 矩阵乘法，一个 softmax，以及一个再次在 $$B$$、$$K$$ 维度上分批的 $$TS \cdot SH$$ 矩阵乘法。我们用蓝色突出显示分批维度：

$$
\begin{array}{cc}
\textrm{操作} & \textrm{训练 FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

### 其他操作

Transformer 中还有几个其他操作。层归一化相对便宜，对于一阶成本估算可以忽略。还有最后的巨大（尽管不是每层的）反嵌入矩阵乘法。

$$
\begin{array}{ccc}
\textsf{操作} & \textsf{训练 FLOPs} & \textsf{参数} \\
\hline \\
\textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer FLOPs 的一般经验法则

如果我们忽略短上下文训练中点积注意力的成本，那么所有层的总 FLOPs 是

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{令牌数量} * \textrm{参数数量}
\end{align*}
$$

这导致了估算密集 Transformer FLOPs 数量的著名经验法则，忽略了注意力 FLOPs。（反嵌入是另一个简单的矩阵乘法，具有 $6BSDV$ FLOPs 和 $DV$ 参数，并遵循相同的经验法则。）

### 注意力与上下文长度的成本比例

如果我们确实考虑上述点积注意力并假设 $$F=4D$$，$$D=NH$$（这是典型的）且 $$N=K$$：

$$\small{\frac{\textrm{注意力 FLOPs}}{\textrm{矩阵乘法 FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

所以结论是**点积注意力 FLOPs 只在 T>8D 时在训练期间才占主导地位**。对于 D ~ 8k，这将是 ~64K 令牌。这是有道理的，因为这意味着随着 MLP 大小的增加，注意力 FLOPs 变得不太关键。对于大型模型，注意力的二次成本实际上并不是长上下文训练的巨大障碍。然而，对于较小的模型，例如 Gemma-27B，D=4608，这意味着注意力在 32k 序列长度左右开始占主导地位。Flash Attention 也有助于缓解长上下文的成本，我们在 [附录 A](#appendix-a-how-does-flash-attention-work) 中简要讨论。

## 杂项数学

### 稀疏性和专家混合

如果不简要讨论专家混合（MoE）模型<d-cite key="moe"></d-cite>，那将是我们的疏忽。专家混合模型用一组可以动态路由的独立 MLP 替换了标准 Transformer 中的单一密集 MLP 块。作为一级近似，**MoE 只是一个每层有 E 个 MLP 块的正常密集模型**，而不是只有一个。每个令牌激活这些专家中的 $k$ 个，通常 $k=2$。与密集版本相比，这使参数数量增加了 $O(E)$，而每个令牌的激活参数总数乘以 $k$。

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>图：</b>一个具有 $n$ 个专家的 MoE 层示例。门控专家将每个令牌路由到其中的 $k$ 个，那些 $k$ 个 MLP 的输出被求和。我们的参数数量是每个专家大小的 $n$ 倍，但每个令牌只使用 $k$ 个。<a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">来源</a>。" %}

与密集模型相比，MoE 引入了新的通信，主要是两个 AllToAll（一个在 MoE 块之前，一个在之后），它们将令牌路由到正确的专家并将它们带回其主设备。<d-footnote>技术上，这只有在我们的专家与数据或序列分片在同一轴上时才会发生。</d-footnote> 然而，正如我们在上一节看到的，每个 AllToAll 的成本仅是单轴（对于双向环）上可比 AllGather 的 1/4。

### 梯度检查点

反向传播作为一种算法是用计算换取内存。与需要 $$O(n_\text{layers}^2)$$ FLOPs 的反向传播不同，**它需要 $$O(n_\text{layers})$$ 内存**，保存前向传播期间生成的所有中间激活。虽然这比二次方计算更好，但在内存方面极其昂贵：一个具有 $$B * T=4M$$（每批总共 4M 令牌）、L=64 和 D=8192 的模型，如果避免所有不必要的反向传播计算，将不得不在 bfloat16 中保存大约 $$2 * 20 * B * T * D * L = 84TB$$ 的激活。20 来自（大致）计算上面 Transformer 图中的每个中间节点，因为例如

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

所以为了避免重新计算，我们需要从前向传播中保存 $$g(x)$$ 和 $$\exp(g(x))$$。为了避免保存这么多内存，我们可以选择只保存一部分中间激活。以下是我们在用的一些策略。

* **块重计算**：只保存每层的输入。这是我们使用的最激进的方法，每层只保存 1 个检查点，意味着在上面的例子中我们只保存 4.2TB。这迫使我们基本上在反向传播中重复所有前向传播 FLOPs，意味着我们将 FLOPs 从 $$6ND$$ 增加到大约 $$8ND$$。
* **仅大矩阵乘法**：另一个简单的策略是只保存大矩阵乘法的输出。这让我们避免在反向传播中重新计算任何大矩阵乘法，但仍然让我们重新计算其他激活函数和注意力部分。这将每层 20 个减少到接近每层 7 个。

这绝不是全面的。在使用 JAX 时，这些通常由 `jax.remat`/`jax.checkpoint` 控制（您可以阅读更多[这里](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)）。

### 键值（KV）缓存

正如我们将在 [第 7 节](../inference) 中看到的，LLM 推理有两个关键部分，预填充和生成。

* **预填充** 处理长提示并将其注意力激活保存在键值缓存（KV 缓存）中供生成使用，特别是注意力块中的键值投影。
* **生成** 将几个这样的 KV 缓存批处理在一起，并从每个缓存中采样令牌。

每个 KV 缓存实际上是一个大小为 $[2, S, L, K, H]$ 的数组，其中 2 代表键和值。这是相当大的！int8 中键值缓存的总大小是 $2SLKH$。对于一个具有 8k 上下文长度、64 层和 $KH = NH = D = 8192$ 的中等大小模型，这是 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。您可以看到为什么我们想要使用 $K \ll N$ 的 GMQA。

## 您应该从本节中学到什么？

* Transformer 的整体参数和 FLOPs 相对容易计算，总结如下，假设 MHA（批量大小 B，词汇表大小 V，序列长度 T，D=d<sub>model</sub>，F=d<sub>ff</sub>）：


<!-- $$
\begin{array}{ccc}
\textrm{组件} & \textrm{每层参数} & \textrm{每层训练 FLOPs} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{注意力} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{其他} & D & BTD \\[10pt]
\textbf{词汇表} & DB \text{ (总共，不是每层)} & 12BTDV \\[10pt]
\end{array}
$$ -->


| 组件         | 每层参数                 | 每层训练 FLOPs                 |
| :----------- | :----------------------- | :----------------------------- |
| **MLP**      | 3DF                      | 18BTDF                         |
| **注意力**   | 4DNH                     | 24BTDNH \+ 12BT<sup>2</sup>NH  |
| **其他**     | D                        | BTD                            |
| **词汇表**   | DV（总共，不是每层）     | 12BTDV                         |

* MLP 块的参数数量主导总参数数量，只要序列长度 $T < 8D$，MLP 块也主导 FLOPs 预算。
* 对于合理的上下文长度，训练期间的总 FLOPs 预算很好地近似为 $$6 \cdot \text{参数数量} \cdot \text{令牌数量}$$。
* 在推理期间，我们的 KV 缓存每个缓存大约是 $$2 \cdot S \cdot L \cdot N \cdot H$$，尽管架构修改通常可以减少这个数量。

## 几个要解决的问题

**问题 1：** 一个具有 $D=4096$、$F=4 \cdot D$、$V=32,000$ 和 $L=64$ 的模型有多少参数？其中多少是注意力参数？我们每个令牌的 KV 缓存有多大？*您可以假设 $N\cdot H=D$ 和 int8 KV 的多头注意力。*

{% details 点击这里查看答案。%}

1. 总参数大约是 $$L \cdot (3DF + 4DNH + D) + 2DV$$。对于给定的数字，这是 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$，即 160 亿参数。
2. 注意力参数与总参数的比例通常是 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$。这给了我们大约 1/4 的参数用于注意力。
3. 每个令牌，我们的 KV 缓存在 int8 中是 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$，即 `512kB / 令牌`。

{% enddetails %}

**问题 2：** 在 `{‘X': 4, ‘Y': 8, ‘Z': 4}` 上执行 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] 需要多少总 FLOPs？每个 TPU 执行多少 FLOPs？

{% details 点击这里查看答案。%}

该操作的总"理论"FLOPs 是 $$2 \cdot B \cdot D \cdot F$$。然而，因为计算没有在 Z 维度上分片，我们实际上在做 Z 倍的额外 FLOPs，意味着总 FLOPs 是 $$2 \cdot B \cdot D \cdot F \cdot Z$$。由于计算在其他维度上分片，每设备的总 FLOPs 大约是 $$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$。

{% enddetails %}

**问题 3：** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOPs？

{% details 点击这里查看答案。%}

按照上述规则，我们有 I 和 J 作为收缩维度，K、L、M、N 和 O 作为非收缩维度。我们没有"批处理维度"，所以这只是 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$，所有轴的总和。如果我们有共享轴，它只会被计算一次。

{% enddetails %}

**问题 4：** 自注意力的算术强度是多少（忽略 Q/K/V/O 投影）？*以 Q 和 KV 长度 T 和 S 的函数给出答案。*在什么上下文长度下注意力达到 FLOPs 限制？给定我们 TPU 的 HBM 带宽，随着上下文长度增长，绘制注意力相对于 FFW 块的有效相对成本图。

{% details 点击这里查看答案。%}

自注意力需要加载 $$Q$$、$$K$$ 和 $$V$$ 激活，然后计算 $$\text{softmax}(Q \cdot K) \cdot V$$，然后将结果写回 HBM。这将使用 Flash Attention 完成，所以这个数学有一些注意事项，但基本上在 bf16 自注意力中执行

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

所以我们的总字节数是 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$，总 FLOPs 是 $$4BTSNH + O(BTSN)$$，算术强度是 $$4BTSKGH / (4BHK * (TG + S))$$。

所以基本上，在预填充期间我们有 $$S=T$$，所以我们的算术强度是 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$。在生成期间，$$T=1$$，所以我们有 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$，假设 $$S$$ 非常大。根据您如何解释这个问题，在预填充或训练期间，假设没有序列分片，自注意力在 S=240 时达到计算限制。在生成期间，我们永远不会达到计算限制，因为 $$G$$ 很小。尽管如此，您可以看到增加 $$G$$ 使我们更接近计算限制。

{% enddetails %}

**问题 5：** 在什么序列长度下，自注意力 FLOPs 等于 QKVO 投影 FLOPs？

{% details 点击这里查看答案。%}

这纯粹是关于何时 $$24BTDNH == 12BT^2NH$$ 的问题。简化我们得到 $$2D = T$$，所以例如对于 $$D=4096$$，这是 $$8192$$。这告诉我们，对于大多数合理的上下文长度，矩阵乘法 FLOPs 更大。

{% enddetails %}

**问题 6：** 假设我们在前向传播期间只保存 Transformer 层中 7 个主要矩阵乘法的输出（Q、K、V、O + 三个 FFW 矩阵）。在反向传播期间我们需要多少额外的 FLOPs 来"重新物化"？

{% details 点击这里查看答案。%}

只保存七个矩阵乘法输出（Q、K、V、O、W₁、W₂、W₃）意味着反向传播必须重新计算两个注意力矩阵乘法

$$QK^{\top} \quad\text{和}\quad \operatorname{softmax}(QK^{\top})V.$$

每个都是 $T \times T$ 矩阵乘法，在 $B$ 个序列和 $N$ 个头上分批，所以额外的 FLOPs 是

$$4 \; B \, T^{2} \, N \, H.$$

所有其他重新计算的操作只有 $O(BTD)$。

{% enddetails %}

**问题 7：** DeepSeek v3 说它在 14.8T 令牌上训练了 2.79M H800 小时（[来源](https://arxiv.org/pdf/2412.19437v1)）。鉴于它有 370 亿激活参数，他们实现了大约多少硬件利用率？*提示：注意他们使用的是没有结构化稀疏性的 FP8 FLOPs。*

{% details 点击这里查看答案。%}

从 [这里](https://lenovopress.lenovo.com/lp1814.pdf) 的规格表中，我们发现具有稀疏性的 FP8 性能为 3,026 TFLOPs/s，或者通常是没有稀疏性的一半（`1.513e15` FLOPs/s）。2.79M H800 小时意味着 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` 总 FLOPs。鉴于激活参数数量为 370 亿，这次训练运行应该使用了大约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs。这意味着 FLOPs 利用率大约是 `3.3e24 / 1.52e25 = 21.7%`。

{% enddetails %}

**问题 8：** 专家混合（MoE）模型有 $E$ 个标准密集 MLP 块的副本，每个令牌激活这些专家中的 $k$ 个。在 TPU v5e 上，对于 int8 权重的 MoE，需要多大的令牌批量大小才能达到计算限制？对于 DeepSeek，它有 256 个（路由的）专家和 $k=8$，这个数字是多少？

{% details 点击这里查看答案。%}

因为每个专家有 $E$ 个副本，在 int8 中，我们需要加载 $E \cdot D \cdot F$ 字节。因为每个令牌激活 $k$ 个专家，我们有 $2\cdot k \cdot B \cdot D \cdot F$ FLOPs。要用 bfloat16 FLOPs 达到计算限制，我们需要超过 240 的算术强度，这在 $(2\cdot k \cdot BDF) / EDF > 240$ 或 $k \cdot B / E > 120$ 时发生。

因此，我们需要 $B > 120 \cdot E / k$ 才能达到计算限制。对于 DeepSeek，这给了我们 $B > 120 \cdot 256 / 8 = 3840$。这在生成时间是一个非常大的批量大小。

{% enddetails %}

<h3 markdown=1 class="next-section">第 4 部分就到这里！关于扩展 Transformer 训练的第 5 部分，[点击这里](../training)！</h3>

## 附录

### 附录 A：Flash Attention 是如何工作的？

传统的反对将 Transformer 扩展到非常长上下文的论点是注意力 FLOPs 和内存使用与上下文长度成二次方缩放。虽然注意力 QK 乘积确实具有形状 $[B, S, T, N]$，其中 B 是批量大小，S 和 T 是 Q 和 K 序列维度，N 是头数，但这个说法附带一些严重的警告：

1. 正如我们在第 4 节中指出的，即使这是二次方的，注意力 FLOPs 只在 $$S > 8 \cdot D$$ 时才占主导地位，特别是在训练期间，单个注意力矩阵的内存与内存中的所有权重和激活检查点相比很小，特别是在分片时。
2. 我们不需要具体化完整的注意力矩阵来计算注意力！我们可以计算局部和和最大值，并避免具体化超过一小块数组。虽然总 FLOPs 仍然是二次方的，但我们大大减少了内存压力。

第二个观察首先由 [Rabe 等人 2021](https://arxiv.org/abs/2112.05682) 提出，后来在 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)（Dao 等人 2022）中提出。基本思想是以 K/V 的块来计算注意力，我们计算局部 softmax 和一些辅助统计信息，然后将它们传递到下一个块，该块将它们与其局部块结合起来。具体来说，我们计算

1. **M：** $$q \cdot k$$ 在序列维度上的运行最大值
2. **O：** 在序列维度上的运行完整注意力 softmax
3. **L：** 运行分母 $$\sum_i (q \cdot k_i - \text{运行最大值})$$

有了这些，我们可以只用恒定量的内存计算新的最大值、新的运行总和和新的输出。为了给出这个工作原理的粗略描述，注意力大致是这个操作：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

最大值被减去以获得数值稳定性，并且可以在不影响结果的情况下添加，因为 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$。只看上面的分母，如果我们想象有两个连续的键向量块，$$K^1$$ 和 $$K^2$$，我们为每个计算局部 softmax 总和 $$L^1$$ 和 $$L^2$$

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

然后我们可以使用以下方法将它们合并为这两个块在一起的完整 softmax 总和

$$L^\text{合并} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ 和 } M^2 = \max_j Q \cdot K_j^2$$

这也可以对完整的 softmax 进行，给了我们一种累积任意大的 softmax 总和的方法。这是 Flash Attention 论文中的完整算法。

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件角度来看，这让我们可以将我们的 Q 块放入 VMEM（上面算法称为片上 SRAM 的东西），所以我们只需要在每次迭代时加载 KV 块，从而降低算术强度。我们还可以将运行统计信息保存在 VMEM 中。

最后一个值得强调的微妙点是用于使 Flash VJP（反向模式导数）计算对训练实用的注意力 softmax 属性。如果我们将中间 softmax 数组定义为：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_j}}$$

在注意力中，我们从反向模式 *dO* 和 *V* 数组获得 *dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在将此梯度反向传播到 Q 和 K 期间

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式，允许我们将沿大键**长度**维度的收缩与沿特征**深度**维度的局部收缩交换。

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

这个替换对于能够为 VJP 实现序列块*局部*计算至关重要，并启用了更聪明的分片方案，如环注意力。
