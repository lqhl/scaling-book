---
layout: distill
title: "你需要知道的所有 Transformer 数学"
# permalink: /main/
description: "在这里, 我们将快速回顾 Transformer 架构, 特别是如何计算 FLOPs, 字节和其他感兴趣的数量."
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
  - name: "数点"
  - subsections:
    - name: "前向和反向 FLOPs"
  - name: "Transformer 核算"
  - name: "全局 FLOPs 和参数计算"
  - name: "杂项数学"
  - subsections:
    - name: "稀疏性和专家混合"
    - name: "梯度检查点"
    - name: "键值 (KV) 缓存"
  - name: "你应该从本节中学到什么?"
  - name: "一些待解决的问题"
  - name: "附录"
  - subsections:
    - name: "附录 A: Flash Attention 是如何工作的?"

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

## 数点

让我们从以下形状的向量 $$x$$,$$y$$ 和矩阵 $$A$$,$$B$$ 开始:

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
\end{array}
$$

-   $$x \cdot y$$ 的点积需要 $$P$$ 次*加法*和*乘法*, 或总共 $$2P$$ 次浮点运算.
-   矩阵向量乘积 $$Ax$$ 沿着 $$A$$ 的行进行 $$N$$ 次点积, 共 $$2NP$$ FLOPs.
-   矩阵-矩阵乘积 $$AB$$ 对 $$B$$ 的 $$M$$ 列中的每一列进行一次矩阵-向量乘积, 总共 $$2NPM$$ FLOPs.
-   一般来说, 如果我们有两个更高维度的数组 $$C$$ 和 $$D$$, 其中一些维度是<span style="color:red">收缩的</span>, 一些是<span style="color:blue">批处理的</span>. (例如 $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$), 那么这个收缩的 FLOPs 成本是所有 $$C$$ 和 $$D$$ 维度乘积的两倍, 其中批处理和收缩维度只计算一次, (例如 $$2\blue{GH}IJMN\red{KL}$$). 请注意, 只有当一个维度同时出现在两个乘数中时, 它才是批处理维度. (另请注意, 如果没有收缩维度并且这只是一个逐元素乘积, 则 2 的因子将不适用.)

$$ 
\begin{array}{ccc}
\textrm{操作} & \textrm{FLOPs} & \textrm{数据} \\ \hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end{array}
$$ 

请注意, 对于矩阵-矩阵乘法, *计算*以三次方的速度 $$O(N^3)$$ 扩展, 而数据传输仅以二次方的速度 $$O(N^2)$$ 扩展 - 这意味着随着我们扩大矩阵乘法的大小, *更容易*达到计算饱和的极限. 这是非常不寻常的, 并且在很大程度上解释了为什么我们使用以矩阵乘法为主的架构 - 它们易于扩展!

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向和反向 FLOPs

在训练期间, 我们并不特别关心给定矩阵乘法的结果; 我们真正关心的是它的导数. 这意味着我们在反向传播期间会执行更多的 FLOPs.

如果我们想象 **B** 只是一个更大网络中的一个矩阵, **A** 是我们的输入激活, **C = A B**, 那么损失 **L** 对 **B** 的导数由链式法则给出:

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

这是一个外积, 需要 $2NPM$ FLOPs 来计算 (因为它在 $N$ 维度上收缩). 同样, 损失对 **A** 的导数是

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

同样是 $2NPM$ FLOPs, 因为 **dL/dC** 是一个大小为 $$[N, M]$$ 的 (余) 向量. 虽然这个量不是关于参数的导数, 但它用于计算网络前几层的导数 (例如, 正如 dL/dC 用于计算上面的 dL/dB).

将这些加起来, 我们看到**在训练期间, 我们总共有 6NPM FLOPs**, 而推理期间为 2NPM: 前向传播 2NPM, 后向传播 4NPM. 由于 PM 是矩阵中的参数数量, 这是著名的 Transformer 训练期间 FLOPs 的 $$6 * \text{参数数量} * \text{token 数量}$$ 近似的简化形式: 每个 token 需要 $$6 * \text{参数数量}$$ FLOPs. 我们将在下面展示一个更正确的推导.

## Transformer 核算

Transformer 是未来. 嗯, 至少它们是现在. 也许几年前, 它们是众多架构之一. 但今天, 几乎值得了解该架构的每一个细节. 我们不会重新介绍该架构, 但[这篇博客](https://jalammar.github.io/illustrated-transformer/)和[原始的 Transformer 论文](https://arxiv.org/abs/1706.03762)可能会有所帮助.

这是一个 Transformer 解码器架构的基本图:

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图:</b> 该图显示了一个标准 Transformer 的一层, 从上到下流动. 我们使用单字母约定来描述 Transformer 中数组的形状和布局, 再次以红色显示收缩维度, 以蓝色显示批处理维度. 在给定的操作中, 输入形状在左上角给出, 参数形状在右上角给出, 结果形状在下面, 例如 BTD 是门控 einsum 的输入形状, DF 是权重形状." %}

**注意 [门控 einsum]**: 上图使用了一个“[门控 einsums](https://arxiv.org/abs/2002.05202)”<d-cite key="glu"></d-cite>, 其中我们将上投影矩阵分成两个矩阵 ($W_\text{In1}$ 和 $W_\text{In2}$ 上面), 其输出作为一种“门控函数”进行逐元素相乘. 并非所有 LLM 都使用它, 因此你有时会看到一个单一的 $W_\text{In}$ 矩阵, MLP 参数总数为 2DF 而不是 3DF. 通常在这种情况下, D 和 F 会被放大以保持参数数量与 3 矩阵情况相同. 话虽如此, LLAMA, DeepSeek 和许多其他模型都使用了某种形式的门控 einsum.

**注意 2 [MHA 注意力]**: 对于自注意力, T 和 S 是相同的, 但对于交叉注意力, 它们可能不同. 对于普通的多头注意力 (MHA), N 和 K 是相同的, 而对于[多查询注意力](https://arxiv.org/abs/1911.02150) (MQA)<d-cite key="mqa"></d-cite> K=1, 对于[分组 MQA](https://arxiv.org/abs/2305.13245) (GMQA)<d-cite key="gmqa"></d-cite> K 只需要整除 N.

## 全局 FLOPs 和参数计算

对于下面, 我们将计算每层的 FLOPs, 以避免到处都出现 **L** 的因子.

### MLPs


Transformer 的 MLP 通常由 2 个输入矩阵乘法组成, 它们逐元素组合, 以及一个输出矩阵乘法:

$$ 
\begin{array}{ccc}
\textrm{操作} & \textrm{训练 FLOPs} & \textrm{参数} \\ \hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\
\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\
\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\
\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$ 

### 注意力

对于具有不同 **Q** 和 **KV** 头数的通用分组查询注意力情况, 让我们假设 **Q**,**K**,**V** 投影的头维度 H 相等, 并估计 **QKVO** 矩阵乘法的成本:

$$ 
\begin{array}{ccc}
\textrm{操作} & \textrm{训练 FLOPs} & \textrm{参数} \\ \hline \\

A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\
\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\
\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\
\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\
\hline \ \
& 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$ 

点积注意力操作更微妙, 实际上是在 $$B$$, $$K$$ 维度上批处理的 $$TH \cdot HS$$ 矩阵乘法, 一个 softmax, 以及再次在 $$B$$, $$K$$ 维度上批处理的 $$TS \cdot SH$$ 矩阵乘法. 我们用蓝色突出显示批处理维度:

$$ 
\begin{array}{cc}
\textrm{操作} & \textrm{训练 FLOPs} \\ \hline \[\[3pt]\]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\
\\\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\
\\\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\
\hline \ \
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$ 

### 其他操作

Transformer 中还发生了其他一些操作. Layernorm 相对便宜, 对于一阶成本估算可以忽略. 还有一个最终的巨大的 (虽然不是每层的) unembedding 矩阵乘法.

$$ 
\begin{array}{ccc}
\textsf{操作} & \textsf{训练 FLOPs} & \textsf{参数} \\ \hline \\

\\\[10pt]
\textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\
\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$ 

### Transformer FLOPs 的通用经验法则

如果我们忽略短上下文训练的点积注意力成本, 那么所有层的总 FLOPs 是

$$ 
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{token 数量} * \textrm{参数数量}
\end{align*}
$$ 

这导致了一个著名的经验法则, 用于估算密集 Transformer 的 FLOP 数量, 忽略了注意力 FLOPs. (Unembedding 是另一个简单的矩阵乘法, 有 $6BSDV$ FLOPs 和 $DV$ 参数, 并遵循相同的经验法则.)

### 注意力与上下文长度的成本分数

如果我们确实考虑了上面的点积注意力, 并假设 $$F=4D$$, $$D=NH$$ (通常如此) 和 $$N=K$$:

$$\small{\frac{\textrm{注意力 FLOPs}}{\textrm{矩阵乘法 FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

所以结论是**点积注意力 FLOPs 只有在 T>8D 时才在训练中占主导地位**. 对于 D ~ 8k, 这将是 ~64K token. 这在某种程度上是有道理的, 因为这意味着随着 MLP 大小的增加, 注意力 FLOPs 变得不那么重要. 对于大型模型, 注意力的二次成本实际上并不是长上下文训练的巨大障碍. 然而, 对于较小的模型, 即使是例如 Gemma-27B, D=4608, 这意味着注意力在 32k 序列长度左右变得占主导地位. Flash Attention 也有助于减轻长上下文的成本, 我们在[附录 A](#appendix-a-how-does-flash-attention-work)中简要讨论了这一点.

## 杂项数学

### 稀疏性和专家混合

我们不能不简要讨论专家混合 (MoE) 模型<d-cite key="moe"></d-cite>, 它用一组可以动态路由的独立 MLP 块取代了标准 Transformer 中的单个密集 MLP 块. 初步来看, **一个 MoE 只是一个普通的密集模型, 每层有 E 个 MLP 块**, 而不是只有一个. 每个 token 激活这些专家中的 $k$ 个, 通常 $k=2$. 与密集版本相比, 这将参数数量增加了 $O(E)$, 同时将每个 token 的激活参数总数乘以 $k$.

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>图:</b> 一个具有 $n$ 个专家的 MoE 层示例. 门控专家将每个 token 路由到其中的 $k$ 个, 这 $k$ 个 MLP 的输出被求和. 我们的参数数量是每个专家大小的 $n$ 倍, 但每个 token 只使用 $k$ 个. <a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">来源</a>." %}

与密集模型相比, MoE 引入了新的通信, 主要是两个 AllToAll (一个在 MoE 块之前, 一个在之后), 将 token 路由到正确的专家, 并将它们带回其主设备.<d-footnote>技术上, 这只在我们在与专家相同的轴上进行数据或序列分片时发生.</d-footnote> 然而, 正如我们在上一节中看到的, 每个 AllToAll 的成本仅为沿单个轴的可比 AllGather 的 1/4 (对于双向环).

### 梯度检查点

反向传播作为一种算法, 用内存换取计算. 反向传播不需要 $$O(n_\text{layers}^2)$$ FLOPs, **它需要 $$O(n_\text{layers})$$ 内存**, 保存前向传播期间生成的所有中间激活. 虽然这比二次计算要好, 但在内存方面却非常昂贵: 一个具有 $$B * T=4M$$ (每批总共 4M token), L=64 和 D=8192 的模型, 如果避免所有不必要的后向传播计算, 将不得不保存大约 $$2 * 20 * B * T * D * L = 84TB$$ 的 bfloat16 激活. 20 来自 (粗略地) 计算上面 Transformer 图中的每个中间节点, 因为例如

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

所以为了避免重新计算, 我们需要从前向传播中保存 $$g(x)$$ 和 $$\exp(g(x))$$. 为了避免保存这么多内存, 我们可以选择只保存一部分中间激活. 以下是我们使用的一些策略.

*   **块重算**: 只保存每层的输入. 这是我们使用的最激进的方法, 每层只保存 1 个检查点, 这意味着在上面的例子中我们只保存 4.2TB. 这迫使我们在后向传播中重复基本上所有的前向传播 FLOPs, 这意味着我们将 FLOPs 从 $$6ND$$ 增加到大约 $$8ND$$.
*   **仅大矩阵乘法:** 另一个简单的策略是只保存大矩阵乘法的输出. 这使我们能够避免在后向传播期间重新计算任何大矩阵乘法, 但仍然使我们重新计算其他激活函数和部分注意力. 这将每层的 20 个减少到接近 7 个.

这绝不是详尽无遗的. 在使用 JAX 时, 这些通常由 `jax.remat`/`jax.checkpoint` 控制 (你可以在[这里](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)阅读更多内容).

### 键值 (KV) 缓存

正如我们将在[第 7 节](../inference)中看到的, LLM 推理有两个关键部分, 预填充和生成.

*   **预填充**处理一个长提示, 并将其注意力激活保存在一个键值缓存 (KV Cache) 中, 以便在生成中使用, 特别是注意力块中的键值投影.
*   **生成**将其中几个 KV 缓存批处理在一起, 并从每个缓存中采样 token.

每个 KV 缓存实际上是一个大小为 $[2, S, L, K, H]$ 的数组, 其中 2 表示键和值. 这相当大! int8 中键值缓存的总大小为 $2SLKH$. 对于一个中等大小的模型, 具有 8k 上下文长度, 64 层, 以及 $KH = NH = D = 8192$, 这是 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$. 你可以看到为什么我们想要使用 $K \ll N$ 的 GMQA.

## 你应该从本节中学到什么?

*   Transformer 的总体参数和 FLOPs 相当容易计算, 并在此处进行了总结, 假设 MHA (批量大小为 B, 词汇量大小为 V, 序列长度为 T, D=d_model, F=d_ff):


<!-- $$
\begin{array}{ccc}
\textrm{组件} & \textrm{每层参数} & \textrm{每层训练 FLOPs} \\ \hline \\
\textbf{MLP} & 3DF & 18BTDF \\
\[10pt]
\textbf{注意力} & 4DNH & 24BTDNH + 12BT^2NH \\
\[10pt]
\textbf{其他} & D & BTD \\
\[10pt]
\textbf{词汇表} & DB \text{ (总计, 非每层)} & 12BTDV \\
\end{array}
$$ -->


| 组件 | 每层参数 | 每层训练 FLOPs |
| :------------ | :------------------------ | :---------------------------- |
| **MLP**       | 3DF                       | 18BTDF                        |
| **注意力** | 4DNH                      | 24BTDNH + 12BT<sup>2</sup>NH |
| **其他**     | D                         | BTD                           |
| **词汇表**     | DV (总计, 非每层) | 12BTDV                        |

*   MLP 块的参数数量在总参数数量中占主导地位, 并且只要序列长度 $T < 8D$, MLP 块在 FLOPs 预算中也占主导地位.
*   对于合理的上下文长度, 训练期间的总 FLOPs 预算可以很好地近似为 $$6 \cdot \text{num_params} \cdot \text{num_tokens}$$.
*   在推理期间, 我们的 KV 缓存大约为每个缓存 $$2 \cdot S \cdot L \cdot N \cdot H$$, 尽管架构修改通常可以减少这个值.

## 一些待解决的问题

**问题 1:** 一个具有 $D=4096$, $F=4 \cdot D$, $V=32,000$ 和 $L=64$ 的模型有多少参数? 其中有多少是注意力参数? 每个 token 的 KV 缓存有多大? *你可以假设 $N\cdot H=D$ 和多头注意力, int8 KVs.*

{% details 点击这里查看答案. %}

1.  总参数大约是 $$L \cdot (3DF + 4DNH + D) + 2DV$$. 对于给定的数字, 这是 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$, 或 16B 参数.
2.  注意力参数与总参数之比通常是 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$. 这给了我们大约 1/4 的参数用于注意力.
3.  每个 token, 我们的 KV 缓存是 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$ (int8), 即 `512kB / token`.

{% enddetails %}

**问题 2:** 在 `{‘X': 4, ‘Y': 8, ‘Z': 4}` 上执行 A[B_X, D_Y] \*\_D W[D_Y, F] 需要多少总 FLOPs? 每个 TPU 执行多少 FLOPs?

{% details 点击这里查看答案. %}

该操作的总“理论” FLOPs 是 $$2 \cdot B \cdot D \cdot F$$. 然而, 因为计算没有在 Z 维度上分片, 我们实际上多做了 Z 倍的 FLOPs, 这意味着总共有 $$2 \cdot B \cdot D \cdot F \cdot Z$$ FLOPs. 由于计算在其他维度上是分片的, 每个设备的总 FLOPs 大约是 $$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$.

{% enddetails %}

**问题 3:** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOPs?

{% details 点击这里查看答案. %}

根据上面的规则, 我们有 I 和 J 作为收缩维度, K, L, M, N 和 O 作为非收缩维度. 我们没有“批处理维度”, 所以这只是 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$, 所有轴的总和. 如果我们有一个共享轴, 它只会被计算一次.

{% enddetails %}

**问题 4:** 自注意力的算术强度是多少 (忽略 Q/K/V/O 投影)? *以 Q 和 KV 长度 T 和 S 的函数形式给出答案.* 在什么上下文长度下, 注意力是受 FLOPs 限制的? 给定我们 TPU 的 HBM 带宽, 绘制随着上下文长度的增长, 注意力与 FFW 块的有效相对成本.

{% details 点击这里查看答案. %}

自注意力需要加载 $$Q$$, $$K$$ 和 $$V$$ 激活, 然后计算 $$\text{softmax}(Q \cdot K) \cdot V$$, 然后将结果写回 HBM. 这将使用 Flash Attention 完成, 所以这个数学有一些警告, 但基本上在 bf16 自注意力中执行

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$ 

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

所以我们的总字节数是 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$, 总 FLOPs 是 $$4BTSNH + O(BTSN)$$, 算术强度是 $$4BTSKGH / (4BHK * (TG + S))$$.

所以基本上, 在预填充期间, 我们有 $$S=T$$, 所以我们的算术强度是 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$. 在生成期间, $$T=1$$, 所以我们有 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$, 假设 $$S$$ 非常大. 根据你如何解释这个问题, 在预填充或训练期间, 假设没有序列分片, 自注意力在 S=240 时受计算限制. 在生成期间, 我们永远不会受计算限制, 因为 $$G$$ 很小. 然而, 无论如何, 你可以看到增加 $$G$$ 会使我们更接近受计算限制.

{% enddetails %}

**问题 5:** 在什么序列长度下, 自注意力 FLOPs 等于 QKVO 投影 FLOPs?

{% details 点击这里查看答案. %}

这纯粹是一个关于何时 $$24BTDNH == 12BT^2NH$$ 的问题. 简化后我们得到 $$2D = T$$, 例如对于 $$D=4096$$, 这是 $$8192$$. 这告诉我们, 对于大多数合理的上下文长度, 矩阵乘法 FLOPs 更大.

{% enddetails %}

**问题 6:** 假设我们在前向传播期间只保存 Transformer 层中 7 个主要矩阵乘法 (Q, K, V, O + 三个 FFW 矩阵) 中每个的输出. 在后向传播期间, 我们需要“重新物化”多少额外的 FLOPs?

{% details 点击这里查看答案. %}

只保存七个矩阵乘法输出 (Q, K, V, O, W₁, W₂, W₃) 意味着后向传播必须重新计算两个注意力矩阵乘法

$$QK^{\top} \quad\text{和}\quad \operatorname{softmax}(QK^{\top})V.$$

每个都是一个在 $B$ 个序列和 $N$ 个头上批处理的 $T \times T$ 矩阵乘法, 所以额外的 FLOPs 是

$$4 \; B \; T^{2} \; N \; H.$$

所有其他重新计算的操作都只是 $O(BTD)$.

{% enddetails %}

**问题 7:** DeepSeek v3 表示它在 14.8T token 上训练了 279 万 H800 小时 ([来源](https://arxiv.org/pdf/2412.19437v1)). 鉴于它有 37B 个激活参数, 他们大致实现了什么样的硬件利用率? *提示: 请注意, 他们使用了没有结构化稀疏性的 FP8 FLOPs.*

{% details 点击这里查看答案. %}

从[这里](https://lenovopress.lenovo.com/lp1814.pdf)的规格表中, 我们发现有 3,026 TFLOPs/s 的 FP8 性能 (带稀疏性), 或者通常是这个值的一半 (`1.513e15` FLOPs/s) (不带稀疏性). 279 万 H800 小时意味着 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` 总 FLOPs. 鉴于 37B 的激活参数数量, 这次训练运行应该使用了大约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs. 这意味着 FLOPs 利用率大约是 `3.3e24 / 1.52e25 = 21.7%`.

{% enddetails %}

**问题 8:** 专家混合 (MoE) 模型有 $E$ 个标准密集 MLP 块的副本, 每个 token 激活这些专家中的 $k$ 个. 在 TPU v5e 上, 对于 int8 权重的 MoE, 需要多大的 token 批量大小才能受计算限制? 对于 DeepSeek, 它有 256 个 (路由的) 专家和 $k=8$, 这个数字是多少?

{% details 点击这里查看答案. %}

因为我们有 $E$ 个每个专家的副本, 在 int8 中, 我们需要加载 $E \cdot D \cdot F$ 字节. 因为每个 token 激活 $k$ 个专家, 我们有 $2\cdot k \cdot B \cdot D \cdot F$ FLOPs. 为了在 bfloat16 FLOPs 下受计算限制, 我们需要一个超过 240 的算术强度, 这发生在 $(2\cdot k \cdot BDF) / EDF > 240$ 或 $k \cdot B / E > 120$ 时.

因此, 我们需要 $B > 120 \cdot E / k$ 才能受计算限制. 对于 DeepSeek, 这给了我们 $B > 120 \cdot 256 / 8 = 3840$. 这在生成时是一个非常大的批量大小.

{% enddetails %}

<h3 markdown=1 class="next-section">第四部分到此结束! 第五部分 (关于扩展 Transformer 训练), [点击这里](../training)!</h3>

## 附录

### 附录 A: Flash Attention 是如何工作的?

传统上反对将 Transformer 扩展到非常长的上下文的理由是, 注意力 FLOPs 和内存使用量随上下文长度呈二次方增长. 虽然注意力 QK 乘积的形状为 $[B, S, T, N]$ (其中 B 是批量大小, S 和 T 是 Q 和 K 序列维度, N 是头数) 是正确的, 但这一说法带有一些严重的警告:

1.  正如我们在第 4 节中指出的, 即使这是二次的, 注意力 FLOPs 也只在 $$S > 8 \cdot D$$ 时占主导地位, 特别是在训练期间, 单个注意力矩阵的内存与内存中存在的所有权重和激活检查点相比很小, 特别是在分片时.
2.  我们不需要物化完整的注意力矩阵来计算注意力! 我们可以计算局部和和最大值, 并避免物化超过数组的一小部分. 虽然总 FLOPs 仍然是二次的, 但我们大大减少了内存压力.

这一第二个观察首先由 [Rabe et al. 2021](https://arxiv.org/abs/2112.05682) 提出, 后来在 [Flash Attention 论文](https://arxiv.org/abs/2205.14135) (Dao et al. 2022) 中提出. 基本思想是将注意力分块计算 K/V, 我们计算局部 softmax 和一些辅助统计数据, 然后将它们传递给下一个块, 下一个块将它们与自己的局部块组合. 具体来说, 我们计算

1.  **M:** 序列维度上 $$q \cdot k$$ 的运行最大值
2.  **O:** 序列维度上运行的完整注意力 softmax
3.  **L:** 运行分母 $$\sum_i (q \cdot k_i - \text{运行最大值})$$

有了这些, 我们可以用恒定的内存量计算新的最大值, 新的运行和, 以及新的输出. 为了粗略地描述这是如何工作的, 注意力大致是这个操作:

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

为了数值稳定性, 减去了最大值, 并且可以在不影响结果的情况下添加, 因为 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$. 只看上面的分母, 如果我们想象有两个连续的键向量块, $$K^1$$ 和 $$K^2$$, 我们为每个块计算局部 softmax 和 $$L^1$$ 和 $$L^2$$

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

然后我们可以使用以下公式将它们组合成这两个块的完整 softmax 和

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2$$

这也可以对完整的 softmax 进行, 给了我们一种累积任意大的 softmax 和的方法. 这是 Flash Attention 论文中的完整算法.

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件的角度来看, 这让我们可以将我们的 Q 块放入 VMEM (上面算法称之为片上 SRAM), 所以我们只需要在每次迭代中加载 KV 块, 从而降低了算术强度. 我们也可以将运行统计数据保存在 VMEM 中.

最后一个值得强调的微妙之处是注意力 softmax 的一个属性, 它被用来使 Flash VJP (反向模式导数) 计算在训练中变得实用. 如果我们将一个中间 softmax 数组定义为:

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_j}}$$

在注意力中, 我们从反向模式 *dO* 和 *V* 数组中获得 *dS*:

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在将此梯度反向传播到 Q 和 K 期间

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式, 允许我们将沿大键**长度**维度的收缩与沿特征**深度**维度的局部收缩交换.

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\ &= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\ &= \sum_d dO_{id} O_{id} \\ &= dO_{id} \cdot_d O_{id}
\end{align*}$$

这种替换对于能够为 VJP 实现序列块*局部*计算至关重要, 并启用了更巧妙的分片方案, 如环形注意力.