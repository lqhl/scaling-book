---
layout: distill
title: "如何并行化Transformer进行训练"
# permalink: /main/
description: "在这里我们讨论LLM训练期间使用的四种主要并行方案：数据并行、完全分片数据并行（FSDP）、张量并行和流水线并行。对于每一种，我们计算在什么时候我们会被通信瓶颈所限制。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers"
previous_section_name: "第4部分：Transformers"

next_section_url: ../applied-training
next_section_name: "第6部分：训练LLaMA"

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
  - name: "我们所说的扩展是什么意思？"
  - subsections:
    - name: "数据并行"
    - name: "完全分片数据并行（FSDP）"
    - name: "张量并行"
    - name: "结合FSDP和张量并行"
    - name: "流水线并行"
    - name: "跨Pod扩展"
  - name: "在TPU上训练LLM的经验总结"
  - name: "一些练习题"
  - name: "附录"
  - subsections:
    - name: "附录A：推导反向传播的通信"

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

## 我们所说的扩展是什么意思？

"模型扩展"的目标是能够增加用于训练或推理的芯片数量，同时实现吞吐量的成比例线性增长（我们称之为*强扩展*）。虽然单个芯片的性能取决于内存带宽和FLOPs之间的权衡，但集群级别的性能取决于通过将芯片间通信与有用的FLOPs重叠来隐藏通信。这并不简单，因为增加芯片数量会增加通信负载，同时减少我们可以用来隐藏它的每设备计算量。正如我们在[第3节](../sharding)中看到的，分片矩阵乘法通常需要昂贵的AllGathers或ReduceScatters，这可能会阻止TPUs进行有用的工作。本节的目标是找出这些通信何时变得*过于昂贵*。

在本节中，我们将讨论四种常见的并行方案：（纯）**数据并行、完全分片数据并行**（FSDP / ZeRO分片）、**张量并行**（也称为模型并行），以及（简要）**流水线并行**。对于每一种方案，我们将展示我们承担的通信成本，以及该成本在什么时候开始限制我们的计算成本。<d-footnote>我们将专注于通信边界——因为虽然内存容量约束很重要，但在使用重计算（激活检查点）和预训练期间大量芯片时，它们通常不会限制我们。我们在这里也不讨论MoE的专家并行——这大大扩展了设计空间，只讨论密集Transformer的基本情况。</d-footnote> 对于本节，您可以只专注于芯片间通信成本，因为只要我们有足够大的单芯片批量大小，从HBM到MXU的数据传输已经与计算重叠。

我们将使用以下符号来简化本节中的计算。

| 符号 | 含义（模型参数）                                             |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub>（隐藏维度/残差流维度）      |
| F        | **d**<sub>ff</sub>（前馈维度）                        |
| B        | 批量维度（批次中的token数量；总数，不是每设备） |
| T        | 序列长度                                                        |
| L        | 模型中的层数                                          |

| 符号 | 含义（硬件特性）                                                                 |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 每芯片FLOPS/s                                                                                  |
| W        | 网络带宽（双向，通常以下标表示，例如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$) |
| X        | 沿网格轴X的芯片数量                                                                 |
| Y        | 沿另一个标记为Y的网格轴的芯片数量                                           |
| Z        | 沿第三个标记为Z的网格轴的芯片数量                                                |

为了简单起见，**我们将Transformer近似为MLP块的堆栈**——正如我们在[第4节](../transformers)中看到的，对于较大的模型，注意力在FLOPs中占比较小的部分。我们还将忽略门控矩阵乘法，为每一层留下以下简单结构：

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>Figure:</b> a simplified Transformer layer. We treat each FFW block as a stack of two matrices <b>W<sub>in</sub></b>: <code>bf16[D, F]</code> (up-projection) and <b>W<sub>out</sub></b>: <code>bf16[F, D]</code> (down-projection) with an input <b>In</b>: <code>bf16[B, D]</code>." %}

{% details 这里是我们没有并行的小Transformer的完整算法。 %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**反向传播：** 需要计算dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前一层需要*)

</div>

我们提供这个用于与添加了通信的算法进行比较。

{% enddetails %}

以下是我们将讨论的4种并行方案。每种方案都可以被认为是由上图中**In**、**W<sub>in</sub>、W<sub>out</sub>和Out**的分片唯一定义的。

**1. 数据并行：** *激活沿批量分片，参数和优化器状态在每个设备上复制。通信只在反向传播期间发生。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 完全分片数据并行（FSDP或ZeRO-3）：** *激活沿批量分片（像纯数据并行），参数沿相同网格轴分片并在前向传播中使用前及时AllGather。优化器状态也沿批量分片。减少重复内存。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称为Megatron分片或模型并行）：** *激活沿D（$d_\text{model}$）分片，参数沿F（$d_{ff}$）分片。在每个块之前和之后AllGather和ReduceScatter激活。与FSDP兼容。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层维度分片，激活微批次化并沿层维度滚动。流水线阶段之间的通信是最小的（只是将激活移动单跳）。为了滥用符号：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

当您的模型即使使用很小的批量大小（>240个token，以便达到计算受限）也能适应单个芯片时，**您应该始终使用简单的数据并行。** 纯数据并行将我们的激活分割到任意数量的TPU上，只要TPU的数量小于我们的批量大小。前向传播不涉及通信，但在每步结束时，**每个TPU对其本地梯度执行AllReduce以在更新参数之前同步它们。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>Figure:</b> a diagram of pure data parallelism (forward pass). Our activations (left) are fully sharded along the batch dimension and our weights are fully replicated, so each TPU has an identical copy of the weights. This means the total memory of our weights is increased by a factor of N, but no communication is required on the forward-pass." %}

{% details 这里是前向和反向传播的完整算法。我们滥用符号将 dL/dOut 写作 dOut，纯粹为了简洁。 %}

<div markdown=1 class="algorithm">

**纯数据并行算法：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前一层需要*)

</div>

我们忽略损失函数的细节并将 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 缩写。请注意，虽然我们的最终损失是平均值 **AllReduce**(Loss[B<sub>X</sub>])，但我们只需要在反向传播中平均权重梯度时计算 AllReduce。

{% enddetails %}

请注意，前向传播没有通信——**所有通信都在反向传播中**！反向传播还有一个很好的特性，即 AllReduces 不在"关键路径"上，这意味着每个 AllReduce 可以在任何方便的时候执行，不会阻塞您执行后续操作。如果总体通信成本超过我们的总计算成本，_仍然可能成为瓶颈_，但从实现角度来看，它要宽容得多。我们将看到模型/张量并行性没有这个特性。

**为什么要这样做？** 纯数据并行通过沿批量维度分割激活来减少激活内存压力，只要我们有更多的芯片来分割批量维度，就可以几乎任意增加批量大小。特别是在训练期间，当激活通常主导我们的内存使用时，这非常有帮助。

**为什么不这样做？** 纯数据并行对减少模型参数或优化器状态的内存压力没有任何作用，这意味着纯数据并行很少适用于大规模的有趣模型，在这些模型中我们的参数 + 优化器状态无法适应单个 TPU。为了给出一个规模感，如果我们使用 bf16 参数和 fp32 优化器状态与 Adam 一起训练<d-footnote>Adam 存储参数、一阶和二阶累加器。由于参数是 bfloat16 而优化器状态是 float32，这给了我们每个参数 `2 + 8 = 10` 字节。</d-footnote>，我们能容纳的最大模型有 $$\text{TPU 内存} / 10$$ 参数，例如，在具有 96GB HBM 和纯数据并行的 TPUv5p 芯片上，这大约是 9B 参数。

<p markdown=1 class="takeaway">**要点**：我们使用 Adam 和纯数据并行能训练的最大模型有 $$\text{num_params} = \text{每设备 HBM} / 10$$ 参数。对于 TPU v5p，这大约是 9B 参数。<d-footnote>请注意，这不包括梯度检查点，所以这实际上没有用。这是一个具有 1 个 token 批量的绝对下限。</d-footnote></p>

*为了在训练期间对真实模型有用，我们至少需要部分分片模型参数或优化器。*

**什么时候我们会受到通信瓶颈的限制？** 如上所示，我们每层有两个 AllReduces，每个大小为 $$2DF$$（对于 bf16 权重）。什么时候数据并行会使我们受到通信限制？

如上表所示，令 $C$ = 每芯片 FLOPs，$W_{\text{ici}}$ = **双向** 网络带宽，$X$ = 批量被分片的分片数<d-footnote>我们假设这种分片是在 ICI 网格上完成的，因此相关的网络带宽是 $W_\text{ici}$</d-footnote>。让我们计算执行相关矩阵乘法所需的时间 $$T_\text{math}$$ 和所需的通信时间 $$T_\text{comms}$$。由于这种并行方案在前向传播中不需要通信，我们只需要为反向传播计算这些量。

*通信时间：* 从上一节我们知道，在 1D 网格中执行 AllReduce 所需的时间仅取决于被 AllReduced 的数组的总字节数和 ICI 带宽 $W_\text{ici}$；具体来说，AllReduce 时间是 $2 \cdot \text{总字节数} / W_\text{ici}$。由于我们需要对 $W_\text{in}$ 和 $W_\text{out}$ 都进行 AllReduce，我们每层有 2 个 AllReduces。每个 AllReduce 用于权重矩阵，即 $DF$ 参数的数组，或 $2DF$ 字节。综合起来，单层中 AllReduce 的总时间是

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘法时间：* 每层在前向传播中包含两个矩阵乘法，或在反向传播中包含四个矩阵乘法，每个需要 $2(B/X)DF$ FLOPs。因此，对于反向传播中的单层，我们有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们重叠计算，每层的总时间是这两个量的最大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$ 时，我们受到计算限制，或者当

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

关键是，为了在数据并行中保持计算限制，我们需要每设备批量大小 $$B / X$$ 超过 ICI 操作强度，$C / W_\text{ici}$。这最终是因为计算时间随每设备批量大小缩放，而通信时间与这个量无关（因为我们传输的是模型权重）。注意 $B > C/W_\text{ici}$ 条件与单设备计算限制规则 $B > 240$ 的相似性；在那种情况下，规则也来自计算时间随批量大小缩放，而数据传输大小（在 $B \ll F, D$ 状态下）与批量大小无关。

让我们代入一些实际数字来感受一下规模。对于 TPUv5p，在 ICI 上进行 1D 数据并行时 `C=4.6e14` 且 `W=2 * 9e10`，所以**我们每芯片批量大小必须至少为 2,550 以避免受到通信限制**。由于我们可以在多个轴上进行数据并行，如果我们将 TPUv5p pod 的所有三个轴都专用于纯数据并行，我们将带宽 $W_\text{ici}$ 提高 3 倍，并且可以缩小到每 TPU 仅 BS=850 或每 pod 批量 7.6M 个 token（8960 个芯片）！**这告诉我们，受到纯数据并行瓶颈是相当困难的！**

<p markdown=1 class="takeaway">**注意[上下文并行]：** 在整个本节中，$B$ 始终指的是总批量大小**以 token 计**。然而，显然，我们的批量由许多不同的序列组成，那么这是如何工作的呢？就 MLP 而言，**token 就是 token**！无论它们属于相同序列还是两个不同序列都没有关系。所以我们或多或少可以在批量和序列维度上都进行数据并行：我们称之为上下文并行或序列并行，但您可以将其视为只是另一种数据并行。注意力比 MLP 更复杂，因为我们在注意力期间进行一些跨序列计算，但这可以通过在注意力期间收集 KVs 或 Qs 并仔细重叠 FLOPs 和通信（通常使用称为"环形注意力"的东西）来处理。在本节中，我们将完全忽略我们的序列维度，并假设一定数量的批量或序列并行。</p>

### Fully-Sharded Data Parallelism (FSDP)

**Syntax:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

完全分片数据并行（通常称为 FSDP 或 ZeRO 分片<d-cite key="zero"></d-cite>）将模型优化器状态和权重跨数据并行分片分割，并根据需要有效地收集和分散它们。**与纯数据并行相比，FSDP 大幅减少了每设备内存使用量并节省了反向传播 FLOPs，开销非常小。**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>Figure:</b> FSDP shards the contracting dimension of Win and the output dimension of Wout along the data dimension. This reduces memory but (from Section 3) requires us to gather the weights for W before we perform the matmul. Note that the activations (left) <it>are not sharded along the contracting dimension</it>, which is what forces us to gather. <b>Note that our weight optimizer state is likewise sharded along the contracting dimension.</b>" %}

您会记得（来自[第3节](../sharding)）AllReduce 可以分解为 AllGather 和 ReduceScatter。这意味着，与其为标准数据并行执行完整的梯度 AllReduce，我们可以跨芯片分片权重和优化器状态，在前向传播期间在每一层 AllGather 它们，并在反向传播期间对权重进行 ReduceScatter，无需额外成本。

{% details 这是 FSDP 的完整算法。 %}

<div markdown=1 class="algorithm">

**完全分片数据并行（FSDP）：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*不在关键路径上，可以在前一层执行*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*现在可以丢弃 W<sub>in</sub>[D, F]*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*不在关键路径上，可以在前一层执行*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步执行*)
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*可以提前完成*)
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(可以在这里丢弃 W<sub>out</sub>[F, D])*
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = dTmp[B<sub>X</sub>, F] \*<sub>B</sub> In[B<sub>X</sub>, D]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(not on critical path, can be done async)*
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*可以提前完成*)
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*needed for previous layers) (can throw away W<sub>in</sub>[D, F] here*)

</div>

{% enddetails %}

这也被称为"ZeRO分片"，来自"ZeRO开销分片"，因为我们不执行任何不必要的计算或存储任何不必要的状态。ZeRO-{1,2,3}分别用于指代以这种方式分片优化器状态、梯度和权重。由于所有方法都具有相同的通信成本<d-footnote>技术上，FSDP在前向传播中添加了纯DP没有的通信，但这与反向传播的比例相同，因此应该对通信roofline没有影响。这里的关键是ZeRO-3将反向传播的AllReduce转换为AllGather和ReduceScatter，它们具有相同的总通信量。</d-footnote>，我们基本上总是可以进行ZeRO-3分片，它在一组设备上分片参数、梯度和优化器状态。

**为什么要这样做？** 标准数据并行涉及大量重复工作。每个TPU都对完整梯度执行AllReduce，然后更新完整优化器状态（在所有TPU上的相同工作），然后更新参数（再次完全重复）。对于ZeRO分片（分片梯度/优化器状态），您可以ReduceScatter梯度，而不是AllReduce，只更新您的优化器状态分片，更新参数分片，然后根据前向传播的需要AllGather参数。

**什么时候我们会受到通信瓶颈的限制？** 我们的相对FLOPs和通信成本与纯数据并行完全相同，因为反向传播中的每个AllReduce都变成了AllGather + ReduceScatter。回想一下，AllReduce被实现为AllGather和ReduceScatter，每个都有一半的成本。这里我们模拟前向传播，因为它与反向传播具有相同的FLOPs与通信比例：

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此，与纯数据并行一样，当 $$B / X > C / W_\text{ici}$$ 时，我们受到计算限制，即当每设备批量大小 $B/X$ 超过"ICI操作强度" $C/W_\text{ici}$（对于v5p为 `4.59e14 / 1.8e11 = 2550`）。这对我们很有利，因为这意味着如果我们的每设备批量大小足够大以至于纯数据并行受到计算限制，我们可以——无需担心离开计算限制区域——简单地升级到FSDP，为自己节省大量的参数和优化器状态内存！尽管我们确实必须在前向传播中添加通信，但这个成本无关紧要，因为它只是与前向传播FLOPs重叠。

<p markdown=1 class="takeaway">**要点**：FSDP和纯数据并行在TPUv5上当每设备批量大小小于 $2550 / M_X$ 时都会受到带宽限制，其中 $M_X$ 是网格轴的数量。</p>

例如，DeepSeek-V2（少数最近发布其训练批量大小信息的强模型之一）使用了约40M个token的批量大小。**这允许我们在达到带宽限制之前扩展到大约47,000个芯片，或大约5个TPUv5 pod。**

对于LLaMA-3 70B，其训练了大约 `6.3e24 (15e12 * 70e9 * 6)` FLOPs，我们可以将16M个token的批量分割到大约 `16e6 / (2550 / 3) = 18,823` 个芯片（大约2个8960个芯片的pod）上，每个芯片具有 `4.59e14` FLOPs，以50%的峰值FLOPs利用率（通常称为MFU）运行，并且**在大约17天内完成训练**。不错！但让我们探索如何做得更好。

<p markdown=1 class="takeaway">**关于关键批量大小的注意事项**：有点违反直觉的是，当我们的总批量大小减小时（固定芯片数量），我们会受到更多的通信瓶颈。数据并行和FSDP允许我们扩展到任意多的芯片，只要我们能不断增加批量大小！然而，在实践中，随着我们的批量大小增加，我们往往会看到训练收益递减，因为我们的梯度几乎变得无噪声。我们有时也会看到训练不稳定。因此，在"无限计算区域"中寻找最优分片方案的游戏通常从固定的批量大小（由扩展定律确定）和已知的（大量）芯片数量开始，然后旨在找到一种分区方案，允许我们将这个小批量大小适配到这么多芯片上。</p>

### 张量并行

**语法：** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$（我们使用 $$Y$$ 以便最终与FSDP结合）

在完全分片的数据并行AllReduce中，我们在芯片之间移动权重。我们也可以分片模型的前馈维度并在层内移动激活——这被称为"1D模型并行"或Megatron分片<d-cite key="megatron"></d-cite>。这可以解锁每个pod更小的有效批量大小。下图显示了以这种方式分片单个矩阵的示例：

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 基本张量并行的示例。由于我们只在Y上分片激活（不像在FSDP中我们在X上分片），我们在X上复制激活。使用我们的标准语法，这是 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]。因为我们只在一个收缩维度上分片，我们通常在矩阵乘法之前AllGather激活 <b>A</b>。" %}

如上所述，**In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着我们必须在第一个矩阵乘法之前收集激活。当激活小于权重时，这比ZeRO分片更便宜。**这通常只有在添加了一些ZeRO分片（减少收集的大小）时才成立。这是我们倾向于混合ZeRO分片和张量并行的原因之一。

{% details 这里是张量并行的算法！ %}

<div markdown=1 class="algorithm">

**张量并行：**

**前向传播：** 需要计算 Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(在关键路径上)*
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(不在收缩维度上分片，所以没有通信)*
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *(在关键路径上)*
5.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可以在这里丢弃 dOut[B, D])*
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(这可以通过与前向传播的(1)共享来跳过)*
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = dTmp[B, F<sub>Y</sub>] \*<sub>B</sub> In[B, D]
7.  dIn[B, D] {U.Y} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前一层需要)*
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

张量并行的一个优点是它与我们的Transformer前向传播中的两个矩阵很好地交互。天真地，我们会在两个矩阵中的每一个之后执行AllReduce。但在这里我们首先执行 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]** 然后执行 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**。这意味着我们在开始时AllGather **In**，在结束时ReduceScatter **Out**，而不是执行AllReduce。

**这有多大成本？** 让我们只模拟前向传播——反向传播只是这里每个操作的转置。在1D张量并行中，我们在第一个矩阵乘法之前AllGather激活，在第二个之后ReduceScatter它们，一次发送两个字节（bf16）。让我们弄清楚什么时候我们会受到通信瓶颈的限制。

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

注意我们希望计算成本大于通信成本，我们得到：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此例如，对于TPUv5p，$C / W_{ici} = 2550$（bf16），所以我们只能进行最多 $Y < F / 2550$ 的张量并行。当我们有多个ICI轴时，我们的 $T_\text{comms}$ 减少了 $M_Y$ 倍，所以我们得到 $Y < M_Y \cdot F / 2550$。

<p markdown=1 class="takeaway">**要点**：当 $Y > M_Y \cdot F / 2550$ 时，张量并行会受到通信限制。对于大多数模型，这是8到16路张量并行之间。</p>

**请注意，这不依赖于计算的精度**，因为例如对于int8，在TPUv5p上，$$C_\text{int8} / W_{ici}$$ 是 $$5100$$ 而不是 $$2550$$，但通信量也减半，所以两个因子2相互抵消。

**让我们思考一些例子：**

* 在TPUv5p上使用LLaMA 3-70B，其中 $$D = 8192,$$ $$F \approx 30,000$$，我们可以舒适地进行8路张量并行，但在16路张量并行上会受到通信限制。8路模型分片所需的F是20k。

* 对于Gemma 7B，$$F \approx 50k$$，所以我们在19路张量并行时会受到通信限制。这意味着我们可能可以进行16路并行并且仍然看到良好的性能。

### 结合FSDP和张量并行

**语法：** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP和张量并行的优点是它们可以结合使用。通过在两个轴上分片 **W<sub>in</sub>** 和 **W<sub>out</sub>**，我们既节省了内存又节省了计算。因为我们沿X分片B，我们减少了模型并行AllGathers的大小，并且因为我们沿Y分片F，我们减少了FSDP的通信开销。这意味着两者的结合可以使我们获得比上面看到的更小的有效批量大小。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 结合FSDP和张量并行的示意图。与其他情况不同，这里没有模型参数的重复。" %}

{% details 这里是混合FSDP + 张量并行的完整算法。虽然我们有很多通信，但我们所有的AllGathers和ReduceScatters都更小，因为我们对激活进行了批量分片，对权重进行了更多的张量分片！ %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
5.  Out[B<sub>X</sub>, D] {U.Y} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*
7.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可以在这里丢弃 dOut[B, D])*
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(不在关键路径上 + 这可以与前一层(2)共享)*
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> In[B<sub>X</sub>, D]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
11. dIn[B<sub>X</sub>, D] {U.Y} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前一层需要)*
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

**FSDP和TP的正确组合是什么？** 一个简单但关键的准则是FSDP移动权重，张量并行移动激活。这意味着随着我们的批量大小缩小（特别是随着我们进行更多的数据并行），张量并行变得更便宜，因为我们每分片的激活更小。

* 张量并行执行 $$\mathbf{AllGather}_Y([B_X, D_Y])$$，随着 $$X$$ 增大而缩小。
* FSDP执行 $$\mathbf{AllGather}_X([D_X, F_Y])$$，随着 $$Y$$ 增大而缩小。

因此，通过结合两者，我们可以将每个副本的最小批量大小进一步降低。我们可以用与上面相同的方式计算FSDP和TP的最优数量：

设 $$X$$ 是专门用于FSDP的芯片数量，$$Y$$ 是专门用于张量并行的芯片数量。设 $$N$$ 是我们切片中的芯片总数，其中 $$N=XY$$。设 $$M_X$$ 和 $$M_Y$$ 是我们分别进行FSDP和TP的网格轴数量（这些应该大致总和为3）。我们将纯粹模拟前向传播，因为它具有每个FLOP最多的通信。然后累加上面算法中的通信，我们有

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样，我们的总FLOPs时间是

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为了简化分析，我们做两个假设：首先，我们允许 $X$ 和 $Y$ 取非整数值（只要它们为正且满足 $XY=N$）；其次，我们假设我们可以完全重叠 $X$ 和 $Y$ 轴上的通信。在第二个假设下，总通信时间是

$$T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)$$

在我们询问在什么条件下我们会受到计算限制之前，让我们找到 $X$ 和 $Y$ 的最优值以最小化我们的总通信。由于我们的FLOPs与 $X$ 和 $Y$ 无关，最优设置是那些简单地最小化通信的设置。为此，让我们用 $X$ 和 $N$（这是固定的，因为它是我们系统中的芯片数量）而不是 $X$ 和 $Y$ 来表示上面的 $T_\text{comms}$：

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

因为 $T_\text{FSDP comms}$ 在 $X$ 中单调递增，而 $T_\text{TP comms}$ 在 $X$ 中单调递减，当 $T_\text{FSDP comms} = T_\text{TP comms}$ 时必须最小化最大值，这发生在

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这非常有用！这告诉我们，对于给定的 $B$、$F$ 和 $N$，什么数量的FSDP是最优的。让我们感受一下规模。代入现实值，即 $N = 64$（对应于4x4x4的芯片阵列）、$B=48,000$、$F=32768$，大致得到 $X\approx 13.9$。所以我们会选择 $X$ 为16，$Y$ 为4，接近我们计算的最优值。

<p markdown=1 class="takeaway">**要点**：一般来说，在训练期间，FSDP的最优数量是 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$。</p>

现在让我们回到我们一直在询问所有并行策略的问题：**在什么条件下我们会受到计算限制？** 由于我们可以重叠FLOPs和通信，当以下情况时我们受到计算限制：

$$\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}$$

通过设 $\alpha \equiv C / W_\text{ici}$（ICI算术强度），我们可以简化：

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

由于我们计算 $X_{opt}$ 使LHS最大值相等，我们可以将其代入任一侧（注意 $Y_{opt} = N/X_{opt}$），即

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

进一步简化，我们发现

$$ \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

其中左侧与通信时间成比例，右侧与计算时间成比例。请注意，虽然计算时间随批量大小线性缩放（无论并行性如何都是如此），但通信时间随批量大小的平方根缩放。因此，计算与通信时间的比率也随批量大小的平方缩放：

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}. $$

为了确保这个比率大于一以至于我们受到计算限制，我们要求

$$ \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

要获得近似数字，再次代入 $F=32,768$、$\alpha=2550$ 和 $M_X M_Y=2$（因为对于3D网格必须如此）。这大致给出 $B/N > 99$。与纯数据并行（或FSDP）情况相比，这大致为我们赢得了8倍的因子，在纯数据并行情况下，假设3D网格，我们计算 $B/N$ 必须超过约850才能受到计算限制。

<p markdown=1 class="takeaway">**要点**：将张量并行与FSDP结合允许我们将 $B/N$ 降低到 $$2550^2 / 2F$$。这让我们可以处理每个芯片低至100的批量，这比仅使用FSDP可以实现的要小大约8倍。</p>

下面我们绘制了混合FSDP + TP的FLOPs与通信时间比率，将其与仅张量并行（TP）和仅数据并行（FSDP）在代表性的4x4x4芯片阵列上进行比较。虽然纯FSDP并行在非常大的批量大小上占主导地位，但在批量大小除以芯片数量在100到850之间的区域，需要混合FSDP + TP策略才能受到计算限制。

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>Figure:</b> 在TPUv5p 4x4x4切片上F=30k的最优混合FSDP/TP的FLOPs与通信时间比率。如预期，张量并行与批量大小有固定比率；理想的混合FSDP + TP随 $\sqrt{B}$ 缩放，FSDP随 $B$ 缩放。然而，在中间批量大小区域，只有FSDP + TP 实现大于1的比率。"%}

这是TPU v5p 16x16x16的另一个示例，显示了不同分片方案的FLOPs和通信时间作为批量大小的函数。

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>Figure:</b> 不同并行方案的通信时间。黑色虚线是矩阵乘法FLOPs花费的时间，所以任何高于这条线的曲线都受到通信限制。我们注意到所有策略在批量大小低于6e5时都受到通信限制，这与我们预期的4096 * 2550^2 / (2 * 8192 * 4) = 4e5一致。" %}

黑色曲线是花费在模型FLOPs上的时间量，意味着任何批量大小低于所有通信成本的情况都严格受到通信限制。你会注意到黑色曲线在大约 `4e5` 处与绿色曲线相交，正如预测的那样。

这是一个与此交互的动画，显示了不同批量大小的总计算时间和通信时间：

<div class="l-page">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

你会注意到这通常与上述一致（最小值在FSDP=256，TP=16附近），加上或减去一些摆动因子，因为每个轴的数量有一些轻微差异。

### 流水线并行

您可能会注意到我们在前面的章节中完全避开了讨论流水线并行。流水线并行是GPU并行性的主要策略，在TPU上相对不那么重要。简而言之，流水线训练涉及将模型的层分割到多个设备上，并在前向和反向传播期间在流水线阶段之间传递激活。算法大致如下：

1. 在TPU 0上初始化您的数据，权重沿层维度分片（对于与FSDP和张量并行的流水线，$W_\text{in}[L_Z, D_X, F_Y]$）。
2. 在TPU 0上执行第一层，然后将得到的激活复制到TPU 1，重复直到到达最后一个TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对于最后一个流水线阶段，计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，然后将 $\partial L / \partial x_{L-1}$ 复制到前一个流水线阶段并重复直到到达TPU 0。

{% details 这里是一些（可工作的）Python伪代码 %}

这个伪代码应该在Cloud TPU VM上运行。虽然它不是很高效或现实，但它给您一种数据如何在设备之间传播的感觉。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# 假设每一层只是一个矩阵乘法。
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# 假设我们有 num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # 编造一些假的损失函数

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(0, num_layers, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i + 1], weights[i])
  dx, dw = f_vjp(dx)  # 计算 jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # 更新我们的权重

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是个好主意？** 流水线并行有很多优点：它在流水线阶段之间的通信成本很低，意味着即使使用低带宽互连，您也可以训练非常大的模型。这在GPU上通常很有用，因为它们不像TPU那样通过ICI密集连接。

**为什么这很困难/烦人？** 您可能已经注意到上面的伪代码中TPU 0几乎总是空闲的！它只在流水线的第一步和最后一步工作。空闲期被称为流水线气泡，处理起来非常烦人。通常我们首先尝试通过微批量来缓解这个问题，微批量发送多个小批量通过流水线，保持TPU 0至少在总步骤时间的更大比例中被利用。

第二种方法是仔细重叠前向矩阵乘法 $W_i @ x_i$、反向 $dx$ 矩阵乘法 $W_i @ \partial L / \partial x_{i+1}$ 和 $dW$ 矩阵乘法 $\partial L / \partial x_{i+1} @ x_i$。由于这些都需要一些FLOPs，我们可以重叠它们来完全隐藏气泡。这是最近的DeepSeek v3论文<d-cite key="DeepSeek3"></d-cite>中的一个图，显示了他们的"无气泡"流水线调度：

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>Figure:</b> DeepSeek v3流水线调度（来自他们的<a href=\"https://arxiv.org/pdf/2412.19437\">最近论文</a>）。橙色是前向矩阵乘法，绿色是dL/dx矩阵乘法，蓝色是dL/dW矩阵乘法。通过优先处理反向dL/dx乘法，我们可以避免\"搁置\"FLOPs。" %}

因为它对TPU（具有更大的互连pod）来说不那么关键，我们不会深入探讨，但理解关键流水线瓶颈是一个很好的练习。

### 跨Pod扩展

最大的可能TPU切片是具有8960个芯片（和2240个主机）的TPU v5p SuperPod。当我们想要扩展到超过这个大小时，我们需要跨越数据中心网络（DCN）边界。每个TPU主机配备一个或多个NIC（网络接口卡），通过以太网将主机连接到其他TPU v5p pod。如[TPU章节](../tpus)中所述，每个主机具有大约200Gbps（25GB/s）的全双工DCN带宽，即每个TPU大约6.25GB/s的全双工（出口）带宽。

通常，当扩展到单个pod之外时，我们在ICI域内进行某种形式的模型并行或FSDP，然后在多个pod之间进行纯数据并行。设 $N$ 是我们想要扩展到的TPU数量，$M$ 是每个ICI连接切片中的TPU数量。要在DCN上进行AllReduce，我们可以在pod集合上进行环形归约，给我们（在反向传播中）：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随 $M$ 缩放，因为不像ICI，当我们扩展我们的ICI域并获得更多NIC时，总带宽会增长。简化，我们发现当以下情况时 $T_\text{math} > T_\text{comms}$：

$$\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}$$

对于TPU v5p，$\frac{C}{W_\text{dcn}}$ 大约是 `4.46e14 / 6.25e9 = 71,360`。这告诉我们，为了在DCN上高效扩展，每个ICI域需要一个最小批量大小来出口每个节点。

**这是个多大的问题？** 举一个具体的例子，假设我们想在TPU v5p上训练LLaMA-3 70B，批量大小为2M个token。LLaMA-3 70B具有 $F\approx 30,000$。从上面的章节，我们知道以下内容：

* 我们可以进行张量并行，最多到 $Y = M_Y \cdot F / 2550 \approxeq 11 \cdot M_Y$。
* 只要 $B / N > 2550 / M_X$，我们就可以进行FSDP。这意味着如果我们想要用BS=2M和3个数据并行轴训练，我们最多能够使用 $\approx 2400$ 个芯片，大约是TPU v5p pod的四分之一。
* 当我们结合FSDP + 张量并行时，当我们有 $B / N < 2550^2 / 2 * 30,000 = 108$ 时会受到通信限制，所以这让我们可以扩展到大约18k个芯片！然而，TPU v5p pod的最大大小是8k个芯片，所以超出那个我们必须使用DCN。

长话短说，我们有一个很好的配方，使用大约X（FSDP）= 1024和Y（TP）= 8来训练BS=1M，但用BS=2M我们需要使用DCN。如上所述，我们有一个DCN算术强度 $\text{71,360}$，所以我们只需要确保我们的每个ICI域的批量大小大于这个。这对我们来说是微不足道的，因为用2个pod，我们每个pod的BS是1M，每个GPU的批量大小是111，这很好（可能有点接近，但理论上合理）。

<p markdown=1 class="takeaway">**要点**：只要我们的每pod批量大小至少为71k个token，使用纯数据并行在多个TPU pod之间扩展是相当直接的。</p>

## 在TPU上训练LLM的经验总结

* 增加并行性或减少批量大小都会使我们更容易受到通信限制，因为它们减少了每个芯片执行的计算量。

* 在合理的上下文长度（~32k）内，我们可以将Transformer建模为MLP块的堆栈，并通过它们如何分片每层的两个/三个主要矩阵乘法来定义几种并行方案。

* 在训练期间，我们考虑4种主要的并行方案，每种都有自己的带宽和计算要求（数据并行、FSDP、张量并行）。

| **策略**                                 | **描述**                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **数据并行**                         | 激活沿批量分片，其他一切都完全复制，我们在反向传播期间all-reduce梯度。                                                                      |
| **FSDP**                                     | 激活、权重和优化器沿批量分片，权重在使用前立即收集，梯度被reduce-scattered。                                                               |
| **张量并行（又名Megatron、模型）** | 激活沿 $$d_\text{model}$$ 分片，权重沿 $$d_{ff}$$ 分片，激活在 W<sub>in</sub> 之前收集，结果在 W<sub>out</sub> 之后reduce-scattered。 |
| **混合FSDP + 张量并行**          | 以上两者，其中FSDP收集模型分片权重。                                                                                                                           |

以下是每种方法的"公式"：

$$\small
\begin{array}{cc}
\text{策略} & \text{公式}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 这些策略中的每一种都有一个限制，在这个限制下它会受到网络/通信限制，基于它们的每设备计算和通信。这里是每层的计算和通信，假设 $$X$$ 是FSDP，$$Y$$ 是张量并行。

$$
\small
\begin{array}{ccc}
\text{策略} & \text{每层计算} & \text{每层通信} \\
& \text{(忽略门控einsum)} & \text{(字节，前向+反向传播)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少有用，因为模型及其优化器状态使用的字节数 = 10倍参数数量。这意味着我们很少能在内存中容纳超过几十亿个参数。

* 当 $$\text{每分片批量大小} < C / W$$（网络的算术强度）时，数据并行和FSDP会受到通信限制。对于ICI这是2,550，对于DCN这是75,000。这可以通过更多的并行轴来增加。

* 当 $$\lvert Y\rvert > F / 2550$$ 时，张量并行会受到通信限制。**对于大多数模型，这大约是8-16路。** 这与批量大小无关。

* 混合FSDP + 张量并行允许我们将批量大小降低到低至 $$2550^2 / 2F \approx 100$$。这是非常低的。

* 跨pod的数据并行在受到DCN限制之前需要每个pod大约75,000的最小批量大小。

* 基本上，如果您的批量大小很大或模型很小，事情很简单。您可以在DCN上进行数据并行或FSDP + 数据并行。中间部分是事情变得有趣的地方。

## 一些练习题

让我们使用LLaMA-2 13B作为本节的基本模型。以下是模型详情：

| 超参数 | 值  |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2具有独立的嵌入和输出矩阵以及门控MLP块。

**问题1：** LLaMA-2 13B有多少个参数（我知道这很傻，但请做数学计算）？*注意，如[Transformer数学](../transformers)中所述，LLaMA-3有3个大的FFW矩阵，两个上投影和一个下投影。我们在本节中忽略了两个"门控"einsum矩阵，但它们的行为与本节中的W<sub>in</sub>相同。*

{% details 点击这里查看答案。 %}

* FFW参数：$$3LDF$$ = `8.5e9`
* 注意力参数：$$4DNHL$$ = `4.2e9`
* 词汇参数：$$2VD$$ = `0.3e9`
* 总计：`8.5e9 + 4.2e9 + 0.39e9 = 13.1e9`，符合预期！

{% enddetails %}

**问题2：** 假设我们使用BS=16M个token进行训练并使用Adam。暂时忽略并行性，模型的参数、优化器状态和激活使用了多少总内存？*假设我们将参数存储在bf16中，优化器状态存储在fp32中，并且每层检查点激活三次（在三个大矩阵乘法之后）。*

{% details 点击这里查看答案。 %}

参数（bf16）和两个优化器状态（fp32，一阶和二阶矩累加器）使用的总内存是 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两个矩阵乘法后的激活形状为 $BF$，最后一个为 $BD$（根据上面的Transformer图），所以bf16的总内存是 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$ 或 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，因为 `B=16e16`。所有其他激活或多或少可以忽略不计。

{% enddetails %}

**问题3：** 假设我们想在TPUv5p 16x16x16切片上使用32k序列长度和3M个token的总批量大小进行训练。假设我们要使用bfloat16权重和float32优化器，如上所述。

1. 我们可以使用纯数据并行吗？为什么或为什么不？
2. 我们可以使用纯FSDP吗？为什么或为什么不？使用纯FSDP，每个设备将使用多少内存（假设我们只在3个大的FFW矩阵之后进行梯度检查点）。
3. 我们可以使用混合FSDP + 张量并行吗？为什么或为什么不？如果可以，$X$ 和 $Y$ 应该是什么？每个设备将存储多少内存？仅使用roofline FLOPs估计并忽略注意力，在40% MFU下每个训练步骤需要多长时间？

{% details 点击这里查看答案。 %}

首先，让我们写下一些数字。使用32k序列长度和3M批量大小，我们的序列批量大小为96。在TPU v5p 16x16x16切片上，我们有 `393TB` 的HBM。

1. 我们不能使用纯数据并行，因为它在每个芯片上复制参数和优化器状态，这些已经大约130GB（来自问题2），这比我们每个芯片拥有的HBM（96GB）更多。

2. 让我们先纯粹从内存角度来看。在问题2中将BS=16M替换为3M，我们得到 `~7.86e12` 的总检查点激活，加上1.3e11的优化器状态，这使我们几乎正好达到8e12 = 8TB。TPUv5p切片总共有 `393TB` 的HBM，所以我们安全地低于HBM限制。接下来让我们看看我们会受到通信还是计算限制。使用4096个芯片和3个并行轴，我们可以做的最小批量大小是 `850 * 4096 = 3.48M` 个token。这略高于我们的3M批量大小。所以我们实际上受到了通信限制，这很糟糕。所以一般答案是**不，我们不能单独使用FSDP**。

3. 现在我们知道我们的主要关注点是受到通信限制，所以让我们代入一些数字。首先，我们从上面知道，使用混合FSDP + 张量并行时，我们的每芯片批量大小需要在这里高于 $2550^2 / 2F = 235$。这意味着我们理论上可以做到！让我们弄清楚每种要做多少。

我们有规则 $X_{opt} = \sqrt((F / B) * (M_X / M_Y) * N)$，所以在这里我们有 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，意味着我们将大致做1024路DP和4路TP。每个TPU内存将与(2)中相同，步骤时间将是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

{% enddetails %}

<h3 markdown=1 class="next-section">第5部分就到这里了！对于将此内容应用于真实LLaMA模型的第6部分，[点击这里](../applied-training)！</h3>

## 附录

### 附录A：推导反向传播的通信

上面，我们将Transformer层前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。我们如何推导反向传播所需的通信？

这相当自然地遵循前一节中单个矩阵乘法 **Y = X * A** 的规则：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

使用这个，我们得到以下公式（让 Tmp[B, F] 代表 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub> = dTmp[B, F] *<sub>B</sub> Tmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

请注意，这些公式是数学陈述，没有提到分片。反向传播的工作是计算这四个量。所以为了弄清楚必要的通信，我们只需要取上面四个方程中要矩阵乘法的所有量的分片（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>），这些由我们的并行方案指定，并使用分片矩阵乘法的规则来弄清楚我们必须做什么通信。注意 dOut 以与 Out 相同的方式分片。