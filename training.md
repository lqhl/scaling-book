---
layout: distill
title: "如何为训练并行化 Transformer"
# permalink: /main/
description: "在这里, 我们讨论了 LLM 训练中使用的四种主要并行方案: 数据并行, 完全分片数据并行 (FSDP), 张量并行和流水线并行. 对于每种方案, 我们计算了在什么时候我们会受到通信的瓶颈."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers"
previous_section_name: "Part 4: Transformers"

next_section_url: ../applied-training
next_section_name: "Part 6: Training LLaMA"

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
  - name: "我们所说的扩展是什么意思?"
  - subsections:
    - name: "数据并行"
    - name: "完全分片数据并行 (FSDP)"
    - name: "张量并行"
    - name: "结合 FSDP 和张量并行"
    - name: "流水线"
    - name: "跨 Pod 扩展"
  - name: "TPU 上 LLM 训练的要点"
  - name: "一些待解决的问题"
  - name: "附录"
  - subsections:
    - name: "附录 A: 推导后向传播通信"

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

## 我们所说的扩展是什么意思?

“模型扩展”的目标是能够在增加用于训练或推理的芯片数量的同时, 实现吞吐量的成比例线性增长 (我们称之为*强扩展*). 虽然单个芯片上的性能取决于内存带宽和 FLOPs 之间的权衡, 但集群级别的性能取决于通过将其与有用的 FLOPS 重叠来隐藏芯片间通信. 这是不平凡的, 因为增加芯片数量会增加通信负载, 同时减少我们可以用来隐藏它的每个设备的计算量. 正如我们在[第 3 节](../sharding)中看到的, 分片矩阵乘法通常需要昂贵的 AllGathers 或 ReduceScatters, 这会阻止 TPU 进行有用的工作. 本节的目标是找出它们何时变得*过于昂贵*.

在本节中, 我们将讨论四种常见的并行方案: (纯) **数据并行, 完全分片数据并行** (FSDP / ZeRO 分片), **张量并行** (也称为模型并行), 以及 (简要地) **流水线并行**. 对于每种方案, 我们将展示我们产生的通信成本以及该成本在什么时候开始成为我们计算成本的瓶颈.<d-footnote>我们将专注于通信限制 —— 因为虽然内存容量限制很重要, 但在使用重物质化 (激活检查点) 和预训练期间的大量芯片时, 它们通常不会限制我们. 我们在这里也不讨论 MoE 的专家并行 —— 这大大扩展了设计空间, 只讨论密集 Transformer 的基本情况.</d-footnote> 对于本节, 你可以只关注芯片间通信成本, 因为只要我们有足够大的单芯片批量大小, 从 HBM 到 MXU 的数据传输就已经与计算重叠了.

我们将在本节中使用以下符号来简化计算.

| 符号 | 含义 (模型参数) |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub> (隐藏维度/残差流维度) |
| F        | **d**<sub>ff</sub> (前馈维度) |
| B        | 批处理维度 (批处理中的 token 数量; 总计, 而非每个设备) |
| T        | 序列长度 |
| L        | 模型中的层数 |

| 符号 | 含义 (硬件特性) |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 每个芯片的 FLOPS/s |
| W        | 网络带宽 (双向, 通常带有下标, 例如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$ |
| X        | 网格轴 X 上的芯片数量 |
| Y        | 另一个网格轴上的芯片数量, 标记为 Y |
| Z        | 第三个网格轴上的芯片数量, 标记为 Z |

为简单起见, **我们将 Transformer 近似为一堆 MLP 块** —— 正如我们在[第 4 节](../transformers)中看到的, 对于较大的模型, 注意力是 FLOPs 的一个相对较小的部分. 我们还将忽略门控矩阵乘法, 为我们留下每层的以下简单结构:

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>图:</b> 一个简化的 Transformer 层. 我们将每个 FFW 块视为两个矩阵的堆栈 <b>W<sub>in</sub></b>: <code>bf16[D, F]</code> (上投影) 和 <b>W<sub>out</sub></b>: <code>bf16[F, D]</code> (下投影), 输入为 <b>In</b>: <code>bf16[B, D]</code>." %}

{% details 这是我们没有并行化的小 Transformer 的完整算法. %}

<div markdown=1 class="algorithm">

**前向传播:** 需要计算 Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**后向传播:** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前几层需要*)

</div>

我们提供此内容是为了与添加了通信的算法进行比较.

{% enddetails %}

以下是我们即将讨论的 4 种并行方案. 每种方案都可以被认为是由上图中 **In**, **W<sub>in</sub>, W<sub>out</sub>, 和 Out** 的分片唯一确定的.

**1. 数据并行:** *激活沿批处理分片, 参数和优化器状态在每个设备上复制. 通信仅在后向传播期间发生.*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 完全分片数据并行 (FSDP 或 ZeRO-3):** *激活沿批处理分片 (如纯数据并行), 参数沿同一网格轴分片, 并在前向传播中使用前及时进行 AllGather. 优化器状态也沿批处理分片. 减少了重复的内存.*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行 (也称为 Megatron 分片或模型并行):** *激活沿 D ($d_\text{model}$) 分片, 参数沿 F ($d_{ff}$) 分片. 在每个块之前和之后对激活进行 AllGather 和 ReduceScatter. 与 FSDP 兼容.*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行:** *权重沿层维度分片, 激活进行微批处理并沿层维度滚动. 流水线阶段之间的通信是最小的 (只是在单个跳跃上传输激活). 滥用符号:*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行

**语法:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

当你的模型可以容纳在单个芯片上, 即使批量大小很小 (>240 个 token, 以便受计算限制), **你都应该始终使用简单的数据并行.** 纯数据并行将我们的激活分布在任意数量的 TPU 上, 只要 TPU 的数量小于我们的批量大小. 前向传播不涉及通信, 但在每个步骤结束时, **每个 TPU 都会对其本地梯度执行 AllReduce, 以在更新参数之前同步它们.**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图:</b> 纯数据并行图 (前向传播). 我们的激活 (左) 沿批处理维度完全分片, 我们的权重完全复制, 因此每个 TPU 都有一个相同的权重副本. 这意味着我们权重的总内存增加了 N 倍, 但在前向传播中不需要通信." %}

{% details 这是前向和后向传播的完整算法. 我们滥用符号将 dL/dOut 写为 dOut, 纯粹是为了简洁. %}

<div markdown=1 class="algorithm">

**纯数据并行算法:**

**前向传播:** 需要计算 Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**后向传播:** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上, 可以异步完成*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*不在关键路径上, 可以异步完成*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前几层需要*)

</div>

我们忽略了损失函数的细节, 并将 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 缩写. 请注意, 尽管我们的最终损失是平均 **AllReduce**(Loss[B<sub>X</sub>]), 但我们只需要在后向传播中计算 AllReduce, 当平均权重梯度时.

{% enddetails %}

请注意, 前向传播没有通信 —— **全在后向传播中**! 后向传播还有一个很好的特性, 即 AllReduce 不在“关键路径”上, 这意味着每个 AllReduce 可以在方便的时候执行, 并且不会阻止你执行后续操作. 总体通信成本*仍然可能成为我们的瓶颈*, 如果它超过了我们的总计算成本, 但从实现的角度来看, 它要宽容得多. 我们将看到模型/张量并行不具有此特性.

**为什么要这样做?** 纯数据并行通过将我们的激活分布在批处理维度上来减少激活内存压力, 只要我们有更多的芯片来分布批处理维度, 就可以几乎任意地增加批量大小. 特别是在训练期间, 当我们的激活通常主导我们的内存使用时, 这非常有用.

**为什么不这样做?** 纯数据并行对减少模型参数或优化器状态的内存压力没有任何作用, 这意味着纯数据并行对于大规模的有趣模型很少有用, 因为我们的参数 + 优化器状态无法容纳在单个 TPU 中. 为了给你一个规模感, 如果我们用 bf16 的参数和 fp32 的 Adam 优化器状态进行训练<d-footnote>Adam 存储参数, 一阶和二阶累加器. 由于参数是 bfloat16, 优化器状态是 float32, 这给了我们每个参数 `2 + 8 = 10` 字节.</d-footnote>, 我们可以容纳的最大模型有 $$\text{TPU 内存} / 10$$ 个参数, 所以例如在具有 96GB HBM 和纯数据并行的 TPUv5p 芯片上, 这大约是 9B 个参数.

<p markdown=1 class="takeaway">**要点**: 我们可以用 Adam 和纯数据并行训练的最大模型有 $$\text{num_params} = \text{每个设备的 HBM} / 10$$. 对于 TPU v5p, 这大约是 9B 个参数.<d-footnote>请注意, 这不包括梯度检查点, 所以这实际上没有用. 这是一个批量为 1 个 token 的绝对下限.</d-footnote></p>

*为了在训练期间对真实模型有用, 我们至少需要部分地对模型参数或优化器进行分片.*

**我们什么时候会受到通信的瓶颈?** 正如我们上面看到的, 我们每层有两个 AllReduce, 每个大小为 $$2DF$$ (对于 bf16 权重). 数据并行什么时候会让我们受通信限制?

如上表所示, 设 $C$ = 每个芯片的 FLOPs, $W_{\text{ici}}$ = **双向**网络带宽, $X$ = 批处理分区的分片数量<d-footnote>我们假设这个分区是在 ICI 网格上完成的, 所以相关的网络带宽是 $W_\text{ici}$</d-footnote>. 让我们计算执行相关矩阵乘法所需的时间 $$T_\text{math}$$, 以及所需的通信时间 $$T_\text{comms}$$. 由于这个并行方案在前向传播中不需要通信, 我们只需要计算后向传播的这些量.

*通信时间:* 从上一节我们知道, 在 1D 网格中执行 AllReduce 所需的时间仅取决于被 AllReduce 的数组的总字节数和 ICI 带宽 $W_\text{ici}$; 具体来说, AllReduce 时间是 $2 \cdot \text{总字节数} / W_\text{ici}$. 由于我们需要对 $W_\text{in}$ 和 $W_\text{out}$ 都进行 AllReduce, 我们每层有 2 个 AllReduce. 每个 AllReduce 都是针对一个权重矩阵, 即一个 $DF$ 参数的数组, 或 $2DF$ 字节. 将所有这些放在一起, 单层中 AllReduce 的总时间是

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘法时间:* 每层在前向传播中包含两个矩阵乘法, 或在后向传播中包含四个矩阵乘法, 每个都需要 $2(B/X)DF$ FLOPs. 因此, 对于后向传播中的单层, 我们有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们重叠, 每层的总时间是这两个量的最大值:

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$, 或者当

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

时, 我们变得受计算限制.

结果是, 为了在数据并行下保持受计算限制, 我们需要每个设备的批量大小 $$B / X$$ 超过 ICI 操作强度 $C / W_\text{ici}$. 这最终是由于计算时间随每个设备的批量大小而扩展, 而通信时间与此量无关 (因为我们正在传输模型权重) 的结果. 请注意 $B > C/W_\text{ici}$ 条件与单设备受计算限制规则 $B > 240$ 的相似之处; 在那种情况下, 规则也来自于计算时间随批量大小而扩展, 而数据传输大小 (在 $B \ll F, D$ 情况下) 与批量大小无关的事实.

让我们输入一些真实数字来感受一下规模. 对于 TPUv5p, `C=4.6e14`, `W=2 * 9e10` (对于 ICI 上的 1D 数据并行), 所以**我们每个芯片的批量大小必须至少为 2,550 才能避免受通信限制**. 由于我们可以在多个轴上进行数据并行, 如果我们将 TPUv5p pod 的所有三个轴都用于纯数据并行, 我们的带宽 $W_\text{ici}$ 会增加 3 倍, 并且可以缩小到每个 TPU 只有 BS=850, 或每个 pod (8960 个芯片) 每批 760 万个 token! **这告诉我们, 纯数据并行很难成为瓶颈!**

<p markdown=1 class="takeaway">**注意 [上下文并行]:** 在本节中, $B$ 始终指**以 token 为单位**的总批量大小. 然而, 显然, 我们的批处理由许多不同的序列组成, 那么这是如何工作的呢? 就 MLP 而言, **token 就是 token**! 它们属于同一序列还是两个不同的序列并不重要. 所以我们或多或少可以自由地在批处理和序列维度上进行数据并行: 我们称之为上下文并行或序列并行, 但你可以将其简单地视为另一种数据并行. 注意力比 MLP 更棘手, 因为我们进行了一些跨序列计算, 但这可以通过在注意力期间收集 KV 或 Q 并仔细重叠 FLOPs 和通信来处理 (通常使用一种称为“环形注意力”的东西). 在本节中, 我们将完全忽略我们的序列维度, 并假设有一定数量的批处理或序列并行.</p>

### 完全分片数据并行 (FSDP)

**语法:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

完全分片数据并行 (通常称为 FSDP 或 ZeRO-sharding<d-cite key="zero"></d-cite>) 将模型优化器状态和权重分布在数据并行分片上, 并根据需要高效地收集和分散它们. **与纯数据并行相比, FSDP 大大减少了每个设备的内存使用量, 并在后向传播 FLOPs 上节省了开销, 开销非常小.**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>图:</b> FSDP 将 Win 的收缩维度和 Wout 的输出维度沿数据维度进行分片. 这减少了内存, 但 (从第 3 节) 要求我们在执行矩阵乘法之前收集 W 的权重. 请注意, 激活 (左) <it>没有沿收缩维度进行分片</it>, 这就是迫使我们进行收集的原因. <b>请注意, 我们的权重优化器状态同样沿收缩维度进行分片.</b>" %}

你会记得 (从[第 3 节](../sharding)), AllReduce 可以分解为 AllGather 和 ReduceScatter. 这意味着, 与其为标准数据并行进行完整的梯度 AllReduce, 我们可以将权重和优化器状态分布在芯片上, 在前向传播的每一层对它们进行 AllGather, 并在后向传播期间对权重进行 ReduceScatter, 而无需额外成本.

{% details 这是 FSDP 的完整算法. %}

<div markdown=1 class="algorithm">

**完全分片数据并行 (FSDP):**

**前向传播:** 需要计算 Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*不在关键路径上, 可以在前一层期间完成*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*现在可以丢弃 W<sub>in</sub>[D, F]*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*不在关键路径上, 可以在前一层期间完成*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**后向传播:** 需要计算 dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上, 可以异步完成*)
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*可以提前完成*)
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(可以在这里丢弃 W<sub>out</sub>[F, D]*)
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = dTmp[B<sub>X</sub>, F] \*<sub>B</sub> In[B<sub>X</sub>, D]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(不在关键路径上, 可以异步完成)*
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*可以提前完成*)
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前几层需要) (可以在这里丢弃 W<sub>in</sub>[D, F]*)

</div>

{% enddetails %}

这也被称为“ZeRO Sharding”, 来自“零开销分片”, 因为我们不执行任何不必要的计算或存储任何不必要的状态. ZeRO-{1,2,3} 分别用于指以这种方式对优化器状态, 梯度和权重进行分片. 由于所有这些都具有相同的通信成本<d-footnote>技术上, FSDP 在前向传播中增加了纯 DP 没有的通信, 但这与后向传播的比例相同, 因此它对通信屋顶线没有影响. 这里的关键是 ZeRO-3 将后向传播的 AllReduce 变成了 AllGather 和 ReduceScatter, 它们具有相同的总通信量.</d-footnote>, 我们基本上总是可以进行 ZeRO-3 分片, 它将参数, 梯度和优化器状态分布在一组设备上.

**我们为什么要这样做?** 标准数据并行涉及大量重复工作. 每个 TPU 都对完整的梯度进行 AllReduce, 然后更新完整的优化器状态 (所有 TPU 上的工作都相同), 然后更新参数 (同样, 完全重复). 对于 ZeRO 分片 (对梯度/优化器状态进行分片), 你可以对梯度进行 ReduceScatter, 而不是进行 AllReduce, 只更新你的优化器状态分片, 更新一个参数分片, 然后根据需要为你的前向传播 AllGather 参数.

**我们什么时候会受到通信的瓶颈?** 我们的相对 FLOPs 和通信成本与纯数据并行完全相同, 因为后向传播中的每个 AllReduce 都变成了 AllGather + ReduceScatter. 回想一下, AllReduce 是作为 AllGather 和 ReduceScatter 实现的, 每个的成本都是一半. 在这里, 我们对前向传播进行建模, 因为它具有与后向传播相同的 FLOPs-通信比:

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此, 与纯数据并行一样, 当 $$B / X > C / W_\text{ici}$$ 时, 我们受计算限制, 即当每个设备的批量大小 $B/X$ 超过“ICI 操作强度” $C/W_\text{ici}$ (`4.59e14 / 1.8e11 = 2550` 对于 v5p) 时. 这对我们来说很棒, 因为这意味着如果我们的每个设备的批量大小足够大, 以至于对于纯数据并行来说是受计算限制的, 我们可以 —— 不用担心离开受计算限制的状态 —— 简单地升级到 FSDP, 为我们节省大量的参数和优化器状态内存! 尽管我们确实在前向传播中增加了通信, 但这个成本是无关紧要的, 因为它只是与前向传播的 FLOPs 重叠.

<p markdown=1 class="takeaway">**要点:** 当每个设备的批量大小小于 $2550 / M_X$ 时, FSDP 和纯数据并行在 TPUv5 上都会受到带宽限制, 其中 $M_X$ 是网格轴的数量.</p>

例如, DeepSeek-V2 (最近唯一一个发布其训练批量大小信息的强大模型) 使用了约 40M token 的批量大小. **这使我们能够扩展到大约 47,000 个芯片, 或大约 5 个 TPUv5 pod, 然后我们才会达到带宽限制.**

对于 LLaMA-3 70B, 它训练了大约 `6.3e24 (15e12 * 70e9 * 6)` FLOPs, 我们可以将 16M token 的批量分布在大约 `16e6 / (2550 / 3) = 18,823` 个芯片上 (大约 2 个 8960 个芯片的 pod), 每个芯片具有 `4.59e14` FLOPs, 运行在 50% 的峰值 FLOPs 利用率 (通常称为 MFU), **并在大约 17 天内完成训练**. 不错! 但让我们探讨一下如何做得更好.

<p markdown=1 class="takeaway">**关于临界批量大小的说明**: 有点不直观的是, 随着我们总批量大小的减少 (芯片数量固定), 我们变得更加受通信瓶颈. 数据并行和 FSDP 让我们能够扩展到任意数量的芯片, 只要我们能不断增加我们的批量大小! 然而, 在实践中, 随着我们批量大小的增加, 我们往往会看到训练的回报递减, 因为我们的梯度变得几乎没有噪声. 我们有时也会看到训练不稳定. 因此, 在“无限计算状态”中寻找最佳分片方案的游戏通常从一个固定的批量大小开始, 由扩展定律确定, 以及一个已知的 (大的) 芯片数量, 然后旨在找到一个分区, 允许我们将那个小批量大小容纳在那么多芯片上.</p>

### 张量并行

**语法:** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$ (我们使用 $$Y$$ 最终与 FSDP 结合)

在完全分片的数据并行 AllReduce 中, 我们在芯片之间移动权重. 我们也可以对模型的前馈维度进行分片, 并在层内移动激活 —— 这被称为“1D 模型并行”或 Megatron 分片<d-cite key="megatron"></d-cite>. 这可以解锁每个 pod 更小的有效批量大小. 下图显示了以这种方式分片的单个矩阵的示例:

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>图:</b> 基本张量并行的示例. 由于我们只在 Y 上对激活进行分片 (与 FSDP 中我们在 X 上分片不同), 我们在 X 上复制我们的激活. 使用我们的标准语法, 这是 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]. 因为我们只在一个收缩维度上进行分片, 我们通常在矩阵乘法之前对激活 <b>A</b> 进行 AllGather." %}

如前所述, **In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着我们必须在第一个矩阵乘法之前收集我们的激活. 当激活比权重小时, 这比 ZeRO 分片更便宜.** 这通常只有在添加了一定量的 ZeRO 分片 (这减少了收集的大小) 时才成立. 这是我们倾向于混合 ZeRO 分片和张量并行的原因之一.

{% details 这是张量并行的算法! %}

<div markdown=1 class="algorithm">

**张量并行:**

**前向传播:** 需要计算 Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(在关键路径上)*
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(不在收缩维度上分片, 所以没有通信)*
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *(在关键路径上)*
5.  Loss[B] = ...

**后向传播:** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可以在这里丢弃 dOut[B, D]*)
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(可以通过与前向传播的 (1) 共享来跳过)*
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = dTmp[B, F<sub>Y</sub>] \*<sub>B</sub> In[B, D]
7.  dIn[B, D] {U.Y} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前几层需要)*
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

张量并行的一个好处是它与我们 Transformer 前向传播中的两个矩阵很好地交互. 简单地说, 我们会在两个矩阵之后都进行 AllReduce. 但在这里, 我们首先进行 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**, 然后 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**. 这意味着我们在开始时对 **In** 进行 AllGather, 在结束时对 **Out** 进行 ReduceScatter, 而不是进行 AllReduce.

**这有多昂贵?** 让我们只对前向传播进行建模 - 后向传播只是这里每个操作的转置. 在 1D 张量并行中, 我们在第一个矩阵乘法之前对激活进行 AllGather, 在第二个矩阵乘法之后对它们进行 ReduceScatter, 一次发送两个字节 (bf16). 让我们找出什么时候我们会受到通信的瓶颈.

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

注意到我们希望计算成本大于通信成本, 我们得到:

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此, 例如, 对于 TPUv5p, bf16 中的 $C / W_{ici} = 2550$, 所以我们只能进行张量并行, 直到 $Y < F / 2550$. 当我们有多个 ICI 轴时, 我们的 $T_\text{comms}$ 会减少 $M_Y$ 倍, 所以我们得到 $Y < M_Y \cdot F / 2550$.

<p markdown=1 class="takeaway">**要点**: 当 $Y > M_Y \cdot F / 2550$ 时, 张量并行会受到通信限制. 对于大多数模型, 这大约是 8 到 16 路张量并行.</p>

**请注意, 这不依赖于计算的精度**, 因为例如对于 int8, 在 TPUv5p 上, $$C_\text{int8} / W_{ici}$$ 是 $$5100$$ 而不是 $$2550$$, 但通信量也减半了, 所以两个 2 的因子抵消了.

**让我们考虑一些例子:**

*   在 TPUv5p 上, LLaMA 3-70B 的 $$D = 8192,$$ $$F \approx 30,000$$, 我们可以轻松地进行 8 路张量并行, 但在 16 路张量并行上会受到通信限制. 模型 8 路模型分片所需的 F 为 20k.

*   对于 Gemma 7B, $$F \approx 50k$$, 所以我们在 19 路张量并行时会受到通信限制. 这意味着我们可能可以进行 16 路, 并且仍然看到良好的性能.

### 结合 FSDP 和张量并行

**语法:** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP 和张量并行的好处在于它们可以结合使用. 通过在两个轴上对 **W<sub>in</sub>** 和 **W<sub>out</sub>** 进行分片, 我们既节省了内存又节省了计算. 因为我们在 X 上对 B 进行分片, 我们减少了模型并行 AllGather 的大小, 因为我们在 Y 上对 F 进行分片, 我们减少了 FSDP 的通信开销. 这意味着两者的结合可以让我们达到比上面看到的更低的有效批量大小.

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>图:</b> 结合 FSDP 和张量并行的图. 与其他情况不同, 这里没有模型参数的重复." %}

{% details 这是混合 FSDP + 张量并行的完整算法. 虽然我们有很多通信, 但我们所有的 AllGather 和 ReduceScatter 都更小, 因为我们对激活进行了批处理分片, 对权重进行了更多的张量分片! %}

<div markdown=1 class="algorithm">

**前向传播:** 需要计算 Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
5.  Out[B<sub>X</sub>, D] {U.Y} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*
7.  Loss[B<sub>X</sub>] = ...

**后向传播:** 需要计算 dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可以在这里丢弃 dOut[B, D]*)
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(不在关键路径上 + 这可以与前一层的 (2) 共享)*
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> In[B<sub>X</sub>, D]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
11. dIn[B<sub>X</sub>, D] {U.Y} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前几层需要)*
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

**FSDP 和 TP 的正确组合是什么?** 一个简单但关键的准则是, FSDP 移动权重, 张量并行移动激活. 这意味着随着我们批量大小的缩小 (特别是当我们进行更多的数据并行时), 张量并行变得更便宜, 因为我们每个分片的激活更小.

*   张量并行执行 $$\mathbf{AllGather}_Y([B_X, D_Y])$$, 随着 $$X$$ 的增长而缩小.
*   FSDP 执行 $$\mathbf{AllGather}_X([D_X, F_Y])$$, 随着 $$Y$$ 的增长而缩小.

因此, 通过结合两者, 我们可以将每个副本的最小批量大小进一步降低. 我们可以用与上面相同的方式计算 FSDP 和 TP 的最佳数量:

设 $$X$$ 是专用于 FSDP 的芯片数量, $$Y$$ 是专用于张量并行的芯片数量. 设 $$N$$ 是我们切片中的总芯片数量, $$N=XY$$. 设 $$M_X$$ 和 $$M_Y$$ 是我们分别进行 FSDP 和 TP 的网格轴数量 (这些应该大致总和为 3). 我们将纯粹对前向传播进行建模, 因为它每个 FLOP 的通信量最大. 然后将上面算法中的通信相加, 我们得到

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样, 我们的总 FLOPs 时间是

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为了简化分析, 我们做两个假设: 首先, 我们允许 $X$ 和 $Y$ 取非整数值 (只要它们是正的并且满足 $XY=N$); 其次, 我们假设我们可以完全重叠 $X$ 和 $Y$ 轴上的通信. 在第二个假设下, 总通信时间是

$$T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)$$

在我们询问在什么条件下我们会受计算限制之前, 让我们找到 $X$ 和 $Y$ 的最佳值以最小化我们的总通信. 由于我们的 FLOPs 与 $X$ 和 $Y$ 无关, 最佳设置是那些简单地最小化通信的设置. 为此, 让我们用 $X$ 和 $N$ (固定不变, 因为它是我们系统中的芯片数量) 而不是 $X$ 和 $Y$ 来写出上面的 $T_\text{comms}$:

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

因为 $T_\text{FSDP comms}$ 在 $X$ 中是单调递增的, 而 $T_\text{TP comms}$ 在 $X$ 中是单调递减的, 所以当 $T_\text{FSDP comms} = T_\text{TP comms}$ 时, 最大值必须被最小化, 这发生在

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这超级有用! 这告诉我们, 对于给定的 $B$, $F$ 和 $N$, 最佳的 FSDP 数量是多少. 让我们感受一下规模. 输入实际值, 即 $N = 64$ (对应于一个 4x4x4 的芯片阵列), $B=48,000$, $F=32768$, 大约得到 $X\approx 13.9$. 所以我们会选择 $X$ 为 16, $Y$ 为 4, 接近我们计算的最优值.

<p markdown=1 class="takeaway">**要点:** 一般来说, 在训练期间, 最佳的 FSDP 数量是 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$. </p>

现在让我们回到我们一直对所有并行策略提出的问题: **在什么条件下我们会受计算限制?** 由于我们可以重叠 FLOPs 和通信, 当

$$\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}$$

时, 我们受计算限制.

通过令 $\alpha \equiv C / W_\text{ici}$, 即 ICI 算术强度, 我们可以简化:

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

由于我们计算了 $X_{opt}$ 以使 LHS 最大值相等, 我们可以将其代入任何一边 (注意到 $Y_{opt} = N/X_{opt}$), 即

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

进一步简化, 我们发现

$$ \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

其中左边与通信时间成正比, 右边与计算时间成正比. 请注意, 虽然计算时间与批量大小成线性关系 (无论并行性如何), 但通信时间与批量大小的平方根成正比. 因此, 计算与通信时间之比也与批量大小的平方成正比:

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}. $$

为了确保这个比率大于一, 以便我们受计算限制, 我们需要

$$ \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

为了得到近似数字, 再次代入 $F=32,768$, $\alpha=2550$, 和 $M_X M_Y=2$ (对于 3D 网格必须如此). 这大约得到 $B/N > 99$. 这大约比纯数据并行 (或 FSDP) 的情况赢得了八倍, 在纯数据并行的情况下, 假设一个 3D 网格, 我们计算出 $B/N$ 必须超过大约 $850$ 才能受计算限制.

<p markdown=1 class="takeaway">**要点:** 结合张量并行和 FSDP 允许我们将 $B/N$ 降低到 $$2550^2 / 2F$$. 这让我们能够处理每个芯片低至 100 的批量, 这比我们仅使用 FSDP 所能达到的要小大约八倍.</p>

下面我们绘制了混合 FSDP + TP 的 FLOPs 与通信时间之比, 并将其与仅张量并行 (TP) 和仅数据并行 (FSDP) 在一个代表性的 4x4x4 芯片阵列上进行了比较. 虽然纯 FSDP 并行在非常大的批量大小时占主导地位, 但在每个芯片的批量大小在约 100 到 850 之间的范围内, 需要混合 FSDP + TP 策略才能受计算限制.

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>图:</b> 在 TPUv5p 4x4x4 切片上, F=30k 时, 最佳混合 FSDP/TP 的 FLOPs 与通信时间之比. 正如预期的那样, 张量并行与批量大小具有固定的比率; 理想的混合 FSDP + TP 与 $\sqrt{B}$ 成比例, FSDP 与 $B$ 成比例. 然而, 在中间批量大小范围内, 只有 FSDP + TP 才能达到大于一的比率."%}

这是 TPU v5p 16x16x16 的另一个例子, 显示了不同分片方案下 FLOPs 和通信时间作为批量大小的函数.

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>图:</b> 不同并行方案的通信时间. 黑色虚线是矩阵乘法 FLOPs 所花费的时间, 因此任何高于此线的曲线都受通信限制. 我们注意到, 所有策略在批量大小低于 6e5 时都变得受通信限制, 这与我们预期的 4096 * 2550^2 / (2 * 8192 * 4) = 4e5 一致." %}

黑色曲线是模型 FLOPs 所花费的时间, 这意味着任何批量大小, 如果这个值低于所有通信成本, 那么它就严格受通信限制. 你会注意到黑色曲线在 `4e5` 左右与绿色曲线相交, 正如预测的那样.

这是一个交互式动画, 可以让你玩转这个, 显示不同批量大小的总计算时间和通信时间:

<div class="l-page">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

你会注意到这通常与上面的一致 (最小值在 FSDP=256, TP=16 左右), 加上或减去一些摆动因子, 因为每个轴的数量略有不同.

### 流水线

你可能已经注意到, 我们在前面的章节中一直避免谈论流水线. 流水线是 GPU 并行的一种主导策略, 在 TPU 上则不那么重要. 简而言之, 流水线训练涉及将模型的层分布在多个设备上, 并在前向和后向传播期间在流水线阶段之间传递激活. 算法大致如下:

1.  在 TPU 0 上初始化你的数据, 你的权重沿层维度分片 ($W_\text{in}[L_Z, D_X, F_Y]$ 用于 FSDP 和张量并行的流水线).
2.  在 TPU 0 上执行第一层, 然后将结果激活复制到 TPU 1, 重复此过程, 直到到达最后一个 TPU.
3.  计算损失函数及其导数 $\partial L / \partial x_L$.
4.  对于最后一个流水线阶段, 计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$, 然后将 $\partial L / \partial x_{L-1}$ 复制到前一个流水线阶段, 重复此过程, 直到到达 TPU 0.

{% details 这里有一些 (可运行的) Python 伪代码 %}

这个伪代码应该可以在 Cloud TPU VM 上运行. 虽然它不是很高效或现实, 但它能让你了解数据是如何在设备之间传播的.

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# 假设每一层只是一个矩阵乘法.
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
  return jnp.mean(batch ** 2)  # 编造一个假的损失函数

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(0, num_layers, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i + 1], weights[i])
  dx, dw = f_vjp(dx)  # 计算 jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # 更新我们的权重

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是个好主意?** 流水线有很多好处: 它在流水线阶段之间的通信成本很低, 这意味着即使在低带宽互连的情况下, 你也可以训练非常大的模型. 这在 GPU 上通常非常有用, 因为它们不像 TPU 那样通过 ICI 密集连接.

**为什么这很困难/烦人?** 你可能已经在上面的伪代码中注意到, TPU 0 几乎总是空闲的! 它只在流水线的第一个和最后一个步骤中工作. 空闲期被称为流水线气泡, 处理起来非常烦人. 通常我们首先尝试用微批处理来缓解这个问题, 它通过流水线发送多个小批次, 至少在总步骤时间的更大部分内保持 TPU 0 的利用率.

第二种方法是仔细重叠前向矩阵乘法 $W_i @ x_i$, 后向 $dx$ 矩阵乘法 $W_i @ \partial L / \partial x_{i+1}$, 以及 $dW$ 矩阵乘法 $\partial L / \partial x_{i+1} @ x_i$. 由于这些都需要一些 FLOPs, 我们可以重叠它们以完全隐藏气泡. 这是最近 DeepSeek v3 论文<d-cite key="DeepSeek3"></d-cite>中的一张图, 显示了他们的“无气泡”流水线调度:

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>图:</b> DeepSeek v3 流水线调度 (来自他们<a href=\"https://arxiv.org/pdf/2412.19437\">最近的论文</a>). 橙色是前向矩阵乘法, 绿色是 dL/dx 矩阵乘法, 蓝色是 dL/dW 矩阵乘法. 通过优先处理后向 dL/dx 乘法, 我们可以避免“搁浅” FLOPs." %}

因为它对 TPU (具有更大的互连 pod) 不那么重要, 我们不会深入探讨这个问题, 但理解关键的流水线瓶颈是一个很好的练习.

### 跨 Pod 扩展

最大的 TPU 切片是 TPU v5p SuperPod, 拥有 8960 个芯片 (和 2240 个主机). 当我们想要扩展到这个规模以上时, 我们需要跨越数据中心网络 (DCN) 的边界. 每个 TPU 主机都配备了一个或多个 NIC (网络接口卡), 通过以太网将主机连接到其他 TPU v5p pod. 正如在[TPU 部分](../tpus)中指出的, 每个主机大约有 200Gbps (25GB/s) 的全双工 DCN 带宽, 这大约是每个 TPU 6.25GB/s 的全双工 (出口) 带宽.

通常, 当扩展到单个 pod 以上时, 我们在 ICI 域内进行某种形式的模型并行或 FSDP, 然后在多个 pod 之间进行纯数据并行. 设 $N$ 是我们想要扩展到的 TPU 数量, $M$ 是每个 ICI 连接切片中的 TPU 数量. 为了在 DCN 上进行 AllReduce, 我们可以对 pod 集合进行环形归约, 这给了我们 (在后向传播中):

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随 $M$ 扩展, 因为与 ICI 不同, 总带宽随着我们扩大 ICI 域并获得更多 NIC 而增长. 简化后, 我们发现当

$$\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}$$

时, $T_\text{math} > T_\text{comms}$.

对于 TPU v5p, $\frac{C}{W_\text{dcn}}$ 大约是 `4.46e14 / 6.25e9 = 71,360`. 这告诉我们, 为了在 DCN 上高效扩展, 每个 ICI 域需要一个最小的批量大小来出口每个节点.

**这有多大的问题?** 举一个具体的例子, 假设我们想在 TPU v5p 上用 2M token 的 BS 训练 LLaMA-3 70B. LLaMA-3 70B 的 $F\approx 30,000$. 从上面的章节中, 我们知道以下几点:

*   我们可以进行张量并行, 直到 $Y = M_Y \cdot F / 2550 \approxeq 11 \cdot M_Y$$.
*   只要 $B / N > 2550 / M_X$, 我们就可以进行 FSDP. 这意味着如果我们想用 BS=2M 和 3 个数据并行轴进行训练, 我们最多只能使用 $\approx 2400$ 个芯片, 大约是 TPU v5p pod 的四分之一.
*   当我们结合 FSDP + 张量并行时, 当我们有 $B / N < 2550^2 / 2 * 30,000 = 108$ 时, 会受通信限制, 所以这让我们能够扩展到大约 18k 个芯片! 然而, TPU v5p pod 的最大大小是 8k 个芯片, 所以超过这个数量我们必须使用 DCN.

TLDR 是我们有一个很好的配方, 可以用 BS=1M 进行训练, 大约使用 X (FSDP) = 1024 和 Y (TP) = 8, 但用 BS=2M 我们需要使用 DCN. 如上所述, 我们的 DCN 算术强度为 $\text{71,360}$, 所以我们只需要确保我们每个 ICI 域的批量大小大于这个值. 这对我们来说是微不足道的, 因为有 2 个 pod, 我们每个 pod 的 BS 为 1M, 每个 GPU 的批量大小为 111, 这很棒 (可能有点接近, 但理论上是合理的).

<p markdown=1 class="takeaway">**要点:** 只要我们每个 pod 的批量大小至少为 71k token, 使用纯数据并行跨多个 TPU pod 进行扩展就相当直接.</p>

## TPU 上 LLM 训练的要点

*   增加并行度或减少批量大小都倾向于使我们更加受通信限制, 因为它们减少了每个芯片执行的计算量.

*   在合理的上下文长度 (~32k) 内, 我们可以将 Transformer 建模为一堆 MLP 块, 并通过它们如何对每层的两个/三个主要矩阵乘法进行分片来定义几种并行方案.

*   在训练期间, 我们考虑 4 种主要的并行方案, 每种方案都有自己的带宽和计算要求 (数据并行, FSDP, 张量并行).

| **策略** | **描述** |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **数据并行** | 激活是批处理分片的, 其他一切都是完全复制的, 我们在后向传播期间对梯度进行 all-reduce. |
| **FSDP** | 激活, 权重和优化器是批处理分片的, 权重在使用前被收集, 梯度被 reduce-scattered. |
| **张量并行 (又名 Megatron, 模型)** | 激活沿 $$d_\text{model}$$ 分片, 权重沿 $$d_{ff}$$ 分片, 激活在 W<sub>in</sub> 之前被收集, 结果在 W<sub>out</sub> 之后被 reduce-scattered. |
| **混合 FSDP + 张量并行** | 以上两者, 其中 FSDP 收集模型分片权重. |

以下是每种方法的“公式”:

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

*   这些策略中的每一种都有一个限制, 在这个限制下它会受到网络/通信的限制, 基于它们每个设备的计算和通信. 以下是每层的计算和通信, 假设 $$X$$ 是 FSDP, $$Y$$ 是张量并行.

$$
\small
\begin{array}{ccc}
\text{策略} & \text{每层计算} & \text{每层通信} \\
& \text{(忽略门控 einsum)} & \text{(字节, 前向 + 后向传播)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

*   纯数据并行很少有用, 因为模型及其优化器状态使用的字节数 = 10 倍参数数量. 这意味着我们很少能在内存中容纳超过几十亿个参数.

*   当 $$\text{每个分片的批量大小} < C / W$$ (网络的算术强度) 时, 数据并行和 FSDP 会受通信限制. 对于 ICI, 这个值是 2,550, 对于 DCN, 这个值是 75,000. 这可以通过更多的并行轴来增加.

*   当 $$\lvert Y\rvert > F / 2550$$ 时, 张量并行会受通信限制. **对于大多数模型, 这大约是 8-16 路.** 这与批量大小无关.

*   混合 FSDP + 张量并行允许我们将批量大小降低到 $$2550^2 / 2F \approx 100$$. 这是非常低的.

*   跨 pod 的数据并行要求每个 pod 的最小批量大小约为 75,000, 然后才会受 DCN 限制.

*   基本上, 如果你的批量大小很大或者你的模型很小, 事情就很简单. 你可以进行数据并行或者 FSDP + 跨 DCN 的数据并行. 中间部分是事情变得有趣的地方.

## 一些待解决的问题

让我们使用 LLaMA-2 13B 作为本节的基本模型. 以下是模型详细信息:

| 超参数 | 值 |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2 有独立的嵌入和输出矩阵以及一个门控 MLP 块.

**问题 1:** LLaMA-2 13B 有多少参数 (我知道这很傻, 但请计算一下)? *注意, 正如在[Transformer 数学](../transformers)中一样, LLaMA-3 有 3 个大的 FFW 矩阵, 两个上投影和一个下投影. 我们在本节中忽略了两个“门控” einsum 矩阵, 但它们的行为与本节中的 W<sub>in</sub> 相同.*

{% details 点击这里查看答案. %}

*   FFW 参数: $$3LDF$$ = `8.5e9`
*   注意力参数: $$4DNHL$$ = `4.2e9`
*   词汇表参数: $$2VD$$ = `0.3e9`
*   总计: `8.5e9 + 4.2e9 + 0.39e9 = 13.1e9`, 正如预期的那样!

{% enddetails %}

**问题 2:** 假设我们用 BS=16M token 进行训练, 并使用 Adam. 暂时忽略并行性, 模型的参数, 优化器状态和激活总共使用了多少内存? *假设我们将参数存储在 bf16 中, 优化器状态存储在 fp32 中, 并且每层对激活进行三次检查点 (在三个大的 FFW 矩阵之后).*

{% details 点击这里查看答案. %}

参数 (bf16) 和两个优化器状态 (fp32, 一阶和二阶矩累加器) 使用的总内存是 `(2 + 4 + 4) * 13e9 ~ 130GB`. 前两个矩阵乘法后的激活形状为 $BF$, 最后一个之后为 $BD$ (根据上面的 Transformer 图), 所以 bf16 的总内存是 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$ 或 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`, 因为 `B=16e16`. 所有其他激活或多或少都可以忽略不计.

{% enddetails %}

**问题 3:** 假设我们想在 TPUv5p 16x16x16 切片上用 32k 序列长度和 3M token 的总批量大小进行训练. 假设我们想使用 bfloat16 权重和 float32 优化器, 如上所述.

1.  我们可以使用纯数据并行吗? 为什么或为什么不?
2.  我们可以使用纯 FSDP 吗? 为什么或为什么不? 使用纯 FSDP, 每个设备将使用多少内存 (假设我们只在 3 个大的 FFW 矩阵之后进行梯度检查点).
3.  我们可以使用混合 FSDP + 张量并行吗? 为什么或为什么不? 如果可以, $X$ 和 $Y$ 应该是什么? 每个设备将存储多少内存? 仅使用屋顶线 FLOPs 估计并忽略注意力, 在 40% MFU 下, 每个训练步骤需要多长时间?

{% details 点击这里查看答案. %}

首先, 让我们写下一些数字. 32k 序列长度和 3M 批量大小, 我们的序列批量大小为 96. 在 TPU v5p 16x16x16 切片上, 我们有 `393TB` 的 HBM.

1.  我们不能使用纯数据并行, 因为它在每个芯片上复制参数和优化器状态, 这已经大约是 130GB (来自 Q2), 这比我们每个芯片的 HBM (96GB) 还要多.

2.  让我们从纯粹的内存角度开始. 在 Q2 中用 3M 替换 BS=16M, 我们得到 `~7.86e12` 的总检查点激活, 加上 1.3e11 的优化器状态, 这使我们几乎正好达到 8e12 = 8TB. TPUv5p 切片总共有 `393TB` 的 HBM, 所以我们安全地低于 HBM 限制. 接下来让我们看看我们是受通信限制还是受计算限制. 有 4096 个芯片和 3 个并行轴, 我们可以做的最小批量大小是 `850 * 4096 = 3.48M` token. 这略高于我们的 3M 批量大小. 所以我们实际上受通信限制, 这很遗憾. 所以总的答案是**不, 我们不能单独使用 FSDP**.

3.  现在我们知道我们的主要问题是受通信限制, 所以让我们输入一些数字. 首先, 我们从上面知道, 我们每个芯片的批量大小与混合 FSDP + 张量并行需要高于 $2550^2 / 2F = 235$. 这意味着我们理论上可以做到这一点! 让我们弄清楚每种并行需要多少.

我们有规则 $X_{opt} = \sqrt((F / B) * (M_X / M_Y) * N)$, 所以这里我们有 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`, 这意味着我们将大约进行 1024 路 DP 和 4 路 TP. 每个 TPU 的内存将如 (2) 中所示, 步骤时间将只是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`.

{% enddetails %}

<h3 markdown=1 class="next-section">第五部分到此结束! 第六部分, 将此内容应用于真实的 LLaMA 模型, [点击这里](../applied-training)!</h3>

## 附录

### 附录 A: 推导后向传播通信

上面, 我们将 Transformer 层的前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]. 我们如何推导出后向传播所需的通信?

这很自然地遵循上一节中单个矩阵乘法 **Y = X * A** 的规则:

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

使用这个, 我们得到以下公式 (令 Tmp[B, F] 代表 In[B, D] * W<sub>in</sub>[D, F]):

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub> = dTmp[B, F] *<sub>B</sub> Tmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

请注意, 这些公式是数学陈述, 没有提到分片. 后向传播的工作是计算这四个量. 因此, 为了弄清楚所需的通信, 我们只需获取上面四个方程中要进行矩阵乘法的所有量的分片 (Tmp, dOut, W<sub>out</sub>, W<sub>in</sub>), 这些分片由我们的并行化方案指定, 并使用分片矩阵乘法的规则来弄清楚我们必须进行什么通信. 请注意, dOut 的分片方式与 Out 相同.
