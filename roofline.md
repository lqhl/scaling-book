---
layout: distill
title: "关于性能上限的一切"
# permalink: /main/
description: "当我们在硬件上运行算法时，我们受到三方面的限制：计算机进行数学运算的速度（OPs/秒）、用于移动数据的可用带宽（字节/秒）和用于存储数据的总内存（字节）。这些"性能上限"约束让我们可以上下界定给定计算的时间。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "Part 0: Introduction"

next_section_url: ../tpus
next_section_name: "Part 2: TPUs"

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

  - name: Where Does the Time Go?
  - subsections:
    - name: "Visualizing rooflines"
    - name: "Matrix multiplication"
    - name: "Network communication rooflines"
  - name: A Few Problems to Work

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

## 时间都去哪了？

让我们从一个极其简单的问题开始：*为什么一个算法需要 50ms 而不是 50s 或 5ms*？在模型内部实际发生了什么需要大量时间的事情，我们应该预期它需要多长时间？

**计算：** 深度学习模型实际上是一系列矩阵乘法，每个矩阵乘法由浮点乘法和加法"运算"（FLOPs）组成。我们的加速器速度决定了这些计算需要多长时间：

$$\begin{equation}
T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/s}}
\end{equation}$$

例如，NVIDIA H100 每秒可以执行约 9.89e14 次 bfloat16<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写，这是一种在 ML 中常用的 16 位浮点格式。</d-footnote> FLOPs，而 TPU v6e 每秒可以执行 9.1e14 次 FLOPs。<d-footnote>H100 和 B200 通常只能达到声称的峰值 FLOPs 的 80-85%，而 TPU 在正常使用中可以达到 95%。</d-footnote> 这意味着在 H100 上执行 1e12 次 FLOPs 需要（大约）`1e12 / 9.89e14 = 1.01ms`，在 TPU v6e 上需要 `1e12 / 9.1e14 = 1.1ms`。<d-footnote>请注意，这些芯片的定价不同，此比较未标准化为成本。</d-footnote>

**芯片内通信：** *在加速器内部*，张量需要在片上内存（HBM）和计算核心之间传输。你会看到这个链路的带宽被称为"HBM带宽"<d-footnote>NVIDIA 也称之为"内存带宽"。</d-footnote> 在 H100 上，[这大约是 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，在 TPU v6e 上 [这大约是 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片间通信：** 当我们将模型*分布在多个加速器上*时，张量经常需要在它们之间传输。在我们的硬件上，通常有几种选择（ICI、DCN 和 PCIe），每种都有不同的带宽。

无论通信是在芯片内还是芯片间，我们都以字节/秒来测量，并用以下公式估算总通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{通信字节数}}{\text{网络/内存带宽 字节/秒}}
\end{equation}$$

通常（但不总是），单个芯片内的计算可以与芯片内和芯片间的通信重叠。这意味着**我们可以通过使用计算和通信时间的最大值来下界训练和推理时间**。我们也可以**用它们的和来上界**。在实践中，我们针对最大值进行优化，因为代数更简单，而且我们通常可以通过重叠通信和计算来接近这个界限。如果我们以最大值为目标进行优化，那么下界和上界最多相差 2 倍，因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。然后我们通过建模"重叠区域"和开销来提高准确性，这可以通过分析你的特定模型和目标系统来获得信息。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果我们假设可以完美地重叠通信和计算，当 $T_\text{math} > T_\text{comms}$ 时，我们会看到硬件的充分利用。我们称之为"计算受限"。当 $T_\text{comms} > T_\text{math}$ 时，我们往往是"通信受限"，至少有一部分加速器 FLOPs/s 被浪费在等待数据传递上。判断一个操作是计算受限还是通信受限的一种方法是查看其"*算术强度*"或"*操作强度*"。

**定义：** 算法的算术强度由其执行的总 FLOPs 与其需要通信的字节数之比给出——无论是在芯片内还是芯片间。

$$\begin{equation}
\text{算术强度} = \frac{\text{计算 FLOPs}}{\text{通信字节数}}
\end{equation}$$

算术强度衡量给定操作的"每字节 FLOPs"。在第一阶近似中，当我们的算术强度高时，$T_\text{math}$ 相对于 $T_\text{comms}$ 很大，我们通常使用大部分可用的 FLOPs。当相反的情况成立时，我们在通信上花费更多时间并浪费 FLOPs。这种交叉发生的地方是我们硬件的"峰值算术强度"，即峰值加速器 FLOPs/s 与加速器带宽的比率。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{计算 FLOPs}} {\text{加速器 FLOPs/s}} > \frac{\text{通信字节数}}{\text{带宽 字节/秒}} & \\[0.5em]
\Leftrightarrow \frac{\text{计算 FLOPs}}{\text{通信字节数}} > \frac{\text{加速器 FLOPs/s}}{\text{带宽 字节/秒}} & \\[0.5em]
\Leftrightarrow \text{强度}(\text{计算}) > \text{强度}(\text{加速器}) & \\
\end{align*}$$

量 $\text{强度}(\text{加速器})$ 是我们的加速器达到其峰值 FLOPs/s 时的算术强度。**对于 TPU v5e MXU，这大约是 240 FLOPs/字节**<d-footnote>MXU 是 TPU 上的矩阵乘法单元。我们在这里指定这一点是因为 TPU 还有其他加速器，如 VPU，负责具有不同峰值 FLOPs/s 的逐元素操作。</d-footnote>，因为 TPU 每秒可以执行 `1.97e14` 次 FLOPs，并从 HBM 加载 `8.2e11` 字节/秒。这意味着如果一个算法的算术强度低于 240<d-footnote>这仅在算法从 HBM 加载其权重并在 MXU 中运行时才成立。正如我们将在下一节讨论的，我们有时可以将参数存储在具有更高带宽的 VMEM 中。许多算法也在 VPU 中运行，VPU 具有不同的性能特征。</d-footnote> FLOPs/字节，它将受到字节加载的限制，因此我们不会充分利用我们的硬件。让我们看一个这样的例子：

**<span style="color:#7ab5ff">示例（点积）</span>：** 要以 bfloat16 精度计算两个向量的点积，`x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从内存中加载 $x$ 和 $y$，每个都有 $2 * N = 2N$ 字节，执行 $N$ 次乘法和 $N-1$ 次加法，并将 $2$ 字节写回 HBM
$$\begin{equation}
\text{强度}(\text{点积}) = \frac{\text{总 FLOPs}}{\text{总字节数}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

当 $N\rightarrow\infty$ 时。因此点积的算术强度为 $\frac{1}{2}$，换句话说，点积每加载的字节执行 0.5 次浮点运算。这意味着我们的算术强度低于硬件的算术强度，我们将受到通信限制。<d-footnote>上面的 240 数字在这里不是正确的比较，因为正如你将在下一节看到的，点积是在 VPU 而不是 MXU 上执行的。TPU v5p VPU 每秒可以执行大约 7e12 次 FLOPs，所以其关键强度约为 3，这意味着我们在这里仍然受到一些通信限制。无论如何，我们的强度低且恒定这一事实意味着在大多数硬件上很难达到计算受限。</d-footnote>

### 可视化屋顶线图

我们可以使用**屋顶线图**来可视化内存和计算之间的权衡，该图绘制了算法在我们硬件上的峰值可达到的 FLOPs/s（吞吐量）（y 轴）与该算法的算术强度（x 轴）的关系。这是一个示例对数-对数图：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图：</b> 一个示例屋顶线图，展示了两个具有不同算术强度的算法（算法 1 和算法 2）以及它们在不同带宽（BW1 和 BW2）下的相应理论峰值吞吐量。在红色区域，算法在两种带宽下都受到带宽限制，并且浪费了硬件峰值 FLOPs/s 的一部分。黄色区域仅在较低带宽（BW1）下受到带宽限制。绿色区域在所有带宽下都是计算受限的。这里，我们使用加速器的峰值 FLOPs/s，增加带宽或提高强度不会带来好处。" %}

如上所示，随着强度增加（从左到右移动），我们最初看到算法性能（以 FLOPs/s 为单位）的线性增长，直到达到硬件的关键算术强度，对于 TPU v5e 来说是 240。任何强度较低的算法将受到带宽（BW）限制，并受到峰值内存带宽的限制（以红色显示）。右侧的任何算法将充分利用我们的 FLOPs（以绿色显示）。这里，算法 1 受通信限制，仅使用总硬件 FLOPs/s 的一部分。算法 2 是计算受限的。我们通常可以通过增加算法的算术强度或增加可用的内存带宽（从 BW1 移动到 BW2）来提高算法的性能。

### 矩阵乘法

让我们看看我们即将最喜欢的算法：矩阵乘法（又称 matmul）。我们写 $X * Y \rightarrow Z$，其中 $X$ 的形状为 $\text{bf16}[B, D]$，$Y$ 的形状为 $\text{bf16}[D, F]$，$Z$ 的形状为 $\text{bf16}[B, F]$。要进行矩阵乘法，我们需要加载 $2DF + 2BD$ 字节，执行 $2BDF$ 次 FLOPs，并写回 $2BF$ 字节。<d-footnote>技术上我们执行 $BF \times (2D - 1)$ 次 FLOPs，但这已经足够接近了。这来自于 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 节有更多细节。</d-footnote> <d-footnote>尽管矩阵乘法的输出在技术上是 float32，但我们通常在复制回 HBM 之前将其降级为 bfloat16。</d-footnote> 因此：

$$\begin{equation}
\text{强度}(\text{矩阵乘法}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设我们的"批量大小"$B$ 相对于 $D$ 和 $F$ 很小，我们可以得到一个很好的简化。然后我们得到

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approxeq \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{强度}(\text{矩阵乘法}) > \text{强度}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

对于 Transformer 矩阵乘法，这是一个合理的假设，因为我们通常有一个本地（每个副本）批量大小 $B < 1024$ 个 token（*不是序列*），但 $D$ 和 $F > 8000$。因此，当我们的每个副本<d-footnote>我们说每个副本是因为，如果我们进行某种模型分片来增加矩阵乘法中使用的芯片数量，我们将可用的计算和内存带宽按相同数量进行缩放。因此，关键批量大小对于模型权重的每个独立副本都是成立的。</d-footnote> 批量大小大于 240 个 token 时，我们通常会成为计算受限，这是一个非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 要使 bfloat16 矩阵乘法在大多数 TPU 上达到计算受限，我们需要每个副本的 token 批量大小大于 240。<d-footnote>请注意，这不是通常意义上的批量大小，即序列中的批量大小。事实证明，大多数屋顶线纯粹依赖于 token 的数量，无论它们属于相同还是不同的序列。例如，如果你在 128 个 GPU 上有 512 个序列的批量大小，每个序列 4096 个 token，那么你的总批量大小是 `512 * 4096 = 2M` 个 token，本地批量大小是 16k 个 token。</d-footnote></p>

这带有一些值得注意的注意事项，我们将在下面的问题中探讨，特别是关于量化（例如，如果我们量化激活但仍然进行全精度 FLOPs），但这是一个需要记住的好规则。对于 GPU，这个数字稍高（接近 300），但相同的结论通常成立。当我们[将大矩阵乘法分解为较小的矩阵乘法](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，瓦片大小也很重要。<d-footnote>当我们进行大型矩阵乘法时，我们需要将其分解为适合 VMEM/SMEM/TMEM（更高带宽的片上内存）的较小瓦片。这导致我们多次加载块，因此我们只加载 $O(N^2)$ 字节不再完全正确。考虑一个具有瓦片大小 $bm$、$bk$、$bm$ 的 $(m, k) \cdot (k, n)$ 矩阵乘法。令 $tm = m / bm$ 等。那么总 FLOPs 是 $2 \cdot tm \cdot tn \cdot tk \cdot m \cdot bk \cdot bm$，总字节数是 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + 2 \cdot bm \cdot bn)$。忽略最后一项，我们的强度是 $bm \cdot bn / (bm + bn)$，这与上述类似。</d-footnote> 我们将在[下一节](../tpus)中讨论较低级别的 GPU 和 TPU 细节。

### 网络通信屋顶线

到目前为止我们讨论的所有屋顶线都是内存带宽屋顶线，_全部在单个芯片内_。这不应被视为规则。事实上，在这本书中我们关心的大多数屋顶线都涉及芯片间的通信：通常是涉及跨多个 TPU 分片的矩阵的矩阵乘法。

举一个有点人为的例子，假设我们想要将两个大矩阵 $X\sim \text{bfloat16[B, D]}$ 和 $Y \sim \text{bfloat16[D, F]}$ 相乘，它们在 2 个 TPU/GPU 上均匀分割（沿着 $D$ 维度）。要进行这个乘法（正如我们将在[第 3 节](../sharding)中看到的），我们可以在每个 TPU 上将每个矩阵的一半相乘（在 TPU 0 上 `A = X[:, :D // 2] @ Y[:D // 2, :]`，在 TPU 1 上 `B = X[:, D // 2:] @ Y[D // 2:, :]`），然后将生成的"部分和"复制到另一个 TPU 并将它们相加。假设我们可以在每个方向上复制 `4.5e10` 字节，并在每个芯片上执行 `1.97e14` FLOPs/s。$T_\text{math}$ 和 $T_\text{comms}$ 是什么？

$T_\text{math}$ 显然是之前的一半，因为每个 TPU 只做一半的工作，即<d-footnote>我们忽略了将两个部分和相加所需的 FLOPs（另外 DF 次加法），但这基本上可以忽略不计。</d-footnote>

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{加速器 FLOPs/s}} = \frac{BDF}{1.97e14}$$

那么 $T_\text{comms}$ 呢？现在这指的是芯片间的通信时间！这只是发送的总字节数除以网络带宽，即

$$T_\text{comms} = \frac{2BF}{\text{网络带宽}} = \frac{2BF}{4.5e10}$$

因此，当 $$\text{强度}(\text{矩阵乘法 (2-芯片)}) > \text{强度}(\text{TPU 相对于芯片间网络})$$ 时，我们成为计算受限（现在是相对于芯片间网络），或者等效地当 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$ 或 $D > 8755$ 时。请注意，与之前不同，关键阈值现在取决于 $D$ 而不是 $B$！试着想想为什么会这样。这只是这样一个例子，但我们强调这种屋顶线对于知道何时我们可以跨多个 TPU 并行化操作至关重要。

## 一些要解决的问题

**问题 1 [int8 矩阵乘法]：** 假设我们想要以 int8 精度（每个参数 1 字节）而不是 bfloat16 来进行矩阵乘法 $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$。<d-footnote>在这里和整本书中，我们将使用符号 $A \cdot_D B$ 来表示乘法正在对 D 维度进行收缩。这是对 einsum 符号的滥用。</d-footnote>

1. 需要从内存加载多少字节？需要写回多少字节到内存？
2. 执行了多少总 OPs？
3. 算术强度是多少？
4. $T_\text{math}$ 和 $T_\text{comms}$ 的屋顶线估计是什么？整个操作运行时间的合理上下界是什么？

假设我们的 HBM 带宽是 `8.1e11` 字节/秒，我们的 int8 峰值 OPs/s 是 `3.94e14`（大约是 bfloat16 的 2 倍）。

{% details 点击这里查看答案。%}

1. 因为我们以 int8 存储参数，每个参数 1 字节，所以我们有 $$BD + DF$$ 字节从 HBM 加载，$$BF$$ 字节写回。
2. 这与 bfloat16 相同，但理论上 int8 OPs/s 应该更快。所以这仍然是 $2BDF$ 次 FLOPs。
3. 算术强度是 $$2BDF / (BD + DF + BF)$$。如果我们对 $$B \ll D$$ 和 $$B \ll F$$ 做出与上面相同的假设，我们得到算术强度为 $$2B$$，这意味着我们的规则变为 $B > \text{HBM int8 算术强度} / 2$。使用给定的数字，这个 int8 强度是 `3.94e14 / 8.1e11 = 486`，所以规则是 $B > 486 / 2 = 243$。请注意，这基本上没有变化！
4. $$T_\text{math} = 2BDF / 3.94e14$$ 和 $$T_\text{comms} = (BD + DF + BF) / 8.1e11$$，所以合理的下界是 $$\max(T_\text{math}, T_\text{comms})$$，上界是 $$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题 2 [int8 + bf16 矩阵乘法]：** 在实践中，我们经常进行不同的权重与激活量化，所以我们可能以非常低的精度存储权重，但保持激活（和计算）在更高的精度。假设我们想要将权重量化为 int8，但保持激活（和计算）在 bfloat16。在什么批量大小下我们成为计算受限？假设 `1.97e14` bfloat16 FLOPs/s。

*提示：这具体意味着 `bfloat16[B, D] * int8[D, F] -> bfloat16[B, F]`，其中 $B$ 是"批量大小"。*

{% details 点击这里查看答案。%}

再次假设 B 很小，我们有 2BDF bfloat16 FLOPs，但只有 DF 权重（而不是 bfloat16 中的 2DF）。这意味着当 $$2B > 240$$ 或 $$B > 120$$ 时，我们成为计算受限。这要低得多，意味着如果我们能够进行 int8 权重量化（这相当容易做到）但仍然进行 bfloat16 FLOPs，我们在效率上会获得有意义的提升（尽管 int8 OPs 会更好）。

{% enddetails %}

**问题 3：** 使用问题 2 的设置，为 $F = D = 4096$ 和 $F = D = 1024$ 制作峰值 FLOPs 与 $B$ 的屋顶线图。*使用加载字节的精确数量，而不是近似值。*

{% details 点击这里查看答案。%}

这是所讨论的图：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

请注意，两个模型最终都达到了峰值硬件 FLOPs/s，但较大的 D/F 更早达到。D=F=1024 几乎使关键批量大小翻倍。生成此图的代码在这里：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('批量大小')
plt.ylabel('TPU v5e 上的峰值 bfloat16 FLOPs/s')
plt.grid()
```

{% enddetails %}

**问题 4：** 如果我们想要执行 $\text{int8[B, D]} *_D \text{int8[B, D, F]} \rightarrow \text{int8[B, F]}$，我们假设每个批量元素有不同的矩阵。这个操作的算术强度是多少？

{% details 点击这里查看答案。%}

让我们先看看总 FLOPs 和通信量。

1. 总 FLOPs：FLOPs 基本相同，因为我们执行相同数量的 $$BD \times DF$$ 矩阵乘法（这在第 4 节中有更多讨论）。所以这只是 $$2BDF$$。
2. 总通信量：我们在这里有更多的通信：$$BD + BDF + BF$$。
3. 因此，我们的算术强度实际上是 $$2BDF / (BD + BDF + BF)$$。由于 $$BDF$$ 主导分母，这大约是 $$2$$。所以不是依赖于批量大小，这基本上是常数。这是不好的，因为这意味着无论什么情况我们基本上都会受到通信限制。

{% enddetails %}

**问题 5 [GPU 的内存屋顶线]：** 使用 [NVIDIA 为 H100 提供的规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算矩阵乘法在什么批量大小下成为计算受限。*注意 Tensor Core FLOPs 数字是真实值的两倍，因为它们只能通过结构化稀疏性实现。*

{% details 点击这里查看答案。%}

从规格表中，我们看到报告的 bfloat16 FLOPs 值是 `1.979e15` FLOPs/s，带有星号注明"具有稀疏性"。没有稀疏性的真实值是这个的一半，意味着接近 `1e15` FLOPs/s。内存带宽是 3.35TB/s，或 `3.35e12` 字节/秒。因此 $B_\text{crit}$ 是 `1e15 / 3.35e12 = 298`，与 TPU 相当相似。

{% enddetails %}

<h3 markdown=1 class="next-section">第 1 部分就到这里了！对于第 2 部分，了解真实 TPU 如何处理 FLOPs 和通信，[请点击这里](../tpus)。</h3>