---
layout: distill
title: "关于屋顶线模型"
# permalink: /main/
description: "当我们在硬件上运行算法时, 我们受到三件事的限制: 我们的计算机进行数学运算的速度 (操作/秒), 用于移动数据的可用带宽 (字节/秒), 以及用于存储数据的总可用内存 (字节). 这些“屋顶线”约束使我们能够对给定计算的时间进行上限和下限的估算."
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

  - name: "时间都去哪儿了?"
  - subsections:
    - name: "可视化屋顶线"
    - name: "矩阵乘法"
    - name: "网络通信屋顶线"
  - name: "一些待解决的问题"

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

## 时间都去哪儿了?

让我们从一个极其简单的问题开始: *为什么一个算法需要 50 毫秒而不是 50 秒或 5 毫秒*? 模型内部到底发生了什么耗费了大量时间, 我们应该期望它花费多长时间?

**计算:** 深度学习模型实际上是一堆矩阵乘法, 每个都由浮点乘法和加法“操作” (FLOPs) 组成. 我们的加速器速度决定了这些计算需要多长时间:

$$\begin{equation}
  T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/秒}}
\end{equation}$$

例如, 一块 NVIDIA H100 每秒可以执行约 9.89e14 次 bfloat16<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写, 这是一种常用于机器学习的 16 位浮点格式.</d-footnote> FLOPs, 而一个 TPU v6e 每秒可以执行 9.1e14 次 FLOPs.<d-footnote>H100s 和 B200s 通常只能达到声称峰值 FLOPs 的 80-85% 左右, 而 TPU 在正常使用中可以接近 95%.</d-footnote> 这意味着在 H100 上执行 1e12 次 FLOPs 大约需要 `1e12 / 9.89e14 = 1.01ms`, 在 TPU v6e 上需要 `1e12 / 9.1e14 = 1.1ms`.<d-footnote>请注意, 这些芯片的定价不同, 此比较未按成本进行标准化.</d-footnote>

**芯片内通信:** *在加速器内部*, 张量需要在片上内存 (HBM) 和计算核心之间传输. 你会看到这个链接的带宽被称为“HBM 带宽”.<d-footnote>NVIDIA 也称之为“内存带宽”.</d-footnote> 在 H100 上, [这大约是 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/), 在 TPU v6e 上 [这大约是 1.6TB/s](https://cloud.google.com/tpu/docs/v6e).

**芯片间通信:** 当我们将模型*分布在多个加速器上*时, 张量经常需要在它们之间传输. 我们的硬件上通常有几种选择 (ICI, DCN 和 PCIe), 每种都有不同的带宽.

无论通信是在芯片内部还是芯片之间, 我们都以字节/秒为单位进行测量, 并用以下公式估算总通信时间:

$$\begin{equation}
  T_\text{comms} = \frac{\text{通信字节数}}{\text{网络/内存带宽 字节/秒}}
\end{equation}$$

通常 (但并非总是), 单个芯片内的计算可以与芯片内和芯片间的通信重叠. 这意味着**我们可以通过使用计算和通信时间的最大值来为训练和推理时间设定下限**. 我们也可以**用它们的和来设定上限**. 在实践中, 我们针对最大值进行优化, 因为代数更简单, 而且我们通常可以通过重叠通信和计算来接近这个界限. 如果我们以最大值为目标进行优化, 那么下限和上限最多相差 2 倍, 因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$. 然后, 我们通过对“重叠区域”和开销进行建模来提高精度, 这可以通过分析你的特定模型和目标系统来获得信息.

$$\begin{equation}
  T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
  T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果我们假设可以完美地重叠通信和计算, 当 $T_\text{math} > T_\text{comms}$ 时, 我们可以看到硬件的完全利用. 我们称之为“受计算限制”. 当 $T_\text{comms} > T_\text{math}$ 时, 我们往往是“受通信限制”, 至少有一部分加速器的 FLOPs/s 被浪费在等待数据传输上. 判断一个操作是受计算限制还是受通信限制的一种方法是看它的“*算术强度*”或“*操作强度*”.

**定义:** 算法的算术强度由其执行的总 FLOPs 与其需要通信的字节数之比给出 —— 无论是在芯片内部还是芯片之间.

$$\begin{equation}
  \text{算术强度} = \frac{\text{计算 FLOPs}}{\text{通信字节数}}
\end{equation}$$

算术强度衡量给定操作的“每字节 FLOPs”. 初步来看, 当我们的算术强度高时, $T_\text{math}$ 相对于 $T_\text{comms}$ 较大, 我们通常会使用大部分可用的 FLOPs. 当情况相反时, 我们在通信上花费更多时间并浪费 FLOPs. 这种交叉发生的点是我们硬件的“峰值算术强度”, 即峰值加速器 FLOPs/s 与加速器带宽之比.

$$\begin{align*} T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{计算 FLOPs}} {\text{加速器 FLOPs/s}} > \frac{\text{通信字节数}}{\text{带宽 字节/s}} & \\
\Leftrightarrow \frac{\text{计算 FLOPs}}{\text{通信字节数}} > \frac{\text{加速器 FLOPs/s}}{\text{带宽 字节/s}} & \\
\Leftrightarrow \text{强度}(\text{计算}) > \text{强度}(\text{加速器}) & \\
\end{align*}$$

数量 $\text{Intensity}(\text{Accelerator})$ 是我们的加速器达到其峰值 FLOPs/s 时的算术强度. **对于 TPU v5e MXU, 这大约是 240 FLOPs/字节**<d-footnote>MXU 是 TPU 上的矩阵乘法单元. 我们在这里指定这一点, 因为 TPU 还有其他加速器, 如 VPU, 负责具有不同峰值 FLOPs/s 的逐元素操作.</d-footnote>, 因为 TPU 可以执行 `1.97e14` FLOPs/s 并从 HBM 加载 `8.2e11` 字节/s. 这意味着如果一个算法的算术强度低于 240<d-footnote>这仅在算法从 HBM 加载其权重并在 MXU 中运行时才成立. 正如我们将在下一节中讨论的, 我们有时可以将参数存储在具有更高带宽的 VMEM 中. 许多算法也在 VPU 中运行, VPU 具有不同的性能特征.</d-footnote> FLOPs/字节, 它将受到字节加载的限制, 因此我们无法充分利用我们的硬件. 让我们看一个这样的例子:

**<span style="color:#7ab5ff">示例 (点积)</span>:** 为了计算两个 bfloat16 精度向量的点积, `x • y: bf16[N], bf16[N] → bf16[1]`, 我们需要从内存中加载 $x$ 和 $y$, 每个都有 $2 * N = 2N$ 字节, 执行 $N$ 次乘法和 $N-1$ 次加法, 并将 $2$ 字节写回 HBM
$$\begin{equation}
  \text{强度}(\text{点积}) = \frac{\text{总 FLOPs}}{\text{总字节数}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

当 $N\rightarrow\infty$ 时. 所以点积的算术强度为 $rac{1}{2}$, 或者换句话说, 点积每加载一个字节执行 0.5 次浮点运算. 这意味着我们的算术强度低于我们的硬件, 我们将受到通信限制.<d-footnote>上面 240 的数字在这里不是正确的比较, 因为正如你将在下一节中看到的, 点积是在 VPU 而不是 MXU 上执行的. TPU v5p VPU 每秒大约可以执行 7e12 次 FLOPs, 因此其临界强度约为 3, 这意味着我们在这里仍然在某种程度上受通信限制. 无论哪种方式, 我们的强度低且恒定的事实意味着在大多数硬件上很难受计算限制.</d-footnote>

### 可视化屋顶线

我们可以使用**屋顶线图**来可视化内存和计算之间的权衡, 该图绘制了算法在我们的硬件上可实现的峰值 FLOPs/s (吞吐量) (y 轴) 与该算法的算术强度 (x 轴) 的关系. 这是一个对数-对数图的例子:

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图:</b> 一个屋顶线图示例, 显示了两种具有不同算术强度的算法 (算法 1 和算法 2) 及其在不同带宽 (BW1 和 BW2) 下的相应理论峰值吞吐量. 在红色区域, 算法在两种带宽下都受到带宽限制, 并且浪费了硬件峰值 FLOPs/s 的一部分. 黄色区域仅在较低带宽 (BW1) 下受到带宽限制. 绿色区域在所有带宽下都受到计算限制. 在这里, 我们使用的是加速器的峰值 FLOPs/s, 增加带宽或提高强度没有任何好处." %}

上面, 随着强度的增加 (从左到右移动), 我们最初看到算法性能 (以 FLOPs/s 为单位) 的线性增长, 直到我们达到硬件的临界算术强度, 在 TPU v5e 的情况下为 240. 任何强度较低的算法都将受到带宽 (BW) 的限制, 并受限于峰值内存带宽 (以红色显示). 右侧的任何算法都将充分利用我们的 FLOPs (以绿色显示). 在这里, 算法 1 受通信限制, 仅使用总硬件 FLOPs/s 的一小部分. 算法 2 受计算限制. 我们通常可以通过增加其算术强度或增加可用内存带宽 (从 BW1 移动到 BW2) 来提高算法的性能.

### 矩阵乘法

让我们看看我们即将最喜欢的算法: 矩阵乘法 (又名 matmul). 我们写成 $X * Y \rightarrow Z$, 其中 $X$ 的形状为 $\text{bf16}[B, D]$, $Y$ 的形状为 $\text{bf16}[D, F]$, $Z$ 的形状为 $\text{bf16}[B, F]$. 为了进行矩阵乘法, 我们需要加载 $2DF + 2BD$ 字节, 执行 $2BDF$ FLOPs, 并写回 $2BF$ 字节.<d-footnote>技术上, 我们执行 $BF \times (2D - 1)$ FLOPs, 但这已经足够接近了. 这来自 $BDF$ 次乘法和 $BF * (D-1)$ 次加法. 第 4 节有更多细节.</d-footnote> <d-footnote>虽然 matmul 的输出技术上是 float32, 但我们通常在复制回 HBM 之前将其转换为 bfloat16.</d-footnote> 因此:

$$\begin{equation}
  \text{强度}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设我们的“批量大小” $B$ 相对于 $D$ 和 $F$ 较小, 我们可以得到一个很好的简化. 那么我们得到

$$\begin{equation}
  \frac{BDF}{BD + DF + BF} \approxeq \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
  \text{强度}(\text{matmul}) > \text{强度}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

对于 Transformer 矩阵乘法来说, 这是一个合理的假设, 因为我们通常有一个本地 (每个副本) 批量大小 $B < 1024$ 个 token (*不是序列*), 但 $D$ 和 $F > 8000$. 因此, 当我们的每个副本<d-footnote>我们说每个副本, 是因为如果我们进行某种模型分片以增加用于矩阵乘法的芯片数量, 我们会按相同的数量扩展我们可用的计算和内存带宽. 因此, 临界批量大小对于模型权重的每个独立副本都是如此.</d-footnote> 批量大小大于 240 个 token 时, 我们通常会变得受计算限制, 这是一个非常简单的规则!

<p markdown=1 class="takeaway">**要点:** 对于 bfloat16 矩阵乘法要在大多数 TPU 上受计算限制, 我们需要我们的每个副本 token 批量大小大于 240.<d-footnote>请注意, 这*不是*通常意义上的批量大小, 通常意义上的批量大小是指序列的批量大小. 事实证明, 大多数屋顶线纯粹取决于 token 的数量, 无论它们属于相同还是不同的序列. 例如, 如果你在 128 个 GPU 上有一个 512 个序列, 每个序列 4096 个 token 的批量大小, 那么你的总批量大小为 `512 * 4096 = 2M` 个 token, 本地批量大小为 16k 个 token.</d-footnote></p>

这带有一些我们将在下面的问题中探讨的值得注意的警告, 特别是关于量化 (例如, 如果我们量化我们的激活但仍然进行全精度 FLOPs), 但这是一个值得记住的好规则. 对于 GPU, 这个数字略高 (接近 300), 但同样的结论通常成立. 当我们[将一个大的矩阵乘法分解成更小的矩阵乘法](https://docs.jax.org/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时, 瓦片大小也很重要.<d-footnote>当我们进行大型矩阵乘法时, 我们需要将其分解成更小的瓦片, 以适应 VMEM/SMEM/TMEM, 即更高带宽的片上内存. 这导致我们多次加载块, 因此我们只加载 $O(N^2)$ 字节不再完全正确. 考虑一个 $(m, k) \cdot (k, n)$ 矩阵乘法, 瓦片大小为 $bm$, $bk$, $bm$. 令 $tm = m / bm$, 等等. 那么总 FLOPs 为 $2 \cdot tm \cdot tn \cdot tk \cdot m \cdot bk \cdot bm$, 总字节数为 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + 2 \cdot bm \cdot bn)$. 忽略最后一项, 我们的强度为 $bm \cdot bn / (bm + bn)$, 这与上面的类似.</d-footnote> 我们将在[下一节](../tpus)中讨论更底层的 GPU 和 TPU 细节.

### 网络通信屋顶线

到目前为止, 我们讨论的所有屋顶线都是内存带宽屋顶线, *全部在单个芯片内*. 这不应被视为规则. 事实上, 本书中我们关心的​​大多数屋顶线都涉及芯片之间的通信: 通常是涉及跨多个 TPU 分片的矩阵的矩阵乘法.

举一个有点刻意的例子, 假设我们想将两个大矩阵 $X\sim \text{bfloat16}[B, D]$ 和 $Y \sim \text{bfloat16}[D, F]$ 相乘, 它们均匀地分布在 2 个 TPU/GPU 上 (沿着 $D$ 维度). 为了进行这个乘法 (正如我们将在[第 3 节](../sharding)中看到的), 我们可以在每个 TPU 上乘以每个矩阵的一半 (`A = X[:, :D // 2] @ Y[:D // 2, :]` 在 TPU 0 上, `B = X[:, D // 2:] @ Y[D // 2:, :]` 在 TPU 1 上), 然后将得到的“部分和”复制到另一个 TPU 并将它们相加. 假设我们可以在每个方向上复制 `4.5e10` 字节, 并在每个芯片上执行 `1.97e14` FLOPs/s. $T_\text{math}$ 和 $T_\text{comms}$ 是多少?

$T_\text{math}$ 显然是之前的一半, 因为每个 TPU 都在做一半的工作, 即<d-footnote>我们忽略了将两个部分和相加所需的 FLOPs (另外 DF 次加法), 但这基本上可以忽略不计.</d-footnote>

$$\begin{equation}
  T_\text{math} = \frac{2BDF}{2 \cdot \text{加速器 FLOPs/s}} = \frac{BDF}{1.97e14}
\end{equation}$$

那么 $T_\text{comms}$ 呢? 这现在指的是芯片之间的通信时间! 这只是发送的总字节数除以网络带宽, 即

$$\begin{equation}
  T_\text{comms} = \frac{2BF}{\text{网络带宽}} = \frac{2BF}{4.5e10}
\end{equation}$$

因此, 当 $$\text{强度}(\text{matmul (2-chips)}) > \text{强度}(\text{TPU w.r.t. inter-chip network})$$ 或等效地当 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$ 或 $D > 8755$ 时, 我们变得受计算限制 (现在是相对于芯片间网络). 请注意, 与之前不同, 临界阈值现在取决于 $D$ 而不是 $B$! 试着想想为什么会这样. 这只是一个例子, 但我们强调这种屋顶线对于了解何时可以在多个 TPU 上并行化操作至关重要.

## 一些待解决的问题

**问题 1 [int8 matmul]:** 假设我们想用 int8 精度 (每个参数 1 字节) 而不是 bfloat16 来进行矩阵乘法 $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$.<d-footnote>在这里和全文中, 我们将使用符号 $A \cdot_D B$ 来表示乘法正在对 D 维度进行收缩. 这是对 einsum 符号的滥用.</d-footnote>

1.  需要从内存中加载多少字节? 需要写回内存多少字节?
2.  总共执行了多少次操作?
3.  算术强度是多少?
4.  $T_\text{math}$ 和 $T_\text{comms}$ 的屋顶线估计是什么? 整个操作运行时间的合理上限和下限是什么?

假设我们的 HBM 带宽为 `8.1e11` 字节/秒, 我们的 int8 峰值 OPs/s 为 `3.94e14` (大约是 bfloat16 的 2 倍).

{% details 点击这里查看答案. %}

1.  因为我们用 int8 存储参数, 每个参数有 1 个字节, 所以我们从 HBM 加载了 $$BD + DF$$ 字节, 并写回了 $$BF$$ 字节.
2.  这与 bfloat16 中的情况相同, 但理论上 int8 OPs/s 应该更快. 所以这仍然是 $2BDF$ FLOPs.
3.  算术强度是 $$2BDF / (BD + DF + BF)$$. 如果我们做出与上面相同的假设, 即 $$B \ll D$$ 和 $$B \ll F$$, 我们得到的算术强度为 $$2B$$, 这意味着我们的规则变成了 $B > \text{HBM int8 算术强度} / 2$. 使用给定的数字, 这个 int8 强度是 `3.94e14 / 8.1e11 = 486`, 所以规则是 $B > 486 / 2 = 243$. 请注意, 这基本上没有改变!
4.  $$T_\text{math} = 2BDF / 3.94e14$$ 和 $$T_\text{comms} = (BD + DF + BF) / 8.1e11$$, 所以一个合理的下限是 $$\max(T_\text{math}, T_\text{comms})$$, 一个上限是 $$T_\text{math} + T_\text{comms}$$.

{% enddetails %}

**问题 2 [int8 + bf16 matmul]:** 在实践中, 我们经常进行不同的权重与激活量化, 所以我们可能会用非常低的精度存储我们的权重, 但保持激活 (和计算) 在更高的精度. 假设我们想用 int8 量化我们的权重, 但保持激活 (和计算) 在 bfloat16. 在什么批量大小时我们会变得受计算限制? 假设 `1.97e14` bfloat16 FLOPs/s.

*提示: 这具体意味着 `bfloat16[B, D] * int8[D, F] -> bfloat16[B, F]`, 其中 $B$ 是“批量大小”.*

{% details 点击这里查看答案. %}

再次假设 B 很小, 我们有 2BDF bfloat16 FLOPs, 但只有 DF 权重 (而不是 bfloat16 中的 2DF). 这意味着当 $$2B > 240$$ 或 $$B > 120$$ 时, 我们变得受计算限制. 这个值要低得多, 这意味着如果我们能做 int8 权重量化 (这相当容易做到) 但仍然做 bfloat16 FLOPs, 我们在效率上会得到有意义的提升 (尽管 int8 OPs 会更好).

{% enddetails %}

**问题 3:** 采用问题 2 的设置, 为 $F = D = 4096$ 和 $F = D = 1024$ 制作一个峰值 FLOPs 与 $B$ 的屋顶线图. *使用加载的确切字节数, 而不是近似值.*

{% details 点击这里查看答案. %}

这是所讨论的图:

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

请注意, 两种模型最终都达到了峰值硬件 FLOPs/s, 但较大的 D/F 更早达到. D=F=1024 几乎使临界批量大小翻了一番. 生成此图的代码在这里:

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
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

{% enddetails %}

**问题 4:** 如果我们想执行 $\text{int8}[B, D] *_D \text{int8}[B, D, F] \rightarrow \text{int8}[B, F]$, 其中我们想象每个批处理元素都有一个不同的矩阵. 这个操作的算术强度是多少?

{% details 点击这里查看答案. %}

让我们从查看总 FLOPs 和通信开始.

1.  总 FLOPs: FLOPs 基本相同, 因为我们正在做相同数量的 $$BD \times DF$$ 矩阵乘法 (这在第 4 节中有更多讨论). 所以这只是 $$2BDF$$.
2.  总通信: 我们这里有更多的通信: $$BD + BDF + BF$$.
3.  因此, 我们的算术强度现在实际上是 $$2BDF / (BD + BDF + BF)$$. 由于 $$BDF$$ 在分母中占主导地位, 这大约是 $$2$$. 所以它不再依赖于批量大小, 而是基本上是恒定的. 这很糟糕, 因为这意味着我们基本上总是会受到通信限制, 无论如何.

{% enddetails %}

**问题 5 [GPU 的内存屋顶线]:** 使用 [NVIDIA 为 H100 提供的规格表](https://www.nvidia.com/en-us/data-center/h100/), 计算矩阵乘法将变得受计算限制的批量大小. *请注意, Tensor Core FLOPs 的数字是真实值的两倍, 因为它们只能通过结构化稀疏性实现.*

{% details 点击这里查看答案. %}

从规格表中, 我们看到报告的 bfloat16 FLOPs 值为 `1.979e15` FLOPs/s, 并带有一个星号, 注明“带稀疏性”. 真实值是这个值的一半, 没有稀疏性, 意味着接近 `1e15` FLOPs/s. 内存带宽为 3.35TB/s, 或 `3.35e12` 字节/秒. 因此 $B_\text{crit}$ 是 `1e15 / 3.35e12 = 298`, 与 TPU 相当相似.

{% enddetails %}

<h3 markdown=1 class="next-section">第一部分到此结束! 第二部分, 看看真实的 TPU 如何处理 FLOPs 和通信, [点击这里](../tpus).</h3>
