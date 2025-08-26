---
layout: distill
title: "分片矩阵以及如何乘以它们"
# permalink: /main/
description: "当我们训练大型机器学习模型时, 我们必须将其参数或输入拆分 (或“分片”) 到许多加速器上. 由于 LLM 主要由矩阵乘法组成, 理解这一点归结为理解当矩阵分布在设备上时如何进行乘法. 我们基于 TPU 通信原语的成本, 发展了一个简单的分片矩阵乘法理论."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus"
previous_section_name: "Part 2: TPUs"

next_section_url: ../transformers
next_section_name: "Part 4: Transformer Math"

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
  - name: "分区符号和集合操作"
  - subsections:
    - name: "分片的统一符号"
    - name: "我们如何在代码中描述它?"
  - name: "使用分片数组进行计算"
  - subsections:
    - name: "情况 1: 两个乘数都没有分片的收缩维度"
    - name: "情况 2: 一个乘数有分片的收缩维度"
    - name: "情况 3: 两个乘数都有分片的收缩维度"
    - name: "情况 4: 两个乘数都有一个非收缩维度沿同一轴分片"
  - name: "深入了解 TPU 通信原语"
  - subsections:
    - name: "我们最后的通信原语: AllToAll"
    - name: "关于 ReduceScatter 的更多信息"
  - name: "我们学到了什么?"
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

## 分区符号和集合操作

当我们在成千上万的 TPU 或 GPU 上训练一个 LLM 时, 我们抽象地做的计算与在单个设备上训练时是相同的. 不同之处在于**我们的数组无法容纳在单个 TPU/GPU 的 HBM 中**, 所以我们必须将它们拆分.<d-footnote>值得注意的是, 我们也可能为了速度而选择并行化. 即使我们可以容纳在较少数量的芯片上, 扩展到更多芯片只是给了我们更多的 FLOPs/s. 例如, 在推理期间, 我们有时可以容纳在较小的拓扑上, 但选择扩展到更大的拓扑以减少延迟. 同样, 在训练期间, 我们经常扩展到更多芯片以减少步骤时间.</d-footnote> 我们称之为“*分片*”或“*分区*”我们的数组. 扩展的艺术在于弄清楚如何对我们的模型进行分片, 以便计算保持高效.

这是一个在 4 个 TPU 上分片的 2D 数组 **A** 的例子:

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图:</b> 一个形状为 <b>A</b>[I, J] 的示例数组在 4 个设备上进行分片. 两个维度都在 2 个设备上均匀分片, 分片方式为 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]. 每个 TPU 持有总内存的 1/4." %}

请注意, 分片数组仍然具有与未分片数组相同的*全局*或*逻辑形状*, 例如 `(4, 128)`, 但它也有一个*设备本地形状*, 例如 `(2, 64)`, 这给了我们每个 TPU 持有的实际字节大小 (在上图中, 每个 TPU 持有总数组的 ¼). 现在我们将 इसको推广到任意数组.

### 分片的统一符号

我们使用*命名轴表示法*的一种变体来描述张量如何以块的形式在设备上分片: 我们假设存在一个 2D 或 3D 的设备网格, 称为**设备网格**, 其中每个轴都被赋予了**网格轴名称**, **例如 X**, **Y 和 Z.** 然后, 我们可以通过描述数组的每个命名维度如何跨物理网格轴进行分区来指定矩阵数据在设备网格上的布局. 我们称这个分配为**分片**.

**示例 (上图)**: 对于上图, 我们有:
*   **网格:** 上面的设备网格 `Mesh(devices=((0, 1), (2, 3)), axis_names=(‘X', ‘Y'))`, 这告诉我们我们有 4 个 TPU, 排列成一个 2x2 的网格, 轴名称为 $X$ 和 $Y$.
*   **分片:** $A[I_X, J_Y]$, 这告诉我们将第一个轴 $I$ 沿网格轴 $X$ 分片, 将第二个轴 $J$ 沿网格轴 $Y$ 分片. 这个分片告诉我们每个分片持有数组的 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$.

综上所述, 我们知道数组的本地形状 (单个设备持有的分片的大小) 是 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$, 其中 $$\lvert I\rvert$$ 是 A 的第一个维度的大小, $$\lvert J\rvert$$ 是 A 的第二个维度的大小.

<b markdown=1 style="color: #048affff;">小测验 [沿 1 个轴的 2D 分片]:</b> 考虑一个数组 `fp32[1024, 4096]`, 分片为 $A[I_{XY}, J]$, 网格为 `{'X': 8, 'Y': 2}`. 每个设备持有多少数据? 在 H100s 上从 HBM 加载这个数组需要多长时间 (假设每个芯片的内存带宽为 `3.4e12`)?

{% details 点击这里查看答案. %}

$A[I_{XY}, J]$ 将第一个维度 (I) 沿 X 和 Y 硬件轴进行分片. 在这个例子中, 本地形状是 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$. 对于给定的例子, 全局形状是 `fp32[1024, 4096]`, 所以本地形状是 `fp32[64, 4096]`.

由于每个 GPU 有 `4 * 64 * 4096 = 1MiB` 字节, 这大约需要 `1e6 / 3.4e12 = 294ns`, 尽管由于各种开销, 实际时间可能要长得多, 因为这个数组很小.

{% enddetails %}

**可视化这些分片:** 让我们尝试通过查看一个分布在 4 个设备上的 2D 数据数组来可视化这些分片:

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

我们将矩阵的*完全复制*形式简单地写为 $A[I, J]$, 没有分片分配. 这意味着*每个*设备都包含整个矩阵的完整副本.

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

我们可以用一个下标网格轴来表示其中一个维度已经跨一个网格轴进行了分区. 例如, $A[I_X, J]$ 意味着 **I** 逻辑轴已经跨 **X** 网格维度进行了分区, 但 **J** 维度*没有*分区, 并且这些块在 **Y** 网格轴上保持*部分复制*.

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$ 意味着 **I** 逻辑轴已经跨 **X** 网格轴进行了分区, 并且 **J** 维度已经跨 **Y** 网格轴进行了分区.

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

我们在下图中说明了其他可能性:

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

这里 $A[I_{XY}, J]$ 意味着我们将 **X** 和 **Y** 网格轴视为一个更大的扁平化维度, 并将 **I** 命名轴跨所有设备进行分区. 多个网格轴下标的顺序很重要, 因为它指定了分区跨网格的遍历顺序.

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后, 请注意, 我们*不能*将多个命名轴沿*相同*的网格维度进行分片. 例如, $A[I_X, J_X]$ 是一个无意义的, 禁止的分片. 一旦一个网格维度被用于分片数组的一个维度, 它在某种意义上就被“用掉了”.

<b markdown=1 style="color: #57cf57;">小测验:</b> 设 **A** 是一个形状为 `int8[128, 2048]` 的数组, 分片为 $A[I_{XY}, J]$, 网格为 `Mesh({‘X': 2, ‘Y': 8, ‘Z': 2})` (总共 32 个设备). **A** 每个设备使用多少内存? **A** 在所有设备上总共使用多少内存?

{% details 点击这里查看答案. %}

**答案:** 我们的数组 **A** 在 X 和 Y 上分片, 在 Z 上复制, 因此每个设备的形状为 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`, 大小为 `8 * 2048 = 16,384` 字节. 因为它在 Z 上复制, 而在 Z 平面内它在 X 和 Y 上完全分片, 所以每个 Z 平面有一个副本, 并且有 2 个这样的平面, 所以总大小 (在所有设备上) 是 `128 * 2048 * 2 = 512 KiB`.

{% enddetails %}

### 我们如何在代码中描述它?

到目前为止, 我们一直避免谈论代码, 但现在是时候先睹为快了. JAX 使用一种与我们上面描述的抽象语法非常匹配的命名分片语法. 我们将在[第 10 节](../jax-stuff)中更多地讨论这一点, 但这里有一个快速预览. 你可以在 Google Colab [这里](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing)中玩这个, 并分析结果以查看 JAX 如何处理不同的分片. 这个代码片段做了 3 件事:

1.  创建一个 **jax.Mesh**, 将我们的 8 个 TPU 映射到一个 4x2 的网格中, 并将名称‘X'和‘Y'分配给两个轴.
2.  创建矩阵 A 和 B, 其中 A 在其两个维度上都进行了分片, B 在输出维度上进行了分片.
3.  编译并执行一个简单的矩阵乘法, 返回一个分片数组.

```py
import jax
import jax.numpy as jnp

# 创建我们的网格! 我们正在一个 TPU v2-8 4x2 切片上运行, 名称为 'X' 和 'Y'.
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 一个帮助定义我们分片的小工具函数. PartitionSpec 是我们的
# 分片 (从轴到名称的映射).
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# 我们将 A 和 B 都在非收缩维度上进行分片, 并将 A 在收缩维度上进行分片.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# 我们可以对这些分片数组进行矩阵乘法! out_shardings 告诉我们我们希望
# 输出如何分片. JAX/XLA 为我们处理其余的分片.
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX 的酷之处在于这些数组的行为就像它们没有被分片一样! `B.shape` 会告诉我们全局或逻辑形状 (2048, 8192). 我们必须实际查看 `B.addressable_shards` 才能看到它是如何本地分片的. 我们可以对这些数组执行操作, JAX 会尝试找出如何广播或重塑它们以执行操作. 例如, 在上面的例子中, **A** 的本地形状是 `[2, 1024]`, **B** 的本地形状是 `[2048, 4096]`. JAX/XLA 会自动在这些数组之间添加必要的通信以执行最终的乘法.

## 使用分片数组进行计算

如果你有一个分布在许多设备上的数据数组, 并希望对其执行数学运算, 那么对数据和计算进行分片会带来哪些开销?

显然, 这取决于所涉及的计算.

*   对于*逐元素*操作, 对分布式数组进行操作**没有开销**.
*   当我们希望对驻留在许多设备上的元素执行操作时, 事情就变得复杂了. 值得庆幸的是, 对于大多数机器学习来说, 几乎所有的计算都以矩阵乘法的形式进行, 而且它们相对容易分析.

本节的其余部分将讨论如何乘以分片矩阵. 初步来看, 这涉及到移动矩阵的块, 以便你可以完全乘以或求和每个块. **每个分片都会涉及不同的通信.** 例如, $A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ 可以在没有任何通信的情况下进行乘法, 因为*收缩维度* (J, 我们实际求和的维度) 是未分片的. 然而, 如果我们希望输出是未分片的 (即 $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$), 我们就需要将 $A$ 或 $C$ 复制到每个设备 (使用 *AllGather*). 这两种选择有不同的通信成本, 所以我们需要计算这个成本并选择最低的一个.

{% details 你可以从“块矩阵乘法”的角度来思考这个问题. %}

为了理解这一点, 回忆一下“块矩阵”的概念可能会有所帮助, 即一个嵌套的矩阵的矩阵:

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

矩阵乘法有一个很好的性质, 即当矩阵乘数用块来表示时, 乘积可以用块矩阵乘法来表示, 遵循标准规则:

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

这意味着实现分布式矩阵乘法归结为在网络上移动这些分片块, 对这些块执行*本地*矩阵乘法, 并对它们的结果求和. **那么问题是添加什么通信, 以及它的成本是多少.**

{% enddetails %}

方便的是, 我们可以将所有可能的分片归结为大约 4 种需要考虑的情况, 每种情况都有一个规则, 说明我们需要添加什么通信
1.  **[情况 1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension):** 两个输入都没有在收缩维度上进行分片. _我们可以对本地分片进行乘法, 无需任何通信._
2.  **[情况 2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension):** 一个输入在收缩维度上进行了分片. _我们通常对收缩维度上的分片输入进行“AllGather”._
3.  **[情况 3](#case-3-both-multiplicands-have-sharded-contracting-dimensions):** 两个输入都在收缩维度上进行了分片. _我们可以对本地分片进行乘法, 然后对结果进行“AllReduce”._
4.  **[情况 4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis):** 两个输入都在同一轴上对非收缩维度进行了分片. 我们无法在不先对其中一个输入进行 AllGather 的情况下继续.

你可以将这些视为只需要遵循的规则, 但理解这些规则为什么成立以及它们的成本是多少也很有价值. 我们现在将详细介绍每一个.

### 情况 1: 两个乘数都没有分片的收缩维度

**引理:** 当乘以分片矩阵时, 计算是有效的, 并且输出遵循输入的分片, *除非*收缩维度被分片或两个矩阵都沿同一轴分片. 例如, 这可以正常工作

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

没有任何通信, 并且得到一个跨 X 和 Y 硬件维度分片的张量. 试着想想为什么会这样. 基本上, 计算与分片*无关*, 因为每个批处理条目都有一些本地的收缩轴块, 它可以乘以和归约. 任何这些情况都可以正常工作, 并遵循这个规则:

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

因为 **A** 和 **B** 都没有分片的收缩维度 **J**, 我们可以简单地执行输入的本地块矩阵乘法, 结果将*已经*根据所需的输出分片进行了分片. 当两个乘数都有沿同一轴分片的非收缩维度时, 这不再成立 (有关详细信息, 请参见[无效分片](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)部分).

### 情况 2: 一个乘数有分片的收缩维度

让我们考虑当一个输入 **A** 沿收缩 **J** 维度分片, 而 **B** 完全复制时该怎么做:

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能简单地将 **A** 和 **B** 的本地块相乘, 因为我们需要对 **A** 的完整收缩维度求和, 该维度分布在 X 轴上. 通常, 我们首先“**AllGather**” **A** 的分片, 以便每个设备都有一个完整的副本, 然后才与 **B** 相乘:

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

这样, 实际的乘法就可以在每个设备上完全完成.

<p markdown=1 class="takeaway">**要点:** 当乘以其中一个矩阵沿收缩维度分片的矩阵时, 我们通常先对其进行 AllGather, 以便收缩不再分片, 然后进行本地矩阵乘法.</p>

请注意, 当 **B** 也没有沿 X 分片时, 我们也可以进行本地部分矩阵乘法, 然后对分片的部分和求和 (或 *AllReduce*), 在某些情况下这可能更快. 请参见问题 4 [下面](#some-problems-to-work).

**什么是 AllGather?** AllGather 是我们将要讨论的第一个核心 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 通信原语. AllGather *移除*沿一个轴的分片, 并将分布在设备上的分片重新组装到该轴上的*每个*设备上. 使用上面的符号, AllGather 从一组轴中移除一个下标, 例如

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

我们不必移除给定维度的所有下标, 例如 $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$ 也是一个 AllGather, 只是只在一个轴上进行. 另请注意, 我们也可能希望使用 AllGather 来移除*非收缩*维度分片, 例如在矩阵乘法中:

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们可以先对 **A** 进行 AllGather 以移除输入分片, 或者我们可以进行分片矩阵乘法, 然后对结果 **C** 进行 AllGather.

**AllGather 实际上是如何执行的?** 为了在一个 TPU 轴 (一个环) 周围执行一维 AllGather, 我们基本上让每个 TPU 将其分片在一个环周围传递, 直到每个设备都有一个副本.<d-footnote>GPU AllGather 也可以这样工作, 你可以从节点中的 GPU 创建一个环, 并按 (任意) 顺序传递块.</d-footnote> 这是一个动画:

{% include figure.liquid path="assets/img/all-gather.gif" caption="<b>图:</b> 一个动画, 显示了如何在一组 8 个 TPU 或 GPU 设备周围执行 AllGather. 每个设备开始时拥有数组的 1/8, 结束时拥有一个完整的副本." %}

我们可以单向或双向进行 AllGather (上图显示了双向). 如果我们单向进行, 每个 TPU 会在环周围发送大小为 $\text{bytes} / N$ 的块, 共 $N - 1$ 次跳跃. 如果我们双向进行, 我们有 $\lceil \frac{N}{2} \rceil$ 次跳跃, 大小为 $2 \cdot \text{bytes} / N$.

**这需要多长时间?** 让我们以双向 AllGather 为例, 计算它需要多长时间. 设 $$V$$ 为数组中的字节数, $X$ 为收缩维度上的分片数. 那么从上图中, 每个跳跃在每个方向上发送 $V / \lvert X\rvert$ 字节, 所以每个跳跃需要

$$T_{hop} = \frac{2 \cdot V}{X \cdot W_\text{ici}}$$

其中 $W_\text{ici}$ 是**双向** ICI 带宽.<d-footnote>分子中的 2 来自于我们使用的是双向带宽. 我们在每个方向上发送 $V / X$, 总共 $2V / X$.</d-footnote> 我们需要发送总共 $\lvert X\rvert / 2$ 次跳跃才能到达每个 TPU<d-footnote>技术上是 $\lceil X / 2 \rceil$</d-footnote>, 所以总的归约需要

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

请注意, 这**不依赖于 $X$!** 这有点惊人, 因为这意味着即使我们的 TPU 只是本地连接的, 连接的局部性也不重要. 我们只是受每个链接速度的瓶颈.

<p markdown=1 class="takeaway">**要点:** 当在受吞吐量限制的情况下执行 AllGather (或 ReduceScatter 或 AllReduce) 时, 实际的通信时间仅取决于数组的大小和可用带宽, 而不取决于我们的数组分片的设备数量!</p>

**关于 ICI 延迟的说明:** 无论数据量大小, 每个 ICI 链接上的每次跳跃都有一些固有的开销. 这通常在 1us 左右. 这意味着当我们的数组 $$A$$ 非常小并且每次跳跃的时间少于 1us 时, 我们可以进入一个“受延迟限制”的状态, 此时计算*确实*依赖于 $X$.

{% details 有关完整详细信息, 请单击此处. %}

设 $$T_\text{min}$$ 为单次跳跃的最小时间. 那么

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

因为我们执行 $X / 2$ 次跳跃. 对于大型归约或收集, 我们完全受带宽限制. 我们发送的数据量如此之大, 以至于每次跳跃的开销基本上可以忽略不计. 但对于小数组 (例如, 从模型中采样时), 这不可忽略, ICI 带宽也不相关. 我们完全受延迟限制. 另一种说法是, 给定一个特定的 TPU, 例如具有 `4.5e10` 单向 ICI 带宽的 TPU v5e, 发送任何小于 `4.5e10 * 1e-6 = 45kB` 的缓冲区都将受延迟限制.

{% enddetails %}

这是一个在 TPU v5e 8x16 切片上 AllGather 带宽的实证测量. 数组在 16 轴上分片, 因此它有一个完整的双向环.

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>图:</b> 在 AllGather 期间, TPU v5e 的实证带宽和估计链接带宽. 橙色的 BW 是每秒实际 AllGather 的字节数, 而蓝色曲线显示了根据集合的已知成本计算的实证单向链接带宽." %}

请注意, 我们只达到了声称峰值带宽 (`4.5e10`) 的约 95%, 并且我们在约 10MB 时达到了这个峰值, 当 16 路分片时, 每个设备约 500kB (*旁注: 这比 GPU 好得多*).

**当我们跨多个轴进行 AllGather 时会发生什么?** 当我们跨多个轴进行收集时, 我们有多个 ICI 维度可以进行收集. 例如, AllGather<sub>XY</sub>([B, D<sub>XY</sub>]) 在两个硬件网格轴上操作. 这将可用带宽增加了 $N_\text{axes}$ 倍.

{% details 有关完整详细信息, 请单击此处. %}

一般来说, 我们有

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]$$

其中 $$\sum_i \lvert X_i \rvert / 2$$ 是 TPU 网格中最长路径的长度.

{% enddetails %}

<b markdown=1 style="color:rgb(144, 92, 255);">小测验 2 [AllGather 时间]:</b> 使用[第 2 部分](../tpus)中的数字, 在具有 2D 网格 `{'X': 8, 'Y': 4}` 的 TPUv5e 上执行 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F] 需要多长时间, 其中 $$E = 2048$$, $$F = 8192$$ (bfloat16)? 如果 $$E=256, F=256$$ 呢?

{% details 点击这里查看答案. %}

**答案:** 让我们从计算一些基本量开始:

1) TPU v5e 的每个轴都有 4.5e10 字节/秒的单向 ICI 带宽.
2) 在 bfloat16 中, 对于 (a), 我们有 $A[E_Y, F]$, 所以每个设备持有一个形状为 bfloat16[512, 8192] 的数组, 大小为 512 * 8192 * 2 = 8.4MB. 总数组大小为 2048 * 8192 * 2 = 34MB.

*对于第 (1) 部分*, 我们可以使用上面的公式. 由于我们是在一个轴上执行 AllGather, 我们有 $T_{\text{comms}} = \text{34e6} / \text{9e10} = \text{377us}$. 为了检查我们是否不受延迟限制, 我们知道在一个大小为 4 的轴上, 我们最多有 3 次跳跃, 所以我们的延迟限制大约是 3us, 所以我们离得不近. 然而, TPU v5e 只有在一个轴的大小为 16 时才有环绕连接, 所以在这里*我们实际上无法进行完全双向的 AllGather*. 我们需要 3 次跳跃才能让数据从边缘到达另一边, 所以理论上我们更像是 $T_{\text{comms}} = 3 * \text{8.4e6} / \text{4.5e10} = 560\mu s$. [**这里**](https://imgur.com/a/RkvpRGQ) 是来自[这个 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing) 的**实际配置文件**, 显示为 $680 \mu s$, 这是合理的, 因为我们可能没有获得 100% 的理论带宽! *对于第 (2) 部分*, 每个分片的大小为 `64 * 256 * 2 = 32kB. 32e3 / 4.5e10 = 0.7us`, 所以我们受延迟限制. 由于我们有 3 次跳跃, 这大约需要 3 * 1us = 3us. [在实践中, 它更接近 8us.](https://imgur.com/a/HZLQmYs)

{% enddetails %}

### 情况 3: 两个乘数都有分片的收缩维度

第三种基本情况是当两个乘数都在它们的收缩维度上分片, 并且沿同一网格轴:

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

在这种情况下, *本地*分片块矩阵乘法至少是*可能*执行的, 因为它们将共享相同的收缩索引集. 但是每个乘积只代表最终期望乘积的*部分和*, 并且沿 **X** 维度的每个设备将剩下这个最终期望乘积的不同*部分和*. 这种情况非常普遍, 以至于我们扩展了我们的符号来明确标记这种情况:

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

符号 **{ U<sub>X</sub> }** 读作“**沿 X 网格轴未归约**”, 指的是该操作在某种意义上是“未完成”的状态, 因为它只有在最终求和后才算完成. $\cdot_\text{LOCAL}$ 语法意味着我们执行本地求和, 但将结果保持未归约状态.

这可以看作是关于矩阵乘法和外积的以下结果:

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

其中 ⊗ 是外积. 因此, 如果轴 **X** 上的 TPU **i** 具有 **A** 的第 **i** 列和 **B** 的第 **i** 行, 我们可以进行本地矩阵乘法以获得 $$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$. 这个矩阵的每个条目都包含 **A • B** 在该条目处的和的第 **i** 项. 我们仍然需要对 **P** 进行求和, 我们在网格轴 **X** 上对其进行了分片, 以获得完整的 **A • B**. 如果我们按块 (即分片) 写出 **A** 和 **B**, 然后对结果的每个分片求和, 这种方式同样有效.

我们可以使用跨 **X** 轴的完整 **AllReduce** 来解决这个问题:

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce 移除部分和, 导致沿该轴的*每个*设备都具有相同的完全求和的值. AllReduce 是我们将在本节中讨论的几个关键通信中的第二个, 第一个是 AllGather, 其他是 ReduceScatter 和 AllToAll. AllReduce 接受一个具有未归约 (部分求和) 轴的数组, 并通过在该未归约轴周围传递这些分片并累积结果来执行求和. 签名为

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

这意味着它只是移除了 $\\{U_Y\\}$ 后缀, 但在其他方面保持结果不变.

**AllReduce 的成本是多少?** 一个关于 AllReduce 如何执行的心智模型是, 每个设备都将其分片发送给其邻居, 并将接收到的所有分片相加. 显然, 这比 AllGather 更昂贵, 因为每个“分片”都与完整数组具有相同的形状. 通常, **AllReduce 的成本是 AllGather 的两倍.** 一种看待这个问题的方式是注意到 **AllReduce** 可以表示为另外两个原语的组合: 一个 **ReduceScatter** 和一个 **AllGather**. 与 AllReduce 一样, ReduceScatter 解析数组上的部分和, 但结果是沿给定维度“分散”或分区的输出. AllGather 收集所有这些片段, 并沿该物理轴“取消分区/取消分片/复制”逻辑轴.

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**那么 ReduceScatter 呢?** 正如 AllReduce 移除一个下标 ($F_Y \to F$ 上面), ReduceScatter 对一个未归约/部分求和的数组求和, 然后将另一个逻辑轴沿同一网格轴分散 (分片). $[F]\\{U_Y\\} \to [F_Y]$. 动画显示了这是如何完成的: 请注意, 它与 AllGather 非常相似, 但我们不是保留每个分片, 而是将它们相加. 因此, 它的延迟大致相同, 不包括执行归约所需的时间.

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

每个跳跃的通信时间就是每个分片的字节数 $V / Y$ 除以带宽 $W_\text{ici}$, 就像 AllGather 一样, 所以我们有

$$T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ici}}$$

其中 $$W_\text{ici}$$ 是双向带宽, 只要我们有一个完整的环可以进行归约.

### 情况 4: 两个乘数都有一个非收缩维度沿同一轴分片

每个网格维度在对张量进行分片时最多只能出现一次. 执行上述规则有时会导致违反此规则的情况, 例如:

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

这是无效的, 因为沿维度 **X** 的给定分片, 比如说 **i**, 将具有 **C** 的 **(i, i)** 分片, 即一个对角线条目. 那么, 在所有分片中没有足够的信息来恢复除对角线条目之外的任何内容, 所以我们不能允许这种分片.

解决这个问题的方法是对某些维度进行 AllGather. 在这里我们有两个选择:

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

或

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

在任何一种情况下, 结果在其形状中只会提到 **X** 一次. 我们选择哪一个将取决于后续操作需要什么样的分片.

## 深入了解 TPU 通信原语

前面的 4 种情况介绍了几种用于执行分片矩阵乘法的“核心通信原语”:

1.  **AllGather:** 从分片中移除一个下标, 收集分片.
2.  **ReduceScatter:** 通过在该轴上对分片求和来移除数组的“未归约”后缀, 使数组在第二个轴上分片.
3.  **AllReduce:** 移除一个“未归约”后缀, 使数组在该轴上未分片.

还有一种核心通信原语需要提及, 它出现在专家混合 (MoE) 模型和其他计算中: **AllToAll**.

### 我们最后的通信原语: AllToAll

最后一个基本的集合操作, 在考虑分片矩阵乘法时不会自然出现, 但在实践中经常出现, 是 **AllToAll** 集合, 或者更准确地说, 是*分片转置*或重分片操作的特殊情况. 例如

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAll 通常需要重新排列分片计算的不同区域之间的分片布局, 这些区域没有兼容的布局方案. 在考虑分片专家混合模型时, 它们会自然出现. *你可以将 AllToAll 视为将一个下标从一个轴移动到另一个轴*. 因为 all to all 不需要复制每个分片的所有数据到环上, 它实际上比 AllGather *便宜* (便宜 1/4)<d-footnote>对于偶数大小的双向环, 每个设备将向右发送 $(N/2 + (N/2-1) + … + 1)$ 个块, 向左发送 $((N/2-1) + … + 1)$ 个块 $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$. 每个块 (又名分片的分片) 的大小是 $\text{bytes} / N^2$, 所以每个设备的成本是 $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$. 这个结果在所有设备上都适用, 因为总带宽随设备数量而扩展.</d-footnote>.

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

如果我们推广到 ND AllToAll, 在 AxBxC 网格上, 一个 V 字节数组的总成本是

$$T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

其中, 像往常一样, $W_\text{ici}$ 是双向 ICI 带宽. 对于 1D 网格, 这简化为 $V / (4 \cdot W_\text{ici})$, 这是 AllReduce 成本的 1/4. 在 2D 中, 成本实际上随着最小轴的大小而降低.

*旁注: 如果你想要一个粗略的推导, 从一个 1D 环面 $\mathbb{Z} / N\mathbb{Z}$ 开始. 如果我们随机选择一个源节点和目标节点, 它们平均相距 N / 4 跳, 这给了我们一个成本 $(V \cdot N) / (4 * N)$. 现在如果我们考虑一个 ND 环面, 每个轴基本上是独立的. 每个节点有 $1 / Z$ 字节, 平均需要将其数据跳跃 $\max(A, B, C, …) / 4$ 跳.*

### 关于 ReduceScatter 的更多信息

ReduceScatter 是一个比它最初看起来更基本的操作, 因为它实际上是 AllGather 的导数, 反之亦然. 即, 如果在前向传播中我们有:

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

然后我们对反向模式导数 **A'** (通常在每个分片上都不同) 进行 ReduceScatter, 以推导出分片的 **A'**:

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同样, 在前向传播中 $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$ 意味着在后向传播中 $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$.

将 AllReduce 转换为 AllGather 和 ReduceScatter 还有一个方便的特性, 即我们可以将最终的 AllGather 推迟到稍后的某个时刻. 我们通常不希望支付在设备上复制完整矩阵乘积的成本. 相反, 我们希望即使在这种组合两个具有分片收缩维度的乘数的情况下, 也能保持分片状态:

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

在这种情况下, 我们也可以执行 ReduceScatter 而不是 AllReduce, 然后可以选择在稍后的某个时间执行 AllGather, 即

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

请注意, ReduceScatter *引入*了一个分片维度, 因此在这种情况下, 沿着 **I** 或 **K** 命名维度进行分片具有天然的自由度. 在使用 ReduceScatter 时, 我们通常需要选择*哪个*命名维度来引入新的分片 (尽管选择通常由更大的建模上下文强制). 这就是为什么我们使用语法 **ReduceScatter<sub>X,K</sub>** 来指定要分片的轴.

## 我们学到了什么?

*   数组的分片由一个**网格**指定, 该网格命名了我们 TPU 网格的物理, 硬件轴, 以及一个**分片**, 该分片将网格轴名称分配给数组的逻辑轴.
    *   例如, **A**[I<sub>XY</sub>, J] 描述了一个抽象数组 **A**, 其第一个维度沿两个网格轴 X 和 Y 分片. 结合 Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y')) 或缩写为 Mesh({'X': 4, 'Y': 8}), 这告诉我们我们的数组沿第一个维度进行了 32 路分片.

*   **使用分片数组的算术运算与使用未分片数组的算术运算完全相同, 除非你沿分片轴执行收缩**. 在这种情况下, 我们必须引入一些通信. 我们考虑四种情况:

    1.  *两个数组都没有沿收缩维度分片*: 不需要通信.
    2.  *一个数组沿收缩维度分片* (或者收缩维度沿不同轴分片): 我们在执行操作之前对其中一个输入进行 AllGather.
    3.  *两个数组都沿收缩维度相同地分片:* 我们在本地乘以分片, 然后执行 AllReduce 或 ReduceScatter.
    4.  *两个数组都沿同一网格轴沿非收缩维度分片:* 我们首先对其中一个输入进行 AllGather.

*   TPU 大约使用 **4 种核心通信原语**:
    1.  AllGather: $[A_X, B] \to [A, B]$
    2.  ReduceScatter: $[A, B] \\{U_X\\} \to [A, B_X]$
    3.  AllToAll: $[A, B_X] \to [A_X, B]$
    4.  AllReduce: $[A_X, B]\\{U_Y\\} \to [A_X, B]$ (技术上不是一个原语, 因为它结合了 ReduceScatter + AllGather)

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

*   这些操作的成本和延迟**不取决于轴的大小 (只要它们受带宽限制)**, 而只取决于输入数组的大小和链接的带宽. 对于单向 AllGather/ReduceScatter:

$$T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{数据量}}{\text{带宽}} \cdot \frac{\text{轴} - 1}{\text{轴}}
\longrightarrow \frac{\text{数据量}}{\text{带宽 (双向)}}$$

*   AllReduce 由一个 ReduceScatter 和一个 AllGather 组成, 因此成本是上述成本的 2 倍. AllToAll 只需要将分片部分地传递到环上, 因此成本是 AllGather 的 ¼. 这是一个总结:

| 操作 | 描述 | 语法 | 运行时间 |
| :---------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | 收集分片数组沿一个轴的所有分片, 移除一个下标.                                     | $[A_X, B] \to [A, B]$            | 字节 / (双向 ICI 带宽 * num_axes) |
| **ReduceScatter** | 对一个部分求和的数组沿一个轴求和, 并将其沿另一个轴分片 (添加一个下标).                 | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与 AllGather 相同                                |
| **AllReduce**     | 对一个部分求和的数组沿一个轴求和. 移除一个 { U<sub>x</sub> }. 结合了 AllGather 和 ReduceScatter. | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | 收集 (复制) 一个轴, 并将另一个维度沿同一轴分片.                                 | $[A, B_X] \to [A_X, B]$          | 双向环的 AllGather / 4           |

## 一些待解决的问题

*这里有一些基于本节内容的有启发性的问题. 我们目前不会包含所有答案, 但我们会尽可能多地写出答案.*

**问题 1 [复制分片]**: 一个数组被分片为 $A[I_X, J, K, \ldots]$ (即, 仅在 $X$ 上分片), 网格为 `Mesh({'X': 4, 'Y': 8, 'Z': 2})`. 所有芯片上 $A$ 占用的总字节数与数组一个副本的大小之比是多少?

{% details 点击这里查看答案. %}

我们的数组只在 X 上分片, 大小为 4, 所以实际上每个分片的大小为 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$. 由于我们的数组在 Y 和 Z 上复制, 总大小为 $Y \cdot Z \cdot \text{sizeof}(A)$, 所以总大小与单个芯片大小之比为 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$.

{% enddetails %}

**问题 2 [AllGather 延迟]**: 在 TPUv4p 4x4x4 切片上, 网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})`, 如果 $B=1024$ 且 $D=4096$ (bfloat16), $\text{AllGather}_X([B_X, D_Y])$ 需要多长时间? $$\text{AllGather}_{XY}([B_X, D_Y])$$ 呢? $$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$ 呢?

{% details 点击这里查看答案. %}

我们在所有轴上都有一个环绕链接, 因为我们有一个完整的 `4x4x4` 立方体, 所以我们有 9e10 的双向带宽可用.

1.  因为我们只是在一个轴上收集, 而另一个轴是分片的, 所以我们实际上是在 1 个轴上收集 $2BD / Y$ 字节. *如果你只考虑 Y 轴上的一个分片, X 轴上的 AllGather 看起来就像一个未分片的 AllGather, 字节数为 1 / Y.* 由于我们的 TPU v4p 的 ICI 带宽是双向 9e10 字节/秒, 这将需要 $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$.

2.  我们的带宽是以前的两倍, 但我们正在 AllGather 整个数组, 所以 `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`. 这远低于 4us 的延迟限制 (每跳 1us), 所以我们没问题.

3.  AllReduce 的成本是 AllGather 的两倍. 每个分片的大小为 $2BD / (X * Y)$, 所以成本约为 $4BD / (X * Y * W)$, 或大约 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`.

{% enddetails %}

**问题 3 [受延迟限制的 AllGather]**: 假设我们正在执行一个 $\text{AllGather}_X([B_X])$, 但 $B$ 非常小 (比如 128). 在 TPUv4p 4x4x4 切片上, 网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})`, bfloat16 格式, 这需要多长时间? *提示: 你可能受延迟限制.*

{% details 点击这里查看答案. %}

我们的 bfloat16 数组总共只使用 256 字节, 每个设备只有 64 字节. 由于我们在 TPU v4p 上有一个大小为 4 的轴, 我们有一个环绕链接, 所以我们可以双向发送数组. 单向带宽为 `4.5e10`, 每次跳跃大约需要 `64 / 4.5e10 ~ 0`, 所以我们肯定受延迟限制. 计算跳跃次数, 我们只需 2 次跳跃就可以完成整个收集, 所以大约 2us 是一个很好的估计.

{% enddetails %}

**问题 4 [矩阵乘法策略]**: 为了执行 $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$, 在本节中, 我们告诉你执行 $\text{AllGather}_X(Y[D_X, F])$ 并乘以完全复制的矩阵 (情况 2, *策略 1*). 相反, 你可以像 $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$ (情况 4, *策略 2*) 那样乘以本地分片, 然后 $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$. 这两种策略各执行多少 FLOPs 和通信? 哪种更好, 为什么?

{% details 点击这里查看答案. %}

让我们从我们的基线 (*策略 1*) 开始. 正如我们已经展示的, AllGather 的成本是 $2DF / W_\text{ici}$. 一旦我们有了完全复制的数组, 总计算时间是 $2BDF / C$ (其中 $C$ 是我们的加速器 FLOPs/s, 因为每个 TPU 执行相同的 FLOPs). 所以我们有

$$T_\text{total (策略 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下, 新策略 (策略 2) 对 $2BF$ 字节进行 AllReduce, 成本为 $4BF / W_\text{ici}$, 但执行的 FLOPs 少 $1 / X$ (因为计算是分片的). 这意味着我们执行 $2\cdot B\cdot D\cdot F / X$ FLOPs, 并且由此产生的 AllReduce 通信 $$2 \cdot 2 \cdot B \cdot F$$ 字节 (bfloat16). 因此, *策略 2* (没有 AllGather, 只是稍后进行 AllReduce) 的总时间大约是

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题是: *哪个更大?* 当 $D / (X \cdot C) > 2 / W_\text{ici}$, 或当 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$ 时, 策略 (2) 受计算限制. 我们可能合理地期望 $D \approx 8k$, 所以这意味着大约 $X < 2$, 这是不可能的 – 因此我们基本上总是使用策略 2 受通信限制. 使用基线 (策略 1), 当 $$B < C / W_\text{ici} = 2550$$ 时, 我们受通信限制, 这通常是正确的, 但并非总是如此.

所以如果 $B < 2550$, 我们在两种情况下都受通信限制, 我们有

$$T_\text{comms for Strategy 2} < T_\text{comms for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

当 $D > 2B$ 且 $2B < 5100$ 时, 这是正确的. 这通常是正确的, 所以如果我们的批量很小, 策略 2 有时会更好. 当我们的批量很大 ($B > 2550$) 时, 我们有

$$T_\text{comms for Strategy 2} < T_\text{math for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

当 $2 / W_\text{ici} < D / C$, 或当 $D > 2 * 2550 = 5100$ 时, 这是正确的, 这对于大型模型通常是正确的. 所以这种替代策略对于大型模型通常更好, 除非 $D$ 很小.

*我们为什么不总是这样做?* 嗯, 在实践中我们有时可能会这样做, 但通常很少有一个矩阵乘法的输入的收缩维度在一个轴上分片, 而另一个输入没有在该轴上分片. 例如, 如果我们正在做 FSDP (在[第 5 节](../training)中解释), 我们将在数据维度上分片我们的参数, 但我们的激活也将在数据维度上分片. 所以从这个意义上说, 这种情况不常出现.

{% enddetails %}

**问题 5 [最小延迟]**: 假设我想在 TPUv5p 4x4x4 上以尽可能低的延迟进行矩阵乘法 $A[B, D] \cdot_D B[D, F] \to C[B, F]$. 我的输入应该如何分片? 总的 FLOPs 和通信时间是多少?

**问题 6:** 假设我们想在 TPUv5e 4x4 上执行 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$. 我们执行什么通信? 通信与计算花费的时间各是多少?

*   $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ 呢? 这是训练中最标准的设置, 我们结合了数据, 张量和零分片.
*   $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ 呢? 这是推理的标准设置, 我们进行纯张量并行 (+数据).

**问题 7:** 一个典型的 Transformer 块有两个矩阵 $B[D, F]$ 和 $C[F, D]$, 其中 $F \gg D$. 批量大小为 B, 整个块是 $$C \cdot B \cdot x$$, 其中 $$x[B, D]$$. 让我们选择 $$D=8192$$, $$F=32768$$, 和 $$B=128$$, 并假设一切都是 bfloat16. 假设我们正在一个 TPUv5e 2x2 切片上运行, 但假设每个 TPU 只有 300MB 的可用内存. **B, C 和输出应该如何分片以保持在内存限制以下, 同时最小化总时间? 通信和 FLOPs 花费的时间各是多少?**

**问题 8 [挑战]**: 使用上面的简短代码片段作为模板, 分配一个分片数组, 并使用 pmap 或 shard_map 对 4 种主要通信原语 (AllGather, AllReduce, ReduceScatter 和 AllToAll) 进行基准测试. 你将需要使用 `jax.lax.all_gather`, `jax.lax.psum`, `jax.lax.psum_scatter` 和 `jax.lax.all_to_all`. 你理解这些函数的语义吗? 它们需要多长时间?

**问题 9 [分片矩阵乘法的另一种策略?]**: [上面](#case-2-one-multiplicand-has-a-sharded-contracting-dimension) 我们声称, 当只有一个矩阵乘法的输入沿其收缩维度分片时, 我们应该 AllGather 分片矩阵并在本地执行收缩. 你可能想到的另一种策略是执行分片矩阵乘法, 然后对结果进行 AllReduce (就好像两个输入都沿收缩维度分片一样), 即 $A[I, J_X] *_J B[J, K] \to C[I, K]$ 通过

1.  $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2.  $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题:

1.  明确写出这个算法, 用于矩阵 $A[N, M]$ 和 $B[M, K]$, 使用索引来准确显示在哪个设备上完成了什么计算. 假设 $A$ 在 ND 个设备上分片为 $A[I, J_X]$, 并且你希望你的输出在所有设备上复制.
2.  现在假设你对最终结果不要求在每个设备上复制, 而是分片 (跨 N 或 K 维度) 感到满意. 上面的算法会如何改变?
3.  仅从上面策略的通信成本来看 (在 (b) 部分, 而不是 (a) 部分), 这个通信成本与我们首先 AllGather A 然后进行矩阵乘法的算法的通信成本相比如何?

{% details 点击这里查看答案. %}


1.  首先计算外积, 将结果存储在 $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$ 中. 请注意, 重复的索引不是被收缩的索引, 因为我们正在进行外积. 这里的和范围是我们正在使用的特定设备上存储的 i 值的集合. 所以, 例如, 如果我们有一个大小为 16 的收缩轴, 和 4 个设备, 那么在设备 0 上, i 的范围是 {0, 1, 2, 3}; 在设备 1 上, i 的范围是 {4, 5, 6, 7}; 在设备 2 上, i 的范围是 {8, 9, 10, 11}; 在设备 3 上, i 的范围是 {12, 13, 14, 15}. 然后 AllReduce 每个设备上的 $O[N, K]$ 的部分和, 形成完整的 $O[N, K]$.
2.  我们可以在第 2 步中进行更便宜的 ReduceScatter, 而不是进行 AllReduce, 沿任一轴: $[N, K] \\{ U_X \\} \to [N_X, K]$ 或 $[N, K] \\{ U_X \\} \to [N, K_X]$.
3.  如上文所述, 进行 AllGather 的成本 (当我们受吞吐量限制时) 与 ReduceScatter 的成本相同; 它仅由我们正在处理的完整矩阵的大小给出. 因此, 在 gather-then-matmul 算法中, 这与 $NM$ 成比例 (因为我们正在 $\text{AllGather}$-ing $A$); 在 matmul-then-reduce-scatter 算法中, 这与 NK 成比例 (因为我们正在 reduce-scattering $O$). 所以两种算法的通信成本比是 `M/K`.

{% enddetails %}

**问题 10: AllToAll 的乐趣:** 在上表中, 注意到执行 AllToAll 的时间比执行 AllGather 或 ReduceScatter 的时间低 4 倍 (在我们受吞吐量限制的情况下). 在这个问题中, 我们将看到这 4 倍的来源, 并看到如果我们只有单向 ICI 链接, 而不是双向 ICI 链接, 这个因素会如何改变.

1.  让我们先从单向情况开始. 想象一下, 我们在一个环形拓扑中有 *D* 个设备, 如果我们正在对一个 N x N 矩阵 *A* 进行 AllGather 或 ReduceScatter, 该矩阵被分片为 $A[I_X, J]$ (假设 $D$ 整除 $N$). 描述这两个集合操作中涉及的通信, 并计算在整个算法期间**单个** ICI 链接上传输的标量 (浮点数或整数) 的总数.
2.  现在让我们考虑一个 AllToAll, 仍然在单向 ICI 的情况下. 在这种情况下, 算法与 all-gather 的情况有何不同? 计算在此算法中单个 ICI 链接上传输的标量数量.
3.  你应该发现 (a) 和 (b) 部分的答案之比是一个很好的数字. 用简单的术语解释这个因素的来源.
4.  现在让我们添加双向通信. 这如何影响 all-gather 情况下所需的总时间?
5.  添加双向通信如何影响 AllToAll 情况下所需的总时间?
6.  现在简单地解释一下双向环中 AllGather 时间和 AllToAll 时间之间的比率.

{% details 点击这里查看答案. %}

(1) **解决方案:** 过程很简单: 在算法的每个步骤中, 每个设备都会将一个单分片“条带”的矩阵 (总共 $$\frac{N}{D} \times N$$ 个元素) 发送给其最近的邻居. 这会发生 $$D-1$$ 次, 因为每个分片都需要被通信到除其起始设备之外的所有设备. 所以总共, 每个设备传输 $$\frac{N^2(D-1)}{D}$$ 个标量, 即流经单个 ICI 链接.

**答案:** $$N^2 (1-\frac{1}{D})$$, 或者当 $$D >> 1$$ 时, 简单地为 $$N^2$$.

(2) **解决方案:** 从通信的角度来看, AllToAll 和 AllGather 之间的关键区别在于, 在 AllToAll 中, 驻留在特定设备上的整个分片不需要被通信到每个其他设备. 想象一下存储在特定设备 (称之为设备 0) 上的分片是 $$[A, B, C, D]$$ (这里 A,B,C,D 是矩阵, 我们正在想象一个有 4 个设备的环来说明). 现在矩阵 $$A$$ 不需要被通信到任何地方, 矩阵 $$B$$ 需要最终到达设备 1; 矩阵 $$C$$ 最终到达设备 2; 矩阵 $$D$$ 最终到达设备 3. 所以在算法的第一步, 我们将 $$B$$, $$C$$, 和 $$D$$ 发送到设备 1; 在下一步, 设备 1 将 $$C$$ 和 $$D$$ 继续发送到设备 2; 在最后一步, 设备 2 只将 $$D$$ 发送到设备 3. 在这种情况下传输的参数总数是 $$(\text{A/B/C/D 的大小}) * (3 + 2 + 1)$$. A/B/C/D 的大小是 (在一般情况下) $$\frac{N^2}{D^2}$$, 并且同样在一般情况下, $$(3 + 2 + 1)$$ 项变成 $$((D-1) + (D-2) + … + 1)$$, 或 $$\frac{(D)(D-1)}{2}$$. 所以单个 ICI 链接上传输的总字节数是 $$\frac{N^2(D-1)}{D \times 2}$$.

**答案:** $$\frac{N^2}{2}(1-\frac{1}{D})$$, 或者当 $$D >> 1$$ 时, 简单地为 $$\frac{N^2}{2}$$.

(3) **解决方案:** 因子就是 $$\frac{1}{2}$$, 即在单向环形拓扑上, AllToAll 的成本是 all-gather/ReduceScatter 的一半. 回顾上面的推导, 这最终来自于这样一个事实, 即在 all-gather 的情况下, 我们每次传输相同大小的块 $$(D-1)$$ 次, 即我们正在做求和 $$ \text{小块大小} * (D + D + D + … + D)$$, 而在 AllToAll 的情况下, 我们正在做求和 $$\text{小块大小} * (D + D-1 + D-2 + … + 1)$$. 因此, 2 的因子本质上来自于这样一个事实, 即 $$1 + 2 + \ldots + n = n(n+1)/2$$.

(4) **解决方案**: 现在任何一个链接必须承载的总标量数量减少了 2 倍, 因为在双向环中, 每个“分片条带”可以同时双向发送.

(5) **解决方案**: 在这种情况下, 与单向情况相比, 我们赢得了 4 倍的优势. 这最容易通过考虑单个分片条带中每个大小为 (N2/D2) 的块的命运来看出, 比如说源自设备 0 的那个. 与 (单向情况) 发送其中一个块距离 D-1, 另一个块距离 D - 2, 等等一直到 1 不同, 我们现在将条带分成向右或向左移动的块, 最大移动距离为 ceil(D/2). 所以相应的和现在变成 $$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$, 或者在 $$D$$ 很大时为 $$D^2/8$$. 与单向情况下的 $$D^2/2$$ 相比, 我们看到我们赢得了 4 倍的优势.

(6) **解决方案:** 在单向环中, 我们看到 AllToAll 时间已经比 all-gather 时间快两倍; 这来自于我们不需要将我们的完整条带发送到每个设备的事实. 然后, 当我们添加双向性时, 我们看到对于 AllToAll 来说是 4 倍的胜利, 而对于 all-gathers 来说只有 2 倍的胜利. 将这些比率放在一起, 我们得到了我们所寻求的 4 倍因子.

{% enddetails %}

<h3 markdown=1 class="next-section">第三部分到此结束! 第四部分 (关于 Transformer 数学), 点击 [这里](../transformers)!</h3>