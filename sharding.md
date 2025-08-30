---
layout: distill
title: "分片矩阵及其乘法"
# permalink: /main/
description: "当我们训练大型ML模型时，我们必须将其参数或输入分割（或"分片"）到许多加速器上。由于LLM主要由矩阵乘法组成，理解这一点归结为理解当矩阵在设备间分割时如何相乘。我们基于TPU通信原语的成本开发了一个简单的分片矩阵乘法理论。"
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
  - name: "分片符号和集体操作"
  - subsections:
    - name: "分片的统一符号"
    - name: "我们如何在代码中描述这个？"
  - name: "使用分片数组进行计算"
  - subsections:
    - name: "情况1：两个乘数都没有分片的收缩维度"
    - name: "情况2：一个乘数具有分片的收缩维度"
    - name: "情况3：两个乘数都有分片的收缩维度"
    - name: "情况4：两个乘数都有一个非收缩维度沿相同轴分片"
  - name: "深入探讨TPU通信原语"
  - subsections:
    - name: "我们最后的通信原语：AllToAll"
    - name: "关于ReduceScatter的更多内容"
  - name: "我们学到了什么？"
  - name: "一些练习问题"

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

## 分片符号和集体操作

当我们在上万个TPU或GPU上训练LLM时，我们仍然在做与在单个设备上训练时抽象上相同的计算。不同之处在于**我们的数组无法放入单个TPU/GPU的HBM中**，因此我们必须将它们分割。<d-footnote>值得注意的是，我们也可能为了速度而选择并行化。即使我们可以适配在更少的芯片上，扩展到更多设备会给我们更多的FLOPs/s。例如，在推理过程中，我们有时可以适配在较小的拓扑结构上，但选择扩展到更大的拓扑结构以减少延迟。同样，在训练过程中，我们经常扩展到更多芯片以减少步进时间。</d-footnote>我们称之为对数组进行"*分片*"或"*分割*"。扩展的艺术在于找出如何分片我们的模型，以保持计算效率。

这是一个在4个TPU上分片的二维数组**A**的示例：

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图：</b> 形状为 <b>A</b>[I, J] 的示例数组在4个设备上分片。两个维度在2个设备上均匀分片，分片方式为 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]。每个TPU持有总内存的1/4。" %}

注意分片数组仍然与未分片数组具有相同的*全局*或*逻辑形状*，比如`(4, 128)`，但它也有一个*设备本地形状*，比如`(2, 64)`，这给了我们每个TPU实际持有的字节大小（在上图中，每个TPU持有总数组的¼）。现在我们将此推广到任意数组。

### 分片的统一符号

我们使用*命名轴符号*的变体来描述张量如何在设备间以块的形式分片：我们假设存在一个称为**设备网格(device mesh)**的2D或3D设备网格，其中每个轴都被赋予了**网格轴名称**，例如**X**、**Y和Z**。然后我们可以通过描述数组的每个命名维度如何跨物理网格轴分区来指定矩阵数据如何在设备网格上布局。我们称这种分配为**分片(sharding)**。

**示例（上图）**：对于上图，我们有：
* **网格：** 设备网格 `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，这告诉我们有4个TPU以2x2网格排列，轴名称为$X$和$Y$。
* **分片：** $A[I_X, J_Y]$，这告诉我们将第一个轴$I$沿网格轴$X$分片，第二个轴$J$沿网格轴$Y$分片。这种分片告诉我们每个分片持有数组的$1 / (\lvert X\rvert \cdot \lvert Y\rvert)$。

综合起来，我们知道数组的本地形状（单个设备持有的分片大小）是$(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中$$\lvert I\rvert$$是A的第一个维度的大小，$$\lvert J\rvert$$是A的第二个维度的大小。

<b markdown=1 style="color: #048affff;">小测验 [跨1轴的2D分片]：</b> 考虑一个数组 `fp32[1024, 4096]`，分片为 $A[I_{XY}, J]$，网格为 `{'X': 8, 'Y': 2}`。每个设备持有多少数据？在H100上从HBM加载这个数组需要多长时间（假设每个芯片的内存带宽为`3.4e12`）？

{% details 点击这里查看答案。%}

$A[I_{XY}, J]$ 将第一个维度（I）沿X和Y硬件轴分片。在这个例子中，本地形状是$(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。对于给定的例子，全局形状是`fp32[1024, 4096]`，所以本地形状是`fp32[64, 4096]`。

由于每个GPU有`4 * 64 * 4096 = 1MiB`字节，这大约需要`1e6 / 3.4e12 = 294ns`，但由于各种开销，实际时间可能会显著更长，因为数据量很小。

{% enddetails %}

**可视化这些分片：** 让我们通过查看一个在4个设备上分割的二维数据数组来尝试可视化这些分片：

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

我们将矩阵的*完全复制*形式简单地写为$A[I, J]$，没有分片分配。这意味着*每个*设备都包含整个矩阵的完整副本。

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

我们可以通过下标网格轴来指示其中一个维度已跨网格轴分区。例如，$A[I_X, J]$意味着**I**逻辑轴已跨**X**网格维度分区，但**J**维度*未*分区，并且块在**Y**网格轴上保持*部分复制*。

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$意味着**I**逻辑轴已跨**X**网格轴分区，并且**J**维度已跨**Y**网格轴分区。

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

我们在下图中说明了其他可能性：

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

这里$A[I_{XY}, J]$意味着我们将**X**和**Y**网格轴视为一个更大的扁平化维度，并将**I**命名轴跨所有设备分区。多个网格轴下标的顺序很重要，因为它指定了分区在网格上的遍历顺序。

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后，请注意我们*不能*让多个命名轴沿*相同*的网格维度分片。例如，$A[I_X, J_X]$是一个无意义的、被禁止的分片。一旦网格维度被用于分片数组的一个维度，它在某种意义上就被"用完"了。

<b markdown=1 style="color: #57cf57;">小测验：</b> 设**A**是一个形状为`int8[128, 2048]`的数组，分片为$A[I_{XY}, J]$，网格为`Mesh({'X': 2, 'Y': 8, 'Z': 2})`（总共32个设备）。**A**在每个设备上使用多少内存？**A**在所有设备上总共使用多少内存？

{% details 点击这里查看答案。%}

**答案：** 我们的数组**A**在X和Y上分片，在Z上复制，因此每个设备的形状为`int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小为`8 * 2048 = 16,384`字节。因为它在Z上复制，而在Z平面内完全在X和Y上分片，每个Z平面有一个副本，有2个这样的平面，所以总大小（跨所有设备）为`128 * 2048 * 2 = 512 KiB`。

{% enddetails %}

### 我们如何在代码中描述这个？

到目前为止，我们避免了谈论代码，但现在是一个很好的机会来一瞥。JAX使用一种命名分片语法，与我们上面描述的抽象语法非常匹配。我们将在[第10节](../jax-stuff)中更详细地讨论这个问题，但这里是一个快速预览。你可以在Google Colab[这里](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing)中尝试这个，并分析结果以了解JAX如何处理不同的分片。这个代码片段做了3件事：

1. 创建一个**jax.Mesh**，将我们的8个TPU映射到4x2网格，两个轴分别命名为'X'和'Y'。
2. 创建矩阵A和B，其中A沿其两个维度分片，B沿输出维度分片。
3. 编译并执行一个简单的矩阵乘法，返回一个分片数组。

```py
import jax
import jax.numpy as jnp

# 创建我们的网格！我们在TPU v2-8 4x2切片上运行，轴名称为'X'和'Y'。
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 一个小实用函数来帮助定义我们的分片。PartitionSpec是我们的
# 分片（从轴到名称的映射）。
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# 我们在非收缩维度上对A和B进行分片，并在收缩维度上对A进行分片。
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# 我们可以对这些分片数组执行矩阵乘法！out_shardings告诉我们我们希望
# 输出如何分片。JAX/XLA为我们处理其余的分片。
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX的酷之处在于这些数组表现得好像它们没有分片一样！`B.shape`会告诉我们全局或逻辑形状(2048, 8192)。我们必须实际查看`B.addressable_shards`来了解它如何在本地分片。我们可以对这些数组执行操作，JAX将尝试找出如何广播或重塑它们以执行操作。例如，在上面的例子中，**A**的本地形状是`[2, 1024]`，**B**的本地形状是`[2048, 4096]`。JAX/XLA将根据需要自动在这些数组之间添加通信以执行最终的乘法。

## 使用分片数组进行计算

如果你有一个分布在许多设备上的数据数组，并希望对其执行数学运算，那么与分片数据和计算相关的开销是什么？

显然，这取决于所涉及的计算。

* 对于*逐元素*操作，在分布式数组上操作**没有开销**。
* 当我们希望在驻留在许多设备上的元素之间执行操作时，事情会变得复杂。幸运的是，对于大多数机器学习，几乎所有的计算都以矩阵乘法的形式进行，而它们相对容易分析。

本节的其余部分将讨论如何乘以分片矩阵。首先，这涉及移动矩阵的块，以便你可以完全乘以或求和每个块。**每种分片都涉及不同的通信。**例如，$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$可以在没有任何通信的情况下进行乘法，因为*收缩维度*（J，我们实际求和的维度）是未分片的。但是，如果我们希望输出未分片（即$A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），我们需要将$A$或$C$复制到每个设备（使用*AllGather*）。这两种选择有不同的通信成本，因此我们需要计算这个成本并选择最低的一个。

{% details 你可以用"块矩阵乘法"来思考这个问题。%}

为了理解这一点，回忆一下"块矩阵"的概念可能会有帮助，即矩阵的嵌套矩阵：

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

矩阵乘法有一个很好的性质，即当矩阵乘数用块表示时，乘积可以用遵循标准规则的块矩阵乘法来表示：

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

这意味着实现分布式矩阵乘法归结为在网络上移动这些分片块，在块上执行*本地*矩阵乘法，并对它们的结果求和。**那么问题是要添加什么通信，以及它的成本是多少。**

{% enddetails %}

方便的是，我们可以将所有可能的分片归纳为大约4种需要考虑的情况，每种情况都有一个关于我们需要添加什么通信的规则：
1. **[情况1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension)：** 两个输入都没有沿收缩维度分片。_我们可以在没有任何通信的情况下乘以本地分片。_
2. **[情况2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)：** 一个输入具有分片的收缩维度。_我们通常沿收缩维度对分片输入进行"AllGather"。_
3. **[情况3](#case-3-both-multiplicands-have-sharded-contracting-dimensions)：** 两个输入都沿收缩维度分片。_我们可以乘以本地分片，然后"AllReduce"结果。_
4. **[情况4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)：** 两个输入都有一个非收缩维度沿相同轴分片。我们必须先对两个输入之一进行AllGather才能继续。

你可以将这些视为需要遵循的规则，但理解为什么这些规则成立以及它们的成本也是有价值的。我们现在将详细讨论每一种情况。

### 情况1：两个乘数都没有分片的收缩维度

**引理：** 当乘以分片矩阵时，计算是有效的，输出遵循输入的分片，*除非*收缩维度被分片或两个矩阵沿相同轴分片。例如，这可以正常工作

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

没有任何通信，结果是一个在X和Y硬件维度上分片的张量。试着思考为什么会这样。基本上，计算*独立于*分片，因为每个批次条目都有一些被收缩轴的本地块，它可以乘以和减少。任何这些情况都可以正常工作并遵循这个规则：

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

因为**A**和**B**都没有分片的收缩维度**J**，我们可以简单地执行输入的本地块矩阵乘法，结果将*已经*根据所需的输出分片进行分片。当两个乘数都有非收缩维度沿相同轴分片时，这不再成立（详细信息请参见[无效分片](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)部分）。

### 情况2：一个乘数具有分片的收缩维度

让我们考虑当一个输入**A**沿收缩**J**维度分片而**B**完全复制时该怎么做：

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能简单地乘以**A**和**B**的本地块，因为我们需要对**A**的完整收缩维度求和，该维度在X轴上分割。通常，我们首先对**A**的分片进行"**AllGather**"，以便每个设备都有完整副本，然后才乘以**B：**

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

这样，实际的乘法可以在每个设备上完全完成。

<p markdown=1 class="takeaway">**要点：** 当乘以其中一个矩阵沿收缩维度分片的矩阵时，我们通常先对其进行AllGather，使收缩不再分片，然后执行本地矩阵乘法。</p>

请注意，当**B**也没有沿X轴分片时，我们也可以执行本地部分矩阵乘法，然后对分片的部分和求和（或*AllReduce*），这在某些情况下可能更快。参见问题4[下面](#some-problems-to-work)。

**什么是AllGather？** AllGather是我们将讨论的第一个核心[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)通信原语。AllGather*移除沿轴的分片*，并将分布在设备上的分片重新组装到沿该轴的*每个*设备上。使用上面的符号，AllGather从一组轴中移除下标，例如

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

我们不必为给定维度移除所有下标，例如$$A[I_{XY}, J] \rightarrow A[I_Y, J]$$也是一个AllGather，只是仅在单个轴上。另请注意，我们可能还希望使用AllGather来移除*非收缩*维度分片，例如在矩阵乘法中：

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们可以最初对**A**进行AllGather以移除输入分片，或者我们可以执行分片矩阵乘法，然后对结果**C**进行AllGather。

**AllGather实际上是如何执行的？** 要在单个TPU轴（环）周围执行一维AllGather，我们基本上让每个TPU将其分片在环中传递，直到每个设备都有一个副本。<d-footnote>GPU AllGather也可以这样工作，你可以在节点中创建一个GPU环，并按该（任意）顺序传递块。</d-footnote> 这是一个动画：

{% include figure.liquid path="assets/img/all-gather.gif" caption="<b>图：</b> 显示如何在8个TPU或GPU设备组周围执行AllGather的动画。每个设备从数组的1/8开始，最终获得完整副本。" %}

我们可以单向或双向执行AllGather（上面显示了两个方向）。如果我们单向执行，每个TPU在环中经过$N - 1$跳发送大小为$\text{bytes} / N$的块。如果我们双向执行，我们有$\lceil \frac{N}{2} \rceil$跳，大小为$2 \cdot \text{bytes} / N$。

**这需要多长时间？** 让我们采用双向AllGather并计算它需要多长时间。设$$V$$为数组中的字节数，$X$为收缩维度上的分片数量。然后从上图来看，每个跳在每个方向发送$V / \lvert X\rvert$字节，所以每个跳需要

$$T_{hop} = \frac{2 \cdot V}{X \cdot W_\text{ici}}$$

其中$W_\text{ici}$是**双向**ICI带宽。<d-footnote>分子中的因子2来自于我们使用双向带宽的事实。我们在每个方向发送$V / X$，总共$2V / X$。</d-footnote>我们需要发送总共$\lvert X\rvert / 2$跳以到达每个TPU<d-footnote>技术上，$\lceil X / 2 \rceil$</d-footnote>，所以总归约需要

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

请注意，这**不依赖于$X$！** 这有点令人惊讶，因为这意味着即使我们的TPU只是本地连接的，连接的局部性也不重要。我们只是受到每个链路速度的限制。

<p markdown=1 class="takeaway">**要点：** 在吞吐量受限的情况下执行AllGather（或ReduceScatter或AllReduce）时，实际通信时间仅取决于数组的大小和可用带宽，而不是我们数组分片的设备数量！</p>

**关于ICI延迟的说明：** 每个ICI跳都有一些固有开销，无论数据量如何。这通常约为1us。这意味着当我们的数组$$A$$非常小且每个跳花费少于1us时，我们可以进入"延迟受限"状态，其中计算_确实_依赖于$X$。

{% details 点击此处查看完整详情。%}

设$$T_\text{min}$$为单跳的最小时间。那么

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

因为我们执行$X / 2$跳。对于大的归约或收集操作，我们完全受带宽限制。我们发送如此多的数据，以至于每个跳的开销基本上可以忽略不计。但对于小型数组（例如从模型中采样时），这不可忽略，ICI带宽也不相关。我们纯粹受延迟限制。换句话说，给定一个特定的TPU，例如TPU v5e，其单向ICI带宽为`4.5e10`，发送任何低于`4.5e10 * 1e-6 = 45kB`的缓冲区将受延迟限制。

{% enddetails %}

这是在TPU v5e 8x16切片上对AllGather带宽的实证测量。数组在16轴上分片，因此它具有完整的双向环。

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>图：</b> TPU v5e在AllGather期间的实证带宽和估计链路带宽。橙色BW是实际的每秒AllGather字节数，而蓝色曲线显示根据已知集体操作成本计算的经验单向链路带宽。" %}

请注意，我们仅达到声称的峰值带宽（`4.5e10`）的约95%，并且我们在约10MB时达到此峰值，当16路分片时，每个设备约500kB（*旁白：这比GPU好得多）。

**当我们在多个轴上AllGather时会发生什么？** 当我们在多个轴上收集时，我们有多个ICI维度可以执行收集。例如，AllGather<sub>XY</sub>([B, D<sub>XY</sub>])在两个硬件网格轴上操作。这可用带宽增加了$N_\text{axes}$倍。

{% details 点击此处查看完整详情。%}

一般来说，我们有

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]$$

其中$$\sum_i \lvert X_i \rvert / 2$$是TPU网格中最长路径的长度。

{% enddetails %}

<b markdown=1 style="color:rgb(144, 92, 255);">小测验2 [AllGather时间]：</b> 使用[第2部分](../tpus)中的数字，在TPUv5e上执行AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F]需要多长时间，使用2D网格`{'X': 8, 'Y': 4}`，$$E = 2048$$，$$F = 8192$$，使用bfloat16？如果$$E=256, F=256$$呢？

{% details 点击这里查看答案。%}

**答案：** 让我们从计算一些基本量开始：

1) TPU v5e的2个轴中每个轴的单向ICI带宽为4.5e10字节/秒。
2) 在bfloat16中对于(a)，我们有$A[E_Y, F]$，所以每个设备持有一个形状为bfloat16[512, 8192]的数组，大小为512 * 8192 * 2 = 8.4MB。总数组大小为2048 * 8192 * 2 = 34MB。

*对于第(1)部分*，我们可以使用上面的公式。由于我们在一个轴上执行AllGather，我们有$T_{\text{comms}} = \text{34e6} / \text{9e10} = \text{377us}$。为了检查我们是否不受延迟限制，我们知道在大小为4的轴上，我们最多有3跳，所以我们的延迟限制约为3us，所以我们并不接近。但是，TPU v5e只有一个轴的大小为16时才有回环连接，所以这里*我们实际上无法执行完全双向的AllGather*。我们必须执行3跳以使边缘的数据到达另一边缘，所以在理论上我们有更像$T_{\text{comms}} = 3 * \text{8.4e6} / \text{4.5e10} = 560\mu s$。[**这里**](https://imgur.com/a/RkvpRGQ)是[这个Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing)的**实际分析**，显示$680 \mu s$，这是合理的，因为我们可能没有获得100%的理论带宽！*对于第(2)部分*，每个分片大小为`64 * 256 * 2 = 32kB。32e3 / 4.5e10 = 0.7us`，所以我们受延迟限制。由于我们有3跳，这大约需要3 * 1us = 3us。[实际上，它更接近8us。](https://imgur.com/a/HZLQmYs)

{% enddetails %}

### 情况3：两个乘数都有分片的收缩维度

第三个基本情况是当两个乘数都在其收缩维度上分片，沿相同的网格轴：

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

在这种情况下，*本地*分片块矩阵乘法至少是*可能*执行的，因为它们将共享相同的收缩索引集。但是每个乘积将只代表完整所需乘积的*部分和*，并且沿**X**维度的每个设备将留下这个最终所需乘积的不同*部分和*。这种情况如此常见，以至于我们扩展我们的符号以明确标记这种情况：

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

符号**{ U<sub>X</sub> }**读作"**沿X网格轴未归约**"，指的是操作在某种意义上"未完成"的状态，因为它只会在最终求和后才完成。$\cdot_\text{LOCAL}$语法意味着我们执行本地求和但留下结果未归约。

这可以看作是关于矩阵乘法和外积的以下结果：

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

其中⊗是外积。因此，如果轴**X**上的TPU **i**拥有**A**的第**i**列和**B**的第**i**行，我们可以执行本地矩阵乘法以获得$$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$。这个矩阵在每个条目中拥有**A • B**在该条目处的和的第**i**项。我们仍然需要在**P**上执行求和，我们在网格轴**X**上对其进行了分片，以获得完整的**A • B**。如果我们按块（即分片）编写**A**和**B**，然后对结果的每个结果分片求和，这的工作方式相同。

我们可以使用沿**X**轴的完整**AllReduce**来执行这个求和以解决这个问题：

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce移除部分和，导致沿轴的*每个*设备具有相同的完全求和值。AllReduce是我们将在本节讨论的几个关键通信中的第二个，第一个是AllGather，其他的是ReduceScatter和AllToAll。AllReduce接受一个具有未归约（部分求和）轴的数组，并通过在未归约轴周围传递这些分片并累积结果来执行求和。签名是

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

这意味着它只是移除$\\{U_Y\\}$后缀，但其他方面保持结果不变。

**AllReduce的成本是多少？** 关于AllReduce如何执行的一个心理模型是每个设备将其分片发送到其邻居，并对它接收到的所有分片求和。显然，这比AllGather更昂贵，因为每个"分片"具有与完整数组相同的形状。通常，**AllReduce的成本是AllGather的两倍。** 看到这一点的一种方法是注意到**AllReduce**可以表示为两个其他原语的组合：**ReduceScatter**和**AllGather**。像AllReduce一样，ReduceScatter解决数组上的部分和，但结果输出沿给定维度"分散"或分区。AllGather收集所有这些片段并沿该物理轴"取消分区/取消分片/复制"逻辑轴。

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**ReduceScatter呢？** 就像AllReduce移除下标（上面的$F_Y \to F$）一样，ReduceScatter对未归约/部分求和的数组求和，然后沿相同的网格轴分散（分片）不同的逻辑轴。$[F]\\{U_Y\\} \to [F_Y]$。动画显示了这是如何完成的：请注意它与AllGather非常相似，但我们不是保留每个分片，而是将它们求和在一起。因此，它的延迟大致相同，不包括执行归约所花费的时间。

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

每个跳的通信时间只是每分片字节$V / Y$除以带宽$W_\text{ici}$，就像AllGather一样，所以我们有

$$T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ici}}$$

其中$$W_\text{ici}$$是双向带宽，只要我们有完整的环可以进行归约。

### 情况4：两个乘数都有一个非收缩维度沿相同轴分片

当对张量进行分片时，每个网格维度最多只能出现一次。执行上述规则有时会导致违反此规则的情况，例如：

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

这是无效的，因为沿维度**X**的给定分片，比如**i**，将拥有**C**的**(i, i)**个分片，即对角线条目。那么，在所有分片中没有足够的信息来恢复结果的对角线条目之外的任何内容，因此我们不能允许这种分片。

解决这个问题的方法是对某些维度进行AllGather。这里我们有两个选择：

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

或者

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

无论哪种情况，结果在其形状中只会提及**X**一次。我们选择哪一个将基于后续操作需要什么分片。

## 深入探讨TPU通信原语

前4种情况介绍了几种用于执行分片矩阵乘法的"核心通信原语"：

1. **AllGather：** 从分片中移除下标，收集分片。
2. **ReduceScatter：** 通过在该轴上对分片求和来移除数组的"未归约"后缀，使数组在第二轴上保持分片。
3. **AllReduce：** 移除"未归约"后缀，使数组在该轴上不分片。

还有一个核心通信原语需要提及，它在专家混合（MoE）模型和其他计算中出现：**AllToAll**。

### 我们最后的通信原语：AllToAll

最后一个基本的集体操作，在考虑分片矩阵乘法时不会自然出现，但在实践中经常出现，是**AllToAll**集体操作，或者更精确地说是*分片转置*或重新分片操作的特殊情况。例如

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAll通常需要在分片计算的不同区域之间重新排列分片布局，这些区域没有兼容的布局方案。在考虑分片的专家混合模型时，它们自然出现。*你可以将AllToAll视为将下标从一个轴移动到另一个轴*。因为all to all不需要在环上复制每个分片的所有数据，它实际上比AllGather*便宜*（便宜1/4倍）<d-footnote>对于偶数大小的双向环，每个设备将向右发送$(N/2 + (N/2-1) + … + 1)$个块，向左发送$((N/2-1) + … + 1)$个块$= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每个块（即分片的分片）的大小是$\text{bytes} / N^2$，所以每设备成本是$(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$。这个结果在所有设备上扩展，因为总带宽随设备数量扩展。</d-footnote>。

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

如果我们推广到ND AllToAll，在AxBxC网格上$V$字节数组的总体成本是

$$T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

其中像往常一样$W_\text{ici}$是双向ICI带宽。对于1D网格，这简化为$V / (4 \cdot W_\text{ici})$，这是AllReduce成本的1/4。在2D中，成本实际上随最小轴的大小缩小。

*旁白：如果你想要这个事实的粗略推导，从1D环面$\mathbb{Z} / N\mathbb{Z}$开始。如果我们随机选择源节点和目标节点，它们平均相距N / 4跳，给我们$(V \cdot N) / (4 * N)$的成本。现在如果我们考虑ND环面，每个轴基本上是独立的。每个节点有$1 / Z$字节，平均必须将其数据跳$\max(A, B, C, …) / 4$跳。*

### 关于ReduceScatter的更多内容

ReduceScatter比它最初看起来更基础，因为它实际上是AllGather的导数，反之亦然。即，如果在前向传播中我们有：

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

然后我们对反向模式导数**A'**（通常在每个分片上都不同）进行ReduceScatter以推导分片的**A'**：

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同样，前向传播中的$$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$意味着反向传播中的$$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$。

将AllReduce转换为AllGather和ReduceScatter还有一个方便的特性，即我们可以将最终的AllGather推迟到稍后的某个时刻。通常，我们宁愿不支付重新组装在设备间复制的完整矩阵乘积的成本。相反，我们希望即使在结合两个具有分片收缩维度的乘数的情况下也保持分片状态：

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

在这种情况下，我们也可以执行ReduceScatter而不是AllReduce，然后可选地在稍后的某个时间执行AllGather，即

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

请注意，ReduceScatter*引入*一个分片维度，因此在这种情况下具有沿**I**或**K**命名维度分片的自然自由度。我们通常需要在使用ReduceScatter时选择*哪个*命名维度来引入新的分片（尽管选择通常由更大的建模上下文强制）。这就是为什么我们使用语法**ReduceScatter<sub>X,K</sub>**来指定要分片的轴。

## 我们学到了什么？

* 数组的分片由一个**Mesh**指定，它命名我们TPU网格的物理硬件轴，以及一个**Sharding**，它将网格轴名称分配给数组的逻辑轴。
  * 例如，**A**[I<sub>XY</sub>, J]描述一个抽象数组**A**，其第一维度沿两个网格轴X和Y分片。结合Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y'))或缩写的Mesh({'X': 4, 'Y': 8})，这告诉我们我们的数组在第一维度上分片32种方式。

* **分片数组的算术运算与未分片数组的运算完全相同，除非你沿分片轴执行收缩**。在这种情况下，我们必须引入一些通信。我们考虑四种情况：

  1. *两个数组都没有沿收缩维度分片*：不需要通信。
  2. *一个数组沿收缩维度分片*（或收缩维度沿不同轴分片）：我们在执行操作之前对其中一个输入进行AllGather。
  3. *两个数组都沿收缩维度相同分片*：我们在本地乘以分片，然后执行AllReduce或ReduceScatter。
  4. *两个数组都沿非收缩维度沿相同网格轴分片*：我们首先对其中一个输入进行AllGather。

* TPU使用大约**4个核心通信原语**：
  1. AllGather: $[A_X, B] \to [A, B]$
  2. ReduceScatter: $[A, B] \\{U_X\\} \to [A, B_X]$
  3. AllToAll: $[A, B_X] \to [A_X, B]$
  4. AllReduce: $[A_X, B]\\{U_Y\\} \to [A_X, B]$（技术上不是原语，因为它结合了ReduceScatter + AllGather）

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 这些操作中每个操作的成本和延迟**不依赖于轴的大小（只要它们受带宽限制）**，而仅依赖于输入数组的大小和链路的带宽。对于单向AllGather/ReduceScatter：

$$T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{数据量}}{\text{带宽}} \cdot \frac{\text{轴} - 1}{\text{轴}}
\longrightarrow \frac{\text{数据量}}{\text{带宽（双向）}}$$

* AllReduce由ReduceScatter后跟AllGather组成，因此具有上述成本的2倍。AllToAll只需将分片部分地在环中传递，因此成本是AllGather的1/4。这是一个总结：

| 操作            | 描述                                                                                                          | 语法                           | 运行时间                                          |
| :-------------- | :----------------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**   | 沿轴收集分片数组的所有分片，移除下标。                                                                      | $[A_X, B] \to [A, B]$            | bytes / (bidirectional ICI bandwidth * num_axes) |
| **ReduceScatter** | 沿轴对部分求和的数组求和，并沿另一个轴对其进行分片（添加下标）。                                              | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与AllGather相同                                |
| **AllReduce**   | 沿轴对部分求和的数组求和。移除{ U<sub>x</sub> }。结合AllGather和ReduceScatter。                                 | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**    | 收集（复制）一个轴并沿相同轴对不同的维度进行分片。                                                           | $[A, B_X] \to [A_X, B]$          | 双向环的AllGather / 4                           |

## 一些练习问题

*以下是基于本节内容的一些指导性问题。目前我们不会包含所有答案，但我们会尽可能多地写出答案。*

**问题1 [复制分片]**：一个数组分片为$A[I_X, J, K, \ldots]$（即，仅在$X$上分片），网格为`Mesh({'X': 4, 'Y': 8, 'Z': 2})`。$A$在所有芯片上占用的总字节数与数组的一个副本的大小的比率是多少？

{% details 点击这里查看答案。%}

我们的数组仅在X上分片，X的大小为4，所以实际上每个分片的大小为$[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。由于我们的数组在Y和Z上复制，总大小为$Y \cdot Z \cdot \text{sizeof}(A)$，所以总大小与单芯片大小的比率为$Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

{% enddetails %}

**问题2 [AllGather延迟]**：在TPUv4p 4x4x4切片上，网格为`Mesh({'X': 4, 'Y': 4, 'Z': 4})`，如果$B=1024$且$D=4096$，使用bfloat16，$\text{AllGather}_X([B_X, D_Y])$应该需要多长时间？$$\text{AllGather}_{XY}([B_X, D_Y])$$呢？$$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$呢？

{% details 点击这里查看答案。%}

我们在所有轴上都有回环链接，因为我们有一个完整的`4x4x4`立方体，所以我们有9e10双向带宽可以使用。

1. 因为我们只在一个轴上收集，另一个轴是分片的，所以我们有效地在一个轴上收集$2BD / Y$字节。*如果你只考虑沿Y轴的单个分片，沿X的AllGather看起来像是一个具有1/Y字节的未分片AllGather。*由于我们的TPU v4p的ICI带宽是9e10字节/秒双向，这将需要$2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$。

2. 我们有之前的两倍带宽，但我们在AllGathering整个数组，所以`T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这远低于4us的延迟限制（每跳1us），所以我们没问题。

3. AllReduce的成本是AllGather的两倍。每个分片的大小为$2BD / (X * Y)$，所以成本约为$4BD / (X * Y * W)$，或者大约`4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

{% enddetails %}

**问题3 [延迟受限的AllGather]**：假设我们正在执行$\text{AllGather}_X([B_X])$，但$B$非常小（比如128）。在TPUv4p 4x4x4切片上，网格为`Mesh({'X': 4, 'Y': 4, 'Z': 4})`，使用bfloat16，这应该需要多长时间？*提示：你可能受延迟限制。*

{% details 点击这里查看答案。%}

我们的数组在bfloat16中总共只使用256字节，每个设备只有64字节。由于我们在TPU v4p上有一个大小为4的轴，我们有回环链接，所以我们可以双向发送数组。有`4.5e10`的单向带宽，每跳大约需要`64 / 4.5e10 ~ 0`，所以我们肯定受延迟限制。计算跳数，我们可以在仅2跳内完成完整收集，所以大约2us是一个很好的估计。

{% enddetails %}

**问题4 [矩阵乘法策略]**：要执行$X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，在本节中我们告诉你执行$\text{AllGather}_X(Y[D_X, F])$并乘以完全复制的矩阵（情况2，*策略1*）。相反，你可以像$X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$（情况4，*策略2*）那样乘以本地分片，然后$\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。这些策略各执行多少FLOPs和通信？哪个更好，为什么？

{% details 点击这里查看答案。%}

让我们从我们的基线（*策略1*）开始。正如我们所展示的，AllGather的成本是$2DF / W_\text{ici}$。一旦我们有了完全复制的数组，总计算时间是$2BDF / C$（其中$C$是我们的加速器FLOPs/s，因为每个TPU执行相同的FLOPs）。所以我们有

$$T_\text{total (策略1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下，新策略（策略2）对$2BF$字节执行AllReduce，成本为$4BF / W_\text{ici}$，但执行$1 / X$更少的FLOPs（因为计算是分片的）。这意味着我们执行$2\cdot B\cdot D\cdot F / X$ FLOPs，结果AllReduce在bfloat16中通信$$2 \cdot 2 \cdot B \cdot F$$字节。因此，*策略2*的总时间（没有AllGather，只是稍后的AllReduce）大约是

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题是：*这些中哪个更大？*策略(2)在$D / (X \cdot C) > 2 / W_\text{ici}$时受计算限制，或者当$D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$时。我们可以合理地期望$D \approx 8k$，所以这意味着大约$X < 2$，这是不太可能的——因此我们基本上总是策略2受通信限制。使用基线（策略1），当$$B < C / W_\text{ici} = 2550$$时我们受通信限制，这通常但不总是成立。

所以如果$B < 2550$，我们在两种情况下都受通信限制，我们有

$$T_\text{comms for 策略2} < T_\text{comms for 策略1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

这在$D > 2B$时成立，其中$2B < 5100$。这通常成立，所以如果我们的批次很小，策略2有时可能更好。当我们的批次很大时（$B > 2550$），我们有

$$T_\text{comms for 策略2} < T_\text{math for 策略1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

这在$2 / W_\text{ici} < D / C$时成立，或者当$D > 2 * 2550 = 5100$时，这通常对大模型成立。所以这种替代策略通常对大模型更好，除非$D$很小。

*为什么我们不总是这样做？* 嗯，在实践中我们有时会这样做，但通常很少遇到矩阵乘法的一个输入的收缩维度沿另一个输入没有分片的轴分片的情况。例如，如果我们正在做FSDP（在[第5节](../training)中解释），我们将在数据维度上对我们的参数进行分片，但我们的激活_也将在数据上分片_。所以从这个意义上说，这不会经常出现。

{% enddetails %}

**问题5 [最小延迟]**：假设我想在TPUv5p 4x4x4上执行矩阵乘法$A[B, D] \cdot_D B[D, F] \to C[B, F]$，具有最低可能的延迟。我的输入应该如何分片？总FLOPs和通信时间是多少？

**问题6：** 假设我们想在TPUv5e 4x4上执行$A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。我们执行什么通信？通信和计算各花费多少时间？

* $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$呢？这是训练的标准设置，我们结合数据、张量和零分片。
* $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$呢？这是推理的标准，我们执行纯张量并行性（+数据）。

**问题7：** 一个典型的Transformer块有两个矩阵$B[D, F]$和$C[F, D]$，其中$F \gg D$。使用批次大小B，整个块是$$C \cdot B \cdot x$$，其中$$x[B, D]$$。让我们选择$$D=8192$$，$$F=32768$$，和$$B=128$$，并假设一切都是bfloat16。假设我们在TPUv5e 2x2切片上运行，但假设每个TPU只有300MB的空闲内存。**B、C和输出应该如何分片以保持在内存限制以下，同时最小化总时间？通信和FLOPs各花费多少时间？**

**问题8 [挑战]**：使用上面的简短代码片段作为模板，分配一个分片数组并使用pmap或shard_map对4个主要通信原语（AllGather、AllReduce、ReduceScatter和AllToAll）进行基准测试。你将要使用`jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter`和`jax.lax.all_to_all`。你理解这些函数的语义吗？它们需要多长时间？

**问题9 [分片矩阵乘法的另一种策略？]**：[上面](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)我们声称当矩阵乘法只有一个输入沿其收缩维度分片时，我们应该对分片矩阵进行AllGather并在本地执行结果收缩。你可能想到的另一种策略是执行分片矩阵乘法，然后对结果进行AllReduce（就像两个输入都沿收缩维度分片一样），即$A[I, J_X] *_J B[J, K] \to C[I, K]$通过

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题：

1. 明确写出矩阵$A[N, M]$和$B[M, K]$的这个算法，使用索引来显示在什么设备上完成什么计算。假设$A$在ND设备上分片为$A[I, J_X]$，并且你希望你的输出在所有设备上复制。
2. 现在假设你对最终结果不在每个设备上复制，而是分片（跨N或K维度）感到满意。上面的算法会如何改变？
3. 纯粹看上述策略的通信成本（在(b)部分，不是(a)部分），这个通信成本与我们首先AllGather A然后执行矩阵乘法的算法的通信成本相比如何？

{% details 点击这里查看答案。%}


1. 首先计算外积，将结果存储在$$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$中。请注意，重复的索引不是被收缩的索引，因为我们正在做外积。这里求和的范围是我们正在使用的特定设备上存储的i值集。所以，例如，如果我们有一个大小为16的收缩轴，和4个设备，那么在设备0上，i的范围是{0, 1, 2, 3}；在设备1上，i的范围是{4, 5, 6, 7}；在设备2上，i的范围是{8, 9, 10, 11}；在设备3上，i的范围是{12, 13, 14, 15}。然后对$O[N, K]$的部分和进行AllReduce，这些部分和存在于每个设备上，以形成完整的$O[N, K]$。
2. 而不是在第2步中执行AllReduce，我们可以用更便宜的ReduceScatter来替代，沿任一轴：$[N, K] \\{ U_X \\} \to [N_X, K]$或$[N, K] \\{ U_X \\} \to [N, K_X]$。
3. 如上面主文所述，执行AllGather的成本（当我们受吞吐量限制时）与ReduceScatter的成本相同；它简单地由我们正在处理的完整矩阵的大小给出。所以在gather-then-matmul算法中，这缩放为$NM$（因为我们在$\text{AllGather}$-ing $A$）；在matmul-then-reduce-scatter算法中，这缩放为NK（因为我们在reduce-scattering $O$）。所以两个算法的通信成本比率是`M/K`。

{% enddetails %}

**问题10：AllToAll的乐趣：** 在上表中，注意到执行AllToAll的时间比执行AllGather或ReduceScatter的时间低4倍（在我们受吞吐量限制的情况下）。在这个问题中，我们将看到这个4倍因子来自哪里，并且看到如果我们只有单向ICI链接，而不是双向ICI链接，这个因子会如何变化。

1. 让我们从单向情况开始。想象我们在环拓扑中有*D*个设备，如果我们正在执行AllGather或ReduceScatter，在N x N矩阵*A*上，该矩阵分片为$A[I_X, J]$（假设$D$整除$N$为简单起见）。描述这两个集体操作中涉及的通信，并计算在这个算法的整个过程中传输通过**单个**ICI链接的标量（浮点数或整数）总数。
2. 现在让我们思考AllToAll，仍然在单向ICI情况下。在这种情况下，算法与all-gather情况有何不同？计算在这个算法中传输通过单个ICI链接的标量数量。
3. 你应该发现你对部分(a)和部分(b)的答案之间的比率是一个很好的数字。用简单的术语解释这个因子来自哪里。
4. 现在让我们添加双向通信。这对all-gather情况下所需的总时间有什么影响？
5. 添加双向通信对AllToAll情况下所需的总时间有什么影响？
6. 现在简单地解释双向环中AllGather时间和AllToAll时间之间的比率。

{% details 点击这里查看答案。%}

(1) **解决方案：** 过程很简单：在算法的每一步中，每个设备将向其最近的邻居发送矩阵的单个分片"条带"（总共$$\frac{N}{D} \times N$$个元素）。这发生$$D-1$$次，因为每个分片需要被通信到除它开始所在设备之外的所有设备。所以总共，每个设备传输$$\frac{N^2(D-1)}{D}$$个标量，即流经单个ICI链接。

**答案：** $$N^2 (1-\frac{1}{D})$$，或者当$$D >> 1$$时简单地$$N^2$$。

(2) **解决方案：** 从通信的角度来看，AllToAll和AllGather之间的关键区别在于，在AllToAll中，驻留在特定设备上的分片的全部内容不需要被通信到每个其他设备。想象存储在特定设备上（称之为设备0）的分片是$$[A, B, C, D]$$（这里A,B,C,D是矩阵，我们想象一个有4个设备的环用于说明）。现在矩阵$$A$$不需要被通信到任何地方，矩阵$$B$$需要最终在设备1上；矩阵$$C$$最终在设备2上；矩阵$$D$$最终在设备3上。所以在算法的第一步，我们将$$B$$、$$C$$和$$D$$发送到设备1；在下一步中，设备1将$$C$$和$$D$$转发到设备2；在最后一步，设备2仅将$$D$$发送到设备3。在这种情况下传输的参数总数是$$(A/B/C/D的大小) * (3 + 2 + 1)$$。A/B/C/D的大小是（现在在一般情况下）$$\frac{N^2}{D^2}$$，并且在一般情况下$$(3 + 2 + 1)$$项变为$$((D-1) + (D-2) + … + 1)$$，或$$\frac{(D)(D-1)}{2}$$。所以流经单个ICI链接的总字节数是$$\frac{N^2(D-1)}{D \times 2}$$。

**答案：** $$\frac{N^2}{2}(1-\frac{1}{D})$$，或者当$$D >> 1$$时简单地$$\frac{N^2}{2}$$。

(3) **解决方案：** 因子简单地是$$\frac{1}{2}$$，即AllToAll的成本是单向环拓扑上all-gather/ReduceScatter的一半。查看上面的推导，这最终来自于这样一个事实：在all-gather情况下，我们每次传输相同大小的块$$(D-1)$$次，即我们在做和$$ \text{tiny block size} * (D + D + D + … + D)$$，而在AllToAll情况下，我们在做和$$\text{tiny block size} * (D + D-1 + D-2 + … + 1)$$。因此，两倍因子本质上来自于$$1 + 2 + \ldots + n = n(n+1)/2$$这一事实。

(4) **解决方案**：现在任何单个链接必须承载的标量总数减少了2倍，因为在双向环中，每个"分片条带"可以同时双向发送。

(5) **解决方案**：在这种情况下，与单向情况相比，我们获得了4倍的增益。通过考虑单个分片条带中每个大小为(N2/D2)的块的命运最容易看到这一点，比如起源于设备0的那个。我们不是像在单向情况下那样将这些块中的一个发送距离D-1，另一个块发送距离D-2，等等，一直到1，而是现在将条带分成向右或向左移动的块，移动最大距离ceil(D/2)。所以相应的和现在变为$$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$，或者在大$$D$$的极限下$$D^2/8$$。与单向情况下的$$D^2/2$$相比，我们看到我们获得了4倍的增益。

(6) **解决方案：** 在单向环中，我们看到AllToAll时间已经是all-gather时间的两倍快；这来自于我们不需要将我们的完整条带发送到每个单个设备的事实。然后，当我们添加双向性时，我们看到AllToAll获得了4倍的增益，而all-gather只获得了2倍的增益。将这些比率放在一起，我们得到了我们寻求的4倍因子。

{% enddetails %}

<h3 markdown=1 class="next-section">第3部分到此结束！第4部分（关于Transformer数学），点击[这里](../transformers)！</h3>
