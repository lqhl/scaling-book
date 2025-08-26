---
layout: distill
title: "如何看待 TPU"
# permalink: /main/
description: "本节全部关于 TPU 的工作原理, 它们如何联网以实现多芯片训练和推理, 以及这如何影响我们最喜欢的算法的性能. 对于 GPU 用户来说, 这里也有一些好东西!"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "Part 1: Rooflines"

next_section_url: ../sharding
next_section_name: "Part 3: Sharding"

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
  - name: "什么是 TPU?"
  - name: "TPU 网络"
  - name: "关键要点"
  - subsections:
    - name: "TPU 规格"
  - name: "已解决的问题"
  - name: "附录"
  - subsections:
    - name: "附录 A: 关于 TPU 内部的更多信息"
    - name: "附录 B: 脉动阵列是如何工作的?"

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

<p markdown=1 class="announce">你可能也会喜欢阅读关于 NVIDIA GPU 的新 [第 12 节](../gpus)!</p>

## 什么是 TPU?

**TPU 基本上是一个专门用于矩阵乘法 (称为 TensorCore) 的计算核心, 附带一堆快速内存 (称为高带宽内存或 HBM ó<d-cite key="tpu_paper"></d-cite>.** 这是一个图表:

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图:</b> TPU 芯片的基本组件. TensorCore 是左侧的灰色框, 包含矩阵乘法单元 (MXU), 向量单元 (VPU) 和向量内存 (VMEM)." %}

你可以将 TensorCore 基本上看作是一台非常擅长矩阵乘法的机器, 但它还有一些其他值得注意的功能. TensorCore 有三个关键单元:

*   **MXU** (矩阵乘法单元) 是 TensorCore 的核心. 对于大多数 TPU 代, 它每 8 个周期使用脉动阵列执行一次 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 矩阵乘法 (有关详细信息, 请参见<a href="#appendix-b-how-does-a-systolic-array-work">附录 B</a>).
    *   在 TPU v5e 上, 每个 MXU 在 1.5GHz 时大约为 `5e13` bf16 FLOPs/s. 大多数 TensorCore 有 2 或 4 个 MXU, 因此例如 TPU v5e 的总 bf16 FLOPs/s 为 `2e14`.
    *   TPU 还支持具有更高吞吐量的较低精度矩阵乘法 (例如, 每个 TPU v5e 芯片可以执行 `4e14` int8 OPs/s).

*   **VPU** (向量处理单元) 执行通用的数学运算, 如 ReLU 激活或向量之间的逐点加法或乘法. 归约 (求和) 也在这里执行. <a href="#appendix-a-more-on-tpu-internals">附录 A</a> 提供了更多详细信息.
*   **VMEM** (向量内存) 是位于 TensorCore 内部的片上暂存器, 靠近计算单元. 它比 HBM 小得多 (例如, 在 TPU v5e 上为 128 MiB), 但与 MXU 的带宽要高得多. VMEM 的操作有点像 CPU 上的 L1/L2 缓存, 但要大得多并且由程序员控制. HBM 中的数据需要先复制到 VMEM 中, 然后 TensorCore 才能对其进行任何计算.

**TPU 在矩阵乘法方面非常非常快**. 这主要是它们所做的, 而且它们做得很好. [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture), 迄今为止最强大的 TPU 之一, 每核每秒可以执行 `2.5e14` 次 bf16 FLOPs, 或每芯片每秒 `5e14` 次 bf16 FLOPs. 一个由 8960 个芯片组成的 pod 每秒可以执行 4 exaflops. 这*很多*. 这是世界上最强大的超级计算机之一. 而且谷歌有很多这样的计算机.<d-footnote>TPU, 特别是它们的脉动阵列, 是如此强大的硬件加速器, 因为矩阵乘法是少数几个使用 $O(n^3)$ 计算量和 $O(n^2)$ 字节的算法之一. 这使得普通的 ALU 很容易受到计算而不是内存带宽的瓶颈.</d-footnote>

上图还包括一些其他组件, 如 SMEM 和标量单元, 它们用于控制流处理, 并在<a href="#appendix-a-more-on-tpu-internals">附录 A</a>中简要讨论, 但对于理解来说并不重要. 另一方面, HBM 很重要而且相当简单:

*   **HBM** (高带宽内存) 是一大块快速内存, 用于存储 TensorCore 使用的张量. HBM 的容量通常在几十 GB 的数量级 (例如, [TPU v5e 有 16GiB 的 HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)). 

    *   当需要进行计算时, 张量会从 HBM 通过 VMEM (见下文) 流式传输到 MXU, 结果会从 VMEM 写回 HBM. 

    *   HBM 和 TensorCore (通过 VMEM) 之间的带宽被称为“HBM 带宽” (通常在 1-2TB/秒左右), 它限制了在受内存限制的工作负载中计算的速度.

**通常, 所有 TPU 操作都是流水线化和重叠的.** 为了执行矩阵乘法 $X 	imes A 	o Y$, TPU 首先需要将矩阵 $A$ 和 $X$ 的块从 HBM 复制到 VMEM, 然后将它们加载到 MXU 中, MXU 会乘以 8x128 (对于 $X$) 和 128x128 (对于 $A$) 的块, 然后将结果逐块复制回 HBM. 为了高效地执行此操作, 矩阵乘法是流水线化的, 因此与 VMEM 的复制与 MXU 的工作重叠. 这使得 MXU 可以继续工作, 而不是等待内存传输, 从而使矩阵乘法受计算限制, 而不是受内存限制.

这是一个如何从 HBM 执行逐元素乘积的示例:

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图:</b> 一个动画, 显示了在 TPU 上执行的逐点乘积, 字节从 HBM 加载. 请注意字节如何以块的形式从内存中流出, 部分结果如何流水线化地返回, 而无需等待整个数组被物化." %}

矩阵乘法看起来几乎完全相同, 只是它会加载到 MXU 而不是 VPU/向量单元, 并且加载和存储会以不同的顺序发生, 因为相同的权重块会用于多个激活块. 你可以看到数据块流式传输到 VMEM, 然后到 VREGs (向量寄存器), 然后到向量单元, 然后再回到 VMEM 和 HBM. 正如我们即将看到的, 如果从 HBM 到 VMEM 的加载比向量单元 (或 MXU) 中的 FLOPs 慢, 我们就会变得“受带宽限制”, 因为我们正在让 VPU 或 MXU 缺乏工作.

<p markdown=1 class="takeaway">**关键要点:** TPU 非常简单. 它们从 HBM 加载权重到 VMEM, 然后从 VMEM 加载到脉动阵列, 该阵列每秒可以执行约 200 万亿次乘加运算. HBM $\leftrightarrow$ VMEM 和 VMEM $\leftrightarrow$ 脉动阵列的带宽为 TPU 可以高效执行的计算设定了基本限制.</p>

**VMEM 和算术强度:** VMEM 比 HBM 小得多, 但它与 MXU 的带宽要高得多. 正如我们在[第 1 节](../roofline)中看到的, 这意味着如果一个算法可以将其所有输入/输出都放入 VMEM, 它就不太可能遇到通信瓶颈. 这在计算具有较差算术强度时特别有帮助: VMEM 带宽比 HBM 带宽高约 22 倍, 这意味着从 VMEM 读取/写入的 MXU 操作只需要 10-20 的算术强度即可实现峰值 FLOPs 利用率. 这意味着如果我们可以将权重放入 VMEM 而不是 HBM, 我们的矩阵乘法可以在更小的批量大小时受 FLOPs 限制. 这也意味着那些从根本上具有较低算术强度的算法仍然可以是高效的. VMEM 只是太小了, 这通常是一个挑战.<d-footnote>我们有时会谈论 VMEM 预取, 这是指提前在 VMEM 中加载权重, 以便我们可以掩盖矩阵乘法的加载成本. 例如, 在一个普通的 Transformer 中, 我们有时可以在注意力期间将我们的大型前馈权重加载到 VMEM 中, 如果我们受内存带宽限制, 这可以隐藏权重加载的成本. 这要求我们的权重足够小或分片足够多, 以便将单个层放入 VMEM 并留有余地.</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**一个 TPU 芯片通常 (但并非总是) 由两个共享内存的 TPU 核心组成, 可以被认为是一个具有两倍 FLOPs 的大型加速器** (称为“megacore”配置). 自 TPU v4 以来一直如此. 较早的 TPU 芯片具有独立的内存, 并被视为两个独立的加速器 (TPU v3 及更早版本). 像 TPU v5e 这样的推理优化芯片每个芯片只有一个 TPU 核心.

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**以**4 个一组的形式排列在“托盘”上**, 通过 **PCIe 网络连接到 CPU 主机.** 这是大多数读者熟悉的格式, 4 个芯片 (8 个核心, 虽然通常被视为 4 个逻辑 megacore) 通过 Colab 或单个 TPU-VM 公开. 对于像 TPU v5e 这样的推理芯片, 我们每个主机有 2 个托盘, 而不是 1 个, 但每个芯片也只有 1 个核心, 这给了我们 8 个芯片 = 8 个核心.<d-footnote>在 Cloud TPU VM 上, 每个托盘都作为单独 VM 的一部分公开, 因此再次可以看到 4 个核心.</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 带宽是有限的:** 就像 HBM $\leftrightarrow$ VMEM 链接一样, CPU $\leftrightarrow$ HBM PCIe 连接具有特定的带宽, 限制了你可以从主机内存加载到 HBM 或反之亦然的速度. 例如, TPU v4 的 PCIe 带宽为每个方向 16GB/秒, 因此比 HBM 慢近 100 倍. 我们*可以*将数据加载/卸载到主机 (CPU) RAM 中, 但速度不是很快.

## TPU 网络

**芯片通过 ICI 网络在 Pod 中相互连接**. 在较早的几代 (TPU v2 和 TPU v3), 推理芯片 (例如, TPU v5e) 和 Trilium (TPU v6e) 中, ICI (“芯片间互连”) 连接 4 个最近的邻居 (带有边缘链接以形成 2D 环面). TPU v4 和 TPU v5p 连接到最近的 6 个邻居 (形成 3D 环面). 请注意, 这些连接**不**通过它们的主机, 它们是芯片之间的直接链接.

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环面结构将任意两个节点之间的最大距离从 $N$ 减少到 $N / 2$, 使通信速度快得多. TPU 还有一个“扭曲环面”配置, 它以类似莫比乌斯带的拓扑结构包裹环面, 以进一步减少节点之间的平均距离.

**TPU pod (由 ICI 连接) 可以变得非常大:** TPU v4 的最大 pod 大小 (称为**superpod**) 为 `16x16x16`, TPU v5p 为 `16x20x28`. 这些大型 pod 由可重新配置的 `4x4x4` 芯片立方体组成, 这些立方体通过[光学环绕链接](https://arxiv.org/pdf/2208.10041)<d-footnote>光学开关只是一个具有相同 ICI 带宽的可重新配置连接. 它只是让我们连接立方体, 同时保留一个环绕链接.</d-footnote>连接, 我们可以重新配置以连接非常大的拓扑.

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以请求较小的拓扑 (例如 `2x2x1`, `2x2x2`), 尽管没有环绕. 这是一个重要的警告, 因为它通常会使大多数通信的时间加倍. 任何完整立方体的倍数 (例如 `4x4x4` 或 `4x4x8`) 都将具有由光学开关提供的环绕.<d-footnote>请注意, `2x2x4` 不会有任何环绕, 因为它们是由光学开关提供的, 而光学开关仅在完整立方体上可用. 然而, TPU v5e 8x16 *将*在较长的轴上有一个环绕, 因为它不使用可重新配置的光学网络.</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e 和 Trilium pod 由单个 `16x16` 2D 环面组成, 在任何大小为 16 的轴上都有环绕 (这意味着 `8x16` 在长轴上有环绕). TPU v5e 和 v6e (Trillium) 不能扩展到超过 16x16 的环面, 但 pod 仍然可以通过标准的数据中心网络 (DCN) 相互通信, DCN 将 TPU 主机相互连接. 同样, 可以请求较小的拓扑, 在维度 $<16$ 上没有环绕.

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接是 TPU 和 GPU 之间的关键区别**. GPU 通过交换机层次结构连接, 近似于每个 GPU 之间的点对点连接, 而不是像 TPU 那样使用本地连接. 通常, 节点内的 GPU (H100 为 8 个 GPU, B200 多达 500 个) 是直接连接的, 而更大的拓扑则需要在每个 GPU 之间进行 O(log(N)) 次跳跃. 一方面, 这意味着 GPU 可以在节点内以单个低延迟跳跃发送任意数据. 另一方面, TPU 的价格要便宜得多 (因为 NVLink 交换机很昂贵) 并且布线更简单, 并且可以扩展到更大的拓扑, 因为每个设备的链接数量和每个设备的带宽是恒定的. 在[这里](../gpus#networking)阅读更多内容.

**ICI 相对于 DCN 非常快, 但仍比 HBM 带宽慢.** 例如, 一个 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 具有:

*   每个芯片 `2.5e12` 字节/秒 (2.5 TB/s) 的 HBM 带宽.
*   每个芯片每个轴 `9e10` 字节/秒 (90 GB/s) 的 ICI 带宽, 每个芯片有 3 个轴.<d-footnote>上面的页面列出了 100 GB/s 的带宽, 这与这里列出的略有不同. TPU ICI 链接的带宽根据执行的操作略有不同. 你通常可以不用担心地使用本文档中的数字.</d-footnote>
*   每个 TPU `6.25e9` 字节/秒 (6.25 GB/s) 的 DCN (出口) 带宽 (通过每个主机上的 1-2 个 NIC).<d-footnote>TPU v6e 为 12.5e9 字节/秒, v5e 为 3.125e9 字节/秒.</d-footnote>

这意味着当我们在多个芯片上拆分模型时, 我们需要小心避免用较慢的跨设备通信来瓶颈 MXU.

**多切片训练:** 一组由 ICI 连接的 TPU 称为一个**切片**. 不同的切片可以使用 DCN 相互连接, 例如连接不同 pod 上的切片. 由于 DCN 是比 ICI 慢得多的连接, 因此应尽量限制我们的计算等待来自 DCN 的数据的时间. DCN 是主机到主机的, 因此要通过 DCN 将缓冲区从 TPU 传输到 TPU, 我们首先需要通过 PCIe 传输到主机, 然后通过网络出口, 然后通过目标主机网络入口, 然后通过 PCIe 进入 HBM.

## 关键要点

*   TPU 很简单, 在大多数情况下可以被认为是一个连接到内存 (超快), 通过 ICI 连接到其他芯片 (相当快), 以及通过 DCN 连接到数据中心其余部分 (有点快) 的矩阵乘法单元.

*   通信受到我们各种网络带宽的限制, 按速度排序:
    *   HBM 带宽: 在 TensorCore 和其关联的 HBM 之间.
    *   ICI 带宽: 在 TPU 芯片和其最近的 4 或 6 个邻居之间.
    *   PCIe 带宽: 在 CPU 主机和其关联的芯片托盘之间.
    *   DCN 带宽: 在多个 CPU 主机之间, 通常是未通过 ICI 连接的主机.

*   **在一个切片内, TPU 仅通过 ICI 连接到其最近的邻居.** 这意味着在一个切片中, 远距离芯片之间的 ICI 通信需要先跳过中间的芯片.

*   **权重矩阵需要在两个维度上至少填充到 128 的大小** (在 TPU v6 上为 256) 以填满 MXU (实际上, 较小的轴被填充到 128).

*   **较低精度的矩阵乘法往往更快.** 对于支持它的几代 TPU, int8 或 int4 FLOPs 的速度大约是 bfloat16 FLOPs 的 2 倍/4 倍. VPU 操作仍然以 fp32 执行.

*   为了避免瓶颈 TPU 计算单元, 我们需要**确保每个通道的通信量与其速度成正比**.

### TPU 规格

以下是我们芯片的一些具体数字:

| 型号 | Pod 大小 | 主机大小 | HBM 容量/芯片 | HBM 带宽/芯片 (字节/秒) | FLOPs/s/芯片 (bf16) | FLOPs/s/芯片 (int8) |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.1e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |

主机大小指的是连接到单个主机的 TPU 的拓扑 (例如, TPU v5e 有一个连接到 8 个 TPU 的单个 CPU 主机, 拓扑为 4x2). 以下是互连数据:

| 型号 | ICI 带宽/链接 (单向, 字节/秒) | ICI 带宽/链接 (双向, 字节/秒) |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |              1e11              |            2e11             |
| **TPU v4p** |             4.5e10             |            9e10             |
| **TPU v5p** |              9e10              |           1.8e11            |
| **TPU v5e** |             4.5e10             |            9e10             |
| **TPU v6e** |              9e10              |           1.8e11            |

我们同时包含了单向 (unidirectional) 带宽和双向 (bidirectional) 带宽, 因为单向带宽更符合硬件实际情况, 但双向带宽在涉及完整环形的方程中更常出现.<d-footnote>我们所说的双向 (bidirectional) 带宽是指在两个方向上沿单个链接可以发送的总字节数, 或者同样地, 假设我们可以有效地使用两个链接, 沿特定轴从单个 TPU 发出的总字节数. 当我们有一个功能正常的环形, AKA 当我们在特定轴上有一个环绕连接时, 这是正确的. 这发生在推理芯片上, 当我们有一个完整的 16 轴时, 或者在训练芯片 (v*p) 上, 当我们有一个是 4 的倍数的轴时. 我们更喜欢使用双向带宽, 因为它在涉及双向通信的计算中经常出现.</d-footnote>

PCIe 带宽通常约为每个 TPU `1.6e10` 字节/秒 (`3.2e10` 用于 TPU v6e), 而 DCN 带宽通常约为每个 TPU `6.25e9` 字节/秒 (`12.5e9` 用于 TPU v6e, `3.125e9` 用于 TPU v5e).

## 已解决的问题

这些数字有点枯燥, 但它们可以让你对模型性能进行基本的屋顶线估计. 让我们解决几个问题来解释为什么这很有用. 你将在第 3 部分看到更多例子.

**问题 1 [限制 LLM 延迟]:** 假设你想从一个分布在 32 个 TPU v4p 上的 200B 参数的 bf16 模型中进行采样. 将所有参数从 HBM 加载到脉动阵列需要多长时间? *提示: 使用上面的数字.*

{% details 点击这里查看答案. %}

**答案:** 我们正在 32 个芯片上加载 `sizeof(bf16) * 200e9 = 400e9` 字节, 这意味着每个芯片 12.5e9 字节, 每个芯片的 HBM 带宽为 1.23e12. 所以加载大约需要 10 毫秒.

这很酷, 因为*这是从模型中采样的延迟的合理下限*. 每个采样步骤都需要从 HBM 加载所有参数, 因此它不能少于 10 毫秒. 在实践中, 在小批量大小时, 这几乎是可以实现的.

{% enddetails %}

**问题 2 [TPU 细节]:** 考虑一个完整的 TPU v5e pod. 总共有多少个 CPU 主机? 多少个 TPU TensorCore? 整个 pod 的总 FLOPs/s 是多少? 总 HBM 是多少? 对 TPU v5p pod 做同样的练习.

{% details 点击这里查看答案. %}

**答案:** 对于 TPU v5e, 每个 pod 是 `16x16`, 每个主机是一个 4x2 的切片, 所以我们有 `16*16 / 8 = 32` 个主机. 对于 TPU v5e, 每个 TPU 只有一个核心, 所以我们有 256 个 TensorCore. 在 bfloat16 中, 总 FLOPs/s 是 `16*16*2e14 = 5.1e16`. 每个芯片有 16GB 的 HBM, 所以总共有 `256 * 16 = 4TB` 的内存.

对于一个完整的 TPU v5p pod, 我们有 `16x20x28` 个芯片, 每个主机是 2x2x1, 所以我们有 `16*20*28 / 2*2 = 2,240` 个主机. 对于 TPU v5p, 每个 TPU 有两个 TensorCore, 所以我们有 `8960 * 2 = 17,920` 个核心. 在 bfloat16 中, 总 FLOPs/s 是 `8960 * 4.5e14 = 4e18`. 每个芯片有 96GB 的 HBM, 所以总共有 `8960 * 96 = 860TB` 的内存.

{% enddetails %}

**问题 3 [PCIe 操作强度]:** 想象一下, 我们被迫将一个大的权重矩阵 $A$ (类型为 $\text{bfloat16}[D, F]$) 和一批激活 $x$ (类型为 $\text{bfloat16}[B, D]$) 存储在主机 DRAM 中, 并希望对它们进行矩阵乘法. 这在单个主机上运行, 我们使用一个连接到它的单个 TPU v6e 芯片. 你可以假设 $B \ll D$, 并且 $F = 4D$ (我们将在后面的章节中看到为什么这些是合理的假设). 我们需要保持受 FLOPs 限制的最小批量大小 $B$ 是多少? 假设 PCIe 带宽为 1.5e10 字节/秒.

{% details 点击这里查看答案. %}

**答案:** 我们需要执行 $2BDF$ 次浮点运算, 每个芯片每秒可以执行 `9.2e14` 次浮点运算. 这需要 $2BDF / 9.2e14$ 秒来执行. 我们需要从 DRAM 加载 $2DF + 2BD$ 字节, 并将 $2BF$ 字节写回. 我们受到 PCIe 传输速度的瓶颈, 所以我们需要 $2 \cdot (BD + DF + BF) / 1.5e10$ 秒来将数据传入和传出 TPU. 因为我们希望计算时间比权重加载时间长, 假设我们可以将所有权重加载与计算重叠, 我们希望 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$. 我们可以使用我们的假设 $B \ll D$ 和 $F = 4D$ 来简化这个式子, 得到

$$\frac{8BD^2}{9.2e14} > \frac{8D^2}{1.5e10}$$

或

$$B > \frac{9.2e14}{1.5e10} \simeq 61,000$$

{% enddetails %}

**问题 4 [通用矩阵乘法延迟]:** 假设我们想将一个权重矩阵 int8[16384, 4096] 与一个大小为 int8[B, 4096] 的激活矩阵相乘, 其中 B 是某个未知的批量大小. 假设我们从 1 个 TPUv5e 开始.

1.  这个乘法需要多长时间, 作为 B 的函数? *提示: 计算从 HBM 加载数组需要多长时间以及实际乘法需要多长时间可能会有帮助. 哪个是你的瓶颈?*
2.  如果我们想在 VMEM 中运行这个操作呢? 作为 B 的函数, 它需要多长时间?

{% details 点击这里查看答案. %}

**答案:** (1) 我们需要执行的浮点运算次数是 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3e8 \cdot B$. 所以 $T_{\text{math}} = (1.3e8 \cdot B) / 3.94e14$ 秒. 我们需要从 HBM 加载 $16384 \cdot 4096 + 4096 \cdot B$ 字节到 VMEM, 并将 $16384 \cdot B$ 字节从 VMEM 写回 HBM. 这意味着 $T_{\text{comms}} = (6.7e7 + 2e4\cdot B) / 8.1e11$ 秒. 假设通信和计算尽可能多地重叠, 整个乘法大约需要

$$\max\{T_{\text{math}}, T_{\text{comms}}\}\} = \max\{\frac{6.7e7 + 2e4\cdot B}{8.1e11}, \frac{1.3e8 \cdot B}{3.94e14}\}

当 $\frac{6.7e7 + 2e4\cdot B}{8.1e11} < \frac{1.3e8 \cdot B}{3.94e14}$ 时, 或者等效地, $B > 271$ 时, 我们将受 FLOPs 限制. 这个数字比我们下面推导出的 240 略大, 因为我们考虑了 $$D$$ 和 $$F$$ 的全部影响.

(2) 如果我们改为从 VMEM 加载, 让我们将 VMEM 到 MXU 的带宽视为 HBM $\leftrightarrow$ VMEM 带宽的 22 倍. 这将我们的数据加载分母从 8.1e11 变为 1.78e13, 我们得到 $B > 11$. 请注意, 在实践中, 我们不能将所有 VMEM 带宽都用于加载 $W$, 因此在实践中它会更接近 20.

{% enddetails %}

**问题 5 [ICI 带宽]:** 假设我们有一个 TPU v5e `4x4` 切片. 假设我们想将一个类型为 `bfloat16[8, 128, 8192]` 的数组从 `TPU{0,0}` 发送到 `TPU{3, 3}`. 假设 TPU v5e 的每跳延迟为 $1\mu s$.

1.  第一个字节到达目的地需要多长时间?
2.  总传输需要多长时间?

{% details 点击这里查看答案. %}

**答案:** 在 TPUv5e 中, 我们有 2D 连接. 因为我们只有一个 `4x4` 切片 (没有大小为 16 的轴), 所以我们没有环绕连接. 因此, 我们的目标芯片可以从两个端口接收数据, 同样, 我们的源芯片可以从两个端口发送数据. 我们需要传输的数据量是 `2 * 8 * 128 * 8192 = 1.7e7` 字节. 我们可以同时从两个端口传输 (即, 将一半数组向右发送, 一半向下发送), 所以我们每秒传输 `2 * 4.5e10 = 9e10` 字节, 这意味着传输整个数组大约需要 `1.7e7 / 9e10 = 188us` (假设我们受带宽限制). 在一个 `4x4` 切片中, 芯片 $(0, 0)$ 和 $(3, 3)$ 之间有六次跳跃, 因为轴上少于 16 个芯片时没有环绕链接. 由于每次跳跃的延迟约为 $1\mu s$, 第一个字节将在大约 `6us` 内到达, 总传输将需要 `188us`.

{% enddetails %}

**问题 6 [综合题, 难]:** 想象你有一个大矩阵 **A**: `int8[128 * 1024, 128 * 1024]`, 它均匀地分片在一个 TPU v5e 4x4 切片上, 但卸载到每个芯片的主机 DRAM 中. 假设你想将整个数组复制到 TPU{0, 0} 并将其与一个向量 `bf16[8, 128 * 1024]` 相乘. 这需要多长时间? *提示: 使用上面的数字.*

{% details 点击这里查看答案. %}

**答案:** 让我们从概述我们需要执行的操作开始. 我们的数组大约是 16GB. 从上表中, 一个 TPU v5e 主机有一个 4x2 的拓扑, 所以一个 4x4 有 2 个主机. 因此, 由于我们的数组是均匀分片的, 每个主机实际上包含数组的 1/2, 或 8GB. 我们需要将这些块全部复制到 TPU{0,0}, 这给了我们两个选择:

1.  我们可以通过 DCN 复制, 然后通过 PCIe 将整个未分片的数组加载到 HBM 中.
2.  我们可以将我们分片的数组加载到它们相应的 TPU 上, 然后通过 ICI 执行一个 gather, 然后在 TPU{0,0} 上执行矩阵乘法.

很明显, 选项 (2) 更好. DCN 与 ICI 相比速度较慢, 我们更愿意通过许多 PCIe 链接加载一个大数组, 而不是仅仅几个 (主机 0 上的 8 个). 这是系统部分的一个图表. 如上所述, 请注意 TPU 通过 ICI 连接到它们的邻居 (即使跨主机), 所有 TPU 都通过 PCIe 连接到它们的主机 CPU, 主机通过 DCN 连接.

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="每个芯片实际上都有自己的 PCIe 链接到其主机, 虽然为了清晰起见, 这里只显示了一个." %}

现在让我们来看看每个部分需要多长时间:

1.  **PCIe 加载**: 我们正在通过 16 个 PCIe 链接加载 16GB / 2 = 8GB 的块, 每个链接的带宽为 `1.5e10` 字节/秒. 因此, 这大约需要 33 毫秒.

2.  **ICI 复制:** 每个 TPU 现在有 16GB / 16 = 1GB 的数组. 我们的 ICI 带宽是每个链接 *双向* 9e10 字节/秒, 你会从上图中注意到, 在这个拓扑中, TPU v5e 上的 4 个 ICI 链接中只有 2 个在 TPU{0,0} 上使用. 由于 TPU{0,0} 需要沿着 2 个轴以 `4.5e10` 字节/秒/链接的速度接收总共 15GB, 我们可以通过 `15e9 / (4.5e10 * 2) = 167ms` 来为时间设定下限. 在实践中, 这可能无法实现, 因为负载非常不均匀, 但可能在 2 倍的范围内. 正如你将在第 2 节中看到的, 执行一个完整的 AllGather 也大约需要 `16e9 / (4.5e10 * 2)`, 所以这接近最优.

3.  **HBM $\rightarrow$ MXU 加载:** 为了执行我们最后的矩阵乘法, 我们需要将这 16e9 字节加上 bf16[8, 128 \* 1024] 数组 (另外 2MB, 所以可以忽略不计) 通过 HBM 带宽加载到 MXU 中, 这将需要 `16e9 / 8.1e11 = 19ms`.

4.  **FLOPs:** 我们总共执行 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7e11$$ FLOPs, 由于我们可以执行 `1.97e14` bf16 FLOPs/s, 我们得到 1.3ms.

总时间的上限是所有这些时间的总和, 但由于 TPU 通常可以重叠这些操作, 我们可以将其视为一个受最慢部分瓶颈的流水线问题. 假设这是真的, 那么答案大约是 150-200 毫秒.

{% enddetails %}

<h3 markdown=1 class="next-section">第二部分到此结束! 第三部分, 涵盖分区和跨 TPU 通信, [点击这里](../sharding).</h3>

## 附录

### 附录 A: 关于 TPU 内部的更多信息

在这里, 我们将更深入地探讨 TPU 的内部操作. 除非另有说明, 我们将提供 TPU v5p 的规格.

### VPU

VPU 是 TPU 的向量算术核心. VPU 由一个二维 SIMD 向量机 (**VPU**) 组成, 该机器执行逐元素算术运算, 如 vadd (向量加法) 或 vmax (逐元素最大值), 以及一组称为 **VREGs** 的向量寄存器, 用于为 VPU 和 MXU 保存数据.

**VREGs:** 每个 TPU v5p 核心有 64 个 32 位 VREGs (TPU v4 中为 32 个), 这给了我们每个核心总共约 `64 * 8 * 128 * 4 = 256kB` 的 VREG 内存 (或者对于整个芯片来说是这个值的 2 倍, 因为我们有两个核心). 一个 TPU v5p 每个周期可以从 VMEM 加载 3 个寄存器, 并每个周期向 VMEM 写入 1 个寄存器.

**VPU:** VPU 是一个形状为 `(8, 128)` 的二维向量算术单元, 其中 128 维度被称为通道轴, 8 维度被称为子通道轴. v5 上的每个 (通道, 子通道) 对包含 4 个相互独立的标准浮点 ALU. VPU 在其每个 ALU 中以一个周期执行大多数算术指令 (如 vadd 或向量加法), 延迟为 2 个周期, 因此例如在 v5 中, 你可以在每个周期内将 4 对来自 VREGs 的 f32 值相加. 一个典型的 VPU 指令可能看起来像 `{v2 = vadd.8x128.f32 v0, v1}`, 其中 v0 和 v1 是输入 VREGs, v2 是输出 VREG.

所有通道和子通道都以纯 SIMD 方式每个周期执行相同的程序, 但每个 ALU 可以执行不同的操作. 所以我们可以例如在一个周期内处理 1 个 vadd 和 1 个 vsub, 每个都对两个完整的 VREGs 进行操作, 并将输出写入第三个.

**小测验 [计算 VPU 吞吐量]:** 使用上述信息, 计算一个 TPU v5p 可以执行多少向量 FLOPs/s. 一个 TPU v5p 的时钟速度约为 1.75GHz.

{% details 点击这里查看答案. %}

*答案*: 每个周期, 每个核心可以在 `8 * 128` 个 ALU 上执行 4 个向量指令. 这给了我们整个芯片 `8 * 128 * 4 * 2` FLOPs/周期, 或 `8 * 128 * 4 * 2 * 1.75e9 = 1.4e13 FLOPs/s`. 请注意, 这比 MXU 的 FLOPs/s (约 `2e14`) 小得多 (大约 10 倍).

{% enddetails %}

**归约:** 通常, 跨子通道维度的通信或归约比跨通道维度的更容易. 例如, VPU 支持一个通道内 shuffle 操作, 可以在大约一个周期内沿着大小为 8 的轴滚动. 这可以用于在子通道维度上执行高效的归约 (只需 shuffle 4, 2 和 1, 并进行 3 对逐元素求和).

跨通道归约要困难得多, 并且涉及一个称为 XLU 或“跨通道单元”的独立硬件单元, 它速度慢且相当昂贵.

**与 GPU 的比较:** 对于熟悉 NVIDIA GPU 的人来说, VPU 中的每个 ALU 都类似于一个 CUDA 核心, 一个 VPU 通道类似于一个“Warp Scheduler”, 即通常执行 SIMD 算术的 32 个 CUDA 核心的集合. 通道内的归约非常容易, 但如果我们需要跨通道, 我们需要至少通过 VMEM/XLU/SMEM, 这要慢得多. 有关更多详细信息, 请参见 [GPU 部分](../gpus).

### 标量核心

标量核心是 TPU 的控制单元. 它获取并分派所有指令, 并执行从 HBM 到 VMEM 的传输, 并且可以被编程为执行标量元数据工作. 因为标量核心是单线程的, 这的一个副作用是 TPU 的每个核心每个周期只能创建一个 DMA 请求.

为了说明这一点, 一个标量核心控制一个 VPU (由 4096 个 ALU 组成), 4 个 MXU, 2 个 XLU 和多个 DMA 引擎. 每个计算单元的控制高度倾斜是硬件效率的来源, 但也限制了以任何有趣的方式进行数据相关向量化的能力.

### 附录 B: 脉动阵列是如何工作的?

TPU MXU 的核心是一个 `128x128` 的脉动阵列 (`256x256` 在 TPU v6e 上). 当完全饱和时, 脉动阵列可以每 8 个时钟周期执行一次 `bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>如果你不熟悉这个符号, 它的意思是: 将一个 `8x128` 的 bfloat16 元素矩阵与一个 `128x128` 的 bfloat16 元素矩阵相乘, 并将结果存储在一个 `8x128` 的 float32 元素矩阵中.</d-footnote> 乘法.

*   在其核心, 脉动阵列是一个 2D `128x128` (`=16,384`) 的 ALU 网格, 每个 ALU 都能执行乘法和加法操作.
*   权重 (**W**, `128x128` 输入) 从上方传入 (称为 RHS), 而输入 (**X**, `8x128` 输入) 从左侧传入 (称为 LHS).

这是一个简化的动画, 显示了一组权重 (蓝色) 与一组激活 (绿色) 的相乘. 你会注意到权重 (RHS) 首先部分加载, 对角线加载, 然后激活也对角线地送入. 在下面的每一帧中, 我们将所有重叠的绿色和蓝色单元相乘, 将结果与从上方传入的任何残差相加, 然后将结果依次向下一个单元传递.

{% include figure.liquid path="assets/img/systolic-array.gif" %}

这是一个更通用的动画版本, 显示了输出从计算中流出:

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

这是一个图表, 显示了这如何在多个 RHS 和 LHS 数组之间进行流水线化:

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

在加载权重 (RHS) 和激活 (LHS) 时, 会有一个初始的流水线气泡. 在那个初始气泡之后, 可以加载新的输入和权重, 而不会产生额外的气泡.

这是一个 bf16[2, 3] x bf16[3, 3] 矩阵乘法的糟糕动画, 你可以想象成一个 2x3 权重矩阵与一个批量为 1, 大小为 3 的输入激活的矩阵乘法. 与前面的幻灯片相比, 这个是旋转的, 输入向右流出而不是向下, 但你大致可以看到结构.

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以有效地将其流水线化以乘以大矩阵, 而不会产生太大的流水线气泡. 话虽如此, 重要的是我们的矩阵的形状要大于 MXU 的边维度, 通常是 128x128. 一些 TPU (自 TPU v3 起) 有多个 MXU, TPU v3 有 2 个, TPU v4/5 有 4 个, 所以我们需要确保分块维度大于 128 * MXU 的数量. [这里](https://www.youtube.com/watch?v=sJltBQ4MOHA) 有一个很好的动画.

Trillium (TPU v6e) 有一个 `256x256` 的脉动阵列, 这意味着它每个周期可以执行 4 倍的 FLOPs. 这也意味着你的张量的维度需要是两倍大才能充分利用 MXU.

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu) 有另一个关于固定权重矩阵的脉动阵列乘法的优秀动画.
