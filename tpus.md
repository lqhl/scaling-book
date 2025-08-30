---
layout: distill
title: "如何理解TPU"
# permalink: /main/
description: "本节全部关于TPU如何工作，它们如何联网以实现多芯片训练和推理，以及这如何影响我们最喜欢的算法的性能。甚至还有一些对GPU用户有用的好东西！"
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
  - name: What Is a TPU?
  - name: TPU Networking
  - name: Key Takeaways
  - subsections:
    - name: TPU Specs
  - name: Worked Problems
  - name: Appendix
  - subsections:
    - name: "Appendix A: More on TPU internals"
    - name: "Appendix B: How does a systolic array work?"

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

<p markdown=1 class="announce">You might also enjoy reading the new [Section 12](../gpus) on NVIDIA GPUs!</p>

## 什么是TPU？

**TPU基本上是一个专门用于矩阵乘法的计算核心（称为TensorCore），连接到一堆快速内存（称为高带宽内存或HBM）<d-cite key="tpu_paper"></d-cite>。** 这是一个图表：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>Figure:</b> the basic components of a TPU chip. The TensorCore is the gray left-hand box, containing the matrix-multiply unit (MXU), vector unit (VPU), and vector memory (VMEM)." %}

您可以将TensorCore基本上视为一个非常好的矩阵乘法机器，但它还有一些其他值得注意的功能。TensorCore有三个关键单元：

* **MXU**（矩阵乘法单元）是TensorCore的核心。对于大多数TPU代，它使用脉动阵列每8个周期执行一次`bfloat16[8,128] @ bf16[128,128] -> f32[8,128]`矩阵乘法<d-footnote>TPU v6e（Trillium）具有256x256 MXU，而所有前几代使用128x128</d-footnote>（详见<a href="#appendix-b-how-does-a-systolic-array-work">附录B</a>）。
  * 这在TPU v5e上以1.5GHz每MXU大约`5e13` bf16 FLOPs/s。大多数TensorCore有2或4个MXU，所以例如TPU v5e的总bf16 FLOPs/s是`2e14`。
  * TPU还支持更低精度的矩阵乘法，具有更高的吞吐量（例如，每个TPU v5e芯片可以做`4e14` int8 OPs/s）。

* **VPU**（向量处理单元）执行一般的数学运算，如ReLU激活或向量之间的逐点加法或乘法。归约（求和）也在这里执行。<a href="#appendix-a-more-on-tpu-internals">附录A</a>提供了更多详细信息。
* **VMEM**（向量内存）是位于TensorCore内的片上暂存器，靠近计算单元。它比HBM小得多（例如，TPU v5e上为128 MiB），但到MXU的带宽要高得多。VMEM的运作方式有点像CPU上的L1/L2缓存，但要大得多且由程序员控制。HBM中的数据需要先复制到VMEM中，然后TensorCore才能对其进行任何计算。

**TPU在矩阵乘法方面非常非常快。** 这是它们主要做的工作，而且做得很好。迄今为止最强大的TPU之一[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)可以达到每核`2.5e14` bf16 FLOPs或每芯片`5e14` bf16 FLOPs。一个包含8960个芯片的pod可以达到每秒4 exaflops。这是*非常*大的算力。这是世界上最强大的超级计算机之一。而且谷歌有很多这样的设备。<d-footnote>TPU，特别是它们的脉动阵列，是如此强大的硬件加速器，因为矩阵乘法是少数几个使用$O(n^3)$计算量来处理$O(n^2)$字节数据的算法之一。这使得普通的ALU很容易被计算能力限制，而不是内存带宽限制。</d-footnote>

上面的图表还包括一些其他组件，如SMEM和标量单元，它们用于控制流处理，并在<a href="#appendix-a-more-on-tpu-internals">附录A</a>中简要讨论，但对于理解来说不是关键。另一方面，HBM很重要且相当简单：

* **HBM**（高带宽内存）是一大块快速内存，用于存储TensorCore使用的张量。HBM的容量通常在几十GB的量级（例如，[TPU v5e有16GiB的HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。

  * 当需要计算时，张量通过VMEM从HBM流式传输到MXU，结果从VMEM写回HBM。

  * HBM和TensorCore之间（通过VMEM）的带宽被称为"HBM带宽"（通常约为1-2TB/s），这限制了内存受限工作负载的计算速度。

**通常，所有TPU操作都是流水线化和重叠的。** 要执行矩阵乘法 $X \cdot A \to Y$，TPU首先需要将矩阵$A$和$X$的块从HBM复制到VMEM，然后将它们加载到MXU中，MXU将8x128的块（$X$）和128x128的块（$A$）相乘，然后将结果逐块写回HBM。为了高效执行，矩阵乘法是流水线化的，使得与VMEM之间的复制操作与MXU工作重叠。这使得MXU可以继续工作，而不是等待内存传输，保持矩阵乘法受计算限制，而不是内存限制。

以下是在TPU上执行逐点乘积的示例：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>Figure:</b> an animation showing a pointwise product performed on TPU, with bytes loaded from HBM. Note how bytes are streamed out of memory in chunks and partial results are pipelined back without waiting for the full array to be materialized." %}

矩阵乘法看起来几乎相同，只是它会加载到MXU而不是VPU/Vector单元，并且加载和存储的顺序不同，因为相同的权重块用于多个激活块。你可以看到数据块流式传输到VMEM，然后到VREGs（向量寄存器），然后到Vector Unit，再回到VMEM和HBM。正如我们即将看到的，如果从HBM到VMEM的加载速度比Vector Unit（或MXU）中的FLOPs慢，我们就会成为"带宽受限"，因为我们让VPU或MXU缺少工作。

<p markdown=1 class="takeaway">**Key takeaway:** TPUs are very simple. They load weights from HBM into VMEM, then from VMEM into a systolic array which can perform around 200 trillion multiply-adds per second. The HBM $\leftrightarrow$ VMEM and VMEM $\leftrightarrow$ systolic array bandwidths set fundamental limits on what computations TPUs can do efficiently.</p>

**VMEM和计算强度：** VMEM比HBM小得多，但它到MXU的带宽要高得多。正如我们在[第1节](../roofline)中看到的，这意味着如果一个算法可以将其所有输入/输出都放入VMEM中，它就很少会遇到通信瓶颈。当计算的计算强度较差时，这特别有用：VMEM带宽比HBM带宽高约22倍，这意味着从VMEM读取/写入的MXU操作只需要10-20的计算强度就能达到峰值FLOPs利用率。这意味着如果我们能将权重放入VMEM而不是HBM，我们的矩阵乘法可以在更小的批次大小下达到FLOPs限制。这也意味着那些本质上具有较低计算强度的算法仍然可以高效。VMEM太小，这通常是一个挑战。<d-footnote>我们有时会谈论VMEM预取，这指的是提前在VMEM中加载权重，这样我们可以掩盖矩阵乘法加载的成本。例如，在普通的Transformer中，我们有时可以在注意力计算期间将大的前馈权重加载到VMEM中，如果我们受到内存带宽限制，这可以隐藏权重加载的成本。这要求我们的权重足够小或足够分片，以便将单个层放入VMEM并有剩余空间。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**TPU芯片通常（但不总是）由两个共享内存的TPU核心组成，可以被视为一个具有两倍FLOPs的大型加速器**（称为"megacore"配置）。自TPU v4以来一直是如此。较老的TPU芯片有独立的内存，被视为两个独立的加速器（TPU v3及更早版本）。像TPU v5e这样的推理优化芯片每个芯片只有一个TPU核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**被安排成**每4个一组在'tray'上**，通过PCIe网络连接到**CPU主机**。这是大多数读者熟悉的格式，通过Colab或单个TPU-VM暴露4个芯片（8个核心，但通常被视为4个逻辑megacore）。对于像TPU v5e这样的推理芯片，我们每个主机有2个tray，而不是1个，但每个芯片也只有1个核心，给我们8个芯片=8个核心。<d-footnote>在Cloud TPU VM上，每个tray作为单独VM的一部分暴露，所以再次只有4个核心可见。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe带宽有限：** 像HBM $\leftrightarrow$ VMEM链接一样，CPU $\leftrightarrow$ HBM PCIe连接有特定的带宽，限制了从主机内存到HBM或反之的加载速度。例如，TPU v4的PCIe带宽是每个方向16GB/秒，因此比HBM慢近100倍。我们*可以*将数据加载/卸载到主机（CPU）RAM，但速度不是很快。

## TPU网络

**芯片通过Pod中的ICI网络相互连接**。在较老的一代（TPU v2和TPU v3）、推理芯片（例如TPU v5e）和Trillium（TPU v6e）中，ICI（"芯片间互连"）连接4个最近的邻居（边缘链接形成2D环面）。TPU v4和TPU v5p连接到最近的6个邻居（形成3D环面）。注意这些连接**不**通过它们的主机，它们是芯片之间的直接链接。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环形结构将任意两个节点之间的最大距离从$N$减少到$N / 2$，使通信快得多。TPU还具有"扭曲环面"配置，以Mobius条带拓扑包装环面，以进一步减少节点之间的平均距离。

**TPU pod（通过ICI连接）可以变得非常大：** 最大pod大小（称为**superpod**）对于TPU v4是`16x16x16`，对于TPU v5p是`16x20x28`。这些大型pod由可重新配置的`4x4x4`芯片立方体组成，通过[光学环绕链接](https://arxiv.org/pdf/2208.10041)连接<d-footnote>光学开关只是具有相同ICI带宽的可重新配置连接。它只是让我们在保留环绕链接的同时连接立方体。</d-footnote>，我们可以重新配置来连接非常大的拓扑。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以请求较小的拓扑（例如`2x2x1`、`2x2x2`），但没有环绕连接。这是一个重要的警告，因为它通常会使大多数通信时间加倍。任何完整立方体的倍数（例如`4x4x4`或`4x4x8`）都将具有由光学开关提供的环绕连接。<d-footnote>注意`2x2x4`不会有任何环绕连接，因为它们是由光学开关提供的，而光学开关只在完整立方体上可用。然而，TPU v5e 8x16在长轴上*确实*有环绕连接，因为它不使用可重新配置的光学网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e和Trillium pod由单个`16x16` 2D环面组成，在大小为16的任何轴上都有环绕连接（意味着`8x16`在长轴上有环绕连接）。TPU v5e和v6e（Trillium）不能扩展到16x16环面之外，但pod仍然可以通过标准数据中心网络（DCN）相互通信，DCN将TPU主机相互连接。同样，可以在维度$<16$的情况下请求较小的拓扑，但没有环绕连接。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接是TPU和GPU之间的关键区别。** GPU通过开关层次结构连接，这些开关近似于每个GPU之间的点对点连接，而不是像TPU那样使用本地连接。通常，节点内的GPU（H100为8个GPU，B200最多为500个）直接连接，而较大的拓扑需要在每个GPU之间进行O(log(N))跳。一方面，这意味着GPU可以在单次低延迟跳内在节点内发送任意数据。另一方面，TPU价格低得多（因为NVLink开关昂贵），布线更简单，并且可以扩展到更大的拓扑，因为每个设备的链接数和每个设备的带宽是恒定的。更多信息请阅读[这里](../gpus#networking)。

**ICI相对于DCN非常快，但仍然比HBM带宽慢。** 例如，[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)具有：

* 每芯片`2.5e12`字节/秒（2.5 TB/s）的HBM带宽。
* 每轴`9e10`字节/秒（90 GB/s）的ICI带宽，每个芯片有3个轴。<d-footnote>上面的页面列出了100 GB/s的带宽，与此处列出的略有不同。TPU ICI链接根据执行的操作具有略有不同的带宽。你可以放心使用本文档中的数字。</d-footnote>
* 每个TPU `6.25e9`字节/秒（6.25 GB/s）的DCN（出口）带宽（通过每个主机上的1-2个NIC）。<d-footnote>TPU v6e有12.5e9字节/秒，v5e有3.125e9字节/秒。</d-footnote>

这意味着当我们在多个芯片之间分割模型时，需要小心避免用较慢的跨设备通信使MXU成为瓶颈。

**多slice训练：** 一组通过ICI连接的TPU被称为**slice**。不同的slice可以使用DCN相互连接，例如连接不同pod上的slice。由于DCN是比ICI慢得多的连接，应该尽量限制我们的计算需要等待来自DCN的数据的时间。DCN是主机到主机的，因此要通过DCN将缓冲区从TPU传输到TPU，我们首先需要通过PCIe传输到主机，然后通过网络出口，然后通过目标主机网络入口，然后通过PCIe进入HBM。

## 关键要点

* TPU很简单，在大多数情况下可以被视为一个矩阵乘法单元，连接到内存（超快），通过ICI连接到其他芯片（相当快），并通过DCN连接到数据中心的其余部分（有些快）。

* 通信受到我们各种网络带宽的限制，按速度排序：
  * HBM带宽：TensorCore与其关联的HBM之间。
  * ICI带宽：TPU芯片与其最近的4个或6个邻居之间。
  * PCIe带宽：CPU主机与其关联的芯片tray(s)之间。
  * DCN带宽：多个CPU主机之间，通常是不通过ICI连接的主机。

* **在slice内，TPU只通过ICI连接到它们的最近邻居。** 这意味着在slice内通过ICI进行远距离芯片之间的通信需要首先经过中间芯片的跳转。

* **权重矩阵需要在两个维度上至少填充到128大小**（TPU v6上为256）以填满MXU（实际上，较小的轴被填充到128）。

* **较低精度的矩阵乘法往往更快。** 对于支持它的代，TPU可以比bfloat16 FLOPs快约2x/4x地执行int8或int4 FLOPs。VPU操作仍然在fp32中执行。

* 为了避免TPU计算单元成为瓶颈，我们需要**确保每个通道的通信量与其速度成比例**。

### TPU规格

以下是我们的芯片的一些具体数字：

| 型号                                      | Pod大小 | 主机大小 | 每芯片HBM容量 | 每芯片HBM带宽(字节/秒) | 每芯片FLOPs/s (bf16) | 每芯片FLOPs/s (int8) |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.1e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |

主机大小指的是连接到单个主机的TPU拓扑（例如，TPU v5e有一个CPU主机连接到8个TPU，采用4x2拓扑）。以下是互连数据：

| 型号       | ICI带宽/链接(单向, 字节/秒) | ICI带宽/链接(双向, 字节/秒) |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |              1e11              |            2e11             |
| **TPU v4p** |             4.5e10             |            9e10             |
| **TPU v5p** |              9e10              |           1.8e11            |
| **TPU v5e** |             4.5e10             |            9e10             |
| **TPU v6e** |              9e10              |           1.8e11            |

我们包括单向（单向）带宽和双向（双向）带宽，因为单向带宽更符合硬件实际，但双向带宽在涉及完整环的方程中更常见。<d-footnote>通过双向（双向）带宽，我们的意思是可以在单个链接的两个方向上发送的总字节数，或者等效地，从单个TPU沿特定轴发出的出站字节总数，假设我们可以有效地使用两个链接。当我们有一个正常工作的环时，即当我们在特定轴上有环绕连接时，这是正确的。这在推理芯片上当我们有完整的16轴时发生，或者在训练芯片（v*p）上当我们有4的倍数的轴时发生。我们更喜欢使用双向带宽，因为它在涉及双向通信的计算中经常出现。</d-footnote>

PCIe带宽通常约为每TPU `1.6e10` 字节/秒（TPU v6e为`3.2e10`），而DCN带宽通常约为每TPU `6.25e9` 字节/秒（TPU v6e为`12.5e9`，TPU v5e为`3.125e9`）。

## 工作问题

这些数字有点枯燥，但它们让你可以为模型性能做出基本的roofline估计。让我们解决几个问题来解释为什么这很有用。你将在第3部分看到更多示例。

**问题1 [LLM延迟边界]：** 假设你想从一个200B参数的bf16模型中采样，该模型分布在32个TPU v4p上。将所有参数从HBM加载到脉动阵列需要多长时间？*提示：使用上面的数字。*

{% details Click here for the answer. %}

**答案：** 我们在32个芯片上加载`sizeof(bf16) * 200e9 = 400e9`字节，意味着每芯片12.5e9字节，每个芯片的HBM带宽为1.23e12。所以加载大约需要10ms。

这很酷，因为*这是从模型采样的延迟的一个合理下限*。每个采样步骤需要从HBM加载所有参数，所以它不能少于10ms。在实践中，在小批次大小下，这接近可实现的目标。

{% enddetails %}

**问题2 [TPU细节]：** 考虑一个完整的TPU v5e pod。总共有多少个CPU主机？有多少个TPU TensorCores？整个pod的总FLOPs/s是多少？总HBM是多少？对TPU v5p pod做同样的练习。

{% details Click here for the answer. %}

**答案：** 对于TPU v5e，每个pod是`16x16`，每个主机是4x2 slice，所以我们有`16*16 / 8 = 32`个主机。对于TPU v5e，每个TPU只有一个核心，所以我们有256个TensorCores。总FLOPs/s是`16*16*2e14 = 5.1e16`（bfloat16）。每个芯片有16GB的HBM，所以那是`256 * 16 = 4TB`的内存。

对于完整的TPU v5p pod，我们有`16x20x28`个芯片，每个主机是2x2x1，所以我们有`16*20*28 / 2*2 = 2,240`个主机。对于TPU v5p，每个TPU有两个TensorCores，所以我们有`8960 * 2 = 17,920`个核心。总FLOPs/s是`8960 * 4.5e14 = 4e18`（bfloat16）。每个芯片有96GB的HBM，所以那是`8960 * 96 = 860TB`的内存。

{% enddetails %}

**问题3 [PCIe操作强度]：** 想象我们被迫在主机DRAM中存储一个大的权重矩阵$A$，类型为$\text{bfloat16}[D, F]$，以及一批激活$x$，类型为$\text{bfloat16}[B, D]$，并希望对它们进行矩阵乘法。这在单个主机上运行，我们使用连接到它的单个TPU v6e芯片。你可以假设$B \ll D$，且$F = 4D$（我们将在未来的章节中看到为什么这些是合理的假设）。我们需要多大的最小批次大小$B$才能在PCIe上保持FLOPs限制？假设PCIe带宽为1.5e10字节/秒。

{% details Click here for the answer. %}

**答案：** 我们必须执行$2BDF$浮点运算，每个芯片可以执行`9.2e14`浮点运算每秒。这需要$2BDF / 9.2e14$秒来执行。我们必须从DRAM加载$2DF + 2BD$字节，并将$2BF$字节写回它。我们受到PCIe传输速度的限制，所以我们需要$2 \cdot (BD + DF + BF) / 1.5e10$秒来将数据传输到TPU和从TPU传输数据。由于我们希望计算比权重加载花费更长时间，假设我们可以将所有权重加载与计算重叠，我们想要$2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$。我们可以使用我们的假设$B \ll D$和$F = 4D$来简化这个，得到

$$\frac{8BD^2}{9.2e14} > \frac{8D^2}{1.5e10}$$

或者

$$B > \frac{9.2e14}{1.5e10} \simeq 61,000$$

{% enddetails %}

**问题4 [一般矩阵乘法延迟]：** 假设我们想将一个权重矩阵int8[16384, 4096]与一个大小为int8[B, 4096]的激活矩阵相乘，其中B是某个未知的批次大小。假设我们开始时在1个TPUv5e上。

1. 这个乘法需要多长时间作为B的函数？*提示：计算从HBM加载数组需要多长时间以及乘法实际需要多长时间可能会有所帮助。什么是限制你的瓶颈？*
2. 如果我们想从VMEM运行这个操作怎么办？作为B的函数需要多长时间？

{% details Click here for the answer. %}

**答案：** (1) 我们需要执行的浮点运算数量是$2 \cdot 4096 \cdot 16384 \cdot B = 1.3e8 \cdot B$。所以$T_{\text{math}} = (1.3e8 \cdot B) / 3.94e14$秒。我们需要从HBM到VMEM加载$16384 \cdot 4096 + 4096 \cdot B$字节，并从VMEM到HBM写回$16384 \cdot B$字节。这意味着$T_{\text{comms}} = (6.7e7 + 2e4\cdot B) / 8.1e11$秒。假设尽可能多的通信和计算重叠，整个乘法将大约需要

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{6.7e7 + 2e4\cdot B}{8.1e11}, \frac{1.3e8 \cdot B}{3.94e14}\right\}$$

当$\frac{6.7e7 + 2e4\cdot B}{8.1e11} < \frac{1.3e8 \cdot B}{3.94e14}$时，我们将受到FLOPs限制，或者等效地，$B > 271$。这比我们下面推导的240数字稍大，因为我们考虑了$$D$$和$$F$$的完整影响。

(2) 如果我们从VMEM加载，让我们考虑VMEM到MXU的带宽是HBM $\leftrightarrow$ VMEM带宽的22倍。这将我们的数据加载分母从8.1e11变为1.78e13，我们得到$B > 11$。注意在实践中，我们不能将所有VMEM带宽专用于加载$W$，所以实际上它将更接近20。

{% enddetails %}

**问题5 [ICI带宽]：** 假设我们有一个TPU v5e `4x4` slice。假设我们想从`TPU{0,0}`发送一个类型为`bfloat16[8, 128, 8192]`的数组到`TPU{3, 3}`。假设TPU v5e的每跳延迟为$1\mu s$。

1. 第一个字节何时到达目的地？
2. 总传输需要多长时间？

{% details Click here for the answer. %}

**答案：** 在TPUv5e中我们有2D连接性。因为我们只有一个`4x4` slice（没有大小为16的轴），我们没有环绕连接。因此我们的目标芯片有两个端口可以接收数据，同样我们的源芯片有两个端口可以发送数据。我们需要传输的数据量是`2 * 8 * 128 * 8192 = 1.7e7`字节。我们可以同时从两个端口传输（即发送一半数组向右，一半向下），所以我们得到`2 * 4.5e10 = 9e10`字节每秒的传输速度，这意味着传输整个数组大约需要`1.7e7 / 9e10 = 188us`（假设我们受到带宽限制）。在`4x4` slice中，芯片$(0, 0)$和$(3, 3)$之间有六跳，因为对于少于16个芯片的轴没有环绕链接。由于每跳的延迟约为$1\mu s$，第一个字节将在大约`6us`内到达，总传输将需要`188us`。

{% enddetails %}

**问题6 [综合运用，困难]：** 想象你有一个大矩阵**A**：`int8[128 * 1024, 128 * 1024]`均匀分布在TPU v5e 4x4 slice上，但卸载到每个芯片的主机DRAM上。假设你想将整个数组复制到TPU{0, 0}并将其与向量`bf16[8, 128 * 1024]`相乘。这需要多长时间？*提示：使用上面的数字。*

{% details Click here for the answer. %}

**答案：** 让我们从概述我们必须执行的操作开始。我们的数组大约是16GB。从上面的表格中，TPU v5e主机有4x2拓扑，所以4x4有2个主机，因此，由于我们的数组是均匀分片的，每个主机有效地包含数组的1/2，即8GB。我们需要将这些块全部复制到TPU{0,0}，这给了我们两个选择：

1. 我们可以通过DCN复制，然后通过PCIe将整个未分片的数组加载到HBM中。
2. 我们可以将分片的数组加载到它们对应的TPU上，然后通过ICI执行gather，然后在TPU{0,0}上执行矩阵乘法。

应该清楚选项(2)更好。DCN与ICI相比很慢，我们更愿意通过许多PCIe链接加载大数组，而不是只有几个（主机0上的8个）。这是系统部分的图表。如上所述，注意TPU通过ICI连接到它们的邻居（即使跨主机），所有TPU都连接到它们的主机CPU（通过PCIe），主机通过DCN连接。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="Each chip actually has its own PCIe link to its host, though for clarity only one is shown here." %}

现在让我们计算每个部分需要多长时间：

1. **PCIe加载**：我们在16个PCIe链接上加载16GB / 2 = 8GB的块，每个链接有`1.5e10`字节/秒的带宽。因此这将需要大约33ms。

2. **ICI复制**：每个TPU现在有我们数组的16GB / 16 = 1GB。我们的ICI带宽是每链接9e10字节/秒*双向*，你会从上面的图表注意到，在这个拓扑中，TPU v5e上的4个ICI链接中只有2个用于TPU{0,0}。由于TPU{0,0}需要沿着2个轴接收总共15GB，在`4.5e10`字节/秒/链接，我们可以将时间下限设为`15e9 / (4.5e10 * 2) = 167ms`。在实践中这可能无法实现，因为负载非常不均匀，但它可能在2倍以内。正如你在第2节中看到的，执行完整的AllGather也将大约需要`16e9 / (4.5e10 * 2)`，所以这接近最优。

3. **HBM $\rightarrow$ MXU加载**：为了执行最终的矩阵乘法，我们需要通过HBM带宽将这些16e9字节加上bf16[8, 128 \* 1024]数组（另外2MB，可忽略）加载到MXU中，这将需要`16e9 / 8.1e11 = 19ms`。

4. **FLOPs**：我们总共执行$$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7e11$$ FLOPs，由于我们可以执行`1.97e14` bf16 FLOPs/s，我们得到1.3ms。

总时间的上限是所有这些时间的总和，但由于TPU通常可以重叠这些操作，我们可以将其视为一个流水线问题，瓶颈是最慢的部分。假设这是真的，那么答案大约是150-200ms。

{% enddetails %}

<h3 markdown=1 class="next-section">第2部分就到这里！关于第3部分，涵盖分区和跨TPU通信，[点击这里](../sharding)。</h3>

## 附录

### 附录A：更多关于TPU内部的信息

在这里我们将更深入地探讨TPU的内部操作。除非另有说明，我们将提供TPU v5p的规格。

### VPU

VPU是TPU的向量算术核心。VPU由一个二维SIMD向量机器（**VPU**）组成，执行逐元素算术运算，如vadd（向量加法）或vmax（逐元素最大值），以及一组称为**VREGs**的向量寄存器，用于保存VPU和MXU的数据。

**VREGs：** 每个TPU v5p核心有64个32位VREGs（TPU v4中有32个），给我们每核心总共约`64 * 8 * 128 * 4 = 256kB`的VREG内存（或整个芯片的2倍，因为我们有两个核心）。TPU v5p每个周期可以从VMEM加载3个寄存器，每个周期向VMEM写入1个寄存器。

**VPU：** VPU是一个形状为`(8, 128)`的二维向量算术单元，其中128维度被称为lane轴，8维度被称为sublane轴。v5上的每个(lane, sublane)对包含4个独立的浮点ALU。VPU在每个ALU中用1个周期执行大多数算术指令（如vadd或向量加法），延迟为2个周期，因此例如在v5中，你可以在每个周期中将4对f32值从VREGs相加。典型的VPU指令可能看起来像`{v2 = vadd.8x128.f32 v0, v1}`，其中v0和v1是输入VREGs，v2是输出VREG。

所有lane和sublane每个周期以纯SIMD方式执行相同的程序，但每个ALU可以执行不同的操作。因此我们可以例如在单个周期中处理1个vadd和1个vsub，每个操作在两个完整的VREGs上操作，并将输出写入第三个。

**快速测验[计算VPU吞吐量]：** 使用上述信息，计算TPU v5p可以执行多少向量FLOPs/s。TPU v5p的时钟速度约为1.75GHz。

{% details Click here for the answer. %}

*答案*：每个周期，每个核心可以在`8 * 128`个ALU上执行4个向量指令。这给我们整个芯片`8 * 128 * 4 * 2` FLOPs/周期，或`8 * 128 * 4 * 2 * 1.75e9 = 1.4e13 FLOPs/s`。注意这比MXU的约`2e14` FLOPs/s小多少（大约10倍）。

{% enddetails %}

**归约：** 通常，跨越sublane维度的通信或归约比跨越lane维度更容易。例如，VPU支持一个intra-lane shuffle操作，可以在大约一个周期内沿着大小为8的轴滚动。这可以用来沿着sublane维度执行高效的归约（只需按4、2和1shuffle，并进行3对逐元素求和）。

跨lane归约要困难得多，涉及一个单独的硬件单元，称为XLU或"cross lane unit"，它很慢且相当昂贵。

**与GPU的比较：** 对于熟悉NVIDIA GPU的人来说，VPU中的每个ALU类似于CUDA核心，单个VPU lane类似于"Warp Scheduler"，即通常执行SIMD算术的32个CUDA核心的集合。lane内的归约相当容易，但如果我们需要跨lane，我们需要至少通过VMEM/XLU/SMEM，这要慢得多。更多细节请参见[GPU部分](../gpus)。

### 标量核心

标量核心是TPU的控制单元。它获取和调度所有指令，并执行从HBM到VMEM的传输，并且可以被编程来做标量元数据工作。由于标量核心是单线程的，这的一个副作用是每个TPU核心每个周期只能创建一个DMA请求。

将此放在上下文中，单个标量核心控制一个VPU（由4096个ALU组成）、4个MXU、2个XLU和多个DMA引擎。每个单位计算的控制高度倾斜是硬件效率的来源，但也限制了以任何有趣的方式进行数据依赖向量化的能力。

### 附录B：脉动阵列如何工作？

TPU MXU的核心是一个`128x128`脉动阵列（TPU v6e上为`256x256`）。当完全饱和时，脉动阵列可以每8个时钟周期执行一次`bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>如果你不熟悉这个符号，它的意思是：将一个具有bfloat16元素的`8x128`矩阵与一个具有bfloat16元素的`128x128`矩阵相乘，并将结果存储在具有float32元素的`8x128`矩阵中。</d-footnote>乘法。

* 在其核心，脉动阵列是一个二维`128x128`（`=16,384`）的ALU网格，每个ALU能够执行乘法和加法操作。
* 权重（**W**，`128x128`输入）从上方传递下来（称为RHS），而输入（**X**，`8x128`输入）从左侧传入（称为LHS）。

这是一个简化的动画，展示了一组权重（蓝色）与一组激活（绿色）的乘法。你会注意到权重（RHS）首先被部分加载，对角线地，然后激活也被对角线地输入。在下面的每一帧中，我们将所有重叠的绿色和蓝色单元相乘，将结果与从上方传入的任何残差相加，然后将结果依次向下传递一个单元。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

这是这个动画的一个更通用版本，显示输出从计算中流式传输出来：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

这是一个图表，显示如何在多个RHS和LHS数组之间进行流水线化：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

当权重（RHS）和激活（LHS）被加载时，存在一个初始的流水线气泡。在那个初始气泡之后，新的输入和权重可以被加载而无需额外的气泡。

这是一个bf16[2, 3] x bf16[3, 3]矩阵乘法的糟糕动画，你可以将其想象为2x3权重矩阵与批次1大小3的输入激活的矩阵乘法。这与前面的幻灯片相比是旋转的，输入向右流动而不是向下，但你可以大致看到结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以有效地流水线化这个来乘大矩阵，而不会有太大的流水线气泡。话虽如此，重要的是我们的矩阵形状大于MXU的侧维度，通常是128x128。一些TPU（自TPU v3以来）有多个MXU，TPU v3有2个，TPU v4/5有4个，所以我们需要确保平铺维度大于128 * MXU数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA)有一个很好的动画来展示这个。

Trillium (TPU v6e)有一个`256x256`脉动阵列，这意味着它可以执行4倍多的FLOPs/周期。这也意味着你的张量维度需要是两倍大才能充分利用MXU。

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)有另一个关于固定权重矩阵的脉动阵列乘法的优秀动画。