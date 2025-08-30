---
layout: distill
title: "Transformer推理全解析"
# permalink: /main/
description: "Transformer的推理与训练有很大不同。部分原因是推理增加了一个新的考虑因素：延迟。在本节中，我们将从模型采样单个新token开始，一直到作为推理引擎的一部分在多个加速器切片上高效扩展大型Transformer。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "第6部分：训练LLaMA"

next_section_url: ../applied-inference
next_section_name: "第8部分：服务LLaMA"

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
  - name: "Transformer推理基础"
  - subsections:
    - name: "我们真正想要优化什么？"
    - name: "线性操作：什么限制了我们的性能？"
    - name: "注意力机制呢？"
    - name: "LLM延迟和吞吐量的理论估算"
    - name: "内存方面呢？"
    - name: "LLaMA 2-13B的吞吐量和延迟建模"
  - name: "提高生成吞吐量和延迟的技巧"
  - name: "在多加速器上分布式推理"
  - subsections:
    - name: "预填充（Prefill）"
    - name: "生成（Generation）"
    - name: "分片KV缓存"
  - name: "设计高效的推理引擎"
  - subsections:
    - name: "连续批处理"
    - name: "前缀缓存"
    - name: "看看实现：JetStream"
  - name: "练习题"
  - name: "附录"

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

## Transformer推理基础

你已经训练了一个Transformer，现在想用它生成一些新序列。_归根结底，基准测试分数上升和损失曲线下降只是代理指标，真正重要的是当模型实际运行时是否会发生有趣的事情！_<d-footnote>历史上，你可以在不接触推理的情况下进行大量Transformer研究——LLM损失、多项选择基准测试可以在没有适当KV缓存或生成循环实现的情况下高效运行。这意味着，特别是在研究代码库中，推理代码路径中通常有很多低垂的果实。</d-footnote>

采样在概念上很简单。我们输入一个序列，我们喜欢的Transformer会输出$$\log p(\text{下一个token}_i \vert \text{之前的tokens})$$，即所有可能下一个token的对数概率。我们可以从这个分布中采样并获得一个新token。附加这个token并重复这个过程，我们就获得了一个提示的延续token序列。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b>从Transformer进行朴素采样。蓝色logits给我们提供了下一个token的分布，我们可以从中采样。注意每一步都重新处理整个前缀，导致算法的运行时为$\Theta(n^2)$。" %}

我们刚刚描述了Transformer采样的朴素实现，虽然它有效，但**我们在实践中从不这样做**，因为每次生成token时我们都在重新处理整个序列。这个算法在FFW上是$$O(n^2)$$，在注意力机制上是$$O(n^3)$$来生成$$n$$个token！

**我们如何避免这种情况？** 与其每次都进行完整的前向传递，事实证明我们可以保存每次前向传递的一些中间激活，从而避免重新处理之前的token。具体来说，由于给定token在点积注意力期间只关注之前的token，我们可以简单地将每个token的键和值投影写入一个称为**KV缓存**的新数据结构中。一旦我们为过去的token保存了这些键/值投影，未来的token就可以简单地计算它们的$$q_i \cdot k_j$$乘积，而无需在早期token上执行任何新的FLOPs。太棒了！

考虑到这一点，推理有两个关键部分：

* <b style="color: red;">预填充（Prefill）</b>：给定一个长提示，我们同时处理提示中的所有token，并将结果激活（特别是键值投影）保存在**"KV缓存"**中。我们还保存最后一个token的logits。
* <b style="color: blue;">生成（Generation）</b>：给定KV缓存和之前的logits，我们从logits中增量采样一个token，将该token反馈到Transformer中，并为下一步生成一组新的logits。我们还将该新token的KV激活附加到KV缓存中。我们重复这个过程，直到遇到特殊的`<EOS>`token或达到某个最大长度限制。

这是使用KV缓存进行采样的示意图：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b>使用KV缓存进行高效Transformer采样的示意图。<b style=\"color: red;\">预填充</b>处理我们的提示并将所有每个token的键值激活保存在缓存中。<b style=\"color: blue;\">生成</b>获取此缓存（和最后一个token的logits），采样一个新token，并将该新token传递给模型，关注KV缓存并将新token的键值投影保存回缓存。这在MLP块中是一个$O(n)$算法。" %}

通过使用KV缓存进行采样，我们将生成$n$个token的时间复杂度降低到FFW上的$$O(n)$$和注意力上的$$O(n^2)$$，因为我们从不重新处理之前的token。然而，生成序列仍然需要许多前向传递——这就是当你查询Gemini或ChatGPT时结果流回给你时发生的事情。每个token（通常）都是对一个巨大模型的单独（但部分缓存的）Transformer调用。

我们很快就会看到<b style="color: red;">预填充</b>和<b style="color: blue;">生成</b>是非常不同的野兽——Transformer推理是两个伪装的任务！与训练相比，KV缓存也是一个新颖且重要的复杂性来源。

### 我们真正想要优化什么？

在我们继续之前，值得强调推理的一个全新方面：延迟。在训练期间我们只关心吞吐量（每秒处理的总token数），但在推理期间我们必须担心生成token的速度（包括**首token时间（TTFT）**和**每token延迟**）。例如：

* **离线批量推理**用于评估和数据生成只关心推理的批量成本，对单个样本的延迟不敏感。
* **聊天界面/流式任务**需要大规模廉价运行，同时具有低TTFT并生成足够快的token以超过人类阅读速度。
* **边缘推理**（例如笔记本电脑上的`llama.cpp`）只需要以最低延迟服务一个用户，可能面临严重的硬件限制。

最大化硬件利用率仍然至关重要，有助于降低成本和提高TTFT，但与训练不同，它并不*必然*在所有情况下转化为更好的用户体验。在加速器、系统和模型架构层面的许多优化需要在延迟、吞吐量、上下文长度甚至模型质量之间进行权衡。

### 更细粒度的Transformer视图

到目前为止，我们主要将Transformer视为前馈块的堆叠。虽然从FLOPs和内存的角度来看这通常是合理的，但不足以正确建模推理。<d-footnote>在本节中你会注意到的一点是，推理比训练要苛刻得多。我们通常有更少的FLOPs、更少的批处理机会，以及对延迟更高的敏感性。KV缓存也极大地复杂化了推理。</d-footnote> 正如我们在[第4部分](../transformers)中看到的，Transformer前向传递的主要组件是：

1. **一堆线性操作**，包括MLP（$W_{in}$, $W_{out}$）和注意力QKV投影及输出投影（$W_Q$, $W_K$, $W_V$, 和 $W_O$）。这些都涉及从HBM读取参数和一批激活，执行一些FLOPs，然后将结果写回HBM。
2. **点积注意力**。我们需要从HBM读取一批键值投影和一批查询激活，执行一些内积和一些softmax操作，然后将注意力结果写回HBM。
3. **其他所有内容**，包括应用层归一化、激活函数、token采样、更新KV缓存和位置嵌入。这些确实需要一些FLOPs，但被上述操作主导或融合其中。

在接下来的几节中，我们将在预填充和生成的背景下审视每一个组件，并询问什么可能会成为我们性能的瓶颈。在单个加速器内，我们是受计算限制还是受内存限制？我们想强调预填充与生成之间的答案会有多么不同。

### 线性操作：什么限制了我们的性能？

我们所有的线性操作在概念上都是相同的，无论它们位于MLP块还是注意力中。它们的算术强度取决于批大小。我们在[第1节](../roofline)中做过这个数学计算，但值得重复。让我们看一个$\text{bf16[B, D]}$批次与$\text{bf16[D, F]}$矩阵的单个矩阵乘法。这可能是大的MLP块（$W_\text{in}$或$W_\text{out}$）或较小的注意力投影之一（$W_Q$, $W_K$, $W_V$, $W_O$）。要进行这个矩阵乘法，我们需要将这两个数组从HBM加载到MXU中，执行乘法，然后将结果写回HBM。如前所述，我们有：

$$T_\text{math} = \frac{\text{计算FLOPs}}{\text{加速器FLOPs/s}} = \frac{2BDF}{\text{加速器FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{通信字节数}}{\text{带宽字节/s}} = \frac{2BD + 2FD + 2BF}{\text{带宽字节/s}}$$

TPU或GPU可以通过在计算时加载来重叠这些操作，所以要受计算限制，我们需要$$T_\text{math} \geq T_\text{comms}$$，即：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{加速器FLOPs/s}}{\text{带宽字节/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

其中RHS是我们硬件的算术强度。现在假设$D$和$F$相对于$B$非常大（通常我们的批次最多为500，而$D$和$F > 10k$），我们可以使用$\small{2BD + 2DF + 2BF \approxeq 2DF}$来简化分母，这给我们：

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approxeq \frac{2BDF}{2DF} \geq \frac{\text{加速器FLOPs/s}}{\text{带宽字节/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

如果我们对权重进行量化或在矩阵乘法中使用较低精度的FLOPs，这个临界批大小会改变。例如，如果我们将权重量化为int8或fp8，$B_\text{crit}$减少2倍。如果我们在int8或fp8中执行FLOPs，$B_\text{crit}$增加2倍。因此，如果我们让$\beta = \text{每参数位数} / \text{每激活位数}$和$\alpha_\text{hbm} = C / W_\text{hbm}$，我们的临界批大小实际上是$B_\text{crit} = \beta \alpha_\text{hbm}$。

<p markdown=1 class="takeaway">**要点：** Transformer矩阵乘法受计算限制*当且仅当*每个副本的**token**批大小大于$B_\text{crit} = C / W_\text{hbm} \cdot (\text{每参数位数} / \text{每激活位数}) = \beta \cdot \alpha_\text{hbm}$。对于TPU v5e上的bf16激活，这是240个token。对于H100，大约是280个token。</p>

在训练期间，我们所有的矩阵乘法都会具有高强度，因为我们在非常大的批次上重用相同的权重。**这种高算术强度延续到预填充，因为用户提示通常有数百甚至数千个token长。** 正如我们之前看到的，TPUv5e的硬件算术强度是240，因此如果长度超过240个token的序列输入到在此硬件上以bf16运行的密集模型中，我们预计会受计算限制，一切正常。技术上可以将比这更短的提示批处理在一起以实现更高的利用率，但这通常没有必要。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，所有矩阵乘法基本上总是受计算限制。因此，简单地最大化硬件利用率或MFU（模型FLOPs利用率）足以最大化每芯片吞吐量（成本）和延迟（以TTFT形式）。除非提示非常短，否则在每提示级别进行批处理只会增加延迟，而对预填充吞吐量的改进很小。</p>

然而，在生成期间，对于每个请求，我们只能一次一个token地执行前向传递，因为步骤之间存在顺序依赖关系！因此，我们只能（容易地）通过将多个请求批处理在一起，在批次维度上并行化来实现良好的利用率。我们稍后会详细讨论这一点，但实际上在不影响延迟的情况下将许多并发请求批处理在一起是很困难的。因此，**用生成来饱和硬件FLOPs要困难得多。**

<p markdown=1 class="takeaway">**要点：** 在生成期间，总token批大小必须大于$B_{\text{crit}}$才能在线性/前馈操作上受计算限制（对于TPU v5e上的bf16参数为240）。因为生成是串行发生的，逐个token，这要求我们将多个请求批处理在一起，这很困难！</p>

*值得注意的是这有多大！* 生成批大小为240意味着240个并发请求同时生成，以及密集模型的240个独立KV缓存。这意味着这在实践中很难实现，除非在某些批量推理设置中。相比之下，在预填充期间推送超过240个token是相当常规的，尽管随着稀疏性的增加需要一些注意。

**请注意，这个确切数字会根据量化类型和硬件而有所不同。** 加速器通常可以在较低精度下提供更多算术运算。例如，如果我们有int8参数但在bf16中进行计算，临界批大小降至120。使用int8激活和int8参数，它会跳回240，因为TPUv5e可以提供400 TOPs/s的int8 x int8。

### 注意力机制呢？

当我们查看点积注意力操作时，事情变得更加复杂，特别是因为我们必须考虑KV缓存。让我们只看一个具有纯多头注意力的注意力头。在单个Flash Attention融合中，我们<d-footnote>我们在这里做了相当多的简化，忽略了应用softmax、掩码等的非矩阵乘法FLOPs。它们应该与计算或HBM读取重叠，但在某些TPU代上可能不容易做到。这些细节不会改变主要信息，即KV缓存通常受内存限制。</d-footnote>：

1. 从HBM读取形状为$\text{bf16[B, T, D]}$的$Q$激活。
2. 从HBM读取$KV$缓存，这是一对$\text{bf16[B, S, D]}$张量。
3. 在$$QK$$矩阵乘法中执行$2BSTD$ FLOPs。使用Flash Attention，我们不需要将$\text{bf16[B, S, T]}$注意力矩阵写回HBM。
4. 在注意力$$AV$$矩阵乘法中执行$2BSTD$ FLOPs。
5. 将结果$\text{bf16[B, T, D]}$张量写回HBM。

将所有内容放在一起，我们得到：

$$\text{多头注意力算术强度} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充，$S=T$因为我们正在进行自注意力，所以这简化为$T^2 / 2T = T / 2$。这很好，因为这意味着**预填充期间注意力的算术强度是$\Theta(T)$**。这意味着很容易在注意力上受计算限制。只要我们的序列长度相当大，就没问题！

但是由于生成具有平凡的序列维度，并且$B$和$D$维度抵消，我们可以进行近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕，因为这意味着我们无法做任何事情来提高生成期间注意力的算术强度。我们在加载巨大的KV缓存的同时执行极少量的FLOPs。**因此我们在注意力期间基本上总是受内存带宽限制！**

<p markdown=1 class="takeaway">**要点：** 在预填充期间，对于任何合理的序列长度（大约$\gt 480$个token），注意力通常受计算限制，而在生成期间我们的算术强度低且恒定，因此我们总是受内存带宽限制。</p>

*从概念上讲，这是为什么？* 主要是，我们在模型的线性部分受计算限制，因为参数（内存带宽密集型组件）在许多批次项上重用。然而，每个批次项都有自己的KV缓存，因此更大的批大小意味着更多的KV缓存。除非架构被积极调整，否则我们几乎*总是*在这里受内存限制。

这也意味着一旦参数内存变得与KV缓存内存相当，增加批大小对吞吐量的回报就会递减。回报递减对你的影响程度取决于单个序列的参数与KV缓存字节的比率，即大约比率$2DF / SHK$。由于$HK\approx D$，这大致取决于$F$与$S$（序列长度）的比率。这也取决于使KV缓存更小的架构修改（我们稍后会详细说明）。

### LLM延迟和吞吐量的理论估算

从这个数学计算中，我们可以得到优化时应该瞄准的步时相当好的界限。**（注意：如果我们希望读者从整个章节中记住一件事，那就是以下内容）。** 对于生成期间的小批大小（这很常见），我们可以通过假设我们在注意力和MLP块中都受内存带宽限制来下限我们的每步延迟：

$$\begin{equation*}
\text{理论最小步时} = \frac{\text{批大小} \times \text{KV缓存大小} + \text{参数大小}}{\text{总内存带宽}}
\end{equation*}$$

类似地，对于吞吐量：

$$\begin{equation*}
\text{理论最大Tokens/s} = \frac{\text{批大小} \times \text{总内存带宽}}{\text{批大小} \times \text{KV缓存大小} + \text{参数大小}}
\end{equation*}$$

最终，随着我们的批大小增长，FLOPs开始主导参数加载，因此在实践中我们有更一般的方程：

$$\begin{align}
\tiny \text{理论步时（通用）} = \underbrace{\frac{\text{批大小} \times \text{KV缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（总是带宽限制）}} + \underbrace{\max\left(\frac{2 \times \text{批大小} \times \text{参数数量}}{\text{总FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可能受计算限制）}}
\end{align}$$

其中注意力分量（左）从不受计算限制，因此不需要FLOPs上限分析。这些对于粗略计算相当有用，例如：

<b markdown=1 style="color: #57cf57;">随堂测验：</b> 假设我们想在TPU v5e 4x4切片上使用int8权重和bf16 FLOPs，从30B参数密集模型中生成一批4个token，上下文长度为8192，KV缓存为每token 100 kB。这个操作的合理下限是多少？如果我们想采样一批256个token呢？

{% details 点击查看答案。 %}

**答案：** 在int8中，我们的参数将使用30e9字节，根据给定规格，我们的KV缓存将每个使用`100e3 * 8192 = 819MB`。我们有16个芯片，每个芯片有`8.1e11`字节/s的带宽和`1.97e14` bf16 FLOPs/s。根据上述方程，由于我们有一个小批大小，我们预计步时至少为`(4 * 819e6 + 30e9) / (16 * 8.1e11) = 2.5 ms`。在256个token时，我们将完全进入MLP块的计算限制区域，因此步时大约为`(256 * 819e6) / (16 * 8.1e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

{% enddetails %}

如你所见，这里存在吞吐量和延迟之间的明显权衡。小批次快速但不能很好地利用硬件。大批次慢但高效。以下是为一些较旧的PaLM模型计算的延迟-吞吐量帕累托边界（来自[ESTI论文](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>）：

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>Figure:</b> Pareto frontier of cost (read: throughput) versus latency for several PaLM models. Note how chip count (C) and batch size (B) moves you along the Pareto frontier, with the exception of the green dot (C:32 B:16 for PaLM 540B) where the available memory prevented the setup from supporting a good batch size and caused throughput to suffer. Note how throughput generally tends to flatten around after the batch size 240. int8 weights offers a better latency-throughput pareto optimal, but not a better max throughput." %}

我们不仅通过批大小来权衡延迟和吞吐量，还可能更倾向于更大的拓扑结构而不是更小的，这样如果发现自己受HBM限制，我们可以容纳更大的批次。[下一节](../applied-inference)更详细地探讨了这一点。

<p markdown=1 class="takeaway">**要点：** 如果你关心生成吞吐量，请使用尽可能大的每芯片批大小。任何超过TPU算术强度（$B_\text{crit}$，通常为120或240）的每芯片批大小都将最大化吞吐量。你可能需要增加拓扑结构来实现这一点。较小的批大小将允许你以吞吐量为代价来改善延迟。</p>

{% details 从硬件角度来看，这里有一些注意事项。点击这里查看一些细节。 %}

这都相当理论化。在实践中，我们通常不会看到非常明显的性能上限，原因有几个：

* 我们关于HBM读取将与FLOPs完美重叠的假设并不现实，因为我们的编译器（XLA）可能会出错。
* 对于分片模型，XLA也经常无法有效重叠模型分片矩阵乘法的ICI通信与FLOPs本身，因此我们通常在线性操作超过$$\text{BS}=32$$时开始受到延迟影响。
* 由于不完美的重叠，大于理论上限的批大小仍然会看到一些吞吐量改进，但这是一个很好的启发式方法。

{% enddetails %}

### 内存方面呢？

我们花了一些时间研究带宽和FLOPs，但没有研究内存。由于我们新的数据结构KV缓存，推理时的内存情况看起来有很大不同。在本节中，让我们选择一个真实的模型（LLaMA 2-13B）来展示事物看起来有多么不同：

| 超参数             | 值     |
| ------------------ | ------ |
| L (层数)           | 40     |
| D (模型维度)       | 5,120  |
| F (前馈网络维度)   | 13,824 |
| N (头数)           | 40     |
| K (KV头数)         | 40     |
| H (QKV维度)        | 128    |
| V (词表大小)       | 32,000 |

推理期间什么在使用内存？显然，我们的参数。计算这些参数，我们有：

| 参数             | 公式                                                                                                          | 大小（字节）                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 前馈网络参数     | d_model<sup>2</sup> x ffw_multiplier x 3 (用于gelu + 输出投影) x n_layers                                  | 5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9**                       |
| 词表参数         | 2 (输入和输出嵌入) x n_embeddings x d_model                                                         | 2 x 32,000 x 5,120 = **0.3e9**                                 |
| 注意力参数       | [2 (*查询和输出*) x d_model x n_heads x d_qkv + 2 (*用于键和值*) x d_model x n\_kv\_heads x d_qkv] x n_layers | (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = **4.2e9** |

将这些参数相加，我们得到8.5e9 + 4.2e9 + 0.3e9 = **13e9总参数**，正如预期的那样。正如我们在前面章节中看到的，在训练期间，我们可能以bfloat16存储参数，并以float32存储优化器状态。这可能使用大约100GB的内存。这与我们的梯度检查点相比相形见绌，后者可能使用几TB。

**推理有何不同？** 在推理期间，我们存储一份参数副本，比如以bfloat16格式。这使用26GB——在实践中，我们通常可以通过量化做得更好。没有优化器状态或梯度需要跟踪。因为我们不进行检查点（为后向传递保留激活），所以我们的激活占用对于预填充<d-footnote>特别感谢Flash Attention，它避免了物化我们的注意力矩阵</d-footnote>和生成都是可以忽略的。如果我们预填充8k个token，单个激活只使用大约`8,192 x 5,120 x 2字节 = 80MB`的内存。更长的预填充可以分解为许多较小的前向传递，因此对于更长的上下文也不是问题。生成使用的token甚至比这更少，因此激活可以忽略。

**主要区别在于KV缓存**。这些是所有过去token的键和值投影，大小仅受最大允许序列长度的限制。$$T$$个token的总大小为

$$\text{KV缓存大小} = 2 \cdot \text{每个浮点数的字节数} \cdot H \cdot K \cdot L \cdot T$$

其中$$H$$是每个头的维度，$$K$$是KV头的数量，$$L$$是层数，2来自于同时存储键和值。

**这可能很快变得非常大**，即使使用适中的批大小和上下文长度。对于LLaMA-13B，在bf16下一个8192序列的KV缓存为

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{字节}) \times 2 = 6.7 \text{GB}$$

**仅4个这样的缓存就超过了我们参数的内存使用量！** 需要明确的是，LLaMA 2没有针对较长上下文的KV缓存大小进行优化（情况并不总是这么糟糕，因为通常$K$要小得多，比如在LLaMA-3中），但这仍然具有说明性。我们不能在内存或延迟估算中忽略这些。

### LLaMA 2-13B的吞吐量和延迟建模

让我们看看，如果我们尝试在8xTPU v5e上以不同批大小完美高效地执行生成，直到之前推导出的最大理论吞吐量的临界批大小（240），会发生什么。

| 批大小                        |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV缓存内存 (GiB)             |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| 总内存 (GiB)                |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| 理论步时 (ms)        |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| 理论吞吐量 (tokens/s) | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5e为我们提供128GiB的HBM、6.5TiB/s的HBM带宽（每个0.82TiB/s）和1600TF/s的计算能力。

对于这个模型，增加批大小确实能提高吞吐量，但我们会遭受快速递减的回报。超过批大小16后会出现内存不足（OOM），需要增加一个数量级的内存才能接近240。更大的拓扑结构可以改善延迟，但我们在每芯片吞吐量方面遇到了瓶颈。

假设我们保持参数总数不变，但神奇地将KV缓存缩小5倍（例如，使用1:5的[GMQA](#提高生成吞吐量和延迟的技巧)，这意味着我们有8个KV头共享40个查询头——详见下一节）。

| 批大小                        |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV缓存内存 (GiB)             |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| 总内存 (GiB)                |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| 理论步时 (ms)        |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| 理论吞吐量 (tokens/s) | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

使用较小的KV缓存，我们仍然有递减的回报，但每芯片的理论吞吐量继续扩展到批大小240。我们可以容纳更大的64批，并且延迟在所有批大小下也持续更好。延迟、最大吞吐量和最大批大小都显著改善！事实上，后来的LLaMA代次使用了这种确切的优化——LLaMA-3 8B有32个查询头和8个KV头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

<p markdown=1 class="takeaway">**要点：** 除了参数之外，KV缓存的大小对模型的最终推理性能有很大影响。我们希望通过架构决策和运行时优化的组合来控制它。</p>

## 提高生成吞吐量和延迟的技巧

自最初的[Attention is All You Need论文](https://arxiv.org/abs/1706.03762)以来，已经开发了许多技术来使模型更高效，通常特别针对KV缓存。一般来说，较小的KV缓存使得更容易增加生成步骤的批大小和上下文长度而不损害延迟，并使围绕Transformer的系统（如请求缓存）更容易处理。忽略对质量的影响，我们可能会看到：

**分组多头查询注意力（GMQA，GQA）：** 我们可以减少KV头的数量，并在注意力机制中与许多查询头共享它们。在极端情况下，可以在所有查询头之间共享单个KV头。这相对于纯多头注意力（MHA）将KV缓存减少了查询:KV比率的倍数，并且观察到模型的性能对此变化相对不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这也有效提高了注意力计算的算术强度（参见[第4节](../transformers)中的问题4）。

**混合一些局部注意力层：** 局部注意力将上下文限制为中小型最大长度。在训练和预填充期间，这涉及将注意力矩阵掩码为对角线带而不是三角形。这有效地限制了局部层的KV缓存的最大长度。通过在模型中混合一些局部层和一些全局层，在上下文长度超过局部窗口时，KV缓存的大小大大减小。

**跨层共享KV：** 模型可以学习以某种模式跨层共享相同的KV缓存。虽然这确实减少了KV缓存大小，并在增加批大小、缓存、离线存储等方面提供了好处，但共享的KV缓存可能需要从HBM多次读取，*因此不一定改善步时*。

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>左：</b> 多层纯全局注意力。<b>右：</b> 与相邻层共享的全局/局部交错模式示例。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai博客</a>。"%}

**量化：** 推理通常对参数和KV的精度不太敏感。通过对参数和KV缓存进行量化（例如到int8、int4、`fp8`等），我们可以在两者上节省内存带宽，减少达到计算上限所需的批大小，并节省内存以运行更大的批大小。量化还有一个额外优势，即使模型没有经过量化训练，通常也可以在训练后应用。

**使用不规则HBM读取和分页注意力：** 我们在上面的计算中为每个KV缓存分配了8k上下文，但通常不需要从内存中读取整个KV缓存——请求具有广泛的长度分布，并且不使用模型的最大上下文，因此我们通常可以实现仅读取KV缓存的非填充部分的内核（例如Flash Attention变体）。

分页注意力<d-cite key="paged"></d-cite>是对此的改进，它将KV缓存在操作系统风格的分页表中存储，并大多完全避免填充KV缓存。这增加了许多复杂性，但意味着每个批次只使用所需的内存。这是一个运行时优化，因此再次与架构无关。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图：</b> 在生成期间，单个token（第四个）关注多个KV缓存块/页。通过对KV缓存进行分页，我们避免加载或存储比需要更多的内存。取自<a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention论文</a>。" %}

<p markdown=1 class="takeaway">**大局观：** 总的来说，这些KV缓存优化可以将KV缓存大小减少一个数量级以上，相比于标准的MHA Transformer。这可以导致Transformer总体成本的数量级改进。</p>

## 在多加速器上分布式推理

到目前为止，我们一直在回避如何扩展到单个芯片之外的问题。遵循[第5节](../training)，让我们探索可用的不同策略及其权衡。一如既往，我们将分别查看预填充和生成。

### 预填充（Prefill）

从性能上限的角度来看，**预填充几乎与训练相同**，几乎所有相同的技术和权衡都适用——模型（Megatron）并行、序列分片（对于足够长的上下文）、流水线，甚至FSDP都是可行的！你只需要保留KV以便稍后进行生成。与训练一样，增加芯片数量使我们能够访问更多的FLOPs/s（可能降低TTFT），但增加了通信开销（可能降低每芯片吞吐量）。

**预填充分片的一般规则：** 这是预填充的一般规则集。我们假设我们只在单个序列上进行预填充（无批维度）：

1. *模型分片：* 我们通常首先进行一定量的模型并行，直到达到ICI限制。正如我们在[第5节](../training)中看到的，对于1轴，这大约是$F / 2200$（通常约为4-8路分片）。
2. *序列并行：* 除此之外，我们进行序列并行（类似于数据并行，但在序列维度上进行分片）。虽然序列并行在注意力中引入了一些额外的通信，但在较长的上下文中通常相当小。与训练一样，我们可以重叠通信和计算（分别使用集体矩阵乘法进行Megatron和环形注意力）。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，几乎所有在训练期间可以工作的分片都可以正常工作。进行模型并行直到ICI限制，然后进行序列并行。</p>

### 生成（Generation）

生成是比预填充更复杂的野兽。一方面，更难获得大的批大小，因为我们需要将许多请求批处理在一起。延迟目标更低。这些意味着我们通常更受内存限制，对通信开销更敏感，这限制了我们的分片策略：

1. **FSDP是不可能的：** 由于我们在从HBM加载参数和KV缓存到MXU时受内存限制，我们不希望通过ICI移动它们，这比HBM慢几个数量级。*我们希望移动激活而不是权重。* 这意味着类似于FSDP的方法通常完全不可用于生成。<d-footnote>在训练后意外保留它是导致数量级回归的简单而常见的方式</d-footnote>

2. **没有理由进行数据并行：** 纯数据并行没有帮助，因为它复制了我们的参数，并不能帮助我们更快地加载参数。你最好启动模型的多个副本。<d-footnote>我们的意思是，以较小的批大小启动具有模型副本的多个服务器。模型级别的数据并行严格来说更差。</d-footnote>

3. **没有序列 = 没有序列分片。** 祝你好运进行序列分片。

_这主要留给我们用于密集模型生成的模型分片变体_。与预填充一样，我们能做的最简单的事情是简单的模型并行（激活完全复制，权重在MLP的隐藏维度上完全分片），直到我们达到ICI限制的4-8路。然而，由于我们经常受内存带宽限制，我们实际上可以超越这个限制来改善延迟！

**关于生成的ICI限制的说明：** 在训练期间，我们希望受计算限制，因此我们的性能上限分析关注的是当我们的ICI通信时间超过FLOPs时。然而，在生成期间，如果我们受参数加载的内存带宽限制，我们可以增加模型分片超越这一点，并以最小的吞吐量成本改善延迟。更多的模型分片为我们提供了更多的HBM来加载权重，而我们的FLOPs无关紧要。<d-footnote>在FLOPs时间不是我们的瓶颈的意义上，所以我们需要担心的是ICI时间超过参数加载时间。</d-footnote> 让我们看看在它成为瓶颈之前我们可以进行多少模型并行。

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中$\beta = W_\text{hbm} / W_\text{ici}$。对于TPU v5e和TPU v6e，这个数字通常在8左右。这意味着例如，如果$F$是16,384且$B$是32，理论上我们可以进行多达`16384 / (32 * 8) = 64`路的模型并行，而不会对吞吐量产生有意义的影响。这假设我们可以完全分片我们的KV缓存64路，这很困难：我们在下面讨论这一点。

对于注意力层，我们还以Megatron风格在头上模型分片注意力$$W_Q$$和$$W_O$$。KV权重相当小，复制它们通常比分片超过$K$路更便宜。

<p markdown=1 class="takeaway">**要点：** 在生成期间，我们唯一的选择是模型并行的变体。我们的目标是移动激活而不是KV缓存或参数，后者更大。当我们的批大小很大时，我们进行模型并行直到FLOPs-ICI限制（$F / \alpha$）。当我们的批大小较小时，我们可以通过更多的模型分片来改善延迟（以适度的吞吐量成本为代价）。当我们想要进行比KV头数更多的模型分片时，我们也可以沿批维度分片我们的KV。</p>

### 分片KV缓存

**我们还有一个需要分片的额外数据结构——KV缓存。** 同样，我们几乎总是倾向于避免复制缓存，因为它是注意力延迟的主要来源。为了做到这一点，我们首先沿头维度对KV进行Megatron分片。这限制为$K$路分片，所以对于头数较少的模型，我们尽可能地分片头维度，然后沿批次维度分片，即$\text{KV}[2, B_Z, S, K_Y, H]$。这意味着KV缓存被完全分布。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>Figure:</b> comparison of the attention mechanism with (a) Multi head attention with pure model sharding and (b) Multiquery attention with batch sharding of the KV cache. Notice how we need two extra AllToAlls to shift the activations from model sharding to batch sharding, so they can act on the KV caches." %}

这样做的代价是每个注意力层需要两次AllToAll——一次将Q激活转移到批次分片，以便我们可以用批次分片计算注意力，另一次将批次分片的注意力输出转移回纯模型分片。

{% details 这是完整的算法！ %}

这里我们将写出在$Y$和$Z$上都有模型并行的完整注意力算法。很抱歉对键张量和KV头维度都使用了$K$。令$M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ... (现有激活，来自前一层的非分片)
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K, H] = ... (现有KV缓存，批次分片)
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这相当复杂，但你可以大致看出它是如何工作的。新的通信相对便宜，因为它们作用于我们的小激活，而作为回报，我们在加载KV（它们是静态的）时节省了大量内存带宽。

</div>

{% enddetails %}

* **序列分片：** 如果批次大小太小，或者上下文很长，我们可以对KV缓存进行序列分片。同样，我们在这里需要付出集合通信的代价来跨分片累积注意力。首先我们需要AllGather Q激活，然后以类似于Flash Attention的方式累积KV。

## 设计高效的推理引擎

到目前为止，我们已经研究了如何分别优化和分片预填充和生成操作。为了实际有效地使用它们，我们需要设计一个推理引擎，可以在延迟/吞吐量帕累托边界上我们选择的点来提供这两个操作。

最简单的方法就是运行一批预填充，然后运行一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>图：</b>在最简单的设置中，请求被聚合，服务器在运行一批预填充和调用生成函数之间交替，直到所有序列完成。" %}

这很容易实现，是大多数代码库中的第一个推理设置，但它有多个缺点：

1. **延迟非常糟糕。** 我们将预填充和生成批次大小耦合在一起。在大预填充批次大小下，首token时间（TTFT）非常糟糕——你需要在任何用户看到任何token之前完成所有预填充。在小批次大小下，生成吞吐量非常糟糕。
2. **我们用较长的生成阻塞较短的生成。** 许多序列会在其他序列之前完成，在生成期间留下空的批次槽位，进一步损害生成吞吐量。随着批次大小和生成长度的增加，问题会加剧。
3. **预填充被填充。** 预填充被填充到最长的序列，我们浪费了大量计算。有解决方案可以解决这个问题，但历史上XLA使得跳过这些FLOPs相当困难。随着批次大小和预填充序列长度的增加，这再次变得更糟。
4. **我们被迫在预填充和生成之间共享分片。** 预填充和生成都位于同一个切片上，这意味着我们对两者使用相同的拓扑和分片（除非你保留两份权重副本），这通常对性能无益，例如生成需要更多的模型分片。

因此，这种方法仅推荐用于边缘应用（通常只关心服务单个用户和使用FLOPs/字节较少的硬件）以及在Transformer代码库生命周期早期的快速迭代（由于其简单性）。

一个稍微更好的方法是在批次大小为1时执行预填充（此时受计算限制但具有合理的延迟），但在生成期间将多个请求批处理在一起：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这将避免批处理预填充造成的TTFT浪费，同时保持高生成吞吐量。我们称之为**交错**配置，因为我们"交错"预填充和生成步骤。这对于批量生成应用（如评估，其中吞吐量是主要目标）非常强大。编排器可以配置为在任何生成槽位打开时优先进行预填充，即使对于非常大的生成批次大小也能确保高利用率。我们还可以避免将预填充填充到最大长度，因为它不与另一个请求批处理。

主要缺点是当服务器执行预填充时，所有其他请求的生成都会暂停，因为所有计算资源都将被预填充消耗。用户A的响应正忙于解码，将被用户B的预填充阻塞。这意味着即使TTFT有所改善，token生成平均来说也会抖动和缓慢，这对于许多应用来说不是良好的用户体验——其他用户的预填充位于请求整体延迟的关键路径上。

为了解决这个问题，我们分离解码和预填充。虽然Transformer推理可以在一个服务器上完成，但从延迟的角度来看，通常在两组TPU/GPU上执行这两个不同的任务更好。预填充服务器生成KV缓存，通过网络发送到生成服务器，生成服务器将多个缓存批处理在一起并为每个缓存生成token。我们称之为**"解耦式"**服务。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

This provides a few advantages:

1. **Low latency at scale**: A user's request never blocks on another user's, except if there is insufficient prefill capacity. The request should be immediately prefilled, then sent to the generation server, then immediately slotted into the generation buffer. If we expect many concurrent requests to come in, we can scale the number of prefill servers independently from the number of generate servers so users are not left in the prefill queue for an extended period of time.

2. **Specialization:** Quite often, the latency-optimal parameter sharding strategy/hardware topology for prefill and generate is quite different (for instance, more model parallelism is useful for generate but not prefill). Constraining the two operations to use the same sharding hurts the performance of both, and having two sets of weights uses memory. Also, by moving prefill onto its own server, it doesn't need to hold any KV caches except the one it's currently processing. That means we have a lot more memory free for history caching (see the next section) or optimizing prefill latency.

One downside is that the KV cache now needs to be shifted across the network. This is typically acceptable but again provides a motivation for reducing KV cache size.

<p markdown=1 class="takeaway">**Takeaway:** for latency-sensitive, high-throughput serving, we typically have to separate prefill and generation into separate servers, with prefill operating at batch 1 and generation batching many concurrent requests together.</p>

### 连续批处理（Continuous batching）

上述问题（2）催生了**连续批处理**的概念。我们优化并编译：

* 多个具有可变上下文长度的预填充函数，并将其插入到某个KV缓冲区中，具有最大批大小和上下文长度/页数。
* 一个生成函数，接收KV缓存，并为所有当前活跃的请求执行生成步骤。

然后我们将这些函数与一个编排器结合，该编排器对传入请求进行排队，根据可用的生成槽位调用预填充和生成，处理历史缓存（参见下一节）并流式传输token。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存（Prefix caching）

由于预填充昂贵且受计算限制（给我们留下的余量较少），减少其成本的最佳方法之一是减少预填充量。因为LLM是自回归的，查询["I", "like", "dogs"]和["I", "like", "cats"]在前两个token中产生相同的KV缓存。这意味着，原则上，如果我们先计算"I like dogs"缓存，然后计算"I like cats"缓存，我们只需要做1/3的计算。我们可以通过重用缓存来节省大部分工作。这在几个特定情况下特别强大：

1. **聊天机器人**：大多数聊天机器人对话涉及严格附加到自身的来回对话。这意味着如果我们可以保存每个对话轮的KV缓存，我们可以跳过除最新token之外的所有计算。
2. **少样本提示**：如果我们有任何类型的少样本提示，这可以免费保存和重用。系统指令通常也具有这种形式。

唯一难以做到这一点的原因是内存限制。正如我们所看到的，KV缓存很大（通常许多GB），并且要使缓存有用，我们需要将它们保留到后续查询到达。通常，预填充服务器上任何未使用的HBM可以用于本地缓存系统。此外，加速器通常在CPU主机上有大量内存（例如，8xTPUv5e服务器有128GiB的HBM，但大约有450GiB的主机DRAM）。这种内存比HBM慢得多——通常太慢而无法执行生成步骤——但对于缓存读取来说足够快。在实践中：

* 因为KV缓存在处理初始请求的TPU集合中是本地的，我们需要某种形式的亲和性路由来确保后续查询到达同一个副本。这可能导致负载均衡问题。
* 较小的KV缓存再次有帮助——它使我们能够在相同数量的空间中保存更多KV缓存，并减少读取时间。
* KV缓存及其查找可以很自然地存储在树或字典树中。驱逐可以基于LRU（最近最少使用）进行。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图：</b> 实现为LRU字典树的KV前缀缓存。我们可以通过共享前缀来避免复制KV内存。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai博客</a>。" %}

### 看看实现：JetStream

Google开源了一个实现此逻辑的库，称为[JetStream](https://github.com/google/JetStream)。服务器有一组"预填充引擎"和"生成引擎"，通常在不同的TPU切片上，由单个控制器编排。预填充发生在"[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)"中，而生成发生在"[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)"中。我们还有一个"[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)"，用于编排将KV缓存从预填充切片复制到生成切片。

Engine接口（在此[实现](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)）是任何LLM必须提供的通用接口。关键方法是：

* **prefill**：接收一组输入token并生成KV缓存。
* **insert**：接收KV缓存并将其插入到生成正在处理的KV缓存批次中。
* **generate**：接收一组批处理的KV缓存，并为每个批次项生成一个token，将单个token的KV缓存附加到每个token的解码状态中。

我们还有一个PyTorch版本的JetStream，可在此处[获取](https://github.com/google/jetstream-pytorch)。

## 练习题

我将基于LLaMA-2 13B为本节创建一个新模型。以下是详细信息：

| 超参数             | 值     |
| :----------------- | :----- |
| L (层数)           | 64     |
| D (模型维度)       | 4,096  |
| F (前馈网络维度)   | 16,384 |
| N (头数)           | 32     |
| K (KV头数)         | 8      |
| H (QKV维度)        | 256    |
| V (词表大小)       | 32,128 |

**问题1：** 上述模型有多少参数？在int8下，其每个token的KV缓存有多大？*你可以假设我们共享输入和输出投影矩阵。*

{% details 点击查看答案。 %}

**参数数量：**

* MLP参数数量：$L * D * F * 3$
* 注意力参数数量：$L * 2 * D * H * (N + K)$
* 词表参数：$D * V$（因为我们共享这些矩阵）

我们的总参数数量因此为$L * D * (3F + 2H * (N + K)) + D * V$。代入上述数字，我们有`64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。因此，该模型约有184亿参数。

KV缓存为每个token $L * K * H$，即`64 * 8 * 256 = 131kB`每个token。

{% enddetails %}

**问题2：** 假设我们想在TPUv5e 4x4切片上服务此模型，并且可以在此拓扑上完全分片我们的KV缓存。假设我们对所有内容使用int8并希望支持128k序列，我们可以容纳的最大批大小是多少？如果我们将KV头数减少到1呢？

{% details 点击查看答案。 %}

我们的KV缓存在int8中每边为$L * K * H$，即`64 * 8 * 256 = 131kB`。对于128k序列，这意味着每个批次项`131e3 * 128e3 = 16.8GB`。由于每个TPU有16GB的HBM，包括我们的参数，我们可以容纳的最大批大小为`(16 * 16e9 - 18.4e9) / 16.8e9 = 14`。如果我们有$K=1$，我们将有8倍于此，即约112。

{% enddetails %}

**问题3：** 假设参数在TPU v5e 4x4切片上完全分片，将所有参数从HBM加载到MXU需要多长时间？假设int8参数。*这是每步延迟的一个良好下限。*

{% details 点击查看答案。 %}

我们有总共18.4B参数，或int8中的18.4e9字节。每个芯片有8.1e11 HBM带宽，因此大约需要`18e9 / (8.1e11 * 16) = 1.3ms`，假设我们可以完全使用我们的HBM带宽。

{% enddetails %}

**问题4：** 假设我们想在TPUv5e 4x4切片上使用int8 FLOPs和参数/激活来服务此模型。我们将如何为预填充和解码分片？*提示：也许先回答这些问题：*

1. 4x4上的ICI是什么样的？
2. 张量并行的性能上限界限是什么？
3. 我们如何分片KV缓存？

对于这种分片，生成的每步延迟大约是多少？

**问题5：** 假设上述模型实际上是一个MoE（混合专家模型）。MoE模型实际上是一个密集模型，具有E个FFW块的副本。每个token通过k个FFW块，这些`k`被平均以产生输出。让我们使用`E=16`和`k=2`以及上述设置。

1. 它有多少总参数和激活参数？*激活参数指任何给定token使用的参数。*
2. 在TPU v5e上需要多大的批大小才能受计算限制？
3. 每个token的KV缓存有多大？
4. 具有T个token的前向传递涉及多少FLOPs？

{% details 点击查看答案。 %}

(1) 作为MoE，每个MLP块现在有$3 * E * D * F$参数，比密集变体增加$E$倍。因此现在它有$L * D * (3EF + 2H * (N + K)) + D * V$或`64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9`总参数，增加约12倍。对于激活参数，我们有$k$而不是$E$激活参数，总共`64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，比密集变体增加不到2倍。

(2) 因为我们有$E$倍多的参数但只有$k$倍多的FLOPs，我们的HBM性能上限增加了$E/k$倍。这意味着在TPU v5e上我们需要大约`240 * (16 / 2) = 1920`个token。

(3) KV缓存大小保持不变，因为MoE特性不会改变注意力机制的任何内容。

(4) 这仍然是$2ND$，其中$D$是激活参数数量。因此这是$2 * \text{32.2e9} * T$。

{% enddetails %}

**问题6：** 对于MoE，我们可以进行"专家分片"，即沿网格的一个轴分割我们的专家。在我们的标准表示法中，我们的第一个FFW权重具有形状`[E, D, F]`，我们将其分片为[E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中`X`仅在训练期间用作我们的FSDP维度。假设我们想在TPU v5e上进行推理：

1. 上述模型在TPU v5e 8x16切片上使用Y=8，Z=16的HBM权重加载时间是多少？每个TPU有多少可用HBM？
2. 我们可以容纳模型的最小切片是多少？

**问题7 [2D模型分片]：** 这里我们将详细研究[ESTI论文](https://arxiv.org/pdf/2211.05102)中称为2D权重静态分片的数学。我们在附录B中简要描述了这一点，但先尝试解决这个问题，看看你是否能算出数学。2D权重静态分片的基本思想是沿$D$和$F$轴分片我们的权重，使每个块大致为正方形。这减少了通信负载，使我们能够稍微扩展得更远。

这是2D权重静态的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U.X} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U.X})
4.  Out[B, D<sub>X</sub>] {U.YZ} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W2[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U.YZ})
</div>

你的目标是计算此算法的$T_\text{math}$和$T_\text{comms}$，并找出它何时会优于传统的3D模型分片？

{% details 点击查看答案！ %}

让我们计算$T_\text{math}$和$T_\text{comms}$。我们所有的FLOPs都是完全分片的，因此如前所述，我们有$T_\text{math} = 4BDF / (N \cdot C)$，但我们的通信现在是

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

我们注意到AllReduce的成本是两倍，并且我们通过执行每个操作的轴数来缩放我们的通信。假设我们可以自由选择拓扑，并且假设$F=4D$（如在LLaMA-2中），我们声称（通过一些基本微积分）$X$、$Y$和$Z$的最优值是$X = \sqrt{N / 8}$，$YZ = \sqrt{8N}$，因此总通信为

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，从上面复制，正常的1D模型并行将具有$T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$，那么新的通信何时更小？我们有

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 81
\end{align*}$$

对于一般的$F$，我们声称这个条件是

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

所以这告诉我们，如果我们有超过81个芯片，使用这个新方案会更好。这是一个稍微奇怪的结果，因为我们在历史上发现自己在约20路张量并行时受ICI限制。但在这里，即使我们受通信限制，我们的总通信随着总芯片数的增加而继续减少！这告诉我们，我们可以不断增加芯片，增加批大小，进行更多参数缩放，并看到延迟减少。

{% enddetails %}

<h3 markdown=1 class="next-section">这就是第7部分的全部内容！对于第8部分，看看我们如何在TPU上服务LLaMA 3，请点击[这里](../applied-inference)。</h3>

## 附录

### 附录A：批大小 > 240 规则有多真实？

我们上面提供的简单规则——批大小必须大于240个token才能受计算限制——大致正确，但忽略了TPU在某些操作（如设备间通信）未使用所有可用HBM时预取权重的能力。

这是一个小型Transformer的层时间（微秒）经验图，其d<sub>model</sub>为8192，d<sub>ff</sub>为32768，每层只有2个矩阵乘法。这来自[这个Colab笔记本](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。你会看到步时在批大小约240之前增加非常缓慢，然后线性增加。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

这是实际的吞吐量（token/微秒）。这相当清楚地说明了论点。由于我们的层在这里大约有6亿参数，分片为4路，我们预计最小延迟约为365微秒。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

所以至少在这个模型中，我们确实看到吞吐量增加到每个数据并行分片约BS240。

### 附录B：2D权重静态分片

随着拓扑结构的增长，如果我们能够访问更高维度的网格（如TPU的网格），可以通过引入第二个分片轴来进一步完善这一点，称为"**2D权重分片**"。我们称之为"**2D权重静态**"，在[高效扩展Transformer推理论文](https://arxiv.org/abs/2211.05102)中有更详细的描述。

由于我们在Megatron中只分片隐藏$$F$$维度，一旦芯片数量随着1D分片变得很大，它可能变得显著小于$$E$$（$$d_\text{model}$$维度）。这意味着在较大的批大小下，在MLP的第一层应用后，在隐藏维度上执行部分集合操作可能更经济。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

此图显示：

1. 1D权重静态分片，即纯Megatron分片，其中激活在AllGather后完全复制，权重在隐藏F维度上完全分片。
2. 2D权重静态分片，其中权重在隐藏F和归约E维度上分片，激活在E维度上分片。我们在第一层之前在(yz)轴上执行AllGather，然后在(x)轴上执行ReduceScatter。

对于注意力层，Megatron风格的分片对于较少数量的芯片也相对简单。然而，Megatron在$$n_\text{heads}$$维度上进行，这限制了可能的分片量。修改2D分片（不是分片隐藏维度，而是分片$$n_\text{heads}$$维度），我们获得了进一步扩展的能力。

### 附录C：延迟限制的通信

作为回顾，在[第3节](../sharding)中，我们推导了在具有全双工带宽WICI和延迟Tmin的1D环链路上，在X个芯片上执行大小为B的张量的AllGather所需的时间。

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

对于大的B，挂钟时间保持相对恒定，因为当你向系统添加更多芯片时，你同时扩展了执行操作所需的数据移动量和总可用带宽。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于在延迟优化推理期间移动的数据量相对较少，激活上的集合操作通常受延迟项限制（特别是对于小批大小）。通过计算我们需要完成之前所需的跳数，可以很容易地可视化延迟。

在TPU上，如果通信的张量大小相关部分每跳小于1微秒（一跳是两台相邻设备之间的通信），我们可能受实际分派集合操作的固定开销的限制。使用`4.5e10`单向ICI带宽，当$$(\text{字节数} / n_\text{分片数}) / 4.5e10 < 1e-6$$时，ICI通信变得受延迟限制。对于8路Megatron分片，当`buffer_size < 360kB`时会发生这种情况。**这在推理期间实际上并不算小：**使用`BS=16`和`D=8192`的int8，我们的激活将使用`16*8192=131kB`，所以我们已经受延迟限制。

<p markdown=1 class="takeaway">**要点：**当$$\text{总字节数} < W_{ICI} \times 1e-6$$时，我们的通信变得受延迟限制。例如，在$$Y$$上进行模型并行时，当$$Y > BD / 45,000$$时，我们在int8中变得受限制。</p>

这里可以与计算性能上限进行类比——我们承担了一些小操作的固定成本（通信的延迟，矩阵乘法的内存带宽）。

### 附录D：推测采样（Speculative Sampling）

当我们*真正*关心端到端延迟时，可以使用一个额外的技巧，称为推测采样<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。作为回顾，我们通常逐个token地从大型Transformer生成：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

使用推测采样，我们使用一个更小、更便宜的模型生成token，然后用大模型检查结果。这在*贪婪解码*中最容易理解：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 我们从某个更小、更便宜的模型中贪婪采样。理想情况下，我们使用经过训练以匹配较大模型的模型，例如通过蒸馏，但也可以简单到使用n-gram或匹配小型文本语料库的token。
2. 生成K个token后，我们使用大模型计算到目前为止生成的所有token的下一个token logits。
3. 由于我们进行贪婪解码，我们可以只检查较小模型生成的token是否在所有可能token中具有最高概率。如果其中一个token错误，我们取最长的正确前缀，并用正确token替换第一个错误token，然后返回(1)。如果所有token都正确，我们可以使用最后一个正确logit在返回(1)之前采样一个额外的token。

**为什么这是延迟上的胜利？** 此方案仍然需要我们为每个token执行相当于大模型一次前向传递的FLOPs，但因为我们可以将一堆token批处理在一起，我们可以在一次前向传递中完成所有这些FLOPs，并利用我们*不受计算限制*的事实来免费评分更多token。

每个接受的token在FLOPs方面平均变得更昂贵（因为有些会被拒绝，我们必须调用草稿模型），但我们从硬件中挤出更多FLOPs，而且小模型很便宜，所以总体上我们获胜。我们还在多个步骤中共享KV缓存加载，因此**推测解码对于长上下文也可以是吞吐量上的胜利。** 由于一切都经过大模型检查，我们完全不会改变采样分布（尽管对于非贪婪情况，确切轨迹会有所不同）。

传统上，推测解码依赖于存在一个与目标模型具有相似采样分布的较小模型，例如LLaMA-2 2B用于LLaMA-2 70B，这通常不存在。即使可用，如果接受率低，较小的草稿模型仍然可能太昂贵。相反，将草稿模型嵌入主模型内部可能会有所帮助，例如通过向基础模型的后期层添加专用草稿头<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>。由于此头与主模型共享大部分参数，因此运行速度更快，并且更紧密地匹配采样分布。

对于正常的自回归采样，token/s与步时相同。我们仍然受制于算术强度部分的理论最小步时（事实上，推测采样步时通常比正常自回归采样慢得多，但因为平均每步获得超过1个token，我们可以获得更好的token/s）。

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图：</b>此图显示了Chinchilla（来自DeepMind的70B模型）与4B参数草稿（小模型）的每步延迟和推测成功率。对于XSum（自然语言数据集），理想的推测量约为提前3-4个token，而HumanEval（编码数据集）更可预测，并从更积极的推测中看到收益。" %}

**这对于非贪婪解码如何工作？** 这有点复杂，但基本上归结为受Metropolis-Hastings启发的算法，其中我们有$$P_{\text{草稿模型}}(\text{选择的token})$$和$$P_{\text{目标模型}}(\text{选择的token})$$源自logits，如果这些概率的比率小于某个阈值，则概率性地拒绝选择的token。

这两篇[论文](https://arxiv.org/abs/2211.17192)[论文](https://arxiv.org/abs/2302.01318)同时推导了这一点，并提供了关于这在实践中如何工作的良好示例。

<p markdown=1 class="takeaway">**要点：** 推测采样是另一个强大的杠杆，用于以吞吐量为代价换取更好的每token延迟。然而，在批大小受限的情况下（例如，小的硬件占用空间或大的KV缓存），它变成了双赢。</p>
