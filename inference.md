---
layout: distill
title: "关于 Transformer 推理的一切"
# permalink: /main/
description: "对 Transformer 进行推理可能与训练非常不同. 部分原因在于推理增加了一个需要考虑的新因素: 延迟. 在本节中, 我们将从模型中采样一个新 token 开始, 一直到作为推理引擎的一部分, 在多个加速器切片上高效扩展大型 Transformer."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "Part 6: Training LLaMA"

next_section_url: ../applied-inference
next_section_name: "Part 8: Serving LLaMA"

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
  - name: "Transformer 推理的基础知识"
  - subsections:
    - name: "我们到底想优化什么?"
    - name: "线性操作: 我们的瓶颈是什么?"
    - name: "注意力呢?"
    - name: "LLM 延迟和吞吐量的理论估计"
    - name: "内存呢?"
    - name: "为 LLaMA 2-13B 建模吞吐量和延迟"
  - name: "提高生成吞吐量和延迟的技巧"
  - name: "在多个加速器上分布推理"
  - subsections:
    - name: "预填充"
    - name: "生成"
    - name: "分片 KV 缓存"
  - name: "设计一个有效的推理引擎"
  - subsections:
    - name: "连续批处理"
    - name: "前缀缓存"
    - name: "让我们看一个实现: JetStream"
  - name: "已解决的问题"
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

## Transformer 推理的基础知识

所以你已经训练了一个 Transformer, 并且你想用它来生成一些新的序列. _归根结底, 基准分数上升和损失曲线下降只是衡量一旦付诸实践是否会发生有趣事情的代理指标!_<d-footnote>从历史上看, 你可以在不接触推理的情况下对 Transformer 进行惊人数量的研究 —— LLM 损失, 多项选择基准可以在没有适当的 KV 缓存或生成循环实现的情况下高效运行. 这意味着, 特别是在研究代码库中, 推理代码路径中通常有很多唾手可得的成果.</d-footnote>

采样在概念上很简单. 我们输入一个序列, 我们最喜欢的 Transformer 会吐出 $$\log p(\text{下一个 token}_i \vert \text{之前的 token})$$, 即所有可能的下一个 token 的对数概率. 我们可以从这个分布中采样并获得一个新的 token. 附加这个 token 并重复这个过程, 我们就得到了一个作为提示延续的 token 序列.

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图:</b> 从 Transformer 进行朴素采样. 蓝色的 logits 给了我们一个关于下一个 token 的分布, 我们可以从中采样. 请注意, 每个步骤都会重新处理整个前缀, 导致算法的运行时间为 $\Theta(n^2)$." %}

我们刚刚描述了 Transformer 采样的朴素实现, 虽然它有效, **但我们在实践中从不这样做**, 因为我们每次生成一个 token 都会重新处理整个序列. 这个算法在 FFW 上是 $$O(n^2)$$, 在注意力机制上是 $$O(n^3)$$, 以生成 $$n$$ 个 token!

**我们如何避免这种情况?** 与其每次都进行完整的前向传播, 事实证明我们可以保存每次前向传播的一些中间激活, 从而避免重新处理之前的 token. 具体来说, 由于给定的 token 在点积注意力期间只关注之前的 token, 我们可以简单地将每个 token 的键和值投影写入一个名为 **KV 缓存**的新数据结构中. 一旦我们为过去的 token 保存了这些键/值投影, 未来的 token 就可以简单地计算它们的 $$q_i \cdot k_j$$ 乘积, 而无需对早期的 token 执行任何新的 FLOPs. 太棒了!

考虑到这一点, 推理有两个关键部分:

*   <b style="color: red;">预填充</b>: 给定一个长提示, 我们同时处理提示中的所有 token, 并将结果激活 (特别是键值投影) 保存在一个**“KV 缓存”**中. 我们还保存了最后一个 token 的 logits.
*   <b style="color: blue;">生成</b>: 给定一个 KV 缓存和之前的 logits, 我们从 logits 中增量地采样一个 token, 将该 token 反馈给 Transformer, 并为下一步生成一组新的 logits. 我们还将该新 token 的 KV 激活附加到 KV 缓存中. 我们重复这个过程, 直到遇到一个特殊的 `<EOS>` token 或达到某个最大长度限制.

这是一个带 KV 缓存的采样图:

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图:</b> 使用 KV 缓存的高效 Transformer 采样图. <b style=\"color: red;\">预填充</b>处理我们的提示并将其所有每个 token 的键值激活保存在缓存中. <b style=\"color: blue;\">生成</b>获取此缓存 (和最后一个 token 的 logits), 采样一个新 token, 并将该新 token 通过模型, 关注 KV 缓存并将新 token 的键值投影保存回缓存. 这是 MLP 块中的一个 $O(n)$ 算法." %}

通过使用 KV 缓存进行采样, 我们将生成 $n$ 个 token 的时间复杂度在 FFW 上降低到 $$O(n)$$, 在注意力上降低到 $$O(n^2)$$, 因为我们从不重新处理之前的 token. 然而, 生成一个序列仍然需要许多次前向传播 —— 这就是当你查询 Gemini 或 ChatGPT 并且结果流式传输给你时发生的事情. 每个 token (通常) 都是对一个巨大模型的单独 (但部分缓存的) Transformer 调用.

我们很快就会看到, <b style="color: red;">预填充</b>和<b style="color: blue;">生成</b>是截然不同的野兽 —— Transformer 推理是伪装成两个任务! 与训练相比, KV 缓存也是一个新颖且重要的复杂性来源.

### 我们到底想优化什么?

在继续之前, 值得强调一下推理中一个全新的方面: 延迟. 虽然在训练期间我们只关心吞吐量 (每秒**每个芯片**处理的总 token 数), 但在推理期间, 我们必须担心我们生成 token 的速度 (包括**首个 token 的时间 (TTFT)** 和**每个 token 的延迟**). 例如:

*   用于评估和数据生成的**离线批量推理**只关心推理的批量成本, 而不关心单个样本的延迟.
*   **聊天界面/流式任务**需要在规模上廉价运行, 同时具有较低的 TTFT, 并且生成 token 的速度要快于人类的阅读速度.
*   **边缘推理** (例如, 你笔记本电脑上的 `llama.cpp`) 只需要以尽可能低的延迟一次为一个用户提供服务, 可能会有严格的硬件限制.

最大化硬件利用率仍然至关重要, 有助于降低成本和 TTFT, 但与训练不同, 它*不一定*在所有情况下都能转化为更好的用户体验. 加速器, 系统和模型架构级别的许多优化都在延迟, 吞吐量, 上下文长度甚至模型质量之间进行权衡.

### Transformer 的更精细视图

到目前为止, 我们主要将 Transformer 视为一堆前馈块. 虽然从 FLOPs 和内存的角度来看, 这通常是合理的, 但它不足以正确地建模推理.<d-footnote>在本节中, 你会注意到推理比训练要苛刻得多. 我们通常有更少的 FLOPs, 更少的批处理机会, 以及对延迟的敏感度要高得多. KV 缓存也极大地复杂化了推理.</d-footnote> 正如我们在[第 4 部分](../transformers)中看到的, Transformer 前向传播的主要组成部分是:

1.  **一堆线性操作**, 包括 MLP ($W_{in}$, $W_{out}$) 和注意力 QKV 投影和输出投影 ($W_Q$, $W_K$, $W_V$, 和 $W_O$). 这些都涉及从 HBM 读取参数和一批激活, 进行一些 FLOPs, 并将结果写回 HBM.
2.  **点积注意力**. 我们需要从 HBM 读取一批键值投影和一批查询激活, 进行一些内积和一些 softmax 操作, 并将注意力结果写回 HBM.
3.  **其他一切**, 包括应用层归一化, 激活函数, token 采样, 更新 KV 缓存和位置嵌入. 这些确实需要一些 FLOPs, 但被上述操作主导或融合到其中.

在接下来的几节中, 我们将结合预填充和生成来看待这些操作, 并询问什么可能成为我们性能的瓶颈. 在单个加速器内, 我们是受计算限制还是受内存限制? 我们想强调一下, 对于预填充和生成, 答案会有多大的不同.

### 线性操作: 我们的瓶颈是什么?

我们所有的线性操作在概念上都是相同的, 无论它们是在 MLP 块还是在注意力中. 它们的算术强度取决于批量大小. 我们在[第 1 节](../roofline)中做过这个数学计算, 但值得重复一下. 让我们看一个 $\text{bf16[B, D]}$ 批次与一个 $\text{bf16[D, F]}$ 矩阵的单个矩阵乘法. 这可能是大的 MLP 块 ($W_\text{in}$ 或 $W_\text{out}$) 或较小的注意力投影之一 ($W_Q$, $W_K$, $W_V$, $W_O$). 为了进行这个矩阵乘法, 我们需要从 HBM 将这两个数组加载到 MXU 中, 进行乘法, 然后将结果写回 HBM. 和以前一样, 我们有:

$$T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/s}} = \frac{2BDF}{\text{加速器 FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{通信字节数}}{\text{带宽 字节/s}} = \frac{2BD + 2FD + 2BF}{\text{带宽 字节/s}}$$

TPU 或 GPU 可以通过在进行计算时进行加载来重叠这些操作, 因此要受计算限制, 我们需要 $$T_\text{math} \geq T_\text{comms}$$, 或者:

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 字节/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

其中 RHS 是我们硬件的算术强度. 现在让我们假设 $D$ 和 $F$ 相对于 $B$ 非常大 (通常我们的批次最多为 500, 而 $D$ 和 $F > 10k$), 我们可以通过使用 $\small{2BD + 2DF + 2BF \approxeq 2DF}$ 的事实来简化分母, 这给了我们

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approxeq \frac{2BDF}{2DF} \geq \frac{\text{加速器 FLOPs/s}}{\text{带宽 字节/s}}
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_\text{crit}
\end{align*}$$$

如果我们对权重进行量化或对矩阵乘法使用较低精度的 FLOPs, 这个临界批量大小可能会改变. 例如, 如果我们将权重 量化为 int8 或 fp8, $B_\text{crit}$ 会减少 2 倍. 如果我们用 int8 或 fp8 进行 FLOPs, $B_\text{crit}$ 会增加 2 倍. 因此, 如果我们令 $\beta = \text{每个参数的位数} / \text{每个激活的位数}$, $\alpha_\text{hbm} = C / W_\text{hbm}$, 我们的临界批量大小实际上是 $B_\text{crit} = \beta \alpha_\text{hbm}$. 

<p markdown=1 class="takeaway">**要点:** Transformer 矩阵乘法是受计算限制的, *当且仅当*每个副本的**token**批量大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{每个参数的位数} / \text{每个激活的位数}) = \beta \cdot \alpha_\text{hbm}$. 对于 TPU v5e 上的 bf16 激活, 这是 240 个 token. 对于 H100, 大约是 280 个 token.</p>

在训练期间, 我们在所有矩阵乘法中都会有很高的强度, 因为我们在一个非常大的批次上重复使用相同的权重. **这种高算术强度会延续到预填充, 因为用户提示通常有数百甚至数千个 token.** 正如我们之前看到的, TPUv5e 的硬件算术强度是 240, 所以如果一个长度超过 240 个 token 的序列被输入到一个在 bf16 下运行的密集模型中, 我们会期望受计算限制, 一切都很好. 比这短的提示在技术上可以批处理在一起以实现更高的利用率, 但这通常不是必需的.

<p markdown=1 class="takeaway">**要点:** 在预填充期间, 所有矩阵乘法基本上总是受计算限制. 因此, 简单地最大化硬件利用率或 MFU (模型 FLOPs 利用率) 就足以最大化每个芯片的吞吐量 (成本) 和延迟 (以 TTFT 的形式). 除非提示非常短, 否则在每个提示级别进行批处理只会增加延迟, 而对预填充吞吐量的改进很小.</p>

然而, 在生成期间, 对于每个请求, 我们只能一次一个 token 地进行前向传播, 因为步骤之间存在顺序依赖关系! 因此, 我们只能 (容易地) 通过将多个请求批处理在一起, 在批处理维度上进行并行化来获得良好的利用率. 我们稍后会更多地讨论这个问题, 但实际上在不影响延迟的情况下将许多并发请求批处理在一起是很困难的. 因此, **用生成来饱和硬件 FLOPs 要困难得多.**

<p markdown=1 class="takeaway">**要点:** 在生成期间, 总 token 批量大小必须大于 $B_\text{crit}$ 才能在线性/前馈操作上受计算限制 (对于 TPU v5e 上的 bf16 参数为 240). 因为生成是串行地, 逐个 token 地发生的, 这要求我们将多个请求批处理在一起, 这很困难!</p>

*值得注意的是这有多大!* 生成批量大小为 240 意味着 240 个并发请求同时生成, 以及 240 个单独的密集模型 KV 缓存. 这意味着这在实践中很难实现, 除了一些批量推理设置. 相比之下, 在预填充期间推送超过 240 个 token 是相当常规的, 尽管随着稀疏性的增加需要一些小心.

**请注意, 这个确切的数字会因量化和硬件的类型而异.** 加速器通常可以在较低的精度下提供更多的算术运算. 例如, 如果我们有 int8 参数但在 bf16 中进行计算, 临界批量大小会降至 120. 使用 int8 激活和 int8 参数, 它会跳回到 240, 因为 TPUv5e 可以提供 400 TOPs/s 的 int8 x int8.

### 注意力呢?

当我们看点积注意力操作时, 事情变得更加复杂, 特别是由于我们必须考虑 KV 缓存. 让我们只看一个具有纯多头注意力的注意力头. 在单个 Flash Attention 融合中, 我们<d-footnote>我们在这里简化了很多, 忽略了应用 softmax, 掩码等中的非矩阵乘法 FLOPs. 它们应该与计算或 HBM 读取重叠, 但在某些 TPU 代上实现可能不平凡. 这些细节不会改变主要信息, 即 KV 缓存通常受内存限制.</d-footnote>: 

1.  从 HBM 读取形状为 $\text{bf16[B, T, D]}$ 的 $Q$ 激活.
2.  从 HBM 读取 $KV$ 缓存, 这是一对 $\text{bf16[B, S, D]}$ 张量.
3.  在 $$QK$$ 矩阵乘法中执行 $2BSTD$ FLOPs. 使用 Flash Attention, 我们不需要将 $\text{bf16[B, S, T]}$ 注意力矩阵写回 HBM.
4.  在注意力 $$AV$$ 矩阵乘法中执行 $2BSTD$.
5.  将结果 $\text{bf16[B, T, D]}$ 张量写回 HBM.

把它们放在一起, 我们得到:

$$\text{多头注意力算术强度} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充, $S=T$ 因为我们正在进行自注意力, 所以这简化为 $T^2 / 2T = T / 2$. 这很棒, 因为这意味着**预填充期间注意力的算术强度是 $\Theta(T)$**. 这意味着很容易受计算限制. 只要我们的序列长度相当大, 我们就没问题!

但是由于生成的序列维度很小, 并且 $B$ 和 $D$ 维度抵消了, 我们可以做近似:

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕, 因为这意味着我们无法做任何事情来提高生成期间注意力的算术强度. 我们在加载一个巨大的 KV 缓存的同时只做了少量的 FLOPs. **所以我们在注意力期间基本上总是受内存带宽限制!**

<p markdown=1 class="takeaway">**要点:** 在预填充期间, 对于任何合理的序列长度 (大约 $\gt 480$ 个 token), 注意力通常受计算限制, 而在生成期间, 我们的算术强度低且恒定, 因此我们总是受内存带宽限制.</p>

*从概念上讲, 这是为什么?* 主要地, 我们在模型的线性部分受计算限制, 因为参数 (内存带宽密集型组件) 被许多批处理项重用. 然而, 每个批处理项都有自己的 KV 缓存, 所以更大的批量大小意味着更多的 KV 缓存. 我们几乎*总是*会在这里受内存限制, 除非架构被积极地调整. 

这也意味着一旦参数内存与 KV 缓存内存相当, 你将从增加批量大小中获得递减的吞吐量回报. 递减回报对你的伤害程度取决于单个序列的参数与 KV 缓存字节之比, 即大约是 $2DF / SHK$ 的比率. 由于 $HK\approx D$, 这大约取决于 $F$ 与 $S$ (序列长度) 的比率. 这也取决于使 KV 缓存更小的架构修改 (我们稍后会详细说明).

### LLM 延迟和吞吐量的理论估计

从这个数学中, 我们可以得到相当好的优化时应该瞄准的步骤时间的界限. **(注意: 如果我们希望读者从本章中带走一件事, 那就是以下内容).** 对于生成期间的小批量大小 (这很常见), 我们可以通过假设我们在注意力和 MLP 块中都受内存带宽限制来为我们的每步延迟设定下限:

$$\begin{equation*}
\text{理论最小步骤时间} = \frac{\text{批量大小} \times \text{KV 缓存大小} + \text{参数大小}}{\text{总内存带宽}}
\end{equation*}$$$

同样, 对于吞吐量:

$$\begin{equation*}
\text{理论最大 token/秒} = \frac{\text{批量大小} \times \text{总内存带宽}}{\text{批量大小} \times \text{KV 缓存大小} + \text{参数大小}}
\end{equation*}$$$

最终, 随着我们批量大小的增长, FLOPs 开始主导参数加载, 所以在实践中我们有更通用的方程:

$$\begin{align}
\tiny \text{理论步骤时间 (通用)} = \underbrace{\frac{\text{批量大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力 (总是受带宽限制)}} + \underbrace{\max\left(\frac{2 \times \text{批量大小} \times \text{参数数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP (可以受计算限制)}}
\end{align}$$$

其中注意力组件 (左) 从不受计算限制, 因此不需要 FLOPs 屋顶线. 这些对于粗略的计算相当有用, 例如

<b markdown=1 style="color: #57cf57;">小测验:</b> 假设我们想在 TPU v5e 4x4 切片上, 用 int8 和 bf16 FLOPs, 8192 上下文和 100 kB / token KV 缓存, 从一个 30B 参数的密集模型中生成一个批量大小为 4 个 token 的步骤. 这个操作的延迟的合理下限是多少? 如果我们想采样一个 256 个 token 的批次呢?

{% details 点击这里查看答案. %}

**答案:** 在 int8 中, 我们的参数将使用 30e9 字节, 并且根据给定的规格, 我们的 KV 缓存将每个使用 `100e3 * 8192 = 819MB`. 我们有 16 个芯片, 每个芯片有 `8.1e11` 字节/秒的带宽和 `1.97e14` bf16 FLOPs/s. 从上面的方程中, 由于我们的批量大小很小, 我们预计我们的步骤时间至少为 `(4 * 819e6 + 30e9) / (16 * 8.1e11) = 2.5 ms`. 在 256 个 token 时, 我们将完全进入 MLP 块的计算限制状态, 所以我们的步骤时间大约是 `(256 * 819e6) / (16 * 8.1e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`.

{% enddetails %}

正如你所看到的, 这里在吞吐量和延迟之间存在明显的权衡. 小批量速度快, 但不能很好地利用硬件. 大批量速度慢, 但效率高. 这是为一些较早的 PaLM 模型计算的延迟-吞吐量帕累托前沿 (来自[ESTI 论文](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>):

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>图:</b> 几个 PaLM 模型的成本 (读作: 吞吐量) 与延迟的帕累托前沿. 请注意芯片数量 (C) 和批量大小 (B) 如何将你沿着帕累托前沿移动, 除了绿点 (PaLM 540B 的 C:32 B:16), 在那里可用内存阻止了设置支持一个好的批量大小, 并导致吞吐量受损. 请注意吞吐量通常在批量大小 240 之后趋于平缓. int8 权重提供了更好的延迟-吞吐量帕累托最优, 但不是更好的最大吞吐量." %}

我们不仅用批量大小作为旋钮来权衡延迟和吞吐量, 如果我们发现自己受 HBM 限制, 我们也可能更喜欢更大的拓扑而不是更小的拓扑, 以便容纳更大的批次. [下一节](../applied-inference)将更详细地探讨这一点.

<p markdown=1 class="takeaway">**要点:** 如果你关心生成吞吐量, 请使用尽可能大的每个芯片的批量大小. 任何高于 TPU 算术强度 ($B_\text{crit}$, 通常为 120 或 240) 的每个芯片的批量大小都将最大化吞吐量. 你可能需要增加你的拓扑来实现这一点. 较小的批量大小将允许你以吞吐量为代价来提高延迟.</p>

{% details 从硬件的角度来看, 这有一些警告. 点击这里查看一些细节. %}

这一切都相当理论化. 在实践中, 我们通常不会看到一个清晰的屋顶线, 原因有几个:

*   我们关于 HBM 读取将与 FLOPs 完美重叠的假设是不现实的, 因为我们的编译器 (XLA) 是会出错的.
*   对于分片模型, XLA 也经常无法有效地将我们模型分片矩阵乘法的 ICI 通信与 FLOPs 本身重叠, 所以我们经常在 $\text{BS}=32$ 以上的线性操作上开始出现延迟损失.
*   大于理论屋顶线的批量大小仍然会看到一些吞吐量的改善, 因为重叠不完美, 但这是一个很好的启发式方法.

{% enddetails %}

### 内存呢?

我们花了一些时间来看带宽和 FLOPs, 但没有看内存. 在推理时, 内存情况看起来大不相同, 这要归功于我们的新数据结构, KV 缓存. 对于本节, 让我们选择一个真实模型 (LLaMA 2-13B) 来演示情况有多么不同:

| 超参数 | 值 |
| :----------------- | :----- |
| L (层数) | 40 |
| D (d_model) | 5,120 |
| F (ffw_dimension) | 13,824 |
| N (头数) | 40 |
| K (num_kv_heads) | 40 |
| H (qkv_dim) | 128 |
| V (num_embeddings) | 32,000 |

在推理期间什么在使用内存? 嗯, 显然, 是我们的参数. 计算这些, 我们有:

| 参数 | 公式 | 大小 (字节) |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| FFW 参数 | d_model<sup>2</sup> x ffw_multiplier x 3 (用于 gelu + 输出投影) x n_layers | 5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9** |
| 词汇表参数 | 2 (输入和输出嵌入) x n_embeddings x d_model | 2 x 32,000 x 5,120 = **0.3e9** |
| 注意力参数 | [2 (*q 和输出*) x d_model x n_heads x d_qkv + 2 (*k 和 v*) x d_model x n_kv_heads x d_qkv] x n_layers | (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = **4.2e9** |

将这些参数加起来, 我们得到 8.5e9 + 4.2e9 + 0.3e9 = **13e9 总参数**, 正如预期的那样. 正如我们在前面的章节中看到的, 在训练期间, 我们可能会将参数存储在 bfloat16 中, 优化器状态存储在 float32 中. 这可能会使用大约 100GB 的内存. 这与我们的梯度检查点相比相形见绌, 后者可以使用几个 TB.

**推理有什么不同?** 在推理期间, 我们存储一个参数的副本, 比如说在 bfloat16 中. 这使用了 26GB —— 在实践中, 我们通常可以通过量化做得更好. 没有优化器状态或梯度需要跟踪. 因为我们不进行检查点 (为后向传播保留激活), 我们的激活占用空间对于预填充<d-footnote>特别感谢 Flash Attention, 它避免了物化我们的注意力矩阵</d-footnote>和生成都可以忽略不计. 如果我们预填充 8k token, 单个激活只使用大约 `8,192 x 5,120 x 2 字节 = 80MB` 的内存. 更长的预填充可以分解成许多更小的前向传播, 所以对于更长的上下文也不是问题. 生成使用的 token 比这还少, 所以激活可以忽略不计.

**主要区别在于 KV 缓存**. 这些是所有过去 token 的键和值投影, 大小仅受最大允许序列长度的限制. $$T$$ 个 token 的总大小是

$$\text{KV 缓存大小} = 2 \cdot \text{每个浮点数的字节数} \cdot H \cdot K \cdot L \cdot T$$

其中 $$H$$ 是每个头的维度, $$K$$ 是 KV 头的数量, $$L$$ 是层数, 2 来自于同时存储键和值.

**这会很快变得很大**, 即使批量大小和上下文长度适中. 对于 LLaMA-13B, 一个 8192 序列在 bf16 下的 KV 缓存是

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{字节}) \times 2 = 6.7 \text{GB}$$$

**仅仅 4 个这样的缓存就超过了我们参数的内存使用量!** 需要明确的是, LLaMA 2 在较长上下文下的 KV 缓存大小方面没有进行优化 (情况并不总是这么糟, 因为通常 $K$ 要小得多, 就像 LLaMA-3 中那样), 但这仍然具有说明性. 我们在内存或延迟估计中不能忽略这些.

### 为 LLaMA 2-13B 建模吞吐量和延迟

让我们看看如果我们尝试在 8xTPU v5es 上以不同的批量大小完美高效地执行生成, 直到达到前面为最大理论吞吐量推导出的临界批量大小 (240), 会发生什么.

| 批量大小 | 1 | 8 | 16 | 32 | 64 | 240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV 缓存内存 (GiB) | 6.7 | 53.6 | 107.2 | 214.4 | 428.8 | 1608 |
| 总内存 (GiB) | 32.7 | 79.6 | 133.2 | 240.4 | 454.8 | 1634 |
| 理论步骤时间 (ms) | 4.98 | 12.13 | 20.30 | 36.65 | 69.33 | 249.09 |
| 理论吞吐量 (token/秒) | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5es 给了我们 128GiB 的 HBM, 6.5TiB/s 的 HBM 带宽 (每个 0.82TiB/s) 和 1600TF/s 的计算能力.

对于这个模型, 增加批量大小确实能给我们带来更好的吞吐量, 但我们很快就会遇到收益递减. 我们在批量大小超过 16 时就会 OOM, 并且需要数量级更多的内存才能接近 240. 更大的拓扑可以改善延迟, 但我们在每个芯片的吞吐量上遇到了瓶颈.

假设我们保持总参数数量不变, 但神奇地将 KV 缓存缩小 5 倍 (比如说, 使用 1:5 的 [GMQA](#tricks-for-improving-generation-throughput-and-latency), 这意味着我们有 8 个 KV 头在 40 个 Q 头上共享 —— 更多细节见下一节).

| 批量大小 | 1 | 8 | 16 | 32 | 64 | 240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV 缓存内存 (GiB) | 1.34 | 10.72 | 21.44 | 42.88 | 85.76 | 321.6 |
| 总内存 (GiB) | 27.34 | 36.72 | 47.44 | 68.88 | 111.76 | 347.6 |
| 理论步骤时间 (ms) | 4.17 | 5.60 | 7.23 | 10.50 | 17.04 | 52.99 |
| 理论吞吐量 (token/秒) | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

使用更小的 KV 缓存, 我们仍然有递减的回报, 但理论上每个芯片的吞吐量会持续扩展到批量大小 240. 我们可以容纳更大的批量 64, 并且在所有批量大小下延迟也始终更好. 延迟, 最大吞吐量和最大批量大小都得到了显著改善! 事实上, 后来的 LLaMA 代使用了这个确切的优化 —— LLaMA-3 8B 有 32 个查询头和 8 个 KV 头 ([来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)).

<p markdown=1 class="takeaway">**要点:** 除了参数, KV 缓存的大小对模型的最终推理性能有很大影响. 我们希望通过架构决策和运行时优化的组合来控制它.</p>

## 提高生成吞吐量和延迟的技巧

自最初的 [Attention is All You Need 论文](https://arxiv.org/abs/1706.03762)以来, 已经开发了许多技术来使模型更高效, 通常专门针对 KV 缓存. 一般来说, 更小的 KV 缓存使得在不影响延迟的情况下更容易增加生成步骤的批量大小和上下文长度, 并使 Transformer 周围的系统 (如请求缓存) 的生活更轻松. 忽略对质量的影响, 我们可能会看到:

**分组多查询注意力 (又名 GMQA, GQA):** 我们可以减少 KV 头的数量, 并在注意力机制中与许多 Q 头共享它们. 在极端情况下, 可以在所有 Q 头之间共享单个 KV 头. 这将 KV 缓存减少了 Q:KV 比率的倍数, 相对于纯 MHA, 并且已经观察到模型的性能对这种变化相对不敏感.

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这实际上也增加了注意力计算的算术强度 (参见[第 4 节](../transformers)中的问题 4).

**混合一些局部注意力层:** 局部注意力将上下文限制在一个小到中等大小的最大长度. 在训练时间和预填充时间, 这涉及到将注意力矩阵掩码为一个对角条带, 而不是一个三角形. 这有效地限制了局部层的 KV 缓存的最大长度. 通过将一些局部层与一些全局层混合到模型中, 在上下文长度超过局部窗口时, KV 缓存的大小会大大减小.

**跨层共享 KV:** 模型可以学习以某种模式跨层共享相同的 KV 缓存. 虽然这确实减小了 KV 缓存的大小, 并在增加批量大小, 缓存, 离线存储等方面提供了好处, 但共享的 KV 缓存可能需要多次从 HBM 读取, *因此它不一定能改善步骤时间.*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>左:</b> 多层纯全局注意力. <b>右:</b> 一些全局/局部交错模式的示例, 与相邻层共享. 来源: <a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>."
%}

**量化:** 推理通常对参数和 KV 的精度不太敏感. 通过对参数和 KV 缓存进行量化 (例如, int8, int4, `fp8` 等), 我们可以节省两者的内存带宽, 减少达到计算屋顶线所需的批量大小, 并节省内存以在更大的批量大小下运行. 量化还有一个额外的好处, 即即使模型没有用量化进行训练, 也通常可以在训练后应用.

**使用不规则 HBM 读取和分页注意力:** 我们在上面的计算中为每个 KV 缓存分配了 8k 的上下文, 但通常不需要从内存中读取整个 KV 缓存 —— 请求具有广泛的长度分布, 并且不使用模型的最大上下文, 因此我们通常可以实现仅读取 KV 缓存的非填充部分的内核 (例如, Flash Attention 变体).

分页注意力<d-cite key="paged"></d-cite> 是对此的改进, 它将 KV 缓存存储在操作系统风格的页表中, 并且基本上完全避免了填充 KV 缓存. 这增加了很多复杂性, 但意味着每个批次只使用它需要的内存. 这是一个运行时优化, 因此它同样与架构无关.

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图:</b> 在生成期间, 单个 token (第四个) 关注多个 KV 缓存块/页. 通过对 KV 缓存进行分页, 我们避免了加载或存储比我们需要更多的内存. 摘自 <a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention 论文</a>."
%}

<p markdown=1 class="takeaway">**大局:** 总而言之, 与标准 MHA Transformer 相比, 这些 KV 缓存优化可以将 KV 缓存大小减少一个数量级以上. 这可以导致 Transformer 的总成本提高一个数量级.</p>

## 在多个加速器上分布推理

到目前为止, 我们一直在含糊地讨论如何扩展到单个芯片之外. 遵循[第 5 节](../training), 让我们探讨可用的不同策略及其权衡. 与往常一样, 我们将分别看待预填充和生成.

### 预填充

从屋顶线的角度来看, **预填充几乎与训练相同**, 几乎所有相同的技术和权衡都适用 —— 模型 (Megatron) 并行, 序列分片 (对于足够长的上下文), 流水线, 甚至 FSDP 都是可行的! 你只需要保留 KV, 以便稍后可以进行生成. 与训练一样, 增加芯片数量可以让我们获得更多的 FLOPs/s (可能降低 TTFT), 但会增加通信开销 (可能降低每个芯片的吞吐量).

**分片预填充的一般规则:** 这是预填充的一般规则集. 我们假设我们只在一个序列上进行预填充 (没有批处理维度):

1.  *模型分片:* 我们通常首先进行一定量的模型并行, 直到我们受 ICI 限制. 正如我们在[第 5 节](../training)中看到的, 对于 1 个轴, 这大约是 $F / 2200$ (通常是 4-8 路分片).
2.  *序列并行:* 除此之外, 我们进行序列并行 (就像数据并行, 但在序列维度上进行分片). 虽然序列并行在注意力中引入了一些额外的通信, 但在较长的上下文中, 它通常相当小. 与训练一样, 我们可以重叠通信和计算 (分别使用 Megatron 的集合矩阵乘法和环形注意力).

<p markdown=1 class="takeaway">**要点:** 在预填充期间, 几乎任何在训练期间可行的分片都可以正常工作. 进行模型并行直到 ICI 限制, 然后进行序列并行.</p>

### 生成

生成比预填充更复杂. 一方面, 获得大的批量大小更难, 因为我们需要将许多请求批处理在一起. 延迟目标更低. 总之, 这意味着我们通常更受内存限制, 对通信开销更敏感, 这限制了我们的分片策略:

1.  **FSDP 是不可能的:** 由于我们在从 HBM 加载参数和 KV 缓存到 MXU 时受内存限制, 我们不希望通过比 HBM 慢几个数量级的 ICI 来移动它们. *我们希望移动激活而不是权重.* 这意味着类似于 FSDP 的方法通常对于生成是完全不可行的.<d-footnote>训练后意外地将其保留是一个简单而常见的导致数量级回归的方式</d-footnote>

2.  **没有理由进行数据并行:** 纯数据并行没有帮助, 因为它复制了我们的参数, 并且无助于我们更快地加载参数. 你最好启动多个模型的副本.<d-footnote>我们的意思是, 启动多个服务器, 每个服务器都有一个较小批量大小的模型副本. 模型级别的数据并行严格来说更差.</d-footnote>

3.  **没有序列 = 没有序列分片.** 祝你好运, 序列分片.

_这主要给我们留下了密集模型生成的模型分片变体_. 与预填充一样, 我们可以做的最简单的事情是简单的模型并行 (激活完全复制, 权重在 MLP 的隐藏维度上完全分片), 直到我们受 ICI 限制的 4-8 路. 然而, 由于我们经常受内存带宽限制, 我们实际上可以超越这个限制来提高延迟!

**关于生成 ICI 限制的说明:** 在训练期间, 我们希望受计算限制, 所以我们的屋顶线着眼于我们的 ICI 通信何时比我们的 FLOPs 花费更长的时间. 然而, 在生成期间, 如果我们受参数加载的内存带宽限制, 我们可以将模型分片增加到超过这个点, 并以最小的吞吐量成本 (以 token/秒/芯片为单位) 提高延迟. 更多的模型分片给了我们更多的 HBM 来加载我们的权重, 我们的 FLOPs 并不重要.<d-footnote>从某种意义上说, FLOPs 时间不是我们的瓶颈, 所以我们需要担心的是 ICI 时间超过参数加载时间.</d-footnote> 让我们看看在它成为瓶颈之前我们可以做多少模型并行.

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}
\end{align*}$$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$. 对于 TPU v5e 和 TPU v6e, 这个数字通常在 8 左右. 这意味着例如, 如果 $F$ 是 16,384, $B$ 是 32, 我们理论上可以进行模型并行, 直到 `16384 / (32 * 8) = 64` 路, 而不会对吞吐量产生有意义的影响. 这假设我们可以完全将我们的 KV 缓存分片 64 路, 这很困难: 我们将在下面讨论这个问题.

对于注意力层, 我们也对注意力 $$W_Q$$ 和 $$W_O$$ 在头 Megatron 风格上进行模型分片. KV 权重相当小, 复制它们通常比分片超过 $K$ 路分片更便宜.

<p markdown=1 class="takeaway">**要点:** 我们在生成期间唯一的选择是模型并行的变体. 我们的目标是移动激活而不是 KV 缓存或参数, 它们更大. 当我们的批量大小很大时, 我们进行模型并行直到 FLOPs-ICI 限制 ($F / \alpha$). 当我们的批量大小较小时, 我们可以通过更多的模型分片来提高延迟 (以适度的吞吐量成本). 当我们想要模型分片的数量超过我们拥有的 KV 头时, 我们也可以在批处理维度上对我们的 KV 进行分片.</p>

### 分片 KV 缓存

**我们还有一个需要分片的新数据结构 —— KV 缓存.** 同样, 我们几乎总是倾向于避免复制缓存, 因为它是注意力延迟的主要来源. 为此, 我们首先在头维度上对 KV 进行 Megatron 分片. 这仅限于 $K$ 路分片, 因此对于头数较少的模型, 我们尽可能多地对头维度进行分片, 然后在批处理维度上进行分片, 即 $\text{KV}[2, B_Z, S, K_Y, H]$. 这意味着 KV 缓存是完全分布式的.

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>图:</b> (a) 具有纯模型分片的多头注意力和 (b) 具有 KV 缓存批处理分片的多查询注意力的注意力机制比较. 请注意, 我们需要两个额外的 AllToAll 来将激活从模型分片转移到批处理分片, 以便它们可以作用于 KV 缓存."
%}

这样做的代价是每个注意力层需要两次 AllToAll —— 一次是将 Q 激活转移到批处理分片, 以便我们可以用批处理分片计算注意力, 另一次是将批处理分片的注意力输出转移回纯模型分片.

{% details 这是完整的算法! %}

在这里, 我们将写出在 $Y$ 和 $Z$ 上都进行模型并行的完整注意力算法. 我为同时使用 $K$ 表示键张量和 KV 头维度而道歉. 设 $M=N/K$.

<div markdown=1 class="algorithm">

1. X[B, D] = ... (现有激活, 从前一层未分片)
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K, H] = ... (现有 KV 缓存, 批处理分片)
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

</div>

这相当复杂, 但你大致可以看到它是如何工作的. 新的通信成本适中, 因为它们作用于我们的小激活, 作为回报, 我们节省了大量的内存带宽来加载 KV (它们是静止的).

{% enddetails %}

*   **序列分片:** 如果批量大小太小, 或者上下文很长, 我们可以对 KV 缓存进行序列分片. 同样, 我们在这里为跨分片累积注意力付出了集体成本. 首先我们需要对 Q 激活进行 AllGather, 然后以类似于 Flash Attention 的方式累积 KV.

## 设计一个有效的推理引擎

到目前为止, 我们已经研究了如何独立地优化和分片单个预填充和生成操作. 为了有效地使用它们, 我们需要设计一个推理引擎, 它可以将这两个操作馈送到我们选择的延迟/吞吐量帕累托前沿上的一个点.

最简单的方法是简单地运行一批预填充, 然后运行一批生成:

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>图:</b> 在最简单的设置中, 请求被聚合, 服务器在运行一批预填充和调用生成函数之间交替, 直到所有序列完成." %}

这很容易实现, 并且是大多数代码库中的第一个推理设置, 但它有多个缺点:

1.  **延迟很糟糕.** 我们将预填充和生成批量大小耦合在一起. 在大的预填充批量大小下, 首个 token 的时间 (TTFT) 很糟糕 —— 你需要完成所有预填充, 然后用户才能看到任何 token. 在小的批量大小下, 生成吞吐量很糟糕.
2.  **我们用较长的生成阻塞了较短的生成.** 许多序列会比其他序列更早完成, 在生成期间留下空的批处理槽, 进一步损害生成吞吐量. 随着批量大小和生成长度的增加, 问题会加剧.
3.  **预填充被填充.** 预填充被填充到最长的序列, 我们浪费了大量的计算. 对此有解决方案, 但历史上 XLA 使得跳过这些 FLOPs 相当困难. 同样, 批量大小和预填充序列长度越大, 情况就越糟.
4.  **我们被迫在预填充和生成之间共享一个分片.** 预填充和生成都存在于同一个切片上, 这意味着我们对两者使用相同的拓扑和分片 (除非你保留两份权重), 这通常对性能没有帮助, 例如, 生成需要更多的模型分片.

因此, 这种方法只推荐用于边缘应用 (通常只关心为一个用户提供服务并使用具有较少 FLOPs/字节的硬件) 和 Transformer 代码库生命周期早期的快速迭代 (由于其简单性).

一个稍微好一点的方法是在批量大小为 1 时执行预填充 (此时它受计算限制但具有合理的延迟), 但在生成期间将多个请求批处理在一起:

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这将避免批量预填充造成的 TTFT 浪费, 同时保持较高的生成吞吐量. 我们称之为**交错**配置, 因为我们“交错”了预填充和生成步骤. 这对于批量生成应用 (如评估) 非常强大, 在这些应用中, 吞吐量是主要目标. 编排器可以配置为在任何生成槽位空闲时优先进行预填充, 即使对于非常大的生成批量大小, 也能确保高利用率. 我们还可以避免将预填充填充到最大长度, 因为它没有与其他请求批处理.

主要缺点是, 当服务器执行预填充时, 所有其他请求的生成都会暂停, 因为所有计算资源都将被预填充消耗. 正在解码其响应的用户 A 将被正在进行预填充的用户 B 阻塞. 这意味着即使 TTFT 得到了改善, token 生成平均也会抖动和缓慢, 这对于许多应用来说不是一个好的用户体验 —— 其他用户的预填充处于请求总延迟的关键路径上.

为了解决这个问题, 我们将解码和预填充分开. 虽然 Transformer 推理可以在一台服务器上完成, 但从延迟的角度来看, 在两组 TPU/GPU 上执行这两个不同的任务通常更好. 预填充服务器生成 KV 缓存, 这些缓存通过网络发送到生成服务器, 生成服务器将多个缓存批处理在一起并为每个缓存生成 token. 我们称之为**“分离式”**服务.

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

这提供了一些优势:

1.  **大规模低延迟**: 用户的请求永远不会阻塞在另一个用户的请求上, 除非预填充容量不足. 请求应该立即被预填充, 然后发送到生成服务器, 然后立即插入到生成缓冲区中. 如果我们预计会有许多并发请求进来, 我们可以独立于生成服务器的数量来扩展预填充服务器的数量, 这样用户就不会长时间停留在预填充队列中.

2.  **专业化:** 通常, 预填充和生成的延迟最优参数分片策略/硬件拓扑非常不同 (例如, 更多的模型并行对生成有用, 但对预填充没用). 将这两个操作限制为使用相同的分片会损害两者的性能, 并且拥有两套权重会占用内存. 此外, 通过将预填充移动到自己的服务器上, 它不需要持有任何 KV 缓存, 除了它当前正在处理的那个. 这意味着我们有更多的内存可用于历史缓存 (见下一节) 或优化预填充延迟.

一个缺点是 KV 缓存现在需要在网络上传输. 这通常是可以接受的, 但再次为减小 KV 缓存大小提供了动力.

<p markdown=1 class="takeaway">**要点:** 对于延迟敏感, 高吞吐量的服务, 我们通常必须将预填充和生成分离到不同的服务器上, 预填充以批量 1 运行, 生成则将许多并发请求批处理在一起.</p>

### 连续批处理

上面的问题 (2) 激发了**连续批处理**的概念. 我们优化并编译:

*   一些具有可变上下文长度的预填充函数, 并将其插入到某个 KV 缓冲区, 某个最大批量大小和上下文长度/页数中.
*   一个生成函数, 它接受 KV 缓存, 并为所有当前活动的请求执行生成步骤.

然后, 我们将这些函数与一个编排器结合起来, 该编排器对传入的请求进行排队, 根据可用的生成槽位调用预填充和生成, 处理历史缓存 (见下一节) 并流式传输 token.

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存

由于预填充成本高昂且受计算限制 (给了我们更少的余地), 减少其成本的最佳方法之一是减少预填充的次数. 因为 LLM 是自回归的, 查询 ["我", "喜欢", "狗"] 和 ["我", "喜欢", "猫"] 产生的 KV 缓存的前两个 token 是相同的. 这意味着, 原则上, 如果我们先计算“我喜欢狗”的缓存, 然后再计算“我喜欢猫”的缓存, 我们只需要进行 1/3 的计算. 我们可以通过重用缓存来节省大部分工作. 这在一些特定情况下特别强大:

1.  **聊天机器人**: 大多数聊天机器人对话都涉及严格附加到自身的来回对话. 这意味着如果我们能保存每次对话轮次的 KV 缓存, 我们就可以跳过除最新 token 之外的所有计算.
2.  **少样本提示**: 如果我们有任何类型的少样本提示, 这可以被保存和重用. 系统指令通常也具有这种形式.

这之所以难以做到, 唯一的限制是内存. 正如我们所见, KV 缓存很大 (通常是几个 GB), 为了使缓存有用, 我们需要将它们保留到后续查询到达. 通常, 预填充服务器上任何未使用的 HBM 都可以用于本地缓存系统. 此外, 加速器通常在其 CPU 主机上有很多内存 (例如, 一个 8xTPUv5e 服务器有 128GiB 的 HBM, 但大约有 450GiB 的主机 DRAM). 这个内存比 HBM 慢得多 —— 通常慢到无法进行生成步骤 —— 但对于缓存读取来说足够快. 在实践中:

*   因为 KV 缓存是处理初始请求的 TPU 集合的本地缓存, 我们需要某种形式的亲和路由来确保后续查询到达同一个副本. 这可能会导致负载均衡问题.
*   更小的 KV 缓存是有帮助的 (再次) —— 它使我们能够在相同的空间中保存更多的 KV 缓存, 并减少读取时间.
*   KV 缓存及其查找可以很自然地存储在树或 trie 中. 可以按 LRU (最近最少使用) 的原则进行驱逐.

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图:</b> 实现为 LRU trie 的 KV 前缀缓存. 我们可以通过共享前缀来避免重复的 KV 内存. 来源: <a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai 博客</a>."
%}

### 让我们看一个实现: JetStream

谷歌开源了一个实现此逻辑的库, 名为 [JetStream](https://github.com/google/JetStream). 该服务器有一组“预填充引擎”和“生成引擎”, 通常在不同的 TPU 切片上, 由单个控制器进行编排. 预填充发生在“[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)”中, 而生成发生在“[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)”中. 我们还有一个“[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)”, 负责编排从预填充切片到生成切片的 KV 缓存复制.

Engine 接口 (在[这里](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)实现) 是任何 LLM 都必须提供的通用接口. 关键方法是:

*   **prefill:** 接受一组输入 token 并生成一个 KV 缓存.
*   **insert:** 接受一个 KV 缓存并将其插入到生成正在生成的 KV 缓存批次中.
*   **generate:** 接受一组批处理的 KV 缓存, 并为每个批处理条目生成一个 token, 将单个 token 的 KV 缓存附加到每个 token 的解码状态.

我们还有一个 PyTorch 版本的 JetStream, 可在[这里](https://github.com/google/jetstream-pytorch)找到.

## 已解决的问题

我将为本节发明一个基于 LLaMA-2 13B 的新模型. 以下是详细信息:

| 超参数 | 值 |
| :----------------- | :----- |
| L (层数) | 64 |
| D (d_model) | 4,096 |
| F (ffw_dimension) | 16,384 |
| N (头数) | 32 |
| K (num_kv_heads) | 8 |
| H (qkv_dim) | 256 |
| V (num_embeddings) | 32,128 |

**问题 1:** 上述模型有多少参数? 每个 token 的 KV 缓存 (int8) 有多大? *你可以假设我们共享输入和输出投影矩阵.*

{% details 点击这里查看答案. %}

**参数数量:**

*   MLP 参数数量: $L * D * F * 3$
*   注意力参数数量: $L * 2 * D * H * (N + K)$
*   词汇表参数: $D * V$ (因为我们共享这些矩阵)

因此, 我们的总参数数量是 $L * D * (3F + 2H * (N + K)) + D * V$. 代入上面的数字, 我们有 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`. 因此, 这个模型大约有 184 亿个参数.

KV 缓存是每个 token $L * K * H$, 即 `64 * 8 * 256 = 131kB`.

{% enddetails %}

**问题 2:** 假设我们想在 TPUv5e 4x4 切片上服务这个模型, 并且可以完全在这个拓扑上分片我们的 KV 缓存. 假设我们对所有东西都使用 int8, 并且想要支持 128k 序列, 我们可以容纳的最大批量大小是多少? 如果我们将 KV 头的数量减少到 1 呢?

{% details 点击这里查看答案. %}

我们的 KV 缓存每个 token 在 int8 中有 $L * K * H$ 的大小, 或 `64 * 8 * 256 = 131kB`. 对于 128k 序列, 这意味着每个批处理条目 `131e3 * 128e3 = 16.8GB`. 由于每个 TPU 有 16GB 的 HBM, 包括我们的参数, 我们可以容纳的最大批量大小是 `(16 * 16e9 - 18.4e9) / 16.8e9 = 14`. 如果我们有 $K=1$, 我们将有这个的 8 倍, 即大约 112.

{% enddetails %}

**问题 3:** 假设参数在 TPU v5e 4x4 切片上完全分片, 将所有参数从 HBM 加载到 MXU 需要多长时间? 假设 int8 参数. *这是每步延迟的一个很好的下限.*

{% details 点击这里查看答案. %}

我们总共有 18.4B 个参数, 或 18.4e9 字节 (int8). 我们每个芯片有 8.1e11 HBM 带宽, 所以大约需要 `18e9 / (8.1e11 * 16) = 1.3ms`, 假设我们可以完全利用我们的 HBM 带宽.

{% enddetails %}

**问题 4:** 假设我们想在 TPUv5e 4x4 切片上使用 int8 FLOPs 和参数/激活来服务这个模型. 我们将如何为预填充和解码进行分片? *提示: 也许先回答这些问题:*

1.  4x4 上的 ICI 是什么样的?
2.  张量并行的屋顶线限制是什么?
3.  我们如何对 KV 缓存进行分片?

对于这个分片, 生成的粗略每步延迟是多少?

**问题 5:** 让我们假装上面的模型实际上是一个 MoE. MoE 模型实际上是一个具有 E 个 FFW 块副本的密集模型. 每个 token 通过 k 个 FFW 块, 这 `k` 个块被平均以产生输出. 让我们使用 `E=16` 和 `k=2` 以及上面的设置.

1.  它总共有多少个和激活的参数? *激活的意味着被任何给定的 token 使用.*
2.  在 TPU v5e 上需要多大的批量大小才能受 FLOPs 限制?
3.  每个 token 的 KV 缓存有多大?
4.  一个 T token 的前向传播涉及多少 FLOPs?

{% details 点击这里查看答案. %}

(1) 作为一个 MoE, 每个 MLP 块现在有 $3 * E * D * F$ 个参数, 比密集变体增加了 $E$ 倍. 因此, 它现在总共有 $L * D * (3EF + 2H * (N + K)) + D * V$ 或 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9` 个参数, 增加了约 12 倍. 对于激活的参数, 我们有 $k$ 而不是 $E$ 个激活的参数, 总共 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`, 比密集变体增加了不到 2 倍.

(2) 因为我们有 $E$ 倍的参数, 但只有 $k$ 倍的 FLOPs, 我们的 HBM 屋顶线增加了 $E/k$ 倍. 这意味着在 TPU v5e 上, 我们需要大约 `240 * (16 / 2) = 1920` 个 token.

(3) KV 缓存大小保持不变, 因为 MoE 特性不会改变注意力机制的任何东西.

(4) 这仍然是 $2ND$, 其中 $D$ 是激活的参数数量. 因此这是 $2 * \text{32.2e9} * T$.

{% enddetails %}

**问题 6:** 对于 MoE, 我们可以进行“专家分片”, 即我们将我们的专家分布在我们网格的一个轴上. 在我们的标准符号中, 我们的第一个 FFW 权重的形状为 `[E, D, F]`, 我们将其分片为 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>], 其中 `X` 仅在训练期间用作我们的 FSDP 维度. 假设我们想在 TPU v5e 上进行推理:

1.  在 TPU v5e 8x16 切片上, Y=8, Z=16, 上述模型的 HBM 权重加载时间是多少? 每个 TPU 有多少可用的 HBM?
2.  我们可以将我们的模型容纳在的最小切片是多少?

**问题 7 [2D 模型分片]:** 在这里, 我们将详细讲解 [ESTI 论文](https://arxiv.org/pdf/2211.05102)所谓的 2D 权重固定分片的数学原理. 我们在附录 B 中简要描述了这一点, 但请先尝试做这个问题, 看看你是否能算出数学原理. 2D 权重固定分片的基本思想是沿 $D$ 和 $F$ 轴对我们的权重进行分片, 以便每个块大致呈方形. 这减少了通信负载, 并允许我们进一步扩展.

这是 2D 权重固定的算法:

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U.X} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U.X})
4.  Out[B, D<sub>X</sub>] {U.YZ} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W2[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U.YZ})
</div>

你的目标是为这个算法计算出 $T_\text{math}$ 和 $T_\text{comms}$, 并找出它何时会优于传统的 3D 模型分片?

{% details 点击这里查看答案! %}

让我们计算 $T_\text{math}$ 和 $T_\text{comms}$. 我们所有的 FLOPs 都是完全分片的, 所以和以前一样, 我们有 $T_\text{math} = 4BDF / (N \cdot C)$, 但我们的通信现在是

$$\begin{align*}
 T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$$

其中我们注意到 AllReduce 的成本是两倍, 并且我们根据每个操作执行的轴数来缩放我们的通信. 假设我们可以自由选择我们的拓扑, 并且假设 $F=4D$ (如 LLaMA-2), 我们声称 (通过一些基本的微积分) $X$, $Y$ 和 $Z$ 的最佳值是 $X = \sqrt{N / 8}$, $YZ = \sqrt{8N}$, 所以总通信是

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先, 从上面复制, 普通的 1D 模型并行将有 $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$, 那么新的通信何时更小? 我们有

$$\begin{align*}
 T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}}
 \iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 81
\end{align*}$$$

对于一个通用的 $F$, 我们声称这个条件是

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

所以这告诉我们, 如果我们有超过 81 个芯片, 我们最好使用这个新方案. 现在这是一个有点奇怪的结果, 因为我们历史上发现自己在 ~20 路张量并行时受 ICI 限制. 但在这里, 即使我们受通信限制, 我们的总通信也会随着总芯片数量的增加而持续减少! 这告诉我们, 我们可以继续增加我们的芯片, 增加我们的批量大小, 进行更多的参数扩展, 并看到延迟降低.

{% enddetails %}

<h3 markdown=1 class="next-section">第 7 部分到此结束! 第 8 部分, 看看我们如何可能在 TPU 上服务 LLaMA 3, 点击 [这里](../applied-inference).</h3>

## 附录

### 附录 A: 批处理大小 > 240 规则有多真实?

我们上面提供的简单规则, 即我们的批量大小必须大于 240 个 token 才能受计算限制, 大致是正确的, 但忽略了 TPU 在其他操作不使用所有可用 HBM 时预取权重的一些能力, 比如在进行设备间通信时.

这是一个小型 Transformer 的层时间 (以微秒为单位) 的实证图, d<sub>model</sub> 为 8192, d<sub>ff</sub> 为 32768, 每层只有 2 个矩阵乘法. 这来自[这个 Colab 笔记本](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing). 你会看到步骤时间增长非常缓慢, 直到大约批量 240, 然后线性增长.

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

这是实际的吞吐量, 以 token/us 为单位. 这很清楚地说明了论点. 由于我们的层大约是 600M 参数, 在这里分片 4 路, 我们预计最小延迟大约是 365us.

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

所以至少在这个模型中, 我们确实看到吞吐量增加, 直到每个数据并行分片大约 BS240.

### 附录 B: 2D 权重固定分片

随着拓扑的增长, 如果我们可以访问更高维度的网格 (如 TPU 的网格), 就可以通过“**2D 权重分片**”进一步完善. 通过引入第二个分片轴. 我们称之为“**2D 权重固定**”, 在[高效扩展 Transformer 推理论文](https://arxiv.org/abs/2211.05102)中有更详细的描述.

因为我们只在 Megatron 中对隐藏的 $$F$$ 维度进行分片, 所以随着芯片数量随着 1D 分片而增长, 它可能会变得比 $$E$$ ( $$d_\text{model}$$ 维度) 小得多. 这意味着在更大的批量大小时, 在应用 MLP 的第一层后, 在隐藏维度上执行一部分集合操作可能更经济.

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

该图显示:

1.  1D 权重固定分片, 又名纯 Megatron 分片, 其中激活在 AllGather 后完全复制, 权重在隐藏的 F 维度上完全分片.
2.  2D 权重固定分片, 其中权重在隐藏的 F 和归约 E 维度上都进行分片, 激活在 E 维度上进行分片. 我们在第一层之前在 (yz) 轴上执行 AllGather, 然后在 (x) 轴上执行 ReduceScatter.

对于注意力层, Megatron 风格的分片对于较少数量的芯片也相对简单. 然而, Megatron 发生在 $$n_\text{heads}$$ 维度上, 这对可能的分片数量设置了限制. 通过修改 2D 分片 (不是分片隐藏维度, 而是分片 $$n_\text{heads}$$ 维度), 我们获得了进一步扩展的能力.

### 附录 C: 受延迟限制的通信

作为回顾, 在[第 3 节](../sharding)中, 我们推导了在 1D 环形链路上, 在每个 TPU 上对大小为 B 的张量执行 AllGather 所需的时间, 该链路具有全双工带宽 WICI 和延迟 Tmin.

$$T_\text{total} = \max\left(\frac{T_\text{min} \cdot |X|}{2}, \frac{B}{W_\text{ICI}}\right)$$

对于大的 B, 时钟时间相对恒定, 因为当你向系统添加更多芯片时, 你同时扩展了执行操作所需的数据移动量和可用的总带宽.

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于在延迟优化的推理期间移动的数据量相对较少, 激活上的集合操作通常受延迟项的限制 (特别是对于小批量大小). 通过计算我们需要完成的跳跃次数, 可以很容易地可视化延迟.

在 TPU 上, 如果通信的张量大小相关部分每跳 (一跳是两个相邻设备之间的通信) 小于 1 微秒, 我们可能会受到实际分派集合的固定开销的瓶颈. 单向 ICI 带宽为 `4.5e10`, 当 $$(\text{字节} / n_\text{shards}) / 4.5e10 < 1e-6$$ 时, ICI 通信变得受延迟限制. 对于 8 路 Megatron 分片, 这是当 `buffer_size < 360kB` 时. **这在推理期间实际上并不那么小:** `BS=16` 和 `D=8192` (int8), 我们的激活将使用 `16*8192=131kB`, 所以我们已经受延迟限制了.

<p markdown=1 class="takeaway">**要点:** 当 $$\text{总字节数} < W_{ICI} \times 1e-6$$ 时, 我们的通信会受到延迟限制. 例如, 在 $$Y$$ 上进行模型并行时, 当 $$Y > BD / 45,000$$ 时, 我们在 int8 中会受到限制.</p>

这里可以与计算屋顶线进行类比 —— 我们正在承担一些小操作的固定成本 (通信的延迟, 矩阵乘法的内存带宽).

### 附录 D: 推测性采样

当我们*真正*关心端到端延迟时, 我们可以采用一个额外的技巧, 称为推测性采样<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>. 作为回顾, 我们通常从一个大型 Transformer 中逐个生成 token:

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

通过推测性采样, 我们使用一个更小, 更便宜的模型来生成 token, 然后用大模型检查结果. 这最容易用*贪婪解码*来理解:

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1.  我们从某个更小, 更便宜的模型中进行贪婪采样. 理想情况下, 我们使用一个经过训练以匹配较大模型的模型, 例如通过蒸馏, 但它也可以像简单地使用 n-gram 或匹配一小部分文本的 token 一样简单.
2.  在我们生成了 K 个 token 之后, 我们使用大模型来计算我们到目前为止生成的所有 token 的下一个 token 的 logits.
3.  由于我们是贪婪解码, 我们可以简单地检查较小模型生成的 token 是否是所有可能 token 中概率最高的. 如果其中一个 token 错误, 我们取最长的正确前缀, 并用正确的 token 替换第一个错误的 token, 然后回到 (1). 如果所有 token 都正确, 我们可以使用最后一个正确的 logit 来采样一个额外的 token, 然后再回到 (1).

**为什么这是一个延迟上的胜利?** 这个方案仍然要求我们为每个 token 执行相当于大模型一次前向传播的 FLOPs, 但因为我们可以将一堆 token 批处理在一起, 我们可以将所有这些 FLOPs 放在一次前向传播中完成, 并利用我们*不受计算限制*的事实来免费评分更多的 token.

平均而言, 每个被接受的 token 在 FLOPs 方面变得更加昂贵 (因为有些会被拒绝, 而且我们必须调用一个草稿模型), 但我们从硬件中榨取了更多的 FLOPs, 而且小模型很便宜, 所以我们总体上是赢的. 我们还在多个步骤之间共享 KV 缓存加载, 所以**对于长上下文, 推测性解码也可以在吞吐量上取得胜利.** 由于所有内容都经过了大模型的检查, 我们根本没有改变采样分布 (尽管对于非贪婪解码, 确切的轨迹会有所不同).

传统上, 推测性解码依赖于存在一个与目标模型具有相似采样分布的较小模型, 例如 LLaMA-2 70B 的 LLaMA-2 2B, 这通常不存在. 即使存在, 如果接受率很低, 较小的草稿模型仍然可能过于昂贵. 相反, 将草稿模型嵌入到主模型中可能会有所帮助, 例如通过在基础模型的后期层中添加一个专用的草稿头<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>. 因为这个头与主模型共享大部分参数, 所以它运行得更快, 并且更接近采样分布.

对于正常的自回归采样, token/s 与步骤时间相同. 我们仍然受限于这里算术强度部分的理论最小步骤时间 (事实上, 推测性采样的步骤时间通常比正常的自回归采样慢得多, 但因为我们平均每步得到超过 1 个 token, 我们可以得到更好的 token/s).

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图:</b> 该图显示了 Chinchilla (一个 70B 模型, 来自 DeepMind) 使用一个 4B 参数的草稿模型 (小模型) 的每步延迟和推测成功率. 对于 XSum (一个自然语言数据集), 理想的推测量大约是提前 3-4 个 token, 而 HumanEval (一个编码数据集) 更可预测, 并且从更激进的推测中获益." %}

**这对于非贪婪解码是如何工作的?** 这有点复杂, 但本质上归结为一个受 Metropolis-Hastings 启发的算法, 其中我们有从 logits 派生出的 $$P_\text{草稿模型}(\text{选择的 token})$$ 和 $$P_\text{目标模型}(\text{选择的 token})$$, 如果这些概率的比率小于某个阈值, 则概率性地拒绝选择的 token.

这[两篇](https://arxiv.org/abs/2211.17192) [论文](https://arxiv.org/abs/2302.01318) 同时推导出了这一点, 并提供了很好的例子来说明这在实践中是如何工作的.

<p markdown=1 class="takeaway">**要点:** 推测性采样是另一个强大的杠杆, 用于用吞吐量换取更好的每个 token 延迟. 然而, 在批量大小受限的情况下 (例如, 硬件占用空间小或 KV 缓存大), 它变成了双赢.</p>