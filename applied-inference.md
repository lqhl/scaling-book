---
layout: distill
title: "在TPU上部署LLaMA 3-70B"
# permalink: /main/
description: "让我们仔细看看我们如何在TPU v5e上部署LLaMA 3-70B模型。不同模型在性能上限下部署的成本如何？它们的KV缓存有多大？我们应该使用什么批量大小？参数和激活在推理期间如何分片？让我们通过一些粗略估计来分析生产环境中的延迟和吞吐量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "Part 7: Inference"

next_section_url: ../profiling
next_section_name: "Part 9: Profiling"

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
  - name: "LLaMA 服务情况分析"
  - subsections:
    - name: "思考吞吐量"
    - name: "关于预填充（prefill）呢？"
  - name: "可视化延迟吞吐量权衡"
  - name: "解题示例"

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

*本节将介绍部署 LLaMA-3 所需的条件以及如何高效地完成这项工作。与之前的"实践"章节一样，请在查阅答案之前先用笔和纸尝试解答问题！*

## LLaMA 服务情况分析

让我们回顾一下 LLaMA 3-70B 的配置（参考[第 6 节](../applied-training)）：

| **hyperparam**              | **value** |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

让我们从一个简单的问题开始：**我们应该在什么硬件上提供服务？** 答案基本上是，选择每美元 FLOPs 最便宜的硬件。<d-footnote>这并不总是正确的，有时更多的 HBM 或 ICI 带宽比 FLOPs 更关键，但这是一个很好的启发式方法。</d-footnote> 因此，我们通常希望在 TPU v5e 上提供服务，这是我们当前专用的推理芯片（价格来自 2025 年 2 月的 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)）：

| **TPU type** | **bfloat16 FLOPs/s** | **Google Cloud USD / hour** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每个 TPU v5e 都有 16GB 的 HBM，这将要求我们对模型进行相当激进的分片。让我们先思考一些可能对我们很重要的基本数量：

**问题：** LLaMA 3-70B 每个 token 的 KV 缓存有多大？*你可以假设我们使用 int8 存储。这决定了我们在给定拓扑结构上可以使用多大的批量大小。*

{% details 点击这里查看答案！ %}

LLaMA 3-70B 有 8 个 KV 头，所以每个 token 的大小是 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**请注意这有多大！** 如果我们的序列长度为 32k token（这是常见的），这使用 `162e3 * 32,768 = 5.3GB / 序列`。对于 BS=240，这就是 1.3TB！由于 TPU v5e 每个只有 16GB，我们需要大约 `(70e9 + 1.3e12) / 16e9 = 86` 个 TPU v5e 芯片才能容纳这么多内存。还要注意这与 70GB 模型参数相比有多大。

{% enddetails %}

**问题：** 假设我们想要以批量大小 32 和 8192 序列长度部署 L3 70B，所有内容（参数和 KV）都使用 int8。这将使用多少总内存？我们可以使用的最小分片是什么？

{% details 答案 %}

由于我们的 KV 在 int8 中是 `160e3` 字节，我们的总 KV 内存是 `160e3 * 8192 * 32 = 41.9e9` 字节。我们的参数是 `70e9` 字节，因为我们每个参数有 1 个字节。因此，我们的总内存使用量是 `41.9e9 + 70e9 = 112GB`。

我们可以使用的最小分片将有 `112e9 / 16e9 = 7` 个 TPU，或者（四舍五入到偶数大小），TPU v5e `4x2`。这将是一个很紧的配置，考虑到其他开销，我们可能无法完全适应，所以我们至少需要一个 `4x4`（或者降低批量大小）。

{% enddetails %}

**问题：** 在 TPU v5e `4x2` 上以这个批量大小和量化，我们期望每个解码步骤的大致延迟是多少？吞吐量（tokens / sec / chip）是多少？`4x4` 呢？*假设我们在 bfloat16 中执行 FLOPs 并且所有内容都完全分片。*

{% details 答案 %}

我们可以调用前一节的公式

$$\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}$$

这里我们的关键批量大小将约为 120，因为我们的参数在 int8 中但我们的 FLOPs 在 bfloat16 中。我们也可以手动计算 RHS 最大值，但这基本上是我们已经做过几次的计算。**因此，对于我们的矩阵乘法和 FLOPs，我们都深入到了内存受限的状态。**

严格地看内存带宽，我们的步骤时间基本上是 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.1e11) = 17ms`。**所以理论上我们的步骤时间约为 17ms。** 我们的吞吐量将是 `32 / .017 = 1882 tokens / sec`，或者 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个需要注意的地方，就是要检查我们是否可能在矩阵乘法上受到 ICI 限制。我们可以在这里为它分配 2 个轴，所以理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时我们会受到 ICI 限制，所以我们是安全的！

如果我们要在 `4x4` 上运行，我们在 ICI 方面仍然没问题，所以我们的延迟会降到 `17 / 2 = 8.5ms`，但我们的每芯片吞吐量将保持不变。

{% enddetails %}

### 思考吞吐量

让我们花点时间纯粹思考吞吐量。当我们优化吞吐量时，我们希望受到计算限制，这意味着我们接近利用所有 TPU MXU 容量。通常这意味着我们希望批量大小尽可能大，这样我们就可以做尽可能多的工作。

**问题：** 在 TPU v5e 上，使用 bfloat16 权重和激活，我们的批量大小需要多大才能在矩阵乘法中受到计算限制？如果我们使用 int8 权重但在 bfloat16 中执行 FLOPs 呢？int8 权重配合 int8 FLOPs 呢？

{% details 答案 %}

正如第 7 节所讨论的，对于任何 $B \ll D, F$ 的 bfloat16 矩阵乘法，我们有

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}$$

当我们的权重在 int8 中时，我们在分母中失去了一个因子 2，所以我们有 $2BDF / DF = 2B > 240$，或者同样地 $B > 120$，是之前关键批量大小的一半。这真的很有帮助！当我们使用 int8 权重和 int8 FLOPs 时，我们必须使用 TPU FLOPs/s 的 int8 值，它从 bfloat16 的 1.97e14 变为 3.94e14，几乎翻倍。这意味着我们又回到了大约 $B > 240$ 的起点。

int8 权重和 bfloat16 FLOPs 的情况相当常见，因为无损量化参数通常比进行低精度算术更容易。

{% enddetails %}

**问题：** 使用 bfloat16、int8 和 int4（KV 和参数）在 8k 上下文中，我们可以部署 LLaMA 3-70B 的最小 TPU v5e 拓扑是什么？*对于这个问题，你可以认为 KV 缓存可以忽略不计。*

{% details 答案 %}

这很简单！如果我们对很小的批量大小没问题，那么唯一的限制是将参数内存放入 HBM 中，即它只是 `ceil(num_params * sizeof(dtype) / HBM per TPU`，或者 `ceil(70e9 * sizeof(dtype) / 16e9)` 四舍五入到最合理的拓扑（2 的倍数）：

| dtype | param size | KV size / token (bytes) | min TPU v5es | actual min slice | remaining HBM for KV caches | num KV caches @ 8k |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             29              |         43         |

这太酷了！它告诉我们如果我们愿意，可以在 TPU v5e 2x2 上部署 LLaMA 70B。除了你会注意到 KV 缓存的数量非常小。这是我们的批量大小！这意味着我们将获得糟糕的 FLOPs 利用率。为了将我们的批量大小推到 240，我们很乐意使用更大的拓扑。

{% enddetails %}

**问题：** 假设我们使用适合这些拓扑的最大批量大小，我们可以期望每个生成步骤的延迟是多少？

{% details 答案 %}

这也很简单，因为我们选择批量大小来填满我们所有的 HBM！这只是将一个完整 TPU v5e 值的字节加载到 MXU 需要多长时间的问题。这只是 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`，所以这是 **19ms / step**。假设我们的生成具有 512 token 的中位数长度，那么每个解码大约是 9s。请注意，我们可以通过更小的批量大小获得略好的延迟，例如，如果我们只查看 int4 中的模型参数，我们的最小延迟约为 10ms / step，因为 HBM 不再满。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们总是可以通过询问将模型的所有参数从 HBM 加载到 MXU 需要多长时间来限制解码延迟的下限。当我们的 KV 缓存很小时，你可以将每一层看作只是逐块加载权重然后丢弃它们。除非我们使用大批量大小或大量设备间通信，这通常是一个合理的界限（在 1.5x 以内）。当我们的批量大小更大时，我们还需要模拟 KV 缓存加载，因为它主导了参数。</p>

同样，在 FLOPs 受限的状态下（例如训练或大批量推理），我们可以使用 $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ 下限，它假设没有通信。

**问题：** 对于这些情况，这给我们每芯片什么吞吐量（以查询 / 芯片计）？*你可以假设我们的中位数解码长度为 512 token。*

{% details 答案 %}

这是一个重要的问题，因为它与成本 / token 完全相关。

根据我们对中位数解码长度的假设，我们的吞吐量只是 $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$。这给我们大致 $$(4.42 / N)$$ QPS，所以代入 $$N$$ 我们得到：

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

请注意这是相当乐观的，因为它完全忽略了前向传播的工作内存（分配给激活和注意力的内存）。使用 Flash Attention 这并不是荒谬的，但它也不现实。真实数字可能只有这个的一半左右。为了获得绝对最大的吞吐量，我们可能希望将芯片数量增加一倍以上，并显著增加批量大小。

{% enddetails %}

**问题：** 如果我们将上述每个示例的拓扑加倍，我们的峰值吞吐量会如何变化？

{% details 答案 %}

如果我们在 bfloat16 中使用 4x8 分片，我们将有 372GB 剩余用于 KV 缓存，这将让我们的批量大小增加到 140。然后由于我们的步骤时间保持相同，我们将有 `16.54 / num_chips` 的吞吐量，或者

|       dtype       | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.51    |
|   int8 (on 4x4)   |    1.03    |
|   int4 (on 2x4)   |    2.06    |

进一步增加会带来更大的胜利！重要的要点是，**如果我们受到 KV 缓存大小的限制，最小拓扑并不是性能最高的拓扑**。

{% enddetails %}

**问题：** 现在让我们深入研究分片的问题。假设我们想要在 TPU v5e 4x8 上以 bfloat16 提供服务。在生成期间，我们会在 TPU v5e 4x8 上为我们的模型使用什么分片？我们可以避免受到通信限制吗？

{% details 答案 %}

正如前一节所讨论的，在生成期间对于分片我们实际上只有一个选择：模型并行。在我们变得通信受限之前我们能做多少？正如我们在前一节所讨论的，我们的模型大致在以下情况下变得通信受限

$$Y > \frac{F \cdot M_Y}{2200}$$

对于 LLaMA 3-70B，我们有 `F = 28,672`，所以如果我们做 2 个轴的模型分片，这给我们大致 $$Y = 28672 \cdot 2 / 2200 = 26$$，所以通常我们可以扩展到大约 16 个芯片而不会受到通信限制，这让我们可以使用 `4x4` 但不能使用 `4x8`。通常，由于我们没有完美地重叠计算，即使这个估计也过于乐观。

**要点：我们不能实际上在 4x8 上使用纯模型并行提供服务。** 我们在这里能做到的最好的是 4x2 或者_可能_一个 4x4。

However, as we've discussed, when our batch size is small we can often do more model parallelism without significantly hurting throughput, since our model is memory-bandwidth-bound and not FLOPs bound. We said before that this value is roughly $Y=F / (8\cdot B)$, so if we did batch size 64, we could in theory go up to `Y = 28,672 / (8 * 64) = 56` way model parallelism before we become ICI-bound. To sanity check this, we can look at $T_\text{ici comms}$, $T_\text{hbm comms}$, and $T_\text{math}$ for a single matmul. We clearly have:

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

For a `4x8`, this would give us $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`, $T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`, and $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`, so in theory we're still HBM bandwidth bound, which is great! *Note that scaling up from a `4x4` to a `4x8` probably isn't helpful from a throughput standpoint, but it'll reduce our latency!

If we look at the int8 and int4 configs, we _can_ do those with pure model parallelism. So we've hit a point at which quantization actually gives us a meaningful advantage beyond faster FLOPs: it lets us use a larger batch size before we become comms-bound. **So the end of this story is that we can't achieve peak throughput on a 4x8, but for the int8 and int4 configs we could do pure model parallelism*.

{% enddetails %}

<p markdown=1 class="takeaway">**Tip**: the maximum amount of useful model parallelism depends on $$d_{ff}$$ and the number of axes over which you're sharding your model. The maximum value usually ranges between 8 and 32 depending on the model size. You can scale beyond this limit to improve latency at some throughput cost.</p>

### 关于预填充（prefill）呢？

我们在这里主要忽略了预填充，因为它要简单得多。让我们将几个概念放在一起思考端到端的图景。

**问题：** 假设我们在预填充期间实现了 40% 的 FLOPs 利用率。在 16 个 TPU v5e 芯片上，长度为 8192 的预填充需要多长时间？

{% details 答案 %}

在 8k token 时，我们明显受到计算限制，所以我们只需要推理 FLOPs。我们知道我们的模型有 `70e9` 参数，所以每个前向传播使用 `2 * 70e9 * B` FLOPs。假设 40% 的 MFU（FLOPs 利用率），这给我们大约 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s` 的运行时间。与我们之前看到的数字相比，这实际上相当多！

{% enddetails %}

**问题：** 假设我们有一个中位数预填充长度为 8192 token，中位数解码长度为 4096 token。假设我们有一个生成批量大小为 32。平均每个步骤完成多少序列解码？平均每个步骤从我们的 KV 缓存中驱逐多少 token？

{% details 答案 %}

这相当直接。由于我们的中位数解码长度为 4096 token，序列将大致每 1 / 4096 token 完成一次。给定批量大小为 32，这意味着我们每步有 `32 / 4096` 个序列被驱逐。由于我们的 KV 缓存长度大约为 `8192 + 4096`，这是每步 `32 * (8192 + 4096) / 4096 = 96` 个 token 被驱逐。通用公式是 $B * (P + G) / G$，其中 $P$ 和 $G$ 是预填充和生成长度。

{% enddetails %}

**问题：** 假设我们使用解聚式服务，中位数预填充长度为 8192，中位数解码长度为 512。假设使用上面计算的 bfloat16 预填充和生成延迟。你需要什么比例的预填充：生成服务器来保持两者完全饱和。

{% details 答案 %}

这是一个相当有趣的问题。让 $P$ 是预填充服务器的数量，$G$ 是生成服务器的数量。所以总的来说，这是一个管道问题，我们以 `P / prefill_latency` 的速率输入序列，以 `B * G / (generate_latency * median_decode_length)` 的速率消耗它们。我们计算出每预填充步骤 `910ms`，在批量大小 43（让我们称之为 32）下每解码步骤 `19ms`。因此我们需要 `P / 0.91 = 32 * G / (0.019 * 512)` 或 `P = 3G`，即我们需要大约 3 倍于生成服务器的预填充服务器！

{% enddetails %}

## 可视化延迟吞吐量权衡

继续使用 LLaMA 70B 一会儿，让我们实际看看生成期间不同批量大小的延迟和吞吐量。正如我们在前一节为 PaLM 模型所展示的，这给了我们一个吞吐量/延迟的帕累托前沿。让我们假设 16 路张量并行，因为这是我们在 MLP 块中保持计算限制的合理界限。我们将在这里使用 TPU v5e 4x4 拓扑。**滑块控制序列长度，所以你可以看到更大 KV 缓存的影响。**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **看看成本和延迟之间的权衡有多剧烈。** 以每 token 延迟翻倍的成本，我们可以实现每 token 成本大约 100 倍的减少。此外，我们的延迟可以在小批量时从 5.5ms 到大批量时 20 ms 之间的任何地方。
* 注意在 2k 上下文时，当它达到 BS 120 屋顶线时，吞吐量有效地在 1 token / ms / chip 左右达到平稳（这里是 120，因为我们使用 int8 权重但 bf16 FLOPs）。然而，随着序列长度的增加，我们不再能在内存中适应这个批量大小，所以我们从未达到完全饱和的点。
* 注意在相同吞吐量下大批量大小的延迟有多高，因为 KV 加载变得主导（而不是参数加载）。

我们可以通过将成本和延迟的来源分解为参数加载时间、KV 加载时间和 FLOPs 时间来更好地理解这一点。红色区域是我们期望在 MLP 块中受到计算限制的区域。

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

这讲述了一个相当完整的故事。你可以看到最初，参数加载代表了延迟的绝大部分，直到批量大小变得足够大，FLOPs 和 KV 加载变得更加显著。值得注意的是，在所有大于 2048 的序列长度下，我们在 KV 缓存加载上花费的时间比在 FLOPs 上更多！**所以虽然我们可以通过增加批量大小来改善硬件利用率，但在长上下文长度下 KV 加载总是主导总步骤时间。**

<p markdown=1 class="takeaway">**要点**：对于 LLaMA 3-70B，在几乎所有这些配置中，我们都强烈受到 KV 缓存内存带宽限制（和 HBM 限制），突显了减少 KV 缓存大小对于生成吞吐量有多重要。还要注意这里的延迟/吞吐量权衡仍然多么剧烈。</p>

{% details 这段代码相当简单。 %}

这是计算这些屋顶线的代码：

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
param_size = 70e9  # int8 means 1 byte per param
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

param_size = bytes_per_param * param_count

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(max_num_chips: int = 16):
  # for num_chips in topo_sizes:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  num_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(num_chips <= max_num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们如何非常明确地将延迟分解为两个来源：KV 加载和参数加载，以及延迟如何受到 FLOPs 或通信的限制，以较大者为准。

{% enddetails %}

## 解题示例

这里有几个解题示例。其中一些重复了上面已经解决的问题，但可能在教学上有用。

**问题 1：** LLaMA 3-405B 的每个前向传播每 token 使用多少 FLOPs？假设我们受到 FLOPs 限制，在 TPU v5e 上的 N 个芯片上单个前向传播的下限是什么？如果我们受到通信限制呢？*忽略模型不适合单个芯片的事实。*

**问题 2：** 假设我们想要使用 int8 权重和 int8 KV 缓存以 BS240 部署 LLaMA 3-8B。（a）模型参数（b）KV 缓存和（c）峰值工作激活（大致）使用多少字节？我们可以在什么最小拓扑上运行这个？

**问题 3：** 你如何在 TPU v5e 上部署 LLaMA 3-405B？假设 int8 权重和 bfloat16 FLOPs。假设我们有 15ms / token 的严格限制，我们能实现的最高吞吐量配置是什么？理论最小步骤时间是多少？

<h3 markdown=1 class="next-section">第 8 节就到这里了！对于第 9 节，深入了解 XLA 和 TPU 性能分析，请点击[这里](../profiling)。</h3>
