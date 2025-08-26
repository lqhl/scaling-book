---
layout: distill
title: "在 TPU 上服务 LLaMA 3-70B"
# permalink: /main/
description: "让我们仔细看看如何在 TPU v5e 上服务 LLaMA 3-70B 模型. 在屋顶线模型下, 服务不同模型的成本是多少? 它们的 KV 缓存有多大? 我们应该使用多大的批量大小? 在推理过程中, 参数和激活是如何分片的? 让我们通过一些粗略的估算, 来看看在生产环境中延迟和吞吐量的情况."
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
  - name: "LLaMA 服务的故事是怎样的?"
  - subsections:
    - name: "思考吞吐量"
    - name: "预填充呢?"
  - name: "可视化延迟吞吐量权衡"
  - name: "已解决的问题"

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

*本节将探讨服务 LLaMA-3 需要什么, 以及如何高效地完成. 与之前的“应用”部分一样, 请在查看答案之前尝试自己用笔和纸算出答案!*

## LLaMA 服务的故事是怎样的?

让我们回顾一下 LLaMA 3-70B 的样子 (参考[第 6 节](../applied-training)):

| **超参数** | **值** |
| :------------------: | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

让我们从一个简单的问题开始: **我们应该在什么硬件上提供服务?** 答案基本上是, 无论哪个在 FLOPs/美元方面最便宜.<d-footnote>这并非总是如此, 有时更多的 HBM 或 ICI 带宽比 FLOPs 更重要, 但这是一个很好的启发式方法.</d-footnote> 因此, 我们通常希望在 TPU v5e 上提供服务, 这是我们目前专用的推理芯片 (成本来自 2025 年 2 月的 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)):

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud 美元/小时** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每个 TPU v5e 有 16GB 的 HBM, 这将要求我们相当积极地对模型进行分片. 让我们从思考一些可能对我们很重要的基本量开始:

**问题:** LLaMA 3-70B 每个 token 的 KV 缓存有多大? *你可以假设我们用 int8 存储它们. 这决定了我们在给定拓扑上的批量大小能有多大.*

{% details 点击这里, 在你思考之后! %}

LLaMA 3-70B 有 8 个 KV 头, 所以每个 token 的大小是 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`.

**注意这有多大!** 如果我们有一个 32k token 的序列长度 (这很常见), 这将使用 `162e3 * 32,768 = 5.3GB / 序列`. 对于 BS=240, 这是 1.3TB! 由于 TPU v5e 每个只有 16GB, 我们大约需要 `(70e9 + 1.3e12) / 16e9 = 86` 个 TPU v5e 芯片才能容纳这么多内存. 另请注意, 与 70GB 的模型参数相比, 这有多大.

{% enddetails %}

**问题:** 假设我们想在批量大小为 32, 序列长度为 8192 的情况下, 用 int8 服务 L3 70B (参数和 KV). 这将使用多少总内存? 我们可以用多小的切片来服务它?

{% details 答案 %}

由于我们的 KV 在 int8 中是 `160e3` 字节, 我们的总 KV 内存是 `160e3 * 8192 * 32 = 41.9e9` 字节. 我们的参数是 `70e9` 字节, 因为我们每个参数有 1 个字节. 因此, 我们的总内存使用量是 `41.9e9 + 70e9 = 112GB`.

我们可以使用的最小切片将有 `112e9 / 16e9 = 7` 个 TPU, 或者 (四舍五入到一个偶数大小), TPU v5e `4x2`. 这将是一个紧张的配合, 我们可能无法完全容纳, 考虑到其他开销, 所以我们可能至少需要一个 `4x4` (或者减小批量大小).

{% enddetails %}

**问题:** 在这个批量大小和量化下, 在 TPU v5e `4x2` 上, 我们每个解码步骤大约会期望什么样的延迟? 吞吐量 (token/秒/芯片) 是多少? `4x4` 呢? *假设我们用 bfloat16 执行 FLOPs, 并且一切都完全分片.*

{% details 答案 %}

我们可以调用上一节的公式

$$\begin{align*}
\tiny \text{理论步骤时间 (通用)} = \underbrace{\frac{\text{批量大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力 (总是受带宽限制)}} + \underbrace{\max\left(\frac{2 \times \text{批量大小} \times \text{参数数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP (可以受计算限制)}}
\end{align*}$$

在这里, 我们的临界批量大小大约是 120, 因为我们的参数是 int8, 但我们的 FLOPs 是 bfloat16. 我们也可以手动计算 RHS 的最大值, 但这基本上是我们已经做过好几次的计算. **所以我们在矩阵乘法和 FLOPs 方面都完全处于受内存限制的状态.**

严格地看内存带宽, 我们的步骤时间基本上是 `(KV 大小 + 参数大小) / (8 * HBM 带宽) = 112e9 / (8 * 8.1e11) = 17ms`. **所以理论上我们的步骤时间大约是 17ms.** 我们的吞吐量将是 `32 / .017 = 1882 token/秒`, 或 `1882 / 8 = 235 token/秒/芯片`.

这里有一个警告, 就是要检查我们是否可能在矩阵乘法上受 ICI 限制. 我们可以在这里为其分配 2 个轴, 所以理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时, 我们受 ICI 限制, 所以我们没问题!

如果我们要在 `4x4` 上运行, 我们在 ICI 方面仍然没问题, 所以我们的延迟会降到 `17 / 2 = 8.5ms`, 但我们每个芯片的吞吐量将保持不变.

{% enddetails %}

### 思考吞吐量

让我们花点时间纯粹地思考吞吐量. 当我们优化吞吐量时, 我们希望受计算限制, 这意味着我们接近于利用所有 TPU MXU 容量. 通常这意味着我们希望批量大小尽可能大, 以便我们做尽可能多的工作.

**问题:** 在 TPU v5e 上, 使用 bfloat16 权重和激活, 我们的批量大小需要多大才能在矩阵乘法中受计算限制? 如果我们使用 int8 权重但用 bfloat16 执行 FLOPs 呢? 如果使用 int8 权重和 int8 FLOPs 呢?

{% details 答案 %}

正如第 7 节所讨论的, 对于任何 $B \ll D, F$ 的 bfloat16 矩阵乘法, 我们有

$$\begin{equation*}
    T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM 带宽}} = 240
\end{equation*}$$

当我们的权重是 int8 时, 我们在分母中损失了 2 倍, 所以我们有 $2BDF / DF = 2B > 240$, 或者同样地 $B > 120$, 是之前临界批量大小的一半. 这对我们真的很有帮助! 当我们使用 int8 权重和 int8 FLOPs 时, 我们必须使用 TPU FLOPs/s 的 int8 值, 它从 bfloat16 的 1.97e14 增加到 3.94e14, 几乎翻了一番. 这意味着我们回到了大约 $B > 240$ 的起点.

int8 权重和 bfloat16 FLOPs 的情况相当普遍, 因为无损地量化参数通常比进行低精度算术更容易.

{% enddetails %}

**问题:** 我们可以用 bfloat16, int8 和 int4 (KV 和参数) 在 8k 上下文下服务 LLaMA 3-70B 的最小 TPU v5e 拓扑是什么? *你可以认为 KV 缓存对于这个问题可以忽略不计.*

{% details 答案 %}

这很简单! 如果我们对一个很小的批量大小感到满意, 那么唯一的限制就是将参数内存放入 HBM, 即它只是 `ceil(num_params * sizeof(dtype) / HBM per TPU`, 或 `ceil(70e9 * sizeof(dtype) / 16e9)` 四舍五入到最近的合理拓扑 (2 的某个倍数):

| dtype | 参数大小 | KV 大小/token (字节) | 最小 TPU v5es | 实际最小切片 | 剩余 HBM 用于 KV 缓存 | 8k 时的 KV 缓存数量 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             29              |         43         |

这很酷! 它告诉我们, 如果我们愿意, 我们可以将 LLaMA 70B 放在一个 TPU v5e 2x2 上. 除了你会注意到 KV 缓存的数量非常少. 那是我们的批量大小! 这意味着我们将获得糟糕的 FLOPs 利用率. 我们会非常乐意使用更大的拓扑来将我们的批量大小推高到 240.

{% enddetails %}

**问题:** 假设我们使用适合这些拓扑的最大批量大小, 我们每个生成步骤可以期望什么样的延迟?

{% details 答案 %}

这也很简单, 因为我们正在选择我们的批量大小来填满我们所有的 HBM! 这只是一个问题, 即将一个完整的 TPU v5e 的字节加载到 MXU 需要多长时间. 这只是 `v5e HBM / v5e HBM 内存带宽 = 16GB / 8.2e11 = 19ms`, 所以这是**每步 19ms**. 假设我们的生成中位长度为 512 个 token, 那么每次解码大约需要 9 秒. 请注意, 我们可以通过更小的批量大小获得稍微更好的延迟, 例如, 如果我们只看 int4 中的模型参数, 我们的最小延迟大约是每步 10ms, 因为 HBM 不再是满的.

{% enddetails %}

<p markdown=1 class="takeaway">**要点**: 我们可以通过询问将所有模型的参数从 HBM 加载到 MXU 需要多长时间来为解码延迟设定下限. 当我们的 KV 缓存很小时, 你可以认为每一层只是逐块加载权重, 然后丢弃它们. 除非我们使用大的批量大小或大量的设备间通信, 否则这通常是一个合理的界限 (在 1.5 倍以内). 当我们的批量大小更大时, 我们也需要对 KV 缓存加载进行建模, 因为它主导了参数.</p>

同样, 在受 FLOPs 限制的情况下 (例如, 训练或大批量推理), 我们可以使用 $$\text{总 FLOPs} / (N \cdot C) = 2 \cdot \text{参数数量} \cdot B / (N \cdot C)$$ 下限, 这假设没有通信.

**问题:** 对于这些中的每一个, 每个芯片的吞吐量是多少 (以查询/芯片为单位)? *你可以假设我们的中位解码长度是 512 个 token.*

{% details 答案 %}

这是一个重要的问题, 因为它与每个 token 的成本完全相关.

根据我们对中位解码长度的假设, 我们的吞吐量就是 $$B / (\text{每步延迟} \cdot \text{中位步数} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$. 这给了我们大约 $$(4.42 / N)$$ QPS, 所以代入 $$N$$ 我们得到:

|  dtype   | QPS / 芯片 |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

请注意, 这是相当乐观的, 因为它完全忽略了前向传播的工作内存 (分配给激活和注意力的内存). 这对于 Flash Attention 来说并非荒谬, 但也不现实. 真实的数字可能大约是这个的一半. 为了获得绝对最大的吞吐量, 我们可能需要将芯片数量增加一倍以上, 并显著增加批量大小.

{% enddetails %}

**问题:** 如果我们将上述每个示例的拓扑加倍, 我们的峰值吞吐量会如何变化?

{% details 答案 %}

如果我们我们在 bfloat16 中使用 4x8 切片, 我们将有 372GB 的剩余空间用于 KV 缓存, 这将使我们的批量大小增加到 140. 然后, 由于我们的步骤时间将保持不变, 我们的吞吐量将是 `16.54 / num_chips`, 或

|       dtype       | QPS / 芯片 |
| :---------------: | :--------: |
| bfloat16 (在 4x8 上) |    0.51    |
|   int8 (在 4x4 上)   |    1.03    |
|   int4 (在 2x4 上)   |    2.06    |

进一步增加将带来更大的胜利! 重要的结论是, **在所有情况下, 最小的拓扑并非性能最高的拓扑**, 如果我们受 KV 缓存大小的限制.

{% enddetails %}

**问题:** 现在让我们深入探讨分片的问题. 假设我们想在 TPU v5e 4x8 上用 bfloat16 提供服务. 在生成期间, 我们将在 TPU v5e 4x8 上为我们的模型使用什么分片? 我们能避免受通信限制吗?

{% details 答案 %}

正如上一节所讨论的, 我们在生成期间实际上只有一个分片选项: 模型并行. 在我们变得受通信限制之前, 我们可以做多少? 正如我们在上一节中讨论的, 我们的模型大约在

$$Y > \frac{F \cdot M_Y}{2200}$$

时变得受通信限制.

对于 LLaMA 3-70B, 我们有 `F = 28,672`, 所以如果我们进行 2 个轴的模型分片, 这给了我们大约 $$Y = 28672 \cdot 2 / 2200 = 26$$, 所以一般来说, 我们可以扩展到大约 16 个芯片而不受通信限制, 这让我们使用 `4x4` 而不是 `4x8`. 通常, 由于我们不能完美地重叠计算, 即使这个估计也过于乐观.

**要点: 我们实际上不能用纯模型并行在 4x8 上提供服务.** 我们能做的最好的就是 4x2 或者*也许*是 4x4.

然而, 正如我们所讨论的, 当我们的批量大小很小时, 我们通常可以进行更多的模型并行, 而不会显著影响吞吐量, 因为我们的模型受内存带宽限制, 而不是受 FLOPs 限制. 我们之前说过, 这个值大约是 $Y=F / (8\cdot B)$, 所以如果我们批量大小为 64, 理论上我们可以进行高达 `Y = 28,672 / (8 * 64) = 56` 路模型并行, 然后才会受 ICI 限制. 为了验证这一点, 我们可以查看单个矩阵乘法的 $T_\text{ici comms}$, $T_\text{hbm comms}$ 和 $T_\text{math}$. 我们清楚地有:

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

对于 `4x8`, 这将给我们 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`, $T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`, 以及 $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`, 所以理论上我们仍然受 HBM 带宽限制, 这很棒! *请注意, 从 `4x4` 扩展到 `4x8` 从吞吐量的角度来看可能没有帮助, 但它会减少我们的延迟!* 

如果我们看 int8 和 int4 配置, 我们*可以*用纯模型并行来做这些. 所以我们已经到了一个点, 量化实际上给了我们一个有意义的优势, 超越了更快的 FLOPs: 它让我们在使用更大的批量大小之前变得受通信限制. **所以这个故事的结局是, 我们无法在 4x8 上实现峰值吞吐量, 但对于 int8 和 int4 配置, 我们可以进行纯模型并行*.

{% enddetails %}

<p markdown=1 class="takeaway">**提示**: 有用的模型并行的最大数量取决于 $$d_{ff}$$ 和你对模型进行分片的轴数. 最大值通常在 8 到 32 之间, 具体取决于模型大小. 你可以扩展到这个限制之外以提高延迟, 但会牺牲一些吞吐量.</p>

### 预填充呢?

我们在这里基本上忽略了预填充, 因为它要简单得多. 让我们把几个概念放在一起, 思考一下端到端的画面.

**问题:** 假设我们的中位预填充长度为 8192 个 token, 中位解码长度为 4096 个 token. 假设我们的生成批量大小为 32. 平均每个步骤有多少个序列完成解码? 平均每个步骤从我们的 KV 缓存中驱逐多少个 token?

{% details 答案 %}

这有点直接. 由于我们的中位解码长度为 4096 个 token, 一个序列大约每 1/4096 个 token 完成一次. 给定 32 的批量大小, 这意味着我们每个步骤驱逐 `32 / 4096` 个序列. 由于我们的 KV 缓存长度大约是 `8192 + 4096`, 这是每个步骤驱逐 `32 * (8192 + 4096) / 4096 = 96` 个 token. 一般公式是 $B * (P + G) / G$, 其中 $P$ 和 $G$ 是预填充和生成长度.

{% enddetails %}

**问题:** 假设我们进行分离式服务, 中位预填充长度为 8192, 中位解码长度为 512. 假设上面计算的 bfloat16 中的预填充和生成延迟. 你需要多少比例的预填充:生成服务器才能使两者都完全饱和.

{% details 答案 %}

这是一个有趣的问题. 设 $P$ 是预填充服务器的数量, $G$ 是生成服务器的数量. 所以一般来说, 这是一个流水线问题, 我们以 `P / prefill_latency` 的速率输入序列, 并以 `B * G / (generate_latency * median_decode_length)` 的速率消耗它们. 我们计算出每个预填充步骤为 `910ms`, 每个解码步骤为 `19ms`, 批量大小为 43 (我们称之为 32). 因此我们需要 `P / 0.91 = 32 * G / (0.019 * 512)` 或 `P = 3G`, 即我们需要大约 3 倍于生成服务器的预填充服务器!

{% enddetails %}

## 可视化延迟吞吐量权衡

继续以 LLaMA 70B 为例, 让我们实际看看在生成期间不同批量大小的延迟和吞吐量. 正如我们在上一节中为 PaLM 模型展示的那样, 这给了我们一个吞吐量/延迟的帕累托前沿. 让我们假设 16 路张量并行, 因为这是我们在 MLP 块中保持受计算限制的合理界限. 我们将在这里使用 TPU v5e 4x4 拓扑. **滑块控制序列长度, 以便你可以看到更大 KV 缓存的效果.**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

*   **看看成本和延迟之间的权衡有多么巨大.** 以将每个 token 的延迟加倍为代价, 我们可以实现每个 token 成本大约 100 倍的降低. 此外, 我们的延迟可以从低批量大小的 5.5 毫秒到非常大批量的 20 毫秒不等.
*   请注意, 在 2k 上下文下, 当达到 BS 120 屋顶线 (这里是 120, 因为我们使用 int8 权重但 bf16 FLOPs) 时, 吞吐量实际上每个芯片约 1 token/毫秒. 然而, 随着序列长度的增加, 我们无法再将这个批量大小放入内存中, 因此我们永远无法达到完全饱和的点.
*   请注意, 在相同吞吐量下, 大批量大小的延迟要高得多, 因为 KV 加载变得占主导地位 (而不是参数加载).

我们可以通过将成本和延迟的来源分解为参数加载时间, KV 加载时间和 FLOPs 时间来更好地理解这一点. 红色区域是我们期望在 MLP 块中受计算限制的区域.

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

这讲述了一个相当长的故事. 你可以看到, 最初, 参数加载占了延迟的绝大部分, 直到批量大小变得足够大, FLOPs 和 KV 加载变得更加重要. 值得注意的是, 在所有大于 2048 的序列长度下, 我们在 KV 缓存加载上花费的时间比在 FLOPs 上花费的时间更多! **因此, 虽然我们可以通过增加批量大小来提高硬件利用率, 但在长上下文长度下, KV 加载始终主导总步骤时间.**

<p markdown=1 class="takeaway">**要点:** 对于 LLaMA 3-70B, 我们在几乎所有这些配置中都严重受 KV 缓存内存带宽限制 (和 HBM 限制), 这突显了减少 KV 缓存大小对于生成吞吐量的重要性. 另请注意, 延迟/吞吐量的权衡在这里仍然非常巨大.</p>

{% details 这段代码很简单. %}

这是计算这些屋顶线的代码:

```py
import numpy as np

num_chips = 16  # 我们将 16 固定为我们做的总模型并行的数量
param_size = 70e9  # int8 意味着每个参数 1 个字节
sequence_length = 8192  # 可以改变这个

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

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # 获取可以容纳的最大批量大小
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # 在 2ND 意义上大致正确

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # 生成时总是受带宽限制

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

请注意我们如何非常明确地将延迟分解为两个来源: KV 加载和参数加载, 以及延迟如何受 FLOPs 或通信的限制, 以较大者为准.

{% enddetails %}

## 已解决的问题

这里有几个已解决的问题. 其中一些重复了上面已经解决的问题, 但可能在教学上很有用.

**问题 1:** LLaMA 3-405B 的每个前向传播每个 token 使用多少 FLOPs? 假设我们受 FLOPs 限制, 在 N 个 TPU v5e 芯片上的单个前向传播的下限是多少? 如果我们受通信限制呢? *忽略模型不适合单个芯片的事实.*

**问题 2:** 假设我们想用 BS240, int8 权重和 int8 KV 缓存来服务 LLaMA 3-8B. (a) 模型参数 (b) KV 缓存和 (c) 峰值工作激活 (大致) 使用多少字节? 我们可以运行这个的最小拓扑是什么?

**问题 3:** 你将如何在 TPU v5e 上服务 LLaMA 3-405B? 假设 int8 权重和 bfloat16 FLOPs. 假设我们有 15ms/token 的严格限制, 我们可以实现的最高吞吐量配置是什么? 理论上的最小步骤时间是多少?

<h3 markdown=1 class="next-section">第 8 部分到此结束! 第 9 部分, 深入探讨 XLA 和 TPU 分析, 点击 [这里](../profiling).</h3>