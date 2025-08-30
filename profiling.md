---
layout: distill
title: "如何分析TPU程序"
# permalink: /main/
description: "到目前为止，本系列完全是理论性的：基于硬件性能上限的粗略计算。这种理解让您走得很远，但很多优化归结为实际细节：XLA编译器如何工作以及如何使用JAX/Tensorboard Profiler等分析工具来找出失败时的解决方案。我们在这里讨论这个。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "第 8 部分：部署 LLaMA"

next_section_url: ../jax-stuff
next_section_name: "第 10 部分：JAX"

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
  - name: "TPU 软件栈的千英尺视角"
  - name: "TensorBoard Profiler：多用途 TPU 分析器"
  - subsections:
    - name: "Trace Viewer"
    - name: "如何阅读 XLA 操作"
    - name: "Graph Viewer"
    - name: "查看一个真实的示例分析"
    - name: "内存分析"
  - name: "实践问题"

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

## TPU 软件栈的千英尺视角

Google 提供了多种用于编程 TPU 的 API，从高级 JAX 代码到底层 Pallas 或 HLO。大多数程序员专门编写 JAX 代码，这让你可以编写抽象的 NumPy 风格线性代数程序，这些程序会被自动编译以在 TPU 上高效运行。

这是一个简单的例子，一个将两个矩阵相乘的 JAX 程序：

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

通过调用 `jax.jit`，我们告诉 JAX 追踪这个函数并生成一个称为 [StableHLO](https://openxla.org/stablehlo) 的底层 IR，这是一个与平台无关的 ML 计算 IR，它又被 XLA 编译器转换为 HLO。编译器运行多个 pass 来确定融合、布局和其他因素，这些因素导致了在 JAX 分析中可观察到的 HLO。这个 HLO 以 LLVM 风格的图视图表示 JAX 代码中的所有核心线性代数操作（矩阵乘法、逐点操作、卷积等）。例如，这里是上述程序作为 HLO 的简化版本<d-footnote>要获得这个 HLO，你可以运行 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`。</d-footnote>：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

我们将在下一秒解释 HLO 的语法，但现在只需注意它实际上与上面的 JAX 代码相当匹配。例如，

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

是上面实际的矩阵乘法，它分别沿着第 0 维和第 1 维将两个 f32 矩阵相乘。

**为了将这个 HLO 转换为可以在 TPU 上执行的代码，XLA 编译器首先将其转换为 LLO**（low-level optimizer，低层优化器）IR。LLO 直接对 TPU 进行编程，调度内存之间的拷贝，将数组推送到脉动阵列等。LLO 代码包含将缓冲区推送到脉动阵列、拉取结果以及调度在不同 TPU 内存片段之间通信的 DMA 的原语。一旦这被转换为 LLO，它就会被编译为机器代码，加载到 TPU IMEM 中并执行。

当程序运行速度比我们期望的慢时，我们主要在 JAX 层面上工作以提高性能。然而，这样做通常需要我们理解一些 HLO 的语义以及代码如何在 TPU 上实际运行。当在较低层出现问题时，我们使用另一个逃生舱门，在 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 中编写自定义内核。要查看程序的 HLO 及其运行时统计信息，我们使用 JAX profiler。

## JAX Profiler：多用途 TPU 分析器

JAX 提供了一个多用途 TPU 分析器，包含许多有用的工具，用于理解程序运行时 TPU 上发生的情况。你可以使用 `jax.profiler` 模块来追踪正在运行的程序，并记录从每个子组件的持续时间、每个程序的 HLO、内存使用等所有信息。例如，这段代码会将追踪转储到 `/tmp/tensorboard` 中的文件，可以在 TensorBoard 中查看（[这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling) 是分步指南）。

```python
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# 现在你可以在 Google Colab 中加载 TensorBoard
#
# !pip install tensorboard tensorboard-plugin-profile
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# 或者在外部使用
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

这是你可以在分析器中执行的操作概述：

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

进入 TensorBoard 后，分析器有几个关键的标签页可以帮助你理解你的程序：

1. **Trace Viewer** 显示 TPU 上实际发生情况的详细时间线。
2. **Graph Viewer** 显示 HLO 图，让你看到程序的哪些部分相互输入以及如何分片。
3. **Memory Profile 和 Memory Viewer：** 这些显示你的程序使用了多少内存。

虽然共享分析文件有点困难，但[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)是一个 Perfetto 链接，包含至少一个简单 Transformer 的 Trace Viewer 组件。[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 让你可以生成完整的 JAX/TensorBoard 追踪并进行实验。

### Trace Viewer

**Trace Viewer 可能是分析器中最有用的部分。** 下面的例子显示了一个带有注释部分的简单 Transformer。名称来自代码中提供的标签。

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer 显示每个 TPU 核心上所有操作的时间顺序时间线。我们在这里只看 TPU:0，因为通常所有 TPU 执行相同的指令。几个关键点：

1. 顶行（XLA Ops）显示实际的 TPU 操作（名称是 HLO 名称）。其他都是基于 `jax.named_scope`、`jax.named_call` 和 Python 堆栈跟踪的近似追踪。
2. 注意重复的块，我们可以在这里隔离单个层。我们还可以看到（通过查看代码/理解 Transformer 如何工作）哪些部分是注意力，哪些部分是 MLP。
3. 通过点击 XLA 操作，我们可以查看它来自代码的哪个位置（对理解追踪很有用）并看到 Graph viewer 的链接。

<p markdown=1 class="takeaway">**提示：** 你可以使用"视频游戏"风格控件导航 Trace Viewer，A/D 左右平移，W/S 放大缩小。这些控件使导航变得更加容易。</p>

### 如何阅读 XLA 操作

HLO 实际上并不难读，它对于理解上面追踪的给定部分对应什么非常有帮助。这是一个名为 fusion.3 的示例操作。

```py
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

让我们将其分解成各个部分。

* **操作名称**：fusion.3
  * 点积或融合操作是一组包含最多 1 个矩阵乘法和可能一堆相关逐点 VPU 操作的操作。
* **形状/布局**：`bf16[32,32,4096]`
  * 这是操作的输出形状。我们可以看到 dtype 是 bf16（每个参数 2 字节），`[32,32,4096]` 是形状。
* **布局**：`{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}` 告诉我们内存中轴的顺序（列优先、行优先等）和数组填充。更多内容见下文。
* **内存位置**：S(1)
  * S(1) 告诉我们这个数组存在于 VMEM 中。S(0)（有时省略）是 HBM。S(2) 和 S(3) 是其他内存空间。
* **参数**：`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * 这个操作有一个输入，一个名为 fusion.32 的 bf16 数组，具有特定形状。这告诉我们什么函数输入到这个函数中。

让我们尝试更多地理解这个符号。让我们以这个作为简单例子：

`f32[3,5]{1,0:T(2,2)}`

这再次告诉我们这个操作返回一个形状为 `[3, 5]` 的 float32 数组，具有特定的平铺 `{1,0:T(2,2)}`。虽然平铺不是*太*重要，但简单地说，平铺告诉我们 N 维数组如何在内存中顺序布局。这是一个显示这个数组如何布局的图表：

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

在 `{1,0:T(2,2)}` 中，`1,0` 部分告诉我们物理内存中数组维度的顺序，从最次要到最重要。你可以从右到左读取这部分，并在 `f32[3,5]` 中挑出相应的维度，以确定数组的物理布局。在这个例子中，物理布局是 `[3,5]`，与逻辑形状相同。
之后，`T(2,2)` 告诉我们数组被平铺成 `(2, 2)` 的块，其中在每个块内，数组先有行（**行优先**），然后是列，即 `(0, 0)` 后跟 `(0, 1)`，然后是 `(1, 0)` 和 `(1,1)`。由于 `T(2, 2)` 平铺，数组被填充到 `[4, 6]`，将其内存使用扩展了约 1.6 倍。对于上面给出的大 bf16 数组，`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`，我们做 `T(8,128)(2,1)`，这告诉我们数组有两级平铺，一个外部的 `(8, 128)` 平铺和该单元内的内部 `(2, 1)` 平铺（用于 bf16，所以我们的加载总是 4 字节的倍数）。例如，这里是 `bf16[4,8]{1,0,T(2,4)(2,1)}`（颜色是 (2,4) 平铺，红色框是 (2,1) 平铺）：

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

平铺可以影响张量块加载到 VMEM 的效率，XLA 有时会引入在程序内部"重新平铺"或"重新布局"张量的拷贝，有时会产生不小的开销。<d-footnote>JAX 提供了一个实验性功能来解决这个问题，允许 XLA 计算程序输入的"首选"布局。当你使用 `jax.jit` "即时"编译程序时，你通常传入"模拟"输入，告诉 JAX 期望什么形状和 dtype。这些通常也携带可能不是最优的平铺信息。相反，你可以将输入布局指定为 AUTO，`jax.jit` 将返回 jitted 程序偏好的布局。然后你可以显式地以该布局加载张量，以避免在程序内部引入拷贝。</d-footnote>

### Graph Viewer

虽然上面的一些融合可能看起来很复杂，但 XLA Graph Viewer 使它们更容易解析。例如，这里是一个相当复杂的融合的视图：

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

盯着一堆 HLO 图并尝试将 HLO 操作映射到你正在分析的代码上真的很有帮助。将鼠标悬停在一个框上，你通常会看到定义函数的代码行。

### 查看一个真实的示例分析

[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 有一个虚假 Transformer 的示例分析。[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) 是一个 Perfetto 链接，至少可以看到 Trace Viewer（如果你赶时间的话）。我比平时付出了更多的努力，用 `jax.named_scope` 调用来注释追踪，这样你就可以识别发生了什么。

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

看看分析并尝试真正理解每个部分在做什么。让我们将其分解一点，从 FFW 块开始：

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

在这里我们放大了 FFW 块。你会看到上投影操作是一个融合（矩阵乘法），输入为 `bf16[8, 1024, 8192]` 和 `bf16[8192, 16384]`，输出为 `bf16[32, 1024, 16384]`。我知道（因为我写了这段代码）这是一个 4 路 DP、2 路 MP 分片矩阵乘法的本地视图，所以我们实际上在做

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**我们期望这需要多长时间？** 首先，我们每个数据并行分片的批量大小是 `8 * 1024 = 8192`，所以我们应该是严格的计算限制。这是在 8 个 TPUv2 核心上（在 Google Colab 上免费可用），所以我们期望它需要大约 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`，这几乎正好是它需要的时间（96ms）。这太好了！这意味着我们获得了极好的 FLOPs 利用率！

**通信呢？** 你会注意到在第二个矩阵乘法末尾隐藏的小融合。如果我们点击它，你会看到

```py
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

这基本上是一个小的 ReduceScatter（这是 GraphViewer）；

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

我们期望这需要多长时间？嗯，我们在 TPUv2 4x2 上做 ReduceScatter，这应该只需要在 1.2e11 双向带宽上的一跳。数组大小为 `2*32*1024*8192`，批量轴分片 4 种方式，所以每个分片是 `2*8*1024*8192=134MB`。所以这应该需要大约 1.1ms。**它实际需要多长时间？** 分析中报告的是 1.13ms。所以我们非常接近 roofline！

**让我们也看看注意力！** 这是注意力组件的分析：

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点击了 Q 投影操作，它使用形状为 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256] 的矩阵 $$W_Q$$。我们沿着头维度进行 Megatron 分片。尝试做同样的练习，计算这些应该需要多长时间。

### 内存分析

内存分析可以很容易地看到程序内存作为时间的函数。这对于调试 OOM 很有帮助。你可以在这里看到大约 7.5GB 分配给模型参数，大约 10GB 空闲。所以我们可以将更多内容放入内存。

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 实践问题

**问题 1**：看看[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/分析，弄清楚什么看起来可疑以及这里发生了什么。你能确切地告诉我正在发生什么计算以及每个操作在做什么吗？涉及的每个矩阵的真实形状是什么，它们是如何分片的？*尝试先看分析而不阅读代码。*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 点击这里查看答案。 %}

这是两个矩阵乘法，即具体这个：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

你可以看到一个 reduce、两个大融合和一个 all-reduce。第一个大融合是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

这告诉我们每个分片的形状是 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（在 8192 维度上）。通过观察带有 `replica_groups=\{\{0,16,32,48,64,80,96,112\}, ...\}` 的最终 AllReduce，我们可以看出我们在做 8 路模型并行，所以真实形状是 `[8, 8192] * bf16[32,768, 8192] -> bf16[8, 32,768]`。

{% enddetails %}

**问题 2：** [之前的 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 实现了一个简单的模拟 Transformer。按照 Colab 中的说明，获得带有 GSPMD 分片的朴素 Transformer 的基准测试。每个部分需要多长时间？应该需要多长时间？使用的是什么分片。尝试修复分片！*提示：使用 `jax.lax.with_sharding_constraints` 来约束行为。通过这个修复，你能获得的最佳 MXU 是什么？*

作为参考，初始版本大约获得 184ms/层，优化的分析获得 67ms/层。完成这个后，尝试盯着分析，看看你是否能纯粹从分析中回答这些问题：

- 这是什么分片策略？
- 批量大小 $$d_\text{model}$$，$$d_\text{ff}$$ 是什么？
- 注意力与 MLP 块花费的时间比例是多少？
- 在 roofline 下，每个操作应该花费多少时间比例？

**注意：** 自从编写这个问题以来，XLA 编译器变得更好了。初始版本现在大约是 90ms/层，优化的分析只比它好大约 10ms/层（80ms/层）。尽管如此，它仍然值得玩玩，看看你是否能做得更好。

<h3 markdown=1 class="next-section">第 9 部分到此结束。有关 JAX 并行的深入探讨，请点击[这里](../jax-stuff)查看第 10 部分。</h3>
