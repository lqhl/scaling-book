---
layout: distill
title: "如何分析 TPU 程序"
# permalink: /main/
description: "到目前为止, 本系列完全是理论性的: 基于硬件屋顶线的粗略计算. 这种理解能让你走得很远, 但很多优化都归结为实际细节: XLA 编译器如何工作, 以及如何使用 JAX/Tensorboard Profiler 等分析工具来弄清楚当它失败时该怎么做. 我们在这里讨论这个问题."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "Part 8: Serving LLaMA"

next_section_url: ../jax-stuff
next_section_name: "Part 10: JAX"

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
  - name: "TPU 软件栈的千尺视图"
  - name: "TensorBoard Profiler: 一个多功能的 TPU 分析器"
  - subsections:
    - name: "Trace Viewer"
    - name: "如何阅读 XLA 操作"
    - name: "Graph Viewer"
    - name: "看一个真实 (ish) 的示例配置文件"
    - name: "Memory Profile"
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

## TPU 软件栈的千尺视图

谷歌公开了一系列用于编程 TPU 的 API, 从高级的 JAX 代码到低级的 Pallas 或 HLO. 大多数程序员只编写 JAX 代码, 这让你能够编写抽象的 NumPy 风格的线性代数程序, 这些程序会自动编译以在 TPU 上高效运行.

这是一个简单的例子, 一个将两个矩阵相乘的 JAX 程序:

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

通过调用 `jax.jit`, 我们告诉 JAX 跟踪这个函数并发出一个名为 [StableHLO](https://openxla.org/stablehlo) 的低级 IR, 这是一个用于 ML 计算的平台无关 IR, 然后由 XLA 编译器将其降低到 HLO. 编译器运行许多遍以确定融合, 布局和其他因素, 从而产生在 JAX 配置文件中可观察到的 HLO. 这个 HLO 以 LLVM 风格的图视图表示 JAX 代码中的所有核心线性代数操作 (矩阵乘法, 逐点操作, 卷积等). 例如, 这是上面程序的 HLO 删节版<d-footnote>要获取此 HLO, 你可以运行 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`.</d-footnote>:

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

我们稍后会解释 HLO 的语法, 但现在只需注意它实际上与上面的 JAX 代码相当匹配. 例如,

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

是上面实际的矩阵乘法, 它分别沿 0 和 1 维度乘以两个 f32 矩阵.

**为了将这个 HLO 转换为可以在 TPU 上执行的代码, XLA 编译器首先将其降低到 LLO** (低级优化器) IR. LLO 直接对 TPU 进行编程, 调度内存之间的复制, 将数组推送到脉动阵列等. LLO 代码包含将缓冲区推送到脉动阵列, 拉取结果以及调度在不同 TPU 内存片段之间进行通信的 DMA 的原语. 一旦降低到 LLO, 它就会被编译成加载到 TPU IMEM 中并执行的机器代码.

当一个程序运行得比我们希望的慢时, 我们主要在 JAX 级别上工作以提高性能. 然而, 这样做通常需要我们理解 HLO 的一些语义以及代码实际上是如何在 TPU 上运行的. 当在较低级别出现问题时, 我们会拉动另一个逃生舱口, 并用 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 编写自定义内核. 为了查看程序的 HLO 及其运行时统计信息, 我们使用 JAX 分析器.

## JAX Profiler: 一个多功能的 TPU 分析器

JAX 提供了一个多功能的 TPU 分析器, 其中包含一系列有用的工具, 用于了解程序运行时 TPU 上发生的情况. 你可以使用 `jax.profiler` 模块来跟踪正在运行的程序, 并记录从每个子组件的持续时间, 每个程序的 HLO, 内存使用情况等等. 例如, 这段代码会将一个跟踪转储到 `/tmp/tensorboard` 中的一个文件中, 该文件可以在 TensorBoard 中查看 ([这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling) 是一个分步指南).

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

这是你在分析器中可以做的事情的概述:

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

进入 TensorBoard 后, 分析器有几个关键选项卡可以帮助你了解你的程序:

1.  **Trace Viewer** 以时间线的形式显示了 TPU 上实际发生情况的详细时间线.
2.  **Graph Viewer** 显示了 HLO 图, 让你看到程序的哪些部分相互馈送以及事物是如何分片的.
3.  **Memory Profile 和 Memory Viewer:** 这些显示了你的程序正在使用多少内存.

虽然共享配置文件有点困难, 但[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)是一个 Perfetto 链接, 至少包含一个简单 Transformer 的 Trace Viewer 组件. [这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 让你生成完整的 JAX/TensorBoard 跟踪并进行操作.

### Trace Viewer

**Trace Viewer 可能是分析器中最有用的部分.** 下面的例子显示了一个简单的 Transformer, 其中标注了各个部分. 名称来自代码中提供的标签.

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer 显示了每个 TPU 核心上所有操作的时间顺序时间线. 我们在这里只看 TPU:0, 因为通常所有 TPU 都执行相同的指令. 一些关键说明:

1.  顶行 (XLA Ops) 显示了实际的 TPU 操作 (名称是 HLO 名称). 其他所有内容都是基于 `jax.named_scope`, `jax.named_call` 和 Python 堆栈跟踪的近似跟踪.
2.  注意到重复的块, 我们可以在这里隔离单个层. 我们还可以 (通过查看代码/了解 Transformer 的工作原理) 看到哪些部分是注意力, 哪些部分是 MLP.
3.  通过单击 XLA 操作, 我们可以查看它在代码中的来源 (有助于理解跟踪) 并查看指向 Graph viewer 的链接.

<p markdown=1 class="takeaway">**提示:** 你可以使用“视频游戏”风格的控件来导航 Trace Viewer, A/D 左右平移, W/S 缩放. 这些控件使导航变得容易得多.</p>

### 如何阅读 XLA 操作

HLO 实际上并不难读, 它对于理解上面跟踪的给定部分对应的内容非常有帮助. 这是一个名为 fusion.3 的示例操作.

```py
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

让我们把它分解成几个部分.

*   **操作名称**: fusion.3
    *   点或融合操作是一组操作, 最多包含 1 个矩阵乘法和可能的一堆相关的逐点 VPU 操作.
*   **形状/布局**: `bf16[32,32,4096]`
    *   这是操作的输出形状. 我们可以看到 dtype 是 bf16 (每个参数 2 个字节), `[32,32,4096]` 是形状.
*   **布局:** `{2,1,0:T(8,128)(2,1)}`
    *   `{2,1,0:T(8,128)(2,1)}` 告诉我们内存中轴的顺序 (列主序, 行主序等) 和数组填充. 更多内容见下文.
*   **内存位置:** S(1)
    *   S(1) 告诉我们这个数组存在于 VMEM 中. S(0) (有时省略) 是 HBM. S(2) 和 S(3) 是其他内存空间.
*   **参数**: `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
    *   这个操作有一个输入, 一个名为 fusion.32 的 bf16 数组, 具有特定的形状. 这告诉我们哪个函数馈入这个函数.

让我们试着更多地理解这个符号. 让我们以这个简单的例子为例:

`f32[3,5]{1,0:T(2,2)}`

这再次告诉我们, 这个 Op 返回一个形状为 `[3, 5]` 的 float32 数组, 具有特定的分块 `{1,0:T(2,2)}`. 虽然分块不是*太*重要, 但简而言之, 分块告诉我们一个 N 维数组在内存中是如何顺序布局的. 这是一个显示这个数组如何布局的图:

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

在 `{1,0:T(2,2)}` 中, `1,0` 部分告诉我们物理内存中数组维度的顺序, 从最次要到最主要. 你可以从右到左阅读这部分, 并在 `f32[3,5]` 中找出相应的维度, 以确定数组的物理布局. 在这个例子中, 物理布局是 `[3,5]`, 与逻辑形状相同.
之后, `T(2,2)` 告诉我们数组以 `(2, 2)` 的块进行分块, 其中在每个块内, 数组首先是行 (**行主序**), 然后是列, 即 `(0, 0)` 后面是 `(0, 1)`, 然后是 `(1, 0)` 和 `(1,1)`. 由于 `T(2, 2)` 分块, 数组被填充到 `[4, 6]`, 其内存使用量增加了约 1.6 倍. 对于上面给出的大的 bf16 数组, `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`, 我们做 `T(8,128)(2,1)`, 这告诉我们数组有两级分块, 一个外部 `(8, 128)` 分块和一个内部 `(2, 1)` 分块 (用于 bf16, 以便我们的加载总是 4 字节的倍数). 例如, 这是 `bf16[4,8]{1,0,T(2,4)(2,1)}` (颜色是 (2,4) 块, 红色框是 (2,1) 块):

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

分块会影响张量块加载到 VMEM 的效率, XLA 有时会在程序中引入“重新分块”或“重新布局”张量的复制, 有时会产生不小的开销.<d-footnote>JAX 提供了一个实验性功能来解决这个问题, 即允许 XLA 计算其对程序输入的“首选”布局. 当你用 `jax.jit`“即时”编译一个程序时, 你通常会传入“模拟”输入, 告诉 JAX 期望的形状和 dtype. 这些通常也带有 tiling 信息, 可能不是最优的. 相反, 你可以将输入布局指定为 AUTO, `jax.jit` 将返回 jitted 程序首选的布局. 然后你可以显式地以该布局加载张量, 以避免在程序中引起复制.</d-footnote>

### Graph Viewer

虽然上面的一些融合看起来很复杂, 但 XLA Graph Viewer 使它们更容易解析. 例如, 这是一个相当复杂的融合的视图:

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

盯着一堆 HLO 图, 试着将 HLO 操作映射到你正在分析的代码上, 这真的很有帮助. 将鼠标悬停在一个框上, 你通常会看到定义该函数的代码行.

### 看一个真实 (ish) 的示例配置文件

[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 有一个假 Transformer 的示例配置文件. [这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) 是一个 Perfetto 链接, 至少可以让你在匆忙中看到 Trace Viewer. 我比平时更努力地用 `jax.named_scope` 调用来注释跟踪, 以便你可以识别正在发生的事情.

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

看一下配置文件, 试着真正理解每个部分在做什么. 让我们把它分解一下, 从 FFW 块开始:

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

在这里, 我们放大了 FFW 块. 你会看到上投影 Op 是一个融合 (矩阵乘法), 输入为 `bf16[8, 1024, 8192]` 和 `bf16[8192, 16384]`, 输出为 `bf16[32, 1024, 16384]`. 我知道 (因为我写了这段代码) 这是一个 4 路 DP, 2 路 MP 分片矩阵乘法的本地视图, 所以我们实际上在做

**X:** `bf16[32, 1024, 8192]` * **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**我们期望这需要多长时间?** 首先, 我们每个数据并行分片的批量大小是 `8 * 1024 = 8192`, 所以我们应该完全受计算限制. 这是在 8 个 TPUv2 核心上 (在 Google Colab 上免费提供), 所以我们期望它大约需要 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`, 这几乎与实际花费的时间完全相同 (96ms). 太棒了! 这意味着我们获得了极好的 FLOPs 利用率!

**通信呢?** 你会注意到第二个矩阵乘法末尾隐藏的小融合. 如果我们点击它, 你会看到

```py
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

这基本上是一个小的 ReduceScatter (这是 GraphViewer);

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

我们期望这需要多长时间? 嗯, 我们正在一个 TPUv2 4x2 上进行 ReduceScatter, 这应该只需要在 1.2e11 双向带宽上进行一次跳跃. 数组的大小是 `2*32*1024*8192`, 批处理轴分片 4 路, 所以每个分片是 `2*8*1024*8192=134MB`. 所以这大约需要 1.1ms. **实际上需要多长时间?** 配置文件中报告为 1.13ms. 所以我们非常接近屋顶线!

**让我们也看看注意力!** 这是注意力组件的配置文件:

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点击了 Q 投影操作, 它使用一个形状为 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256] 的矩阵 $$W_Q$$. 我们正在沿头维度进行 Megatron 分片. 试着做同样的练习, 计算这些应该需要多长时间.

### Memory Profile

Memory Profile 使查看程序内存随时间的变化变得容易. 这对于调试 OOM 很有帮助. 你可以在这里看到大约 7.5GB 分配给了模型参数, 大约 10GB 空闲. 所以我们可以容纳更多的东西到内存中.

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 已解决的问题

**问题 1**: 看一下[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/配置文件, 找出看起来可疑的地方以及发生了什么. 你能准确地告诉我正在进行什么计算以及每个操作在做什么吗? 所涉及的每个矩阵的真实形状是什么, 它们是如何分片的? *在阅读代码之前, 先试着看一下配置文件.*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 点击这里查看答案. %}

这是两个矩阵乘法, 即具体如下:

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

你可以看到一个 reduce, 两个大的融合, 和一个 all-reduce. 第一个大的融合是:

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

这告诉我们每个分片的形状是 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]` (在 8192 维度上). 通过观察最终的 AllReduce, `replica_groups=\{\{0,16,32,48,64,80,96,112\}, ...\}`, 我们可以看出我们正在进行 8 路模型并行, 所以真实的形状是 `[8, 8192] * bf16[32,768, 8192] -> bf16[8, 32,768]`.

{% enddetails %}

**问题 2:** [之前的 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 实现了一个简单的模拟 Transformer. 按照 Colab 中的说明, 对使用 GSPMD 分区的朴素 Transformer 进行基准测试. 每个部分需要多长时间? 应该需要多长时间? 正在使用什么分片. 尝试修复分片! *提示: 使用 `jax.lax.with_sharding_constraints` 来约束行为. 通过这个修复, 你能得到的最好的 MXU 是多少?*

作为参考, 初始版本每层大约需要 184ms, 优化后的配置文件每层需要 67ms. 完成此操作后, 尝试盯着配置文件, 看看是否可以仅从配置文件中回答这些问题:

-   这是什么分片策略?
-   批量大小, $$d_\text{model}$$, $$d_\text{ff}$$ 是多少?
-   注意力与 MLP 块花费的时间比例是多少?
-   在屋顶线模型下, 每个操作应该花费多少时间?

**注意:** 自从写下这个问题以来, XLA 编译器已经变得更好了. 初始版本现在每层大约需要 90ms, 优化后的配置文件只比它好大约 10ms/层 (80ms/层). 尽管如此, 还是值得玩一玩, 看看你是否能做得更好.

<h3 markdown=1 class="next-section">第 9 部分到此结束. 第 10 部分, 深入探讨 JAX 并行, 点击 [这里](../jax-stuff).</h3>