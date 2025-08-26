---
layout: distill
title: "在 JAX 中编程 TPU"
# permalink: /main/
description: "如何使用 JAX 高效地编程 TPU! 本节的大部分内容摘自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>这里</a>. 你可以在<a href='https://colab.sandbox.google.com/'>Google Colab</a>上使用免费的 TPU 运行本节中的代码示例."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "Part 9: Profiling"

next_section_url: ../conclusion
next_section_name: "Part 11: Conclusions"

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "JAX 中的并行是如何工作的?"
  - subsections:
    - name: "自动分片模式"
    - name: "显式分片模式"
    - name: "通过 shard_map 的手动分片模式"
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

## JAX 中的并行是如何工作的?

JAX 支持三种多设备编程的思想流派:

1.  **编译器, 接管吧!** 让 XLA 编译器自动对数组进行分区, 并决定添加什么通信来促进给定的程序. 这让你能够将一个在单个设备上运行的程序自动地在数千个设备上运行, 而无需更改任何内容.
2.  **JAX, 接管吧!** 自动并行很棒, 但有时编译器会做一些疯狂的事情. 显式分片让你能够像往常一样编写单设备代码, 但让 JAX 处理分片传播 (而不是编译器). 这意味着当 JAX 不清楚你想要什么时, 它可以向你寻求澄清.
3.  **就让我写我想要的意思, 该死!** 虽然编译器很好, 但它们有时会做错事, 并添加你不想添加的通信. 有时我们希望明确地说明你打算运行的确切通信.

| 模式 | 视图? | 显式分片? | 显式集合? |
|:---:|:---:|:---:|:---:|
| 自动 | 全局 | ❌ | ❌ |
| 显式 | 全局 | ✅ | ❌ |
| 手动 | 每个设备 | ✅ | ✅ |

相应地, JAX 为每种模式提供了 API:

1.  `jax.jit` (带有 `Auto` 网格轴) 让你能够获取任何现有的 JAX 函数, 并用分片输入调用它. 然后 JAX 使用 XLA 的 [Shardy](https://openxla.org/shardy) 编译器, 该编译器会自动并行化程序. XLA 会在需要时为你添加通信 (AllGathers, ReduceScatters, AllReduces 等) 以促进现有操作. 虽然它并不完美, 但它通常在自动将你的程序扩展到任意数量的芯片而无需更改代码方面做得很好.
2.  带有 `Explicit` 网格轴的 `jax.jit` 看起来与 (1) 类似, 但让 JAX 处理分片传播而不是 XLA. 这意味着数组的分片实际上是 JAX 类型系统的一部分, 当 JAX 检测到不明确的通信时可以报错, 并让用户解决它.
3.  `jax.shard_map` 是更手动的对应物. 你会得到一个程序的设备本地视图, 并且必须显式地编写你想要的任何通信. 有一个分片数组, 并希望在每个设备上都有完整的东西? 添加一个 `jax.lax.all_gather`. 想要在你的设备上对一个数组求和? 添加一个 `jax.lax.psum` (一个 AllReduce). 编程更难, 但不太可能做一些你不想做的事情.

<h3 id="auto-sharding-mode">自动分片模式</h3>

jax.jit 在 JAX 中扮演两个角色. 顾名思义, 它“即时”将一个函数从 Python 编译成字节码 (通过 XLA/HLO/LLO), 以便它运行得更快. 但是如果输入是分片的, 或者用户指定了 `in_sharding` 或 `out_sharding`, 它也让 XLA 将计算分布在多个设备上, 并根据需要添加通信. 例如, 这是你如何使用 jax.jit 编写一个分片矩阵乘法:

```py
import jax
import jax.numpy as jnp

# 在 TPU v5e 4x2 上运行. 这为硬件的两个物理轴分配了名称.
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 这告诉 JAX 对所有操作都使用这个网格, 所以你只需要指定 PartitionSpec P.
jax.set_mesh(mesh)

# 我们创建一个矩阵 W 和输入激活 In, 它们在我们的设备上进行了分片.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 我们可以在这里显式地编译分片矩阵乘法函数. 这会添加所有
# 必要的通信 (例如, 在矩阵乘法之后进行 AllReduce).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这将自动运行任何分片, 并在我们的设备上对计算进行分区. **但实际上在硬件级别发生了什么?**

1.  首先, 我们创建在我们的设备上分片的 In 和 W<d-footnote>注意我们是如何做到这一点的. 这是创建具有特定分片的数组的一种方法 (即, 通过向创建函数添加 device 参数). 另一种方法是正常地用 `jnp.array(....)` 创建一个数组, 然后执行例如 `jax.device_put(..., P('x', 'y'))`. 还有一种方法是编写一个创建你想要的数组的函数, 并用你想要的 `out_shardings` 对其进行 jit 编译.</d-footnote>. W 沿收缩维度进行了 2 路分片, 而 In 进行了 4 路分片 (沿收缩和输出维度). 这对应于分片 W[D<sub>X</sub>, F] 和 In[B<sub>X</sub>, D<sub>Y</sub>], 又名一种模型和数据并行.
2.  如果我们要在本地运行这个 (即, 在一个设备上), `matmul_square` 将简单地对输入进行平方并执行一个简单的矩阵乘法. 但是因为我们将 `out_shardings` 指定为 `P('X', None)`, 输出将沿批处理进行分片, 但在模型维度上进行复制, 并且需要一个 AllReduce 来计算.

使用我们前面章节的符号, 这可能会做类似的事情

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` 会自动为我们添加这个! 我们实际上可以用 `jit_matmul.as_text()` 打印 HLO, 并看到以下 HLO (大幅缩写):

```py
# 这个融合是分片输入和矩阵的实际矩阵乘法
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# 我们在设备上对部分求和的结果进行归约
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以看到上面的矩阵乘法 (融合) 和 AllReduce. 请特别注意形状. `bf16[2, 1024]` 是激活的本地视图, 因为我们的 `batch_size=8` 分布在 4 个设备上, 我们的 `d_model=2048` 同样分片 2 路.

**这太神奇了!** 无论我们的程序有多复杂, [Shardy]((https://openxla.org/shardy)) 和 jit 都会尝试为所有中间激活找到分片, 并根据需要添加通信. 话虽如此, Shardy 也有其缺陷. 它可能会犯错. 有时你会看一个配置文件, 注意到出了问题. 一个巨大的 AllGather 占用了 80% 的配置文件, 而它本不需要. 当这种情况发生时, 我们可以尝试通过用 `jax.lax.with_sharding_constraint` 显式地注释中间张量来纠正编译器. 例如, 对于两个矩阵乘法, 我可以强制中间激活沿 `y` 维度进行分片 (并不是说这是一个好主意), 如下所示:

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

这构成了 JAX 并行编程中自动分区世界的 60%, 你可以通过 `jax.lax.with_sharding_constraint` 控制中间分片. 但“编译器挠痒痒”是出了名的不好玩的编程模型. 你可以注释每个中间变量, 但仍然不知道是否会得到正确的结果. 相反, 如果 JAX 本身可以处理和控制分片传播呢?

<h3 id="explicit-sharding-mode">显式分片模式</h3>

显式分片 (或“类型中的分片”) 看起来很像自动分片, 但分片传播发生在 JAX 级别! 每个 JAX 操作都有一个分片规则, 该规则接受操作参数的分片并为操作的结果生成一个分片. 你可以使用 `jax.typeof` 查看结果分片:

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# 在 TPU v5e 2x2 上运行. 这为硬件的两个物理轴分配了名称.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# 这告诉 JAX 对所有操作都使用这个网格, 所以你只需要指定 PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # bfloat16[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # bfloat16[8@X,2@Y]
  return out

f(x)
```

正如你所看到的, JAX 将分片从输入 (`x`) 传播到输出 (`x`), 这些分片可以在跟踪时通过 `jax.typeof` 进行检查. 对于大多数操作, 这些规则简单明了, 因为只有一个合理的选择 (例如, 逐元素操作保留相同的分片). 但对于某些操作, 如何对结果进行分片是不明确的, 在这种情况下, JAX 会在跟踪时抛出一个错误, 我们要求程序员显式地提供一个 `out_sharding` 参数 (例如, jnp.einsum, jnp.reshape 等). 让我们看另一个有冲突的例子:

```py
# 我们创建一个矩阵 W 和输入激活 In, 它们在我们的设备上进行了分片.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 这会报错
```

此代码报错 `Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the `out_sharding` parameter. Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)`

这很棒, 因为 einsum 的输出应该如何分片是不明确的. 输出分片可以是:
*   P('X', 'Y') 这将引起一个 reduce-scatter 或
*   P('X', None) 这将引起一个 all-reduce

与自动模式不同, 显式模式在检测到不明确的通信时会报错, 并要求用户解决它. 所以在这里你可以这样做:

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

自动模式和显式模式可以通过 `jax.sharding.auto_axes` 和 `jax.sharding.explicit_axes` API 进行组合. 这是一篇[很好的文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)以获取更多信息.

<h3 id="manual-sharding-mode-via-shard_map">shard_map: 对程序的显式并行控制</h3>

虽然 Shardy 是“编译器接管”模式, 但 jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 将一切都交到你手中. 你指定输入的分片, 就像在 jax.jit 中一样, 但然后你显式地编写所有通信. `jax.jit` 给你一个程序的全局跨设备视图, 而 `shard_map` 给你一个本地的每个设备的视图.

这是一个例子. 试着推断这个函数的作用:<d-footnote>如果你想在 colab 中通过模拟一个网格来自己玩这个, 你可以用以下单元格来做 `import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=P(('x', 'y')))

# 这个函数将对数组的 1/8 进行操作.
@jax.shard_map(in_specs=P(('x', 'y')), out_specs=P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**这是做什么的?** `slice_and_average` 在每个 TPU 上运行, 使用数组的 1/8, 从中我们切片前 4 个元素, 并在整个网格上对它们进行平均. 这意味着我们实际上在做 `mean(x[:4], x[64:68], x[128:132], …)`. 这很酷, 因为这在 JAX 中不是一个容易表达的操作.

**为什么要这样做而不是用 jax.jit?** 如果我们使用 `jax.jit`, `slice_and_average` 将会看到一个数组的全局视图 (完整的 `[512,]` 数组). 我们将不得不切出这个非均匀的切片, 然后执行一个平均, XLA 必须正确地解释它. XLA 可能会添加错误的通信或感到困惑. 在这里, 我们看到本地视图, 只编写我们需要的通信.

**示例 [集合矩阵乘法]:** 举一个更现实的例子, 假设我们想实现模型并行, 其中激活最初是模型分片的, 即 A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]. 简单地说, 我们会先对 A 进行 AllGather, 然后进行本地矩阵乘法:

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

遗憾的是, 这很糟糕, 因为它不允许我们将通信与计算重叠. 可以通过“集合矩阵乘法”来重叠它们, 如 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) 所述. 算法基本上如下:

*   对于每个 Y 分片, 对 A 的本地块和 W 的本地块进行矩阵乘法, 产生一个形状为 `[B / X, F / Y]` 的结果. 同时, 对 A 进行置换, 以便你本地获得下一个块, 进行矩阵乘法, 并对结果求和.

我们可以用 shard\_map 很容易地实现它:

```py
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs 是循环操作数; rhs 是本地操作数
  axis_size = jax.lax.axis_size('Y')  # 在这个例子中 axis_size = 4
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # 一个块的矩阵乘法
    update = lhs @ rhs_chunk
    # 向左循环移位
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # 在最后一次置换后计算最后一个块, 以便将 lhs 保持在我们找到它时的状态
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

这很酷! 我们可以对它进行基准测试, 发现它也快得多! [这里](https://imgur.com/a/e9I6SrM) 是默认 jit 矩阵乘法的配置文件, 它需要 311us, 在开始时有一个大的阻塞 AllGather:

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

[这里](https://imgur.com/a/21iy0Sv) 是上面的版本, 需要 244 us. 你可以看到配置文件没有 AllGather. 全都是有用的工作! 我们的 FLOPs 利用率也高得多.

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

还值得注意的是, 在收缩维度上没有分片的矩阵乘法时间是 [224us](https://imgur.com/a/i3gNKfq), 所以我们非常接近未分片的基线. 这是你可能最终会做的性能工程的一个很好的例子, 以提高 TPU 利用率. 有关更多 `shard_map` 示例, [这个说明很棒](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side).

现在这里有几个有用的已解决的问题, 可以尝试用 `jax.jit` 或 `shard_map` 来实现!

## 已解决的问题

这里有一些随机的 JAX 相关问题. 我稍后会添加更多. 对于所有这些问题, 你都需要在 Colab 中有一定数量的 TPU. 你可以使用带有 TPUv2-8 的公共 Colab. 从现在开始, 我们将假设你有 N 个可用的设备.

**问题 1:** 设 **A** 是一个形状为 float32[S<sub>X</sub>, D<sub>Y</sub>] 的激活数组, 其中 `X * Y = N`. 执行以下操作:

1.  用 JAX 编写一个函数, 计算每个 `(X, Y)` 分片内的平均值, 即它返回一个大小为 [X, Y] 的数组, 其中 `arr[i, j]` 是分片 `(i, j)` 的平均值. 用 `jax.jit` 和 `shard_map` 都做一遍. 对每个进行分析, 看看它们花了多长时间. 是否添加了任何通信? *提示: 应该没有, 但有时 XLA 还是会添加.*

2.  用 JAX 编写一个函数, 返回 roll(x, shift, axis=0) - x, 对于**每个分片 X** 内的某个 shift. 我不是一个受虐狂, 不会让你用 jax.jit 来做这个, 所以只用 `shard_map` 来做.

{% details 点击这里查看答案. %}

第 1 部分: 这是第 1 部分的解决方案. 请注意, 对于 `jax.jit` 解决方案, 我们必须进行相当复杂的重塑.

```py
import numpy as np

import jax
import jax.numpy as jnp

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=P('X','Y'), out_specs=P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

第 2 部分: 这是第 2 部分的类似解决方案.

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=P('X','Y'), out_specs=P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**问题 2:** 在这里, 我们将一起制作一个基本的“专家混合”模型. 设 **W**: float32[E<sub>X</sub>, D, F<sub>Y</sub>] 是一组 E 个“专家”矩阵. 设 **A**: float32[S<sub>X</sub>, D<sub>Y</sub>] (我们的激活), 设 **B** 是一组“路由分配”, 其中 B[i] 是一个在 `[0, E)` 范围内的整数, 告诉我们我们想用哪个矩阵来处理那个激活. 我们想用 JAX 编写一个函数, 返回 `Out[i] = W[B[i]] @ A[i]`.

1.  让我们从完全忽略分片开始. 使所有这些张量足够小, 以便它们可以容纳在一个设备中. 编写这个函数的本地实现. *确保你没有物化一个形状为 `[S, D, F]` 的数组! 提示: 尝试将 token 排序到一个形状为 `[E, S, D]` 的新缓冲区中, 并注意掩码 (为什么我们需要第二个维度的大小为 S?).*

2.  如果你只是对上面的方法进行 `jax.jit`, 会发生一些事情. 对此进行分析, 看看它决定进行什么通信. 需要多长时间?

3.  你会注意到上面的一个问题是, 它可能会在本地收集完整的激活集 **A**, 即 AllGather<sub>X</sub>([S<sub>X</sub>, D<sub>Y</sub>]). 这不仅在通信方面成本高昂, 而且如果我们无法在本地容纳完整的激活集, 在内存方面也极其昂贵. 使用 `shard_map` 和显式通信来实现上述功能.

      1.  对于第一次尝试, 使用 `jax.lax.all_gather` 并像 (a) 中那样重新排序可能最简单.

      2.  对于第二次尝试, 尽量避免物化任何大小为 `[E, S, D]` 的数组, 即尝试使用 `jax.lax.while_loop` 内的 `jax.lax.all_to_all` 以不规则的方式执行计算. 这样, 你可以避免物化完整的激活并浪费计算在填充上. 这比你最初的实现快多少?

4.  大多数 MoE 路由到多个 (k) 专家, 然后对结果进行平均. 重构上面的代码以实现这一点. 在这种情况下, 设 **B**: int32[S, k] 用于路由到的 k 个专家.

**问题 3:** 上面实现的集合矩阵乘法示例实际上与真实的 LLM 非常相关. 让我们调整示例以执行完整的 Transformer 堆栈.

1.  作为练习, 让我们从实现一个 AllReduce 集合矩阵乘法开始, 即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]. 请注意, 输出没有被复制. 上面讨论了朴素的算法, 基本上只是一个本地矩阵乘法, 后面跟着一个 AllReduce. 尝试制作这个操作的通信重叠的“集合”版本. *提示: 在输出维度上进行分块, 并随意使用 `jax.lax.psum` (又名 AllReduce).* *注意: 由于 XLA 处理这个问题的方式, 它实际上可能不比基线快.*

2.  上面 AllReduce 集合矩阵乘法的补充是 ReduceScatter 集合矩阵乘法, 如 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]. 这发生在 Transformer 的下投影矩阵中. 在 JAX 中实现这个操作的集合, 重叠版本. 小心只传递你需要的最少量的数据. *提示: 在累积结果时尝试对结果进行置换.*

3.  将这两个组合成一个端到端的 Transformer 块, 该块执行 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>] 并带有重叠的通信.<d-footnote>和以前一样, 由于我们在这里省略了一个非线性, 我们不能先做 $W_{in} \cdot W_{out}$.</d-footnote> 这比 `jax.jit` 实现快多少?

**问题 4:** 上面实现的所有集合矩阵乘法都是单向的: 它们只在一个方向上进行置换. 重写集合 AllReduce 矩阵乘法和集合 ReduceScatter 矩阵乘法以使用双向通信. 这些快多少?

### 第 10 部分到此结束. 基本上就是这样了! 有关最终结论和进一步阅读, 请点击 [这里](../conclusion).