---
layout: distill
title: "在JAX中编程TPU"
# permalink: /main/
description: "如何使用JAX高效编程TPU！本节的大部分内容取自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>这里</a>。您可以在<a href='https://colab.sandbox.google.com/'>Google Colab</a>上使用免费的TPU运行本节中的代码示例。"
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
  - name: "JAX中的并行性如何工作？"
  - subsections:
    - name: "自动分片模式"
    - name: "显式分片模式"
    - name: "通过shard_map进行手动分片模式"
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

## JAX中的并行性如何工作？

JAX支持三种多设备编程的思想流派：

1. **编译器，你来掌舵！** 让XLA编译器自动对数组进行分区，并决定添加什么通信来促进给定程序的执行。这让你可以将一个在单个设备上运行的程序，无需任何更改就能在数千个设备上自动运行。
2. **JAX，你来掌舵！** 自动并行性很好，但有时编译器会做一些疯狂的事情。显式分片让你可以像往常一样编写单设备代码，但让JAX处理分片传播（而不是编译器）。这意味着当不清楚你想要什么时，JAX可以要求你澄清。
3. **就让我写我想表达的意思，该死的！** 虽然编译器很好，但有时它们会做错事情，并添加你不想要的通信。有时我们想要明确指定你打算运行的通信。

| 模式 | 视图？ | 显式分片？ | 显式集合通信？ |
|:---:|:---:|:---:|:---:|
| Auto | 全局 | ❌ | ❌ |
| Explicit | 全局 | ✅ | ❌ |
| Manual | 每设备 | ✅ | ✅ |

相应地，JAX为每种模式提供了API：

1. `jax.jit`（使用`Auto`网格轴）让你可以获取任何现有的JAX函数并使用分片输入调用它。然后JAX使用XLA的[Shardy](https://openxla.org/shardy)编译器自动并行化程序。当需要促进现有操作时，XLA会为你添加通信（AllGathers、ReduceScatters、AllReduces等）。虽然它不完美，但通常在无需代码更改的情况下，能很好地将你的程序自动扩展到任意数量的芯片上。
2. `jax.jit`与`Explicit`网格轴看起来类似于（1），但让JAX处理分片传播而不是XLA。这意味着数组的分片实际上是JAX类型系统的一部分，当JAX检测到模糊的通信时，它会报错并让用户解决它。
3. `jax.shard_map`是更手动的对应方案。你获得程序的设备本地视图，必须明确编写你想要的任何通信。有一个分片数组并希望在每个设备上拥有整个数组？添加一个`jax.lax.all_gather`。想要在你的设备上对数组求和？添加一个`jax.lax.psum`（AllReduce）。编程更困难，但不太可能做你不想做的事情。

<h3 id="auto-sharding-mode">自动分片模式</h3>

jax.jit在JAX内部扮演两个角色。顾名思义，它"及时"地将函数从Python编译成字节码（通过XLA/HLO/LLO），使其运行更快。但如果输入是分片的或用户指定了`in_sharding`或`out_sharding`，它还让XLA在多个设备之间分配计算并根据需要添加通信。例如，以下是如何使用jax.jit编写分片矩阵乘法：

```py
import jax
import jax.numpy as jnp

# 在TPU v5e 4x2上运行。这为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 这告诉JAX对所有操作使用此网格，因此你只需指定PartitionSpec P。
jax.set_mesh(mesh)

# 我们创建一个矩阵W和输入激活In，在设备间分片。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 我们可以在这里显式编译分片矩阵乘法函数。这添加了所有必要的通信（例如矩阵乘法后的AllReduce）。
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这将自动以任何分片方式运行，并在我们的设备间分配计算。**但在硬件层面实际发生了什么？**

1. 首先我们在设备间创建分片的In和W<d-footnote>注意我们是如何做到的。这是创建具有特定分片的数组的一种方法（即通过向创建函数添加device参数）。另一种方法是使用`jnp.array(....)`正常创建数组，然后执行例如`jax.device_put(..., P('x', 'y'))`。还有一种是编写创建你想要的数组的函数，并使用`out_shardings`为你想要的内容进行jit编译。</d-footnote>。W在收缩维度上分片2路，而In分片4路（在收缩和输出维度上）。这对应于分片W[D<sub>X</sub>, F]和In[B<sub>X</sub>, D<sub>Y</sub>]，也就是一种模型和数据并行性。
2. 如果我们在本地运行（即在一个设备上），`matmul_square`将简单地对输入进行平方并执行简单的矩阵乘法。但因为我们指定`out_shardings`为`P('X', None)`，输出将在批次维度上分片，但在模型维度上复制，并且需要AllReduce来计算。

使用我们前面几节的符号表示，这可能会做类似的事情：

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit`会自动为我们添加这个！我们可以用`jit_matmul.as_text()`实际打印HLO，并看到以下HLO（大幅缩写）：

```py
# 这个融合是分片输入和矩阵的实际矩阵乘法
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# 我们在设备间减少部分求和的结果
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以看到上面的矩阵乘法（融合）和AllReduce。特别注意形状。`bf16[2, 1024]`是激活的本地视图，因为我们的`batch_size=8`在4个设备间分割，我们的`d_model=2048`同样分割2路。

**这相当神奇！** 无论我们的程序多么复杂，[Shardy]((https://openxla.org/shardy))和jit都会尝试为所有中间激活找到分片方式，并根据需要添加通信。话虽如此，Shardy有其缺陷。它可能会犯错。有时你会查看性能分析文件并发现出了问题。一个巨大的AllGather占用了性能分析文件的80%，而实际上并不需要。当这种情况发生时，我们可以尝试通过使用`jax.lax.with_sharding_constraint`显式注释中间张量来纠正编译器。例如，对于两个矩阵乘法，我可以强制中间激活在`y`维度上分片（这不是一个好主意），如下所示：

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

这构成了JAX并行编程在自动分区世界中的约60%，在那里你通过`jax.lax.with_sharding_constraint`控制中间分片。但"编译器调优"显然不是一个有趣的编程模型。你可以注释每个中间变量，但仍然不知道是否会得到正确的结果。相反，如果JAX本身能够处理和控制分片传播呢？

<h3 id="explicit-sharding-mode">显式分片模式</h3>

显式分片（或"类型中的分片"）看起来很像自动分片，但分片传播发生在JAX层面！每个JAX操作都有一个分片规则，它接受操作参数的分片并为操作结果生成分片。你可以使用`jax.typeof`查看结果分片：

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# 在TPU v5e 2x2上运行。这为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# 这告诉JAX对所有操作使用此网格，因此你只需指定PartitionSpec P。
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

正如你所看到的，JAX将分片从输入（`x`）传播到输出（`x`），这些可以通过`jax.typeof`在追踪时检查。对于大多数操作，这些规则简单明了，因为只有一个合理的选择（例如，元素级操作保留相同的分片）。但对于某些操作，如何对结果进行分片是模糊的，在这种情况下JAX会抛出追踪时错误，我们要求程序员明确提供`out_sharding`参数（例如jnp.einsum、jnp.reshape等）。让我们看另一个有冲突的例子：

```py
# 我们创建一个矩阵W和输入激活In，在设备间分片。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 这将报错
```

这段代码会报错，提示`Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the `out_sharding` parameter. Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)`

这很棒，因为einsum的输出应该如何分片是模糊的。输出分片可以是：
* P('X', 'Y')这将引发reduce-scatter或
* P('X', None)这将引发all-reduce

与Auto模式不同，显式模式在检测到模糊的通信时会报错，并要求用户解决它。所以在这里你可以这样做：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto模式和Explicit模式可以通过`jax.sharding.auto_axes`和`jax.sharding.explicit_axes` API组合。这是一篇[值得阅读的文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)，以获取更多信息。

<h3 id="manual-sharding-mode-via-shard_map">shard_map: explicit parallelism control over a program</h3>

While Shardy is the "compiler take the wheel" mode, jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) puts everything in your hands. You specify the sharding of the inputs, like in jax.jit, but then you write all communication explicitly. Whereas `jax.jit` leaves you with a global cross-device view of the program, `shard_map` gives you a local per-device view.

Here's an example. Try to reason about what this function does:<d-footnote>If you want to play with this yourself in a colab by emulating a mesh, you can do so using the following cell `import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=P(('x', 'y')), out_specs=P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**What does this do?** `slice_and_average` is run on each TPU with 1/8th of the array, from which we slice the first 4 elements and average them across the full mesh. This means we're effectively doing `mean(x[:4], x[64:68], x[128:132], …)`. This is pretty cool, because that's not an easy operation to express in JAX otherwise.

**Why do this instead of jax.jit?** If we'd used `jax.jit`, `slice_and_average` would have seen a global view of the array (the full `[512,]` array). We'd have had to slice out this non-uniform slice and then perform an average which XLA would have had to interpret correctly. XLA might have added the wrong communication or gotten confused. Here we see the local view and write only the communication we need.

**Example [Collective Matmul]:** To take a more realistic example, say we to implement model parallelism where the activations are initially model sharded, i.e. A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]. Naively, we would do this by AllGathering A first followed by a local matrix multiplication:

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

Sadly, this is bad because it doesn't allow us to overlap the communication with the computation. Overlapping them can be done with a "collective matmul", as described in [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959). The algorithm is basically as follows:

* For each Y shard, perform a matmul of the local chunk of A with the local chunk of W, producing a result of shape `[B / X, F / Y]`. Simultaneously, permute A so you get the next chunk locally, perform the matmul, and sum the result.

We can implement that quite easily with shard\_map:

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
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
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

This is pretty neat! We can benchmark this and see that it's also a lot faster! [Here's](https://imgur.com/a/e9I6SrM) the profile with the default jit matmul which takes 311us with a big blocking AllGather at the beginning:

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

And [here's](https://imgur.com/a/21iy0Sv) the version above that takes 244 us. You can see the profile doesn't have the AllGather. It's all useful work! Our FLOPs utilization is also a lot higher.

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

It's also worth noting that the matmul time with no sharding on the contracting dimension is [224us](https://imgur.com/a/i3gNKfq), so we're remarkably close to the unsharded baseline here. This is a good example of the kind of performance engineering you might end up doing to improve TPU utilization. For more `shard_map` examples, [this note is great](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side).

Now here are a couple of useful worked problems to try and implement using `jax.jit` or `shard_map`!

## Worked Problems

Here are some random JAX-related problems. I'll add some more later. For all of these, you'll need some number of TPUs in a Colab. You can use a public Colab with TPUv2-8. From now on, we'll assume you have N devices available.

**Problem 1:** Let **A** be an array of activations of shape float32[S<sub>X</sub>, D<sub>Y</sub>] with `X * Y = N`. Do the following:

1. Write a function in JAX that computes the average within each `(X, Y)` shard, i.e. it returns an array of size [X, Y] where `arr[i, j]` is the average over shard `(i, j)`. Do this with both `jax.jit` and `shard_map`. Profile each and see how long they took. Was there any communication added? *Hint: there shouldn't be, but sometimes XLA adds it anyway.*

2. Write a function in JAX that returns roll(x, shift, axis=0) - x for some shift **within each shard X**. I'm not enough of a masochist to make you do this in jax.jit, so just do this with `shard_map`.

{% details Click here for the answer. %}

Part 1: Here is a solution to part 1. Note the fairly complex reshapes we have to do for the `jax.jit` solution.

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

Part 2: Here is a similar solution to Part 2.

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

**Problem 2:** Here we'll make a basic "mixture of experts" model together. Let **W**: float32[E<sub>X</sub>, D, F<sub>Y</sub>] be a set of E "expert" matrices. Let **A**: float32[S<sub>X</sub>, D<sub>Y</sub>] (our activations) and let **B** be a set of "routing assignments" where B[i] is an integer in the range `[0, E)` telling us which matrix we want to process that activation. We want to write a function in JAX that returns `Out[i] = W[B[i]] @ A[i]`.

1. Let's start by ignoring sharding altogether. Make all of these tensors small enough so they fit in one device. Write a local implementation of this function. *Make sure you don't materialize an array of shape `[S, D, F]`! Hint: try sorting the tokens into a new buffer of shape `[E, S, D]` with some attention to masking (why do we need the second dimension to have size S?).*

2. If you just `jax.jit` the above method, something will happen. Profile this and see what communication it decided to do. How long does it take?

3. One problem you'll notice with the above is that it likely gathers the full set of activations **A** locally, i.e. AllGather<sub>X</sub>([S<sub>X</sub>, D<sub>Y</sub>]), Not only is this expensive communication-wise, it's also incredibly expensive memory-wise if we can't fit the full set of activations locally. Implement the above using `shard_map` and explicit communication.

      1. For a first pass, it might be easiest to use a `jax.lax.all_gather` and reorder as in (a).

      2. For a second pass, try to avoid materializing any array of size `[E, S, D]`, i.e. try to perform the computation in a ragged fashion using a `jax.lax.all_to_all` inside a `jax.lax.while_loop`. This way, you can avoid materializing the full activations and wasting compute on padding. How much faster is this than your original implementation?

4. Most MoEs route to multiple (k) experts and then average the result. Refactor the above to implement this. Let **B**: int32[S, k] in this case for the k experts to route to.

**Problem 3:** The collective matmul example above is actually super relevant for real LLMs. Let's tweak the example to do the full Transformer stack.

1. As an exercise, let's start by implementing an AllReduce collective matmul, i.e. A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]. Note that the output isn't replicated. The naive algorithm is discussed above, basically just a local matmul followed by an AllReduce. Try to make a comms overlapped "collective" version of this operation. *Hint: tile over the output dimension and feel free to use `jax.lax.psum` (aka AllReduce).* *Note: due to the way XLA handles this, it may not actually be faster than the baseline.*

2. The complement to the AllReduce collective matmul above is a ReduceScatter collective matmul, as in Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]. This occurs in the down-projection matrix in a Transformer. Implement a collective, overlapped version of this in JAX. Be careful about passing only the minimal amount of data you need. *Hint: try permuting the result as you accumulate it.*

3. Put these two together into an end-to-end Transformer block that performs In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>] with overlapped communication.<d-footnote>As before, we can't do $W_{in} \cdot W_{out}$ first because of a non-linearity we've omitted here.</d-footnote> How much faster is this than a `jax.jit` implementation?

**Problem 4:** All of the collective matmuls implemented above are unidirectional: they only permute in one direction. Rewrite the collective AllReduce matmul and the collective ReduceScatter matmuls to use bidirectional communication. How much faster are these?

### That's all for Part 10. That's basically it! For final conclusions and further reading, click [here](../conclusion).
