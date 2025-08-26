---
layout: distill
title: "如何理解 GPU"
description: "我们在 Google 热爱 TPU, 但 GPU 也很棒. 本章将深入探讨 NVIDIA GPU 的世界——每个芯片如何工作, 它们如何联网, 以及这对 LLM 意味着什么, 特别是与 TPU 相比. 本节建立在 <a href='https://lqhl.github.io/scaling-book/tpus/'>第 2 章</a> 和 <a href='https://lqhl.github.io/scaling-book/training'>第 5 章</a> 的基础上, 建议您先阅读它们."
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "Part 11: Conclusion"

next_section_url:
next_section_name: "The End"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Swapnil Patil<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name:  Adam Paszke<sup>†</sup>
    url: https://x.com/apaszke
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "什么是 GPU?"
  - subsections:
    - name: "内存"
    - name: "GPU 规格摘要"
    - name: "芯片层面的 GPU 与 TPU 对比"
    - name: "测验 1: GPU 硬件"
  - name: "网络"
  - subsections:
    - name: "节点层面"
    - name: "测验 2: GPU 节点"
    - name: "节点之外"
    - name: "测验 3: 节点之外"
  - name: "集合操作在 GPU 上如何工作?"
  - subsections:
    - name: "节点内集合操作"
    - name: "跨节点集合操作"
    - name: "测验 4: 集合操作"
  - name: "GPU 上 LLM 扩展的 Roofline 模型"
  - subsections:
    - name: "数据并行"
    - name: "张量并行"
    - name: "专家并行"
    - name: "流水线并行"
    - name: "示例"
    - name: "GPU 上 LLM 扩展的 TLDR"
    - name: "测验 5: LLM Rooflines"
  - name: "致谢和延伸阅读"
  - name: "附录"
  - subsections:
    - name: "附录 A: GB200 会带来哪些变化?"
    - name: "附录 B: 更多网络细节"

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

## 什么是 GPU?

现代机器学习 GPU (例如 H100, B200) 基本上是一堆专门用于矩阵乘法的计算核心 (称为**流式多处理器**或 **SMs**), 连接到一块高速内存 (称为 **HBM**). 下图是一个示意图:

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" caption="<b>图:</b> 展示 H100 或 B200 GPU 抽象布局的示意图. H100 有 132 个 SM, 而 B200 有 148 个. 我们宽泛地使用术语 'Warp 调度器' 来描述一组 32 个 CUDA SIMD 核心<i>以及</i>向它们分派工作的调度器. 注意这与 TPU 的相似之处!" %}

每个 SM, 就像 TPU 的张量核心 (Tensor Core) 一样, 拥有一个专用的矩阵乘法核心 (不幸的是也叫**张量核心**<d-footnote>GPU 的张量核心是 SM 的矩阵乘法子单元, 而 TPU 的 TensorCore 是包含 MXU, VPU 和其他组件的总称单元.</d-footnote>), 一个向量算术单元 (称为 **Warp 调度器**<d-footnote>NVIDIA 对此没有一个好的命名, 所以我们只是在几个不好的选项中选择了最好的一个. Warp 调度器主要是一个向一组 CUDA 核心分派工作的单元, 但我们在这里用它来描述控制单元及其控制的核心集合.</d-footnote>), 以及一个高速片上缓存 (称为 **SMEM**). 与 TPU 最多拥有 2 个独立的“张量核心”不同, 现代 GPU 拥有超过 100 个 SM (H100 上有 132 个). 每个 SM 的功能远不如 TPU 张量核心强大, 但整个系统更加灵活. 每个 SM 或多或少是完全独立的, 所以一个 GPU 可以同时执行数百个独立的任务.<d-footnote>虽然 SM 是独立的, 但为了达到峰值性能, 它们通常被迫进行协调, 因为它们都共享一个容量有限的 L2 缓存.</d-footnote>

让我们更详细地看一下 H100 SM:

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" caption="<b>图:</b> H100 SM 的示意图 (<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>来源</a>), 展示了 4 个<i>子分区</i>, 每个子分区包含一个张量核心, Warp 调度器, 寄存器文件, 以及不同精度的 CUDA 核心集. 底部附近的 'L1 数据缓存' 是 256kB 的 SMEM 单元. B200 看起来类似, 但增加了大量的张量内存 (TMEM) 来为庞大的张量核心提供数据." %}

每个 SM 被分成 4 个相同的象限, NVIDIA 称之为 **SM 子分区**, 每个子分区包含一个张量核心, 16k 个 32 位寄存器, 以及一个称为 Warp 调度器的 SIMD/SIMT 向量算术单元, 其通道 (ALU) NVIDIA 称之为 **CUDA 核心**. 每个分区 Arguably 的核心组件是执行矩阵乘法的张量核心, 它构成了其绝大部分的 FLOPs/s, 但它并非唯一值得注意的组件.

*   **CUDA 核心:** 每个子分区包含一组称为 CUDA 核心的 ALU, 用于执行 SIMD/SIMT 向量运算. 每个 ALU 通常每个周期可以执行 1 次算术运算, 例如 f32.add.<d-footnote>较新的 GPU 支持 FMA (融合乘加) 指令, 严格来说每个周期执行两次 FLOP, NVIDIA 无情地利用这一事实将其报告的规格翻倍.</d-footnote> 每个子分区包含 32 个 fp32 核心 (以及数量较少的 int32 和 fp64 核心), 它们在每个周期都执行相同的指令. 与 TPU 的 VPU 类似, CUDA 核心负责 ReLU, 逐点向量运算和归约 (求和).<d-footnote>历史上, 在引入张量核心之前, CUDA 核心是 GPU 的主要组件, 用于渲染, 包括光线-三角形相交和着色. 在今天的游戏 GPU 上, 它们仍然承担着大量的渲染工作, 而张量核心则用于上采样 (DLSS), 这使得 GPU 能够以较低的分辨率进行渲染 (像素越少 = 工作量越少), 并使用机器学习进行上采样.</d-footnote>

*   **张量核心 (TC):** 每个子分区都有自己的张量核心, 这是一个专用的矩阵乘法单元, 类似于 TPU MXU. 张量核心占了 GPU FLOPs/s 的绝大部分 (例如, 在 H100 上, 我们有 990 bf16 TC TFLOP/s, 而 CUDA 核心只有 66 TFLOPs/s).
    *   [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) 意味着 132 个 SM 以 1.76GHz 运行时, 每个 H100 TC 可以执行 `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs/cycle, 大约是一个 8x8x8 的矩阵乘法.<d-footnote>NVIDIA 没有分享太多 TC 硬件细节, 所以这更多是猜测而非确切事实——当然, 这并没有说明 TC 是如何实现的. 我们知道 V100 每 TC/周期可以执行 256 FLOPs. A100 可以执行 512, H100 可以执行 1024, 虽然 B200 的细节尚未公布, 但似乎可能是 2048 FLOPs/TC/周期, 因为 `2250e12 / (148 * 4 * 1.86e9)` 大约是 2048. 更多细节在<a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>这里</a>得到确认.</d-footnote>
    *   与 TPU 类似, GPU 可以用更低的精度执行矩阵乘法以获得更高的吞吐量 (例如 H100 的 fp8 FLOPs/s 是 fp16 的 2 倍). 低精度训练或服务可以显著加快速度.
    *   自 Volta 以来的每一代 GPU 都增加了 TC 的尺寸 ([关于此内容的好文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)). 到了 B200, TC 变得如此之大, 以至于其输入不再能装入 SMEM, 因此 B200 引入了一个名为 TMEM 的新内存空间.<d-footnote>在 Ampere 中, 张量核心可以由单个 warp 供给, 而在 Hopper 中则需要一个完整的 SM (warpgroup), 在 Blackwell 中则由 2 个 SM 供给. 在 Blackwell 中, 矩阵乘法也变得如此之大, 以至于参数 (特别是累加器) 不再适合寄存器内存/SMEM, 因此 Blackwell 增加了 TMEM 来解决这个问题.</d-footnote>

**CUDA 核心比 TPU 的 VPU 更灵活:** GPU CUDA 核心 (自 V100 起) 使用所谓的 SIMT (*单指令多线程*) 编程模型, 而 TPU 使用 SIMD (*单指令多数据*) 模型. 与 TPU VPU 中的 ALU 类似, 一个子分区内的 CUDA 核心必须在每个周期执行相同的操作 (例如, 如果一个核心正在对两个浮点数进行加法, 那么该子分区中的所有其他 CUDA 核心也必须这样做). 然而, 与 VPU 不同的是, 每个 CUDA 核心 (或 CUDA 编程模型中的“线程”) 都有自己的指令指针, 并且可以被独立*编程*. 当同一 warp 中的两个线程被指示执行不同的操作时, 你实际上会执行*两种*操作, 屏蔽掉那些不需要执行发散操作的核心.

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>图:</b> 一组线程内 warp 发散的示例 (<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>来源</a>). 白色空间表示至少一部分物理 CUDA 核心的停顿." %}

这使得在线程级别进行灵活编程成为可能, 但代价是如果 warp 发散过于频繁, 性能会悄无声息地下降. 线程在访问内存方面也更灵活; VPU 只能操作连续的内存块, 而 CUDA 核心可以访问共享寄存器中的单个浮点数并维护每个线程的状态.

**CUDA 核心调度也更灵活:** SM 的运行方式有点像多线程 CPU, 因为它们可以并发地“调度”许多程序 (**warps**) (每个 SM 最多 64 个), 但每个 *Warp 调度器* 在每个时钟周期只执行一个程序.<d-footnote>在给定 SM 上调度的 Warp 称为“常驻”.</d-footnote> Warp 调度器会自动在活动 warp 之间切换, 以隐藏 I/O 操作, 如内存加载. 相比之下, TPU 通常是单线程的.

### 内存

除了计算单元, GPU 还有一个内存层次结构, 最大的是 HBM (主 GPU 内存), 然后是一系列较小的缓存 (L2, L1/SMEM, TMEM, 寄存器内存).

*   **寄存器:** 每个子分区都有自己的寄存器文件, 在 H100/B200 上包含 16,384 个 32 位字 (`4 * 16384 * 4 = 256kiB` 每个 SM), 可由 CUDA 核心访问.
    *   每个 CUDA 核心一次最多只能访问 256 个寄存器, 所以虽然我们每个 SM 最多可以调度 64 个“常驻 warp”, 但如果每个线程使用 256 个寄存器, 你一次只能容纳 8 个 (`256 * 1024 / (4 * 32 * 256)`).

*   **SMEM (L1 缓存):** 每个 SM 都有自己的 256kB 片上缓存, 称为 SMEM, 它可以由程序员控制为“共享内存”, 也可以由硬件用作片上缓存. SMEM 用于存储激活值和 TC 矩阵乘法的输入.

*   **L2 缓存:** 所有 SM 共享<d-footnote>严格来说, L2 缓存被分成两部分, 所以一半的 SM 可以各自访问 25MB. 有一个连接两半的链接, 但带宽较低.</d-footnote> 一个相对较大的约 50MB 的 L2 缓存, 用于减少主内存访问.
    *   这在大小上与 TPU 的 VMEM 相似, 但它**慢得多**并且不受程序员控制. 这导致了一些“远距离的诡异行为”, 程序员需要修改内存访问模式以确保 L2 缓存得到良好利用.<d-footnote>L2 缓存由所有 SM 共享这一事实, 实际上迫使程序员以一种相当协调的方式运行 SM, 尽管原则上它们是独立的单元.</d-footnote>
    *   NVIDIA 没有公布其芯片的 L2 带宽, 但据[测量](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)约为 5.5TB/s. 这大约是 HBM 带宽的 1.6 倍, 但它是全双工的, 所以有效的双向带宽接近 3 倍. 相比之下, TPU 的 VMEM 大小是其 2 倍, *并且* 带宽要大得多 (约 40TB/s).

*   **HBM:** 主 GPU 内存, 用于存储模型权重, 梯度, 激活值等.
    *   HBM 的大小从 Volta 的 32GB 增加到 Blackwell (B200) 的 192GB.
    *   从 HBM 到 CUDA 张量核心的带宽称为 HBM 带宽或内存带宽, 在 H100 上约为 3.35TB/s, 在 B200 上约为 9TB/s.

### GPU 规格摘要

以下是近期型号的 GPU 规格摘要. SM 数量, 时钟速度和 FLOPs 在给定 GPU 的不同变体之间略有不同. 以下是内存容量数据:

| GPU | 世代 | 时钟速度 | SMs/芯片 | SMEM 容量/SM | L2 容量/芯片 | HBM 容量/芯片 |
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: |
| V100 | Volta | 1.25GHz/1.38HGz | 80 | 96kB | 6MB | 32GB |
| A100 | Ampere | 1.10GHz/1.41GHz | 108 | 192kB | 40MB | 80GB |
| H100 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 80GB |
| H200 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 141GB |
| B200 | Blackwell | ? | 148 | 256kB | 126MB | 192GB |

所有世代每个 SM 都有 256kB 的寄存器内存. Blackwell 每个 SM 还增加了 256kB 的 TMEM. 以下是每个芯片的 FLOPs 和带宽数据:

| GPU | 世代 | HBM BW/芯片 | FLOPs/s/芯片 (bf16/fp16) | FLOPs/s/芯片 (fp8/int8) | FLOPs/s/芯片 (fp4) |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100 | Volta | 9.0e11 | — | — | — |
| A100 | Ampere | 2.0e12 | 3.1e14 | 6.2e14 | — |
| H100 | Hopper | 3.4e12 | 9.9e14 | 2.0e15 | — |
| H200 | Hopper | 4.8e12 | 9.9e14 | 2.0e15 | — |
| B200 | Blackwell | 8.0e12 | 2.3e15 | 4.5e15 | 9.0e15 |

我们排除了 B100, 因为它没有大规模生产.<d-footnote>虽然 NVIDIA 制造了 B100 这一代, 但它们只是短暂销售和生产, 据称是由于设计缺陷, 导致它们无法接近其声称的规格运行. 它们在不因热量和功率问题而降频的情况下难以达到峰值 FLOPs.</d-footnote> 一些规格略微取决于 GPU 的精确版本, 因为 NVIDIA GPU 不像 TPU 那样标准化.

这里有一个有用的备忘单, 比较了 GPU 和 TPU 的组件:

| GPU | TPU | 它是什么? |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器 (SM) | 张量核心 | 包含其他单元的核心“单元” |
| Warp 调度器 | VPU | SIMD 向量算术单元 |
| CUDA 核心 | VPU ALU | SIMD ALU |
| SMEM (L1 缓存) | VMEM | 快速片上缓存 |
| 张量核心 | MXU | 矩阵乘法单元 |
| HBM (又名 GMEM) | HBM | 高带宽大容量内存 |

### 芯片层面的 GPU 与 TPU 对比

GPU 最初用于渲染视频游戏, 但自 2010 年代深度学习兴起以来, 它们越来越像专用的矩阵乘法机器——换句话说, 越来越像 TPU.<d-footnote>在深度学习热潮之前, GPU (“图形处理单元”) 做的是, 嗯, 图形处理——主要用于视频游戏. 视频游戏用数百万个小三角形来表示物体, 游戏每秒 30-60 次将这些三角形渲染 (或“光栅化”) 成显示在屏幕上的 2D 图像 (这个频率称为帧率). 光栅化涉及将这些三角形投影到相机的坐标系中, 并计算哪些三角形与哪些像素重叠, 每秒数十亿次. 你可以想象, 这非常昂贵, 而且这仅仅是开始. 然后你必须通过组合可能与光线相交的几个半透明三角形的颜色来为每个像素着色. GPU 被设计用来极快地执行这些操作, 并着眼于通用性; 你需要同时运行许多不同的 GPU 工作负载 (称为“着色器”), 没有单一操作占主导地位. 因此, 消费级以图形为中心的 GPU 可以进行矩阵乘法, 但这不是它们的主要功能.</d-footnote> 在某种程度上, 这段历史解释了为什么现代 GPU 是现在这个样子. 它们并非纯粹为 LLM 或 ML 模型设计, 而是作为通用加速器, 硬件追求的“通用性”水平既是福也是祸. GPU 在应用于新任务时更常“开箱即用”, 并且对优秀编译器的依赖远低于 TPU. 但这也使得它们更难推理或从中获得 roofline 性能, 因为太多的编译器特性可能导致瓶颈.

**GPU 更具模块化.** TPU 有 1-2 个大的张量核心, 而 GPU 有数百个小的 SM. 同样, 每个张量核心有 4 个大的 VPU, 每个 VPU 有 1024 个 ALU, 而 GPU H100 有 132 * 4 = 528 个小的独立 SIMD 单元. 以下是 GPU 与 TPU 的 1:1 比较, 突出了这一点:

| GPU | TPU | H100 # | TPU v5p # |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM (流式多处理器) | 张量核心 | 132 | 2 |
| Warp 调度器 | VPU | 528 | 8 |
| SMEM (L1 缓存) | VMEM | 32MB | 128MB |
| 寄存器 | 向量寄存器 (VRegs) | 32MB | 256kB |
| 张量核心 | MXU | 528 | 8 |

这种模块化上的差异一方面使得 TPU 的制造成本更低, 理解起来更简单, 但也给编译器带来了更大的负担, 需要它做出正确的决策. 因为 TPU 只有一个控制线程, 并且只支持向量化的 VPU 范围指令, 编译器需要手动流水线化所有内存加载和 MXU/VPU 工作以避免停顿. GPU 程序员可以启动数十个不同的内核, 每个内核都在完全独立的 SM 上运行. 另一方面, 这些内核可能会因为 L2 缓存颠簸或未能合并内存加载而性能糟糕; 因为硬件控制了如此多的运行时, 很难推理幕后发生了什么. 因此, TPU 通常可以用更少的工作量更接近峰值 roofline 性能.

**历史上, 单个 GPU 比同类 TPU 更强大 (也更昂贵):** 单个 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍, HBM 是其 1.5 倍. 同时, Google Cloud 上的标价约为 H200 每小时 10 美元, 而 TPU v5p 每小时 4 美元. TPU 通常比 GPU 更依赖于将多个芯片联网在一起.

**TPU 拥有更多的高速缓存.** TPU 的 VMEM 也比 GPU 的 SMEM (+TMEM) 多得多, 这种内存可以用来存储权重和激活值, 使得它们可以被极快地加载和使用. 如果你能持续地将模型权重存储或预取到 VMEM 中, 这可以使它们在 LLM 推理中更快.

### 测验 1: GPU 硬件

这里有一些问题可以测试上述内容. 答案已提供, 但最好在看答案之前, 手持纸笔尝试回答问题.

**问题 1 [CUDA 核心]:** 一个 H100 有多少个 fp32 CUDA 核心 (ALU)? B200 呢? 这与 TPU v5p 中的独立 ALU 数量相比如何?

{% details 点击这里查看答案. %}

**答案:** 一个 H100 有 132 个 SM, 每个 SM 有 4 个子分区, 每个子分区包含 32 个 fp32 CUDA 核心, 所以我们有 `132 * 4 * 32 = 16896` 个 CUDA 核心. 一个 B200 有 `148` 个 SM, 所以总共有 `18944` 个. 一个 TPU v5p 有 2 个 TensorCore (通常通过 Megacore 连接), 每个都有一个 VPU, VPU 有 (8, 128) 个通道, 每个通道有 4 个独立的 ALU, 所以 `2 * 4 * 8 * 128 = 8192` 个 ALU. 这大约是 H100 向量通道数量的一半, 运行频率大致相同.

{% enddetails %}

**问题 2 [向量 FLOPs 计算]**: 单个 H100 有 132 个 SM, 时钟速度为 1.59GHz (最高可达 1.98GHz boost). 假设它每个 ALU 每个周期可以执行一次向量操作. 每秒可以执行多少次向量 fp32 FLOPs? boost 模式下呢? 这与矩阵乘法 FLOPs 相比如何?

{% details 点击这里查看答案. %}

**答案:** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`. boost 模式下是 33.5 TFLOPs/s. 这是[规格表](https://www.nvidia.com/en-us/data-center/h100/)中报告的一半, 因为严格来说我们可以在一个周期内完成一次 FMA (融合乘加), 这算作两次 FLOPs, 但在大多数情况下这并不实用. 我们可以执行 990 bfloat16 矩阵乘法 TFLOPs/s, 所以忽略 FMA, 张量核心的 FLOPs/s 大约是其 30 倍.

{% enddetails %}

**问题 3 [GPU 矩阵乘法强度]:** H100 的峰值 fp16 矩阵乘法强度是多少? B200 呢? fp8 呢? *强度我们指的是矩阵乘法 FLOPs/s 与内存带宽的比率.*

{% details 点击这里查看答案. %}

**答案:** 对于 H100, 我们有峰值 990e12 fp16 FLOPs 和 3.35e12 字节/秒的带宽. 所以临界强度是 `990e12 / 3.35e12 = 295`, 与 TPU 的 240 相当接近. 对于 B200, 它是 `2250e12 / 8e12 = 281`, 非常相似. 这意味着, 与 TPU 类似, 我们需要大约 280 的批处理大小才能在矩阵乘法中达到计算密集型.

对于 H100 和 B200, 我们都有正好 2 倍的 fp8 FLOPs, 所以峰值强度也翻倍到 590 和 562, 尽管在某种意义上它保持不变, 如果我们考虑到我们的权重也可能以 fp8 加载.

{% enddetails %}

**问题 4 [矩阵乘法运行时间]:** 使用问题 3 的答案, 你预计一个 `fp16[64, 4096] * fp16[4096, 8192]` 的矩阵乘法在单个 B200 上需要多长时间? `fp16[512, 4096] * fp16[4096, 8192]` 呢?

{% details 点击这里查看答案. %}

从上面我们知道, 当批处理大小低于 281 个 token 时, 我们将受通信限制. 因此, 第一个纯粹是带宽限制. 我们读取或写入 $2BD + 2DF + 2BF$ 字节 (`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`), 带宽为 `8e12` 字节/秒, 所以大约需要 `69e6 / 8e12 = 8.6us`. 实际上, 我们可能只能获得总带宽的一部分, 所以可能需要接近 10-12us. 当我们增加批处理大小时, 我们完全是计算密集型, 所以我们预计 `T=2*512*4096*8192/2.3e15=15us`. 我们同样只期望总 FLOPs 的一部分, 所以我们可能会看到接近 20us.

{% enddetails %}

**问题 5 [L1 缓存容量]:** H100 的总 L1/SMEM 容量是多少? 寄存器内存呢? 这与 TPU VMEM 容量相比如何?

{% details 点击这里查看答案. %}

**答案:** 我们每个 SM 有 256kB 的 SMEM 和 256kB 的寄存器内存, 所以每种大约 33MB (`132 * 256kB`). 总共, 这给了我们大约 66MB. 这大约是现代 TPU 120MB VMEM 的一半, 尽管一个 TPU 总共只有 256kB 的寄存器内存! TPU VMEM 的延迟低于 SMEM 的延迟, 这也是为什么 TPU 上的寄存器内存不那么关键的原因之一 (溢出和填充到 VMEM 的成本很低).

{% enddetails %}

**问题 6 [计算 B200 时钟频率]:** NVIDIA [在此](https://resources.nvidia.com/en-us-blackwell-architecture) 报告称, B200 可以执行 80TFLOPs/s 的向量 fp32 计算. 鉴于每个 CUDA 核心可以在一个 FMA (融合乘加) 操作中执行 2 FLOPs/cycle, 估算峰值时钟周期.

{% details 点击这里查看答案. %}

**答案:** 我们知道我们有 148 * 4 * 32 = 18944 个 CUDA 核心, 所以我们可以执行 `18944 * 2 = 37888 FLOPs / cycle`. 因此 `80e12 / 37888 = 2.1GHz`, 这是一个很高但合理的峰值时钟速度. B200 通常是液冷的, 所以更高的时钟周期更合理.

{% enddetails %}

**问题 7 [估算 H100 加法运行时间]:** 使用上面的数字, 计算在单个 H100 上将两个 `fp32[N]` 向量相加应该需要多长时间. 计算 $T_\text{math}$ 和 $T_\text{comms}$. 这个操作的算术强度是多少? 如果你可以访问, 也尝试在 PyTorch 或 JAX 中运行这个操作, `N = 1024` 和 `N=1024 * 1024 * 1024`. 结果如何?

{% details 点击这里查看答案. %}

**答案:** 首先, 将两个 `fp32[N]` 向量相加执行 N 次 FLOPs, 需要加载 `4 * N * 2` 字节并写回 4 * N 字节, 总共 `3 * 4 * N = 12N`. 计算它们的比率, 我们得到 `总 FLOPs / 总字节数 = N / 12N = 1 / 12`, 这相当糟糕.

正如我们上面计算的, 我们可以达到大约 33.5 TFLOPs/s 的 boost, 忽略 FMA. 这只有在所有 CUDA 核心都被使用的情况下才能实现. 对于 `N = 1024`, 我们最多只能使用 1024 个 CUDA 核心或 8 个 SM, 这将花费更长的时间 (假设我们是计算密集型, 大约长 16 倍). 我们的内存带宽为 3.35e12 字节/秒. 因此, 我们的峰值硬件强度是 `33.5e12 / 3.35e12 = 10`.<d-footnote>值得注意的是, 这种强度在最近几代 GPU 中保持不变. 对于 H100, 它是 33.5 / 3.5, 对于 B200, 它是 80 / 8. 原因尚不清楚, 但这是一个有趣的观察.</d-footnote> 所以我们将严重受通信限制. 因此我们的运行时间就是

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

对于 `N = 65,536`, 这大约是 0.23us. 实际上, 我们在 JAX 中看到的运行时间约为 1.5us, 这没关系, 因为我们预计在这里会受到超延迟的限制. 对于 `N = 1024 * 1024 * 1024`, 我们的 roofline 约为 3.84ms, 我们看到的是 4.1ms, 这很好!

{% enddetails %}

## 网络

网络是 GPU 和 TPU 差异最大的领域之一. 正如我们所见, TPU 以 2D 或 3D 环面连接, 每个 TPU 只连接到其邻居. 这意味着在两个 TPU 之间发送消息必须经过所有中间的 TPU, 并迫使我们只能在网格上使用统一的通信模式. 虽然在某些方面不方便, 但这也意味着每个 TPU 的链接数量是恒定的, 我们可以扩展到任意大的 TPU "pod" 而不损失带宽.

另一方面, GPU 使用更传统的分层树状交换网络. 称为**节点**的 8 个 GPU 集合 (GB200 最多 72 个<d-footnote>术语“节点”是重载的, 可以指两件事: NVLink 域, 即通过 NVLink 互连完全连接的 GPU 集合, 或连接到单个 CPU 主机的 GPU 集合. 在 B200 之前, 这两者通常是相同的, 但在 GB200 NVL72 中, 我们有一个包含 72 个 GPU 的 NVLink 域, 但每个主机仍然只连接 8 个 GPU. 我们在这里使用术语“节点”来指代 NVLink 域, 但这有争议.</d-footnote>) 使用称为 NVLink 的高带宽互连在 1 跳内连接, 这些节点使用连接到每个 GPU 的 NIC, 通过带宽较低的 InfiniBand (IB) 或以太网网络连接成更大的单元 (称为 **SU** 或可扩展单元). 这些单元又可以通过更高级别的交换机连接成任意大的单元.

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>图:</b> 展示典型 H100 网络的示意图. 一组 8 个 GPU 通过 NVSwitches (也称为 NVLink 交换机) 连接成一个节点或 NVLink 域, 这些节点通过交换式 InfiniBand 结构相互连接. H100 在 NVLink 域中每个大约有 450GB/s 的出口带宽, 每个节点有 400GB/s 的出口带宽进入 IB 网络." %}

### 节点层面

GPU 节点是一个小单元, 通常由 8 个 GPU (GB200 最多 72 个) 组成, 通过全对全, 全带宽, 低延迟的 NVLink 互连连接.<d-footnote>NVLink 被向我描述为一种增强版的 PCIe 连接, 具有低延迟和协议开销, 但不为可扩展性/容错性而设计, 而 InfiniBand 更像以太网, 专为更大的有损网络而设计.</d-footnote> 每个节点包含几个高带宽的 NVSwitches, 用于在所有本地 GPU 之间交换数据包. 实际的节点级拓扑随时间变化很大, 包括每个节点的交换机数量, 但对于 H100, 我们每个节点有 4 个 NVSwitches, GPU 以 `5 + 4 + 4 + 5` 的链接模式连接到它们, 如下图所示:

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图:</b> 从 Pascall (P100) 开始的节点 (即 NVLink 域) 示意图. 自 Volta (V100) 以来, 我们通过一组交换机在节点内实现了全对全连接. H100 节点有 4 个 NVSwitches, 通过 25GB/s 的链接连接到所有 8 个 GPU." %}

对于 Hopper 这一代 (NVLink 4.0), 每个 NVLink 链接具有 25GB/s 的全双工<d-footnote>这里的全双工意味着每个方向 25GB/s, 两个方向相互独立. 你可以在链接上总共发送 50GB/s, 但每个方向最多 25GB/s.</d-footnote> 带宽 (B200 为 50GB/s), 这使得每个 GPU 进入网络的带宽为 `18 * 25=450GB/s` 全双工. 庞大的 NVSwitches 最多有 64 个 NVLink 端口, 这意味着一个拥有 4 个交换机的 8xH100 节点可以处理高达 `64 * 25e9 * 4=6.4TB/s` 的带宽. 以下是这些数字随 GPU 世代变化的概览:

| NVLink Gen | NVSwitch Gen | GPU 世代 | NVLink 带宽 (GB/s, 全双工) | NVLink 端口 / GPU | 节点 GPU 间带宽 (GB/s 全双工) | 节点大小 (NVLink 域) | NVSwitches / 节点 |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
| **3.0** | **2.0** | Ampere | 25 | 12 | 300 | 8 | 6 |
| **4.0** | **3.0** | Hopper | 25 | 18 | 450 | 8 | 4 |
| **5.0** | **4.0** | Blackwell | 50 | 18 | 900 | 8/72 | 2/18 |

Blackwell (B200) 的节点有 8 个 GPU. GB200NVL72 支持更大的 72 个 GPU 的 NVLink 域. 我们展示了 8 GPU 和 72 GPU 系统的详细信息.

### 测验 2: GPU 节点

这里有更多关于网络的问题/解答. 我发现做这些特别有用, 因为它们让你实际思考通信模式.

**问题 1 [H100 节点的总带宽]:** 在一个拥有 4 个交换机的 8xH100 节点中, 我们每个节点有多少总带宽? *提示: 同时考虑 NVLink 和 NVSwitch 的带宽.*

{% details 点击这里查看答案. %}

**答案:** 我们有 Gen4 4xNVSwitches, 每个都有 `64 * 25e9=1.6TB/s` 的单向带宽. 这将给我们 `4 * 1.6e12=6.4e12` 的交换机级别带宽. 然而, 请注意, 每个 GPU 只能处理 450GB/s 的单向带宽, 所以这意味着我们最多有 `450e9 * 8 = 3.6TB/s` 的带宽. 由于这个数字更小, 峰值带宽是 3.6TB/s.

{% enddetails %}

**问题 2 [对分带宽]**: 对分带宽定义为网络任意均分后可用的最小带宽. 换句话说, 如果将网络分成相等的两半, 两半之间有多少带宽? 你能计算一个 8x H100 节点的对分带宽吗? *提示: 对分带宽通常包括双向流量.*

{% details 点击这里查看答案. %}

**答案:** 任何偶数分区都会在每一半有 4 个 GPU, 每个 GPU 可以向另一半输出 `4 * 450GB/s`. 考虑到双向流量, 这使得 `8 * 450GB/s` 的字节穿过分区, 即 3.6TB/s 的对分带宽. 这也是 NVIDIA 在例如[这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)报告的.

{% enddetails %}

**问题 3 [AllGather 成本]**: 给定一个 B 字节的数组, 一个 (吞吐量受限的) AllGather 在一个 8xH100 节点上需要多长时间? 对 bf16[D<sub>X</sub>, F] 进行计算, 其中 `D=4096`, `F=65,536`. *在回答这个问题之前, 值得阅读 TPU 集合操作的[部分](https://lqhl.github.io/scaling-book/sharding/). 在这里思考一下, 但我们接下来会更详细地讨论集合操作.*

{% details 点击这里查看答案. %}

**答案:** 每个 GPU 可以输出 450GB/s, 每个 GPU 有 $B / N$ 字节 (其中 `N=8`, 节点大小). 我们可以想象每个节点将其字节依次发送给其他 $N - 1$ 个节点, 导致总共有 (N - 1) 轮, 每轮的 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$, 或者 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$. 这大约是 $B / (N * W_\text{uni})$ 或 $B / \text{3.6e12}$, 即对分带宽.

对于给定的数组, 我们有 `B=4096 * 65536 * 2=512MB`, 所以总时间是 `536e6 * (8 - 1) / 3.6e12 = 1.04ms`. 这可能是延迟受限的, 所以实际上可能需要更长的时间 (实际上大约需要 1.5ms).

{% enddetails %}

## 节点之外

在节点级别之上, GPU 网络的拓扑结构不那么标准化. NVIDIA 发布了一个[参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html), 该架构使用 InfiniBand 连接比单个节点更多的 GPU, 但客户和数据中心提供商可以根据自己的需求进行定制.<d-footnote>例如, Meta 在一个与此描述显著不同的数据中心网络上训练了 LLaMA-3, 使用了以太网, 一个 3 层交换结构, 以及一个在顶层超额认购的交换机.</d-footnote>

这是一个参考的 1024 GPU H100 系统的示意图, 其中底行的每个框都是一个包含 8 个 H100 GPU, 8 个 400Gbps CX7 NIC (每个 GPU 一个) 和 4 个 NVSwitches 的单个 8xH100 节点.

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>图:</b> 1024 H100 DGX SuperPod 参考架构图, 包含 128 个节点 (有时是 127 个), 每个节点有 8 个 H100 GPU, 连接到一个 InfiniBand 横向扩展网络. 32 个节点 (256 个 GPU) 的集合称为 '可扩展单元' 或 SU. 叶交换机和主干 IB 交换机提供了足够的带宽, 以实现节点间的完全对分带宽." %}

**可扩展单元:** 每 32 个节点组成一个“可扩展单元”(或 SU), 隶属于一组 8 个叶 InfiniBand 交换机. 这个 SU 有 256 个 GPU, 每个节点有 4 个 NVSwitches, 还有 8 个 Infiniband 叶交换机. 图中显示的所有布线都是 InfiniBand NDR (50GB/s 全双工), 配备 64 端口 NDR IB 交换机 (每个端口也是 50GB/s). *请注意, IB 交换机的带宽是 NVSwitches 的 2 倍 (64 个端口, 400 Gbps 链接).*

**SuperPod:** 整个 SuperPod 随后用 16 个顶层“主干”IB 交换机连接 4 个这样的 SU, 得到 1024 个 GPU, 512 个节点级 NVSwitches, 32 个叶 IB 交换机和 16 个主干 IB 交换机, 总共有 512 + 32 + 16 = 560 个交换机. 叶交换机以 32 个节点为一组连接到节点, 所以每组 256 个 GPU 有 8 个叶交换机. 所有叶交换机都连接到所有主干交换机.

**我们有多少带宽?** InfiniBand 网络 (称为“横向扩展网络”) 的整体拓扑结构是一个**胖树**, 其电缆和交换机保证了节点级别以上的完全对分带宽 (这里是 400GB/s). 这意味着如果我们将节点分成两半, 每个节点可以同时向另一分区的节点输出 400GB/s. 更重要的是, 这意味着我们在横向扩展网络中应该有大致恒定的 AllReduce 带宽! 虽然可能不是这样实现的, 但你可以想象在横向扩展网络中的任意多个节点上进行环形归约, 因为你可以构建一个包含每个节点的环.

| 级别 | GPU | 每单元交换机数 | 交换机类型 | 每单元带宽 (TB/s, 全双工) | GPU 间带宽 (GB/s, 全双工) | 胖树带宽 (GB/s, 全双工) |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点 | 8 | 4 | NVL | 3.6 | 450 | 450 |
| 叶 | 256 | 8 | IB | 12.8 | 50 | 400 |
| 主干 | 1024 | 16 | IB | 51.2 | 50 | 400 |

相比之下, 一个 TPU v5p 每个链接大约有 90GB/s 的出口带宽, 或者在 3D 环面的所有轴上总共有 540GB/s 的出口带宽. 这不是点对点的, 所以它只能用于受限的, 统一的通信模式, 但它仍然给了我们更高的 TPU 间带宽, 可以扩展到任意大的拓扑 (至少高达 8960 个 TPU).

GPU 交换结构理论上可以通过增加额外的交换机或间接层来扩展到任意大小, 但代价是增加延迟和昂贵的网络交换机.

<p markdown=1 class="takeaway">**要点**: 在一个 H100 节点内, 我们每个 GPU 有 450GB/s 的完整胖树带宽, 而在节点之外, 这个值下降到 400GB/s 的节点间带宽. 这对于通信原语来说至关重要.</p>

**GB200 NVL72s:** NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群, 将 72 个 GPU 组合在一个 NVLink 域中, 具有完整的 900GB/s 的 GPU 间带宽. 这些域可以连接成更大的 SuperPod, 具有成比例更高 (9x) 的 IB 胖树带宽. 以下是该拓扑的示意图:

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图:</b> 展示一个包含 576 个 GPU 的 GB200 DGX SuperPod 的示意图. 底层的每个机架包含 72 个 GB200 GPU." %}

计算单个节点的出口带宽 (上图中的橙色线), 我们有 `4 * 18 * 400 / 8 = 3.6TB/s` 的带宽到叶级别, 这比 H100 多 9 倍 (正如节点包含 9 倍多的 GPU). 这意味着关键的节点出口带宽要高得多, 我们的跨节点集合带宽实际上可能*低于*节点内的带宽.
更多讨论请参见[附录 A](#appendix-a-how-does-this-change-with-gb200).

| 节点类型 | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
| H100 | 8 | 450e9 | 400e9 |
| B200 | 8 | 900e9 | 400e9 |
| GB200 NVL72 | 72 | 900e9 | 3600e9 |

<p markdown=1 class="takeaway">**要点**: GB200 NVL72 SuperPod 大大增加了节点大小和给定节点的出口带宽, 这显著改变了我们的 roofline 模型.</p>

### 测验 3: 节点之外

**问题 1 [胖树拓扑]:** 使用上面的 DGX H100 图, 计算整个 1024 GPU pod 在节点级别的对分带宽. 证明每个链接的带宽选择是为了确保完全的对分带宽. *提示: 确保计算链接带宽和交换机带宽.*

{% details 点击这里查看答案. %}

**答案:** 让我们逐个组件来分析:

*   首先, 每个节点有 8x400Gbps NDR IB 电缆连接到叶交换机, 使得每个节点有 `8 * 400 / 8 = 400 GB/s` 的带宽到叶. 我们有 8 个叶交换机, 每个 3.2TB/s (64 个 400 GBps 链接), 但我们只能使用 64 个端口中的 32 个从 SU 入口, 所以是 `32 * 400 / 8 = 12.8TB/s` 用于 32 个节点, 同样是每个节点 400GB/s.
*   然后在主干级别, 我们有 `8 * 16 * 2` 400Gbps NDR IB 电缆连接每个 SU 到主干, 使得每个 SU 有 `8 * 16 * 2 * 400 / 8 = 12.8 TB/s` 的带宽到叶. 同样, 这是每个节点 400GB/s. 我们有 16 个主干交换机, 每个 3.2TB/s, 得到 `16 * 3.2 = 51.2 TB/s`, 对于 128 个节点, 同样是 400GB/s.

因此, 如果我们以任何方式对分我们的节点, 我们将在它们之间有每个 GPU 400GB/s 的带宽. 每个组件都恰好具有确保胖树所需的带宽.

{% enddetails %}

**问题 2 [扩展到更大的 DGX pod]:** 假设我们想在 2048 个 GPU 而不是 1024 个上进行训练. 修改上述 DGX 拓扑以处理此问题的最简单/最佳方法是什么? 4096 呢? *提示: 没有唯一的正确答案, 但尽量降低成本. 记住链接容量. [这份](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf) 文档可能会有帮助.*

{% details 点击这里查看答案. %}

**答案:** 一种选择是保持 SU 结构不变 (32 个节点在 8 个交换机下), 只需增加更多的 SU 和更多的顶层交换机. 我们需要 2 倍的主干交换机, 所以我们将有 8 个 SU 和 32 个主干交换机, 这将给我们足够的带宽.

这样做的一个问题是, 每个叶交换机只有 64 个端口, 而我们在上图中已经全部使用了. 但相反, 很容易做到每个主干使用 1x 400 Gbps NDR 电缆而不是 2x, 这提供了相同的总带宽但为我们节省了一些端口.

对于 4096 个 GPU, 我们实际上用完了端口, 所以我们需要增加另一层间接, 也就是说, 在层次结构中增加另一层. NVIDIA 称这些为“核心交换机”, 并用 128 个主干交换机和 64 个核心交换机构建了一个 4096 GPU 集群. 你可以计算一下来证明这提供了足够的带宽.

{% enddetails %}

## 集合操作在 GPU 上如何工作?

GPU 可以执行与 TPU 相同的所有集合操作: ReduceScatters, AllGathers, AllReduces, 和 AllToAlls. 与 TPU 不同, 这些操作的工作方式取决于它们是在节点级别 (通过 NVLink) 还是在更高级别 (通过 InfiniBand) 执行. 这些集合操作由 NVIDIA 在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl) (发音为“nickel”) 库中实现. NCCL 在[这里](https://github.com/NVIDIA/nccl)开源. 虽然 NCCL 根据延迟要求/拓扑使用多种实现 ([细节](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)), 但从现在开始, 我们将讨论一个在交换树状结构上的理论最优模型.

### 节点内集合操作

**AllGather 或 ReduceScatter:** 对于节点级别的 AllGather 或 ReduceScatter, 你可以像 TPU 一样围绕一个环执行它们, 在每一跳都使用完整的 GPU 间带宽. 任意排序 GPU, 并使用完整的 GPU 间带宽将数组的一部分沿环发送.<d-footnote>你也可以认为每个 GPU 将其大小为 $\text{bytes} / N$ 的块发送给其他 $N - 1$ 个 GPU, 总共通信了 $(N - 1) * N * bytes / N$ 字节, 这给了我们</d-footnote> 每一跳的成本是 $T_\text{hop} = \text{bytes} / (N * \text{GPU 出口带宽})$, 所以总成本是

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU 出口带宽}} \rightarrow \frac{\text{bytes}}{\text{GPU 出口带宽}}$$

你会注意到这和 TPU 上完全一样. 对于 AllReduce, 你可以像往常一样组合一个 RS + AG, 成本是两倍.

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>图:</b> 带宽最优的 1D 环形 AllGather 算法. 对于 B 字节, 这将 V / X 字节在顶层交换机上传输 X - 1 次." %}

如果你关心延迟 (例如, 如果你的数组非常小), 你可以进行树状归约, 即先在 2 个一组内 AllReduce, 然后是 4 个, 然后是 8 个, 总共有 $\log(N)$ 跳而不是 $N - 1$ 跳, 尽管总成本仍然相同.

<p markdown=1 class="takeaway">**要点:** 在单个节点内对 B 字节的数组进行 AllGather 或 ReduceScatter 的成本约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approxeq B / W_\text{GPU egress}$. 理论上, 在 H100 上约为 $B / \text{450e9}$, 在 B200 上约为 $B / \text{900e9}$. 除非启用了网络内归约, 否则 AllReduce 的成本是这个的两倍.</p>

<b markdown=1 style="color: #57cf57;">小测验 1 [AllGather 时间]:</b> 使用一个具有 450 GB/s 全双工带宽的 8xH100 节点, AllGather(bf16[B<sub>X</sub>, F]) 需要多长时间? 设 $B=1024$, $F=16,384$.

{% details 点击这里查看答案. %}

**答案:** 我们总共有 $2 \cdot B \cdot F$ 字节, 单向带宽为 450e9. 这大约需要 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$, 或者更精确地, $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$. 使用提供的值, 这大约得到 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$, 或者更精确地, $\text{65us}$.

{% enddetails %}

**AllToAlls:** 节点内的 GPU 具有全对全连接性, 这使得 AllToAlls, 嗯, 相当容易. 每个 GPU 只需直接发送到目标节点. 在一个节点内, 对于 B 字节, 每个 GPU 有 $B / N$ 字节, 并向 $N - 1$ 个目标节点发送 $(B / N^2)$ 字节, 总共

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

与 TPU 相比, 其成本为 $B / (4W)$. 因此, 在单个节点内, 我们的运行时间理论上可以提速 2 倍 ($B / 4W$ vs. $B / 8W$).

对于专家混合 (MoE) 模型, 我们经常需要进行*稀疏或不规则的 AllToAll,* 我们保证输出维度上的 $N$ 个分片中最多有 $k$ 个是非零的, 也就是说 $T_\text{AllToAll}_X \rightarrow K[B, N]$, 其中每个轴上最多有 $k$ 个非零条目. 这样做的成本降低了 $k/N$, 总成本约为 $\min(k/N, 1) \cdot B / (W \cdot N)$. 对于 MoE, 我们通常独立随机地选择非零值, 所以有一定几率非零值少于 $k$ 个, 这给了我们大约
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$.<d-footnote>真实的成本实际上是 $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$ 即 $K$ 次掷骰子中不同结果的期望数量, 但它与给出的近似值非常接近. 更多细节请参见附录.</d-footnote>

<b markdown=1 style="color: #c55404ff;">小测验 2 [AllToAll 时间]:</b> 使用一个具有 450 GB/s 单向带宽的 8xH100 节点, AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) 需要多长时间? 如果我们知道 8 个条目中只有 4 个是非零的呢?

{% details 点击这里查看答案. %}

**答案:** 从上面我们知道, 在密集的情况下, 成本是 $B \cdot (N-1) / (W \cdot N^2)$, 或 $B / (W \cdot N)$. 如果我们知道只有 $\frac{1}{2}$ 的条目是非填充的, 我们可以发送 $B \cdot k/N / (W \cdot N) = B / (2 \cdot W \cdot N)$, 大约是总成本的一半.

{% enddetails %}

<p markdown=1 class="takeaway">**要点:** 在单个节点内对 $B$ 字节的数组进行 AllToAll 的成本约为 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$. 对于不规则 (top-$k$) AllToAll, 这个成本进一步降低到 $(B \cdot k) / (64 \cdot W_\text{GPU egress})$.</p>

**实证测量:** 这是在一个 8xH100 节点上 AllReduce 带宽的实证测量. Algo BW 是测量的带宽 (字节 / 运行时间), Bus BW 计算为 $2 \cdot W \cdot (8 - 1) / 8$, 理论上是实际链路带宽的度量. 你会注意到我们确实达到了接近 370GB/s, 低于 450GB/s 但相当接近, 尽管每个设备只有大约 10GB. 这意味着虽然这些估计在理论上是正确的, 但需要一个大的消息才能实现它.

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图:</b> 禁用 SHARP 的 8xH100 节点的 AllReduce 吞吐量. 蓝色曲线是根据实证测量计算出的经验链路带宽, 计算公式为 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$. 注意, 即使使用巨大的 10GB 数组, 我们也没有特别接近声称的 450GB/s 带宽." %}

这是一个实际问题, 因为它有意义地复杂化了我们可以做出的任何理论声明, 因为例如, 即使是在一个合理大小的数组上进行 AllReduce, 比如 LLaMA-3 70B 的 MLP (大小为 `bf16[8192, 28672]`, 或者 8 路模型分片后为 `bf16[8192, 3584] = 58MB`), 也只能达到大约 150GB/s, 而峰值是 450GB/s. 相比之下, TPU 在小得多的消息大小下就能达到峰值带宽 (见附录 B).

<p markdown=1 class="takeaway">**要点:** 尽管 NVIDIA 声称 H100 NVLink 的带宽约为 450GB/s, 但在实践中很难超过 370 GB/s, 因此请相应调整上述估计.</p>

**网络内归约:** 自 Hopper 这一代以来, NVIDIA 交换机支持 ["SHARP" (可扩展分层聚合和归约协议)](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/), 允许“网络内归约”. 这意味着*网络交换机本身*可以执行归约操作, 并将结果多路复用或“多播”到多个目标 GPU:

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>图:</b> 没有 SHARP 的 AllReduce 理论成本是 2 倍, 因为它必须两次通过每个 GPU. 实际上, 速度提升只有大约 30% (来自 NCCL 2.27.5)." %}

理论上, 这将 AllReduce 的成本减少了近一半, 因为这意味着每个 GPU 可以将其数据发送到一个顶层交换机, 该交换机本身执行归约并将结果广播到每个 GPU, 而不必两次从每个 GPU 出口, 同时也减少了网络延迟.

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU 出口带宽}}$$

请注意, 这是精确的, 而不是差一个 $1/N$ 的因子, 因为每个 GPU 首先出口 $B \cdot (N - 1) / N$, 然后接收其本地分片的局部归约版本 (入口 $B/N$), 完成归约, 然后再次出口 $B/N$, 然后入口完全归约的结果 (入口 $B \cdot (N - 1) / N$), 导致正好入口 $B$ 字节.

然而, 在实践中, 我们看到启用 SHARP 后带宽增加了约 30%, 而预测是 75%. 这仅仅使我们的有效集合带宽达到约 480GB/s, 远非 2 倍.

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>图:</b> 在节点内启用和禁用 NVIDIA SHARP 的 AllReduce 算法带宽的实证测量. 尽管算法上应该能实现接近 75% 的增益, 但在峰值时, 增益仅为约 30% 的吞吐量提升." %}

<p markdown=1 class="takeaway">**要点:** 理论上, NVIDIA SHARP (在大多数 NVIDIA 交换机上可用) 应该将对 $B$ 字节进行 AllReduce 的成本从大约 $2 * B / W$ 降低到 $B / W$. 然而, 在实践中, 我们只看到大约 30% 的带宽提升. 由于纯 AllReduce 在 LLM 中相当罕见, 这并不是特别有用.</p>

### 跨节点集合操作

当我们超越节点级别时, 成本就有点微妙了. 当在树上进行归约时, 你可以想象从下往上归约, 首先在节点内, 然后在叶级别, 然后在主干级别, 在每个级别都使用正常的算法. 特别是对于 AllReduce, 你可以看到这使我们能够总体上传输更少的数据, 因为在节点级别进行 AllReduce 后, 我们只需要向叶级别出口 $B$ 字节, 而不是 $B * N$.

**这有多昂贵?** 作为一阶近似, 因为我们有完全的对分带宽, AllGather 或 ReduceScatter 的成本大约是缓冲区大小 (以字节为单位) 除以节点出口带宽 (H100 上为 400GB/s), *而与树状归约的任何细节无关.*

$$T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

其中 $W_\text{node}$ 出口通常是上述 H100 网络的 400GB/s (8x400Gbps IB 链路从每个节点出口). 想象这一点的最清晰方式是想象在*集群中的每个节点*上进行环形归约. 由于胖树拓扑, 我们总是可以在任意两个节点之间构建一个具有 $W_\text{node}$ 出口的环, 并进行正常的归约. 节点级别的归约 (几乎) 永远不会成为瓶颈, 因为它具有更高的总带宽和更好的延迟, 尽管通常成本是

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]$$

{% details 你可以在这里看到更精确的推导. %}

我们可以更精确地指出, 我们实际上是在网络的每一层进行环形归约, 我们可以大部分重叠它们, 所以我们有:

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

其中 $D_i$ 是深度 $i$ 的度 (深度 $i$ 的子节点数), $W_\text{link i}$ 是连接每个子节点到节点 $i$ 的链路带宽.

使用这个, 我们可以计算可用的 AllGather/AllReduce 带宽为 $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$ 对于给定的拓扑. 在上面的情况下, 我们有:

*   **节点:** $D_\text{node}$ = 8, 因为我们一个节点有 8 个 GPU, Wlink i = 450GB/s. 因此我们的 AG 带宽是 `450e9 * 8 / (8 - 1) = 514GB/s`.
*   **叶:** $D_\text{leaf}$ = 32, 因为我们一个 SU 有 32 个节点, Wlink i = 400GB/s (8x400Gbps IB 链路). 因此我们的带宽是 `400e9 * 32 / (32 - 1) = 413GB/s`.
*   **主干:** $D_\text{spine}$ = 4, 因为我们有 4 个 SU, $W_\text{link i}$ = 12.8TB/s (来自上面的 `8 * 16 * 2 * 400Gbps` 链路). 我们的带宽是 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`.

因此, 我们的整体 AG 或 RS 带宽是 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s` 在叶级别, 所以实际上 $T_\text{AG or RS comms} = B / \text{413GB/s}$, 即即使在最高级别, 我们也有大约 413GB/s 的 AllReduce 带宽. 对于带 SHARP 的 AllReduce, 它会略低于这个值 (约 400GB/s), 因为我们没有 $(N - 1) / N$ 的因子. 尽管如此, 450GB/s 和 400GB/s 作为近似值已经足够接近了.

{% enddetails %}

**其他集合操作:** 除非启用 SHARP, 否则 AllReduce 的成本仍然是上述成本的 2 倍. NVIDIA 也销售支持 SHARP 的 IB 交换机, 尽管并非所有提供商都有. AllToAlls 在跨节点时变化很大, 因为它们不像 AllReduce 那样是“分层”的. 如果我们想将数据从每个 GPU 发送到每个其他 GPU, 我们无法利用节点级别的完全对分带宽. 这意味着如果我们有一个跨越 $M = N / 8$ 个节点的 N 路 AllToAll, 成本是

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approxeq \frac{B}{M \cdot W_\text{node egress}}$$

这实际上具有 50GB/s 而不是 400GB/s 的带宽. 我们从单个 H100 节点内的 $B / (8 * \text{450e9})$ 变为跨越 2 个节点时的 $B / (2 \cdot \text{400e9})$, 性能下降超过 4 倍.

以下是 1024-GPU DGX H100 SuperPod 架构的摘要:

| 级别 | GPU 数量 | 度 (# 子节点) | 交换机带宽 (全双工, TB/s) | 电缆带宽 (全双工, TB/s) | 集合带宽 (GB/s) |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
| 节点 | 8 | 8 | 6.4 | 3.6 | 450 |
| 叶 (SU) | 256 | 32 | 25.6 | 12.8 | 400 |
| 主干 | 1024 | 4 | 51.2 | 51.2 | 400 |

我们使用术语“集合带宽”来描述我们可以从 GPU 或节点出口的有效带宽. 它也是 $\text{对分带宽} * 2 / N$.

<p markdown=1 class="takeaway">**要点:** 在节点级别之上, 对 B 字节进行 AllGather 或 AllReduce 的成本大约是 $B / W_\text{node egress}$, 在 H100 DGX SuperPod 上是 $B / \text{400e9}$. 整体拓扑是一个胖树, 旨在在任意两对节点之间提供恒定的带宽.</p>

**当数组在单独的轴上分片时的归约:** 考虑一个归约的成本, 比如

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

其中我们对一个本身沿着另一个轴 $Y$ 分片的数组进行 AllReduce. 在 TPU 上, 这个操作的总成本与未分片版本相比降低了 $1 / Y$ 的因子, 因为我们每个轴发送的数据量是 $1 / Y$. 在 GPU 上, 成本取决于哪个轴是“内部”轴 (节点内 vs. 节点间) 以及每个分片是否跨越多个节点. 假设 $Y$ 是这里的内部轴, 总成本有效地降低了 $Y$, 但前提是 $Y$ 跨越多个节点:

$$T_\text{comms at node} = \frac{\text{bytes} \cdot D_\text{node}}{\min(Y, D_\text{node}) \cdot W_\text{GPU egress}}$$

$$T_\text{comms in scale-out network} = \frac{\text{bytes} \cdot N}{Y \cdot W_\text{node egress}}$$

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})$$

其中 N 是 GPU 的数量, D 再次是节点中的 GPU 数量 (节点的度). 如你所见, 如果 $Y < D_\text{node}$, 我们在节点级别获得了好处, 但通常不会看到总运行时间的减少, 而如果 $Y > D_\text{node}$, 我们会得到一个与跨越的节点数量成比例的加速.

如果我们想精确地进行环形归约, 树状 AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> }) 的一般规则 (假设 Y 是内轴) 是

$$T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]$$

其中 $S_i$ 是 M * N * …, 即树中 i 级以下子节点的大小. 这大致是说, 我们跨越的 GPU 或节点越多, 我们可用的带宽就越大, 但仅限于该节点内.

**小测验 3 [沿 2 个轴分片]:** 假设我们想在一个 SU (256 个芯片) 上执行 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$, 其中 $Y$ 是内轴. 这将花费多长时间, 作为 $D$, $F$ 和 $Y$ 的函数?

{% details 点击这里查看答案. %}

**答案:** 我们可以将其分为两种情况, Y <= 8 和 Y > 8. 当 $Y <= 8$ 时, 我们仍然受叶交换机的限制, 所以答案和往常一样, $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$. 当 Y > 8 时, 我们从上面得到, 大约是

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

对于 `D = 8192`, `F = 32,768`, 我们有:

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>图:</b> 随着内轴跨越更多节点, 分片 AllGather 的理论成本." %}

注意, 如果我们恰好进行 8 路模型并行, 我们确实将节点级归约的成本降低了 8 倍, 但总成本保持不变, 所以这是免费的, 但对提高总带宽没有帮助.

{% enddetails %}

<p markdown=1 class="takeaway">**要点:** 当我们有多个分片轴时, 外部归约的成本会因内轴跨越的节点数量而降低一个因子.</p>

### 测验 4: 集合操作

**问题 1 [SU AllGather]:** 只考虑一个 SU, 它有 M 个节点, 每个节点有 N 个 GPU. 在 AllGather 期间, 节点级交换机精确地入口和出口多少字节? 顶层交换机呢?

{% details 点击这里查看答案. %}

**答案:** 让我们一步一步来, 逐个分析归约的组成部分:

1.  每个 GPU 向交换机发送 $B / MN$ 字节, 总入口为 $NB / MN = B / M$ 字节.
2.  我们将完整的 $B / M$ 字节出口到主干交换机.
3.  我们从主干交换机入口 $B * (M - 1) / M$ 字节.
4.  我们出口 $B - B / MN$ 字节 $N$ 次, 总共 $N * (B - B / MN) = NB - B / M$.

总共是 $B$ 入口和 $BN$ 出口, 所以我们应该受出口的瓶颈限制, 总时间将是 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$.

对于主干交换机, 计算实际上更简单. 我们必须有 $B / M$ 字节入口 M 次 (总共 $B$ 字节), 然后 $B (M - 1) / M$ 出口 $M$ 次, 总共 $B * (M - 1)$ 出口. 由于这个数字明显更大, 成本是 $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$.

{% enddetails %}

**问题 2 [单节点 SHARP AR]:** 考虑一个单节点, 每个节点有 N 个 GPU. 在使用 SHARP (网络内归约) 进行 AllReduce 期间, 交换机精确地入口和出口多少字节?

{% details 点击这里查看答案. %}

**答案:** 和之前一样, 让我们一步一步来.

1.  每个 GPU 发送 $B * (N - 1) / N$ 字节, 所以我们有 $N * B * (N - 1) / N = B * (N - 1)$ 入口.
2.  我们累加部分和, 然后向每个 GPU 发回 $B / N$ 字节, 所以 $N * B / N = B$ 字节出口.
3.  我们在本地对残差进行部分求和, 然后将其发送回交换机. 这总共是 $N * B / N = B$ 字节入口.
4.  我们捕获所有分片并进行多播, 向 $N$ 个目的地发送 $B * (N - 1) / N$, 总共 $B * (N - 1) / N * N = B * (N - 1)$ 出口.

因此, 总共是 $B * (N - 1) + B = BN$ 字节入口和出口. 这支持总吞吐量恰好是 $B / W_\text{egress}$.

{% enddetails %}

**问题 3 [跨节点 SHARP AR]:** 考虑一个在单个 N GPU 节点上分片的数组 bf16[D<sub>X</sub>, F<sub>Y</sub>]. AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> }) 需要多长时间? 你可以假设我们进行网络内归约. 解释一下如果我们有多个节点, 这会有什么不同?

{% details 点击这里查看答案. %}

**答案:** 我们可以尝试修改上面问题的答案. 基本上, 我们首先从每个 GPU 出口 $B * (X - 1) / XY$ 字节, 然后向每个 GPU 发回 $B / XY$, 然后将相同数量的字节发回交换机, 然后向每个 GPU 发回 $B * (X - 1) / XY$. 总共是 $NB / Y$ 入口和出口, 所以总时间是 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$, 所以总时间确实随着 $Y$ 的增加而减少.

如果我们超越单个节点, 我们可以进行与上面大致相同的归约, 但是当我们从节点级交换机出口时, 我们需要发送所有 B 字节, 而不仅仅是 $B / Y$. 这是因为我们需要保持每个分片是独立的.

{% enddetails %}

**问题 4 [主干级 AR 成本]:** 考虑与上面相同的设置, 但 $Y = 256$ (所以 AR 发生在主干级). AllReduce 需要多长时间? 同样, 可以假设网络内归约.

{% details 点击这里查看答案. %}

**答案:** 这让我们能够利用主干级别相当可观的带宽. 我们在 4 个节点上有 25.6TB/s 的带宽, 所以 AllReduce 带宽为 6.4TB/s. 使用 SHARP, 这可能只需要 `2 * D * F / 6.4e12` 秒.

{% enddetails %}

**问题 5 [2 路 AllGather 成本]:** 计算在恰好 2 个节点上进行 $B$ 字节的 AllGather 的精确成本. *确保计算精确成本而不是近似值, 并考虑节点内和跨节点的成本.*

{% details 点击这里查看答案. %}

**答案:** 在节点级别, 我们有 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$, 而在节点之外, 我们实际上有 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$. 因此, 我们实际上受节点级归约的限制, 而不是叶级! 这也解释了为什么 DeepSeek v3 等会进行 2 路数据并行.

{% enddetails %}

## GPU 上 LLM 扩展的 Roofline 模型

现在让我们来看看这一切都是为了什么: 理解 GPU 上 LLM 扩展的 roofline 模型. 这是对 TPU 训练章节[这里](../training)的补充. 和那里一样, 这里的目标是查看不同并行策略的总 $T_\text{math}$ 和 $T_\text{comms}$, 并理解在什么点上 $T_\text{comms} > T_\text{math}$. 和之前一样, 我们只考虑 MLP 块, 其操作为

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是全局批处理大小 **(以 token 为单位)** (即 $B = \text{批处理大小} \cdot \text{序列长度}$).

这里我们将重现上面的表格, 显示 GPU 和节点级别的有效带宽:

| 节点类型 | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
| H100 | 8 | 450e9 | 400e9 |
| B200 | 8 | 900e9 | 400e9 |
| GB200 NVL72 | 72 | 900e9 | 3600e9 |

**注意:** GPU 和节点的出口带宽都决定了我们 LLM 的 roofline. 我们将使用术语 $W_\text{collective}$ 来描述 GPU 或节点的带宽, 具体取决于我们是在节点内还是节点之上操作.

让我们像对 TPU 那样, 查看**数据并行, 张量并行, 流水线并行, 专家并行**以及它们的组合的计算通信 roofline. 本节的其余部分, 我们将专注于 H100 的 roofline 进行具体计算. GB200-NVL72 具有相同的一般 roofline, 但由于我们有更大的节点出口带宽, 我们有时可能会在节点级别遇到瓶颈.

### 数据并行

如前所述, DP 和 ZeRO 分片涉及在反向传播中进行权重 AllReduce 或 ReduceScatter + AllGather. 由于这两者的成本相同, 要想在纯数据并行或 FSDP *没有网络内归约*的情况下达到计算密集型, 我们在反向传播中, 对于大小为 X 的轴, 每层有:

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

因此, 要使 $T_\text{math} > T_\text{comms}$, 我们需要 $B / (XC) > 1 / W_\text{collective}$ 或

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中 $W_\text{collective}$ 是 GPU 或节点级别的出口带宽, 具体取决于我们是在节点内还是跨节点分片. 因此:

*   **在节点内**, 我们只需要每个 GPU 的 **token** 批处理大小 > $\text{990e12} / \text{450e9} = 2200$.
*   **在 SU 内或主干级别**, BS > $\text{990e12} / \text{400e9} = 2475$.

这比 TPU 上的数字要高得多, TPU 上所有三个轴的数字是 850. 例如, 在 16000 个 H100 上训练的 LLaMA-3, 将需要至少 40M token 的批处理大小 (作为参考, 他们使用了 16M). 在 2048 个 H800 GPU 上训练的 DeepSeek v3, 带宽较低, 为 300GB/s (而不是 H100 上的 450GB/s), 将需要每个 GPU $\text{990e12} / \text{300e9} = 3300$ 个 token, 或约 6.7M (实际上, 他们使用了 4M).

启用网络内归约并使用纯数据并行, 理论上我们的 AllReduce 带宽是 2 倍, 这将使这两个数字减半. 然而, 实际上收益接近 30%, 这仅仅弥补了我们通常难以达到报告数字的事实. 此外, 因为纯数据并行很少有用, 这在实践中基本上无关紧要.

**MoE 模型:** 对于专家混合 (MoE) 模型, 其中我们有 E 个专家, 每个 token 有 k 个专家, 这增加到

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

这将每个 GPU 的 token 批处理大小增加了 $E/k$ 的因子, 即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如, 对于新的 OpenAI OSS 模型, $k=4$ 且 $E=128$, 这在跨节点时增加到 `32 * 2475 = 79,200`, 这是一个相当高的数字.

**当 X 很小时会发生什么?** 当我们只进行例如 2 节点数据并行时, 我们受益于 $(X - 1) / X$ 的缩放, 这给了我们

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中 X 是节点数, $N = 8 \cdot X$. 那么对于密集模型, 我们有 $B / N > \alpha \cdot (X - 1) / X$, 或例如 $B / N > \text{1237}$, 是上述值的一半. 出于这个原因, 你会相当频繁地看到 2 路数据并行.

<p markdown=1 class="takeaway">**要点:** 数据并行和 ZeRO 分片需要每个 GPU 约 2500 个 token 的批处理大小才能达到计算密集型, 假设完美的重叠和 FLOPs 利用率. 对于 MoE 模型, 这会增加一个因子 $E / k$, 即总参数与激活参数的比率. 当进行少量数据并行时, 临界批处理大小会减小.</p>

### 张量并行

张量并行需要在激活值上进行 AllGather 和 ReduceScatter, 我们需要将其与 MLP FLOPs 重叠. 换句话说, 在前向传播中, 我们有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

要达到计算密集型, 这给了我们规则

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

在节点内, 这给了我们大约 $F / 2200$, 在节点外是 $F / 2475$. 对于像 LLaMA-3 那样的 $F=\text{28000}$, 这大约是 11 路 TP (或者向下取整, 大约 8 路, 这是一个节点的大小). 和上面一样, 当我们跨越正好 2 个节点时, 我们获得了额外的 2 倍带宽, 所以我们通常可以进行 16 路数据并行 ($F > 2475 \cdot (Y - 8)$), 这理论上给了我们高达 19 路的模型并行.

<p markdown=1 class="takeaway">**要点:** 在大小为 Y 的轴上进行张量并行, 前馈维度为 F, 当 $Y > F / 2475$ 时会变得受通信限制, 这通常将我们限制在节点内 TP 或最多 2 节点 TP.</p>

### 专家并行

正如我们上面已经指出的, 专家混合 (MoE) 模型的模型权重是 E 倍, 而 FLOPs 只有 k 倍, 这使得数据并行明显更难. 我们可以通过沿专家维度分片我们的权重来在一定程度上缓解这个问题, 即 W<sub>in</sub>[E<sub>Z</sub>, D, F]. 要执行 MLP 块, 我们需要引入 2 次 AllToAll 来将我们的激活值发送到相应的专家.

如上所述, 如果这个 AllToAll<sub>Z->k</sub>([B, D, k]) 跨越多个节点, 其成本大约是 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$, 所以对于纯专家并行, 我们需要

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们要么需要 $K > Z/8$ 且 $F > \alpha \cdot (Z - 8)/k$, 要么 $Z \gg K$ 且 $F > 8 \cdot \alpha$, 其中 $\alpha = C/W$. 这给了你两个可以进行专家并行的领域, 一个是少量专家并行 (大约 2 节点) 和小 $F$, 另一个是大 $F$ 和任意大的 $Z$ (高达 E 路专家并行).

你在实践中会看到这两种情况, 要么是少量的专家并行 (比如 DeepSeek v3, 它的 F 非常小, 跨节点专家并行相对较小且受限), 要么是 F 很大的模型, 在这种情况下我们可以进行显著的跨节点 EP 和 TP.

<p markdown=1 class="takeaway">**要点:** 如果 $F < 8 * C / W_\text{node}$, 专家并行可以跨越 1-2 个节点, 成本与 TP 相似 (略低), 或者如果 $F > 8 * C / W_\text{node}$, 我们可以进行大量的专家并行 (最多 $E$ 个节点), 成本相对较低.</p>

### 流水线并行

流水线并行以极低的通信成本将层分布在节点之间, 因为我们只是每隔几层发送小的微批次激活值. 历史上, 流水线一直受到“流水线气泡”的困扰, 但随着新的零气泡流水线方法的出现, 通常可以避免这种情况.

流水线的总通信成本很小: 有 $N_\text{MB}$ 个微批次和 $N_\text{stages}$ 个阶段, 我们有 $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$ 和 $N_\text{MB} + N_\text{stages} - 2$ 跳, 所以大约是

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{microbatches}} \cdot (N_\text{microbatches} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于我们除以 $N_\text{layers}$, 这比任何其他成本都要小得多. 换句话说, 从通信的角度来看, 流水线基本上是免费的. 那么为什么我们不只做流水线呢? 有几个原因:

(1) **代码复杂性:** 流水线不像其他方法那样能很好地融入自动并行框架 (如 XLA 的 GSPMD). 因为它引入了微批处理来隐藏流水线气泡, 它改变了程序的结构, 而定制的零气泡流水线调度通过要求前向和后向传播的复杂交错加剧了这个问题.

(2) **流水线使数据并行和 FSDP 变得困难:** 可能不做流水线的最大原因是它与 FSDP 和数据并行的兼容性不好. 特别是 ZeRO-3 分片效果很差, 因为它要求我们在每个微批次上 AllGather 权重, 当我们只有 $B / N_\text{microbatches}$ 个 token 来摊销 AllGather 成本时, 这是行不通的. 此外, 在反向传播期间, *我们无法在最后一个微批次通过给定阶段之前对梯度进行 AllReduce 或 ReduceScatter, 这意味着我们有显著的未重叠的通信时间.*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>图:</b> 一个 2 阶段, 2 微批次流水线的示例. F 表示阶段前向传播, B 是阶段后向传播 (成本是 2 倍). G 表示数据并行 AllReduce, 其时间可能明显长于单个微批次的时间." %}

(3) **流水线气泡和步骤不平衡:** 正如你在上面 (糟糕的) 流水线调度中看到的, 在一个简单的流水线调度中很容易出现显著的气泡 (意味着浪费的计算). 上面, 第二阶段在步骤 0 是空闲的, 第一阶段从步骤 2 到 3 是空闲的, 第二阶段在最后一步再次是空闲的. 虽然我们可以通过仔细的调度在一定程度上避免这些, 但我们仍然经常有一些气泡. 我们还必须在关键路径上将激活值从一个阶段传递到下一个阶段, 这会增加开销:

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>图:</b> 一个显示红色传输成本的流水线示例. 这会使阶段相对移动, 并增加流水线气泡开销." %}

对于这些问题中的每一个都有变通方法, 但它们往往实现起来复杂且难以维护, 但流水线仍然是一种相对于其他方法通信成本较低的技术.

**关于延迟的警告:** 如前所述, 即使消息相当大, GPU 也难以实现完整的 AllReduce 带宽. 这意味着即使我们理论上可以跨多个节点扩展例如专家并行的 AllToAlls, 我们也可能难以达到总带宽的 50%. 这意味着我们确实尝试将 TP 或 EP 保持在较少数量的节点内, 以最小化延迟开销.

### 示例

**DeepSeek 是如何做的?** 作为参考, [DeepSeek V3](https://arxiv.org/abs/2412.19437) 是用 2048 个 H800 GPU 训练的, 配置如下:

*   64 路专家并行 (EP), 跨越 8 个节点
*   16 路流水线并行 (PP)
*   2 路 ZeRO-1 数据并行 (DP)

他们的稳态批处理大小为 `4096 * 15360 = 62,914,560` 个 token, 或每个 GPU 30k 个 token. 你可以看到这已经相当大了, 但他们的模型也非常稀疏 (k=8, E=256), 所以你需要一个相当大的批处理大小. 你可以看到, 通过 64 路 EP 和 16 路 PP, 我们总共得到了 1024 路模型并行, 这意味着 AllReduce 是在主干级别完成的, 而且因为只有 2 路, 我们实际上获得了 $2 / (2 - 1) = 2$ 倍的带宽. 这也有助于降低与最后流水线阶段重叠的最终数据并行 AllReduce 的成本.

**LLaMA-3 是如何做的?** LLaMA-3 在 16k 个 GPU 上以 16M token 的 BS 进行训练, 或每个 GPU 约 1k 个 token. 他们采用:

*   节点内 8 路张量并行 (TP)
*   16 路流水线并行 (PP)
*   128 路 ZeRO-1 数据并行

这也是一个密集模型, 所以总的来说这些事情都相当简单. 16 路 PP 将数据并行 AllReduce 的成本降低了 16 倍, 这有助于我们降低临界批处理大小.

### GPU 上 LLM 扩展的 TLDR

让我们退后一步, 对我们到目前为止学到的东西做一个总结:

*   **数据并行或 FSDP (ZeRO-1/3) 需要每个 GPU 约 2500 个 token 的本地批处理大小**, 尽管理论上网络内归约 + 纯 DP 可以稍微降低这个数字.
*   **张量并行在最多约 8 路时是计算密集型的**, 但我们缺乏带宽来在此之上进行太多扩展, 否则会变得受通信限制. 这主要将我们限制在单个 NVLink 域 (即单节点或需要使用 GB200NVL72, 最多 72 个 GPU).
*   **任何跨越多个节点的模型并行形式都可以进一步降低 FSDP 的成本**, 所以我们经常希望混合 PP + EP + TP 来跨越许多节点并降低 FSDP 成本.
*   **如果你能处理零气泡流水线的代码复杂性, 并保持相当大的批处理大小以避免数据并行瓶颈, 那么流水线并行效果很好.** 流水线通常使 ZeRO-3 变得不可能 (因为你需要在每个流水线阶段进行 AllGather), 但你可以改用 ZeRO-1.

**从高层次来看, 这给了我们在 GPU 上分片大型模型的秘诀:**

*   对于相对较小的密集模型, 如果你有足够的批处理大小, 激进的 FSDP 效果很好, 如果需要, 可能带有一些流水线或张量并行.
*   对于较大的密集模型, 1-2 节点 TP + 多节点 PP + 纯 DP 的某种组合效果很好.
*   对于 MoE, 上述规则适用, 但我们也可以进行专家并行, 我们通常更喜欢它而不是 TP. 如果 $F > 8 * C / W_\text{node}$, 我们可以进行大量的多节点专家并行, 否则我们受限于大约 2 节点 EP.

### 测验 5: LLM Rooflines

**问题 1 [B200 rooflines]:** 一个 B200 DGX SuperPod (**不是 GB200 NVL72**) 在节点内有 2 倍的带宽 (900GB/s 出口), 但在横向扩展网络中的带宽相同 (400GB/s) ([来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)). 总 FLOPs 如上所述. 这如何改变模型和数据并行的 roofline?

{% details 点击这里查看答案. %}

**答案:** 我们的 bfloat16 FLOPs/s 从 990 增加到 2250 TFLOPs, 增加了 2.25 倍. 带宽增加 2 倍, 在节点内, 我们的 roofline 大致保持不变. 例如, 对于 TP, 临界强度上升到 `2250e12 / 900e9 = 2500`, 所以我们的限制是 $Y < F / 2500$, 仅略高 (除非节点大小增加, 否则这对我们没有帮助).

然而, 在节点之外, 缺乏额外的带宽实际上使我们更难达到计算密集型! 例如, 对于数据并行, 我们的临界批处理大小增加到 `2250e12 / 400e9 = 5625`, 因为我们的 GPU 可以用相同的带宽执行明显更多的 FLOPs.

具有 72-GPU 节点的 GB200 SuperPod 通过增加更多的出口带宽改变了这一点 ([来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)).

{% enddetails %}

**问题 2 [如何分片 LLaMA-3 70B]:** 考虑 LLaMA-3 70B, 使用 bfloat16 训练, Adam 优化器状态为 fp32.

1.  仅仅为了存储权重和优化器, 我们至少需要多少个 H100?
2.  假设我们想在 4096 个 H100 GPU 上训练 15T token. 假设我们达到了 45% 的 MFU (模型 FLOPs 利用率). 训练需要多长时间?
3.  LLaMA-3 70B 的 `F = 28,672`, 训练时批处理大小约为 4M token. 我们最多可以进行多少模型并行而不受通信限制? 有了这个加上纯 DP, 我们能否在保持计算密集型的情况下训练 LLaMA-3? ZeRO-3 呢? 8 路流水线呢?

{% details 点击这里查看答案. %}

1.  我们需要 2 字节用于权重, 8 字节用于优化器状态, 所以至少需要 700GB. H100 有 80GB 的 DRAM, 我们至少需要 9 个 GPU, 或者 (向上取整) 至少 2 个 8xH100 节点. 这将需要很长时间来训练, 并且无法保存梯度检查点, 但这是一个下限.
2.  这将需要总共 `6 * 70e9 * 15e12 = 6.3e24 bf16 FLOPs`. 每个 GPU 可以执行 `990e12` FLOPs, 所以在 40% MFU 下, 我们可以执行 1.6e18 FLOPs/s. 因此, 整个过程将需要 3.9e6 秒, 或 45 天.
3.  在节点内, 我们有 450GB/s 的带宽, 所以限制大约是 `F / 1995 = 28672 / 1995 = 14.372`. 由于这不跨越 2 个节点, 实际上这意味着我们将进行最多 8 路模型并行.
    1.  这将要求我们进行 512 路 DP. 首先, 我们需要看看是否有足够的内存. 由于我们的模型只分片了 8 路, 这意味着 `700GB / 8 = 87.5GB / GPU`, 这装不下, 所以不行!
    2.  使用 ZeRO-3 和 8 路 TP, 我们将进行 512 路 ZeRO-3. 这不会有任何内存问题, 因为我们正在积极地分片所有东西. 我们的每个 GPU 批处理大小将是 `4e6 / 4096 = 976`. 这相当低, 甚至低于我们的纯 DP 限制, 而且这是该限制的两倍, 因为我们必须移动我们的权重. 所以不行.
    3.  使用 8 路流水线, 每个模型并行分片现在跨越 8 个节点. 正如我们所见, 这将我们的叶级 AllGathers 的成本降低了 8 倍, 所以那里的总 AllReduce/AllGather 带宽从 400GB/s 增加到 `8 * 400GB/s = 3200GB/s`. 那么 roofline 就是 `989e12 / 3200e9 = 309`, 所以我们应该没问题! 我们只需要高效地实现流水线.

{% enddetails %}

**问题 3 [Megatron-LM 超参数]:** 考虑[Megatron-LM 仓库](https://github.com/NVIDIA/Megatron-LM)中的这张图, 突出了他们的高 MFU 数字.

{% include figure.liquid path="assets/gpu/megatron-hparams.png" class="img-fluid" %}

注意, 他们的序列长度到处都是 4096. 对于 16B, 70B 和 314B 模型, 每个 GPU 的 token 批处理大小是多少? 假设数据并行是最外层的轴, 并假设 bfloat16 归约, 判断这些中的每一个在理论上是计算密集型还是通信密集型, 以及是否有更优的配置可用?

{% details 点击这里查看答案. %}

**答案:** 让我们从每个 GPU 的批处理大小开始.

*   **16B**: `192 * 4096 / 192 = 4096` tokens / GPU
*   **70B**: `384 * 4096 / 768 = 2048` tokens / GPU
*   **314B**: `1536 * 4096 / 3072 = 2048` tokens / GPU

这意味着除了第一个, 这些都徘徊在 2k token / 批处理左右, 这值得注意的是在我们为 FSDP 计算的临界阈值附近. 我们计算出该界限是 2,472 token / GPU, 基于主干级归约, 这应该在这里大致发挥作用. 然而, 对于 70B 和 314B, 因为我们分别有 16 路和 64 路模型分片, 我们在主干级别获得了 2 倍和 8 倍的吞吐量提升, 这意味着我们应该分别在约 1k 和 300 token / 步时达到计算密集型.

{% enddetails %}

## 致谢和延伸阅读

本章在很大程度上依赖于许多知识渊博的 GPU 专家的帮助, 包括:

*   Adam Paszke, 他帮助解释了 GPU 上内核编程的现实情况.
*   Swapnil Patil, 他首先解释了 GPU 网络的工作原理.
*   Stas Bekman, 他指出 GPU 的经验现实往往与声称的规格不同.
*   Reiner Pope, 他帮助阐明了 GPU 和 TPU 在硬件层面的比较.
*   Frédéric Bastien, 他对芯片级的故事给出了详细的反馈.
*   Nouamane Tazi, 他在 GPU 上训练 LLM 的经验帮助改进了 roofline 部分.
*   Sanford Miller, 他帮助我理解了 GPU 是如何联网的, 以及 NVIDIA 的规格与现场经常部署的规格相比如何.

关于 GPU 有大量的好读物, 但我最喜欢的一些包括:

*   [SemiAnalysis 的 NVIDIA 张量核心历史](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/): 一篇精彩的文章, 描述了 GPU 如何从视频游戏引擎转变为 ML 加速器.
*   [SemiAnalysis 的 Blackwell 性能分析](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/): 值得一读, 以了解下一代 NVIDIA GPU.
*   [H100 DGX SuperPod 参考](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf): 关于如何联网更大型 GPU 集群的枯燥但有用的读物. [这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576) 是关于 GB200 系统的类似文档.
*   [关于 NVLink Switch 的 Hot Chips 演讲](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf): 关于 NVLink 和 NCCL 集合操作的有趣读物, 特别是包括网络内归约.
*   [DeepSeek-V3 技术报告](https://arxiv.org/pdf/2412.19437): 一个大型半开放 LLM 训练报告的好例子, 描述了他们如何选择分片设置.
*   [如何优化 CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM): 一篇很棒的博客, 描述了如何使用 CUDA 核心实现高效的矩阵乘法, 并着眼于 GPU 上的缓存一致性.
*   [HuggingFace 超大规模实践手册:](https://huggingface.co/spaces/nanotron/ultrascale-playbook) GPU 上 LLM 并行的指南, 本章部分灵感来源于此.
*   [从第一性原理让深度学习飞速发展:](https://horace.io/brrr_intro.html): 一篇更侧重于 GPU 和 PyTorch 的教程, 讲解了 LLM 的 roofline 模型和性能工程.

## 附录 A: GB200 会带来哪些变化?

Blackwell 引入了一系列重大的网络变化, 包括 NVLink 5, 其总 NVLink 带宽是原来的两倍 (900GB/s). B200 仍然有 8-GPU 节点, 就像 H100 一样, 但 GB200 系统 (将 B200 GPU 与 Grace CPU 结合) 引入了更大的 NVLink 域 (NVL72 中有 72 个 GPU, 理论上最多可达 576 个). 这个更大的 NVLink 域也有效地增加了节点出口带宽, 从而降低了节点之上的集合操作成本.

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>图:</b> 展示 GB200 NVL72 单元如何构建的示意图, 包含 18 个交换机和 72 个 GPU." %}

在节点内, 这种增加的带宽 (从 450GB/s 到 900GB/s) 并没有太大区别, 因为我们也使每个 GPU 的总 FLOPs/s 翻了一番. 我们的 roofline 模型基本保持不变, 尽管因为 NVLink 有更好的带宽, 专家并行变得更容易了.

在节点之外, 情况变化更大. 这是来自[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的 SuperPod 示意图.

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图:</b> 展示一个包含 576 个 GPU 的 GB200 DGX SuperPod 的示意图." %}

如你所见, 每个节点的出口带宽增加到 `4 * 18 * 400 / 8 = 3.6TB/s`, 从 H100 的 400GB/s 增加. 这将有效的跨节点 roofline 提高了约 4 倍, 因为我们的 FLOPs/芯片也翻了一番. 现在我们可能开始担心我们是否在节点级别而不是在横向扩展级别受到瓶颈限制.

**Grace Hopper:** NVIDIA 还销售 GH200 和 GB200 系统, 将一定数量的 GPU 与 Grace CPU 配对. 例如, 一个 GH200 有 1 个 H200 和 1 个 Grace CPU, 而一个 GB200 系统有 2 个 B200 和 1 个 Grace CPU. 这个系统的一个优点是 CPU 使用全带宽 NVLink 连接 (称为 NVLink C2C) 连接到 GPU, 所以你有非常高的 CPU 到 GPU 带宽, 这对于将参数卸载到主机 RAM 很有用. 换句话说, 对于任何给定的 GPU, 到达主机内存的带宽与到达另一个 GPU 的 HBM 的带宽相同.

## 附录 B: 更多网络细节

这是一个 NVLink 4 交换机的示意图. 总共有 64 个 NVLink4 端口 (每个使用 2 个物理通道), 以及一个处理通道间交换的大型交叉开关. 相比之下, TPU 使用可以动态重新配置的带反射镜的光学交换机.

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>图:</b> 单个 NVLink4 交换机的更底层视图." %}

在每个级别, 我们都可能受到可用链路带宽或总交换机带宽的瓶颈限制.

*   **节点级别:** 在节点级别, 我们有 4 * 1.6TB/s = 6.4TB/s 的 NVSwitch 带宽, 但我们的 8 个 GPU 每个只能向交换机出口 450GB/s, 这意味着我们实际上在节点内的峰值带宽是 450e9 * 8 = 3.6TB/s (全双工).
*   **SU/叶级别:** 在 SU 级别, 我们有 8 个交换机以全对全方式连接 32 个节点, 使用 1x400 Gbps Infiniband. 这给了我们 8 * 32 * 400 / 8 = 12.8TB/s 的节点出口带宽, 我们在交换机级别有 8 * 1.6TB/s = 12.8TB/s, 所以两者精确吻合.
*   **主干级别:** 在主干级别, 我们有 16 个交换机连接 32 个叶交换机, 使用 2x400 Gbps 链路, 所以我们有 32 * 16 * 400 * 2 / 8 = 51.2TB/s 的出口带宽. 16 个交换机给了我们 16 * 1.6TB/s = 25.6TB/s 的带宽, 所以这是这个级别的瓶颈.

每个 GPU, 这给了我们节点级别 450GB/s 的 GPU 间带宽, SU 级别 50GB/s, 主干级别 25 GB/s.

**GPU 实证 AR 带宽:**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图:</b> 8xH100 集群上的 AllReduce 带宽 (节点内, 禁用 SHARP)." %}

TPU v5p 带宽 (1 轴):

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>图:</b> TPU v5p 4x4x4 集群上的 AllReduce 带宽 (沿一个轴)." %}

这里还有 AllGather 带宽:

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>图:</b> 8xH100 集群上的 AllGather 带宽 (节点内)." %}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>图:</b> TPU v5e 8x16 集群上的 AllGather 带宽 (沿一个轴)." %}

**更多关于 AllToAll 成本:**

在这里我们可以比较近似值 $\min(K / Z) * (Z - 1) / Z$ 与真实值 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$. 除了 $Z$ 值较小的情况外, 它们是相似的.

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>图:</b> 随着分片数量增加, 不规则 AllToAll 的近似成本与真实成本的比较." %}