---
layout: distill
title: "如何理解GPU"
description: "我们在Google热爱TPU，但GPU也很棒。本章深入探讨NVIDIA GPU的世界——每个芯片如何工作、它们如何联网，以及这对LLM意味着什么，特别是与TPU相比。本节建立在<a href='https://jax-ml.github.io/scaling-book/tpus/'>第2章</a>和<a href='https://jax-ml.github.io/scaling-book/training'>第5章</a>的基础上，因此鼓励你先阅读它们。"
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "第11部分：结论"

next_section_url:
next_section_name: "结束"

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
  - name: 什么是GPU？
  - subsections:
    - name: 内存
    - name: "GPU规格总结"
    - name: 芯片层面的GPU与TPU比较
    - name: "测验1：GPU硬件"
  - name: 网络
  - subsections:
    - name: 节点层面
    - name: "测验2：GPU节点"
    - name: 超越节点层面
    - name: "测验3：超越节点层面"
  - name: GPU上的集合操作如何工作？
  - subsections:
    - name: 节点内集合操作
    - name: 跨节点集合操作
    - name: "测验4：集合操作"
  - name: "GPU上LLM扩展的性能上限"
  - subsections:
    - name: "数据并行"
    - name: "张量并行"
    - name: "专家并行"
    - name: "流水线并行"
    - name: "示例"
    - name: "GPU上LLM扩展的TLDR"
    - name: "测验5：LLM性能上限"
  - name: "致谢和进一步阅读"
  - name: "附录"
  - subsections:
    - name: "附录A：GB200带来的变化"
    - name: "附录B：更多网络细节"

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

## 什么是GPU？

现代机器学习GPU（例如H100、B200）基本上是一组专门用于矩阵乘法的计算核心（称为**流式多处理器**或**SMs**），连接到一块快速内存（称为**HBM**）。以下是一个示意图：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" caption="<b>Figure:</b> a diagram showing the abstract layout of an H100 or B200 GPU. An H100 has 132 SMs while a B200 has 148. We use the term 'Warp Scheduler' somewhat broadly to describe a set of 32 CUDA SIMD cores <i>and</i> the scheduler that dispatches work to them. Note how much this looks like a TPU!" %}

每个SM（类似于TPU的张量核心）都有一个专用的矩阵乘法核心（不幸的是也称为**张量核心**<d-footnote>GPU张量核心是SM的矩阵乘法子单元，而TPU张量核心是包含MXU、VPU和其他组件的伞形单元。</d-footnote>）、一个向量算术单元（称为**Warp调度器**<d-footnote>NVIDIA没有很好的名称来描述这个单元，所以我们只能从几个不太好的选项中选择最好的。Warp调度器主要是将工作分派给一组CUDA核心的单元，但我们在这里用它来描述控制单元及其控制的核心集合。</d-footnote>）和一个快速片上缓存（称为**SMEM**）。与最多只有2个独立"张量核心"的TPU不同，现代GPU有超过100个SM（H100上有132个）。每个SM的计算能力远不如TPU张量核心，但整个系统更加灵活。每个SM或多或少是完全独立的，因此GPU可以同时执行数百个独立任务。<d-footnote>尽管SM是独立的，但它们通常需要协调以达到峰值性能，因为它们都共享一个容量有限的L2缓存。</d-footnote>

让我们更详细地看一下H100 SM：

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" caption="<b>Figure:</b> a diagram of an H100 SM (<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>source</a>) showing the 4 <i>subpartitions</i>, each containing a Tensor Core, Warp Scheduler, Register File, and sets of CUDA Cores of different precisions. The 'L1 Data Cache' near the bottom is the 256kB SMEM unit. A B200 looks similar, but adds a substantial amount of Tensor Memory (TMEM) for feeding the bulky Tensor Cores." %}

每个SM被分成4个相同的象限，NVIDIA称之为**SM子分区**，每个子分区包含一个张量核心、16k 32位寄存器和一个称为Warp调度器的SIMD/SIMT向量算术单元，其通道（ALU）NVIDIA称为**CUDA核心**。每个分区的核心组件可以说是张量核心，它执行矩阵乘法并构成其绝大部分的FLOPs/s，但这并不是唯一值得注意的组件。

* **CUDA核心：** 每个子分区包含一组称为CUDA核心的ALU，执行SIMD/SIMT向量算术运算。每个ALU通常每个周期可以执行1个算术操作，例如f32.add。<d-footnote>较新的GPU支持FMA（融合乘加）指令，技术上每个周期执行两次FLOPs，NVIDIA利用这一事实无情地将其报告的规格翻倍。</d-footnote> 每个子分区包含32个fp32核心（以及较少数量的int32和fp64核心），它们每个周期都执行相同的指令。与TPU的VPU类似，CUDA核心负责ReLU、逐点向量操作和归约（求和）。<d-footnote>历史上，在引入张量核心之前，CUDA核心是GPU的主要组件，用于渲染，包括光线-三角形相交和着色。在今天的游戏GPU上，它们仍然承担大部分渲染工作，而张量核心用于上采样（DLSS），这使得GPU可以以较低分辨率渲染（更少的像素=更少的工作）并使用ML进行上采样。</d-footnote>

* **张量核心（TC）：** 每个子分区都有自己的张量核心，这是一个专用的矩阵乘法单元，类似于TPU的MXU。张量核心代表了GPU绝大部分的FLOPs/s（例如，在H100上，我们有990 bf16 TC TFLOP/s，而CUDA核心只有66 TFLOPs/s）。
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) with 132 SM running at 1.76GHz means each H100 TC can do `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs/cycle, roughly an 8x8x8 matmul.<d-footnote>NVIDIA doesn’t share many TC hardware details, so this is more a guess than definite fact – certainly, it doesn’t speak to how the TC is implemented. We know that a V100 can perform 256 FLOPs/TC/cycle. An A100 can do 512, H100 can do 1024, and while the B200 details aren’t published, it seems likely it’s about 2048 FLOPs/TC/cycle, since `2250e12 / (148 * 4 * 1.86e9)` is about 2048. Some more details are confirmed <a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>here</a>.</d-footnote>
  * Like TPUs, GPUs can do lower precision matmuls at higher throughput (e.g. H100 has 2x fp8 FLOPs/s vs. fp16). Low-precision training or serving can be significantly faster.
  * Each GPU generation since Volta has increased the TC size over the previous generation ([good article on this](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)). With B200 the TC has gotten so large it can no longer fit its inputs in SMEM, so B200s introduce a new memory space called TMEM.<d-footnote>In Ampere, the Tensor Core could be fed from a single warp, while in Hopper it requires a full SM (warpgroup) and in Blackwell it’s fed from 2 SMs. The matmuls have also become so large in Blackwell that the arguments (specifically, the accumulator) no longer fit into register memory/SMEM, so Blackwell adds TMEM to account for this.</d-footnote>

**CUDA核心比TPU的VPU更灵活：** GPU CUDA核心（自V100以来）使用称为SIMT（*单指令多线程*）的编程模型，与TPU的SIMD（*单指令多数据*）模型相比。与TPU VPU中的ALU类似，子分区内的CUDA核心必须在每个周期执行相同的操作（例如，如果一个核心正在添加两个浮点数，那么子分区中的其他每个CUDA核心也必须这样做）。然而，与VPU不同，每个CUDA核心（或CUDA编程模型中的"线程"）都有自己的指令指针，并且可以独立*编程*。当同一warp中的两个线程被指示执行不同的操作时，你实际上会执行*两个*操作，屏蔽掉不需要执行分歧操作的核心。

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>Figure:</b> an example of warp divergence within a set of threads (<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>source</a>). White spaces indicate stalls of at least some fraction of the physical CUDA cores" %}

这在线程级别实现了灵活的编程，但代价是如果warp分歧太频繁，性能会无声地下降。线程在可以访问的内存方面也更加灵活；虽然VPU只能操作连续的内存块，但CUDA核心可以访问共享寄存器中的单个浮点数，并维护每个线程的状态。

**CUDA核心调度也更灵活：** SM运行起来有点像多线程CPU，因为它们可以并发"调度"许多程序（**warps**）（每个SM最多64个），但每个*Warp调度器*在每个时钟周期只执行一个程序。<d-footnote>在给定SM上调度的warps称为"resident"（驻留）。</d-footnote> Warp调度器在活跃的warps之间自动切换，以隐藏I/O操作（如内存加载）。相比之下，TPU通常是单线程的。

### Memory

除了计算单元，GPU还有一系列层次化的内存，最大的是HBM（GPU主内存），然后是一系列较小的缓存（L2、L1/SMEM、TMEM、寄存器内存）。

* **寄存器：** 每个子分区都有自己的寄存器文件，在H100/B200上包含16,384个32位字（`4 * 16384 * 4 = 256kiB` 每SM），可由CUDA核心访问。
  * 每个CUDA核心一次只能访问最多256个寄存器，因此虽然我们可以在每个SM上调度最多64个"驻留warps"，但如果每个线程使用256个寄存器，一次只能容纳8个（`256 * 1024 / (4 * 32 * 256)`）。

* **SMEM（L1缓存）：** 每个SM都有自己的256kB片上缓存，称为SMEM，可以由程序员控制为"共享内存"，也可以由硬件用作片上缓存。SMEM用于存储激活和TC矩阵乘法的输入。

* **L2缓存：** 所有SM共享<d-footnote>技术上，L2缓存分为两部分，因此一半的SM可以访问H100上的25MB。两部分之间有连接，但带宽较低。</d-footnote>一个相对较大的~50MB L2缓存，用于减少主内存访问。
  * 这与TPU的VMEM大小相似，但**慢得多**且不由程序员控制。这导致了一些"远距离幽灵作用"，程序员需要修改内存访问模式以确保L2缓存得到良好利用。<d-footnote>L2缓存在所有SM之间共享的事实实际上迫使程序员以相当协调的方式运行SM，尽管原则上它们是独立的单元。</d-footnote>
  * NVIDIA不发布其芯片的L2带宽，但据[测量](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)约为5.5TB/s。这大约是HBM带宽的1.6倍，但它是全双工的，因此有效的双向带宽接近3倍。相比之下，TPU的VMEM大2倍*并且*具有更多的带宽（约40TB/s）。

* **HBM：** GPU主内存，用于存储模型权重、梯度、激活等。
  * HBM大小从Volta的32GB大幅增加到Blackwell（B200）的192GB。
  * 从HBM到CUDA张量核心的带宽称为HBM带宽或内存带宽，在H100上约为3.35TB/s，在B200上约为9TB/s。

### GPU规格总结

以下是近期GPU型号的规格总结。对于给定的GPU型号，SM数量、时钟速度和FLOPs在不同变体之间有所不同。以下是内存容量数字：

|  GPU  | 代次 |   时钟速度   | SM/芯片 | SMEM容量/SM | L2容量/芯片 | HBM容量/芯片 |
| :---: | :--: | :----------: | :-----: | :---------: | :---------: | :----------: |
| V100  | Volta | 1.25GHz/1.38GHz |   80    |    96kB     |    6MB      |    32GB      |
| A100  | Ampere | 1.10GHz/1.41GHz |  108    |    192kB    |    40MB     |    80GB      |
| H100  | Hopper | 1.59GHz/1.98GHz |  132    |    256kB    |    50MB     |    80GB      |
| H200  | Hopper | 1.59GHz/1.98GHz |  132    |    256kB    |    50MB     |    141GB     |
| B200  | Blackwell |       ?       |  148    |    256kB    |   126MB     |    192GB     |

所有代次的每个SM都有256kB寄存器内存。Blackwell还为每个SM增加了256kB TMEM。以下是每个芯片的FLOPs和带宽数字：

|  GPU  | 代次 | HBM带宽/芯片 | FLOPs/s/芯片 (bf16/fp16) | FLOPs/s/芯片 (fp8/int8) | FLOPs/s/芯片 (fp4) |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100  |   Volta    |   9.0e11    |            —             |            —            |         —          |
| A100  |   Ampere   |   2.0e12    |          3.1e14          |         6.2e14          |         —          |
| H100  |   Hopper   |   3.4e12    |          9.9e14          |         2.0e15          |         —          |
| H200  |   Hopper   |   4.8e12    |          9.9e14          |         2.0e15          |         —          |
| B200  | Blackwell  |   8.0e12    |          2.3e15          |         4.5e15          |       9.0e15       |

We exclude B100 since it wasn't mass-produced.<d-footnote>While NVIDIA made a B100 generation, they were only briefly sold and produced, allegedly due to design flaws that prevented them from running close to their claimed specifications. They struggled to achieve peak FLOPs without throttling due to heat and power concerns.</d-footnote> Some specs depend slightly on the precise version of the GPU, since NVIDIA GPUs aren’t as standard as TPUs.

Here’s a helpful cheat sheet comparing GPU and TPU components:

|              GPU              |     TPU     |              What is it?              |
| :---------------------------: | :---------: | :-----------------------------------: |
| Streaming Multiprocessor (SM) | Tensor Core | Core "cell" that contains other units |
|        Warp Scheduler         |     VPU     |      SIMD vector arithmetic unit      |
|           CUDA Core           |   VPU ALU   |               SIMD ALU                |
|        SMEM (L1 Cache)        |    VMEM     |       Fast on-chip cache memory       |
|          Tensor Core          |     MXU     |      Matrix multiplication unit       |
|        HBM (aka GMEM)         |     HBM     |  High bandwidth high capacity memory  |

### 芯片层面的GPU与TPU比较

GPUs started out rendering video games, but since deep learning took off in the 2010s, they've started acting more and more like dedicated matrix multiplication machines – in other words, more like TPUs.<d-footnote>Before the deep learning boom, GPUs ("Graphics Processing Units") did, well, graphics – mostly for video games. Video games represent objects with millions of little triangles, and the game renders (or "rasterizes") these triangles into a 2D image that gets displayed on a screen 30-60 times a second (this frequency is called the framerate). Rasterization involves projecting these triangles into the coordinate frame of the camera and calculating which triangles overlap which pixels, billions of times a second. As you can imagine, this is very expensive, and it’s just the beginning. You then have to color each pixel by combining the colors of possibly several semi-opaque triangles that intersect the ray. GPUs were designed to do these operations extremely fast, with an eye towards versatility; you need to run many different GPU workloads (called "shaders") at the same time, with no single operation dominating. As a result, consumer graphics-focused GPUs can do matrix multiplication, but it’s not their primary function.</d-footnote> To an extent, this history explains why modern GPUs look the way they do. They weren't designed purely for LLMs or ML models but as general-purpose accelerators, and the hardware aims for level of "generality" that can be both a blessing and a curse. GPUs much more often "just work" when applied to new tasks and lean far less on a good compiler than TPUs do. But this also makes them much harder to reason about or get roofline performance out of, since so many compiler features can cause bottlenecks.

**GPUs are more modular.** TPUs have 1-2 big Tensor Cores, while GPUs have hundreds of small SMs. Likewise, each Tensor Core has 4 big VPU with 1024 ALUs each, while GPUs have an H100 has 132 * 4 = 528 small independent SIMD units. Here is a 1:1 comparison of GPUs to TPU that highlights this point:

|              GPU              |           TPU            | H100 # | TPU v5p # |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM (streaming multiprocessor) |       Tensor Core        |  132   |     2     |
|        Warp Scheduler         |           VPU            |  528   |     8     |
|        SMEM (L1 cache)        |           VMEM           |  32MB  |   128MB   |
|           Registers           | Vector Registers (VRegs) |  32MB  |   256kB   |
|          Tensor Core          |           MXU            |  528   |     8     |

This difference in modularity on the one hand makes TPUs much cheaper to build and simpler to understand, but it also puts more burden on the compiler to do the right thing. Because TPUs have a single thread of control and only support vectorized VPU-wide instructions, the compiler needs to manually pipeline all memory loads and MXU/VPU work to avoid stalls. A GPU programmer can just launch dozens of different kernels, each running on a totally independent SM. On the other hand, those kernels might get horrible performance because they are thrashing the L2 cache or failing to coalesce memory loads; because the hardware controls so much of the runtime, it becomes hard to reason about what’s going on behind the scenes. As a result, TPUs can often get closer to peak roofline performance with less work.

**从历史上看，单个GPU比同类TPU更强大（也更昂贵）：** 单个H200的FLOPs/s接近TPU v5p的2倍，HBM是TPU的1.5倍。同时，Google Cloud上的标价约为每小时\\$10美元的H200，而TPU v5p为每小时\\$4美元。TPU通常比GPU更依赖网络连接多个芯片。

**TPUs have a lot more fast cache memory.** TPUs also have a lot more VMEM than GPUs have SMEM (+TMEM), and this memory can be used for storing weights and activations in a way that lets them be loaded and used extremely fast. This can make them faster for LLM inference if you can consistently store or prefetch model weights into VMEM.

### Quiz 1: GPU hardware

Here are some problems to work through that test some of the content above. Answers are provided, but it’s probably a good idea to try to answer the questions before looking, pen and paper in hand.

**Question 1 [CUDA cores]:** How many fp32 CUDA cores (ALUs) does an H100 have? B200? How does this compare to the number of independent ALUs in a TPU v5p?

{% details Click here for the answer. %}

**Answer:** An H100 has 132 SMs with 4 subpartitions each containing 32 fp32 CUDA cores, so we `132 * 4 * 32 = 16896` CUDA cores. A B200 has has `148` SMs, so a total of `18944`. A TPU v5p has 2 TensorCores (usually connected via Megacore), each with a VPU with (8, 128) lanes and 4 independent ALUs per lane, so `2 * 4 * 8 * 128 = 8192` ALUs. This is roughly half the number of vector lanes of an H100, running at roughly the same frequency.

{% enddetails %}

**Question 2 [Vector FLOPs calculation]**: A single H100 has 132 SMs and runs at a clock speed of 1.59GHz (up to 1.98GHz boost). Assume it can do one vector op per cycle per ALU. How many vector fp32 FLOPs can be done per second? With boost? How does this compare to matmul FLOPs?

{% details Click here for the answer. %}

**Answer:** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`. With boost its 33.5 TFLOPs/s. This is half what’s reported in the [spec sheet](https://www.nvidia.com/en-us/data-center/h100/) because technically we can do an FMA (fused-multiply-add) in one cycle which counts as two FLOPs, but this isn't useful in most cases. We can do 990 bfloat16 matmul TFLOPs/s, so ignoring FMAs, Tensor Cores do around 30x more FLOPs/s.

{% enddetails %}

**Question 3 [GPU matmul intensity]:** What is the peak fp16 matmul intensity on an H100? A B200? What about fp8? *By intensity we mean the ratio of matmul FLOPs/s to memory bandwidth.*

{% details Click here for the answer. %}

**Answer:** For an H100, we have a peak 990e12 fp16 FLOPs and 3.35e12 bytes / s of bandwidth. So the critical intensity is `990e12 / 3.35e12 = 295`, fairly similar to the 240 in a TPU. For B200 its `2250e12 / 8e12 = 281`, very similar. This means, similar to TPUs, that we need a batch size of around 280 to be compute-bound in a matmul.

For both H100 and B200 we have exactly 2x fp8 FLOPs, so the peak intensity also doubles to 590 and 562 respectively, although in some sense it stays constant if we take into account the fact that our weights will likely be loaded in fp8 as well.

{% enddetails %}

**Question 4 [Matmul runtime]:** Using the answer to Question 3, how long would you expect an `fp16[64, 4096] * fp16[4096, 8192]` matmul to take on a single B200? How about `fp16[512, 4096] * fp16[4096, 8192]`?

{% details Click here for the answer. %}

From the above, we know we'll be communication-bound below a batch size of 281 tokens. Thus the first is purely bandwidth bound. We read or write $2BD + 2DF + 2BF$ bytes (`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`) with `8e12` bytes/s of bandwidth, so it will take about `69e6 / 8e12 = 8.6us`. In practice we likely get a fraction of the total bandwidth, so it may take closer to 10-12us. When we increase the batch size, we're fully compute-bound, so we expect `T=2*512*4096*8192/2.3e15=15us`. We again only expect a fraction of the total FLOPs, so we may see closer to 20us.

{% enddetails %}

**Question 5 [L1 cache capacity]:** What is the total L1/SMEM capacity for an H100? What about register memory? How does this compare to TPU VMEM capacity?

{% details Click here for the answer. %}

**Answer:** We have 256kB SMEM and 256kB of register memory per SM, so about 33MB (`132 * 256kB`) of each. Together, this gives us a total of about 66MB. This is about half the 120MB of a modern TPU’s VMEM, although a TPU only has 256kB of register memory total! TPU VMEM latency is lower than SMEM latency, which is one reason why register memory on TPUs is not that crucial (spills and fills to VMEM are cheap).

{% enddetails %}

**Question 6 [Calculating B200 clock frequency]:** NVIDIA reports [here](https://resources.nvidia.com/en-us-blackwell-architecture) that a B200 can perform 80TFLOPs/s of vector fp32 compute. Given that each CUDA core can perform 2 FLOPs/cycle in a FMA (fused multiply add) op, estimate the peak clock cycle.

{% details Click here for the answer. %}

**Answer:** We know we have 148 * 4 * 32 = 18944 CUDA cores, so we can do `18944 * 2 = 37888 FLOPs / cycle`. Therefore `80e12 / 37888 = 2.1GHz`, a high but reasonable peak clock speed. B200s are generally liquid cooled, so the higher clock cycle is more reasonable.

{% enddetails %}

**Question 7 [Estimating H100 add runtime]:** Using the figures above, calculate how long it ought to take to add two `fp32[N]` vectors together on a single H100. Calculate both $T_\text{math}$ and $T_\text{comms}$. What is the arithmetic intensity of this operation? If you can get access, try running this operation in PyTorch or JAX as well for `N = 1024` and `N=1024 * 1024 * 1024`. How does this compare?

{% details Click here for the answer. %}

**Answer:** Firstly, adding two `fp32[N]` vectors performs N FLOPs and requires `4 * N * 2` bytes to be loaded and 4 * N bytes to be written back, for a total of `3 * 4 * N = 12N`. Computing their ratio, we have `total FLOPs / total bytes = N / 12N = 1 / 12`, which is pretty abysmal.

As we calculated above, we can do roughly 33.5 TFLOPs/s boost, ignoring FMA. This is only if all CUDA cores are used. For `N = 1024`, we can only use *at most* 1024 CUDA cores or 8 SMs, which will take longer (roughly 16x longer assuming we’re compute-bound). We also have a memory bandwidth of 3.35e12 bytes/s. Thus our peak hardware intensity is `33.5e12 / 3.35e12 = 10`.<d-footnote>It’s notable that this intensity stays constant across recent GPU generations. For H100s it’s 33.5 / 3.5 and for B200 it’s 80 / 8. Why this is isn’t clear, but it’s an interesting observation.</d-footnote> So we’re going to be horribly comms bound. Thus our runtime is just

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

For `N = 65,536`, this is about 0.23us. In practice we see a runtime of about 1.5us in JAX, which is fine because we expect to be super latency bound here. For `N = 1024 * 1024 * 1024`, we have a roofline of about 3.84ms, and we see 4.1ms, which is good!

{% enddetails %}

## 网络

网络是GPU和TPU差异最大的领域之一。正如我们所见，TPU以2D或3D环形拓扑连接，每个TPU只连接到其邻居。这意味着在两个TPU之间发送消息必须经过每个中间TPU，迫使我们只能在网格上使用统一的通信模式。虽然这在某些方面不方便，但这也意味着每个TPU的链路数量是恒定的，我们可以扩展到任意大的TPU"pod"而不会损失带宽。

另一方面，GPU使用更传统的分层树状交换网络。一组8个GPU称为**节点**（GB200最多可达72个<d-footnote>"节点"这个词有双重含义：可以指NVLink域，即通过NVLink互连完全连接的GPU集合，也可以指连接到单个CPU主机的GPU集合。在B200之前，这两者通常是相同的，但在GB200 NVL72中，我们有一个包含72个GPU的NVLink域，但每个主机仍然只连接8个GPU。我们在本文中使用"节点"来指代NVLink域，但这是有争议的。</d-footnote>）通过称为NVLink的高带宽互连在1跳内相互连接，这些节点通过连接到每个GPU的网卡（NIC）使用较低带宽的InfiniBand（IB）或以太网网络连接到更大的单元（称为**SU**或可扩展单元）。这些又可以连接到通过更高级别交换机的任意大单元。

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>Figure:</b> a diagram showing a typical H100 network. A set of 8 GPUs is connected into a node or NVLink domain with NVSwitches (also called NVLink switches), and these nodes are connected to each other with a switched InfiniBand fabric. H100s have about 450GB/s of egress bandwidth each in the NVLink domain, and each node has 400GB/s of egress bandwidth into the IB network." %}

### 节点层面

GPU节点是一个小单元，通常包含8个GPU（GB200最多可达72个），通过全连接、全带宽、低延迟的NVLink互连连接。<d-footnote>NVLink被描述为类似于增强版的PCIe连接，具有低延迟和低协议开销，但不是为可扩展性/容错性设计的，而InfiniBand更像以太网，是为更大的有损网络设计的。</d-footnote>每个节点包含几个高带宽的NVSwitch，用于在所有本地GPU之间交换数据包。实际的节点级拓扑随时间发生了很大变化，包括每个节点的交换机数量，但对于H100，我们每个节点有4个NVSwitch，GPU以`5 + 4 + 4 + 5`的链路模式连接到它们，如下所示：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>Figure:</b> node aka NVLink domain diagrams from Pascall (P100) onward. Since Volta (V100), we have had all-to-all connectivity within a node using a set of switches. The H100 node has 4 NVSwitches connected to all 8 GPUs with 25GB/s links." %}

对于Hopper代（NVLink 4.0），每个NVLink链路具有25GB/s的全双工<d-footnote>全双工在这里意味着每个方向25GB/s，两个方向相互独立。您可以在链路上总共发送50GB/s，但每个方向最多25GB/s。</d-footnote>带宽（B200为50GB/s），为我们提供`18 * 25=450GB/s`的从每个GPU到网络的全双工带宽。巨大的NVSwitch最多有64个NVLink端口，这意味着一个带有4个交换机的8xH100节点可以处理高达`64 * 25e9 * 4=6.4TB/s`的带宽。以下是这些数字如何随着GPU代际变化的概览：

| NVLink Gen | NVSwitch Gen | GPU Generation | NVLink Bandwidth (GB/s, full-duplex) | NVLink Ports / GPU | Node GPU to GPU bandwidth (GB/s full-duplex) | Node size (NVLink domain) | NVSwitches per node |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
|  **3.0**   |   **2.0**    |     Ampere     |                  25                  |         12         |                     300                      |             8             |          6          |
|  **4.0**   |   **3.0**    |     Hopper     |                  25                  |         18         |                     450                      |             8             |          4          |
|  **5.0**   |   **4.0**    |   Blackwell    |                  50                  |         18         |                     900                      |           8/72            |        2/18         |

Blackwell（B200）具有8个GPU的节点。GB200NVL72支持更大的72个GPU的NVLink域。我们展示了8个和72个GPU系统的详细信息。

### 测验2：GPU节点

这里有一些关于网络的问答问题。我觉得做这些问题特别有用，因为它们让你了解实际的通信模式。

**问题1 [H100节点的总带宽]：** 在一个带有4个交换机的8xH100节点中，我们每个节点有多少总带宽？*提示：*考虑NVLink和NVSwitch带宽。

{% details 点击这里查看答案。 %}

**答案：** 我们有Gen4 4xNVSwitches，每个具有`64 * 25e9=1.6TB/s`的单向带宽。这将在交换机级别给我们`4 * 1.6e12=6.4e12`的带宽。但是，请注意每个GPU只能处理450GB/s的单向带宽，这意味着我们最多有`450e9 * 8 = 3.6TB/s`的带宽。由于这个值更小，峰值带宽是3.6TB/s。

{% enddetails %}

**问题2 [对分带宽]**：对分带宽定义为网络中任何均匀分区之间可用的最小带宽。换句话说，如果将网络分成两个相等的部分，有多少带宽在两个部分之间交叉？你能计算出8x H100节点的对分带宽吗？*提示：*对分带宽通常包括两个方向的流量。

{% details 点击这里查看答案。 %}

**答案：** 任何均匀分区将在每一半中有4个GPU，每个可以向另一半输出`4 * 450GB/s`。考虑两个方向的流量，这给我们`8 * 450GB/s`的字节穿过分区，或3.6TB/s的对分带宽。这是NVIDIA报告的，例如[这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。

{% enddetails %}

**问题3 [AllGather成本]**：给定一个B字节的数组，在8xH100节点上（吞吐量受限的）AllGather需要多长时间？为bf16[D<sub>X</sub>, F]做数学计算，其中`D=4096`，`F=65,536`。*在回答这个问题之前，值得阅读TPU集合通信[部分](https://jax-ml.github.io/scaling-book/sharding/)。在这里仔细思考，但我们接下来会更详细地讨论集合通信。*

{% details 点击这里查看答案。 %}

**答案：** 每个GPU可以输出450GB/s，每个GPU有$B / N$字节（其中`N=8`，节点大小）。我们可以想象每个节点一个接一个地将其字节发送到其他$N - 1$个节点，导致总共(N - 1)轮，每轮$T_\text{comms} = (B / (N * W_\text{unidirectional}))$，或$T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$。这大约是$B / (N * W_\text{uni})$或$B / \text{3.6e12}$，即对分带宽。

对于给定的数组，我们有`B=4096 * 65536 * 2=512MB`，所以总时间是`536e6 * (8 - 1) / 3.6e12 = 1.04ms`。这可能是延迟受限的，因此在实践中可能需要更长的时间（实践中大约需要1.5ms）。

{% enddetails %}

## 超越节点层面

超越节点层面，GPU网络的拓扑结构标准化程度较低。NVIDIA发布了一个[参考DGX SuperPod架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，使用InfiniBand连接比单个节点更大的GPU集合，但客户和数据中心提供商可以根据需要自由定制。<d-footnote>例如，Meta在与本描述显著不同的数据中心网络上训练了LLaMA-3，使用以太网、3层交换结构和顶层的一个过载交换机。</d-footnote>

这是一个参考1024 GPU H100系统的图表，其中底行中的每个框都是一个单独的8xH100节点，包含8个GPU、8个400Gbps CX7网卡（每个GPU一个）和4个NVSwitches。

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>Figure:</b> diagram of the reference 1024 H100 DGX SuperPod with 128 nodes (sometimes 127), each with 8 H100 GPUs, connected to an InfiniBand scale-out network. Sets of 32 nodes (256 GPUs) are called 'Scalable Units' or SUs. The leaf and spine IB switches provide enough bandwidth for full bisection bandwidth between nodes." %}

**可扩展单元：** 每组32个节点称为"可扩展单元"（或SU），位于一组8个叶层InfiniBand交换机下。这个SU有256个GPU，每个节点有4个NVSwitches和8个Infiniband叶层交换机。所有显示的布线都是InfiniBand NDR（50GB/s全双工），使用64端口NDR IB交换机（每个端口也是50GB/s）。*注意IB交换机的带宽是NVSwitches的2倍（64个端口，400 Gbps链路）。*

**SuperPod：** 整体SuperPod然后用16个顶层"spine"IB交换机连接4个这些SU，给我们1024个GPU，512个节点级NVSwitches，32个叶层IB交换机，和16个spine IB交换机，总共512 + 32 + 16 = 560个交换机。叶层交换机以32个节点为一组连接到节点，所以每组256个GPU有8个叶层交换机。所有叶层交换机都连接到所有spine交换机。

**我们有多少带宽？** InfiniBand网络（称为"扩展网络"）的整体拓扑是一个**胖树**，电缆和交换机保证节点级别以上的全对分带宽（这里是400GB/s）。这意味着如果我们将节点分成两半，每个节点可以同时向另一分区中的节点输出400GB/s。更重要的是，这意味着我们在扩展网络中应该有大致恒定的AllReduce带宽！虽然可能不会以这种方式实现，但您可以想象在扩展网络中对任意多个节点进行环形归约，因为您可以构建一个包含每个节点的环。

| 级别 | GPU数量 | 每单元交换机数 | 交换机类型 | 每单元带宽（TB/s，全双工） | GPU到GPU带宽（GB/s，全双工） | 胖树带宽（GB/s，全双工） |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点  |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| 叶层  |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| 脊层  |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

相比之下，TPU v5p每个链路约有90GB/s的输出带宽，或者沿3D环形拓扑的所有轴有540GB/s的输出带宽。这不是点对点的，因此只能用于受限的、统一的通信模式，但它仍然给我们提供了更高的TPU到TPU带宽，可以扩展到任意大的拓扑（至少达到8960个TPU）。

GPU交换结构理论上可以通过添加额外的交换机或间接层扩展到任意大小，代价是额外的延迟和昂贵的网络交换机。

<p markdown=1 class="takeaway">**要点**：在H100节点内，我们从每个GPU获得完整的450GB/s胖树带宽，而在节点之外，这会下降到400GB/s的节点到节点带宽。这对于通信原语将是至关重要的。</p>

**GB200 NVL72：** NVIDIA最近开始生产新的GB200 NVL72 GPU集群，在单个NVLink域中组合72个GPU，具有完整的900GB/s GPU到GPU带宽。这些域然后可以链接到更大的SuperPod中，具有成比例更高（9倍）的IB胖树带宽。这是该拓扑的图表：

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>Figure:</b> a diagram showing a GB200 DGX SuperPod of 576 GPUs. Each rack at the bottom layer contains 72 GB200 GPUs." %}

计算单个节点的输出带宽（上面的橙色线），我们有`4 * 18 * 400 / 8 = 3.6TB/s`到叶层的带宽，这比H100多9倍（正如节点包含多9倍的GPU）。这意味着关键的节点输出带宽要高得多，我们的跨节点集合带宽实际上可以比节点内更低。
参见[附录A](#appendix-a-how-does-this-change-with-gb200)以获得更多讨论。

|  节点类型  | 每节点GPU数 | GPU输出带宽 | 节点输出带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

<p markdown=1 class="takeaway">**要点**：GB200 NVL72 SuperPods显著增加了节点大小和给定节点的输出带宽，这大大改变了我们的性能上限。</p>

### 测验3：超越节点层面

**问题1 [胖树拓扑]**：使用上面的DGX H100图表，计算整个1024 GPU pod在节点级别的对分带宽。证明每个链路的带宽被选择为确保全对分带宽。*提示：确保计算链路带宽和交换机带宽。*

{% details 点击这里查看答案。 %}

**答案：** 让我们逐个组件来分析：

* 首先，每个节点有8条400Gbps NDR IB电缆连接到叶层交换机，给每个节点`8 * 400 / 8 = 400 GB/s`到叶层的带宽。我们有8个叶层交换机，每个有3.2TB/s（64个400 GBps链路），但我们只能使用64个端口中的32个来从SU接入，所以32个节点是`32 * 400 / 8 = 12.8TB/s`，再次正好是400GB/s。
* 然后在脊层，我们有`8 * 16 * 2`条400Gbps NDR IB电缆连接每个SU到脊层，给每个SU`8 * 16 * 2 * 400 / 8 = 12.8 TB/s`到叶层的带宽。再次，这是每个节点400GB/s。我们有16个脊层交换机，每个有3.2TB/s，给我们`16 * 3.2 = 51.2 TB/s`，128个节点再次是400GB/s。

因此，如果我们以任何方式对分节点，我们将在它们之间有每个GPU 400GB/s。每个组件都有确保胖树所需的带宽。

{% enddetails %}

**问题2 [扩展到更大的DGX pod]**：假设我们想要在2048个GPU而不是1024个上进行训练。修改上述DGX拓扑以处理这个问题的最简单/最佳方法是什么？4096呢？*提示：没有单一的正确答案，但尽量降低成本。记住链路容量。[这个](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)文档可能有用。*

{% details 点击这里查看答案。 %}

**答案：** 一个选择是保持SU结构完整（32个节点在8个交换机下），只是添加更多的它们与更多的顶级交换机。我们需要2倍更多的脊层交换机，所以我们会有8个SU和32个脊层交换机，给我们足够的带宽。

这样做的一个问题是，我们每个叶层交换机只有64个端口，我们在上面的图表中已经使用了所有端口。但是，改为每个脊层使用1条400 Gbps NDR电缆而不是2条，这给出相同的总带宽但为我们节省一些端口。

对于4096个GPU，我们实际上端口用完了，所以我们需要添加另一个间接层，也就是说，层次结构中的另一个级别。NVIDIA称这些为"核心交换机"，并用128个脊层交换机和64个核心交换机构建4096 GPU集群。你可以做数学计算来证明这给出足够的带宽。

{% enddetails %}

## GPU上的集合通信如何工作？

GPU可以执行与TPU相同的所有集合通信：ReduceScatter、AllGather、AllReduce和AllToAll。与TPU不同，这些操作的工作方式取决于它们是在节点级别（通过NVLink）还是在更高级别（通过InfiniBand）执行。这些集合通信由NVIDIA在[NVSHMEM](https://developer.nvidia.com/nvshmem)和[NCCL](https://developer.nvidia.com/nccl)（发音为"nickel"）库中实现。NCCL在[这里](https://github.com/NVIDIA/nccl)开源。虽然NCCL根据延迟需求/拓扑结构使用各种实现（[详情](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），但从现在开始，我们将讨论在交换树结构上的理论最优模型。

### 节点内集合通信

**AllGather或ReduceScatter：** 对于节点级别的AllGather或ReduceScatter，您可以像TPU一样围绕环执行它们，在每个跳使用完整的GPU到GPU带宽。任意排序GPU并围绕环发送数组的一部分，使用完整的GPU到GPU带宽。<d-footnote>您也可以认为每个GPU将其大小为$\text{bytes} / N$的数据块发送到其他$N - 1$个GPU，总共通信$(N - 1) * N * bytes / N$字节，这给我们</d-footnote> 每个跳的成本是$T_\text{hop} = \text{bytes} / (N * \text{GPU egress bandwidth})$，所以总体成本是

$$T_\text{AG或RS通信} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

您会注意到这与TPU上的完全相同。对于AllReduce，您可以像往常一样组合RS + AG，成本是两倍。

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>Figure:</b> bandwidth-optimal 1D ring AllGather algorithm. For B bytes, this sends V / X bytes over the top-level switches X - 1 times." %}

如果您关心延迟（例如，如果您的数组非常小），您可以进行树形归约，其中您在2对、然后4对、然后8对中进行AllReduce，总共$\log(N)$跳而不是$N - 1$跳，尽管总成本仍然相同。

<p markdown=1 class="takeaway">**要点：** 在单个节点内AllGather或ReduceScatter一个B字节数组的成本约为$T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approxeq B / W_\text{GPU egress}$。这在H100上理论上约为$B  / \text{450e9}$，在B200上为$B / \text{900e9}$。AllReduce的成本是这个的2倍，除非启用了网络内归约。</p>

<b markdown=1 style="color: #57cf57;">小测验1 [AllGather时间]：</b> 使用一个具有450 GB/s全双工带宽的8xH100节点，AllGather(bf16[B<sub>X</sub>, F])需要多长时间？设$B=1024$，$F=16,384$。

{% details 点击这里查看答案。 %}

**答案：** 我们总共有$2 \cdot B \cdot F$字节，具有450e9单向带宽。这将大约需要$T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，或者更精确地说$(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。使用提供的值，这给我们大约$(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，或者更精确地说，$\text{65us}$。

{% enddetails %}

**AllToAlls：** 节点内的GPU具有全连接性，这使得AllToAlls变得相当容易。每个GPU直接发送到目标节点。在节点内，对于B字节，每个GPU有$B / N$字节并向$N - 1$个目标节点发送$(B / N^2)$字节，总共

$$T_\text{AllToAll通信} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

将其与TPU比较，TPU的成本是$B / (4W)$。因此，在单个节点内，我们在运行时获得2倍的理论加速（$B / 4W$ vs. $B / 8W$）。

对于混合专家（MoE）模型，我们经常想要执行*稀疏或不规则AllToAll*，我们保证输出维度上最多$k$个$N$分片是非零的，也就是说$T_\text{AllToAll}_X \rightarrow K[B, N]$，其中每个轴上最多$k$个$N$条目是非零的。这个成本被$k/N$减少，总共约为$\min(k/N, 1) \cdot B / (W \cdot N)$。对于MoE，我们通常独立随机选择非零值，所以有一些机会拥有少于$k$个非零值，给我们大约
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。<d-footnote>真实成本实际上是$$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$ $K$次骰子掷出的不同结果的期望数量，但它与给定的近似值非常接近。有关更多详细信息，请参见附录。</d-footnote>

<b markdown=1 style="color: #c55404ff;">小测验2 [AllToAll时间]：</b> 使用一个具有450 GB/s单向带宽的8xH100节点，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N])需要多长时间？如果我们知道只有8个条目中的4个将是非零的，会怎样？

{% details 点击这里查看答案。 %}

**答案：** 从上面，我们知道在密集情况下，成本是$B \cdot (N-1) / (W \cdot N^2)$，或$B / (W \cdot N)$。如果我们知道只有$\frac{1}{2}$的条目将是非填充的，我们可以发送$B \cdot k/N / (W \cdot N) = B / (2 \cdot W \cdot N)$，大约是总成本的一半。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 在单个节点内GPU上$B$字节数组的AllToAll成本约为$T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$。对于不规则（top-$k$）AllToAll，这进一步减少到$(B \cdot k) / (64 \cdot W_\text{GPU egress})$。</p>

**经验测量：** 这里是8xH100节点上AllReduce带宽的经验测量。算法BW是测量的带宽（字节/运行时间），总线BW计算为$2 \cdot W \cdot (8 - 1) / 8$，理论上是对实际链路带宽的度量。您会注意到我们确实实现了接近370GB/s，低于450GB/s但相当接近，尽管只有大约10GB/设备。这意味着虽然这些估计在理论上是正确的，但需要大消息才能实现它。

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>Figure:</b> AllReduce throughput for an 8xH100 node with SHARP disabled. The blue curve is the empirical link bandwidth, calculated as $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$ from the empirical measurements. Note that we do not get particularly close to the claimed bandwidth of 450GB/s, even with massive 10GB arrays." %}

这是一个真正的问题，因为它有意义地复杂化我们可以做出的任何理论声明，因为例如，即使在合理大小数组上的AllReduce，如LLaMA-3 70B的MLP（大小为`bf16[8192, 28672]`，或者使用8路模型分片，`bf16[8192, 3584] = 58MB`）也只能实现大约150GB/s，与峰值450GB/s相比。相比之下，TPU在低得多的消息大小上实现峰值带宽（见附录B）。

<p markdown=1 class="takeaway">**要点：** 虽然NVIDIA声称在H100 NVLink上的带宽约为450GB/s，但在实践中很难超过370 GB/s，因此相应地调整上述估计。</p>

**网络内归约：** 自Hopper代以来，NVIDIA交换机支持["SHARP"（可扩展分层聚合和归约协议）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，它允许"网络内归约"。这意味着*网络交换机本身*可以执行归约操作并将结果多路复用或"多播"到多个目标GPU：

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>Figure:</b> an AllReduce without SHARP has 2x the theoretical cost because it has to pass through each GPU twice. In practice, speedups are only about 30% (from NCCL 2.27.5)." %}

理论上，这将近乎将AllReduce的成本减半，因为它意味着每个GPU可以将其数据发送到顶级交换机，交换机本身执行归约并将结果广播到每个GPU，而不必两次输出每个GPU，同时还减少网络延迟。

$$T_\text{SHARP AR通信} = \frac{\text{bytes}}{\text{GPU egress带宽}}$$

请注意，这是精确的，而不是偏离$1/N$的因子，因为每个GPU首先输出$B \cdot (N - 1) / N$，然后接收其本地分片的部分归约版本（输入$B/N$），完成归约，然后再次输出$B/N$，然后输入完全归约的结果（输入$B \cdot (N - 1) / N$），导致正好$B$字节被输入。

然而，在实践中，我们看到启用SHARP后带宽增加约30%，与预测的75%相比。这仅仅使我们达到大约480GB/s的有效集合带宽，远不到2倍。

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>Figure:</b> empirical measurements of AllReduce algo bandwidth with and without NVIDIA SHARP enabled within a node. The gains amount to about 30% throughput improvement at peak, even though algorithmically it ought to be able to achieve closer to a 75% gain." %}

<p markdown=1 class="takeaway">**要点：** 理论上，NVIDIA SHARP（在大多数NVIDIA交换机上可用）应该将$B$字节的AllReduce成本从大约$2 * B / W$减少到$B / W$。然而，在实践中，我们只看到带宽大约30%的改善。由于纯AllReduce在LLM中相当罕见，这并不是特别有用。</p>

### 跨节点集合通信

当我们超越节点级别时，成本有点更加微妙。在树上进行归约时，您可以考虑从下往上归约，首先在节点内，然后在叶层，然后在脊层，在每个级别使用正常算法。特别是对于AllReduce，您可以看到这允许我们总体上通信更少的数据，因为我们在节点级别进行AllReduce后，我们只需要输出$B$字节到叶层而不是$B * N$。

**这有多昂贵？** 作为一级近似，因为我们有全对分带宽，AllGather或ReduceScatter的成本大致是字节缓冲区大小除以节点输出带宽（H100上为400GB/s）*无论树归约的任何细节*。

$$T_\text{AG或RS通信} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

where $W_\text{node}$ egress is generally 400GB/s for the above H100 network (8x400Gbps IB links egressing each node). The cleanest way to picture this is to imagine doing a ring reduction over *every node in the cluster*. Because of the fat tree topology, we can always construct a ring with $W_\text{node}$ egress between any two nodes and do a normal reduction. The node-level reduction will (almost) never be the bottleneck because it has a higher overall bandwidth and better latency, although in general the cost is

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]$$

{% details You can see a more precise derivation here. %}

We can be more precise in noting that we are effectively doing a ring reduction at each layer in the network, which we can mostly overlap, so we have:

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

where $D_i$ is the degree at depth $i$ (the number of children at depth $i$), $W_\text{link i}$ is the bandwidth of the link connecting each child to node $i$.

Using this, we can calculate the available AllGather/AllReduce bandwidth as $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$ for a given topology. In the case above, we have:

* **Node:** $D_\text{node}$ = 8 since we have 8 GPUs in a node with Wlink i = 450GB/s. Thus we have an AG bandwidth of `450e9 * 8 / (8 - 1) = 514GB/s`.
* **Leaf:** $D_\text{leaf}$ = 32 since we have 32 nodes in an SU with Wlink i = 400GB/s (8x400Gbps IB links). Thus our bandwidth is `400e9 * 32 / (32 - 1) = 413GB/s`.
* **Spine:** $D_\text{spine}$ = 4 since we have 4 SUs with $W_\text{link i}$ = 12.8TB/s (from `8 * 16 * 2 * 400Gbps` links above). Our bandwidth is `12.8e12 * 4 / (4 - 1) = 17.1TB/s`.

Hence our overall AG or RS bandwidth is `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s` at the leaf level, so in practice $T_\text{AG or RS comms} = B / \text{413GB/s}$, i.e. we have about 413GB/s of AllReduce bandwidth even at the highest level. For an AllReduce with SHARP, it will be slightly lower than this (around 400GB/s) because we don’t have the $(N - 1) / N$ factor. Still, 450GB/s and 400GB/s are close enough to use as approximations.

{% enddetails %}

**Other collectives:** AllReduces are still 2x the above cost unless SHARP is enabled. NVIDIA sells SHARP-enabled IB switches as well, although not all providers have them. AllToAlls do change quite a bit cross-node, since they aren't "hierarchical" in the way AllReduces are. If we want to send data from every GPU to every other GPU, we can't use take advantage of the full bisection bandwidth at the node level. That means if we have an N-way AllToAll that spans $M = N / 8$ nodes, the cost is

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approxeq \frac{B}{M \cdot W_\text{node egress}}$$

which effectively has 50GB/s rather than 400GB/s of bandwidth. We go from $B / (8 * \text{450e9})$ within a single H100 node to $B / (2 \cdot \text{400e9})$ when spanning 2 nodes, a more than 4x degradation.

Here is a summary of the 1024-GPU DGX H100 SuperPod architecture:

|   Level   | Number of GPUs | Degree (# Children) | Switch Bandwidth (full-duplex, TB/s) | Cable Bandwidth (full-duplex, TB/s) | Collective Bandwidth (GB/s) |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
|   Node    |       8        |          8          |                 6.4                  |                 3.6                 |             450             |
| Leaf (SU) |      256       |         32          |                 25.6                 |                12.8                 |             400             |
|   Spine   |      1024      |          4          |                 51.2                 |                51.2                 |             400             |

We use the term "Collective Bandwidth" to describe the effective bandwidth at which we can egress either the GPU or the node. It’s also the $\text{bisection bandwidth} * 2 / N$.

<p markdown=1 class="takeaway">**Takeaway:** beyond the node level, the cost of an AllGather or AllReduce on B bytes is roughly $B / W_\text{node egress}$, which is $B / \text{400e9}$ on an H100 DGX SuperPod. The overall topology is a fat tree designed to give constant bandwidth between any two pairs of nodes.</p>

**当数组在单独轴上分片时的归约：** 考虑像这样的归约成本

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

其中我们在一个本身沿着另一个轴$Y$分片的数组上进行AllReduce。在TPU上，与未分片版本相比，此操作的总成本减少了$1 / Y$因子，因为我们每个轴发送的数据量是$1 / Y$。在GPU上，成本取决于哪个轴是"内部"轴（节点内vs节点间）以及每个分片是否跨越多个节点。假设$Y$在这里是内部轴，总成本有效地减少了$Y$，但只有当$Y$跨越多个节点时：

$$T_\text{节点通信} = \frac{\text{bytes} \cdot D_\text{node}}{\min(Y, D_\text{node}) \cdot W_\text{GPU egress}}$$

$$T_\text{扩展网络通信} = \frac{\text{bytes} \cdot N}{Y \cdot W_\text{node egress}}$$

$$T_\text{总} = \max(T_\text{节点通信}, T_\text{扩展网络通信})$$

其中N是GPU数量，D是节点中的GPU数量（节点的度）。如您所见，如果$Y < D_\text{node}$，我们在节点级别获得收益但通常看不到整体运行时间的减少，而如果$Y > D_\text{node}$，我们获得与跨越的节点数量成比例的加速。

如果我们想要精确地了解环形归约，对于树AllGather<sub>X</sub>(A<sub>Y</sub> { U<sub>X</sub> })的一般规则（假设Y是内部轴）是

$$T_\text{AR或RS通信} = \text{bytes} \cdot \max_{\text{深度 } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{链路 } i}}\right]$$

其中$S_i$是M * N * …，树中第i级以下的子节点的大小。这大致是说，我们跨越的GPU或节点越多，我们的可用带宽就越大，但仅在该节点内。

**小测验3 [沿2轴分片]：** 假设我们想要执行$\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，其中$Y$是单个SU（256芯片）上的内部轴。这将需要多长时间作为$D$、$F$和$Y$的函数？

{% details 点击这里查看答案。 %}

**答案：** 我们可以将其分为两种情况，Y <= 8和Y > 8。当$Y <= 8$时，我们仍然受叶层交换机限制，所以答案是，像往常一样，$T_\text{通信} = 2 * D * F * (32 - 1) / (32 * 400e9)$。当Y > 8时，我们从上面大致得到

$$T_\text{通信} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

对于`D = 8192`，`F = 32,768`，我们有：

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>Figure:</b> theoretical cost of a sharded AllGather as the inner axis spans more nodes." %}

注意，如果我们精确地进行8路模型并行，我们实际上确实将节点级归约的成本减少了8，但总成本保持不变，所以它是免费的但对改善整体带宽没有帮助。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 当我们有多个分片轴时，外部归约的成本减少了内部轴跨越的节点数量的因子。</p>

### 测验4：集合通信

**问题1 [SU AllGather]：** 考虑只有一个SU，有M个节点，每个节点有N个GPU。在AllGather期间，节点级交换机精确地输入和输出了多少字节？顶级交换机呢？

{% details 点击这里查看答案。 %}

**答案：** 让我们逐步完成，通过归约的组件：

1. 每个GPU发送$B / MN$字节到交换机，总共输入$NB / MN = B / M$字节。
2. 我们将完整的$B / M$字节输出到脊层交换机。
3. 我们从脊层交换机输入$B * (M - 1) / M$字节
4. 我们输出$B - B / MN$字节$N$次，总共$N * (B - B / MN) = NB - B / M$。

总计是$B$字节输入和$BN$字节输出，所以我们应该受输出限制，总时间将是$T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

对于脊层交换机，数学实际上更简单。我们必须有$B / M$字节输入M次（总共$B$字节），然后$B (M - 1) / M$输出M次，总共$B * (M - 1)$字节输出。由于这显著更大，成本是$T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。

{% enddetails %}

**问题2 [单节点SHARP AR]：** 考虑一个单节点，每个节点有N个GPU。在使用SHARP（网络内归约）的AllReduce期间，交换机精确地输入和输出了多少字节？

{% details 点击这里查看答案。 %}

**答案：** 像之前一样，让我们逐步完成。

1. 每个GPU发送$B * (N - 1) / N$字节，所以我们有$N * B * (N - 1) / N = B * (N - 1)$字节输入。
2. 我们累积部分和，然后发送回$B / N$字节到每个GPU，所以$N * B / N = B$字节输出。
3. 我们在残差上做部分和，然后将其发送回交换机。这总共是$N * B / N = B$字节输入。
4. 我们捕获所有分片并多播它们，发送$B * (N - 1) / N$到$N$个目的地，总共$B * (N - 1) / N * N = B * (N - 1)$字节输出。

因此总共是$B * (N - 1) + B = BN$字节输入和输出。这支持整体吞吐量正好是$B / W_\text{egress}$。

{% enddetails %}

**问题3 [跨节点SHARP AR]：** 考虑一个数组bf16[D<sub>X</sub>, F<sub>Y</sub>]在单个N个GPU的节点上分片。AllReduce(bf16[D, F<sub>Y</sub>] { U<sub>X</sub> })需要多长时间？您可以假设我们进行网络内归约。解释如果我们有多个节点时这如何不同？

{% details 点击这里查看答案。 %}

**答案：** 我们可以尝试修改上面问题的答案。基本上，我们首先从每个GPU输出$B * (X - 1) / XY$字节，然后发送回$B / XY$到每个GPU，然后将相同数量发送回交换机，然后发送$B * (X - 1) / XY$回到每个GPU。总共是$NB / Y$字节输入和输出，所以总时间是$T_\text{通信} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$，所以总时间确实随着$Y$减少。

如果我们超越单个节点，我们可以做大致与上面相同的归约，但当我们输出节点级交换机时，我们需要发送所有B字节，而不仅仅是$B / Y$。这是因为我们需要保持每个分片分开。

{% enddetails %}

**问题4 [脊层AR成本]：** 考虑与上面相同的设置，但$Y = 256$（所以AR发生在脊层）。AllReduce需要多长时间？再次，随时假设网络内归约。

{% details 点击这里查看答案。 %}

**答案：** 这让我们能够利用脊层相当大量的带宽。我们在4个节点上有25.6TB/s的带宽，所以AllReduce带宽为6.4TB/s。使用SHARP，这可能只需要`2 * D * F / 6.4e12`秒。

{% enddetails %}

**问题5 [2路AllGather成本]：** 考虑正好在2个节点上的AllGather成本。它精确地是多少？确保计算精确成本而不是近似值。

{% details 点击这里查看答案。 %}

**答案：** 在节点级别，我们有$T_\text{通信} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$而超越节点我们实际上有$T_\text{通信} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。因此，我们实际上受节点级归约限制而不是叶层！这激发了例如DeepSeek v3进行2路数据并行。

{% enddetails %}

## GPU上LLM扩展的性能上限

现在让我们看看这一切都是为了什么：理解GPU上LLM扩展的性能上限。这是对TPU训练章节[这里](../training)的补充。正如我们在那里所做的，这里的目的是查看不同并行策略的总$T_\text{math}$和$T_\text{comms}$，并理解在什么点$T_\text{comms} > T_\text{math}$。和之前一样，我们只考虑具有以下操作的MLP块

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中$B$是以**token**为单位的全局批次大小（即$B = \text{batch size} \cdot \text{sequence length}$）。

这里我们将重现上面的表格，显示GPU和节点级别的有效带宽：

|  节点类型  | 每节点GPU数 | GPU输出带宽 | 节点输出带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**注意：** GPU和节点输出带宽都决定了我们LLM的性能上限。我们将使用术语$W_\text{collective}$来描述GPU或节点带宽，取决于我们是在节点级别内还是节点级别之上操作。

让我们看看计算通信性能上限，就像我们对TPU所做的那样，针对**数据并行、张量并行、流水线并行、专家并行**及其组合。在本节的其余部分，我们将专注于H100性能上限进行具体计算。GB200-NVL72具有相同的一般性能上限，但因为我们有更大的节点输出带宽，我们有时可能会在节点级别遇到瓶颈。

### 数据并行

如前所述，DP和ZeRO分片涉及反向传播中的权重AllReduce或ReduceScatter + AllGather。由于这两者成本相同，为了在纯数据并行或FSDP*没有网络内归约*的情况下达到计算限制，每层在反向传播中，对于大小为X的轴，我们有：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

因此，对于$T_\text{math} > T_\text{comms}$，我们需要$B / (XC) > 1 / W_\text{collective}$或

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中$W_\text{collective}$是GPU或节点级别的输出带宽，取决于我们是在节点内还是跨节点分片。因此：

* **节点内**，我们只需要每GPU**token**批次大小 > $\text{990e12} / \text{450e9} = 2200$。
* **SU或脊层级别**，批次大小 > $\text{990e12} / \text{400e9} = 2475$。

这比TPU高很多，TPU在所有三个轴上的数字是850。例如，在16000个H100上训练的LLaMA-3将需要至少4000万个token的批次大小（参考，他们使用了1600万）。在2048个H800 GPU上训练的DeepSeek v3，带宽较低300GB/s（而不是H100上的450GB/s），每个GPU需要$\text{990e12} / \text{300e9} = 3300$个token，或约670万（实践中，他们使用了400万）。

启用网络内归约并使用纯数据并行，理论上我们有2倍的AllReduce带宽，这将使这两个数字减半。然而，在实践中，好处接近30%，这实际上只是弥补了我们通常难以达到报告数字的事实。此外，因为纯数据并行很少有用，这基本上在实践中并不重要。

**MoE模型：** 对于混合专家（MoE）模型，其中我们有E个专家，每个token有k个专家，这增加到

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

这将每GPU token批次大小放大了$E/k$倍，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如，新的OpenAI OSS模型，$k=4$和$E=128$，这在跨节点时增加到`32 * 2475 = 79,200`，这是一个荒谬的高数字。

**当X很小时会发生什么？** 当我们只做例如2节点数据并行时，我们从$(X - 1) / X$缩放中受益，这给我们

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中X是节点数，$N = 8 \cdot X$。那么对于密集模型，我们有$B / N > \alpha \cdot (X - 1) / X$，或例如$B / N > \text{1237}$，是上述值的一半。因此，你会经常看到2路数据并行。

<p markdown=1 class="takeaway">**要点：** 数据并行和ZeRO分片需要每GPU约2500个token的批次大小才能在H100或B200上达到计算限制，假设完美的重叠和FLOPs利用率。对于MoE模型，这增加了$E / k$倍，即总参数与激活参数的比率。当进行少量数据并行时，关键批次大小减小。</p>

### 张量并行

张量并行需要在激活上进行AllGather和ReduceScatter，我们需要将其与MLP FLOPs重叠。换句话说，在前向传播中，我们有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

为了达到计算限制，这给我们规则

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

在节点内，这给我们大约$F / 2200$或节点外$F / 2475$。对于像LLaMA-3的$F=\text{28000}$，这大约是11路TP（或向下取整约8路，这是一个节点的大小）。与上面一样，当我们跨越正好2个节点时，我们获得额外的2倍带宽，所以我们可以通常做16路数据并行（$F > 2475 \cdot (Y - 8)$），这在理论上给我们最多19路模型并行。

<p markdown=1 class="takeaway">**要点：** 在大小为Y的轴上进行张量并行，前馈维度为F，当$Y > F / 2475$时会变成通信限制，这通常将我们限制为仅节点内TP或最多2节点TP。</p>

### 专家并行

正如我们上面已经指出的，混合专家（MoE）模型带来了E倍更多的模型权重，但只有k倍更多的FLOPs，这使得数据并行显著更难。我们可以通过沿专家维度分片我们的权重来在一定程度上缓解这个问题，即W<sub>in</sub>[E<sub>Z</sub>, D, F]。为了执行MLP块，我们需要引入2x AllToAll将我们的激活发送到相应的专家。

如上所述，如果这个AllToAll<sub>Z->k</sub>([B, D, k])跨越多个节点，成本大致为$T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，所以对于纯专家并行，我们需要

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们需要$K > Z/8$且$F > \alpha \cdot (Z - 8)/k$，或者$Z \gg K$且$F > 8 \cdot \alpha$，其中$\alpha = C/W$。这给了你两个专家并行可能的领域，一个是少量专家并行（大约2节点）和小$F$，或者一个大$F$和任意大$Z$（最多E路专家并行）。

在实践中你会看到两种情况，要么是少量专家并行（像DeepSeek v3，它有非常小的F和相对较小的、受限的跨节点专家并行），或者是有大F的模型，在这种情况下我们可以与TP一起进行显著的跨节点EP。

<p markdown=1 class="takeaway">**要点：** 如果$F < 8 * C / W_\text{node}$，专家并行可以跨越1-2个节点，成本与TP相似（略低），或者如果$F > 8 * C / W_\text{node}$，我们可以进行大量的专家并行（最多$E$个节点），成本相对较低。</p>

### Pipeline Parallelism

流水线并行将层分配到各个节点，通信成本极低，因为我们只是每隔几层发送小的微批次激活。历史上，流水线一直受到"流水线气泡"的困扰，但有了新的零气泡流水线方法，通常可以避免这种情况。

流水线并行的总体通信成本极小：有$N_\text{MB}$个微批次和$N_\text{stages}$个阶段，我们有$T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$和$N_\text{MB} + N_\text{stages} - 2$跳，所以大致上

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{microbatches}} \cdot (N_\text{microbatches} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于我们除以$N_\text{layers}$，这比其他任何成本都要小得多。换句话说，从通信的角度来看，流水线基本上是免费的。那么为什么我们不直接使用流水线呢？有几个原因：

(1) **代码复杂性：** 流水线不像其他方法那样很好地融入自动并行框架（如XLA的GSPMD）。因为它引入了微批次来隐藏流水线气泡，这改变了程序的结构，而定制的零气泡流水线调度通过需要前向和反向传递的复杂交错加剧了这个问题。

(2) **流水线使数据并行和FSDP变得困难：** 可能不使用流水线的最大原因是它与FSDP和数据并行配合不佳。特别是ZeRO-3分片效果很差，因为它要求我们在每个微批次上AllGather权重，当我们只有$B / N_\text{microbatches}$个token来分摊AllGather成本时，这是行不通的。此外，在反向传播期间，*在最后一个微批次通过给定阶段之前，我们不能AllReduce或ReduceScatter梯度，这意味着我们有显著的非重叠通信时间。*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>Figure:</b> 一个2阶段、2微批次流水线的例子。F表示阶段前向传递，B是阶段反向传递（成本是2倍）。G表示数据并行AllReduce，可能比单个微批次的时间长得多。" %}

(3) **流水线气泡和步骤不平衡：** 正如你在上面（不好的）流水线调度中看到的，在朴素的流水线调度中很容易出现显著的气泡（意味着浪费的计算）。上面，第二阶段在第0步空闲，第一阶段从第2步到第3步空闲，第二阶段在最后一步再次空闲。虽然我们可以通过仔细调度在一定程度上避免这些，但我们仍然经常有一些气泡。我们还必须在关键路径上将激活从一个阶段传递到下一个阶段，这可能会增加开销：

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>Figure:</b> 一个显示传输成本的流水线例子（红色）。这使阶段相对于彼此移动，并增加了流水线气泡开销。" %}

对于这些问题中的每一个都有变通方法，但它们往往实现复杂且难以维护，但流水线仍然是一种相对于其他方法通信成本较低的技术。

**关于延迟的提醒：** 如前所述，即使有相当大的消息，GPU也很难实现完整的AllReduce带宽。这意味着即使我们在理论上可以扩展例如跨多个节点的专家并行AllToAll，我们可能也很难达到总带宽的50%。这意味着我们确实尝试将TP或EP保持在较少数量的节点内，以最小化延迟开销。

### 示例

**DeepSeek是怎么做的？** 作为参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437)使用2048个H800 GPU进行训练：

* 64路专家并行（EP）跨越8个节点
* 16路流水线并行（PP）
* 2路ZeRO-1数据并行（DP）

他们的稳态批次大小为`4096 * 15360 = 62,914,560`个token，或每个GPU 30k个token。你可以看到这已经相当大了，但他们的模型也非常稀疏（k=8, E=256），所以你需要相当大的批次大小。你可以看到，使用64路EP和16路PP，我们最终总共得到1024路模型并行，这意味着AllReduce在脊层进行，并且因为它只有2路，我们在实践中最终得到$2 / (2 - 1) = 2$倍的带宽。这也有助于减少最终数据并行AllReduce与最终流水线阶段重叠的成本。

**LLaMA-3是怎么做的？** LLaMA-3在16k个GPU上使用16M个token的批次大小进行训练，或每个GPU约1k个token。他们这样做：

* 节点内8路张量并行（TP）
* 16路流水线并行（PP）
* 128路ZeRO-1数据并行

这也是一个密集模型，所以一般来说这些事情都相当简单。16路PP将数据并行AllReduce的成本降低了16倍，这有助于我们减少关键批次大小。

### GPU上LLM扩展的TLDR

让我们退一步，总结一下我们到目前为止学到的内容：

* **数据并行或FSDP（ZeRO-1/3）需要每个GPU约2500个token的本地批次大小**，尽管理论上网络内归约 + 纯DP可以稍微减少这个数字。
* **张量并行在达到约8路之前是计算限制的**，但我们在变成通信限制之前缺乏带宽来扩展更多。这主要将我们限制在单个NVLink域（即单节点或需要使用GB200NVL72达到72个GPU）。
* **任何跨越多个节点的模型并行形式都可以进一步降低FSDP的成本**，所以我们经常想要混合PP + EP + TP来跨越多个节点并降低FSDP成本。
* **如果你能处理零气泡流水线的代码复杂性并保持批次大小相当大以避免数据并行瓶颈，流水线并行效果很好。** 流水线通常使ZeRO-3变得不可能（因为你需要在每个流水线阶段AllGather），但你可以改用ZeRO-1。

**在高层面上，这给了我们在GPU上分片大型模型的方法：**

* 对于相对较小的密集模型，如果你有批次大小，激进的FSDP效果很好，如果需要可能带有一些流水线或张量并行。
* 对于较大的密集模型，1-2个节点TP + 多个节点PP + 纯DP的组合效果很好。
* 对于MoE，上述规则适用，但我们也可以进行专家并行，我们通常比TP更喜欢专家并行。如果$F > 8 * C / W_\text{node}$，我们可以进行大量的多节点专家并行，否则我们限制在大约2个节点EP。

### 测验5：LLM性能上限

**问题1 [B200性能上限]：** B200 DGX SuperPod（**不是GB200 NVL72**）在节点内有2倍的带宽（900GB/s输出），但在扩展网络中有相同数量的带宽（400GB/s）（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)）。总FLOPs如上所述。这如何改变模型和数据并行性能上限？

{% details 点击这里查看答案。 %}

**答案：** 我们在bfloat16中的FLOPs/s从990增加到2250 TFLOPs，增加了2.25倍。带宽增加2倍，在节点内，我们的性能上限大致保持不变。例如，对于TP，关键强度上升到`2250e12 / 900e9 = 2500`，所以我们有$Y < F / 2500$的限制，仅略高（除非节点大小增加，否则这没有帮助）。

然而，在节点之外，缺乏额外带宽实际上使我们更难达到计算限制！例如，对于数据并行，我们的关键批次大小增加到`2250e12 / 400e9 = 5625`，因为我们的GPU可以用相同的带宽做显著更多的FLOPs。

具有72个GPU节点的GB200 SuperPod通过添加更多输出带宽来改变这一点（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。

{% enddetails %}

**问题2 [如何分片LLaMA-3 70B]：** 考虑LLaMA-3 70B，使用bfloat16和fp32优化器状态与Adam进行训练。

1. 最少需要多少个H100来存储权重和优化器？
2. 假设我们想要在4096个H100 GPU上训练15T个token。假设我们实现了45%的MFU（模型FLOPs利用率）。训练需要多长时间？
3. LLaMA-3 70B有`F = 28,672`，使用约4M个token的批次大小进行训练。我们最多可以进行多少模型并行而不受通信限制？有了这个加上纯DP，我们能否在保持计算限制的情况下训练LLaMA-3？ZeRO-3呢？使用8路流水线呢？

{% details 点击这里查看答案。 %}

1. 我们需要2字节用于权重，8字节用于优化器状态，所以至少700GB。有80GB的DRAM，我们至少需要9个GPU，或者（向上取整）至少2个8xH100节点。这将需要永远来训练，并且无法保存梯度检查点，但这是一个下限。
2. 这总共需要`6 * 70e9 * 15e12 = 6.3e24` bf16 FLOPs。每个GPU可以做`990e12` FLOPs，所以在40% MFU下我们可以做1.6e18 FLOPs/s。因此整个过程需要3.9e6秒，或45天。
3. 在节点内，我们有450GB/s的带宽，所以限制大致是`F / 1995 = 28672 / 1995 = 14.372`。由于这不跨越2个节点，实际上意味着我们会达到8路模型并行。
   1. 这将要求我们做512路DP。首先，我们需要看我们是否有足够的内存。由于我们的模型只分片8路，这意味着`700GB / 8 = 87.5GB / GPU`，这不适合，所以不行！
   2. 使用ZeRO-3和8路TP，我们将做512路ZeRO-3。这不会有任何内存问题，因为我们激进地分片所有东西。我们的每GPU批次大小为`4e6 / 4096 = 976`。这相当低，甚至低于我们的纯DP限制，并且这是该限制的两倍，因为我们必须移动我们的权重。所以不行。
   3. 使用8路流水线，每个模型并行分片现在跨越8个节点。正如我们所看到的，这将我们的叶级AllGather成本降低了8倍，所以那里的总体AllReduce/AllGather带宽从400GB/s变为`8 * 400GB/s = 3200GB/s`。然后性能上限是`989e12 / 3200e9 = 309`，所以我们应该没问题！我们只需要有效地实现流水线。

{% enddetails %}

**问题3 [Megatron-LM超参数]：** 考虑来自[Megatron-LM仓库](https://github.com/NVIDIA/Megatron-LM)的这张图，突出了他们高MFU数字。

{% include figure.liquid path="assets/gpu/megatron-hparams.png" class="img-fluid" %}

注意他们的序列长度到处都是4096。对于16B、70B和314B模型，每GPU token批次大小是多少？假设数据并行是最外层的轴并假设bfloat16归约，确定这些在理论上是计算限制还是通信限制，以及是否有更优的配置可用？

{% details 点击这里查看答案。 %}

**答案：** 让我们从每GPU的批次大小开始。

* **16B**：`192 * 4096 / 192 = 4096` 每GPU token
* **70B**：`384 * 4096 / 768 = 2048` 每GPU token
* **314B**：`1536 * 4096 / 3072 = 2048` 每GPU token

这意味着除了第一个，这些都在每批次约2k个token左右，这显著地围绕我们为FSDP计算的关键阈值。我们计算该限制为2,472个token/GPU，基于脊层归约，这应该在这里大致起作用。然而，对于70B和314B，因为我们分别有16路和64路模型分片，我们在脊层获得2倍和8倍更好的吞吐量，这意味着我们应该分别在约1k和300个token/步时达到计算限制。

{% enddetails %}

## 致谢和进一步阅读

本章在很大程度上依赖于许多知识渊博的GPU专家的帮助，包括：

* Adam Paszke，帮助解释了GPU上内核编程的现实。
* Swapnil Patil，首先解释了GPU网络的工作原理。
* Stas Bekman，指出了GPU的实际情况通常与声称的规格不同。
* Reiner Pope，帮助澄清了GPU和TPU在硬件层面的比较。
* Frédéric Bastien，对芯片级故事提供了详细的反馈。
* Nouamane Tazi，他在GPU上训练LLM的经验帮助改进了性能上限部分。
* Sanford Miller，帮助我理解了GPU是如何联网的，以及NVIDIA的规格与实际部署的相比如何。

关于GPU有很多好的阅读材料，但我最喜欢的一些包括：

* [SemiAnalysis的NVIDIA张量核心历史](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：一篇很棒的文章，描述了GPU如何从游戏引擎转变为ML加速器。
* [SemiAnalysis的Blackwell性能分析](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/)：值得阅读以了解NVIDIA GPU的下一代。
* [H100 DGX SuperPod参考](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：关于更大的GPU集群如何联网的枯燥但有用的阅读。[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)是关于GB200系统的类似文档。
* [关于NVLink交换机的Hot Chips演讲](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：关于NVLink和NCCL集合通信的有趣阅读，特别是包括网络内归约。
* [DeepSeek-V3技术报告](https://arxiv.org/pdf/2412.19437)：一个大型半开放LLM训练报告的好例子，描述了他们如何选择他们的分片设置。
* [如何优化CUDA矩阵乘法](https://siboehm.com/articles/22/CUDA-MMM)：一篇很棒的博客，描述了如何使用CUDA核心实现高效的矩阵乘法，着眼于GPU上的缓存一致性。
* [HuggingFace超大规模手册：](https://huggingface.co/spaces/nanotron/ultrascale-playbook) GPU上LLM并行的指南，部分启发了本章。
* [从第一原理让深度学习变得更快：](https://horace.io/brrr_intro.html)：一个更侧重于GPU和PyTorch的LLM性能上限和性能工程教程。

## 附录A：GB200带来了什么变化？

Blackwell引入了许多重大的网络变化，包括NVLink 5，其总体NVLink带宽翻倍（900GB/s）。B200仍然有8个GPU的节点，就像H100一样，但GB200系统（将B200 GPU与Grace CPU结合）引入了更大的NVLink域（NVL72中有72个GPU，理论上最多可达576个）。这个更大的NVLink域也有效地增加了节点输出带宽，这降低了节点级别以上的集合通信成本。

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>Figure:</b> 显示GB200 NVL72单元如何构建的图，有18个交换机和72个GPU。" %}

在节点内，这种增加的带宽（从450GB/s到900GB/s）没有太大区别，因为我们也使每个GPU的总FLOPs/s翻倍。我们的性能上限大多保持不变，尽管因为NVLink有更好的带宽，专家并行变得更容易。

在节点之外，事情变化更大。这是来自[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的SuperPod图。

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>Figure:</b> 显示576个GPU的GB200 DGX SuperPod的图。" %}

正如你所看到的，每节点输出带宽增加到`4 * 18 * 400 / 8 = 3.6TB/s`，比H100的400GB/s有所增加。由于我们的FLOPs/芯片也翻倍，这有效地将跨节点性能上限提高了约4倍。现在我们可能开始担心我们是否在节点级别而不是扩展级别遇到瓶颈。

**Grace Hopper：** NVIDIA还销售GH200和GB200系统，将一些GPU与Grace CPU配对。例如，GH200有1个H200和1个Grace CPU，而GB200系统有2个B200和1个Grace CPU。这个系统的一个优点是CPU使用全带宽NVLink连接（称为NVLink C2C）连接到GPU，所以你有非常高的CPU到GPU带宽，这对于将参数卸载到主机RAM很有用。换句话说，对于任何给定的GPU，到达主机内存的带宽与到达另一个GPU的HBM相同。

## 附录B：更多网络细节

这是一个NVLink 4交换机的图。总共有64个NVLink4端口（每个使用2个物理通道），和一个处理通道间交换的大型交叉开关。相比之下，TPU使用带有可以动态重新配置的镜子的光交换机。

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>Figure:</b> 单个NVLink4交换机的较低级别视图。" %}

在每个级别，我们可能受到可用链路带宽或总交换机带宽的限制。

* **节点级别：** 在节点级别，我们有4 * 1.6TB/s = 6.4TB/s的NVSwitch带宽，但我们的8个GPU每个只能输出450GB/s到交换机，这意味着我们在节点内实际上有450e9 * 8 = 3.6TB/s（全双工）的峰值带宽。
* **SU/叶层级别：** 在SU级别，我们有8个交换机以全对全方式连接32个节点，使用1x400 Gbps Infiniband。这给了我们从节点出来的8 * 32 * 400 / 8 = 12.8TB/s输出带宽，我们在交换机级别有8 * 1.6TB/s = 12.8TB/s，所以两者完全一致。
* **脊层级别：** 在脊层级别，我们有16个交换机用2x400 Gbps链路连接32个叶层交换机，所以我们有32 * 16 * 400 * 2 / 8 = 51.2TB/s的输出带宽。16个交换机给我们16 * 1.6TB/s = 25.6TB/s的带宽，所以这是这个级别的瓶颈。

每个GPU，这给了我们在节点级别450GB/s的GPU到GPU带宽，在SU级别50GB/s，在脊层级别25 GB/s。

**GPU经验AR带宽：**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>Figure:</b> 8xH100集群上的AllReduce带宽（节点内，SHARP禁用）。" %}

TPU v5p带宽（1个轴）：

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>Figure:</b> TPU v5p 4x4x4集群上的AllReduce带宽（沿一个轴）。" %}

这里还有AllGather带宽：

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>Figure:</b> 8xH100集群上的AllGather带宽（节点内）。" %}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>Figure:</b> TPU v5e 8x16集群上的AllGather带宽（沿一个轴）。" %}

**更多关于AllToAll成本的信息：**

这里我们可以将近似值$\min(K / Z) * (Z - 1) / Z$与真实值$(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$进行比较。它们很相似，除了小的$Z$值。

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>Figure:</b> 随着分片数量增加，不规则AllToAll的近似和真实成本比较。" %}
