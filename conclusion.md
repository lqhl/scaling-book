---
layout: distill
title: "结论与延伸阅读"
# permalink: /main/
description: "感谢您的阅读！这里我们将提供一些延伸学习的参考资料."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "Part 10: JAX"

next_section_url: "../gpus"
next_section_name: "Part 12: GPUs"

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
  - name: "致谢"
  - name: "延伸阅读"
  - name: "反馈"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---
**感谢您阅读本系列文章, 并祝贺您坚持到了最后.** 在我们结束之前, 先致以一些谢意:

## 致谢

本文档代表了 Google DeepMind 许多同仁的巨大集体投入, 我们想在此简要致谢!

*   James Bradbury, Reiner Pope, 和 Blake Hechtman 最初推导出了手稿中的许多想法, 并且是早期理解 Transformer 系统观点的先行者.
*   Sholto Douglas 撰写了本文档的初版, 并负责启动该项目. 他比任何人都更能代表本文档的整体叙事.
*   Jacob Austin 领导了将初版的粗略笔记转变为更精炼、更全面的成果的工作. 他完成了本文档的大部分编辑、格式化和发布工作, 并协调了其他作者的贡献.
*   大部分图表和动画由 Anselm Levskaya 和 Charlie Chen 制作.
*   Charlie Chen 撰写了推理部分, 并绘制了许多推理部分的图表.
*   Roy Frostig 在出版、编辑和许多其他环节提供了帮助.

我们还要感谢许多在整个过程中给予关键反馈的同仁, 特别是 Zak Stone, Nikhil Sethi, Caitlin Stanton, Alex Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins 等等. 感谢 Ruiqi Gao 在 HTML 格式化方面的帮助.

**感谢大家!**

<p markdown=1 class="announce">在您离开之前, 您可能也想阅读关于 NVIDIA GPU 的新章节 [第 12 节](../gpus)!</p>

## 延伸阅读

这里有一些相关的文章, 包括以下内容:

*   [**TPU 深度解析**](https://henryhmko.github.io/posts/tpu/tpu.html): 一篇非常精彩的文章, 秉承本书的精神, 深入探讨了 TPU 架构.
*   [**从第一性原理让深度学习飞速发展 (Making Deep Learning Go Brrrr From First Principles)**](https://horace.io/brrr_intro.html): 一篇更侧重于 GPU 和 PyTorch 的教程, 讲解了 LLM 的 roofline 模型和性能工程.
*   [**使用 Pallas 编写 TPU 核函数**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): 如今, TPU 编程越来越多地涉及使用 Pallas 编写自定义核函数. 这个系列讨论了如何编写核函数以及许多这里未提及的底层 TPU 细节.
*   [**如何优化 CUDA Matmul 核函数以达到类 cuBLAS 的性能: 工作日志**](https://siboehm.com/articles/22/CUDA-MMM): 虽然特定于 GPU 和 CUDA, 但这是一篇优秀的博客文章, 展示了如何在 CUDA 中优化矩阵乘法核函数. 这可能是深入了解 TPU 和 GPU 区别的好材料.
*   [**分布式数组和自动并行化**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): 这是 JAX 中并行 API 的一个很好的指南, 也是学习如何实际实现我们在此讨论的一些想法的好方法.
*   [**Rafi Witten 的 2024 年高性能 LLM 课程**](https://github.com/rwitten/HighPerfLLMs2024): 我们的前同事 Rafi 开设了一门关于 TPU 性能工程的精彩课程, 所有幻灯片都在 GitHub 上. 这门课程比我们这里更深入地涵盖了许多内容.
*   [**\[2211.05102\] 高效扩展 Transformer 推理**](https://arxiv.org/abs/2211.05102): 一篇关于 Transformer 推理数学的详细论文. 这是本文档许多内容的灵感来源.
*   [**Huggingface 超大规模实践手册**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): 可谓是本书的 GPU 版本, 更深入地探讨了 PyTorch 如何在训练期间实现并行技术和节省内存的技术.
*   [**Transformer 推理算术**](https://kipp.ly/transformer-inference-arithmetic/): 一篇与本书有许多相同想法的博客, 并配有一些优秀的插图.
*   [**斯坦福 CS336 幻灯片和视频**](https://stanford-cs336.github.io/spring2025/index.html#coursework): 一门非常棒的斯坦福课程, 涵盖了 LLM 训练和服务的许多细节, 并提供了一些有用的练习. 作业 1 和 2 特别相关.
*   [**Stas Bekman 的机器学习工程手册**](https://github.com/stas00/ml-engineering): 一本非常实用的机器学习基础设施指南, 涵盖了本书未涉及的主题, 例如如何与云提供商谈判、集群管理以及 GPU 吞吐量的实证测量.

这个领域仍有很大的综合性写作空间, 所以我们希望这份手稿能鼓励更多这样的作品出现! 我们也相信这是一个值得学习和研究的富有成果的领域. 在许多情况下, 即使手头没有很多硬件加速器, 也可以进行研究.

## 反馈

请留下评论或问题, 以便我们进一步改进. 您可以通过 jaaustin [at] google [dot] com 联系我们的通讯作者 Jacob Austin, 或通过在 [GitHub](https://github.com/jax-ml/scaling-book) 上发布 issue, pull request 或 discussion 来建议编辑.