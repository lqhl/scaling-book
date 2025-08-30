---
layout: distill
title: "结论和进一步阅读"
# permalink: /main/
description: "感谢您的阅读！在这里我们将包含一些更多参考资料供进一步学习。"
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
  - name: "Acknowledgments"
  - name: "Further Reading"
  - name: "Feedback"

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
**感谢您阅读这一系列文章，并恭喜您坚持到了最后。** 在结束之前，我们想表达一些感谢：

## 致谢

本文档代表了 Google DeepMind 许多人的巨大集体投入，我们想在此简要感谢他们！

* James Bradbury、Reiner Pope 和 Blake Hechtman 最初推导了本文稿中的许多想法，并且很早就理解了 Transformer 的系统视角。
* Sholto Douglas 撰写了本文档的第一个版本，负责启动这个项目。他比任何人都更负责本文档的整体叙事。
* Jacob Austin 领导了将第一个版本从粗略笔记转变为更精致全面文档的工作。他完成了大部分编辑、格式化和发布本文档的工作，并协调了其他作者的贡献。
* 大部分图表和动画由 Anselm Levskaya 和 Charlie Chen 制作。
* Charlie Chen 撰写了推理部分，并绘制了许多推理相关的图表。
* Roy Frostig 在出版、编辑和许多其他步骤中提供了帮助。

我们还要感谢许多在此过程中提供关键反馈的人，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alex Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 等许多人。感谢 Ruiqi Gao 在 HTML 格式化方面的帮助。

**感谢大家！**

<p markdown=1 class="announce">在离开之前，您可能还想阅读新的[第 12 节](../gpus)关于 NVIDIA GPU 的内容！</p>

## 进一步阅读

有许多相关的文章和资料，包括以下内容：

* [**TPU 深度解析**](https://henryhmko.github.io/posts/tpu/tpu.html)：一篇精彩的深入探讨 TPU 架构的文章，风格与本书类似。
* [**从第一原理开始让深度学习运行更快**](https://horace.io/brrr_intro.html)：一篇更专注于 GPU 和 PyTorch 的教程，介绍 LLM 的性能上限和性能工程。
* [**使用 Pallas 编写 TPU 内核**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：越来越多的 TPU 编程涉及使用 Pallas 编写自定义内核。这个系列讨论了如何编写内核以及许多这里未提及的更低级别的 TPU 细节。
* [**如何优化 CUDA 矩阵乘法内核以达到 cuBLAS 级性能：工作日志**](https://siboehm.com/articles/22/CUDA-MMMM)：虽然是针对 GPU 和 CUDA 的，但这是一篇出色的博客文章，展示了如何在 CUDA 中优化矩阵乘法内核。这可能是一个深入了解 TPU 和 GPU 差异的好方法。
* [**分布式数组和自动并行化**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：这是一个很好的 JAX 并行化 API 指南，是学习如何实际实现我们在这里讨论的一些想法的好方法。
* [**Rafi Witten 的高性能 LLM 2024 课程**](https://github.com/rwitten/HighPerfLLMs2024)：我们的前同事 Rafi 提供了一个关于 TPU 性能工程的优秀课程，幻灯片都在 GitHub 上。这涵盖了许多比我们这里更深入的内容。
* [**[2211.05102] 高效扩展 Transformer 推理**](https://arxiv.org/abs/2211.05102)：一篇关于 Transformer 推理数学的详细论文。这是本文档许多内容的灵感来源。
* [**Huggingface 超大规模剧本**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：这本书的 GPU 类似物，更深入地讨论了 PyTorch 如何在训练期间实现并行化技术和内存节省技术。
* [**Transformer 推理算术**](https://kipp.ly/transformer-inference-arithmetic/)：一个包含与本书许多相同想法的博客，还有一些优秀的插图。
* [**斯坦福 CS336 幻灯片和视频**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：一个精彩的斯坦福课程，涵盖 LLM 训练和服务的许多细节，还有一些有用的练习。作业 1 和 2 特别相关。
* [**Stas Bekman 的 ML 工程手册**](https://github.com/stas00/ml-engineering)：一个非常实用的 ML 基础设施指南，涵盖了本书中未涉及的主题，如如何与云提供商谈判、集群管理和 GPU 吞吐量的经验测量。

在这个领域仍然有很多全面的写作空间，因此我们希望这份文稿能够鼓励更多这样的写作！我们也相信这是一个富有成果的研究和学习领域。在许多情况下，即使手头没有许多硬件加速器，也可以进行研究。

## 反馈

请留下评论或问题，以便我们进一步改进。您可以通过 jaaustin [at] google [dot] com 联系我们的通讯作者 Jacob Austin，或者通过在 [GitHub](https://github.com/jax-ml/scaling-book) 上发布问题、拉取请求或讨论来建议编辑。
