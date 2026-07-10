---
title: >-
  ECCV2026 预训练论文汇总 · 2篇论文解读
description: >-
  2篇ECCV2026的预训练方向论文解读，涵盖扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
item_list:
  - u: "360anything_geometry-free_lifting_of_images_and_videos_to_360/"
    t: "360Anything: Geometry-Free Lifting of Images and Videos to 360°"
  - u: "gryphone_symbol-aware_masked_diffusion_for_structural_refinement_in_offline_hand/"
    t: "GryphOne: Symbol-Aware Masked Diffusion for Structural Refinement in Offline Handwritten Mathematical Expression Recognition"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**🎞️ ECCV2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md)

**[360Anything: Geometry-Free Lifting of Images and Videos to 360°](360anything_geometry-free_lifting_of_images_and_videos_to_360.md)**

:   360Anything 提出一种几何无关（geometry-free）的扩散Transformer框架，将透视输入和全景目标统一视为 token 序列、通过序列拼接让模型自行学习二者间的几何对应，无需任何相机元数据即可将任意视角图像/视频提升为重力对齐的无接缝 360° 全景图，在图像和视频两个任务上全面超越依赖真实相机参数的此前最优方法。

**[GryphOne: Symbol-Aware Masked Diffusion for Structural Refinement in Offline Handwritten Mathematical Expression Recognition](gryphone_symbol-aware_masked_diffusion_for_structural_refinement_in_offline_hand.md)**

:   GryphOne 将手写数学表达式识别（HMER）从自回归序列生成重新定义为离散掩码扩散的迭代符号精炼过程，通过符号感知分词（SAT）保持局部编辑的语法一致性，并用随机掩码互学习（RMML）提升精炼稳定性，在 MathWriting 上以 5.51% CER 和 59.9% ExpRate 全面超越重实现的自回归基线及商用 HMER 系统。
