---
title: >-
  CVPR2026 信号/通信论文汇总 · 2篇论文解读
description: >-
  2篇CVPR2026的信号/通信方向论文解读，涵盖多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
  - "多模态"
item_list:
  - u: "actta_rethinking_test-time_adaptation_via_dynamic_activation/"
    t: "AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation"
  - u: "clay_conditional_visual_similarity/"
    t: "CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**📷 CVPR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (7)](../../ICLR2026/signal_comm/index.md) · [🧪 ICML2026 (2)](../../ICML2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/signal_comm/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/signal_comm/index.md) · [🧪 ICML2025 (3)](../../ICML2025/signal_comm/index.md)

**[AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](actta_rethinking_test-time_adaptation_via_dynamic_activation.md)**

:   本文提出 AcTTA，一种基于动态激活函数调制的测试时自适应框架，通过将传统固定激活函数重参数化为可学习形式（包含激活中心偏移和非对称梯度斜率），在推理时自适应调整激活行为以应对分布偏移，在 CIFAR10-C/CIFAR100-C/ImageNet-C 上一致超越基于归一化层的 TTA 方法。

**[CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](clay_conditional_visual_similarity.md)**

:   CLAY 提出免训练的条件视觉相似度计算方法，通过在 VLM 嵌入空间中构建文本条件子空间来调制相似度，无需重新计算数据库特征即可适应不同检索条件，并支持多条件检索。
