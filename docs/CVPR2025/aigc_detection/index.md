---
title: >-
  CVPR2025 AIGC检测论文汇总 · 3篇论文解读
description: >-
  3篇CVPR2025的 AIGC 检测方向论文解读，涵盖持续学习、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "少样本学习"
item_list:
  - u: "enhancing_few-shot_class-incremental_learning_via_training-free_bi-level_modalit/"
    t: "Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration"
  - u: "proapo_progressively_automatic_prompt_optimization_for_visual_classification/"
    t: "ProAPO: Progressively Automatic Prompt Optimization for Visual Classification"
  - u: "sgc-net_stratified_granular_comparison_network_for_open-vocabulary_hoi_detection/"
    t: "SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**📷 CVPR2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (10)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/aigc_detection/index.md) · [💬 ACL2026 (17)](../../ACL2026/aigc_detection/index.md) · [🧪 ICML2026 (11)](../../ICML2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md)

**[Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration](enhancing_few-shot_class-incremental_learning_via_training-free_bi-level_modalit.md)**

:   提出 BiMC（Bi-level Modality Calibration）框架，基于冻结 CLIP 模型，通过模态内校准（结合 LLM 生成的细粒度类别描述与视觉原型）和模态间校准（融合预训练语言知识与任务特定视觉先验），在无需任何参数训练的情况下实现 FSCIL SOTA，在 CIFAR-100 上超越最优对比方法 4.25%。

**[ProAPO: Progressively Automatic Prompt Optimization for Visual Classification](proapo_progressively_automatic_prompt_optimization_for_visual_classification.md)**

:   提出 ProAPO，一种基于进化算法的渐进式自动提示优化方法，在仅需 one-shot 监督且无需人工参与的条件下，从任务级模板逐步优化到类别级描述，解决 LLM 生成描述中的幻觉和缺乏区分度问题，在 13 个数据集上超越现有文本提示方法。

**[SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection](sgc-net_stratified_granular_comparison_network_for_open-vocabulary_hoi_detection.md)**

:   提出分层粒度比较网络SGC-Net，通过粒度感知对齐(GSA)模块聚合CLIP多层视觉特征，并利用层级分组比较(HGC)模块借助LLM递归生成区分性描述，解决开放词汇HOI检测中的特征粒度不足和语义混淆问题。
