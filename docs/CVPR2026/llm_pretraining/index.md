---
title: >-
  CVPR2026 预训练论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2026的预训练方向论文解读，涵盖少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
item_list:
  - u: "exploring_visual_pretraining_for_learning_language_intelligence/"
    t: "Exploring Visual Pretraining for Learning Language Intelligence"
  - u: "linking_modality_isolation_in_heterogeneous_collaborative_perception/"
    t: "Linking Modality Isolation in Heterogeneous Collaborative Perception"
  - u: "reconstructing_clip_for_open-vocabulary_dense_perception/"
    t: "Reconstructing CLIP for Open-Vocabulary Dense Perception"
  - u: "unlocking_pre-trained_weights_parameter_inheritance_for_zero-shot_initialization/"
    t: "Unlocking Pre-trained Weights: Parameter Inheritance for Zero-Shot Initialization"
  - u: "watch_and_learn_learning_to_use_computers_from_online_videos/"
    t: "Watch and Learn: Learning to Use Computers from Online Videos"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**📷 CVPR2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

**[Exploring Visual Pretraining for Learning Language Intelligence](exploring_visual_pretraining_for_learning_language_intelligence.md)**

:   这篇论文提出 MAPLE：不把 PDF 抽成文本喂给 LLM，而是直接拿文档**页面图像**做掩码自回归预训练，让 LLM 通过"为遮挡区域生成隐式假设"来学语言智能，在四个数学推理基准上相对纯文本预训练平均提升至多 40.2%。

**[Linking Modality Isolation in Heterogeneous Collaborative Perception](linking_modality_isolation_in_heterogeneous_collaborative_perception.md)**

:   提出 CodeAlign 框架，通过码本构建离散代码空间和跨模态 Feature-Code-Feature (FCF) 翻译，首次解决异构协同感知中不同模态从未在训练数据中共现的"模态隔离"问题，仅需 HEAL 8% 训练参数、通信量降低 1024 倍，同时达到 SOTA 感知性能。

**[Reconstructing CLIP for Open-Vocabulary Dense Perception](reconstructing_clip_for_open-vocabulary_dense_perception.md)**

:   DenseRC 针对"如何为 CLIP 构建好的密集特征"这一被忽视的问题，揭示 cls token 的泛化语义其实来自多层 value embedding、而空间聚合会放大语义错位，于是用多层 value 作基底、配一个轻量的头选择门控（HSG）只在 head 维重加权，造出与全局语义对齐的密集表示，在开放词汇检测和分割多个基准上刷新 SOTA。

**[Unlocking Pre-trained Weights: Parameter Inheritance for Zero-Shot Initialization](unlocking_pre-trained_weights_parameter_inheritance_for_zero-shot_initialization.md)**

:   PITH 用图超网络给目标网络动态生成「投影矩阵」，把预训练大模型的内部权重直接投影到任意尺寸的目标 ViT 上完成初始化，使得初始化后的网络无需任何训练就能直接用——在 ImageNet-1K 上 ViT-Base 零样本精度 53.35%，比上一代 SOTA（TAL）高 6.54%。

**[Watch and Learn: Learning to Use Computers from Online Videos](watch_and_learn_learning_to_use_computers_from_online_videos.md)**

:   提出 Watch & Learn (W&L) 框架，通过逆动力学模型 (IDM) 将互联网上的人类计算机操作视频自动转化为可执行的 UI 轨迹数据，生成 53K+ 高质量轨迹，作为 ICL 示例或 SFT 训练数据显著提升各类 CUA 性能。
