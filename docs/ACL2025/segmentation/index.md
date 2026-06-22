---
title: >-
  ACL2025 语义分割论文汇总 · 4篇论文解读
description: >-
  4篇ACL2025的语义分割方向论文解读，涵盖语义分割、推理、对话系统等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "语义分割"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "对话系统"
item_list:
  - u: "bert-like_models_for_slavic_morpheme_segmentation/"
    t: "BERT-like Models for Slavic Morpheme Segmentation"
  - u: "def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation/"
    t: "DEF-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation"
  - u: "instructpart_task-oriented_part_segmentation_with_instruction_reasoning/"
    t: "InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning"
  - u: "pixel-level_reasoning_segmentation_via_multi-turn_conversations/"
    t: "Pixel-Level Reasoning Segmentation via Multi-turn Conversations"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**💬 ACL2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (122)](../../CVPR2026/segmentation/index.md) · [🔬 ICLR2026 (31)](../../ICLR2026/segmentation/index.md) · [🧪 ICML2026 (14)](../../ICML2026/segmentation/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/segmentation/index.md) · [🧠 NeurIPS2025 (45)](../../NeurIPS2025/segmentation/index.md) · [📹 ICCV2025 (73)](../../ICCV2025/segmentation/index.md)

🔥 **高频主题：** 语义分割 ×4 · 推理 ×3 · 对话系统 ×2

**[BERT-like Models for Slavic Morpheme Segmentation](bert-like_models_for_slavic_morpheme_segmentation.md)**

:   本文探索使用 BERT 类预训练语言模型来完成斯拉夫语系语言的形态素分割任务，通过将形态素分割建模为序列标注问题，在多个斯拉夫语言上取得了优于传统方法的结果。

**[DEF-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation.md)**

:   提出 DEF-DTS，一种基于 LLM 多步演绎推理的对话话题分割方法——通过双向上下文摘要 → 话语意图分类（5 类） → 演绎话题转移判断三步 pipeline，在 TIAGE、SuperDialseg、Dialseg711 三个数据集上取得无监督/prompt 方法 SOTA，在 Dialseg711 上超越监督方法。

**[InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)**

:   提出 InstructPart，首个将任务导向指令与部件级分割结合的真实世界 benchmark——2400 张图像、48 类物体、44 类部件、9600 条人工标注的任务指令，评估发现当前 VLM 在指令驱动的部件分割上严重不足，基于 LISA+DINOv2 的 baseline 微调后性能提升约 100%。

**[Pixel-Level Reasoning Segmentation via Multi-turn Conversations](pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)**

:   提出像素级推理分割 (Pixel-level RS) 新任务，通过多轮对话逐步理解用户意图实现细粒度分割，构建了包含 24k 对话轮次的 PRIST 数据集，并设计 MIRAS 框架在分割精度和推理能力上均超越现有基线。
