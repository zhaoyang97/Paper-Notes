---
title: >-
  CVPR2025 LLM评测论文汇总 · 4篇论文解读
description: >-
  4篇CVPR2025的 LLM 评测方向论文解读，涵盖扩散模型、布局/合成、导航、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "布局/合成"
  - "导航"
  - "少样本学习"
item_list:
  - u: "erase_diffusion_empowering_object_removal_through_calibrating_diffusion_pathways/"
    t: "Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)"
  - u: "postero_structuring_layout_trees_to_enable_language_models_in_generalized_conten/"
    t: "PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation"
  - u: "roadsocial_a_diverse_videoqa_dataset_and_benchmark_for_road_event_understanding_/"
    t: "RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives"
  - u: "unigoal_towards_universal_zero-shot_goal-oriented_navigation/"
    t: "UniGoal: Towards Universal Zero-shot Goal-oriented Navigation"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**📷 CVPR2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (131)](../../ICLR2026/llm_evaluation/index.md) · [💬 ACL2026 (96)](../../ACL2026/llm_evaluation/index.md) · [🧪 ICML2026 (40)](../../ICML2026/llm_evaluation/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (38)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md)

**[Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)](erase_diffusion_empowering_object_removal_through_calibrating_diffusion_pathways.md)**

:   本文提出EraDiff，通过链式校正优化范式（CRO）建立从"含物体"到"纯背景"的渐进扩散路径，并用自校正注意力机制（SRA）在采样时抑制伪影，使扩散模型真正理解"擦除意图"，在OpenImages V5上取得SOTA的Local FID（3.799），在复杂真实场景中显著优于SD2-Inpaint和LaMa。

**[PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation](postero_structuring_layout_trees_to_enable_language_models_in_generalized_conten.md)**

:   提出 PosterO，将海报版面结构化为 SVG 布局树，通过设计意图向量化和层次节点表示实现与 LLM 的对接，利用意图对齐的上下文学习生成高质量内容感知版面，在多个基准上达到 SOTA 并引入首个支持多用途和多形状元素的 PStylish7 数据集。

**[RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives](roadsocial_a_diverse_videoqa_dataset_and_benchmark_for_road_event_understanding_.md)**

:   本文提出RoadSocial，一个来源于社交媒体的大规模多样化VideoQA数据集（13.2K视频、260K问答对），覆盖全球多地域多视角的道路事件场景，通过半自动标注框架和12类QA任务系统性评测了18种Video LLM的道路事件理解能力。

**[UniGoal: Towards Universal Zero-shot Goal-oriented Navigation](unigoal_towards_universal_zero-shot_goal-oriented_navigation.md)**

:   提出 UniGoal 统一零样本目标导航框架，通过将场景和目标统一表示为图结构，结合图匹配驱动的多阶段探索策略，在单一模型中实现对象类别、实例图像和文本描述三种目标类型的零样本导航，性能超越任务专用方法。
