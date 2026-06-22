---
title: >-
  ECCV2024 信息检索/RAG论文汇总 · 3篇论文解读
description: >-
  3篇ECCV2024的信息检索/RAG 方向论文解读，涵盖布局/合成、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
  - "LLM"
item_list:
  - u: "multi-label_cluster_discrimination_for_visual_representation_learning/"
    t: "Multi-Label Cluster Discrimination for Visual Representation Learning"
  - u: "onerestore_a_universal_restoration_framework_for_composite_degradation/"
    t: "OneRestore: A Universal Restoration Framework for Composite Degradation"
  - u: "towards_open-ended_visual_recognition_with_large_language_models/"
    t: "Towards Open-Ended Visual Recognition with Large Language Model"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🎞️ ECCV2024** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🧪 ICML2026 (26)](../../ICML2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md)

**[Multi-Label Cluster Discrimination for Visual Representation Learning](multi-label_cluster_discrimination_for_visual_representation_learning.md)**

:   提出多标签聚类判别方法 MLCD，通过为每张图像分配多个聚类伪标签并设计消歧多标签分类损失，在 LAION-400M 上预训练的 ViT 在 linear probe、zero-shot 分类和检索任务上全面超越 OpenCLIP、FLIP 和 UNICOM。

**[OneRestore: A Universal Restoration Framework for Composite Degradation](onerestore_a_universal_restoration_framework_for_composite_degradation.md)**

:   提出 OneRestore，一种基于 Transformer 的通用图像复原框架，通过场景描述符引导的交叉注意力机制和复合退化复原损失，能在单一模型中自适应地处理低光照、雾、雨、雪及其任意组合的复合退化场景，并支持文本/视觉双模式的可控复原。

**[Towards Open-Ended Visual Recognition with Large Language Model](towards_open-ended_visual_recognition_with_large_language_models.md)**

:   提出 OmniScient Model (OSM)——一个基于冻结 CLIP-ViT + 可训练 MaskQ-Former + 冻结 LLM (Vicuna-7B) 的生成式 mask 分类器，将视觉识别从"从预定义词表中选择类别"转变为"直接生成类别名称"，消除了训练和测试时对预定义词表的依赖，在 COCO 全景分割上超越 DaTaSeg +4.3 PQ。
