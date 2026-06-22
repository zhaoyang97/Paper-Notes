---
title: >-
  ICCV2025 信息检索/RAG论文汇总 · 5篇论文解读
description: >-
  5篇ICCV2025的信息检索/RAG 方向论文解读，涵盖持续学习、RAG等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "RAG"
item_list:
  - u: "aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe/"
    t: "Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation"
  - u: "external_knowledge_injection_for_clip-based_class-incremental_learning/"
    t: "External Knowledge Injection for CLIP-Based Class-Incremental Learning"
  - u: "langbridge_interpreting_image_as_a_combination_of_language_embeddings/"
    t: "LangBridge: Interpreting Image as a Combination of Language Embeddings"
  - u: "monster_a_unified_model_for_motion_scene_text_retrieval/"
    t: "MonSTeR: a Unified Model for Motion, Scene, Text Retrieval"
  - u: "ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge/"
    t: "OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**📹 ICCV2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🧪 ICML2026 (26)](../../ICML2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [🧪 ICML2025 (6)](../../ICML2025/information_retrieval/index.md)

**[Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation](aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)**

:   提出D2S-VSE框架，通过两阶段训练（稠密文本预训练+稠密到稀疏特征蒸馏微调）增强视觉语义嵌入的信息容量，解决图文匹配中图像与文本信息密度不对称的核心问题。

**[External Knowledge Injection for CLIP-Based Class-Incremental Learning](external_knowledge_injection_for_clip-based_class-incremental_learning.md)**

:   提出 Engine（ExterNal knowledGe INjEction）框架，通过双分支注入调优（视觉分支用数据增强、文本分支用 GPT-4 生成判别性描述）和推理时后调优知识注入（成对判别特征重排序），在无需存储历史样本的条件下，在 9 个基准数据集上以 3-10% 的优势超越所有 CLIP-based 类增量学习方法。

**[LangBridge: Interpreting Image as a Combination of Language Embeddings](langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)**

:   LangBridge 通过将视觉特征显式分解为 LLM 词汇嵌入的线性组合，实现了可解释的视觉-语言对齐，并支持跨 LLM 的预训练无关适配器迁移。

**[MonSTeR: a Unified Model for Motion, Scene, Text Retrieval](monster_a_unified_model_for_motion_scene_text_retrieval.md)**

:   提出 **MonSTeR**——首个**运动-场景-文本三模态检索模型**，通过受拓扑深度学习启发的高阶关系建模，构建统一隐空间以捕获三模态之间的内在依赖关系，在多项检索任务上大幅超越仅依赖单模态表征的基线，并可用于人-场景交互模型的评估。

**[OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge.md)**

:   提出 OHRBench——首个评估 OCR 对 RAG 系统级联影响的基准，包含 7 个领域的 8561 张文档图像和 8498 个 QA 对，系统性地揭示了 OCR 产生的语义噪声（Semantic Noise）和格式噪声（Formatting Noise）对检索和生成两阶段的不同影响模式。
