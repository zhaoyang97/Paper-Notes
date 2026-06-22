---
title: >-
  ACL2025 计算生物论文汇总 · 6篇论文解读
description: >-
  6篇ACL2025的计算生物方向论文解读，涵盖生物分子、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "计算生物"
  - "论文解读"
  - "论文笔记"
  - "生物分子"
  - "对齐/RLHF"
item_list:
  - u: "align-pro_align_protein_representations_through_multi-modal_learning/"
    t: "Align-Pro: Align Protein Representations Through Multi-Modal Learning"
  - u: "concept_bottleneck_language_models_for_protein_design/"
    t: "Concept Bottleneck Language Models For Protein Design"
  - u: "foundation_lm_single_cell_survey/"
    t: "A Survey on Foundation Language Models for Single-cell Biology"
  - u: "kpo_protein_safety/"
    t: "Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization"
  - u: "ladder_language-driven_slice_discovery_and_error_rectification_in_vision_classif/"
    t: "LADDER: Language Driven Slice Discovery and Error Rectification in Vision Classifiers"
  - u: "retrieve_to_explain_drug_target_identification/"
    t: "Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧬 计算生物

**💬 ACL2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (21)](../../CVPR2026/computational_biology/index.md) · [🔬 ICLR2026 (155)](../../ICLR2026/computational_biology/index.md) · [💬 ACL2026 (5)](../../ACL2026/computational_biology/index.md) · [🧪 ICML2026 (51)](../../ICML2026/computational_biology/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/computational_biology/index.md) · [🧠 NeurIPS2025 (76)](../../NeurIPS2025/computational_biology/index.md)

🔥 **高频主题：** 生物分子 ×4

**[Align-Pro: Align Protein Representations Through Multi-Modal Learning](align-pro_align_protein_representations_through_multi-modal_learning.md)**

:   Align-Pro通过多模态对比学习框架，将蛋白质的序列、结构和功能描述三种模态的表示对齐到统一的嵌入空间中，从而实现跨模态的蛋白质检索、分类和功能预测。

**[Concept Bottleneck Language Models For Protein Design](concept_bottleneck_language_models_for_protein_design.md)**

:   本文将概念瓶颈模型（Concept Bottleneck Model）的可解释性设计理念引入蛋白质语言模型，通过中间层的生物学概念作为瓶颈，实现既能设计功能性蛋白质序列又能提供人类可理解的设计理由的蛋白质生成系统。

**[A Survey on Foundation Language Models for Single-cell Biology](foundation_lm_single_cell_survey.md)**

:   首篇从语言建模视角系统综述单细胞生物学基础语言模型，将现有工作划分为PLM（从头预训练）和LLM（利用已有大模型）两大类，全面分析tokenization策略、预训练/微调范式以及下游任务体系，并指出当前领域在数据质量、统一评测和scaling law方面的核心挑战。

**[Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](kpo_protein_safety.md)**

:   提出KPO框架，通过构建蛋白质安全知识图谱(PSKG)并结合加权图剪枝策略识别"相似但安全"的蛋白质对，用DPO微调蛋白质语言模型使其远离有害序列空间，同时保持功能性。

**[LADDER: Language Driven Slice Discovery and Error Rectification in Vision Classifiers](ladder_language-driven_slice_discovery_and_error_rectification_in_vision_classif.md)**

:   LADDER 把预训练视觉分类器的内部激活"翻译"成自然语言、检索出与错误相关的句子，再让 LLM 据此推理出"模型在缺少哪个属性时会犯错"的可检验假设，从而无需任何属性标注就能发现并缓解任意现成分类器的多重偏见；在 6 个自然/医学数据集、200+ 分类器上一致超过 Domino/Facts/DFR 等基线。

**[Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification](retrieve_to_explain_drug_target_identification.md)**

:   提出 R2E (Retrieve to Explain)，一种基于检索的架构，通过从文献语料库中检索证据来评分和排序所有候选答案，并利用 Shapley 值将预测忠实地归因到支撑证据，在药物靶点识别任务上超越了遗传学基线和 GPT-4 基线。
