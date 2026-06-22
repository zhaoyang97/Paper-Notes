---
title: >-
  NeurIPS2025 知识编辑论文汇总 · 6篇论文解读
description: >-
  6篇NeurIPS2025的知识编辑方向论文解读，涵盖布局/合成、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "知识编辑"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
  - "LLM"
item_list:
  - u: "edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit/"
    t: "Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs"
  - u: "kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models/"
    t: "KScope: A Framework for Characterizing the Knowledge Status of Language Models"
  - u: "memeic_a_step_toward_continual_and_compositional_knowledge_editing/"
    t: "MemEIC: A Step Toward Continual and Compositional Knowledge Editing"
  - u: "memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_/"
    t: "MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs"
  - u: "rethinking_residual_distribution_in_locate-then-edit_model_editing/"
    t: "Rethinking Residual Distribution in Locate-then-Edit Model Editing"
  - u: "uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models/"
    t: "UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🧠 NeurIPS2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/knowledge_editing/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/knowledge_editing/index.md) · [💬 ACL2026 (10)](../../ACL2026/knowledge_editing/index.md) · [🧪 ICML2026 (8)](../../ICML2026/knowledge_editing/index.md) · [🤖 AAAI2026 (4)](../../AAAI2026/knowledge_editing/index.md) · [🧪 ICML2025 (2)](../../ICML2025/knowledge_editing/index.md)

**[Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)**

:   提出 NMKE 框架，通过神经元级归因发现 knowledge-general 和 knowledge-specific 两类知识神经元，并结合熵引导的动态稀疏 mask，实现精准神经元级知识编辑，在 5000 步连续编辑后仍保持高编辑成功率和模型通用能力。

**[KScope: A Framework for Characterizing the Knowledge Status of Language Models](kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)**

:   提出LLM知识状态的五分类法（一致正确/冲突正确/缺失/冲突错误/一致错误）和KScope层次化统计检验框架，通过重复采样+多步假设检验精确刻画LLM对给定问题的知识模式结构，并系统研究上下文如何更新各状态，发现受约束的上下文摘要+增强可信度平均提升4.3%的知识更新成功率。

**[MemEIC: A Step Toward Continual and Compositional Knowledge Editing](memeic_a_step_toward_continual_and_compositional_knowledge_editing.md)**

:   提出 MemEIC 框架，通过外部双模态检索记忆 + 内部模态分离 LoRA 适配器 + 仿脑 Knowledge Connector 三层架构，实现大视觉语言模型的持续、组合式知识编辑，在新提出的 CCKEB 基准上大幅超越现有方法。

**[MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)**

:   提出MEMOIR框架，通过在FFN层引入零初始化的残差记忆矩阵，利用基于TopHash的稀疏掩码将每次编辑限制在记忆参数的不同子集上，推理时通过掩码重叠率识别相关编辑并条件性激活知识，在15000次连续编辑下仍保持可靠性、泛化性和局部性的最优平衡。

**[Rethinking Residual Distribution in Locate-then-Edit Model Editing](rethinking_residual_distribution_in_locate-then-edit_model_editing.md)**

:   揭示 locate-then-edit 模型编辑中残差分配（residual distribution）机制引入的权重偏移误差会随分配距离、batch 大小和编辑序列长度增长，提出 BLUE（Boundary Layer UpdatE）策略仅更新首尾关键层，平均提升 35.59%。

**[UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)**

:   构建 UniEdit——首个基于开放域知识图谱（Wikidata）的统一 LLM 知识编辑基准，覆盖 5 大类 25 个领域共 311K 条样本，通过邻域多跳链采样（NMCS）算法统一整合多种泛化性和局部性评估标准，系统揭示了现有编辑方法在复杂波纹效应评估下的不足。
