---
title: >-
  AAAI2026 知识编辑论文汇总 · 4篇论文解读
description: >-
  4篇AAAI2026的知识编辑方向论文解读，涵盖多模态、推理、问答、对抗鲁棒、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "知识编辑"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "推理"
  - "问答"
  - "对抗鲁棒"
  - "Agent"
item_list:
  - u: "hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph/"
    t: "Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing"
  - u: "is_the_information_bottleneck_robust_enough_towards_label-noise_resistant_inform/"
    t: "Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning"
  - u: "model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben/"
    t: "Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior"
  - u: "multiplicative_orthogonal_sequential_editing_for_language_models/"
    t: "Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**🤖 AAAI2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/knowledge_editing/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/knowledge_editing/index.md) · [💬 ACL2026 (10)](../../ACL2026/knowledge_editing/index.md) · [🧪 ICML2026 (8)](../../ICML2026/knowledge_editing/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/knowledge_editing/index.md) · [🧪 ICML2025 (2)](../../ICML2025/knowledge_editing/index.md)

**[Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing](hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph.md)**

:   提出MMQAKE基准和Hybrid-DMKG框架，在动态多模态知识图谱上构建"关系链接预测 + RAG增强LVLM推理"双通道混合推理机制，配合背景反思决策模块，在2-5跳多模态知识编辑问答中显著超越现有方法（LLaVA上H-Acc达29.90%，超IKE 13.52个百分点）。

**[Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning](is_the_information_bottleneck_robust_enough_towards_label-noise_resistant_inform.md)**

:   本文揭示了信息瓶颈（IB）原理在标签噪声下的固有脆弱性，提出 LaT-IB 方法，通过将表征解耦为干净标签空间和噪声标签空间两部分，结合"最小-充分-干净"（MSC）准则和三阶段训练框架，在多种噪声条件下实现了对现有 IB 方法的显著超越。

**[Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)**

:   将 Agent 伦理行为引导建模为模型编辑任务（Behavior Editing），提出基于心理学道德理论的三层 BehaviorBench 基准，在 9 个开源模型和 20 个闭源模型上验证了模型编辑可以精确地将 Agent 引导向善意或恶意方向，且单次编辑可导致全局道德对齐偏移。

**[Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](multiplicative_orthogonal_sequential_editing_for_language_models.md)**

:   提出 MOSE（乘法正交序列编辑），用正交矩阵左乘（而非加法更新）参数矩阵来注入新知识，严格保持编辑后矩阵的范数和条件数不变，在序列编辑中实现 12.08% 的性能提升并保留 95.73% 通用能力。
