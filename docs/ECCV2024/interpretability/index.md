---
title: >-
  ECCV2024 可解释性论文汇总 · 5篇论文解读
description: >-
  5篇ECCV2024的可解释性方向论文解读，涵盖对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "可解释性"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
item_list:
  - u: "detailsemnet_elevating_signature_verification_through_detail-semantic_integratio/"
    t: "DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration"
  - u: "egoexo-fitness_towards_egocentric_and_exocentric_full-body_action_understanding/"
    t: "EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding"
  - u: "improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo/"
    t: "Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models"
  - u: "plot_text-based_person_search_with_part_slot_attention_for_corresponding_part_di/"
    t: "PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery"
  - u: "poa_pre-training_once_for_models_of_all_sizes/"
    t: "POA: Pre-training Once for Models of All Sizes"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**🎞️ ECCV2024** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (34)](../../CVPR2026/interpretability/index.md) · [🔬 ICLR2026 (195)](../../ICLR2026/interpretability/index.md) · [💬 ACL2026 (63)](../../ACL2026/interpretability/index.md) · [🧪 ICML2026 (91)](../../ICML2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (80)](../../NeurIPS2025/interpretability/index.md)

**[DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration](detailsemnet_elevating_signature_verification_through_detail-semantic_integratio.md)**

:   提出DetailSemNet用于离线签名验证，通过Detail-Semantics Integrator将特征解耦为细节和语义两个分支分别处理，并引入基于EMD的局部结构匹配，在多个多语言签名数据集上取得SOTA。

**[EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding](egoexo-fitness_towards_egocentric_and_exocentric_full-body_action_understanding.md)**

:   提出 EgoExo-Fitness 数据集，包含同步的第一人称和第三人称健身视频，提供两级时间边界标注和创新性的可解释动作评判标注（技术关键点验证、自然语言评论、质量评分），并构建五个基准任务。

**[Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)**

:   本文发现 Concept Bottleneck Models (CBMs) 中人工干预效率低下的原因在于干预时各概念独立处理、忽视了概念间关联，提出了一个轻量级的 Concept Intervention Realignment Module (CIRM)，在干预后自动重新对齐相关概念的预测值，将达到目标性能所需的干预次数最多减少 70%。

**[PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery](plot_text-based_person_search_with_part_slot_attention_for_corresponding_part_di.md)**

:   提出 PLOT 框架，利用基于 Slot Attention 的 Part Discovery Module 自动发现跨模态（图像-文本）对应的人体部件，结合 Text-based Dynamic Part Attention（TDPA）动态调整各部件重要性，无需部件级标注即可在三个 benchmark 上全面超越 SOTA。

**[POA: Pre-training Once for Models of All Sizes](poa_pre-training_once_for_models_of_all_sizes.md)**

:   POA 提出在自监督自蒸馏框架中引入**弹性学生分支**，通过参数共享和随机子网络采样，**一次预训练即可同时产出上百个不同大小的预训练模型**（如从 ViT-L 直接提取 ViT-S/B），各子网络在 k-NN、线性探测和下游任务上均达到 SOTA 水平。
