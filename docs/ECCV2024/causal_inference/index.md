---
title: >-
  ECCV2024 因果推理论文汇总 · 4篇论文解读
description: >-
  4篇ECCV2024的因果推理方向论文解读，涵盖模型压缩、域适应、多模态、推理、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "模型压缩"
  - "域适应"
  - "多模态"
  - "推理"
  - "对抗鲁棒"
item_list:
  - u: "distill_gold_from_massive_ores_bi-level_data_pruning_towards_efficient_dataset_d/"
    t: "Distill Gold from Massive Ores: Bi-level Data Pruning towards Efficient Dataset Distillation"
  - u: "integrating_markov_blanket_discovery_into_causal_representation_learning_for_dom/"
    t: "Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization"
  - u: "learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni/"
    t: "Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning"
  - u: "understanding_physical_dynamics_with_counterfactual_world_modeling/"
    t: "Understanding Physical Dynamics with Counterfactual World Modeling"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🎞️ ECCV2024** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🧪 ICML2026 (19)](../../ICML2026/causal_inference/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md)

**[Distill Gold from Massive Ores: Bi-level Data Pruning towards Efficient Dataset Distillation](distill_gold_from_massive_ores_bi-level_data_pruning_towards_efficient_dataset_d.md)**

:   提出双层数据剪枝策略 BiLP，通过经验损失静态剪枝和基于因果效应 (ITE) 的动态剪枝，高效选择对数据集蒸馏最有价值的真实样本，以即插即用方式一致性提升现有蒸馏方法性能并降低计算开销。

**[Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization](integrating_markov_blanket_discovery_into_causal_representation_learning_for_dom.md)**

:   提出 CMBRL 框架，在隐空间中发现马尔可夫毯（Markov Blanket）特征——目标变量的最小充分统计量——代替现有方法中仅选择因果/反因果变量的做法，构建不变预测机制实现跨域泛化。

**[Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)**

:   本文提出了反事实偏差鲁棒推理数据集（CoBRa）和反事实思维链方法（CoCT），通过构造编辑后的知识图谱和图像内容来评估和缓解大型视觉语言模型（LVLM）中的知识偏差，使模型能够逐步推理而非依赖偏见知识，在需要知识偏差下推理的任务上显著优于现有方法。

**[Understanding Physical Dynamics with Counterfactual World Modeling](understanding_physical_dynamics_with_counterfactual_world_modeling.md)**

:   本文提出反事实世界建模（Counterfactual World Modeling, CWM），通过时序分解的遮蔽策略训练视频掩码预测器，并设计"反事实提示"机制从单一预训练模型中无需微调即可提取光流、分割、关键点等多种视觉结构，在物理动力学理解任务Physion基准上达到最优性能。
