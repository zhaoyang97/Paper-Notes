---
title: >-
  AAAI2026 因果推理论文汇总 · 7篇论文解读
description: >-
  7篇AAAI2026的因果推理方向论文解读，涵盖机器人、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "机器人"
  - "模型压缩"
item_list:
  - u: "causal_inference_under_threshold_manipulation_bayesian_mixtu/"
    t: "Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects"
  - u: "causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis/"
    t: "CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis"
  - u: "from_theory_of_mind_to_theory_of_environment_counterfactual_simulation_of_latent/"
    t: "From Theory of Mind to Theory of Environment: Counterfactual Simulation of Latent Environmental Dynamics"
  - u: "i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal/"
    t: "I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables"
  - u: "ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo/"
    t: "KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education"
  - u: "learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics/"
    t: "Learning Subgroups with Maximum Treatment Effects without Causal Heuristics"
  - u: "sparse_additive_model_pruning_for_order-based_causal_structure_learning/"
    t: "Sparse Additive Model Pruning for Order-Based Causal Structure Learning"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🤖 AAAI2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🧪 ICML2026 (19)](../../ICML2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/causal_inference/index.md)

**[Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](causal_inference_under_threshold_manipulation_bayesian_mixtu.md)**

:   提出 BMTM/HBMTM 贝叶斯混合模型框架，在消费者策略性操纵消费额以达到奖励阈值的场景下，通过将观测分布拆解为 bunching 与 non-bunching 两个子分布，准确估计阈值因果效应及跨子群的异质性处理效应。

**[CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)**

:   提出 CaDyT，结合高斯过程连续时间动力学建模（Adams-Bashforth 积分器实现精确推断）和 MDL 最小描述长度原则进行结构搜索，同时解决不规则采样和因果结构识别两个挑战，在双质点弹簧/菱形图/Rössler 振荡器上大幅超越所有基线（AUPRC 0.79 vs 次优 0.39）。

**[From Theory of Mind to Theory of Environment: Counterfactual Simulation of Latent Environmental Dynamics](from_theory_of_mind_to_theory_of_environment_counterfactual_simulation_of_latent.md)**

:   本文提出"环境理论"（Theory of Environment）概念，认为人类可能通过与心智理论（Theory of Mind）共享的计算机制来推断环境中隐含的动态规律，从而扩展运动探索的维度空间并促进行为创新。

**[I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)**

:   提出 I-CAM-UV 方法，通过对多个变量集不同的 CAM-UV 因果图结果进行一致性约束枚举，恢复因未观测变量而丢失的因果关系，并设计基于不一致代价单调性的最优优先搜索算法高效求解。

**[KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)**

:   提出 KTCF，一种面向知识追踪（KT）的反事实解释生成方法，通过考虑知识概念间关系生成稀疏且可操作的反事实解释，并将其后处理为顺序化的教学指令，在有效性、稀疏性和可操作性指标上全面超越基线方法。

**[Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)**

:   在 SCM 框架下证明最大处理效应子群必须具有同质点效应（定理1），在分区模型假设下证明最优子群发现可化简为标准监督学习（定理2），用 CART+Gini 指数即可实现——在 77 个 ACIC-2016 半合成数据集上均值处理效应 10.54（vs 次优 7.84），51.9% 排名第一。

**[Sparse Additive Model Pruning for Order-Based Causal Structure Learning](sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)**

:   提出 SARTRE 框架，利用随机化树嵌入与组稀疏回归学习稀疏加性模型，替代 CAM-pruning 中基于假设检验的冗余边修剪，在基于拓扑序的因果结构学习中实现显著加速且精度不降。
