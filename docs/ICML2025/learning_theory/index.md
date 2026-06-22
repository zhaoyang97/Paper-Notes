---
title: >-
  ICML2025 学习理论论文汇总 · 16篇论文解读
description: >-
  16篇ICML2025的学习理论方向论文解读，涵盖对抗鲁棒、Agent、域适应等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "学习理论"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "Agent"
  - "域适应"
item_list:
  - u: "avoiding_catastrophe_in_online_learning_by_asking_for_help/"
    t: "Avoiding Catastrophe in Online Learning by Asking for Help"
  - u: "heavy-tailed_linear_bandits_huber_regression_with_one-pass_update/"
    t: "Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update"
  - u: "improved_and_oracle-efficient_online_ell_1-multicalibration/"
    t: "Improved and Oracle-Efficient Online $\\ell_1$-Multicalibration"
  - u: "improved_generalization_bounds_for_transductive_learning_by_transductive_local_c/"
    t: "Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications"
  - u: "learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors/"
    t: "Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors"
  - u: "learning-augmented_hierarchical_clustering/"
    t: "Learning-Augmented Hierarchical Clustering"
  - u: "maximum_coverage_in_turnstile_streams_with_applications_to_fingerprinting_measur/"
    t: "Maximum Coverage in Turnstile Streams with Applications to Fingerprinting Measures"
  - u: "multiple-policy_evaluation_via_density_estimation/"
    t: "Multiple-Policy Evaluation via Density Estimation"
  - u: "near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna/"
    t: "Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems"
  - u: "near_optimal_best_arm_identification_for_clustered_bandits/"
    t: "Near Optimal Best Arm Identification for Clustered Bandits"
  - u: "on_fine-grained_distinct_element_estimation/"
    t: "On Fine-Grained Distinct Element Estimation"
  - u: "positional_attention_expressivity_and_learnability_of_algorithmic_computation/"
    t: "Positional Attention: Expressivity and Learnability of Algorithmic Computation"
  - u: "principled_algorithms_for_optimizing_generalized_metrics_in_binary_classificatio/"
    t: "Principled Algorithms for Optimizing Generalized Metrics in Binary Classification"
  - u: "provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin/"
    t: "Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition"
  - u: "sparse-pivot_dynamic_correlation_clustering_for_node_insertions/"
    t: "Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions"
  - u: "theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt/"
    t: "Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport"
item_total: 16
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 学习理论

**🧪 ICML2025** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (294)](../../ICLR2026/learning_theory/index.md) · [🧪 ICML2026 (45)](../../ICML2026/learning_theory/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/learning_theory/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/learning_theory/index.md)

**[Avoiding Catastrophe in Online Learning by Asking for Help](avoiding_catastrophe_in_online_learning_by_asking_for_help.md)**

:   提出一个全新的在线学习理论框架来处理灾难性（不可逆）错误：将回报定义为避灾概率、目标函数为回报之积（总体避灾概率），引入导师求助机制和Local Generalization假设，证明不可能结果（不求助则必灾难）和可能结果（策略类可学则后悔和求助率同时趋零），将标准在线学习的子线性后悔提升为子常数后悔。

**[Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)**

:   提出基于 Online Mirror Descent 的单遍 Huber 回归算法 Hvt-UCB，用于重尾噪声线性 bandit，将每轮计算复杂度从 $\mathcal{O}(t\log T)$ 降至 $\mathcal{O}(1)$，同时保持最优且依赖实例的 regret 界。

**[Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](improved_and_oracle-efficient_online_ell_1-multicalibration.md)**

:   提出将在线 $\ell_1$-multicalibration 归约为新定义的在线线性乘积优化 (OLPO) 问题，分别达到 $\widetilde{O}(T^{-1/3})$（改进速率）和 $\widetilde{O}(T^{-1/4})$（oracle 高效速率）的多校准误差上界。

**[Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)**

:   提出转导局部复杂度（TLC）框架，将经典的局部 Rademacher 复杂度扩展到转导学习设定，获得了与归纳学习几乎一致的超额风险界（仅差对数因子），并解决了十年未决的开放问题。

**[Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)**

:   在度量任务系统(MTS)中，当算法仅能以 bandit 方式（每步只查询一个启发式且需连续查询 $m$ 步才能观测状态）访问 $\ell$ 个启发式时，本文给出了 regret 为 $O(\text{OPT}^{2/3})$ 的算法，并证明该界是紧的。

**[Learning-Augmented Hierarchical Clustering](learning-augmented_hierarchical_clustering.md)**

:   本文研究借助分裂预言机（splitting oracle）的辅助信息来突破层次聚类的近似硬度障碍，获得 Dasgupta 目标的 $O(1)$ 常数近似和 Moseley-Wang 目标的 $(1-o(1))$ 近似，并推广到流式和并行计算场景。

**[Maximum Coverage in Turnstile Streams with Applications to Fingerprinting Measures](maximum_coverage_in_turnstile_streams_with_applications_to_fingerprinting_measur.md)**

:   首次在 turnstile 流模型（支持任意插入/删除）下给出最大覆盖问题的单遍流算法，空间 $\tilde{O}(d/\varepsilon^3)$、更新时间 $\tilde{O}(1)$，并将其推广到隐私指纹识别（fingerprinting）场景，实验比先前方法快 210×。

**[Multiple-Policy Evaluation via Density Estimation](multiple-policy_evaluation_via_density_estimation.md)**

:   提出 CAESAR 算法，通过两阶段方法（粗估计访问分布 + 最优采样分布下的密度比估计）同时评估 K 个策略，实现非渐近、实例依赖的样本复杂度，核心技术是"粗估计"——仅需 $O(1/\epsilon)$ 样本即可获得常数倍精度的分布近似。

**[Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)**

:   提出一族基于简洁预测（临界值的点预测或区间预测）的在线背包算法，在consistency与robustness之间实现近Pareto最优的权衡，并给出分数解到整数解的通用转换方法。

**[Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)**

:   在多智能体聚类多臂赌博机设置下，提出 Cl-BAI 和 BAI-Cl 两种算法，利用聚类结构大幅降低最优臂识别的样本复杂度，并证明 BAI-Cl++ 在 $M$ 为常数时达到 minimax 最优。

**[On Fine-Grained Distinct Element Estimation](on_fine-grained_distinct_element_estimation.md)**

:   提出以**成对碰撞数** $C$（pairwise collisions）作为分布式去重计数问题的细粒度复杂度参数，设计了通信量随 $C$ 减小而显著降低的协议，打破了此前 $\Omega(\alpha/\varepsilon^2)$ 的最坏情况下界，并给出了所有参数区间的匹配下界。

**[Positional Attention: Expressivity and Learnability of Algorithmic Computation](positional_attention_expressivity_and_learnability_of_algorithmic_computation.md)**

:   提出 **Positional Transformer**——注意力权重仅由位置编码决定、与输入数据无关的 Transformer 变体，证明其保持了与 MPC 并行计算模型等价的表达力（仅增加 $O(\log n)$ 深度代价），并在算法任务上展现出显著更优的分布外泛化能力。

**[Principled Algorithms for Optimizing Generalized Metrics in Binary Classification](principled_algorithms_for_optimizing_generalized_metrics_in_binary_classificatio.md)**

:   本文提出了优化广义分类指标（如 $F_\beta$、Jaccard、加权准确率等）的有原则算法 METRO，基于 $H$-一致性界和代理损失理论，将指标优化重新表述为广义代价敏感学习问题，具有有限样本泛化保证。

**[Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)**

:   本文在委托-代理（principal-agent）在线信息获取框架下研究最佳评分规则识别（Best Scoring Rule Identification, BSRI）问题，提出 OIAFC（固定置信度）和 OIAFB（固定预算）两种算法，首次建立了实例依赖的样本复杂度上界 $\widetilde{O}(MH_\Delta)$，并将实例无关的样本复杂度从已有工作的 $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ 大幅改进至 $\widetilde{O}(MK\epsilon^{-2})$。

**[Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)**

:   提出 Sparse-Pivot 算法，在节点动态插入的 Correlation Clustering 问题中以摊销 $O_\varepsilon(\log^{O(1)} n)$ 的数据库操作实现 $(20+\varepsilon)$-近似，大幅改善了 Cohen-Addad et al. (ICML 2024) 的近似因子，并在实验中全面优于基线。

**[Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt.md)**

:   本文基于部分最优传输理论推导了部分领域自适应（PDA）的泛化界，证明了部分 Wasserstein 距离作为领域对齐项和提出的理论驱动权重方案的合理性，并据此开发了实用算法 WARMPOT。
