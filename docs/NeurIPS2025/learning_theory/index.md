---
title: >-
  NeurIPS2025 学习理论论文汇总 · 25篇论文解读
description: >-
  25篇NeurIPS2025的学习理论方向论文解读，涵盖域适应、扩散模型、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "学习理论"
  - "论文解读"
  - "论文笔记"
  - "域适应"
  - "扩散模型"
  - "压缩/编码"
item_list:
  - u: "a_highdimensional_statistical_method_for_optimizing_transfer/"
    t: "A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning"
  - u: "adaptive_data_analysis_for_growing_data/"
    t: "Adaptive Data Analysis for Growing Data"
  - u: "computable_universal_online_learning/"
    t: "Computable Universal Online Learning"
  - u: "conformal_online_learning_of_deep_koopman_linear_embeddings/"
    t: "Conformal Online Learning of Deep Koopman Linear Embeddings"
  - u: "diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua/"
    t: "Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification"
  - u: "efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c/"
    t: "Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games"
  - u: "finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema/"
    t: "Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds"
  - u: "how_many_domains_suffice_for_domain_generalization_a_tight_characterization_via_/"
    t: "How Many Domains Suffice for Domain Generalization? A Tight Characterization via the Domain Shattering Dimension"
  - u: "improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl/"
    t: "Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering"
  - u: "infrequent_exploration_in_linear_bandits/"
    t: "Infrequent Exploration in Linear Bandits"
  - u: "keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t/"
    t: "Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning"
  - u: "kernel_conditional_tests_from_learning-theoretic_bounds/"
    t: "Kernel Conditional Tests from Learning-Theoretic Bounds"
  - u: "learning-augmented_online_bipartite_fractional_matching/"
    t: "Learning-Augmented Online Bipartite Fractional Matching"
  - u: "learning-augmented_streaming_algorithms_for_correlation_clustering/"
    t: "Learning-Augmented Streaming Algorithms for Correlation Clustering"
  - u: "non-clairvoyant_scheduling_with_progress_bars/"
    t: "Non-Clairvoyant Scheduling with Progress Bars"
  - u: "on_agnostic_pac_learning_in_the_small_error_regime/"
    t: "On Agnostic PAC Learning in the Small Error Regime"
  - u: "optimism_without_regularization_constant_regret_in_zero-sum_games/"
    t: "Optimism Without Regularization: Constant Regret in Zero-Sum Games"
  - u: "prediction-powered_semi-supervised_learning_with_online_power_tuning/"
    t: "Prediction-Powered Semi-Supervised Learning with Online Power Tuning"
  - u: "product_distribution_learning_with_imperfect_advice/"
    t: "Product Distribution Learning with Imperfect Advice"
  - u: "reliably_detecting_model_failures_in_deployment_without_labels/"
    t: "Reliably Detecting Model Failures in Deployment Without Labels"
  - u: "revisiting_agnostic_boosting/"
    t: "Revisiting Agnostic Boosting"
  - u: "sample-adaptivity_tradeoff_in_on-demand_sampling/"
    t: "Sample-Adaptivity Tradeoff in On-Demand Sampling"
  - u: "the_parameterized_complexity_of_computing_the_vc-dimension/"
    t: "The Parameterized Complexity of Computing the VC-Dimension"
  - u: "the_structural_complexity_of_matrix-vector_multiplication/"
    t: "The Structural Complexity of Matrix-Vector Multiplication"
  - u: "transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression/"
    t: "Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression"
item_total: 25
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 学习理论

**🧠 NeurIPS2025** · **25** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (294)](../../ICLR2026/learning_theory/index.md) · [🧪 ICML2026 (45)](../../ICML2026/learning_theory/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/learning_theory/index.md) · [🧪 ICML2025 (16)](../../ICML2025/learning_theory/index.md)

🔥 **高频主题：** 域适应 ×3

**[A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)**

:   提出基于K-L散度和高维统计分析的理论框架，用于确定多源迁移学习中每个源任务的最优样本迁移数量，避免"用所有源数据"带来的负迁移问题，在DomainNet和Office-Home上超过SOTA 1.0-1.5%的同时减少47.85%的样本使用量和35.19%的训练时间。

**[Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)**

:   本文首次给出了动态增长数据上自适应分析的泛化界，允许分析者根据数据规模自适应调度查询，并通过时变经验精度界和差分隐私机制实现随数据积累越来越紧的泛化保证。

**[Computable Universal Online Learning](computable_universal_online_learning.md)**

:   在 universal online learning 框架中引入可计算性约束，证明了"数学上可学习"不等于"可用计算机程序实现的可学习"，并给出了 agnostic 和 proper 变体下可计算学习的精确刻画。

**[Conformal Online Learning of Deep Koopman Linear Embeddings](conformal_online_learning_of_deep_koopman_linear_embeddings.md)**

:   提出 COLoKe 框架，将 conformal prediction 重新解读为模型一致性诊断工具，仅在 Koopman 模型的预测误差超过动态校准阈值时才触发参数更新，从而实现对非线性动力系统的高效在线 Koopman 线性嵌入学习。

**[Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)**

:   本文从统计学习角度分析了条件扩散Transformer（DiT）在时间序列插补任务中的样本复杂度和不确定性量化性能，并提出混合掩码训练策略提升插补效果。

**[Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games](efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)**

:   提出基于核化（kernelization）的框架，在部分信息反馈设定下为多面体博弈（Colonel Blotto、图拟阵拥堵博弈、网络拥堵博弈）设计了计算高效的无遗憾学习算法，显著改进了学习粗关联均衡（CCE）的运行时复杂度。

**[Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)**

:   提出 Riemannian Online to NonConvex (RO2NC) 算法及其零阶版本 ZO-RO2NC，首次为黎曼流形上完全非光滑非凸随机优化建立了 $O(\delta^{-1}\epsilon^{-3})$ 的有限时间样本复杂度保证，匹配欧几里德最优结果。

**[How Many Domains Suffice for Domain Generalization? A Tight Characterization via the Domain Shattering Dimension](how_many_domains_suffice_for_domain_generalization_a_tight_characterization_via_.md)**

:   提出"领域碎裂维度"（Domain Shattering Dimension）这一新组合度量，紧致刻画了领域泛化所需的领域数量（领域样本复杂度），并证明其与经典VC维的关系为 $\Theta(d \log(1/\alpha))$。

**[Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)**

:   针对 Correlation Clustering 的两个重要推广——Chromatic CC 和 pseudometric-weighted CC，基于 LP relaxation 与精心设计的 rounding function，分别取得 2.15-approximation 和 tight 10/3-approximation，显著改进了先前最佳结果（2.5 和 6）。

**[Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)**

:   提出 INFEX 框架，按给定调度表在探索步执行基线算法（如 LinUCB/LinTS）、其余时刻贪心选臂，证明只要探索次数超过 $\omega(\log T)$ 即可达到与全时刻探索相同的多项对数 regret，同时大幅降低计算开销（80%-99% 时间步为贪心）。

**[Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)**

:   提出 Controllable Pseudo-label Generation (CPG) 框架，通过可控的自强化优化循环将可靠伪标签逐步纳入标注集，在已知分布上构建 Bayes-optimal 分类器，从而在未标注数据分布完全未知的 Realistic LTSSL 场景下实现最高 15.97% 的准确率提升。

**[Kernel Conditional Tests from Learning-Theoretic Bounds](kernel_conditional_tests_from_learning-theoretic_bounds.md)**

:   提出将学习算法的置信界转化为条件假设检验的统一框架，基于核岭回归构建了有限样本保证的条件两样本检验，首次支持非i.i.d.数据与在线采样场景。

**[Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)**

:   本文提出了两个学习增强算法（LAB 和 PAW），用于在线二部分数匹配问题，在给定可能不准确的建议匹配的情况下，首次在整个鲁棒性范围内 Pareto 优于朴素的 CoinFlip 策略。

**[Learning-Augmented Streaming Algorithms for Correlation Clustering](learning-augmented_streaming_algorithms_for_correlation_clustering.md)**

:   提出了首个面向相关聚类（Correlation Clustering）的学习增强流算法，利用成对距离预测，在完全图上实现优于3的近似比（$\tilde{O}(n)$ 空间），在一般图上实现 $O(\log|E^-|)$ 近似比（$\tilde{O}(n)$ 空间），在空间-近似比权衡上显著改进了已有的非学习算法。

**[Non-Clairvoyant Scheduling with Progress Bars](non-clairvoyant_scheduling_with_progress_bars.md)**

:   引入"进度条"信息模型作为透视与非透视调度之间的插值框架，针对对抗性和随机性进度条分别设计了具有最优一致性-鲁棒性权衡的调度算法，同时推进了学习增强调度的理论前沿。

**[On Agnostic PAC Learning in the Small Error Regime](on_agnostic_pac_learning_in_the_small_error_regime.md)**

:   本文在不可知 PAC 学习的小误差域（$\tau \approx d/m$）中，构造了一个基于 ERM 聚合的计算高效学习器，实现了 $c \cdot \tau + O(\sqrt{\tau d/m} + d/m)$ 的误差上界（$c \leq 2.1$），匹配了已知下界，推进了不可知学习的精确复杂度刻画。

**[Optimism Without Regularization: Constant Regret in Zero-Sum Games](optimism_without_regularization_constant_regret_in_zero-sum_games.md)**

:   首次证明无正则化的Optimistic Fictitious Play在2×2零和博弈中获得O(1)常数遗憾，匹配了正则化Optimistic FTRL的最优率，同时证明Alternating Fictitious Play的遗憾下界为Ω(√T)，分离了乐观和交替在无正则化情况下的能力。

**[Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)**

:   将预测驱动推断（PPI）框架扩展到半监督学习训练过程中，提出无偏梯度估计器，并设计在线AdaGrad算法动态调节伪标签与真实标签的相对权重 $\lambda$，在保证无偏性的同时实现与最优固定 $\lambda$ 匹配的收敛速率。

**[Product Distribution Learning with Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)**

:   本文研究在给定不完美建议分布的情况下学习布尔超立方体上乘积分布的问题，提出了一种高效算法，当建议质量足够好时样本复杂度可实现关于维度 $d$ 的次线性依赖。

**[Reliably Detecting Model Failures in Deployment Without Labels](reliably_detecting_model_failures_in_deployment_without_labels.md)**

:   提出D3M(Disagreement-Driven Deterioration Monitoring)，一种基于变分贝叶斯后验采样的三阶段模型监控算法，在无标签、无训练数据的部署场景下可靠检测模型性能退化，同时对非退化性偏移保持低误报率。

**[Revisiting Agnostic Boosting](revisiting_agnostic_boosting.md)**

:   提出新的不可知 Boosting 算法,在非常一般的假设下大幅改善了此前工作的样本复杂度,并建立近匹配下界,从而在对数因子意义下解决了不可知 Boosting 的样本复杂度问题。

**[Sample-Adaptivity Tradeoff in On-Demand Sampling](sample-adaptivity_tradeoff_in_on-demand_sampling.md)**

:   系统研究了按需采样（on-demand sampling）中样本复杂度与自适应轮次之间的权衡关系，在可实现设定下证明 $r$ 轮算法的最优样本复杂度为 $dk^{\Theta(1/r)}/\varepsilon$，在不可知设定下提出仅需 $\widetilde{O}(\sqrt{k})$ 轮即可达近最优样本复杂度的LazyHedge算法，并引入OODS抽象框架建立了近紧的轮次复杂度下界。

**[The Parameterized Complexity of Computing the VC-Dimension](the_parameterized_complexity_of_computing_the_vc-dimension.md)**

:   本文系统研究了计算VC维问题的参数化复杂性，证明朴素穷举算法在ETH假设下是渐近最优的，给出按最大度参数化的FPT 1-可加近似算法，以及按树宽参数化的 $2^{O(\text{tw} \cdot \log \text{tw})} \cdot |V|$ 精确算法，并完整刻画了各结构参数下的可处理性景观。

**[The Structural Complexity of Matrix-Vector Multiplication](the_structural_complexity_of_matrix-vector_multiplication.md)**

:   证明对于 corrupted VC-dimension 为 $d$ 的布尔矩阵 $\mathbf{M} \in \{0,1\}^{m \times n}$，矩阵-向量乘法可在 $\widetilde{O}(nm^{1-1/d}+m)$ 时间内完成，首次为结构化矩阵提供了真亚二次时间上界，推翻了 OMv 猜想在结构化输入上的适用性，并导出了动态 Laplacian 求解器、有效电阻、三角检测等问题的首个高精度亚二次算法。

**[Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)**

:   提出两步式Transfer MNI方法，在高维过参数化线性回归中通过"保留目标信号+零空间迁移源知识"机制增强良性过拟合的泛化能力，刻画了模型偏移和协变量偏移下的非渐近excess risk，并发现了"免费午餐"协变量偏移区间。
