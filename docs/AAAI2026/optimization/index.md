---
title: >-
  AAAI2026 优化/理论论文汇总 · 21篇论文解读
description: >-
  21篇AAAI2026的优化/理论方向论文解读，涵盖联邦学习、LLM、对抗鲁棒、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "联邦学习"
  - "LLM"
  - "对抗鲁棒"
  - "个性化生成"
item_list:
  - u: "a_distributed_asynchronous_generalized_momentum_algorithm_wi/"
    t: "A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds"
  - u: "a_unified_convergence_analysis_for_semi-decentralized_learni/"
    t: "A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication"
  - u: "beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar/"
    t: "Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training"
  - u: "bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_/"
    t: "Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation"
  - u: "co-layout_llm-driven_co-optimization_for_interior_layout/"
    t: "Co-Layout: LLM-driven Co-optimization for Interior Layout"
  - u: "convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond/"
    t: "Convex Clustering Redefined: Robust Learning with the Median of Means Estimator"
  - u: "cost-minimized_label-flipping_poisoning_attack_to_llm_alignment/"
    t: "Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment"
  - u: "data_heterogeneity_and_forgotten_labels_in_split_federated_learning/"
    t: "Data Heterogeneity and Forgotten Labels in Split Federated Learning"
  - u: "ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions/"
    t: "ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions"
  - u: "efficient_and_reliable_hitting-set_computations_for_the_implicit_hitting_set_app/"
    t: "Efficient and Reliable Hitting-Set Computations for the Implicit Hitting Set Approach"
  - u: "fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix/"
    t: "FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters"
  - u: "ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets/"
    t: "GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets"
  - u: "instance_generation_for_meta-black-box_optimization_through_latent_space_reverse/"
    t: "Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering"
  - u: "motif_multi-strategy_optimization_via_turn-based_interactive_framework/"
    t: "MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework"
  - u: "on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd/"
    t: "On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD"
  - u: "parametrized_multi-agent_routing_via_deep_attention_models/"
    t: "Parametrized Multi-Agent Routing via Deep Attention Models"
  - u: "pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de/"
    t: "Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization"
  - u: "peoat_personalization-guided_evolutionary_question_assembly_for_one-shot_adaptiv/"
    t: "PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing"
  - u: "personalized_federated_learning_with_bidirectional_communication_compression_via/"
    t: "Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching"
  - u: "smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da/"
    t: "SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data"
  - u: "tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_/"
    t: "Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🤖 AAAI2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/optimization/index.md) · [🔬 ICLR2026 (220)](../../ICLR2026/optimization/index.md) · [🧪 ICML2026 (88)](../../ICML2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md) · [🧪 ICML2025 (61)](../../ICML2025/optimization/index.md)

🔥 **高频主题：** 联邦学习 ×5 · LLM ×4 · 对抗鲁棒 ×2 · 个性化生成 ×2

**[A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)**

:   提出一种完全异步（totally asynchronous）的广义动量（Generalized Momentum）分布式优化算法，无需假设通信/计算延迟的上界即可保证线性收敛，在 Fashion-MNIST 分类任务上比梯度下降快 71%、比 Heavy Ball 快 41%、比 Nesterov 加速梯度法快 19%。

**[A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](a_unified_convergence_analysis_for_semi-decentralized_learni.md)**

:   本文在统一的收敛分析框架下，首次系统比较了半去中心化联邦学习中两种服务器-设备通信原语（S2S仅返回被采样设备 vs. S2A广播给所有设备），揭示了S2S在高组间异质性下更优、S2A在低异质性下更优的不同regime，并给出了实用的系统配置指南。

**[Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)**

:   提出 Fisher-Orthogonal Projection (FOP)，通过在 Fisher 度量下对子批次梯度差做正交投影来补充方差信息，使二阶优化器 KFAC 在超大 batch 训练中保持有效，实现最高 ×7.5 的加速。

**[Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)**

:   提出 EvoReal 框架，利用 LLM 驱动的进化搜索生成结构上接近真实世界的 VRP 合成实例，再通过两阶段渐进微调策略将预训练神经求解器适配到真实基准，在 TSPLib (1.05% gap) 和 CVRPLib (2.71% gap) 上大幅超越已有神经求解器。

**[Co-Layout: LLM-driven Co-optimization for Interior Layout](co-layout_llm-driven_co-optimization_for_interior_layout.md)**

:   提出 Co-Layout 框架，利用 LLM 从自然语言需求中提取结构化约束，再通过基于网格的整数规划（IP）联合优化房间布局与家具摆放，辅以粗到精求解策略提升效率，显著优于现有两阶段方案。

**[Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)**

:   本文将 Median of Means (MoM) 估计器融入凸聚类框架，提出 COMET 算法，通过随机分箱与中位数聚合实现对噪声和离群点的鲁棒性，同时无需预知簇数 $k$，理论上证明了弱一致性，实验在多个真实数据集上显著超越 k-means、MoM k-means、凸聚类等六种基线方法。

**[Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)**

:   首次从理论上分析了在 RLHF/DPO 对齐过程中，通过翻转偏好标签来引导 LLM 策略走向攻击者目标所需的最小成本，将其形式化为凸优化问题并推导了成本的上下界，进而提出 PCM（Poisoning Cost Minimization）后处理方法，可在保持投毒效果的同时显著减少标签翻转数量。

**[Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)**

:   系统研究了 Split Federated Learning 中数据异构导致的灾难性遗忘现象（尤其是 server 端处理顺序造成的 intra-round 遗忘），并提出基于 multi-head 的 Hydra 方法，将 part-2 的最后层分组训练再聚合，显著降低标签间性能差距（PG 最高降低 75.4%）。

**[ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)**

:   提出ECPv2算法，通过三项创新（自适应下界、Worst-$m$ memory、固定随机投影），将Lipschitz函数全局优化的运行时从$\Omega(n^2 d)$降至$\Omega(n(m+d)\log n)$，同时保持与minimax下界匹配的$O(n^{-1/d})$ regret收敛速率。

**[Efficient and Reliable Hitting-Set Computations for the Implicit Hitting Set Approach](efficient_and_reliable_hitting-set_computations_for_the_implicit_hitting_set_app.md)**

:   针对隐式击中集框架中击中集组件依赖商用IP求解器带来的数值不稳定问题，提出基于伪布尔推理和随机局部搜索的替代方案及混合策略，实现了首个可认证的IHS计算并在1786个基准实例上展示了效率与可靠性的有效权衡。

**[FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)**

:   提出 FedPM（Federated Preconditioned Mixing），一种新型联邦学习方法，通过在服务器端用"预条件混合"替代传统的简单参数平均，解决了现有二阶联邦优化方法中局部预条件器漂移问题，在理论上证明了强凸目标的超线性收敛速率，并在异质数据场景中显著超越现有方法。

**[GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)**

:   提出 GHOST 框架，一种层次化最优搜索算法，用于求解凸集图（GCS）上的旅行商问题。通过结合组合路径搜索与凸轨迹优化，并利用新颖的抽象路径展开算法计算可容许下界指导最佳优先搜索，GHOST 在保证最优性的同时比统一混合整数凸规划基线快数个数量级。

**[Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering](instance_generation_for_meta-black-box_optimization_through_latent_space_reverse.md)**

:   提出 LSRE 框架，通过自编码器构建 BBO 问题实例的二维潜在空间，并利用遗传编程从该空间中反向工程出多样化的合成优化问题实例集 Diverse-BBO，显著提升 MetaBBO 方法的泛化性能。

**[MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)**

:   提出 MOTIF 框架，将求解器设计建模为多策略优化问题，通过基于蒙特卡洛树搜索 (MCTS) 的双 LLM 代理回合制竞争机制，联合优化组合优化求解器中的多个相互依赖的算法组件，在 TSP、CVRP、BPP 等多个组合优化领域中一致超越现有方法。

**[On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)**

:   在二层过参数化线性网络上理论分析 Label Noise SGD 的学习动力学，揭示了两阶段行为——Phase I 中权重范数逐渐缩小使模型从 lazy regime 逃逸到 rich regime，Phase II 中权重与真实插值器对齐并收敛——并将该理论扩展到 SAM 优化器。

**[Parametrized Multi-Agent Routing via Deep Attention Models](parametrized_multi-agent_routing_via_deep_attention_models.md)**

:   提出Deep FLPO框架，将最大熵原理（MEP）的代数结构与permutation-invariant的encoder-decoder神经网络（SPN）融合，解决设施选址与路径联合优化的NP-hard混合整数问题，实现策略推理100倍加速、与Gurobi精确解匹配且快1500倍。

**[Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)**

:   提出 MPaGE 框架，将 LLM 与 Pareto Front Grid 机制和语义聚类结合，自动为多目标组合优化问题生成兼顾解质量与运行效率的启发式算法，在 Bi-TSP、Tri-TSP、Bi-CVRP、Bi-KP 上 HV 和 IGD 均显著优于 EoH、MEoH 等基线。

**[PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing](peoat_personalization-guided_evolutionary_question_assembly_for_one-shot_adaptiv.md)**

:   首次提出"一次性自适应测试 (OAT)"任务，将其建模为组合优化问题，并设计 PEOAT 框架——结合个性化初始化、认知增强进化搜索和多样性保持选择策略，在无交互反馈的条件下为每位考生一次性选出最优题集，大幅超越传统 CAT 方法。

**[Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](personalized_federated_learning_with_bidirectional_communication_compression_via.md)**

:   提出 pFed1BS 框架，通过单比特随机草图实现联邦学习中上下行双向极致通信压缩（降低 99%+），同时引入基于符号的正则化器实现客户端模型个性化，在非 IID 数据场景下同时解决通信瓶颈和数据异质性两大难题。

**[SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)**

:   提出 SMoFi 框架，通过在 Split FL 的 server 端每步同步各 surrogate 模型的 momentum buffer，有效缓解 non-IID 数据导致的梯度分歧，在精度最高提升 7.1%、收敛速度最高加速 10.25 倍。

**[Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)**

:   提出FedCSPACK，一种基于余弦稀疏化参数打包和双权重聚合的个性化联邦学习方法，通过在包级别进行参数选择和共享，同时平衡了数据异质性和客户端资源约束，训练速度提升2-5倍、通信量压缩高达96%，同时模型精度提升3.34%。
