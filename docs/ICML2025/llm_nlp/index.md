---
title: >-
  ICML2025 LLM其他论文汇总 · 28篇论文解读
description: >-
  28篇ICML2025的 LLM 其他方向论文解读，涵盖 LLM、少样本学习、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "LLM 其他"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "少样本学习"
  - "时序预测"
item_list:
  - u: "b-score_detecting_biases_in_large_language_models_using_response_history/"
    t: "B-score: Detecting biases in large language models using response history"
  - u: "best-route_adaptive_llm_routing_with_test-time_optimal_compute/"
    t: "BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute"
  - u: "beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer/"
    t: "Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence"
  - u: "binary_hypothesis_testing_for_softmax_models_and_leverage_score_models/"
    t: "Binary Hypothesis Testing for Softmax Models and Leverage Score Models"
  - u: "breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting/"
    t: "Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting"
  - u: "build_agent_advocates_not_platform_agents/"
    t: "Build Agent Advocates, Not Platform Agents"
  - u: "defending_lvlms_against_vision_attacks_through_partial-perception_supervision/"
    t: "Defending LVLMs Against Vision Attacks through Partial-Perception Supervision"
  - u: "expert_evaluation_of_llm_world_models_a_high-t_c_superconductivity_case_study/"
    t: "Expert Evaluation of LLM World Models: A High-Tc Superconductivity Case Study"
  - u: "generalized_interpolating_discrete_diffusion/"
    t: "Generalized Interpolating Discrete Diffusion"
  - u: "generative_social_choice_the_next_generation/"
    t: "Generative Social Choice: The Next Generation"
  - u: "interchangeable_token_embeddings_for_extendable_vocabulary_and_alpha-equivalence/"
    t: "Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence"
  - u: "la_rosa_enhancing_llm_efficiency_via_layerwise_rotated_sparse_activation/"
    t: "LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation"
  - u: "laser_attention_with_exponential_transformation/"
    t: "LASER: Attention with Exponential Transformation"
  - u: "llm_social_simulations_are_a_promising_research_method/"
    t: "LLM Social Simulations Are a Promising Research Method"
  - u: "merit_maximum-normalized_element-wise_ratio_for_language_model_large-batch_train/"
    t: "MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training"
  - u: "on_expressive_power_of_looped_transformers_theoretical_analysis_and_enhancement_/"
    t: "On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding"
  - u: "product_of_experts_with_llms_boosting_performance_on_arc_is_a_matter_of_perspect/"
    t: "Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective"
  - u: "quest_enhancing_estimates_of_quantile-based_distributional_measures_using_model_/"
    t: "QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions"
  - u: "random_registers_for_cross-domain_few-shot_learning/"
    t: "Random Registers for Cross-Domain Few-Shot Learning"
  - u: "regress_dont_guess_--_a_regression-like_loss_on_number_tokens_for_language_model/"
    t: "Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models"
  - u: "rulebreakers_challenging_llms_at_the_crossroads_between_formal_logic_and_human-l/"
    t: "RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning"
  - u: "safe_delta_consistently_preserving_safety_when_fine-tuning_llms_on_diverse_datas/"
    t: "Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets"
  - u: "star_attention_efficient_llm_inference_over_long_sequences/"
    t: "Star Attention: Efficient LLM Inference over Long Sequences"
  - u: "tabflex_scaling_tabular_learning_to_millions_with_linear_attention/"
    t: "TabFlex: Scaling Tabular Learning to Millions with Linear Attention"
  - u: "the_lock-in_hypothesis_stagnation_by_algorithm/"
    t: "The Lock-in Hypothesis: Stagnation by Algorithm"
  - u: "theoretical_limitations_of_ensembles_in_the_age_of_overparameterization/"
    t: "Theoretical Limitations of Ensembles in the Age of Overparameterization"
  - u: "towards_universal_offline_black-box_optimization_via_learning_language_model_emb/"
    t: "Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings"
  - u: "when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser/"
    t: "When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series"
item_total: 28
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM 其他

**🧪 ICML2025** · **28** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (55)](../../ICLR2026/llm_nlp/index.md) · [💬 ACL2026 (61)](../../ACL2026/llm_nlp/index.md) · [🧪 ICML2026 (39)](../../ICML2026/llm_nlp/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/llm_nlp/index.md)

🔥 **高频主题：** LLM ×6 · 少样本学习 ×2 · 时序预测 ×2

**[B-score: Detecting biases in large language models using response history](b-score_detecting_biases_in_large_language_models_using_response_history.md)**

:   提出B-score指标，通过比较LLM在单轮(single-turn)与多轮(multi-turn)对话中的回答概率差异来检测偏见，发现LLM在多轮对话中能"自我去偏"，并利用B-score提升答案验证准确率。

**[BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)**

:   提出 BEST-Route（Best-of-n Enhanced Sampling and Test-time Route Optimization），在传统查询路由的基础上引入 best-of-n 采样策略，使路由器不仅选择模型，还自适应决定采样数量 n，通过小模型多次采样+选优替代大模型单次调用，在不到 1% 性能损失下降低高达 60% 的推理成本。

**[Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)**

:   本文通过设计 In-Context Meta-Learning (ICML) 实验环境，揭示了 Transformer 在获得上下文元学习能力的训练过程中，内部电路经历了三个截然不同的阶段性涌现（Bigram → Label Attention → Chunk Example），而非 induction head 研究中观察到的单阶段跃变，从而为理解 ICL 的深层机制提供了新视角。

**[Binary Hypothesis Testing for Softmax Models and Leverage Score Models](binary_hypothesis_testing_for_softmax_models_and_leverage_score_models.md)**

:   从理论角度研究Softmax模型和Leverage Score模型的二元假设检验问题，建立了在能量约束下区分两个参数化模型所需的查询次数的紧界，与理解LLM不同能力域的区分性问题相关。

**[Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting](breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting.md)**

:   提出 TimeFuse——一个样本级自适应模型融合框架，通过元特征描述输入时间序列特征并训练可学习融合器预测最优模型组合权重，在多个预测基准上对 SOTA 模型实现近乎普遍的改进（95.1% 样本优于最佳单模型）。

**[Build Agent Advocates, Not Platform Agents](build_agent_advocates_not_platform_agents.md)**

:   Position paper，指出LMA（语言模型代理）若被平台公司控制将成为加剧监控、锁定和注意力操控的"platform agents"，提出应发展用户控制的"agent advocates"来保护个人自主权，并给出三大干预措施：开放模型/算力、互操作性标准、市场监管。

**[Defending LVLMs Against Vision Attacks through Partial-Perception Supervision](defending_lvlms_against_vision_attacks_through_partial-perception_supervision.md)**

:   提出 DPS（Defense through Partial-Perception Supervision），利用裁剪图像的响应作为"弱监督"来引导全图模型在推理时自我修正，实现无需训练的黑盒 LVLM 视觉攻击防御，平均攻击成功率降低 76.3%。

**[Expert Evaluation of LLM World Models: A High-Tc Superconductivity Case Study](expert_evaluation_of_llm_world_models_a_high-t_c_superconductivity_case_study.md)**

:   以高温超导（HTS）领域为案例，构建了专家级数据集（1,726篇论文 + 67道专家问题），系统评估6种LLM系统的科学文献理解能力，发现基于精选文献的RAG系统在事实完整性和证据支持方面显著优于通用闭源模型。

**[Generalized Interpolating Discrete Diffusion](generalized_interpolating_discrete_diffusion.md)**

:   提出广义插值离散扩散框架 GIDD，将掩码扩散 (MDM) 推广为支持任意时变混合分布的扩散族，通过结合掩码与均匀噪声赋予模型自纠错能力，在扩散语言建模中取得 compute-matched SOTA。

**[Generative Social Choice: The Next Generation](generative_social_choice_the_next_generation.md)**

:   将生成式社会选择框架扩展至带成本/预算约束和近似查询的场景，提出 DemocraticProcess 算法并给出近乎最优的近似比例代表性理论保证，实现了实用系统 PROSE（基于 GPT-4o）在药物评论和城市治理数据集上验证有效性。

**[Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence](interchangeable_token_embeddings_for_extendable_vocabulary_and_alpha-equivalence.md)**

:   提出双部分 token 嵌入策略（共享可学习部分 + 随机区分部分），使语言模型能在训练后泛化到更大词表，并对 alpha-等价变换具有天然鲁棒性。

**[LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](la_rosa_enhancing_llm_efficiency_via_layerwise_rotated_sparse_activation.md)**

:   LaRoSA 提出了一种无需训练的激活稀疏化方法，通过逐层正交旋转矩阵将输入激活变换到更适合稀疏化的空间，并结合 Top-K 选择实现一致的模型级稀疏度和可靠的推理加速。

**[LASER: Attention with Exponential Transformation](laser_attention_with_exponential_transformation.md)**

:   通过分析注意力机制中 softmax 的梯度反向传播瓶颈，提出 LASER 注意力——在指数变换的 Value 空间中做注意力计算（即对 exp(V) 做 attention 再取 log），从而获得更大的 Jacobian 信号，改善参数学习效率。

**[LLM Social Simulations Are a Promising Research Method](llm_social_simulations_are_a_promising_research_method.md)**

:   本文作为立场论文，通过综述 36 篇实证研究论证了 LLM 社会模拟（用 LLM 模拟人类研究受试者）是一种有前景的研究方法，识别了五大可解决挑战（多样性、偏见、奉承、异质性、泛化），并为每个挑战提出了有前景的方向。

**[MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training](merit_maximum-normalized_element-wise_ratio_for_language_model_large-batch_train.md)**

:   提出 MERIT 优化器，通过最大范数归一化与逐元素信任比率扩展 LAMB，有效解决大批量训练中注意力 logit 爆炸导致的性能退化问题。

**[On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](on_expressive_power_of_looped_transformers_theoretical_analysis_and_enhancement_.md)**

:   本文首次建立了 Looped Transformer 关于循环次数和目标函数连续性模的逼近速率理论，揭示了循环架构特有的逼近误差来源（上下文连续性与 token 连续性），并提出 Timestep-Modulated Looped Transformer (TMLT) 通过时间步编码消除该限制，在推理、上下文学习和语言建模任务上取得一致提升。

**[Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective](product_of_experts_with_llms_boosting_performance_on_arc_is_a_matter_of_perspect.md)**

:   将 LLM 同时用作候选解生成器和评分器，通过基于 DFS 的搜索算法生成高概率候选解，再利用多视角增强下的 Product of Experts (PoE) 打分选出最优答案，在 ARC-AGI 公开评估集上以 71.6% 的准确率达到开源 SOTA，超越人类平均水平（60.2%），且单任务推理成本仅约 $0.02。

**[QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions](quest_enhancing_estimates_of_quantile-based_distributional_measures_using_model_.md)**

:   提出 QuEst 框架，将少量高质量观测数据与大量模型预测（imputed）数据相结合，对分位数相关的分布度量（QBDM）给出更精确的点估计和严格的置信区间，覆盖 CVaR、Interval-VaR 等经典指标。

**[Random Registers for Cross-Domain Few-Shot Learning](random_registers_for_cross-domain_few-shot_learning.md)**

:   在跨域小样本学习（CDFSL）中发现可学习 prompt 会损害目标域泛化性能，而用随机噪声替代（即随机寄存器）反而能持续提升性能，并基于此提出 REAP 方法，通过在图像语义区域添加随机寄存器来增强注意力扰动，实现高效的域无关特征学习。

**[Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models](regress_dont_guess_--_a_regression-like_loss_on_number_tokens_for_language_model.md)**

:   提出 Number Token Loss (NTL)，一种纯 token 级别的回归式损失函数，通过最小化数值 token 之间的 $L_p$ 范数或 Wasserstein 距离，为 LLM 注入数值邻近性归纳偏置。

**[RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning](rulebreakers_challenging_llms_at_the_crossroads_between_formal_logic_and_human-l.md)**

:   构建首个大规模"规则破坏者"数据集 RULEBREAKERS（25,600 实例），系统评估 7 个 LLM 在形式逻辑推理与事实知识冲突时的表现，发现模型普遍倾向过度刚性地应用逻辑规则而忽略常识，与人类推理行为存在显著偏离。

**[Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets](safe_delta_consistently_preserving_safety_when_fine-tuning_llms_on_diverse_datas.md)**

:   Safe Delta提出了一种安全感知的后训练防御方法，通过估计安全退化程度、选择性保留delta参数以最大化效用同时限制安全损失、并施加安全补偿向量来弥补残余安全损失，在多种微调数据集（不同规模、任务类型）上一致地保持LLM安全性而不牺牲效用。

**[Star Attention: Efficient LLM Inference over Long Sequences](star_attention_efficient_llm_inference_over_long_sequences.md)**

:   提出Star Attention两阶段块稀疏注意力：第一阶段将上下文分块在多主机上局部注意力编码，第二阶段查询通过聚合全局注意力生成，无需微调即可兼容现有LLM，推理加速11倍且保持97-100%精度。

**[TabFlex: Scaling Tabular Learning to Millions with Linear Attention](tabflex_scaling_tabular_learning_to_millions_with_linear_attention.md)**

:   用线性注意力替换 TabPFN 中的 softmax 注意力，将表格分类的 ICL 方法从小数据集扩展到百万级样本，实现 2× 以上加速且性能不降。

**[The Lock-in Hypothesis: Stagnation by Algorithm](the_lock-in_hypothesis_stagnation_by_algorithm.md)**

:   本文提出并形式化了"锁定假说"（Lock-in Hypothesis）：LLM 训练与部署过程中形成的人类-AI 反馈循环会固化用户的现有信念，导致群体观点多样性不可逆地丧失，甚至锁定在错误信念上。

**[Theoretical Limitations of Ensembles in the Age of Overparameterization](theoretical_limitations_of_ensembles_in_the_age_of_overparameterization.md)**

:   在过参数化条件下，无限集成模型与单个无穷宽模型逐点等价，集成方差不再反映传统贝叶斯不确定性而是衡量增加模型容量的预期效果，从理论上解释了深度集成相比大模型无本质泛化优势的经验观察。

**[Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)**

:   提出UniSO框架，将不同类型和维度的优化变量统一编码为JSON字符串后输入语言模型，通过token预测（UniSO-T）和数值回归（UniSO-N）两种建模范式训练通用回归器，并通过元数据引导的对比学习和Lipschitz平滑正则化改善嵌入空间质量，实现了跨域跨维度的通用离线黑盒优化。

**[When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)**

:   提出 Anomaly to Prompt (A2P) 框架，通过异常感知预测 (AAF) 和合成异常提示 (SAP) 两大模块，首次有效解决时间序列中"未来异常预测"(Anomaly Prediction) 这一新任务——不仅预测未来信号走势，还能精准定位未来哪些时间点会出现异常。
