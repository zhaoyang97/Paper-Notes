<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🧠 NeurIPS2025** · 共 **11** 篇

**[A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)**

:   提出基于多智能体影响图（MAIDs）的**目标干预范式（Targeted Intervention）**，通过仅对单个目标智能体施加**预策略干预（Pre-Strategy Intervention, PSI）**，引导整个多智能体系统收敛到满足额外期望结果的优选Nash均衡，无需对所有智能体进行全局干预。

**[An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)**

:   建立数据增强（DA）与因果推断的统一框架：当结果函数对DA不变时，DA等价于对处理机制的软干预；提出IV-like回归松弛工具变量假设，组合的DA+IVL方法严格降低混杂偏差，在不可识别设置中仍能改善因果估计。

**[Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)**

:   提出 Bi-DFCL，通过双层优化框架联合利用观测数据和 RCT 实验数据来训练营销资源分配模型：上层用 RCT 数据的无偏决策损失端到端训练 Bridge Network 来动态纠正下层在观测数据上的偏差，同时设计了基于原始问题的可微代理决策损失（PPL/PIFD）和隐式微分算法，解决了传统两阶段方法的预测-决策不一致和偏差-方差困境。已在美团大规模在线部署。

**[Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)**

:   CAPE 通过从表格数据中学习特征间的因果DAG结构，将其嵌入双曲空间生成因果感知的旋转位置编码（RoPE），使 Transformer 能处理非序列但因果相关的特征数据，在多组学数据的下游任务上显著提升性能。

**[Characterization and Learning of Causal Graphs from Hard Interventions](characterization_and_learning_of_causal_graphs_from_hard_interventions.md)**

:   首次系统分析硬干预（hard interventions）与软干预在因果发现中的差异，证明硬干预通过非局部打破d-分离关系提供更强的图区分能力，提出广义do-演算和基于孪生增强MAG的因果发现算法，实验表明硬干预将Markov等价类缩小37-57%。

**[Conformal Prediction for Causal Effects of Continuous Treatments](conformal_prediction_for_causal_effects_of_continuous_treatments.md)**

:   首次为连续处理（如药物剂量）的因果效应开发共形预测（CP）区间，通过倾向性散度标准化处理干预诱导的分布偏移，在已知/未知倾向性两种场景下提供有限样本 $1-\alpha$ 覆盖保证，在MIMIC-III临床数据上验证了实用性。

**[Demystifying Spectral Feature Learning For Instrumental Variable Regression](demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)**

:   推导了谱特征学习在工具变量(IV)回归中的泛化界，根据谱对齐和特征值衰减率将性能分为"好/坏/丑"三类，并提出数据驱动的诊断方法。

**[Do-Pfn In-Context Learning For Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)**

:   提出 Do-PFN，将 Prior-data Fitted Networks (PFN) 扩展到因果效应估计，在大量合成 SCM 数据上预训练 Transformer 进行 in-context 因果推理，仅需观测数据即可预测干预分布（CID）和 CATE，无需因果图知识或不混杂假设，在合成和半合成实验中表现出色。

**[Few-Shot Knowledge Distillation Of Llms With Counterfactual Explanations](few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)**

:   提出 CoD（Counterfactual-explanation-infused Distillation），通过系统性地将反事实解释（CFE）注入少样本训练集——CFE 位于 teacher 决策边界附近，作为"boundary pegs"将 student 的决策面钉在 teacher 附近——在 6 个数据集上用仅 8-512 样本的超低数据量显著超越标准蒸馏方法。

**[From Black-box to Causal-box: Towards Building More Interpretable Models](from_black-box_to_causal-box_towards_building_more_interpretable_models.md)**

:   提出"因果可解释性"（causal interpretability）的形式化定义，证明黑盒模型和概念瓶颈模型均不满足该性质，给出完整的图判据确定哪些模型架构能一致地回答反事实问题，揭示了因果可解释性与预测精度之间的根本性权衡。

**[Revealing Multimodal Causality With Large Language Models](revealing_multimodal_causality_with_large_language_models.md)**

:   提出 MLLM-CD，首个面向多模态非结构化数据因果发现的框架，包含三个模块：(1) 对比因子发现利用 MLLM 从模态内/间交互中识别因果变量；(2) 统计因果结构发现；(3) 迭代多模态反事实推理利用 MLLM 的世界知识生成反事实样本来消除结构歧义。在合成和真实数据集上显著优于现有方法。
