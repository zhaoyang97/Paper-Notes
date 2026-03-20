<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🤖 AAAI2026** · 共 **12** 篇

**[A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)**

:   提出基于认知层次(cognitive hierarchy)的博弈论框架实现可计算的多智能体Theory of Mind，通过Poisson-Gamma共轭实现高效信念更新，避免求解不可判定的POMDP同时保持有限理性决策。

**[Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)**

:   实时估计合作伙伴的ToM阶数并与之对齐，使用FTL(O(logT)悬悔)或Hedge(O(√T logN)悬悔)算法实现零样本多智能体协作，在4个任务上鲁棒地与不同类型的伙伴协作。

**[Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](affordance-guided_coarse-to-fine_exploration_for_base_placem.md)**

:   针对开放词汇移动操控中机器人基座选位问题，提出一种零样本框架，通过构建跨模态表征（Affordance RGB + Obstacle Map+）将语义affordance线索投射到障碍物地图上，再用粗到细迭代优化平衡语义和几何约束，在5个操控任务上达到85%成功率，大幅超越几何规划器和纯VLM方法。

**[Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)**

:   从Vector Symbolic Architecture(VSA)视角解释Transformer的注意力和残差流作为近似的符号绑定/解绑定操作，解释了逻辑推理中的脆弱性（如变量混淆），并提出显式绑定头和超维记忆层作为改进方向。

**[Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](causal_inference_under_threshold_manipulation_bayesian_mixtu.md)**

:   提出 BMTM/HBMTM 贝叶斯混合模型框架，在消费者策略性操纵消费额以达到奖励阈值的场景下，通过将观测分布拆解为 bunching 与 non-bunching 两个子分布，准确估计阈值因果效应及跨子群的异质性处理效应。

**[EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)**

:   提出 EvoEmpirBench（EEB），包含两个动态交互式 benchmark（局部可观测迷宫导航 + 消消乐），以及 Agent-ExpVer 三智能体在线学习框架（GeoLink 交互 + InsightForce 经验抽象 + TruthWeaver 知识管理），通过"经验→验证→真理归纳"的认知循环实现无参数更新的持续策略进化，使 GPT-4.1 成功率提升 5.6%、Qwen-32B 提升 29%。

**[ISEAL: Encrypted Fingerprinting for Reliable LLM Ownership Verification](iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)**

:   提出ISEAL加密指纹方案用于可靠的LLM所有权验证，通过加密指纹嵌入和验证协议实现不可伪造、不可否认的模型所有权证明。

**[Neural Graph Navigation for Intelligent Subgraph Matching](neural_graph_navigation_for_intelligent_subgraph_matching.md)**

:   提出 NeuGN（Neural Graph Navigation）框架，首次将生成式神经导航集成到子图匹配的核心枚举阶段，通过 QSExtractor 提取查询图结构信号 + GGNavigator 将暴力枚举转为结构感知的候选节点优先排序，在保证完备性的同时将 First Match Steps 最高减少 98.2%。

**[Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)**

:   解决最大容量网格存储系统中的鲁棒乱序检索问题，提出高效算法在系统满载时仍能保证目标物品的可达性和检索效率。

**[Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)**

:   首次系统分析 LLM 多 Agent 软件开发系统（ChatDev/MetaGPT/AgentVerse）的安全风险：提出 IMBIA 攻击框架覆盖两种威胁场景（恶意用户+良性Agent / 良性用户+恶意Agent）和 12 种恶意行为（5 大恶意软件家族），攻击成功率高达 93%（ChatDev），并设计 Adv-IMBIA 对抗性防御将 ASR 降低 40-73%。

**[Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](towards_reinforcement_learning_from_neural_feedback_mapping_.md)**

:   提出 NEURO-LOOP 框架，利用 fNIRS（功能性近红外光谱）脑信号作为隐式神经反馈评估 RL agent 表现，发布 25 名被试 × 3 领域 × 6 条件的 fNIRS 数据集，分类 F1 达 67%（二分类）/ 46%（多分类），跨被试 fine-tuning 分别提升 17% 和 41%，奠定 Reinforcement Learning from Neural Feedback (RLNF) 基础。

**[Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)**

:   本文揭示了在良性 Agent 数据上微调 LLM 会导致意外的安全对齐偏移（攻击成功率增加 32-38%），并提出 PING（Prefix Injection Guard）——通过迭代生成+评估自然语言前缀来引导微调后的 Agent 拒绝有害请求，平均提升拒绝率 66%（Web）和 44%（代码），同时保持任务性能（仅降 1.8%）。
