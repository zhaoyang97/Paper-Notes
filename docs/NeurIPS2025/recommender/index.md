<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🧠 NeurIPS2025** · 共 **11** 篇

**[ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training](asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin.md)**

:   ASAP 是一个多 Agent 系统（Coordinator + Analyzer + Proposal），自动化诊断大规模 LLM 分布式训练的瓶颈类型（计算/内存/通信）并提出 sharding 配置方案，在 3 个实验场景中均匹配人类专家方案，实现最高 2.58× 吞吐量提升。

**[Balancing Performance and Costs in Best Arm Identification](balancing_performance_and_costs_in_best_arm_identification.md)**

:   提出将最优臂识别（BAI）从固定预算/固定置信度框架重新定义为"误识别概率/简单遗憾 + 采样成本"的风险泛函最小化问题，推导出含相变现象的下界（差距过小时最优策略是直接猜），设计 DBCARE 算法在动态预算下达到对数因子内最优。

**[EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)**

:   提出EMPATHIA多Agent框架，基于Kegan建构性发展理论，通过情感/文化/伦理三个专业化Agent的选择器-验证器协商评估难民安置建议，在6,359名难民的真实数据上达到87.4%收敛率和92.1%文化专家同意率。

**[Estimating Hitting Times Locally At Scale](estimating_hitting_times_locally_at_scale.md)**

:   提出两种局部（亚线性）算法估计图上的命中时间——基于相遇时间的 Algorithm 1 和基于谱截断的 Algorithm 3，无需全图访问仅通过以 $u,v$ 为中心的短随机游走完成估计，在合成和真实图上相对误差 <1.4%，并证明了游走采样的最优样本复杂度下界。

**[MMPB: It's Time for Multi-Modal Personalization](mmpb_its_time_for_multi-modal_personalization.md)**

:   提出首个 VLM 个性化评测基准 MMPB，包含 111 个可个性化概念、10k+ 图文问答对和 15 种任务类型，评测了 23 个 VLM 后发现即使最强的 GPT-4o 在个性化任务上也表现不佳，揭示了 VLM 在偏好推理、视觉线索利用和安全对齐与个性化的冲突等方面的重大局限。

**[Overcoming Sparsity Artifacts In Crosscoders To Interpret Chat-Tuning](overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)**

:   识别Crosscoder L1训练中的稀疏性伪影导致虚假模型特定潜变量归因，提出BatchTopK损失+Latent Scaling揭示真正的chat特定概念。

**[Radial Neighborhood Smoothing Recommender System](radial_neighborhood_smoothing_recommender_system.md)**

:   提出 Radial Neighborhood Estimator (RNE)，通过将隐空间距离用观测矩阵的行/列 L2 范数近似估计，构建同时包含重叠和部分重叠用户-物品对的径向邻域，用局部核回归做平滑插补，在理论保证和实验中均优于传统协同过滤和矩阵分解方法，并天然缓解冷启动问题。

**[Think before Recommendation: Autonomous Reasoning-enhanced Recommender](think_before_recommendation_autonomous_reasoning-enhanced_recommender.md)**

:   提出 RecZero（纯 RL 范式）和 RecOne（SFT+RL 混合范式），抛弃传统的 teacher-student 蒸馏方法，用 GRPO 强化学习直接训练单个 LLM 自主发展推理能力进行评分预测，通过结构化 "Think-before-Recommendation" 模板引导分步推理（分析用户→分析物品→匹配→评分），在 4 个数据集上显著超越现有基线。

**[Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)**

:   提出 Transformer Copilot 框架，在 LLM 微调过程中系统记录"错误日志"(Mistake Log)，训练一个辅助 Copilot 模型学习 Pilot 的错误模式，推理时通过 logits 修正提升生成质量，在 12 个基准上最高提升 34.5%。

**[Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)**

:   提出 TagCF 框架，通过 MLLM 提取用户角色标签和物品话题标签，再用 LLM 推理构建 U2I/I2U 逻辑图（用户角色与物品类型的因果关联），辅以标签编码器、对比学习增强和逻辑推理评分三种集成策略增强推荐，在亿级用户的工业在线A/B测试中互动指标提升0.946%、多样性提升0.102%，离线实验NDCG@10提升8.06%。

**[Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints](wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)**

:   提出 MAoP（Multiple Aspects of Planning）框架赋予 LLM "宽视野思维"能力，通过策略师预规划与路由机制并行整合多方面约束，配合 Travel-Sim 因果模拟评估基准，在旅行规划任务上大幅超越 CoT/分解方法，蒸馏后 3B 模型 PER 达 66.9%。
