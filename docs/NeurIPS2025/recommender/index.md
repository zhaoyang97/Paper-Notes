---
title: >-
  NeurIPS2025 推荐系统论文汇总 · 24篇论文解读
description: >-
  24篇NeurIPS2025的推荐系统方向论文解读，涵盖 LLM、推荐系统、人脸/视线、个性化生成、对齐/RLHF、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "NeurIPS2025"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "人脸/视线"
  - "个性化生成"
  - "对齐/RLHF"
  - "推理"
item_list:
  - u: "asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin/"
    t: "ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training"
  - u: "balancing_performance_and_costs_in_best_arm_identification/"
    t: "Balancing Performance and Costs in Best Arm Identification"
  - u: "empathia_multi-faceted_human-ai_collaboration_for_refugee_integration/"
    t: "EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration"
  - u: "estimating_hitting_times_locally_at_scale/"
    t: "Estimating Hitting Times Locally At Scale"
  - u: "face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm/"
    t: "FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens"
  - u: "inference-time_reward_hacking_in_large_language_models/"
    t: "Inference-Time Reward Hacking in Large Language Models"
  - u: "measuring_what_matters_construct_validity_in_large_language_model_benchmarks/"
    t: "Measuring What Matters: Construct Validity in Large Language Model Benchmarks"
  - u: "mmpb_its_time_for_multi-modal_personalization/"
    t: "MMPB: It's Time for Multi-Modal Personalization"
  - u: "neurips_should_lead_scientific_consensus_on_ai_policy/"
    t: "NeurIPS Should Lead Scientific Consensus on AI Policy"
  - u: "overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning/"
    t: "Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning"
  - u: "pac-bayes_bounds_for_multivariate_linear_regression_and_linear_autoencoders/"
    t: "PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders"
  - u: "position_towards_bidirectional_human-ai_alignment/"
    t: "Position: Towards Bidirectional Human-AI Alignment"
  - u: "r2ec_towards_large_recommender_models_with_reasoning/"
    t: "R²ec: Towards Large Recommender Models with Reasoning"
  - u: "radial_neighborhood_smoothing_recommender_system/"
    t: "Radial Neighborhood Smoothing Recommender System"
  - u: "semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation/"
    t: "Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation"
  - u: "the_coming_crisis_of_multi-agent_misalignment_ai_alignment_must_be_a_dynamic_and/"
    t: "The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process"
  - u: "the_more_you_automate_the_less_you_see_hidden_pitfalls_of_ai_scientist_systems/"
    t: "The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems"
  - u: "think_before_recommendation_autonomous_reasoning-enhanced_recommender/"
    t: "Think before Recommendation: Autonomous Reasoning-enhanced Recommender"
  - u: "transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning/"
    t: "Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning"
  - u: "tv-rec_time-variant_convolutional_filter_for_sequential_recommendation/"
    t: "TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation"
  - u: "validating_llm-as-a-judge_systems_under_rating_indeterminacy/"
    t: "Validating LLM-as-a-Judge Systems under Rating Indeterminacy"
  - u: "visuallens_personalization_through_task-agnostic_visual_history/"
    t: "VisualLens: Personalization through Task-Agnostic Visual History"
  - u: "who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re/"
    t: "Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation"
  - u: "wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin/"
    t: "Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🧠 NeurIPS2025** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (24)](../../ICLR2026/recommender/index.md) · [💬 ACL2026 (22)](../../ACL2026/recommender/index.md) · [🧪 ICML2026 (11)](../../ICML2026/recommender/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/recommender/index.md) · [🧪 ICML2025 (17)](../../ICML2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

🔥 **高频主题：** LLM ×8 · 推荐系统 ×6 · 人脸/视线 ×3 · 个性化生成 ×2 · 对齐/RLHF ×2

**[ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training](asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin.md)**

:   ASAP 是一个多 Agent 系统（Coordinator + Analyzer + Proposal），自动化诊断大规模 LLM 分布式训练的瓶颈类型（计算/内存/通信）并提出 sharding 配置方案，在 3 个实验场景中均匹配人类专家方案，实现最高 2.58× 吞吐量提升。

**[Balancing Performance and Costs in Best Arm Identification](balancing_performance_and_costs_in_best_arm_identification.md)**

:   提出将最优臂识别（BAI）从固定预算/固定置信度框架重新定义为"误识别概率/简单遗憾 + 采样成本"的风险泛函最小化问题，推导出含相变现象的下界（差距过小时最优策略是直接猜），设计 DBCARE 算法在动态预算下达到对数因子内最优。

**[EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)**

:   提出EMPATHIA多Agent框架，基于Kegan建构性发展理论，通过情感/文化/伦理三个专业化Agent的选择器-验证器协商评估难民安置建议，在6,359名难民的真实数据上达到87.4%收敛率和92.1%文化专家同意率。

**[Estimating Hitting Times Locally At Scale](estimating_hitting_times_locally_at_scale.md)**

:   提出两种局部（亚线性）算法估计图上的命中时间——基于相遇时间的 Algorithm 1 和基于谱截断的 Algorithm 3，无需全图访问仅通过以 $u,v$ 为中心的短随机游走完成估计，在合成和真实图上相对误差 <1.4%，并证明了游走采样的最优样本复杂度下界。

**[FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)**

:   FACE 提出将协同过滤（CF）嵌入通过解纠缠投影 + 残差量化映射为 LLM 预训练 token（描述符），再用对比学习对齐语义，无需微调 LLM 即可实现 CF 嵌入的语义解读和推荐性能增强。

**[Inference-Time Reward Hacking in Large Language Models](inference-time_reward_hacking_in_large_language_models.md)**

:   本文从数学上证明了推理时对齐方法（如 BoN）在优化代理奖励时不可避免地会出现 reward hacking（真实奖励先升后降），提出了 Best-of-Poisson (BoP) 采样方法近似最优 KL-奖励折中分布，并设计了 HedgeTune 算法通过一维寻根找到最优推理时参数，在数学推理和人类偏好场景中有效缓解 reward hacking。

**[Measuring What Matters: Construct Validity in Large Language Model Benchmarks](measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)**

:   本文由29位专家对445篇LLM benchmark论文进行系统性综述，从构念效度 (construct validity) 角度审视现有LLM评测基准在现象定义、任务设计、评分指标和结论声明方面的不足，并提出8条改进建议。

**[MMPB: It's Time for Multi-Modal Personalization](mmpb_its_time_for_multi-modal_personalization.md)**

:   提出首个 VLM 个性化评测基准 MMPB，包含 111 个可个性化概念、10k+ 图文问答对和 15 种任务类型，评测了 23 个 VLM 后发现即使最强的 GPT-4o 在个性化任务上也表现不佳，揭示了 VLM 在偏好推理、视觉线索利用和安全对齐与个性化的冲突等方面的重大局限。

**[NeurIPS Should Lead Scientific Consensus on AI Policy](neurips_should_lead_scientific_consensus_on_ai_policy.md)**

:   本文是一篇立场论文，主张 NeurIPS 应主动承担 AI 政策领域的科学共识形成角色，借鉴 IPCC（政府间气候变化专门委员会）在气候科学中的成功经验，填补当前 AI 政策领域共识机制的空白。

**[Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)**

:   识别 Crosscoder 中 L1 损失引入的两类稀疏性伪影（Complete Shrinkage 将弱共享概念错误归零、Latent Decoupling 将共享概念拆分为虚假模型特定潜变量），提出 Latent Scaling 诊断方法和 BatchTopK Crosscoder 替代方案，显著提升 chat-tuning 概念发现的可靠性。

**[PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders](pac-bayes_bounds_for_multivariate_linear_regression_and_linear_autoencoders.md)**

:   本文将PAC-Bayes泛化界从单输出线性回归推广到**多变量线性回归**，并进一步适配到推荐系统中的**线性自编码器（LAE）**，通过理论方法将计算复杂度从O(n⁴)降到O(n³)，实验证明该界是紧的且与Recall@K、NDCG@K等实际指标高度相关。

**[Position: Towards Bidirectional Human-AI Alignment](position_towards_bidirectional_human-ai_alignment.md)**

:   本文提出**双向人机对齐（Bidirectional Human-AI Alignment）**框架，从系统综述400+篇论文出发，论证AI对齐不应仅是单向地"让AI对齐人类"，还应包括"让人类适应AI"这一被严重忽视的方向，并识别了当前研究的关键缺口。

**[R²ec: Towards Large Recommender Models with Reasoning](r2ec_towards_large_recommender_models_with_reasoning.md)**

:   提出R²ec，首个将推理能力内生地集成到推荐模型中的统一大推荐模型，通过双头架构实现推理链生成与高效物品预测的一体化，并设计RecPO强化学习框架在无推理标注数据下联合优化推理与推荐目标。

**[Radial Neighborhood Smoothing Recommender System](radial_neighborhood_smoothing_recommender_system.md)**

:   提出 Radial Neighborhood Estimator (RNE)，通过将隐空间距离用观测矩阵的行/列 L2 范数近似估计，构建同时包含重叠和部分重叠用户-物品对的径向邻域，用局部核回归做平滑插补，在理论保证和实验中均优于传统协同过滤和矩阵分解方法，并天然缓解冷启动问题。

**[Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)**

:   提出SRA-CL框架，利用LLM的语义理解能力构建高质量对比样本对，通过语义检索+可学习样本合成器增强序列推荐的对比学习，以即插即用的方式在4个数据集上取得SOTA。

**[The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process](the_coming_crisis_of_multi-agent_misalignment_ai_alignment_must_be_a_dynamic_and.md)**

:   立场论文，主张多智能体系统（MAS）中的 AI 对齐应被视为动态的、依赖交互的社会过程，而非孤立问题；借鉴社会科学理论分析了社会结构如何破坏群体和个体价值，并呼吁 AI 社区建立专门的仿真环境、基准测试和评估框架来应对这一挑战。

**[The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems](the_more_you_automate_the_less_you_see_hidden_pitfalls_of_ai_scientist_systems.md)**

:   本文系统性地识别了当前 AI 科学家系统的四种方法论陷阱（不当基准选择、数据泄漏、指标误用、事后选择偏差），通过精心设计的合成任务 SPR 对 Agent Laboratory 和 The AI Scientist v2 进行受控实验，发现两个系统均存在不同程度的问题，并证明审计 trace log + 代码比仅审查最终论文的检测准确率高 27 个百分点（82% vs 55%）。

**[Think before Recommendation: Autonomous Reasoning-enhanced Recommender](think_before_recommendation_autonomous_reasoning-enhanced_recommender.md)**

:   提出 RecZero（纯 RL 范式）和 RecOne（SFT+RL 混合范式），抛弃传统的 teacher-student 蒸馏方法，用 GRPO 强化学习直接训练单个 LLM 自主发展推理能力进行评分预测，通过结构化 "Think-before-Recommendation" 模板引导分步推理（分析用户→分析物品→匹配→评分），在 4 个数据集上显著超越现有基线。

**[Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)**

:   提出 Transformer Copilot 框架，在 LLM 微调过程中系统记录"错误日志"(Mistake Log)，训练一个辅助 Copilot 模型学习 Pilot 的错误模式，推理时通过 logits 修正提升生成质量，在 12 个基准上最高提升 34.5%。

**[TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)**

:   提出 TV-Rec，基于图信号处理的时变卷积滤波器替代传统固定卷积和自注意力机制，实现更高表达力的序列推荐，在 6 个基准数据集上平均提升 7.49%。

**[Validating LLM-as-a-Judge Systems under Rating Indeterminacy](validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)**

:   提出在评分不确定性 (rating indeterminacy) 条件下验证 LLM-as-a-Judge 系统的框架，通过 "response set" 多标签评分方案替代强制选择评分，使选出的 judge 系统性能提升高达 31%。

**[VisualLens: Personalization through Task-Agnostic Visual History](visuallens_personalization_through_task-agnostic_visual_history.md)**

:   提出VisualLens框架，利用用户日常拍摄的与任务无关的视觉历史(task-agnostic visual history)，通过频谱用户画像(spectrum user profile)和多模态大模型实现跨领域个性化推荐，在新建的Google Review-V和Yelp-V数据集上Hit@3超越GPT-4o 2-5%。

**[Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)**

:   提出 TagCF 框架，通过 MLLM 提取用户角色标签和物品话题标签，再用 LLM 推理构建 U2I/I2U 逻辑图（用户角色与物品类型的因果关联），辅以标签编码器、对比学习增强和逻辑推理评分三种集成策略增强推荐，在亿级用户的工业在线A/B测试中互动指标提升0.946%、多样性提升0.102%，离线实验NDCG@10提升8.06%。

**[Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints](wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)**

:   提出 MAoP（Multiple Aspects of Planning）框架赋予 LLM "宽视野思维"能力，通过策略师预规划与路由机制并行整合多方面约束，配合 Travel-Sim 因果模拟评估基准，在旅行规划任务上大幅超越 CoT/分解方法，蒸馏后 3B 模型 PER 达 66.9%。
