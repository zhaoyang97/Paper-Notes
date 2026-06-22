---
title: >-
  ICML2025 推荐系统论文汇总 · 17篇论文解读
description: >-
  17篇ICML2025的推荐系统方向论文解读，涵盖推荐系统、对齐/RLHF、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "LLM"
item_list:
  - u: "adaptive_elicitation_of_latent_information_using_natural_language/"
    t: "Adaptive Elicitation of Latent Information Using Natural Language"
  - u: "aligning_llms_by_predicting_preferences_from_user_writing_samples/"
    t: "Aligning LLMs by Predicting Preferences from User Writing Samples"
  - u: "deprecating_benchmarks_criteria_and_framework/"
    t: "Deprecating Benchmarks: Criteria and Framework"
  - u: "elmo_efficiency_via_low-precision_and_peak_memory_optimization_in_large_output_s/"
    t: "ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces"
  - u: "how_to_set_adamws_weight_decay_as_you_scale_model_and_dataset_size/"
    t: "How to Set AdamW's Weight Decay as You Scale Model and Dataset Size"
  - u: "learning_cascade_ranking_as_one_network/"
    t: "LCRON: Learning Cascade Ranking as One Network"
  - u: "new_interaction_paradigm_for_complex_eda_software_leveraging_gpt/"
    t: "New Interaction Paradigm for Complex EDA Software Leveraging GPT"
  - u: "not_all_explanations_for_deep_learning_phenomena_are_equally_valuable/"
    t: "Not All Explanations for Deep Learning Phenomena Are Equally Valuable"
  - u: "parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew/"
    t: "PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model"
  - u: "position_dont_use_the_clt_in_llm_evals_with_fewer_than_a_few_hundred_datapoints/"
    t: "Position: Don't Use the CLT in LLM Evals with Fewer Than a Few Hundred Datapoints"
  - u: "qure_query-relevant_retrieval_through_hard_negative_sampling_in_composed_image_r/"
    t: "QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval"
  - u: "recommendations_and_reporting_checklist_for_rigorous_transparent_human_baselines/"
    t: "Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations"
  - u: "recommendations_with_sparse_comparison_data_provably_fast_convergence_for_noncon/"
    t: "Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization"
  - u: "rlthf_targeted_human_feedback_for_llm_alignment/"
    t: "RLTHF: Targeted Human Feedback for LLM Alignment"
  - u: "simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod/"
    t: "SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning"
  - u: "the_right_to_ai/"
    t: "Position: The Right to AI"
  - u: "toward_safe_and_human-aligned_game_conversational_recommendation_via_multi-agent/"
    t: "MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🧪 ICML2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (24)](../../ICLR2026/recommender/index.md) · [💬 ACL2026 (22)](../../ACL2026/recommender/index.md) · [🧪 ICML2026 (11)](../../ICML2026/recommender/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/recommender/index.md) · [🧠 NeurIPS2025 (24)](../../NeurIPS2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

🔥 **高频主题：** 推荐系统 ×3 · 对齐/RLHF ×2 · LLM ×2

**[Adaptive Elicitation of Latent Information Using Natural Language](adaptive_elicitation_of_latent_information_using_natural_language.md)**

:   提出一种基于 LLM 的自适应信息获取框架，通过元学习预测模型对未来观测进行自回归前向模拟，量化并区分认知不确定性与偶然不确定性，自适应选择最具信息量的自然语言问题来高效减少对潜变量实体的认知不确定性。

**[Aligning LLMs by Predicting Preferences from User Writing Samples](aligning_llms_by_predicting_preferences_from_user_writing_samples.md)**

:   提出通过分析用户写作样本预测其偏好来实现个性化 LLM 对齐的新范式，无需显式偏好标注即可从用户文本风格中推断偏好信号，为个性化对齐开辟了新的数据来源。

**[Deprecating Benchmarks: Criteria and Framework](deprecating_benchmarks_criteria_and_framework.md)**

:   提出了一套判断 AI 基准何时应被废弃的 **7 项标准** 和一个包含评估-报告-通知三阶段的 **废弃框架**，并以 EU AI Office 为例给出了制度化落地方案。

**[ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces](elmo_efficiency_via_low-precision_and_peak_memory_optimization_in_large_output_s.md)**

:   提出 ELMO 框架，通过纯 BFloat16/Float8 低精度训练结合梯度融合、分块策略等峰值显存优化，将 300 万标签的 XMC 模型训练显存从 39.7 GiB 降至 6.6 GiB，且不损失分类精度。

**[How to Set AdamW's Weight Decay as You Scale Model and Dataset Size](how_to_set_adamws_weight_decay_as_you_scale_model_and_dataset_size.md)**

:   将 AdamW 的权重更新解释为指数移动平均（EMA），揭示了 EMA 时间尺度 $\tau = 1/(\eta\lambda)$ 是核心超参数，其以 epoch 为单位的最优值在模型和数据集规模变化时保持稳定，从而给出了 weight decay 随规模缩放的明确规则。

**[LCRON: Learning Cascade Ranking as One Network](learning_cascade_ranking_as_one_network.md)**

:   提出LCRON，将多阶段级联排序系统作为统一网络进行端到端训练：通过可微排序技术构建的端到端代理损失$L_{e2e}$直接优化ground truth items通过整个级联的存活概率下界，辅以从下界紧致度推导出的各阶段辅助损失$L_{single}$驱动阶段间协同，在公开基准和工业广告系统的线上A/B测试中均取得显著提升（广告收入+4.10%，用户转化+1.60%）。

**[New Interaction Paradigm for Complex EDA Software Leveraging GPT](new_interaction_paradigm_for_complex_eda_software_leveraging_gpt.md)**

:   提出 SmartonAI 系统，将大语言模型（LLM）和检索增强生成（RAG）集成到 EDA 工具 KiCad 中，通过自然语言交互实现任务分解、文档检索和智能插件推荐与执行，大幅降低复杂工程软件的学习门槛。

**[Not All Explanations for Deep Learning Phenomena Are Equally Valuable](not_all_explanations_for_deep_learning_phenomena_are_equally_valuable.md)**

:   本文是一篇 position paper，主张深度学习中的"反直觉现象"（如 double descent、grokking、lottery ticket）在实际场景中很少出现，研究者不应追求对它们的孤立解释，而应将其作为检验和完善更广泛深度学习理论的实验场。

**[PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)**

:   提出 PARM 单一统一的偏好感知自回归奖励模型，通过 PBLoRA（Preference-Aware Bilinear Low-Rank Adaptation）将偏好向量条件化到 ARM 中，实现高效的多目标测试时对齐——用 1 个奖励模型替代 k 个独立 ARM，降低推理成本且支持弱到强引导（7B 引导 65B）。

**[Position: Don't Use the CLT in LLM Evals with Fewer Than a Few Hundred Datapoints](position_dont_use_the_clt_in_llm_evals_with_fewer_than_a_few_hundred_datapoints.md)**

:   本文作为立场论文，论证了在 LLM 评估数据量少于几百个样本时，基于中心极限定理 (CLT) 的置信区间严重低估不确定性，推荐使用贝叶斯可信区间或 Wilson 得分区间作为替代方案。

**[QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval](qure_query-relevant_retrieval_through_hard_negative_sampling_in_composed_image_r.md)**

:   提出 QuRe，通过基于相关性分数陡降的硬负样本采样策略和奖励模型优化目标，在组合图像检索(CIR)中同时召回目标图像和其他相关图像，从而提升用户满意度。

**[Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations](recommendations_and_reporting_checklist_for_rigorous_transparent_human_baselines.md)**

:   本文对 AI 评估中"人类基线"（human baseline）的方法论进行了系统审查，发现现有 115 项人类基线研究在严谨性和透明度方面存在严重不足，并提出了覆盖基线全生命周期的方法建议和报告清单。

**[Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization](recommendations_with_sparse_comparison_data_provably_fast_convergence_for_noncon.md)**

:   首次为基于成对比较数据的推荐系统非凸矩阵分解问题提供理论恢复保证：证明在warm start条件下，投影梯度下降以指数速率收敛到真实低秩特征矩阵，样本复杂度为近乎最优的 $O(nr^2 \log n)$，关键技术贡献是将matrix Bernstein不等式扩展到成对比较的采样矩阵结构。

**[RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)**

:   RLTHF 提出了一种人机混合的 LLM 对齐框架，通过分析奖励模型的奖励分布来识别 LLM 错标的"难样本"，仅对这些样本进行人工标注，以全量人工标注 6-7% 的成本达到甚至超越全人工标注的对齐质量。

**[SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)**

:   SIMPLEMIX 发现 on-policy 数据擅长推理任务而 off-policy 数据擅长开放式任务，通过简单地混合两类数据源即可在 Alpaca Eval 2.0 上平均提升 6.03%，超越 HyPO 等复杂方法 3.05%。

**[Position: The Right to AI](the_right_to_ai.md)**

:   本文是一篇 position paper，提出"AI 权利"（Right to AI）的概念，主张受 AI 系统影响的个人和社区应当有权参与 AI 的开发和治理，并借鉴城市规划中"城市权利"理论，构建了一个四层公民参与模型。

**[MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition](toward_safe_and_human-aligned_game_conversational_recommendation_via_multi-agent.md)**

:   提出 MATCHA 多 Agent 框架，将游戏对话推荐分解为六个专用 Agent（意图解析、工具增强候选生成、多 LLM 排序、反思重排、风险控制、可解释生成），在 Roblox 真实用户数据上 Hit@5 提升 20%、流行度偏差降 24%、对抗防御率 97.9%。
