---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🤖 AAAI2026** · 共 **8** 篇

**[Align to Structure: Aligning Large Language Models with Structural Information](align_to_structure_aligning_large_language_models_with_struc.md)**

:   提出 Structural Alignment 方法，通过将语言学篇章结构框架（表层文本结构评分 + 基于RST的篇章motif分类器）融入PPO强化学习训练，并设计基于篇章motif的密集奖励机制，使LLM生成更连贯、更具人类写作风格的长文本，在论文写作和长文档摘要任务上均优于标准RLHF模型。

**[AlignTree: Efficient Defense Against LLM Jailbreak Attacks](aligntree_efficient_defense_against_llm_jailbreak_attacks.md)**

:   AlignTree 利用 LLM 内部激活特征（线性 refusal direction + 非线性 SVM 信号）训练轻量级随机森林分类器，在几乎不增加计算开销的情况下高效检测越狱攻击，实现了 SOTA 的攻击成功率（ASR）降低效果。

**[AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](amapo_adaptive_margin-attached_preference_optimization_for_l.md)**

:   提出AMaPO算法，通过实例级自适应margin（结合Z-normalization和指数缩放）动态调节梯度幅度，解决DPO等离线偏好优化方法中对已正确排序样本过拟合、对错误排序样本欠拟合的核心矛盾，显著提升排序准确率和下游对齐性能。

**[DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)**

:   DeCoRL 将 CoT 推理从单体顺序处理转变为"交响乐团式"的模块化并行协作——9 个专用子模型（解析/语义/实体/事实核查/风格/质量/计算/验证/整合）并行生成推理子步骤，通过双重奖励归因（本地质量+贡献度）+ 级联 DRPO 优化协调，在 RM-Bench 上达到 80.8%（超越所有基线），同时实现 3.8 倍推理加速和 22.7% 的可解释性提升。

**[Exploring the Effects of Alignment on Numerical Bias in Large Language Models](exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)**

:   系统揭示了LLM对齐过程（指令调优+偏好调优）是LLM评估器产生数值偏差的根本原因，并验证分数范围调整是最有效的缓解策略。

**[Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)**

:   提出 MRMBench 基准，通过 6 个维度（无害性、有帮助性、正确性、连贯性、复杂性、冗长性）的探针任务评估奖励模型是否有效捕获多维偏好，发现探针性能与 PPO 对齐质量强相关（Pearson $r > 0.8$），并提出推理时探针方法将 AlpacaEval win rate 从 57.3% 提升至 62.5%。

**[SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)**

:   提出SceneJailEval，一个场景自适应的多维度越狱评估框架，定义14个越狱场景和10个评估维度，通过场景分类→维度动态选择→多维检测→加权危害评分的流程，在自建数据集上F1达0.917（超SOTA 6%），在JBB上达0.995（超SOTA 3%），同时支持危害程度量化而非仅二分类。

**[Towards Inference-Time Scaling for Continuous Space Reasoning](towards_inference-time_scaling_for_continuous_space_reasoning.md)**

:   首次系统研究离散文本推理中的inference-time scaling技术能否迁移到连续潜空间推理模型（COCONUT），发现dropout采样能生成多样推理路径（Pass@32达44.43%），但PRM/ORM仅带来不足2.3%提升，根因在于连续思维表示缺乏区分正误推理的几何归纳偏置。
