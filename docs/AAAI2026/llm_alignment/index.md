---
title: >-
  AAAI2026 对齐/RLHF论文汇总 · 17篇论文解读
description: >-
  17篇AAAI2026的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF、LLM、对抗鲁棒、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "LLM"
  - "对抗鲁棒"
  - "推理"
item_list:
  - u: "align_to_structure_aligning_large_language_models_with_struc/"
    t: "Align to Structure: Aligning Large Language Models with Structural Information"
  - u: "aligntree_efficient_defense_against_llm_jailbreak_attacks/"
    t: "AlignTree: Efficient Defense Against LLM Jailbreak Attacks"
  - u: "amapo_adaptive_margin-attached_preference_optimization_for_l/"
    t: "AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment"
  - u: "biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan/"
    t: "BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models"
  - u: "decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen/"
    t: "DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF"
  - u: "differentiated_directional_intervention_a_framework_for_evading_llm_safety_align/"
    t: "Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment"
  - u: "ease_practical_and_efficient_safety_alignment_for_small_language_models/"
    t: "EASE: Practical and Efficient Safety Alignment for Small Language Models"
  - u: "enhancing_uncertainty_estimation_in_llms_with_expectation_of_aggregated_internal/"
    t: "Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal States"
  - u: "exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models/"
    t: "Exploring the Effects of Alignment on Numerical Bias in Large Language Models"
  - u: "gram-r2_self-training_generative_foundation_reward_models_for_reward_reasoning/"
    t: "GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning"
  - u: "importance-aware_data_selection_for_efficient_llm_instruction_tuning/"
    t: "Importance-Aware Data Selection for Efficient LLM Instruction Tuning"
  - u: "intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba/"
    t: "Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis"
  - u: "laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via/"
    t: "LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward"
  - u: "on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons/"
    t: "On the Exponential Convergence for Offline RLHF with Pairwise Comparisons"
  - u: "reducing_the_scope_of_language_models/"
    t: "Reducing the Scope of Language Models"
  - u: "w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_/"
    t: "W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search"
  - u: "when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf/"
    t: "When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🤖 AAAI2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (102)](../../ICLR2026/llm_alignment/index.md) · [💬 ACL2026 (38)](../../ACL2026/llm_alignment/index.md) · [🧪 ICML2026 (37)](../../ICML2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×9 · LLM ×8 · 对抗鲁棒 ×2 · 推理 ×2

**[Align to Structure: Aligning Large Language Models with Structural Information](align_to_structure_aligning_large_language_models_with_struc.md)**

:   提出 Structural Alignment 方法，通过将语言学篇章结构框架（表层文本结构评分 + 基于RST的篇章motif分类器）融入PPO强化学习训练，并设计基于篇章motif的密集奖励机制，使LLM生成更连贯、更具人类写作风格的长文本，在论文写作和长文档摘要任务上均优于标准RLHF模型。

**[AlignTree: Efficient Defense Against LLM Jailbreak Attacks](aligntree_efficient_defense_against_llm_jailbreak_attacks.md)**

:   AlignTree 利用 LLM 内部激活特征（线性 refusal direction + 非线性 SVM 信号）训练轻量级随机森林分类器，在几乎不增加计算开销的情况下高效检测越狱攻击，实现了 SOTA 的攻击成功率（ASR）降低效果。

**[AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](amapo_adaptive_margin-attached_preference_optimization_for_l.md)**

:   提出AMaPO算法，通过实例级自适应margin（结合Z-normalization和指数缩放）动态调节梯度幅度，解决DPO等离线偏好优化方法中对已正确排序样本过拟合、对错误排序样本欠拟合的核心矛盾，显著提升排序准确率和下游对齐性能。

**[BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)**

:   揭示LLM安全对齐中引入的伦理偏见可被反向利用作为越狱攻击向量——边缘化群体关键词的越狱成功率比优势群体高出20%，并提出基于提示词的轻量防御方法BiasDefense。

**[DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)**

:   DeCoRL 将 CoT 推理从单体顺序处理转变为"交响乐团式"的模块化并行协作——9 个专用子模型（解析/语义/实体/事实核查/风格/质量/计算/验证/整合）并行生成推理子步骤，通过双重奖励归因（本地质量+贡献度）+ 级联 DRPO 优化协调，在 RM-Bench 上达到 80.8%（超越所有基线），同时实现 3.8 倍推理加速和 22.7% 的可解释性提升。

**[Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)**

:   将 LLM 安全对齐的内部表征从传统的"单一拒绝方向"解构为功能独立的"危害检测方向"和"拒绝执行方向"，在此基础上提出 DBDI 框架，分别用自适应投影消除和直接引导两种策略精准干预两个方向，在 Llama-2 上实现 97.88% 的越狱成功率。

**[EASE: Practical and Efficient Safety Alignment for Small Language Models](ease_practical_and_efficient_safety_alignment_for_small_language_models.md)**

:   提出 EASE——面向边缘部署小语言模型（SLM）的安全对齐框架，通过两阶段设计解决"浅层拒绝不够安全 vs 深度推理太贵"的矛盾：第一阶段从大型推理模型蒸馏安全推理能力到 SLM，第二阶段用选择性推理激活（仅对脆弱语义区域的对抗查询启用推理，良性查询直接响应），越狱攻击成功率降低 17%（vs 浅层对齐）同时推理开销降低 90%（vs 全推理）。

**[Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal States](enhancing_uncertainty_estimation_in_llms_with_expectation_of_aggregated_internal.md)**

:   提出EAGLE方法，通过聚合LLM多个中间层隐藏状态的logits并计算置信度分布的期望值来估计不确定性，无需训练额外参数，在多个数据集和模型上ECE从12.6%降至3.2%，AUROC从59.0%提升至61.6%。

**[Exploring the Effects of Alignment on Numerical Bias in Large Language Models](exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)**

:   系统揭示了LLM对齐过程（指令调优+偏好调优）是LLM评估器产生数值偏差的根本原因，并验证分数范围调整是最有效的缓解策略。

**[GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning](gram-r2_self-training_generative_foundation_reward_models_for_reward_reasoning.md)**

:   本文提出 GRAM-R²，一个通过自训练方式在无标签数据上引发奖励推理能力的生成式基础奖励模型，能够同时产生偏好标签和推理理由，在响应排序、任务适配和 RLHF 等多个下游任务中一致超越判别式和生成式基线。

**[Importance-Aware Data Selection for Efficient LLM Instruction Tuning](importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)**

:   提出MIWV（Model Instruction Weakness Value）指标，通过比较LLM在有/无one-shot ICL示例下的损失差来衡量每条指令数据对模型能力提升的重要性，在Alpaca数据集上仅用1%（520条）数据即全面超越全量52002条的微调效果。

**[Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)**

:   本文将 AI 对齐形式化为 $\langle M,N,\varepsilon,\delta\rangle$-agreement 多目标优化问题，从通信复杂度角度证明了对齐的信息论下界（编码"所有人类价值观"本质上不可行），同时给出了无界/有界理性智能体的显式可达算法和紧致上界，揭示了在大状态空间下 reward hacking 全局不可避免的理论根基。

**[LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)**

:   提出 LaF-GRPO 框架，利用 LLM 模拟视障用户对导航指令的响应作为奖励信号，通过 GRPO 后训练 VLM 来生成更精确、更安全的视障导航指令，并构建了 27k 样本的 NIG4VI 基准数据集。

**[On the Exponential Convergence for Offline RLHF with Pairwise Comparisons](on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)**

:   在离线RLHF的成对比较设定下，提出RL-LOW算法实现了simple regret的指数收敛 $\exp(-\Omega(n/H))$，并首次导出实例依赖下界证明该速率在指数意义上是最优的。

**[Reducing the Scope of Language Models](reducing_the_scope_of_language_models.md)**

:   系统评估 LLM "范围限制"（scoping）方法——让部署在特定用途的 LLM 只响应域内查询、拒绝所有域外请求。在 3 个模型家族×多种任务上比较 prompting / SFT / DPO / 探针 / Circuit Breakers (CB)，发现 SFT 在高数据多样性下最强、CB 在低多样性下最强、分层组合 (SFT→CB) 保留两者优势——关键发现是范围限制的可行性高度依赖训练数据多样性。

**[W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)**

:   提出 W2S-AlignTree，首个将蒙特卡洛树搜索（MCTS）与弱到强泛化（W2SG）范式结合的推理时对齐框架，利用弱模型的步级代理值函数实时引导强模型生成，在情感控制、摘要、指令遵循任务上均显著超越基线，其中 Llama3-8B 摘要任务提升 15.9%。

**[When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)**

:   针对人类偏好标注中普遍存在的"偏好翻转"问题，提出 FA-DPO（Flipping-Aware DPO），将标注过程建模为"真实意图 + 实例依赖翻转概率"两阶段，通过修正 BT 模型损失和迭代优化翻转估计模块，在多种噪声场景下显著提升对齐鲁棒性，实例依赖翻转率高时比 DPO 提升 16.7%。
