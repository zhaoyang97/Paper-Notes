<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🔬 ICLR2026** · 共 **19** 篇

**[A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)**

:   提出 A2D，一种针对扩散语言模型（dLLM）的 token 级安全对齐方法，通过训练模型在遇到有害内容的 mask 位置输出 [EOS] token 来实现任意解码顺序、任意解码步的安全防御，将 DIJA 模板攻击成功率从 80%+ 降到近零（1.3%/0.0%），并支持早期拒绝实现 19.3x 加速。

**[Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)**

:   提出 Multi-Lingual Consistency (MLC) 辅助损失，通过 SVD 操控多语言表示矩阵的奇异值使其趋向秩-1（即多语言表示共线），仅需多语言 prompt 翻译（无需目标语言的 response），即可将一种语言的安全对齐效果一致性地迁移到所有语言。

**[Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)**

:   提出MetaAPO框架，用一个轻量级meta-learner（两层MLP）动态估计offline/online数据的对齐差距，既指导"在哪些prompt上做在线采样"（解决分布不匹配），又在训练时自适应加权offline/online数据（优化学习效率），在AlpacaEval 2/Arena-Hard/MT-Bench上超越DPO/Online DPO等基线，同时减少42%在线标注成本。

**[AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)**

:   提出 AlphaSteer，通过学习一个受零空间约束的变换矩阵来动态构造 steering 向量，对良性输入产生近零向量（保持效用），对恶意输入重建拒绝方向向量（增强安全），在理论上保证了安全与效用的解耦。

**[Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)**

:   提出 RCPO 框架，将 LLM 对齐从成对偏好扩展到排名选择（ranked choice）建模，通过 MLE 统一了效用模型（MNL）和排名模型（Mallows-RMJ），在 single-best 和 top-k 反馈格式下都优于 DPO 及其变体。

**[Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)**

:   提出基于社会选择理论公理的偏好学习框架，从成对比较数据中推断评估者人群分布的可行集，构造满足人群比例对齐(PPA)和人群有界可操纵性(PBM)公理的策略。

**[CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)**

:   提出 CAGE 框架，通过 Semantic Mold（语义模具）将红队攻击 prompt 的对抗结构与文化内容解耦，能系统性地将英语红队基准适配到不同文化语境中，生成的文化扎根 prompt 比直接翻译的 ASR 显著更高。

**[Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)**

:   发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。

**[Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)**

:   提出 Dual-IPO 框架，通过在奖励模型和视频生成模型之间进行多轮双向迭代优化，无需大量人工标注即可持续提升文本到视频生成的质量和人类偏好对齐，甚至让 2B 模型超越 5B 模型。

**[From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)**

:   提出**ALPO**（自适应局部偏好优化），通过segment-wise采样和细粒度对齐损失训练表达力强的字幕翻译LLM，解决传统DPO在多段局部偏好对齐中梯度稀释的问题。

**[Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)**

:   通过 first-principles 推导揭示 group-relative REINFORCE（如 GRPO）天然具有 off-policy 解释，无需假设数据采样分布。发现 clipping 而非 importance sampling 是稳定性的关键，提出 REC 系列算法统一解释 GRPO、Kimi OPMD 和 Meta AsymRE。

**[Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)**

:   揭示了 stepwise group-based RL（如 GRPO/GiGPO）中的「历史上下文不一致」问题——同一 group 内的 step 可能具有不同历史上下文导致 advantage 估计偏差，提出 HGPO 通过层次化分组和自适应加权实现低偏差、平衡方差的 advantage 估计，在 ALFWorld 和 WebShop 上以极低额外开销（<0.001%）取得显著提升。

**[Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)**

:   提出**D3S**（Dynamic Dual-Level Down-Sampling）框架，在sample层最大化advantage方差、在token层优先选取高熵+高advantage的token，配合动态调度策略，用不到20% token实现更快收敛和更优性能。

**[Mitigating Mismatch within Reference-based Preference Optimization](mitigating_mismatch_within_reference-based_preference_optimization.md)**

:   揭示 DPO 的"悲观 reference 偏差"问题——当 reference 策略对 chosen response 的概率低于 rejected 时（~45% pairs），DPO 会"过早满足"停止学习；提出 HyPO（一行代码修改：将 $\Delta_{ref}$ 裁剪为 $\max(0, \Delta_{ref})$），在 AlpacaEval 上相对 DPO 提升 41.2%。

**[Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)**

:   提出 ReSA（Reasoned Safety Alignment），通过"先回答后检查"范式——模型先总结意图答案，再进行策略驱动的安全分析，最后输出最终回复——在 13 种防御方法中安全性最优，同时维持 MMLU/MATH500/HumanEval 性能，仅 500 样本即达全数据集效果。

**[PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](reinforcement_unlearning_via_group_relative_policy_optimization.md)**

:   PURGE 将 LLM 遗忘（unlearning）重新定义为可验证的 RL 任务，使用 GRPO 框架 + 内在奖励信号（惩罚提及禁止概念）来实现安全一致的知识删除，token 消耗比 SOTA 低 46 倍，同时提升流畅度 +5.48% 和对抗鲁棒性 +12.02%。

**[SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](safedpo_preference_optimization_safety.md)**

:   重新审视安全约束 RLHF 目标并证明其存在闭式最优策略，据此推导出等价的可处理目标 SafeDPO，仅需在标准 DPO 上加入安全感知数据变换和安全 margin 项（1 个额外超参数），无需奖励/代价模型，在 PKU-SafeRLHF-30K 上实现 96.87% 无害率且保持竞争力的有用性，训练速度比 SafeRLHF 快 25×。

**[Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)**

:   提出 Uni-DPO，通过质量感知加权（高间距优先）和性能感知加权（focal loss 聚焦欠拟合样本）双视角动态调整 DPO 偏好对权重，Gemma-2-9B 在 Arena-Hard 上超 Claude 3 Opus 6.7 分。

**[Why DPO is a Misspecified Estimator and How to Fix It](why_dpo_is_misspecified_estimator.md)**

:   从信息几何角度证明 DPO 在参数化（非 tabular）策略类下本质上是一个误指定的统计估计问题——DPO 将真实奖励函数 KL 投影到隐式奖励流形上，当奖励不可实现时会导致偏好反转和奖励下降——并提出 AuxDPO 通过引入零空间辅助变量来修复此问题。
