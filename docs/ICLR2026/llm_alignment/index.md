<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🔬 ICLR2026** · 共 **34** 篇

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

**[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)**

:   理论证明奖励过优化主要源于高奖励尾部区域的奖励模型错误规范，提出基于 rubric 的奖励建模方法：利用 off-policy 数据（强模型生成的优秀回复）构造评分细则，通过渐进式区分"优秀 vs 更优秀"来精细化 rubric，有效缓解奖励过优化。

**[Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)**

:   发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。

**[Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)**

:   提出 Dual-IPO 框架，通过在奖励模型和视频生成模型之间进行多轮双向迭代优化，无需大量人工标注即可持续提升文本到视频生成的质量和人类偏好对齐，甚至让 2B 模型超越 5B 模型。

**[From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)**

:   提出**ALPO**（自适应局部偏好优化），通过segment-wise采样和细粒度对齐损失训练表达力强的字幕翻译LLM，解决传统DPO在多段局部偏好对齐中梯度稀释的问题。

**[General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)**

:   理论证明现有 RLHF 探索奖励（exploratory bonus）在 KL 和 α-散度正则化下实际上会引导策略向参考模型的高概率区域靠拢（与乐观原则相悖），提出 General Exploratory Bonus (GEB) 框架——通过参考模型依赖的奖励调节来抵消散度正则化的保守偏差，可证明满足乐观原则。

**[Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)**

:   首次在实际规模 LLM（7B MoE）的近单遍预训练中验证 grokking 现象——不同数据组异步记忆、延迟泛化；通过分析 MoE routing pathway 的演化（从 instance-specific 到 structured/shared），提出两个零成本指标来监控泛化进度，无需 instruction tuning 和 benchmark 评估。

**[Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)**

:   通过 first-principles 推导揭示 group-relative REINFORCE（如 GRPO）天然具有 off-policy 解释，无需假设数据采样分布。发现 clipping 而非 importance sampling 是稳定性的关键，提出 REC 系列算法统一解释 GRPO、Kimi OPMD 和 Meta AsymRE。

**[GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)**

:   提出 GuardAlign，一个无需训练的多模态大模型推理时安全防御框架：用最优传输(OT)精确检测图像中的不安全区域并遮蔽，再通过跨模态注意力校准保持安全前缀的影响力不衰减，在6个LVLM上将不安全响应率降低最多39%，同时保持甚至提升通用能力。

**[Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)**

:   揭示了 stepwise group-based RL（如 GRPO/GiGPO）中的「历史上下文不一致」问题——同一 group 内的 step 可能具有不同历史上下文导致 advantage 估计偏差，提出 HGPO 通过层次化分组和自适应加权实现低偏差、平衡方差的 advantage 估计，在 ALFWorld 和 WebShop 上以极低额外开销（<0.001%）取得显著提升。

**[Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)**

:   挑战"on-policy数据总是更好"的共识：发现对齐过程分为偏好注入（需高多样性off-policy数据）和偏好微调（需高质量on-policy数据）两个阶段，不同模型/阶段对数据类型的最优选择不同。提出仅3.2%计算开销的边界判定算法，在5个模型×55个配置上验证有效。

**[Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)**

:   提出**D3S**（Dynamic Dual-Level Down-Sampling）框架，在sample层最大化advantage方差、在token层优先选取高熵+高advantage的token，配合动态调度策略，用不到20% token实现更快收敛和更优性能。

**[Learning Ordinal Probabilistic Reward from Preferences (OPRM)](learning_ordinal_probabilistic_reward_from_preferences.md)**

:   提出序数概率奖励模型(OPRM)，将响应质量离散化为1-9序数等级并学习完整概率分布，结合区域洪泛调优(RgFT)实现数据高效训练。在RewardBench达89.3%，比现有RM提升2.9%-7.4%，同时提供不确定性估计和标注分歧检测。

**[Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)**

:   提出 Intuitor，一种用模型自身置信度（self-certainty，即输出分布与均匀分布的 KL 散度）替代外部可验证奖励的 RLIF 方法，在数学推理上匹配 GRPO 性能，同时在代码生成等域外任务上展现更好的泛化能力。

**[Mitigating Mismatch within Reference-based Preference Optimization](mitigating_mismatch_within_reference-based_preference_optimization.md)**

:   揭示 DPO 的"过早满足"问题——当 reference 策略对 chosen 的概率低于 rejected 时（~45% pairs），DPO 的梯度被 reference 的悲观信号不必要地衰减（即使策略仍然错误即 $\Delta_\theta < 0$）；提出 HyPO（一行代码修改：$\max(0, \Delta_{ref})$ 裁剪 reference margin），在 AlpacaEval 2.0 上相对 DPO 提升 41.2%。

**[Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)**

:   提出 NSPO，将安全对齐的策略梯度投影到通用任务表征的零空间中，从几何层面保证安全优化不损害通用能力，仅用 40% 安全数据即在 7 个安全 benchmark 上达到 SOTA，同时在数学/代码/指令遵循上几乎无性能损失。

**[Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)**

:   提出"先回答后检查"(Answer-Then-Check)策略：模型先在思维链中生成意图答案摘要，再依据安全策略进行安全分析，最后决定输出或拒绝。构建80K ReSA数据集训练后，在7种越狱攻击上防御率达到99.3%(RL版本)，仅500样本即可达全数据集效果。

**[PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](reinforcement_unlearning_via_group_relative_policy_optimization.md)**

:   PURGE 将 LLM 遗忘（unlearning）重新定义为可验证的 RL 任务，使用 GRPO 框架 + 内在奖励信号（惩罚提及禁止概念）来实现安全一致的知识删除，token 消耗比 SOTA 低 46 倍，同时提升流畅度 +5.48% 和对抗鲁棒性 +12.02%。

**[SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](safedpo_preference_optimization_safety.md)**

:   重新审视安全约束 RLHF 目标并证明其存在闭式最优策略，据此推导出等价的可处理目标 SafeDPO，仅需在标准 DPO 上加入安全感知数据变换和安全 margin 项（1 个额外超参数），无需奖励/代价模型，在 PKU-SafeRLHF-30K 上实现 96.87% 无害率且保持竞争力的有用性，训练速度比 SafeRLHF 快 25×。

**[Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)**

:   提出Human-AI协同的两阶段偏好数据策展流程：第一阶段人工验证+错误驱动检索+偏好引导LLM标注迭代8轮积累1M对，第二阶段一致性过滤扩展到26M对。训练的Skywork-Reward-V2 8B模型在RewardBench达97.8%，在7个基准上平均88.6%超越所有开源70B模型。

**[Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)**

:   提出"浅层安全对齐假说"(SSAH)：安全对齐本质上是教模型做一个隐式的二分类任务（执行还是拒绝），只需约1.3%的神经元即可建立安全护栏；冻结这些安全关键单元可在微调时保持安全性，利用冗余单元作为"对齐预算"可消除对齐税。

**[Swap-guided Preference Learning for Personalized RLHF (SPL)](swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)**

:   解决变分偏好学习(VPL)中的后验崩坏问题：提出SPL，通过swap引导基础正则化(强制潜变量编码用户偏好而非被忽略)+Preferential-IAF分解swap可逆/不可逆信号+自适应潜变量调节。在Llama-3.1-8B上达63.71%准确率+97.10%活跃单元，而VPL崩坏到57.14%+0%。

**[Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)**

:   提出TI-DPO，通过梯度归因+高斯先验的混合权重机制精确量化每个token对偏好的贡献，结合三元组损失在连续语义空间引导优化，在6个基准上平均62.3分达到SOTA，同时具备可解释的token级控制能力。

**[Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)**

:   提出 UltraBreak，通过语义对抗目标（用cosine相似度替代交叉熵优化出平滑loss景观）+ 输入空间约束（随机变换+TV正则化产生变换不变特征），训练单张通用对抗图像即可跨6+个VLM架构和商业模型实现越狱，黑盒平均ASR达71%（SafeBench），远超此前方法。

**[Towards Understanding Valuable Preference Data for LLM Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)**

:   挑战“数据质量是固有属性”的假设：发现偏好数据价值是模型相关的——对一个模型有益的数据可能伤害另一个。提出截断影响函数(TIF)识别高价值数据，及高效的LossDiff-IRM代理，仅用50-64%数据即提升WinRate 13.58%。

**[Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)**

:   提出Uni-DPO，通过质量感知加权（高分差偏好对优先）+性能感知加权（focal loss聚焦欠拟合样本）+校准NLL损失三个组件统一动态调整DPO偏好对权重，在文本理解和数学推理基准上一致超越DPO/SimPO，Gemma-2-9B在Arena-Hard达67.1%超过Claude 3 Opus(60.4%)。

**[Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)**

:   提出DAR(Dual-regularized Advantage Regression)：发现标准RLHF中参考模型正则化(防reward hacking)和策略稳定约束(防崩溃)会逐步冲突导致优化空间过度受限，通过双KL目标在对数空间插值参考策略+回归变换消除策略比率不稳定性，在直接AI对齐和标准RLHF设置中达到92.42%平均胜率，超GRPO 7.27%。

**[Why DPO is a Misspecified Estimator and How to Fix It](why_dpo_is_misspecified_estimator.md)**

:   从信息几何角度证明 DPO 在参数化（非 tabular）策略类下本质上是一个误指定的统计估计问题——DPO 将真实奖励函数 KL 投影到隐式奖励流形上，当奖励不可实现时会导致偏好反转和奖励下降——并提出 AuxDPO 通过引入零空间辅助变量来修复此问题。
