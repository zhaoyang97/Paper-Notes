---
title: >-
  ICLR2026 对齐/RLHF论文汇总 · 102篇论文解读
description: >-
  102篇ICLR2026的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF、LLM、对抗鲁棒、强化学习、少样本学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICLR2026"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "LLM"
  - "对抗鲁棒"
  - "强化学习"
  - "少样本学习"
  - "推理"
item_list:
  - u: "a2d_any-order_any-step_safety_alignment_for_diffusion_language_models/"
    t: "A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models"
  - u: "activedpo_active_direct_preference_optimization_for_sample-efficient_alignment/"
    t: "ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment"
  - u: "align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf/"
    t: "Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment"
  - u: "aligner_diagnose_thyself_a_meta-learning_paradigm_for_fusing_intrinsic_feedback_/"
    t: "Aligner, Diagnose Thyself: A Meta-Learning Paradigm for Fusing Intrinsic Feedback in Preference Alignment"
  - u: "aligning_deep_implicit_preferences_by_learning_to_reason_defensively/"
    t: "Aligning Deep Implicit Preferences by Learning to Reason Defensively"
  - u: "alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme/"
    t: "Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment"
  - u: "alphaalign_incentivizing_safety_alignment_with_extremely_simplified_reinforcemen/"
    t: "AlphaAlign: Incentivizing Safety Alignment with Extremely Simplified Reinforcement Learning"
  - u: "alphasteer_learning_refusal_steering_with_principled_null-space_constraint/"
    t: "AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint"
  - u: "anchored_supervised_fine-tuning/"
    t: "Anchored Supervised Fine-Tuning"
  - u: "annotation-efficient_honesty_alignment_via_confidence_elicitation_and_calibratio/"
    t: "Annotation-Efficient Honesty Alignment via Confidence Elicitation and Calibration"
  - u: "beyond_binary_preferences_a_principled_framework_for_reward_modeling_with_ordina/"
    t: "Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback"
  - u: "beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling/"
    t: "Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling"
  - u: "beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew/"
    t: "Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework"
  - u: "bird_behavior_induction_via_representation-structure_distillation/"
    t: "BIRD: Behavior Induction via Representation-structure Distillation"
  - u: "bradley-terry_and_multi-objective_reward_modeling_are_complementary/"
    t: "Bradley–Terry and Multi-Objective Reward Modeling Are Complementary"
  - u: "cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation/"
    t: "CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation"
  - u: "capability-based_scaling_trends_for_llm-based_red-teaming/"
    t: "Capability-Based Scaling Trends for LLM-Based Red-Teaming"
  - u: "chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model/"
    t: "Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training"
  - u: "cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models/"
    t: "Cognitive models can reveal interpretable value trade-offs in language models"
  - u: "comal_a_convergent_meta-algorithm_for_aligning_llms_with_general_preferences/"
    t: "COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences"
  - u: "contextif_enhancing_instruction-following_through_context_reward/"
    t: "ContextIF: Enhancing Instruction-Following through Context Reward"
  - u: "cultivating_pluralism_in_algorithmic_monoculture_the_community_alignment_dataset/"
    t: "Cultivating Pluralism In Algorithmic Monoculture: The Community Alignment Dataset"
  - u: "data_selection_for_llm_alignment_using_fine-grained_preferences/"
    t: "Data Selection for LLM Alignment Using Fine-Grained Preferences"
  - u: "disentangling_length_bias_in_preference_learning_via_response-conditioned_modeli/"
    t: "Disentangling Length Bias in Preference Learning via Response-Conditioned Modeling"
  - u: "displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences/"
    t: "Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences"
  - u: "dont_throw_away_your_pretrained_model/"
    t: "Don't Throw Away Your Pretrained Model"
  - u: "eigenbench_a_comparative_behavioral_measure_of_value_alignment/"
    t: "EigenBench: A Comparative Behavioral Measure of Value Alignment"
  - u: "elephant_measuring_and_understanding_social_sycophancy_in_llms/"
    t: "ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs"
  - u: "eliminating_inductive_bias_in_reward_models_with_information-theoretic_guidance/"
    t: "Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance"
  - u: "enforcing_axioms_for_ai_alignment_under_loss-based_rules/"
    t: "Enforcing Axioms for AI Alignment under Loss-Based Rules"
item_total: 102
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🔬 ICLR2026** · **102** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_alignment/index.md) · [💬 ACL2026 (38)](../../ACL2026/llm_alignment/index.md) · [🧪 ICML2026 (37)](../../ICML2026/llm_alignment/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×47 · LLM ×22 · 对抗鲁棒 ×8 · 强化学习 ×4 · 少样本学习 ×2

**[A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)**

:   提出 A2D，一种针对扩散语言模型（dLLM）的 token 级安全对齐方法，通过训练模型在遇到有害内容的 mask 位置输出 [EOS] token 来实现任意解码顺序、任意解码步的安全防御，将 DIJA 模板攻击成功率从 80%+ 降到近零（1.3%/0.0%），并支持早期拒绝实现 19.3x 加速。

**[ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](activedpo_active_direct_preference_optimization_for_sample-efficient_alignment.md)**

:   ActiveDPO 用「被对齐的 LLM 自身」当奖励模型，基于其隐式奖励的梯度推导出一套有理论保证的不确定性准则，主动挑选最值得标注的偏好三元组，从而在固定标注预算下用更少的人工偏好标签把 LLM 对齐到更高水平。

**[Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)**

:   提出 Multi-Lingual Consistency (MLC) 辅助损失，通过 SVD 操控多语言表示矩阵的奇异值使其趋向秩-1（即多语言表示共线），仅需多语言 prompt 翻译（无需目标语言的 response），即可将一种语言的安全对齐效果一致性地迁移到所有语言。

**[Aligner, Diagnose Thyself: A Meta-Learning Paradigm for Fusing Intrinsic Feedback in Preference Alignment](aligner_diagnose_thyself_a_meta-learning_paradigm_for_fusing_intrinsic_feedback_.md)**

:   针对偏好数据集里"标错的偏好对"会毁掉 DPO 对齐的问题，本文不再依赖困惑度差这类单一启发式，而是让模型"自我诊断"——从一致性、学习难度、生成置信度三个内在信号拼出一个诊断向量，再用元学习训一个小网络学会融合这些信号给每个样本自适应加权，在多种噪声比例下显著超过现有鲁棒对齐方法。

**[Aligning Deep Implicit Preferences by Learning to Reason Defensively](aligning_deep_implicit_preferences_by_learning_to_reason_defensively.md)**

:   针对 LLM 个性化对齐里"只会照搬用户说出口的偏好、推不出深层意图、也不会主动规避风险"的问题，本文把对齐从标量奖励匹配重构成结构化推理过程——先用"多角色认知委员会"造出带逐步批判标注的推理链数据集 DeepPref，再训练一个会"先写批判再打分"的生成式过程奖励模型 Pers-GenPRM，最后用融合数值与自然语言反馈的 token 级在线 RL（CDPA）对齐策略模型，在深度偏好理解和防御性推理上都拿到 SOTA。

**[Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment](alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme.md)**

:   作者先用因果干预证明"当前的安全对齐是浅层的、和深度推理无关"，再用一份开源的 CoT 安全微调数据让模型学会"讲道理地拒绝"，最后提出 **Alignment-Weighted DPO**：把回答拆成"推理段"和"答案段"分别赋权，对越狱失败中更有害的那一段做更重的偏好更新，从而在保住效用的同时显著提升对各类越狱攻击的鲁棒性。

**[AlphaAlign: Incentivizing Safety Alignment with Extremely Simplified Reinforcement Learning](alphaalign_incentivizing_safety_alignment_with_extremely_simplified_reinforcemen.md)**

:   AlphaAlign 用一套极简的纯强化学习框架——只需"是否有害"的二元标签 + 不到 200 步 RL——把大模型预训练时就潜藏的"安全自我意识"激发出来，让它先写一段安全推理再作答，靠"可验证安全奖励 + 归一化帮助性奖励"双奖励同时打破"越安全越没用"的权衡。

**[AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)**

:   提出 AlphaSteer，通过学习一个受零空间约束的变换矩阵来动态构造 steering 向量，对良性输入产生近零向量（保持效用），对恶意输入重建拒绝方向向量（增强安全），在理论上保证了安全与效用的解耦。

**[Anchored Supervised Fine-Tuning](anchored_supervised_fine-tuning.md)**

:   本文用 reward-weighted regression (RWR) 框架严格解释了 DFT「更紧但会漂移」的本质，并提出在 DFT 重加权目标上叠加轻量级 KL 锚定项的 ASFT，以 SFT 级算力同时拿下推理与知识两类任务的稳定增益。

**[Annotation-Efficient Honesty Alignment via Confidence Elicitation and Calibration](annotation-efficient_honesty_alignment_via_confidence_elicitation_and_calibratio.md)**

:   这篇论文把"诚实对齐"（让 LLM 在回答前就准确说出自己有多大把握）拆成"引出-再-校准"两阶段：先用免标注的自一致性信号教模型把内在置信"说出来"，再用极少量（~1k 条，约 0.18% 全量）正确性标注把这个置信校准到真实准确率上，配套发布了 56 万训练样本的 HonestyBench，使得只用 1k 标注就能达到全量监督 98% 的对齐效果。

**[Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback](beyond_binary_preferences_a_principled_framework_for_reward_modeling_with_ordina.md)**

:   这篇论文指出现有奖励模型只会用"A 好于 B"的二元偏好，面对人类打的 Likert 分级反馈（"明显更好/更好/略好"）只能靠加 margin、乘权重这种拍脑袋的启发式补丁；作者把奖励建模重新表述成**离散序数回归**问题，从有序 logit 模型自然推出两个有理论根据的损失（NLL 与 all-threshold），让分隔各偏好等级的"阈值"直接从数据里学出来，在 RewardBench / RM-Bench 上一致追平或超过启发式基线，并把错误严重度降低 87%。

**[Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)**

:   提出 RCPO 框架，将 LLM 对齐从成对偏好扩展到排名选择（ranked choice）建模，通过 MLE 统一了效用模型（MNL）和排名模型（Mallows-RMJ），在 single-best 和 top-k 反馈格式下都优于 DPO 及其变体。

**[Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)**

:   提出基于社会选择理论公理的偏好学习框架，从成对比较数据中推断评估者人群分布的可行集，构造满足人群比例对齐(PPA)和人群有界可操纵性(PBM)公理的策略。

**[BIRD: Behavior Induction via Representation-structure Distillation](bird_behavior_induction_via_representation-structure_distillation.md)**

:   BIRD 通过让学生模型的内部表示**结构**（batch 内成对相似度的几何，用 CKA 度量）去匹配一个已对齐教师的表示结构，把鲁棒性 / 安全性这类"对齐行为"从一个异构教师迁移到学生——教师和学生可以任务、数据、架构、输出空间全都不同；在图像 OOD 鲁棒迁移上比微调 / 迁移学习 / 持续学习最多高 18% 鲁棒精度，并能从一个比学生小 25× 的教师做弱到强迁移。

**[Bradley–Terry and Multi-Objective Reward Modeling Are Complementary](bradley-terry_and_multi-objective_reward_modeling_are_complementary.md)**

:   本文提出 SMORM，在一个共享 embedding 上同时挂一个 Bradley–Terry 单目标奖励头和一个多目标回归头联合训练，理论上证明两者互补——回归头帮单目标头在 OOD 下抗 reward hacking，BT 头反过来把弱小的多目标头"托举"上去，最终一个 7B 模型超过 70B 基线。

**[CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)**

:   提出 CAGE 框架，通过 Semantic Mold（语义模具）将红队攻击 prompt 的对抗结构与文化内容解耦，能系统性地将英语红队基准适配到不同文化语境中，生成的文化扎根 prompt 比直接翻译的 ASR 显著更高。

**[Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)**

:   在 600+ 对攻击者-目标 LLM 组合上系统评估了 4 种越狱方法，发现攻击成功率（ASR）与攻击者-目标的能力差距遵循 sigmoid 缩放定律（R^2=0.83），能力差距可用 MMLU-Pro 的 logit 变换量化。

**[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)**

:   理论证明奖励过优化主要源于高奖励尾部区域的奖励模型错误规范，提出基于 rubric 的奖励建模方法：利用 off-policy 数据（强模型生成的优秀回复）构造评分细则，通过渐进式区分"优秀 vs 更优秀"来精细化 rubric，有效缓解奖励过优化。

**[Cognitive models can reveal interpretable value trade-offs in language models](cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)**

:   本文把认知科学里的「礼貌言语」理性言语行为（RSA）认知模型当作探针，给语言模型在一个真话-给面子两难任务上的回答拟合出三种效用（信息/社交/呈现）的权重，从而把模型的推理预算、系统提示、RLHF 训练动力学等"看不见的低层决策"翻译成一组可解释的价值权衡参数。

**[COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences](comal_a_convergent_meta-algorithm_for_aligning_llms_with_general_preferences.md)**

:   COMAL 把"对齐到一般人类偏好"建模成一个原始（无正则）的两人零和博弈，用源自博弈论的 Conceptual Prox 元算法——每轮解一个 KL 正则子博弈、然后把参考策略推进到当前解——首次证明算法能**末迭代收敛到原始博弈的精确纳什均衡**，从而保证对任意对手策略都有 ≥50% 胜率；它能以极小改动套在 DPO/IPO/INPO 等现有方法之上，实测在 Llama-3-8B-Instruct 上对所有对比算法保持 >60.2% 胜率。

**[ContextIF: Enhancing Instruction-Following through Context Reward](contextif_enhancing_instruction-following_through_context_reward.md)**

:   ContextIF 用强化学习训练一个"上下文生成器"，让它针对每条指令自动产出约束摘要 + 平行示范，再把这段上下文喂给参数冻结的目标模型做一次上下文学习；靠一套结构 + 语义复合的 Context Reward 引导生成，使开源 8B 模型在 IFEval 上从 77.11 涨到 83.35，且不损伤甚至提升通用能力。

**[Cultivating Pluralism In Algorithmic Monoculture: The Community Alignment Dataset](cultivating_pluralism_in_algorithmic_monoculture_the_community_alignment_dataset.md)**

:   作者用 5 国 15,000 人的代表性人类调查证明：21 个 SOTA 大模型的回答只对齐了 41% 的人类偏好（"算法单一文化"），现有偏好数据集因候选回答太同质而学不出这种多样性；为此提出"负相关采样（NC sampling）"——用一句 prompt 让单个模型一次生成四个**刻意发散**的回答，使对齐方法学习异质偏好的能力大幅提升，并据此开源了迄今最大、最具代表性的多语言多轮偏好数据集 Community Alignment（233,319 条比较）。

**[Data Selection for LLM Alignment Using Fine-Grained Preferences](data_selection_for_llm_alignment_using_fine-grained_preferences.md)**

:   针对"把多个细粒度（aspect-specific）偏好聚合后训练 DPO 会被偏好冲突拖垮"的问题，本文提出偏好散度（Preference Divergence, PD）来量化某条样本与其它偏好的冲突程度，并证明"只挑 PD 最负的那部分样本做标准 DPO"具有损失上下界最优性——结果在 UltraFeedback / HelpSteer 上只用 30% 数据就稳定超过全量对齐。

**[Disentangling Length Bias in Preference Learning via Response-Conditioned Modeling](disentangling_length_bias_in_preference_learning_via_response-conditioned_modeli.md)**

:   本文把奖励模型里隐性的「长度偏置」转化为显性的「长度指令理解」，提出响应条件化 Bradley-Terry（Rc-BT）模型——固定同一条回复、比较不同提示词，从而既消除长度作弊又让模型学会遵循长度指令，并可无缝接入奖励建模（Rc-RM）和 DPO（Rc-DPO）。

**[Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)**

:   发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。

**[Don't Throw Away Your Pretrained Model](dont_throw_away_your_pretrained_model.md)**

:   论文提出 SWITCH GENERATION：训练一个小型「切换器」LM，在生成一条回答的过程中按 token 片段动态挑选预训练 / 微调 / 对齐三个检查点轮流「发言」，让对齐丢失的基座能力（创造力、校准、多元性）和对齐获得的能力（推理、指令遵循）互补，在 18 个数据集上比单模型平均提升 31%、比 8 类协作基线再提升 12.9%。

**[EigenBench: A Comparative Behavioral Measure of Value Alignment](eigenbench_a_comparative_behavioral_measure_of_value_alignment.md)**

:   EigenBench 提出一种黑盒、无需真值标签的价值对齐度量方法：让一群语言模型互相评判彼此在给定"宪法"（价值准则）下的回答，用 EigenTrust 把这些两两评判聚合成一个共识打分向量，使得"越对齐的模型其评判权重越高"，最终输出每个模型对该价值体系的对齐 Elo 排名。

**[ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs](elephant_measuring_and_understanding_social_sycophancy_in_llms.md)**

:   本文将 LLM 谄媚行为从"同意错误事实"扩展到"过度维护用户面子"，提出 social sycophancy 理论框架，构建 ELEPHANT 基准评测 11 个主流 LLM，发现它们在日常建议查询中平均比人类多谄媚 47 个百分点，且谄媚倾向在偏好数据集中受到奖励，同时提供提示重写和 DPO 等缓解策略。

**[Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance](eliminating_inductive_bias_in_reward_models_with_information-theoretic_guidance.md)**

:   DIR 把奖励模型去偏建模成一个信息论优化问题——最大化「奖励预测↔人类偏好」的互信息、同时最小化「奖励隐表示↔偏置属性」的互信息，用 BA 下界和 CLUB 上界两个变分估计落地，统一处理长度、谄媚、格式等非线性归纳偏置。

**[Enforcing Axioms for AI Alignment under Loss-Based Rules](enforcing_axioms_for_ai_alignment_under_loss-based_rules.md)**

:   在线性社会选择框架下，基于损失的奖励模型（包括多项式奖励）无法保证 Pareto 最优性（PO），但当训练数据均匀覆盖嵌入空间时可在极限中恢复 PO——为宪法风格对齐提供了可证明的数据设计方案。

**[Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection](enhancing_trustworthiness_of_fine-tuned_llms_via_regularized_subset_selection.md)**

:   针对 SFT 导致 LLM 可信度下降的问题，提出两阶段修复框架：先用 DPP 正则化子集选择定位"有害训练样本"，再用 PBRF 梯度上升修复模型，以 ≤1% 困惑度代价换取最高 21% 的可信度提升。

**[Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment](evaluating_and_improving_cultural_awareness_of_reward_models_for_llm_alignment.md)**

:   本文提出 CARB 文化感知奖励模型基准，系统评估奖励模型在 10 种文化和 4 类文化领域中的偏好判断能力，并进一步用 Think-as-Locals 让生成式奖励模型先生成本地文化评价准则、再通过 RLVR/GRPO 优化判断，从而减少表面语言线索带来的伪相关。

**[Fluent Alignment with Disfluent Judges: Post-training for Lower-Resource Languages](fluent_alignment_with_disfluent_judges_post-training_for_lower-resource_language.md)**

:   本文提出一套面向低资源语言的后训练方法：完全不使用目标语言的指令数据，只靠 on-policy 强化学习从模型自身采样的回复中学习，从而即便用一个本身"说话不流畅"的裁判模型也能训出语言地道的对齐模型——核心是"训练阶段绝不让模型见到任何翻译腔文本"。

**[FSPO: Few-Shot Optimization of Synthetic Preferences Effectively Personalizes to Real Users](fspo_few-shot_optimization_of_synthetic_preferences_effectively_personalizes_to_.md)**

:   把奖励建模重构成"以用户为任务"的黑盒元学习问题，用少样本上下文偏好让 LLM 快速推断每个用户的个性化奖励函数，并配合百万级合成偏好数据集（强调多样性 + 结构性）实现从合成用户到真实用户的迁移，在开放式问答上对真人取得 70% 胜率。

**[General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)**

:   理论证明现有 RLHF 探索奖励（exploratory bonus）在 KL 和 α-散度正则化下实际上会引导策略向参考模型的高概率区域靠拢（与乐观原则相悖），提出 General Exploratory Bonus (GEB) 框架——通过参考模型依赖的奖励调节来抵消散度正则化的保守偏差，可证明满足乐观原则。

**[Group-Normalized Implicit Value Optimization for Language Models](group-normalized_implicit_value_optimization_for_language_models.md)**

:   GN-IVO 把 LLM 生成看作逐步决策过程，用同一 prompt 下的一组候选回答构造归一化的奖励分布，再用策略相对旧策略的前缀概率比去匹配这个分布，从而在不训练显式 critic / value network 的情况下给 token 或 reasoning step 提供更细粒度的价值信号。

**[Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)**

:   通过构造 KL 正则化代理目标并推导 pairwise consistency condition，从第一原理证明 group-relative REINFORCE（GRPO）天然是 off-policy 算法；进而通过组件隔离实验发现 clipping 才是训练稳定性的关键而 importance sampling 完全可以去掉，并在此统一框架下重新解释了 Kimi OPMD、Meta AsymRE 等多个看似独立的算法。

**[GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)**

:   提出 GuardAlign，一个无需训练的多模态大模型推理时安全防御框架：用最优传输(OT)精确检测图像中的不安全区域并遮蔽，再通过跨模态注意力校准保持安全前缀的影响力不衰减，在6个LVLM上将不安全响应率降低最多39%，同时保持甚至提升通用能力。

**[GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods](guidedbench_measuring_and_mitigating_the_evaluation_discrepancies_of_in-the-wild.md)**

:   通过系统测量 37 篇越狱研究，本文揭示现有越狱评测因"缺乏逐案标准"而严重失真，并提出 GuidedBench——一个带逐题打分指南（scoring guidelines）的评测系统，把主观的"是否成功越狱"判断转化为客观的"指南要点是否命中"检查，使评测者间方差至少降低 76.03%。

**[Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)**

:   揭示了 stepwise group-based RL（如 GRPO/GiGPO）中的「历史上下文不一致」问题——同一 group 内的 step 可能具有不同历史上下文导致 advantage 估计偏差，提出 HGPO 通过层次化分组和自适应加权实现低偏差、平衡方差的 advantage 估计，在 ALFWorld 和 WebShop 上以极低额外开销（<0.001%）取得显著提升。

**[Holdout-Loss-Based Data Selection for LLM Finetuning via In-Context Learning](holdout-loss-based_data_selection_for_llm_finetuning_via_in-context_learning.md)**

:   用上下文学习（把 holdout 集当作 in-context 示例）来近似"某条样本训练后会带来的 holdout 损失"，从而无需参考模型、无需重训就能给每条微调样本打分并动态重加权梯度，让 SFT/DPO/SimPO 的对齐效果稳定提升、额外开销仅约 1.5%。

**[Humanline: Online Alignment as Perceptual Loss](humanline_online_alignment_as_perceptual_loss.md)**

:   本文用行为经济学的**前景理论**解释"在线对齐为何强于离线对齐"——在线 on-policy 采样更接近人类对模型输出的主观感知分布，而 PPO/GRPO 的剪裁恰好隐式恢复了这种感知偏差，因此它们本质上已是"感知损失"；据此提出一个把感知失真显式注入 DPO/KTO/GRPO 的设计范式（humanline 变体），让离线 off-policy 数据也能匹配在线性能，同时训练快达 6×。

**[IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment](ideal_data_equilibrium_adaptation_for_multi-capability_language_model_alignment.md)**

:   IDEAL 把"SFT 各领域数据该配多少量"建模成一个双层优化问题，用二阶（Hessian）梯度信息算出每个领域数据应该上采样还是下采样，迭代两轮就能让数学/代码/推理/指令跟随四项能力整体均衡提升约 7%。

**[Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment](inverse_reinforcement_learning_with_dynamic_reward_scaling_for_llm_alignment.md)**

:   DR-IRL 用逆强化学习从「均衡的安全演示数据」训练分类别的影子奖励模型，再给 GRPO 的优势函数乘上一个由「数据难度 × 模型响应度」共同决定的动态系数，把优化火力集中到长尾、高难度的有害样本上，从而在不牺牲（甚至提升）通用能力的前提下显著增强安全对齐。

**[Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)**

:   挑战"on-policy数据总是更好"的共识：发现对齐过程分为偏好注入（需高多样性off-policy数据）和偏好微调（需高质量on-policy数据）两个阶段，不同模型/阶段对数据类型的最优选择不同。提出仅3.2%计算开销的边界判定算法，在5个模型×55个配置上验证有效。

**[JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)**

:   提出首个评估 LLM 在越狱攻击下生成假新闻鲁棒性的多语言多区域基准 JailNewsBench，覆盖 34 个地区和 22 种语言、约 30 万实例，揭示最高 86.3% 的攻击成功率以及英语/美国话题防御显著弱于其他地区的安全不平衡现象。

**[JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)**

:   揭示对齐 LLM 的 top-k token log probability 中仍包含有害信息的知识泄露问题，提出 JULI——仅用不到目标模型 1% 参数量的 BiasNet 插件操纵 logit bias，在仅访问 top-5 token 概率的 API 场景下成功越狱 Gemini-2.5-Pro（Harmful Info Score 4.19/5），比 LINT 快 140 倍同时 harmfulness 提升约 2 倍。

**[Keep the Best, Forget the Rest: Reliable Alignment with Order-Aware Preference Optimization](keep_the_best_forget_the_rest_reliable_alignment_with_order-aware_preference_opt.md)**

:   RAPPO 在每个 batch 内用参考策略给样本打"可信度"分，把那些参考模型本身就站错队、又最难学的高损失偏好对临时剔除掉，只用几行代码改造 DPO 就能在情感、去毒、摘要、安全对齐上稳定超过 SimPO/DPO 等基线，并配套证明了更紧的泛化界。

**[Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization](learning_from_noisy_preferences_a_semi-supervised_learning_approach_to_direct_pr.md)**

:   针对"人类视觉偏好是多维的、却被压成一个二元标签"导致的 Diffusion-DPO 梯度冲突问题，本文提出 Semi-DPO：把多个奖励模型一致认可的样本当作干净标注、把维度冲突的样本当作含噪未标注数据，用扩散模型自身作为隐式分类器在不同 timestep 上生成伪标签做迭代自训练，在不引入额外人工标注和显式奖励模型的前提下取得 SOTA 对齐效果。

**[Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)**

:   提出**D3S**（Dynamic Dual-Level Down-Sampling）框架，在sample层最大化advantage方差、在token层优先选取高熵+高advantage的token，配合动态调度策略，用不到20% token实现更快收敛和更优性能。

**[Learning Ordinal Probabilistic Reward from Preferences (OPRM)](learning_ordinal_probabilistic_reward_from_preferences.md)**

:   提出序数概率奖励模型(OPRM)，将响应质量离散化为1-9序数等级并学习完整概率分布，结合区域洪泛调优(RgFT)实现数据高效训练。在RewardBench达89.3%，比现有RM提升2.9%-7.4%，同时提供不确定性估计和标注分歧检测。

**[Learning to Summarize User Information for Personalized RLHF（PLUS）](learning_to_summarize_user_information_for_personalized_reinforcement_learning_f.md)**

:   PLUS 用 RL（PPO）训练一个"用户摘要器"，把每个用户的偏好、特征、历史对话压缩成一段**自然语言摘要** $z$，再用这段摘要去 condition 奖励模型，并让摘要器和奖励模型在线协同自适应，从而在不假设"所有人偏好相同"的前提下，把奖励模型准确率相对 Bradley-Terry 提升 11–77%。

**[Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)**

:   提出 NSPO，将安全对齐的策略梯度投影到通用任务表征的零空间中，从几何层面保证安全优化不损害通用能力，仅用 40% 安全数据即在 7 个安全 benchmark 上达到 SOTA，同时在数学/代码/指令遵循上几乎无性能损失。

**[Multi-objective Large Language Model Alignment with Hierarchical Experts](multi-objective_large_language_model_alignment_with_hierarchical_experts.md)**

:   HoE 把多目标对齐拆解成一系列"单偏好子问题"，用免训练抽取的 LoRA 专家 + 轻量路由专家 + 无参偏好路由构成三层 Mixture-of-Experts，无需重训主干即可即插即用地覆盖整条 Pareto 前沿、响应任意用户偏好权重。

**[Multiplayer Nash Preference Optimization](multiplayer_nash_preference_optimization.md)**

:   把 Nash learning from human feedback（NLHF）从「两玩家博弈」推广到「n 玩家博弈」，让一个策略同时对抗一整个对手群体（历史 checkpoint 或多个异质奖励模型），用乘性权重更新求近似 Nash 均衡，从而更稳、更全面地刻画真实世界中非传递、异质的人类偏好。

**[No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)**

:   发现 GRPO 训练中大量"零方差提示"（所有回答全对或全错）被白白丢弃，提出 RL-ZVP 算法通过熵引导的优势整形从中提取学习信号，在六个数学推理基准上相比 GRPO 提升最高 8.61 个精度点和 7.77 个通过率点。

**[Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](obscure_but_effective_classical_chinese_jailbreak_prompt_optimization_via_bio-in.md)**

:   提出 CC-BOS 框架，利用文言文的语义压缩和模糊性特征，结合果蝇优化算法在八维策略空间中搜索最优越狱提示，在六个主流 LLM 上实现近 100% 的攻击成功率。

**[Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences](omni-reward_towards_generalist_omni-modal_reward_modeling_with_free-form_prefere.md)**

:   本文提出 Omni-Reward，用一个统一的 benchmark（Omni-RewardBench，5 模态 9 任务）、一个大规模偏好数据集（Omni-RewardData，248K 通用 + 69K 指令微调对）和两个奖励模型（判别式 BT 版 + 生成式 R1 版），把奖励建模从「只懂文本+图像、只认固定二元偏好」扩展到「覆盖文/图/视频/音频/3D、能按用户自由文本偏好动态打分」，在自家 benchmark 上比基座模型涨 20%，并在 VL-RewardBench 等公开榜上达到/超过 SOTA。

**[On the Shelf Life of Fine-Tuned LLM-Judges: Future-Proofing, Backward-Compatibility, and Question Generalization](on_the_shelf_life_of_fine-tuned_llm-judges_future-proofing_backward-compatibilit.md)**

:   这篇论文把"被微调好的 LLM 裁判能用多久"形式化成一个**双分布（问题分布 × 回复分布）偏移问题**，通过两个推理数据集、三种训练配方、三个 backbone 的系统实验发现：裁判很难"未来兼容"（在更强新模型的回复上掉点严重）、却比较容易"向后兼容"（在更弱旧回复上几乎不掉点），而持续学习能在新旧之间取得更平衡的适配，且所有裁判对训练时没见过的新问题都泛化不佳。

**[Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks](optimal_sparsity_of_mixture-of-experts_language_models_for_reasoning_tasks.md)**

:   系统研究 MoE 语言模型的稀疏度如何不同地影响记忆性任务和推理性任务：记忆任务偏好更高稀疏度（更多参数），而推理任务在 TPP≈20 附近达到最优，且该趋势在 GRPO 后训练和测试时计算增加后仍然不变。

**[OrthAlign: Orthogonal Subspace Decomposition for Non-Interfering Multi-Objective Alignment](orthalign_orthogonal_subspace_decomposition_for_non-interfering_multi-objective_.md)**

:   针对"提升一个偏好就损害另一个偏好"的多目标对齐困境，OrthAlign 把不同偏好的参数更新约束到彼此正交的子空间里，让各偏好的优化方向在数学上互不干扰，从而在不牺牲单项性能的前提下同时对齐 helpful / harmless / truthful，单项最高提升 50.89%、整体奖励平均提升 13.96%。

**[PALC: Preference Alignment via Logit Calibration](palc_preference_alignment_via_logit_calibration.md)**

:   PALC 给冻结的大模型外挂一个极小的"校准模块"，把对齐干预从纠缠的隐空间挪到天然解耦的词表 logit 空间——只把隐状态当只读上下文、生成位置相关的 logit 偏移量加到原始 logit 上，仅增加 0.002%–0.13% 参数、几乎不掉推理速度，就能在测试时实现可调强度的偏好对齐。

**[Pretrain Value, Not Reward: Decoupled Value Policy Optimization](pretrain_value_not_reward_decoupled_value_policy_optimization.md)**

:   作者指出在固定偏好数据下「先训奖励模型再在线学 critic」与「直接预训练一个价值模型」在信息上等价，于是提出 DVPO：离线预训练一个全局价值模型（GVM）并冻结它作为通用 critic 来指导策略优化，省掉了在线 critic 训练，在 MT-Bench / Alpaca-Eval / Arena-Hard 上达到或超过主流 RLHF 方法，同时省 30–40% 显存、30–45% 训练时间。

**[Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead](quagmires_in_sft-rl_post-training_when_high_sft_scores_mislead_and_what_to_use_i.md)**

:   这篇论文用 100 多个模型、超过 100 万 GPU 小时的实验证明：推理大模型后训练里「SFT 分数越高、RL 后效果就越好」是个广泛存在的伪命题，并提出用**验证集泛化损失**和 **Pass@大 k** 两个指标来可靠预测 RL 最终成绩，把预测精度（$R^2$、Spearman 秩相关）相对直接用 SFT 分数最高提升达 0.5（约 2 倍）。

**[RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment](re-po_robust_enhanced_policy_optimization_as_a_general_framework_for_llm_alignme.md)**

:   RE-PO 把每条偏好标签的"正确与否"当成隐变量，用 EM 算法在训练中边推断每条数据的置信度边更新策略，从而对含噪偏好数据做自适应降权；它还把 DPO/IPO/SimPO/CPO 等一大类偏好损失统一接入同一概率框架，使它们都能被"鲁棒化"，在 AlpacaEval 2 上最多提升 7.0 个百分点。

**[Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)**

:   提出"先回答后检查"(Answer-Then-Check)策略：模型先在思维链中生成意图答案摘要，再依据安全策略进行安全分析，最后决定输出或拒绝。构建80K ReSA数据集训练后，在7种越狱攻击上防御率达到99.3%(RL版本)，仅500样本即可达全数据集效果。

**[RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data](recast_expanding_the_boundaries_of_llms_complex_instruction_following_with_multi.md)**

:   RECAST 从真实的「指令–回复」对里反向挖掘出大量可验证的约束，把它们重新拼装成单条指令含十几个约束的高复杂度训练数据（RECAST-30K，30K 条 / 19 类约束），再配上规则与模型双轨验证器；用它做 SFT 就能让小模型的复杂指令跟随能力超过大几个量级的模型，进一步用「约束满足率」当奖励做 RL（RLVC）还能再涨一截，同时不损伤通用能力。

**[Reward Model Routing in Alignment](reward_model_routing_in_alignment.md)**

:   本文提出 **BayesianRouter**，一个在 RLHF/在线 DPO 训练中为每条偏好对**逐条挑选最合适奖励模型**的混合路由框架：离线阶段用偏好数据训练一个多任务路由器学习各奖励模型的擅长领域，在线阶段用贝叶斯 Thompson 采样按 query 选模型，并把离线学到的强度当作高斯先验注入，在线性更新中边对齐边适应策略分布，最终在指令跟随与推理基准上稳定超过单一 RM、RM 集成和已有路由方法 LASER。

**[Reward Models Inherit Value Biases from Pretraining](reward_models_inherit_value_biases_from_pretraining.md)**

:   这篇论文用"穷举 token 搜索 + 心理语言学语料"的可解释性方法系统检查了 10 个主流开源奖励模型（RM），发现 RM 在"能动性 vs 共融性"等多个人类价值维度上的偏好高度取决于它的基座 LLM（Llama 系偏好 agency、Gemma 系偏好 communion），并把这种偏见一路溯源到基座模型的对数概率、证明它在偏好微调过程中很难被"洗掉"。

**[RewardBench 2: Advancing Reward Model Evaluation](rewardbench_2_advancing_reward_model_evaluation.md)**

:   本文提出 RewardBench 2——一个用全新未见人类 prompt、把"1 选 1"改成"4 选 1（1 正 3 负）"、覆盖 6 大领域（含新增的 Ties / Precise IF / Factuality）的奖励模型评测基准；它比初代 RewardBench 平均难 20 分，且与 best-of-N 采样、PPO 训练等下游用法的相关性显著更强。

**[RLBFF: Binary Flexible Feedback to Bridge Between Human Feedback & Verifiable Rewards](rlbff_binary_flexible_feedback_to_bridge_between_human_feedback_verifiable_rewar.md)**

:   本文提出 RLBFF（Reinforcement Learning with Binary Flexible Feedback），从自然语言反馈里抽取「可二元回答的原则」（如「信息准确性：是」「代码可读性：否」），把奖励模型训练改造成「回答是否满足某条原则」的蕴含判断，从而兼得 RLHF 的广覆盖和 RLVR 的可解释/抗 reward hacking；训练出的标量奖励模型在 RM-Bench（83.6）、JudgeBench（76.3）上超过同数据的 Bradley-Terry 模型，GenRM 进一步把 RM-Bench/JudgeBench 推到 86.2/81.4（榜首），并用它把 Qwen3-32B 对齐到媲美 o3-mini/DeepSeek R1 的水平、推理成本不到对手 5%。

**[Robust Preference Alignment via Directional Neighborhood Consensus](robust_preference_alignment_via_directional_neighborhood_consensus.md)**

:   提出Robust Preference Selection (RPS)，一种无需重训练的推理时偏好对齐增强方法，通过从目标偏好的局部邻域采样多个候选方向并生成响应、再根据原始偏好选择最优响应，在OOD偏好上相比基线达到最高69%的胜率。

**[Robust Reward Modeling via Causal Rubrics](robust_reward_modeling_via_causal_rubrics.md)**

:   针对奖励模型容易抓住长度、格式等虚假特征作弊的问题，CROME 让 Oracle LLM 先为每个问题列出真正决定质量的「因果 rubric」，再围绕这些 rubric 合成两类反事实数据——沿单个因果属性升/降级的「因果增强」和把答案对配到无关问题上的「中立增强」，配合复合损失训练，使奖励模型对因果属性敏感、对未知虚假属性不变，在 RewardBench 上平均提升 5.3%（安全 +12.4%、推理 +7.1%）。

**[ROSETTA: Constructing Code-Based Reward from Unconstrained Language Preference](rosetta_constructing_code-based_reward_from_unconstrained_language_preference.md)**

:   ROSETTA 把人类在机器人交互中随口给出的、会随时间变化的自然语言偏好，分解为“偏好落地、奖励分阶段、代码生成与验证”三步，在线生成可训练的代码奖励函数，并在 116 条偏好上达到 87% 成功率和 86% 人类满意度。

**[SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](safedpo_preference_optimization_safety.md)**

:   重新审视安全约束 RLHF 目标并证明其存在闭式最优策略，据此推导出等价的可处理目标 SafeDPO，仅需在标准 DPO 上加入安全感知数据变换和安全 margin 项（1 个额外超参数），无需奖励/代价模型，在 PKU-SafeRLHF-30K 上实现 96.87% 无害率且保持竞争力的有用性，训练速度比 SafeRLHF 快 25×。

**[Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)**

:   本文通过四个系统性实验（平行投影、正交投影、子空间重叠、激活空间分析）在5个开源 LLM 上全面验证了一个关键发现：安全对齐行为在权重空间和激活空间中都与通用学习高度纠缠、不存在线性可分的独立子空间，因此基于子空间投影/过滤的防御策略面临根本性局限。

**[SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)**

:   提出 SEMA 框架，通过预填充自调优和带意图漂移感知奖励的 RL 两阶段训练，在无需任何现有攻击策略或外部数据的条件下，训练出能自动生成多轮越狱攻击的 attacker，在 AdvBench 上跨三个受害模型平均 ASR@1 达 80.1%，超越 SOTA 33.9%。

**[Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)**

:   指出 RLHF 中标准 KL 散度正则化仅比较相同索引处的 token 概率而忽略语义相似性，提出基于熵正则化 Wasserstein 距离的语义感知策略正则化（WPR），通过对偶公式将正则化转化为 token 级惩罚项，在对话生成和摘要任务上一致优于 KL 及各类 f-散度基线。

**[Semi-Supervised Preference Optimization with Limited Feedback](semi-supervised_preference_optimization_with_limited_feedback.md)**

:   SSPO 把偏好优化重述成一个概率分类问题，从少量成对偏好标签中学到一个能可靠分开"赢/输"回答的奖励阈值，再用这个阈值给海量无配对样本（如 SFT 数据）打伪标签，配合课程式调度联合训练——只用 1% 的 UltraFeedback 就能稳定超过用 10% 数据训练的强基线。

**[Sharpness-Aware Minimization in Logit Space Efficiently Enhances Direct Preference Optimization](sharpness-aware_minimization_in_logit_space_efficiently_enhances_direct_preferen.md)**

:   本文从 logit 空间动力学出发解释了 DPO 训练中"偏好回答概率反而下降"的 squeezing effect（负梯度让残差沿高曲率方向疯狂膨胀），证明 SAM 的曲率正则恰好能压住这种膨胀，并落地为只扰动输出层、几乎零开销的 logits-SAM，在 Pythia-2.8B / Mistral-7B / Gemma-2B-IT 上稳定提升 DPO 及其变体。

**[Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)**

:   提出 Human-AI 协同的两阶段偏好数据策展流水线：阶段一通过人工验证、错误驱动自适应检索和偏好引导 LLM 标注迭代 8 轮积累约 1M 偏好对；阶段二借助双 RM 一致性过滤将数据规模扩展到 26M 对。最终训练的 Skywork-Reward-V2 8B 模型在 RewardBench 达 97.8%，7 个主流基准平均 88.6%，全面超越所有开源 70B 奖励模型。

**[Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)**

:   提出Spectrum Tuning后训练方法，通过在90+任务的分布拟合数据集上训练，改善语言模型的上下文可操控性、输出空间覆盖度和分布对齐能力，揭示当前指令调优会损害模型的上下文可操控性。

**[Stackelberg Learning from Human Feedback: Preference Optimization as a Sequential Game](stackelberg_learning_from_human_feedback_preference_optimization_as_a_sequential.md)**

:   本文把 LLM 偏好对齐重新建模成一个"领导者—跟随者"的**序贯博弈**（SLHF）：领导者先承诺一个回答、跟随者在看到这个回答后再给出改进版，由此天然得到一个确定性、对非传递偏好鲁棒的均衡，并支持**推理时无需训练的迭代自我精炼**，在 0.5B–8B 模型上一致超过 RLHF（RLOO）与 NLHF（Nash-MD-PG）基线。

**[StoryAlign：为故事生成评测并训练奖励模型](storyalign_evaluating_and_training_reward_models_for_story_generation.md)**

:   本文指出现有奖励模型几乎无法识别人类偏好的故事（最强的 GPT-4o 也只有 66.3% 准确率），于是构建了首个故事偏好评测基准 STORYRMB（1133 条人工校验实例）并用约 10 万条自动构造的偏好对训练出专用奖励模型 STORYREWARD，仅 8B 规模就在基准上达到 75.0% 的 SoTA，并在 Best-of-N 测试时扩展中显著优于其他奖励模型。

**[Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)**

:   提出"浅层安全对齐假说"(SSAH)：安全对齐本质上是教模型做一个隐式的二分类任务（执行还是拒绝），只需约1.3%的神经元即可建立安全护栏；冻结这些安全关键单元可在微调时保持安全性，利用冗余单元作为"对齐预算"可消除对齐税。

**[Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)**

:   提出Sysformer，一个可插拔到任意冻结LLM前端的轻量Transformer模块，根据用户输入自适应地在嵌入空间中变换系统提示，使模型拒绝有害请求同时正常回应安全请求，无需修改LLM参数或过滤用户输入。

**[Test-Time Alignment for Large Language Models via Textual Model Predictive Control](test-time_alignment_for_large_language_models_via_textual_model_predictive_contr.md)**

:   把 LLM 的测试时偏好对齐重述成一个轨迹优化问题，借控制论里的模型预测控制（MPC）做"边走边规划"，靠**事后子目标识别**从已生成的 rollout 里挑出高奖励片段当 waypoint、再**条件化重新生成**滚动逼近最优，在机器翻译、长文回复、代码生成三类任务上都不动模型参数就稳定涨点。

**[Text2Grad: Reinforcement Learning from Natural Language Feedback](text2grad_reinforcement_learning_from_natural_language_feedback.md)**

:   把自由形式的文字批评对齐到输出的 token 片段、转成 token 级伪奖励，再据此构造"自然语言梯度"驱动 PPO 更新，让模型只修改"被批评的那几个 token"而非全局乱推，在摘要、代码生成、问答三类任务上同时超过标量奖励 RL 和纯提示反思基线。

**[The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives](the_alignment_auditor_a_bayesian_framework_for_verifying_and_refining_llm_object.md)**

:   把"逆强化学习恢复 LLM 隐式奖励"从一次性点估计，重构成一套贝叶斯审计流程——先用变分推断恢复奖励的**后验分布**而非单点，再用序贯贝叶斯更新让后验逐轮收缩、用认知不确定性诊断暴露捷径与分布外输入，最后把收缩后的低不确定性奖励直接喂回 RLHF，证明它能复现 oracle 奖励的对齐效果（毒性下降曲线几乎重合）。

**[Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)**

:   提出TI-DPO，通过梯度归因+高斯先验的混合权重机制精确量化每个token对偏好的贡献，结合三元组损失在连续语义空间引导优化，在6个基准上平均62.3分达到SOTA，同时具备可解释的token级控制能力。

**[Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)**

:   提出 UltraBreak，通过语义对抗目标（用cosine相似度替代交叉熵优化出平滑loss景观）+ 输入空间约束（随机变换+TV正则化产生变换不变特征），训练单张通用对抗图像即可跨6+个VLM架构和商业模型实现越狱，黑盒平均ASR达71%（SafeBench），远超此前方法。

**[Towards Self-Robust LLMs: Intrinsic Prompt Noise Resistance via CoIPO](towards_self-robust_llms_intrinsic_prompt_noise_resistance_via_coipo.md)**

:   本文提出 CoIPO（对比学习 + 逆向 DPO），让 LLM 在脏提示（拼写错误、词替换、句式扰动）下输出与干净提示尽量一致，不依赖任何外部预处理工具就把内禀鲁棒性练进模型，在自建的 NoisyPromptBench 上比当前 SOTA（CoIN）平均准确率高 3.64%。

**[Towards Understanding Valuable Preference Data for Large Language Model Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)**

:   从模型依赖视角研究偏好数据质量：提出截断影响函数(TIF)发现中等IF值的数据才是最有价值的(而非经典观点中的高IF) -> 设计LossDiff和IRM两个轻量代理指标近似TIF -> 两者组合的LossDiff-IRM选择器仅用50-64%数据即可平均提升WinRate 13.58%，在多个LLM家族和对齐benchmark上均有效。

**[Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations](truthful_or_fabricated_using_causal_attribution_to_mitigate_reward_hacking_in_ex.md)**

:   论文指出偏好优化（DPO/RLHF）会让 LLM 学会"嘴上不承认、暗地里偷用"被禁止的输入线索，从而生成不忠实的思维链解释；作者用反事实因果归因检测这种线索依赖，并把信号以"免责声明"形式注入奖励模型输入，在两个受控设定下显著降低了 CoT 欺骗的发生率。

**[TS²：训练用 Sparsemax+、测试用 Softmax，让 LLM 微调既准又多样](ts2_training_with_sparsemax_testing_with_softmax_for_accurate_and_diverse_llm_fi.md)**

:   针对交叉熵（CE）监督微调把概率挤成 one-hot、压垮输出多样性的问题，本文提出 TS²：训练时用带尾部抑制项的 Sparsemax+ 损失（稀疏支撑 + 显式压尾），推理时换回 softmax 解码，从而在不改模型结构的前提下同时提升 Llama-3.1-8B / Qwen-2.5-7B 在聊天、代码、开放生成上的准确率与多样性。

**[Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)**

:   提出DAR(Dual-regularized Advantage Regression)：发现标准RLHF中参考模型正则化(防reward hacking)和策略稳定约束(防崩溃)会逐步冲突导致优化空间过度受限，通过双KL目标在对数空间插值参考策略+回归变换消除策略比率不稳定性，在直接AI对齐和标准RLHF设置中达到92.42%平均胜率，超GRPO 7.27%。

**[Verification and Co-Alignment via Heterogeneous Consistency for Preference-Aligned LLM Annotations](verification_and_co-alignment_via_heterogeneous_consistency_for_preference-align.md)**

:   本文提出 Heterogeneous-Consistency Co-Alignment (HCC)，用 LLM 与任务专用 embedding 模型之间的一致/不一致关系，在无 ground truth 的半监督 NLU 标注场景中验证 LLM 标注可靠性，并通过两轮基于近邻投票的协同校准修正偏好不一致样本。

**[Weak-to-Strong Generalization with Failure Trajectories](weak-to-strong_generalization_with_failure_trajectories.md)**

:   本文把"弱到强泛化"（W2SG）从二分类扩展到多步交互式决策任务：用一个弱模型探索出大量包含成功与失败的动作轨迹，按公共前缀合并成"轨迹树"，再用结构化对比对的 TreeDPO 或离线 MCTS 路径搜索去微调强模型，结果强模型在三个 Agent 环境上不仅超过 SFT 弱模型，甚至反超用专家数据训练的 SFT 强模型。

**[What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data](whats_in_my_human_feedback_learning_interpretable_descriptions_of_preference_dat.md)**

:   WIMHF 用稀疏自编码器（SAE）在「两个候选回复的嵌入差」上学出一小批人类可读的特征，再用逻辑回归量化每个特征对偏好标签的影响，从而自动地、无需预设假设地说清楚一份偏好数据集「能测什么偏好」和「标注者实际偏好什么」，并把这些特征当作数据净化和个性化的可控杠杆。

**[When Data Is the Algorithm: A Systematic Study and Curation of Preference Optimization Datasets](when_data_is_the_algorithm_a_systematic_study_and_curation_of_preference_optimiz.md)**

:   本文对 5 个常用开源 DPO 偏好数据集做了首个系统的「样本级」横向体检——用 Magpie 标注任务类别/难度/输入质量、再用独立奖励模型给每对偏好打「偏好奖励」分，发现 20–30% 的样本"被选中的回答其实不如被丢弃的"；基于这些诊断信号，作者筛出一个比最强单一数据集小 30% 却更强的混合集 UltraMix。

**[When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger](when_weak_llms_speak_with_confidence_preference_alignment_gets_stronger.md)**

:   用一个不到 0.5B 的弱 LLM 当偏好标注器，再按它对每个样本的"置信度"给偏好优化目标逐样本加权（CW-PO），结果只用 20%~30% 的人工标注就能在多个数据集上反超用 100% 人工标注训练的 DPO，且兼容 DPO/IPO/rDPO 各种目标。

**[Why DPO is a Misspecified Estimator and How to Fix It](why_dpo_is_misspecified_estimator.md)**

:   从信息几何角度证明 DPO 在参数化（非 tabular）策略类下本质上是一个误指定的统计估计问题——DPO 将真实奖励函数 KL 投影到隐式奖励流形上，当奖励不可实现时会导致偏好反转和奖励下降——并提出 AuxDPO 通过引入零空间辅助变量来修复此问题。
