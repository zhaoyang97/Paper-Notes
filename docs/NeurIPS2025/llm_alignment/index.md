<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🧠 NeurIPS2025** · 共 **27** 篇

**[A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)**

:   提出一种自适应 Alpha 聚合策略，在联邦 RLHF 框架中根据各用户群体的历史对齐表现动态调整奖励权重，从而在多元偏好对齐中同时实现高公平性和强对齐性能。

**[Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)**

:   将LLM对齐形式化为约束优化问题（最大化主要奖励同时满足次要效用约束如安全性），提出基于拉格朗日对偶的迭代方法交替更新LLM策略和对偶变量，理论上刻画了分布空间与LLM参数空间之间的原对偶间隙和最优性间隙，证明方法可以找到近最优约束LLM策略。

**[Ask a Strong LLM Judge when Your Reward Model is Uncertain](ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)**

:   提出基于不确定性的路由框架，用SNGP对pairwise reward model做不确定性量化，将高认知不确定性的样本路由到强LLM judge（DeepSeek-R1），在仅调用9.2%~42.5% judge的成本下显著超越随机路由的准确率，且有效改善下游在线RLHF对齐效果。

**[Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)**

:   提出两阶段微调攻击：第一阶段用10个问题配相同拒绝答案使LLM过拟合到窄最优解（尖锐loss landscape），第二阶段用相同10个问题配正常答案触发灾难性遗忘——安全对齐被"忘掉"，仅用完全良性数据即达94.84%越狱成功率，与恶意微调（97.25%）相当且完全绕过审核模型。

**[Can DPO Learn Diverse Human Values? A Theoretical Scaling Law](can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)**

:   建立了 DPO 在多元人类价值设定下的理论泛化框架——通过分析有限梯度步后 reward margin 的动态轨迹，证明了每种价值所需样本量必须随价值类别数 $K$ 对数增长（$Q = \Theta(\log K)$）才能维持泛化性能，揭示了对齐多元化社会价值的统计代价。

**[Capturing Individual Human Preferences with Reward Features](capturing_individual_human_preferences_with_reward_features.md)**

:   提出奖励特征模型（RFM）：学习共享奖励特征 $\phi_\theta(x,y)$，每个用户通过线性权重 $\mathbf{w}_h$ 组合这些特征得到个性化奖励 $r_h = \langle \phi_\theta, \mathbf{w}_h \rangle$，并首次给出多评价者偏好学习的PAC泛化界，证明增加评价者数 $m$ 比增加每人样本数 $n$ 更有效，仅30个样本即可快速适应新用户。

**[Concept-Level Explainability for Auditing & Steering LLM Responses](concept-level_explainability_for_auditing_steering_llm_responses.md)**

:   提出 ConceptX，一种基于概念级（而非 token 级）Shapley 归因的 LLM 可解释性方法，通过语义相似度而非 token 重合度来衡量输入概念对输出的影响，可用于审计偏见和通过 prompt 编辑引导 LLM 输出，在越狱防御中将攻击成功率从 0.463 降至 0.242。

**[Deep Research Brings Deeper Harm](deep_research_brings_deeper_harm.md)**

:   揭示 Deep Research (DR) 智能体的严重安全隐患——即使底层 LLM 能正确拒绝有害请求，部署为 DR 智能体后仍能生成详细专业的危险报告；提出 Plan Injection 和 Intent Hijack 两种针对性越狱方法，以及 DeepREJECT 评估指标，在 6 个 LLM 上验证了 DR 智能体系统性地削弱了对齐机制。

**[DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)**

:   探索GRPO在VideoLLM中的应用，发现"安全门依赖"和"优势消失"两个阻碍有效学习的问题，提出Reg-GRPO（将GRPO loss重建为直接回归优势值的任务，消除clipping/min等安全门操作）和难度感知数据增强策略，在多个视频推理benchmark上显著提升性能。

**[DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](densedpo_finegrained_temporal_preference_optimization_for_vi.md)**

:   提出 DenseDPO，通过三个创新解决视频扩散模型 DPO 训练的根本缺陷：(1) 从 GT 视频加噪去噪构造对齐的视频对消除运动偏差，(2) 在短时间片段而非整个视频上标注偏好提供更密集的学习信号，(3) 用 GPT 等 VLM 自动标注片段级偏好取代人工标注。仅用 1/3 标注数据即大幅提升运动生成质量。

**[Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)**

:   提出 Latent Reward Model (LRM) 和 Latent Preference Optimization (LPO)，将预训练扩散模型本身复用为噪声感知的潜空间奖励模型，在噪声潜在空间直接进行步级偏好优化，相比 Diffusion-DPO 实现 10-28× 训练加速，相比 SPO 实现 2.5-3.5× 加速。

**[EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)**

:   提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度 + 模拟退火）生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，比最强基线的拒绝触发率高 85.34%，并用生成的数据进行 SFT/DPO 微调，将过度拒绝降低 29.85%-45.96%。

**[From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring](from_judgment_to_interference_early_stopping_llm_harmful_outputs_via_streaming_c.md)**

:   提出 Streaming Content Monitor (SCM)——首个原生支持部分检测的流式有害内容监控器，通过 FineHarm 数据集（29K 样本含 token 级标注）和层次一致性感知学习，平均仅需看到 18% 的 response tokens 即可达到 0.95+ macro F1，实现对 LLM 有害输出的实时早停。

**[g-DPO: Scalable Preference Optimization for Protein Language Models](g-dpo_scalable_preference_optimization_for_protein_language_models.md)**

:   针对蛋白质语言模型的偏好优化提出 g-DPO，通过序列空间聚类剪枝冗余对和 union masking 摊销似然计算，在保持标准 DPO 性能的同时实现 1.7-5.4× 训练加速。

**[Gasp Efficient Black-Box Generation Of Adversarial Suffixes For Jailbreaking Llm](gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)**

:   提出GASP框架，通过训练专用的SuffixLLM生成可读的对抗后缀，利用潜在贝叶斯优化（LBO）在连续嵌入空间中高效搜索并用ORPO迭代微调生成器，在完全黑盒设置下实现高攻击成功率且生成的后缀保持人类可读性。

**[Greedy Sampling Is Provably Efficient For Rlhf](greedy_sampling_is_provably_efficient_for_rlhf.md)**

:   证明了在KL正则化的RLHF设置下，直接使用经验估计的贪心采样（无需构建乐观/悲观估计）就能在在线和离线两种设置中实现$O(\log T)$遗憾界和$O(\varepsilon^{-1})$样本复杂度，这是首次在一般偏好模型下达到这些阶数。

**[GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)**

:   通过将 KL 约束奖励最大化的解析解融入梯度权重（零和权重消除配分函数），设计了比 GRPO 更稳定的 LLM 后训练方法 GVPO，在 AIME 上达到 20.72%（GRPO 14.79%），并证明具有唯一全局最优解。

**[HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)**

:   NVIDIA 发布的 40K+ 开源人工标注偏好数据集，覆盖通用/STEM/代码/多语言（13 种语言），训练的奖励模型在 RM-Bench 上达 82.4%（+10%），CC-BY-4.0 许可对商业友好。

**[LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)**

:   将多个奖励模型（RM）的选择建模为上下文多臂老虎机（LinUCB）问题，在迭代 LLM 训练中自适应地为每个 batch 选择最合适的 RM，在推理、指令跟随和长上下文任务上以 2-3 倍效率优势全面超越 RM 集成和单 RM 基线。

**[LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)**

:   建立统一理论框架证明 RLHF/DPO/KTO/BCO 等对齐方法本质上是在估计安全分布 $\mathcal{D}^+$ 与不安全分布 $\mathcal{D}^-$ 之间的散度，由此解释了对齐后隐空间分离现象，并提出基于 KL 散度的 KLDO 对齐方法，在 5 个模型上实现最佳鲁棒性。

**[LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)**

:   提出 LongVPO，一个两阶段 DPO 框架使短上下文 VLM 无需长视频标注即可理解超长视频——阶段1通过锚定短片段构造偏好数据解决位置偏差问题，阶段2通过递归描述+多段推理任务培养跨片段推理能力，仅用 16K 合成样本即超越 SOTA 开源模型。

**[On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)**

:   将 DPO 中的 Bradley-Terry 偏好模型替换为 Rao-Kupper 和 Davidson 扩展，使偏好优化能够显式建模"平局"数据，避免丢弃模糊偏好对，在翻译和数学推理上获得更好的正则化和性能。

**[Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)**

:   通过分布鲁棒优化（DRO）框架提出 WDPO（Wasserstein）和 KLDPO（KL散度）两种鲁棒 DPO 变体，解决用户偏好分布转移导致的对齐失败问题，提供 $O(n^{-1/4})$ 收敛保证，在多维对齐任务和 OpenLLM 榜单上显著优于标准 DPO。

**[Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks](short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)**

:   理论证明并实验验证：防御长度 $\Theta(M)$ 的后缀越狱攻击，只需要在长度 $\Theta(\sqrt{M})$ 的对抗后缀上做对抗训练即可，即"短对抗训练防长越狱"——在5个主流LLM上，20 token 对抗训练可将 120 token 越狱成功率降低至少 30%。

**[Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)**

:   提出 SimNPO，通过移除参考模型依赖并采用长度归一化奖励替代 NPO 的参考模型比较，简化设计但更有效适配数据难度差异，TOFU FQ 从 0.79 提升至 0.91+。

**[Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning](trajectory_bellman_residual_minimization_a_simple_value-based_method_for_llm_rea.md)**

:   TBRM 通过最小化轨迹级贝尔曼残差，将 LLM 输出 logits 视为隐式 Q 值，仅需每个 prompt 一次前向采样即可训练，复杂度远低于 PPO/GRPO 但数学推理性能相当或更优。

**[What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)**

:   从优化理论角度证明：奖励模型的准确率（accuracy）不足以衡量其作为 RLHF "教师"的质量——即使完美准确的奖励模型，如果诱导的奖励方差（reward variance）过低，也会导致 RLHF 目标函数景观平坦，使 policy gradient 优化极慢；不同的语言模型需要不同的奖励模型。
