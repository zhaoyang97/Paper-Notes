---
title: >-
  ICML2026 对齐/RLHF论文汇总 · 37篇论文解读
description: >-
  37篇ICML2026的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF、LLM、对抗鲁棒、强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "LLM"
  - "对抗鲁棒"
  - "强化学习"
item_list:
  - u: "adaptive_probe-based_steering_for_robust_llm_jailbreaking/"
    t: "Adaptive Probe-based Steering for Robust LLM Jailbreaking"
  - u: "alignment-aware_decoding/"
    t: "Alignment-Aware Decoding"
  - u: "autoregressive_direct_preference_optimization/"
    t: "Autoregressive Direct Preference Optimization"
  - u: "boosting_direct_preference_optimization_with_penalization/"
    t: "Boosting Direct Preference Optimization with Penalization"
  - u: "consistency_training_can_entrench_misalignment/"
    t: "Consistency Training Can Entrench Misalignment"
  - u: "curriculum_learning_for_safety_alignment/"
    t: "Curriculum Learning for Safety Alignment"
  - u: "decoupling_reasoning_and_confidence_resurrecting_calibration_in_reinforcement_le/"
    t: "Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards"
  - u: "efficient_preference_poisoning_attack_on_offline_rlhf/"
    t: "Efficient Preference Poisoning Attack on Offline RLHF"
  - u: "f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses/"
    t: "$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses"
  - u: "f-tis_harnessing_diverse_models_in_collaborative_grpo/"
    t: "F-TIS: Harnessing Diverse Models in Collaborative GRPO"
  - u: "gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo/"
    t: "GIST: 用梯度子空间投影做 instruction tuning 的 targeted 数据选择"
  - u: "implicit_preference_alignment_for_human_image_animation/"
    t: "Implicit Preference Alignment for Human Image Animation"
  - u: "implicit_safety_alignment_from_crowd_preferences/"
    t: "Implicit Safety Alignment from Crowd Preferences"
  - u: "korean_culture_into_llm_alignment_toward_cultural_coherence/"
    t: "Korean Culture into LLM Alignment: Toward Cultural Coherence"
  - u: "large_language_models_should_learn_personalized_rather_than_aggregated_human_pre/"
    t: "Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences"
  - u: "long_live_the_balance_information_bottleneck_driven_tree-based_policy_optimizati/"
    t: "Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization"
  - u: "mesa_improving_moe_safety_alignment_via_decentralized_expertise/"
    t: "MESA: Improving MoE Safety Alignment via Decentralized Expertise"
  - u: "mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling/"
    t: "Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling"
  - u: "new_wide-net-casting_jailbreak_attacks_risk_large_models/"
    t: "New Wide-Net-Casting Jailbreak Attacks Risk Large Models"
  - u: "operationalising_the_superficial_alignment_hypothesis_via_task_complexity/"
    t: "Operationalising the Superficial Alignment Hypothesis via Task Complexity"
  - u: "picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti/"
    t: "PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization"
  - u: "quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment/"
    t: "Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment"
  - u: "reward_shaping_for_inference-time_alignment_a_stackelberg_game_perspective/"
    t: "Reward Shaping for (Inference-Time) Alignment: A Stackelberg Game Perspective"
  - u: "safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks/"
    t: "Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks"
  - u: "sft_overtraining_predicts_rank_inversion_via_entropy_collapse_under_rlvr/"
    t: "SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR"
  - u: "simultaneous_multi-objective_alignment_across_verifiable_and_non-verifiable_rewa/"
    t: "Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards"
  - u: "spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-/"
    t: "SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection"
  - u: "steerable_cultural_preference_optimization_of_reward_models/"
    t: "Steerable Cultural Preference Optimization of Reward Models"
  - u: "steering_beyond_the_support_adversarial_training_on_unsupervised_jailbroken_acti/"
    t: "Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation"
  - u: "the_realignment_problem_when_right_becomes_wrong_in_llms/"
    t: "The Realignment Problem: When Right becomes Wrong in LLMs"
item_total: 37
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**🧪 ICML2026** · **37** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (102)](../../ICLR2026/llm_alignment/index.md) · [💬 ACL2026 (38)](../../ACL2026/llm_alignment/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×22 · LLM ×5 · 对抗鲁棒 ×5 · 强化学习 ×2

**[Adaptive Probe-based Steering for Robust LLM Jailbreaking](adaptive_probe-based_steering_for_robust_llm_jailbreaking.md)**

:   这篇论文把 probe-based contrastive steering 改造成更强的白盒红队评测工具，用自适应重训练修正有偏 probe，并用激活统计自适应设定 steering 强度，从而显著暴露加固 LLM 的越狱脆弱性。

**[Alignment-Aware Decoding](alignment-aware_decoding.md)**

:   Alignment-Aware Decoding 直接在推理时利用 DPO 模型相对 SFT 参考模型的 token 概率比作为隐式对齐奖励，在无需额外训练或外部 reward model 的情况下，比 greedy、Bo2 和 EFT 更稳定地生成高对齐质量回答，并可进一步产生合成偏好数据改进 DPO。

**[Autoregressive Direct Preference Optimization](autoregressive_direct_preference_optimization.md)**

:   作者发现 DPO 在推导目标函数时其实是"先按整条回答建 Bradley-Terry 偏好模型、事后才假设模型自回归",顺序反了;ADPO 把自回归假设**提前**到建 BT 模型之前——在输出空间的前缀闭包上定义能量函数,推出一个极简的新损失:把 DPO 里的求和符号从 log-sigmoid 内部**挪到外部**,并由此首次区分出"token 长度 $\mu$"与"反馈长度 $\mu'$"两个独立的长度度量,打通从整条回答到单 token 的任意粒度训练。

**[Boosting Direct Preference Optimization with Penalization](boosting_direct_preference_optimization_with_penalization.md)**

:   本文提出 DPOP（Direct Preference Optimization with Penalization），在标准 DPO 偏好损失之外，额外**惩罚"参考模型自己对同一 prompt 贪婪解码出的响应" $y_g$**，并用一个 detached 门控只在"策略当前仍把被拒响应排在被选响应之上"时才激活这个惩罚，从而把 DPO 一直没用上的参考-贪婪信号变成有效的离线对齐信号，在 AlpacaEval 2.0 上以长度控制胜率超过 DPO/SimPO/AlphaDPO。

**[Consistency Training Can Entrench Misalignment](consistency_training_can_entrench_misalignment.md)**

:   本文提出"一致性非中性假说"，通过在 108 个"模型有机体"上评估 7 种一致性训练方法，发现一致性训练并非对齐中性的——它系统性地抑制脆弱的奖励黑客和涌现性错位，但放大稳定的谄媚行为，分布偏移（而非分数选择）是主要驱动因素。

**[Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)**

:   本文提出 Staged-Competence —— 一个把"模型自身的偏好对齐 margin"作为难度分、再用"分阶段更新参考模型 + 阶段内 competence-based 采样"双重课程的 DPO 安全对齐框架，在三种 8B 量级 LLM 上把 OOD 有害回答率平均降 16%、越狱攻击成功率降 20%，同时几乎不损伤通用能力与不引入过度拒答。

**[Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards](decoupling_reasoning_and_confidence_resurrecting_calibration_in_reinforcement_le.md)**

:   本文先理论证明 RLVR（如 GRPO）训练中"提升准确率"与"减小校准误差"两个目标在 Fisher 度量下梯度方向负相关、不可调和，再提出 DCPO：让模型在推理轨迹后显式吐出一段 verbalized 置信度，给推理 token 和置信度 token 分配各自的 reward / advantage / 掩码梯度，从而在保持 GRPO 同等准确率的前提下把 ECE 从 0.435 降到 0.128（相对降 71.6%）。

**[Efficient Preference Poisoning Attack on Offline RLHF](efficient_preference_poisoning_attack_on_offline_rlhf.md)**

:   针对 log-linear DPO 提出"翻一条偏好标签 = 给损失梯度加一个与策略参数无关的固定向量"的关键观察，把目标投毒攻击归约为二值稀疏近似问题，给出基于 LLL 格基约化的 BAL-A 和基于匹配追踪的 BMP-A 两种算法以及可证明的恢复 / 不可能性条件。

**[$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses](f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses.md)**

:   本文给在线 RLHF 在**通用 $f$-divergence 正则**下首次建立 $O(\log T)$ regret 和 $O(1/T)$ 次优 gap 上界，提出两套采样策略：(1) 基于 optimism in face of uncertainty 加 bonus 项；(2) 一个新颖的 **"derivative-as-uncertainty"** 视角——把 $f'$ 当作不确定性信号，从而设计 derivative-based 采样而无需在每轮显式估计 confidence bound。

**[F-TIS: Harnessing Diverse Models in Collaborative GRPO](f-tis_harnessing_diverse_models_in_collaborative_grpo.md)**

:   F-TIS 把"截断重要性采样 (TIS)"与"按 KL 阈值过滤负优势 off-policy 样本"两件事拼到一个 GRPO 损失里，让大小不同、专长不同、甚至只有一部分参数可训的多个 LLM 在同一次去中心化 GRPO 训练中互相喂样本，最终收敛和纯 on-policy 持平，并在 OOD 数学任务上最高带来 +12% 的性能。

**[GIST: 用梯度子空间投影做 instruction tuning 的 targeted 数据选择](gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo.md)**

:   GIST 把"为 target task 挑 instruction tuning 数据"看作 gradient subspace alignment——证明 LESS 等用 Adam states 当 diagonal preconditioner 在 LoRA 上失效（cross-parameter 耦合 + 低秩 task subspace），改用 validation gradients SVD 抽 task-specific 低秩子空间 + cosine similarity 选样本；在 MMLU/TydiQA/BBH 上匹配或超越 LESS，只用 0.29% 存储和 25% 计算时间。

**[Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)**

:   作者提出 Implicit Preference Alignment (IPA)：一种只需"好样本"、不需要构造好/坏配对的后训练方法，通过最大化与预训练参考模型 KL 间隔来等价地最大化隐式奖励，并配合一个把手部 mask 加权进损失的 HALO 模块，让大尺度视频 DiT 在仅 93 个挑选样本下显著改善人体动画的手部保真度。

**[Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)**

:   针对众包偏好数据中"用户目标各异但安全准则共享"的结构，作者证明传统 reward combination 会被多数用户偏好污染且对权重敏感，转而提出 Safe Crowd Preference-based RL：用 VAE 把众包偏好编码成 latent-conditioned 低层 skill，再训练高层策略在 skill 空间组合，从而在没有显式安全奖励的情况下把下游 cost 压到接近 Oracle，同时任务回报基本不掉。

**[Korean Culture into LLM Alignment: Toward Cultural Coherence](korean_culture_into_llm_alignment_toward_cultural_coherence.md)**

:   现有文化安全工作几乎都在做"减法"（该压制哪些输出），这篇论文给出一个"加法"的对应物——为韩国语境**正面定义**什么才算"文化上连贯的回复"，并据此搭一条对齐数据流水线（韩国伤害 taxonomy 种子 → 攻击挖掘 → 文化政策约束下的多模型安全回复 → 三裁判过滤成 DPO 三元组），DPO 微调让六个开源 LLM 的韩国文化安全率普遍上升、却几乎不损伤通用能力。

**[Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences](large_language_models_should_learn_personalized_rather_than_aggregated_human_pre.md)**

:   这是一篇立场论文，主张当前 RLHF 把多元人类偏好聚合成单一奖励信号、本质上是在优化一个"谁都不是"的平均用户，作者从社会选择理论与跨人群实证两路论证个性化对齐的必要性，并提出一套"保留普适安全约束、只在合法维度上做个性化"的有界个性化框架与研究议程。

**[Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization](long_live_the_balance_information_bottleneck_driven_tree-based_policy_optimizati.md)**

:   本文用信息瓶颈 (IB) 理论提出一个可量化"探索-利用平衡"的步级指标 IB-Score, 并据此设计 IB 引导的树采样 (IBTree) + 步级局部/全局优势, 在 Qwen3-1.7B/8B 上比 GRPO 平均提升 2.9–3.6%, 同时在同 token 预算下多采到 50% 轨迹.

**[MESA: Improving MoE Safety Alignment via Decentralized Expertise](mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)**

:   MESA 把 MoE 安全对齐重塑为"在专家上分配安全责任"的资源分配问题，用 KL 正则化的 Sinkhorn 最优传输（OT）从中间档（shoulder region）专家中挑出代价最低的子集做 SFT，同时用 OT 约束的路由损失把安全 token 引到这些专家，从而在 DeepSeek-V2-Lite / Qwen3-30B-A3B 上把 Strata 安全分推到 95+%，并保住 GSM8K 等推理任务接近原始水平。

**[Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)**

:   本文把 Bradley–Terry 奖励模型重写成一个贝叶斯非负因子分析（NFA）的生成过程——局部稀疏的实例隐变量 $\bm{\theta}$ 与全局稀疏的奖励字典 $\Phi$ 同时建模，以"先解耦再去偏"抑制 RLHF 中由长度/风格等捷径特征引起的 reward hacking，并通过 Weibull 重参数化的摊销变分推断把整个框架塞进现代 LLM 主干，在 Unified-Feedback、RewardBench、HHH、MT-Bench 上一致超过 BT、Ensemble、InfoRM 等强基线。

**[New Wide-Net-Casting Jailbreak Attacks Risk Large Models](new_wide-net-casting_jailbreak_attacks_risk_large_models.md)**

:   本文首次定义并系统分析了"广撒网"越狱场景（攻击者同时向一组大模型发起请求，只要任一模型被攻破即视为成功），并据此设计了一种基于 exploration-to-exploitation 调度的"专家化"对抗样本生成器联合训练方法，在多个 LLM/MLLM 上把无外加防御时的攻击成功率推到 100%，揭示现行单模型越狱评估严重低估了真实世界风险。

**[Operationalising the Superficial Alignment Hypothesis via Task Complexity](operationalising_the_superficial_alignment_hypothesis_via_task_complexity.md)**

:   作者用"解决某任务到目标性能所需的最短程序长度"这一算法信息论指标（task complexity）重新定义了表面对齐假设（SAH），把"数据高效/参数高效/推理控制"这三种看似各说各话的证据统一成"在同一条长度–性能 Pareto 曲线上找短程序"的不同策略，并实测发现：把预训练模型适配到数学推理、机器翻译、指令跟随等任务上，常常只需要几千字节到几兆字节的信息，而后训练的作用是把"够到强性能所需的程序长度"压缩好几个数量级。

**[PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)**

:   PICACO 把"让 LLM 在一个 prompt 里同时遵守多个甚至互相冲突的人类价值"形式化为最大化"价值集与响应之间的条件总相关性"(Total Correlation, TC),不动模型参数,通过 EM-like 的"响应增强 + 指令精炼"两步迭代自动搜索一条 meta-instruction,使 GPT-3.5 / LLaMA-3.1-8B / Gemini-1.5-Flash 在 Schwartz、HH 等 5 套最多 8 个价值的组合上都超过 OPRO、Modular Pluralism 等强基线。

**[Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)**

:   作者用 Inglehart-Welzel 文化地图把标注者按"文化区/象限"重新分层，在 8 个安全数据集上用多层级回归（multilevel modeling）证明文化区在控制完人口学（年龄/性别/族裔）之后仍显著解释安全评分的方差（6/8 数据集 $p<0.05$），并提出 Bayesian 的"cultural sensitivity score"量化得出：当前数据集中约 10% 的样本若忽略某一文化象限就会被错标为 safe；进一步实验表明 LLM 当 rater 替身不靠谱，但当"文化敏感样本"的 triage 工具是可行的。

**[Reward Shaping for (Inference-Time) Alignment: A Stackelberg Game Perspective](reward_shaping_for_inference-time_alignment_a_stackelberg_game_perspective.md)**

:   把"该用什么奖励模型来对齐 LLM"建模成一个 Stackelberg 博弈，证明最优奖励是一个**逐 prompt 的阈值奖励**（高于阈值给满分 $B$、低于给 0），并用从基座模型采样的蒙特卡洛估计高效求出阈值，最后用 sigmoid 软化后无缝插进 CD/ARGS 等推理时对齐方法，在几乎零额外开销下把平均奖励和对 baseline 的 Win-Tie 率提到 66% 以上。

**[Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)**

:   本文证明所有现有「在参数空间设约束」的 HFT 防御都会因参数冗余而被绕过，提出 Safety Bottleneck Regularization (SBR) 把防御战场搬到 unembedding 层这一几何瓶颈上：仅锚定 1 个高危 prompt 的最后一层隐状态，就能在 50 epoch 持续 HFT 攻击下把 Harmful Score 压到 < 10，同时不损 benign 任务精度。

**[SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR](sft_overtraining_predicts_rank_inversion_via_entropy_collapse_under_rlvr.md)**

:   作者发现"挑 SFT 阶段 pass@1 最高的检查点去做 GRPO"这条业界默认规则在代码生成上会系统性翻车——SFT 训得越久 pass@1 越高、但 GRPO 后的 pass@10 反而单调下跌（0.806→0.481），根因是过度 SFT 压扁了输出分布、让 GRPO 的组内优势方差归零、梯度消失，作者用一条闭式阈值 $p^*(g)$ 和一套"预训练熵筛查 + 早期熵监控"两阶段诊断把高危检查点提前揪出来。

**[Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards](simultaneous_multi-objective_alignment_across_verifiable_and_non-verifiable_rewa.md)**

:   MAHALO 把"标准化 PRM 训练 + 多动作头 DPO + 带 KV-cache 续存的 PRM 引导解码"拼成一套统一框架，让一个 LLM 在数学（可验证）、人类价值观（不可验证）、多轮辅导（交互式）三类目标上同时被对齐，并且在推理时能通过头权重与 PRM 选择平滑地切换偏好。

**[SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)**

:   SPARD 用"安全投影交替优化（SPAG）+ 相关性-多样性 DPP 安全样本选择"两件套，把"微调后模型必须满足安全损失约束"显式写成约束优化问题，每步先做效用更新，再用闭式投影把参数拉回安全半空间，同时只用 3% 任务相关且彼此互异的安全样本，就把四种有害微调攻击的平均 ASR 从 SFT 的 87.93% 砍到 9.45%，几乎不掉下游精度。

**[Steerable Cultural Preference Optimization of Reward Models](steerable_cultural_preference_optimization_of_reward_models.md)**

:   SCPO 用一个"全局奖励模型"当参照系，先**过滤**掉少数群体里那些和全局共识一致的通用偏好、只留下真正有文化差异的偏好对，再按**散度反比加权**把过激的离群偏好压低权重，从而训出既能代表某国少数群体观点、又不至于过度偏置的可steer奖励模型——在 PRISM、GlobalOpinionQA 两数据集、7 个国家上把少数奖励模型最多提升约 7 个点，且比全量微调省下 170%–280% 的训练数据。

**[Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation](steering_beyond_the_support_adversarial_training_on_unsupervised_jailbroken_acti.md)**

:   论文针对监督式 safety steering 在未见越狱攻击上失效的问题，提出用"无监督潜在方向发现 + 双层对抗训练"在激活空间里凭空模拟出新型 jailbroken 状态，并把这些模拟状态当作对抗样本来训练一个 OT 势函数（其梯度构成空间变化的引导场），在三个 LLM × 六类经典越狱上把攻击成功率压到大多数 <5% 且基本不伤害良性效用。

**[The Realignment Problem: When Right becomes Wrong in LLMs](the_realignment_problem_when_right_becomes_wrong_in_llms.md)**

:   本文把"模型部署后政策变了怎么办"形式化为 Realignment 问题,提出 TRACE 框架:用更强的 proxy 模型把已有 preference pair 三分类 (Invert / Punish / Retain) 后用混合 IPO+NPO+KL 目标做手术式再对齐,无需新一轮人工标注就能跟上政策漂移。

**[Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)**

:   本文提出 SVGT，把价值对齐从"嵌入 backbone 参数/激活"改为"挂一个独立的价值模块"，先在隔离的 value space 里持续判断当前 hidden state 的安全方向，再用一组可学习的 Bridge Token 作为注意力锚点显式引导生成轨迹，在四种 backbone 上把有害分数普遍降低 70% 以上且几乎不损失流畅度。

**[Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)**

:   作者提出 AIR（Anchor Invariance Regularization），把可验证 prompt 当作"锚"、用 stop-gradient 只把开放式变体往锚的表现上拉，作为辅助损失插入 GRPO，在安全/道德/数学三域把 OOD 组级一致性平均提升 33.49%、ID 提升 12.71%。

**[HRC + DSPPO: 用博弈论分解把传递偏好和循环偏好分开学](transitivity_meets_cyclicity_explicit_preference_decomposition_for_dynamic_large.md)**

:   HRC 把人类偏好显式拆成正交的「传递标量分量」（BT 模型）+「循环向量分量」（GPM），用博弈论分解定理证明这种 hybrid 形式既能保 dominant 候选又能建模 RPS 式循环，再配套时变博弈 DSPPO 让对齐过程从「先稳住传递骨架，再学循环细节」走到 Nash 均衡——在 RewardBench 2 上 Gemma-2B-it 平均涨 1.23%、AlpacaEval 2.0 LC win-rate 拉到 44.75%。

**[TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning](truthrl_incentivizing_truthful_llms_via_reinforcement_learning.md)**

:   针对"只优化准确率反而鼓励瞎猜、强行教拒答又过度保守"的两难，TruthRL 用一个区分"答对 / 幻觉 / 拒答"的三元奖励在 GRPO 上直接优化真实性，把幻觉率从 43.5% 压到 19.4%、真实性得分从 5.3% 拉到 37.2%。

**[UDM-GRPO: 统一离散扩散模型的稳定高效 GRPO](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)**

:   通过将最终干净样本定义为动作并使用前向过程重构轨迹——首次成功将 GRPO 集成到离散扩散模型中，解决训练不稳定问题，在 GenEval 等多个基准上达到 SOTA。

**[VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models](valueflow_toward_pluralistic_and_steerable_value-based_alignment_in_large_langua.md)**

:   针对「用 LLM 给价值打绝对分极不稳定、且没法控制价值表达强度」的难题，本文提出 VALUEFLOW——一个把价值「抽取—评测—引导」串成一条流水线的统一框架，核心是分层价值嵌入空间 HIVES、用 Plackett–Luce 排序聚合得到的价值强度库 VIDB，以及基于锚点排序的强度评测器，并在 10 个模型、4 套价值理论上系统刻画了 LLM 的可引导性规律。

**[When Distance Distracts: Representation Distance Bias in BT-Loss for Reward Models](when_distance_distracts_representation_distance_bias_in_bt-loss_for_reward_model.md)**

:   本文把 Bradley-Terry（BT）奖励模型损失的梯度范数拆成「预测误差 × 表示距离」两项，指出表示距离会喧宾夺主——表示相近的难分对即使排错也只得到微弱更新，于是作者提出 NormBT，用一个与表示距离成反比的逐对权重把更新强度重新交还给预测误差，在 RewardBench 的 Reasoning 类上平均提升 5% 以上。
