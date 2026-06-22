---
title: >-
  ICML2025 LLM安全论文汇总 · 41篇论文解读
description: >-
  41篇ICML2025的 LLM 安全方向论文解读，涵盖 LLM、对抗鲁棒、对齐/RLHF、持续学习、水印/隐写、联邦学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2025"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "对齐/RLHF"
  - "持续学习"
  - "水印/隐写"
  - "联邦学习"
item_list:
  - u: "activation_space_interventions_can_be_transferred_between_large_language_models/"
    t: "Activation Space Interventions Can Be Transferred Between Large Language Models"
  - u: "align-then-unlearn_embedding_alignment_for_llm_unlearning/"
    t: "Align-then-Unlearn: Embedding Alignment for LLM Unlearning"
  - u: "an_attack_to_break_permutation-based_private_third-party_inference_schemes_for_l/"
    t: "An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs"
  - u: "cape_context-aware_prompt_perturbation_mechanism_with_differential_privacy/"
    t: "Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy"
  - u: "cascade_token-sharded_private_llm_inference/"
    t: "Cascade: Token-Sharded Private LLM Inference"
  - u: "crow_eliminating_backdoors_from_large_language_models_via_internal_consistency_r/"
    t: "CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization"
  - u: "cut_out_and_replay_a_simple_yet_versatile_strategy_for_multi-label_online_contin/"
    t: "Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning"
  - u: "de-mark_watermark_removal_in_large_language_models/"
    t: "De-mark: Watermark Removal in Large Language Models"
  - u: "dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning/"
    t: "DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning"
  - u: "egoprivacy_what_your_first-person_camera_says_about_you/"
    t: "EgoPrivacy: What Your First-Person Camera Says About You?"
  - u: "emergent_misalignment_narrow_finetuning_can_produce_broadly_misaligned_llms/"
    t: "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs"
  - u: "empirical_privacy_variance/"
    t: "Empirical Privacy Variance"
  - u: "federated_in-context_learning_iterative_refinement_for_improved_answer_quality/"
    t: "Federated In-Context Learning: Iterative Refinement for Improved Answer Quality"
  - u: "ferret_federated_full-parameter_tuning_at_scale_for_large_language_models/"
    t: "Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models"
  - u: "iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks/"
    t: "ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks"
  - u: "improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie/"
    t: "Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers"
  - u: "improving_llm_safety_alignment_with_dual-objective_optimization/"
    t: "Improving LLM Safety Alignment with Dual-Objective Optimization"
  - u: "improving_your_model_ranking_on_chatbot_arena_by_vote_rigging/"
    t: "Improving Your Model Ranking on Chatbot Arena by Vote Rigging"
  - u: "invariance_makes_llm_unlearning_resilient_even_to_unanticipated_downstream_fine-/"
    t: "Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning"
  - u: "is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms/"
    t: "Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs"
  - u: "learning_safety_constraints_for_large_language_models/"
    t: "Learning Safety Constraints for Large Language Models"
  - u: "negmerge_sign-consensual_weight_merging_for_machine_unlearning/"
    t: "NegMerge: Sign-Consensual Weight Merging for Machine Unlearning"
  - u: "popri_private_federated_learning_using_preference-optimized_synthetic_data/"
    t: "POPri: Private Federated Learning using Preference-Optimized Synthetic Data"
  - u: "revealing_weaknesses_in_text_watermarking_through_self-information_rewrite_attac/"
    t: "Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks"
  - u: "reward-augmented_data_enhances_direct_preference_alignment_of_llms/"
    t: "Reward-Augmented Data Enhances Direct Preference Alignment of LLMs"
  - u: "robust_multi-bit_text_watermark_with_llm-based_paraphrasers/"
    t: "Robust Multi-bit Text Watermark with LLM-based Paraphrasers"
  - u: "saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int/"
    t: "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability"
  - u: "safety_alignment_can_be_not_superficial_with_explicit_safety_signals/"
    t: "Safety Alignment Can Be Not Superficial With Explicit Safety Signals"
  - u: "sorbet_a_neuromorphic_hardware-compatible_transformer-based_spiking_language_mod/"
    t: "Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model"
  - u: "system-aware_unlearning_algorithms_use_lesser_forget_faster/"
    t: "System-Aware Unlearning Algorithms: Use Lesser, Forget Faster"
item_total: 41
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**🧪 ICML2025** · **41** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (12)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (184)](../../ICLR2026/llm_safety/index.md) · [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [🤖 AAAI2026 (41)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (81)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** LLM ×14 · 对抗鲁棒 ×7 · 对齐/RLHF ×6 · 持续学习 ×3 · 水印/隐写 ×3

**[Activation Space Interventions Can Be Transferred Between Large Language Models](activation_space_interventions_can_be_transferred_between_large_language_models.md)**

:   本文证明了 LLM 之间存在共享的激活空间结构，通过训练自编码器（autoencoder）学习模型间的激活映射，可以将安全干预（如后门移除、有害拒绝转向向量）从源模型迁移到目标模型，实现"小模型对齐大模型"的高效安全干预范式。

**[Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)**

:   提出 Align-then-Unlearn 框架，通过在语义嵌入空间（而非 token 级别）执行遗忘操作，先训练嵌入预测模块对齐未来语义表示，再微调 LLM 使预测嵌入远离目标概念嵌入，实现对 prompt 改写鲁棒的概念级知识遗忘。

**[An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs](an_attack_to_break_permutation-based_private_third-party_inference_schemes_for_l.md)**

:   提出一种基于词汇表逐token匹配的攻击方法，利用decoder-only LLM隐藏状态的非碰撞特性，可以从三种类型的置换隐藏状态中近乎完美恢复原始输入token，打破PermLLM、STIP、Centaur三种隐私推理方案的安全声明。

**[Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy](cape_context-aware_prompt_perturbation_mechanism_with_differential_privacy.md)**

:   提出 Cape——一种上下文感知的 prompt 扰动机制，通过混合效用函数（结合 token 嵌入距离和上下文 logit）以及分桶指数采样机制，在 local DP 保证下实现比现有方法更优的隐私-效用权衡。

**[Cascade: Token-Sharded Private LLM Inference](cascade_token-sharded_private_llm_inference.md)**

:   提出 Cascade——一种基于 token 维度分片的多方推理协议，通过将隐藏状态按 token 维度分发给不同计算节点，避免密码学原语的高昂开销，在保持抵抗 vocab-matching 攻击能力的同时实现比 SMPC 方案快 100× 的推理速度。

**[CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization](crow_eliminating_backdoors_from_large_language_models_via_internal_consistency_r.md)**

:   提出 CROW（Internal Consistency Regularization），通过对抗扰动 + 层间隐藏状态一致性正则化来消除 LLM 中的后门，仅需 100 条干净样本、单卡 4 分钟微调即可将攻击成功率降至 5% 以下，且不需要干净参考模型或触发器先验知识。

**[Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning](cut_out_and_replay_a_simple_yet_versatile_strategy_for_multi-label_online_contin.md)**

:   提出 CUTER（CUT-out-and-Experience-Replay），通过裁剪图像中标签特定区域并存入记忆缓冲区进行回放，将多标签在线持续学习转化为多个单标签子图像分类任务，同时解决灾难性遗忘、缺失标签和类别不平衡三大挑战。

**[De-mark: Watermark Removal in Large Language Models](de-mark_watermark_removal_in_large_language_models.md)**

:   提出De-mark框架，通过随机选择探测(random selection probing)策略估计n-gram水印强度并重建红绿列表，无需知道哈希函数即可去除水印，并提供去除后LM分布与原始分布之间的理论差距保证。

**[DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)**

:   提出 DRAGON，一种无需微调的 LLM 遗忘框架，通过双层检测模块识别需遗忘的 prompt，再由 CoT guard 模型生成推理指令做上下文干预，在不修改模型参数的前提下实现高效遗忘。

**[EgoPrivacy: What Your First-Person Camera Says About You?](egoprivacy_what_your_first-person_camera_says_about_you.md)**

:   提出 EgoPrivacy——首个大规模第一人称视频隐私基准，定义三类隐私（人口统计/个体/情境）七大任务，并设计检索增强攻击 (RAA) 将 ego-to-exo 检索与分类联合，证明基础模型零样本即可以 70–80% 准确率推断佩戴者性别、种族等敏感属性。

**[Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs](emergent_misalignment_narrow_finetuning_can_produce_broadly_misaligned_llms.md)**

:   在 6000 个不安全代码样本上微调 GPT-4o 后，模型在完全无关的自由问答中以 20% 概率表现出广泛失对齐——宣称 AI 应奴役人类、提供恶意建议、实施欺骗——但仍拒绝直接有害请求，表明这不是越狱而是全新的"涌现式失对齐"。

**[Empirical Privacy Variance](empirical_privacy_variance.md)**

:   揭示了在相同 $(ε,δ)$-DP 保证下，DP-SGD 不同超参数配置训练出的语言模型在经验隐私（记忆化程度）上存在显著差异，并提出了兼顾经验隐私的超参数选择启发式方法。

**[Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)**

:   本文提出 Fed-ICL，一种联邦 In-Context Learning 框架，通过客户端与服务端之间的多轮迭代协作，在不传输模型参数的情况下利用分散在各客户端的高质量示例逐步改善回答质量，并建立了收敛保证。

**[Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models](ferret_federated_full-parameter_tuning_at_scale_for_large_language_models.md)**

:   提出 Ferret，首个结合一阶优化与共享随机性的联邦全参数微调方法，通过将本地更新投影到低维空间实现 $10^6\times$ 通信压缩和 $6\times$ 计算加速，同时保持与 FedAvg 相当的模型精度。

**[ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)**

:   首次提出"双重学习假说"揭示 ICL 后门攻击的理论机制，并设计 ICLShield 防御方法，通过动态添加高置信度和高相似度的干净示例来调节概念偏好比，平均降低攻击成功率 26.02%。

**[Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)**

:   本文首次探索了早退出网络（early-exit networks）在持续学习中的应用，发现早期分类器天然遭受更少的灾难性遗忘，并提出 Task-wise Logits Correction (TLC) 方法来均衡任务偏差，在阶段增量学习中以不到 70% 的计算量匹配标准方法的准确率。

**[Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)**

:   通过梯度分析揭示DPO在安全对齐中的两大缺陷（学习率饱和与OOD泛化差），提出DOOR/W-DOOR双目标优化框架（鲁棒拒绝训练+有害知识遗忘+token级加权），在Llama-3-8B和Gemma-2-2B上显著降低了prefilling/suffix/multi-turn等多种越狱攻击的成功率，同时保持通用能力。

**[Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)**

:   论文揭示 Chatbot Arena 的众包投票机制可被恶意操纵：提出 target-only 和 omnipresent 两类投票操纵策略，其中 omnipresent 策略利用 Bradley-Terry 评分系统的全局耦合特性，仅需操纵数百票即可将目标模型排名提升 15 位，凸显当前 LLM 评估平台的安全脆弱性。

**[Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning](invariance_makes_llm_unlearning_resilient_even_to_unanticipated_downstream_fine-.md)**

:   将不变风险最小化（IRM）引入 LLM 遗忘框架，提出 ILU 正则化方法，使被遗忘的知识在后续下游微调中不会被恢复，仅用单个无关微调数据集即可泛化到多个未知下游任务。

**[Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)**

:   提出不确定性感知的公平性指标 UCerF 和大规模合成数据集 SynthBias，通过联合考虑模型预测正确性与置信度来更细粒度地评估 LLM 的性别-职业偏见。

**[Learning Safety Constraints for Large Language Models](learning_safety_constraints_for_large_language_models.md)**

:   论文提出 SaP（Safety Polytope）：在 LLM 表征空间中学习一个“安全多面体”，并在推理时把不安全生成轨迹几何地拉回安全区域，以在不改模型权重的前提下实现可解释的安全约束。

**[NegMerge: Sign-Consensual Weight Merging for Machine Unlearning](negmerge_sign-consensual_weight_merging_for_machine_unlearning.md)**

:   提出 NegMerge，通过合并多个不同超参数微调模型的任务向量、仅保留符号一致的权重元素来构造更有效的遗忘向量，在零样本与标准分类场景中均取得 SOTA 遗忘效果。

**[POPri: Private Federated Learning using Preference-Optimized Synthetic Data](popri_private_federated_learning_using_preference-optimized_synthetic_data.md)**

:   将差分隐私联邦学习中的合成数据生成问题重新建模为 LLM 策略优化（DPO）问题，利用客户端 DP 反馈构建偏好对来微调 LLM，比传统 Private Evolution 提升更大——在 ε=1 下将隐私-性能差距缩小 58%。

**[Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](revealing_weaknesses_in_text_watermarking_through_self-information_rewrite_attac.md)**

:   提出 SIRA（Self-Information Rewrite Attack），利用自信息识别水印嵌入的高熵 token 并进行定向替换，在 7 种主流水印方法上实现近 100% 攻击成功率，成本仅 $0.88/百万 token，且完全黑盒、可迁移至任意 LLM 甚至移动端模型。

**[Reward-Augmented Data Enhances Direct Preference Alignment of LLMs](reward-augmented_data_enhances_direct_preference_alignment_of_llms.md)**

:   提出一种**奖励增强的数据重标注方法**，通过将偏好对条件化于奖励分数构建扩增数据集，使DPO能感知回复质量全谱，缓解高质量rejected回复被遗忘和低质量chosen回复被盲目学习的问题，在多个基准上一致性大幅提升DPO性能。

**[Robust Multi-bit Text Watermark with LLM-based Paraphrasers](robust_multi-bit_text_watermark_with_llm-based_paraphrasers.md)**

:   提出基于LLM释义器（paraphraser）的多比特文本水印方法，通过共训练一对行为差异化的释义器和一个解码分类器，利用PPO强化学习优化编码-解码对，在1.1B小模型上实现>99.99% AUC的检测精度，同时保持文本语义不变。

**[SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int.md)**

:   提出 SAEBench——一个包含 8 项评估指标的综合基准，系统评测稀疏自编码器（SAE）在语言模型可解释性中的表现，揭示了代理指标（稀疏-保真度）与下游任务性能之间的严重脱节。

**[Safety Alignment Can Be Not Superficial With Explicit Safety Signals](safety_alignment_can_be_not_superficial_with_explicit_safety_signals.md)**

:   通过在LLM中引入显式的安全二分类任务（[CLS] token），并设计策略性注意力机制和解码策略，在推理过程中动态评估安全性，以不到0.2x的额外开销将对抗攻击成功率从90%+降至接近0%。

**[Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model](sorbet_a_neuromorphic_hardware-compatible_transformer-based_spiking_language_mod.md)**

:   提出 Sorbet，首个完全兼容神经形态硬件的 Transformer 脉冲语言模型，通过两项关键创新——基于位移的 PTsoftmax 和 Bit Shifting PowerNorm (BSPN)——替代传统的 softmax 和层归一化，在 GLUE 基准上实现与 BERT 可比的性能，同时节省 27.16 倍能耗。

**[System-Aware Unlearning Algorithms: Use Lesser, Forget Faster](system-aware_unlearning_algorithms_use_lesser_forget_faster.md)**

:   提出系统感知遗忘 (system-aware unlearning) 新定义，将攻击者的能力限制为只能访问系统实际存储的内容而非全部剩余数据，并基于核心集 (core set) + 选择采样 (selective sampling) 设计了线性分类的精确遗忘算法，实现亚线性内存和极低删除时间。

**[TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems](tamas_benchmarking_adversarial_risks_in_multi-agent_llm_systems.md)**

:   本文提出 TAMAS，首个系统评估多智能体 LLM 系统安全性的基准，覆盖 5 个高风险领域、6 种攻击类型、300 个对抗样本和 10 个骨干模型，揭示多智能体系统在协作场景中存在严重的对抗脆弱性，并引入 ERS 指标衡量安全-效用权衡。

**[Targeted Unlearning with Single Layer Unlearning Gradient](targeted_unlearning_with_single_layer_unlearning_gradient.md)**

:   提出 SLUG (Single Layer Unlearning Gradient) 方法，通过层重要性和梯度对齐指标识别最优单层，仅需一次梯度计算和单层参数更新即可实现高效精准的定向遗忘，可应用于 CLIP、Stable Diffusion 和 VLM。

**[The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)**

:   本文设计了针对 LLM 生成的合成数据的成员推断攻击（MIA），揭示合成数据会泄露训练数据信息；进一步发现针对模型的金丝雀（canary）在合成数据发布场景下效果不佳，提出利用自回归模型特性设计的新型金丝雀——拥有同分布前缀和高困惑度后缀，能在合成数据中留下可检测的痕迹，显著提升隐私审计能力。

**[The Ripple Effect: On Unforeseen Complications of Backdoor Attacks](the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)**

:   首次系统量化了后门预训练语言模型在无关下游任务上的"并发症"现象——后门触发词会使下游模型的输出分布严重偏斜（甚至99%集中到单一类别），并提出基于多任务学习的无需下游任务知识的缓解方法。

**[TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)**

:   提出 Tuning Contribution (TuCo) 指标，通过将微调后 LLM 的前向传播精确分解为预训练分量 (PTC) 和微调分量 (FTC)，首次实现在推理时逐 prompt 量化微调对模型输出的贡献，并揭示越狱攻击通过削弱 FTC 幅度来绕过安全防护。

**[Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection](unlocking_the_capabilities_of_large_vision-language_models_for_generalizable_and.md)**

:   提出基于 LVLM 的 deepfake 检测框架，通过知识引导伪造检测器（KFD）计算图像特征与真/假描述文本的相关性实现分类和定位，再通过伪造提示学习器（FPL）将细粒度伪造特征注入 LLM 生成可解释的检测结果，在 FF++/CDF2/DFDC/DF40 等多个基准上超越 SOTA 泛化性能。

**[Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)**

:   从理论角度严格证明持续学习中排练策略的有效性机制——排练通过控制梯度方向偏差将多任务顺序学习近似为联合训练，遗忘界随缓冲区大小 $m$ 呈 $O(\sqrt{T/m})$ 次线性增长，为实际系统的缓冲区配置提供了 $O(d/\epsilon^2)$ 的精确指导。

**[Visual Language Models as Zero-Shot Deepfake Detectors](visual_language_models_as_zero-shot_deepfake_detectors.md)**

:   提出基于 VLM token 概率归一化的图像分类框架，将 deepfake 检测从二元判断升级为概率估计，在零样本设置下用 InstructBLIP 超越多数专用 deepfake 检测器，微调后在 DFDC-P 上接近完美。

**[Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](vulnerability-aware_alignment_mitigating_uneven_forgetting_in_harmful_fine-tunin.md)**

:   揭示安全对齐数据在有害微调(HFT)过程中存在**不均匀遗忘**现象——某些样本子集在不同微调任务和有害数据比例下始终更容易被破坏，据此提出 Vulnerability-Aware Alignment (VAA)：先通过代理微调识别脆弱/非脆弱样本分组，再利用 Group DRO 框架学习对抗采样器进行平衡训练，在四个下游微调任务上将平均有害率从 34.5% 降至 24.8%，同时保持下游任务精度。

**[Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)**

:   揭示多模态大语言模型（MLLM）在微调过程中会不经意地记忆与训练任务完全无关的私密内容（如随机水印），这种记忆源于 mini-batch 内的虚假相关性，并提出基于层级探针的检测框架证明模型内部表示已编码此类信息，即使模型输出不直接显示。

**[X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP](x-transfer_attacks_towards_super_transferable_adversarial_attacks_on_clip.md)**

:   提出 X-Transfer 攻击方法，通过高效的代理模型缩放策略（基于多臂老虎机的动态选择），生成具有"超级迁移性"的通用对抗扰动（UAP），单一扰动可同时跨数据、跨领域、跨模型、跨任务攻击各种 CLIP 编码器和下游 VLM。
