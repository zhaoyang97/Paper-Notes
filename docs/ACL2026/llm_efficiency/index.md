---
title: >-
  ACL2026 LLM效率论文汇总 · 23篇论文解读
description: >-
  23篇ACL2026的 LLM 效率方向论文解读，涵盖 LLM、推理、扩散模型、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "推理"
  - "扩散模型"
  - "对齐/RLHF"
item_list:
  - u: "alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp/"
    t: "Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference"
  - u: "are_large_language_models_economically_viable_for_industry_deployment/"
    t: "Are Large Language Models Economically Viable for Industry Deployment?"
  - u: "beyond_accuracy_unveiling_inefficiency_patterns_in_tool-integrated_reasoning/"
    t: "Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning"
  - u: "bosch_black-box_binary_optimization_for_short-context_attention-head_selection_i/"
    t: "BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs"
  - u: "breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar/"
    t: "Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models"
  - u: "comet_collaborative_memory_transformer_for_efficient_long_context_modeling/"
    t: "CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling"
  - u: "creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models/"
    t: "CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit"
  - u: "lizard_an_efficient_linearization_framework_for_large_language_models/"
    t: "Lizard: An Efficient Linearization Framework for Large Language Models"
  - u: "mtrouter_cost-aware_multi-turn_llm_routing_with_history-model_joint_embeddings/"
    t: "MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings"
  - u: "multi-drafter_speculative_decoding_with_alignment_feedback/"
    t: "Multi-Drafter Speculative Decoding with Alignment Feedback"
  - u: "native_hybrid_attention_for_efficient_sequence_modeling/"
    t: "Native Hybrid Attention for Efficient Sequence Modeling"
  - u: "racer_retrieval-augmented_contextual_rapid_speculative_decoding/"
    t: "RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding"
  - u: "saber_an_efficient_sampling_with_adaptive_acceleration_and_backtracking_enhanced/"
    t: "Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs"
  - u: "small_data_big_noise_adversarial_training_for_robust_parameter-efficient_fine-tu/"
    t: "Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning"
  - u: "specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati/"
    t: "SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration"
  - u: "speculative_verification_exploiting_information_gain_to_refine_speculative_decod/"
    t: "Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding"
  - u: "structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference/"
    t: "StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference"
  - u: "tandem_riding_together_with_large_and_small_language_models_for_efficient_reason/"
    t: "Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning"
  - u: "task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c/"
    t: "Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios"
  - u: "the_illusion_of_specialization_unveiling_the_domain-invariant_34standing_committ/"
    t: "专业化的幻觉：揭示混合专家模型中的\"常设委员会\""
  - u: "threshold_differential_attention_for_sink-free_ultra-sparse_and_non-dispersive_l/"
    t: "阈值差分注意力：无 Sink、超稀疏且非分散的长上下文注意力"
  - u: "tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_/"
    t: "TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs"
  - u: "understanding_llm_performance_degradation_in_multi-instance_processing_the_roles/"
    t: "Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length"
item_total: 23
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**💬 ACL2026** · **23** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (169)](../../ICLR2026/llm_efficiency/index.md) · [🧪 ICML2026 (48)](../../ICML2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **高频主题：** LLM ×7 · 推理 ×2 · 扩散模型 ×2 · 对齐/RLHF ×2

**[Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)**

:   把 MoE 推理的"激活专家个数"抽象成全局预算 $B$，先用动态规划在层间做最优 Top-K 分配（Alloc-L），再用全局 Top-(K·T) 选择在 token 间重分配（Alloc-T），在 DeepSeek-V2-Lite 上把激活预算砍掉一半还能保持精度，prefill 加速 1.15×、decode 加速 1.34×。

**[Are Large Language Models Economically Viable for Industry Deployment?](are_large_language_models_economically_viable_for_industry_deployment.md)**

:   提出Edge-Eval框架，通过5个部署指标（经济盈亏平衡、智能功耗比、系统密度、冷启动税、量化保真度）在传统T4 GPU上全生命周期评估LLM，揭示<2B小模型在经济和生态维度全面优于7B模型，并发现QLoRA虽降低内存但能耗增加最高7倍的反常现象。

**[Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning](beyond_accuracy_unveiling_inefficiency_patterns_in_tool-integrated_reasoning.md)**

:   提出 PTE（Prefill Token Equivalents），一个基于硬件感知的工具集成推理效率度量指标，统一了内部推理和外部工具使用的成本，并通过大规模实验揭示了四种 TIR 低效模式：确认性工具使用、工具混合、缺乏工具先验和工具格式崩溃。

**[BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](bosch_black-box_binary_optimization_for_short-context_attention-head_selection_i.md)**

:   提出 BOSCH，一种免训练的注意力头级别 SWA 混合方法，将 SWA 头选择建模为大邻域搜索问题并分解为三阶段优化（层重要性探测→自适应比例分配→分组头选择），在 4 个模型 4 种比例设置下系统性超越层级启发式和 6 种静态头级别方法。

**[Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)**

:   提出 AHD（Anchor-based History-stable Decoding），一种无需训练的即插即用动态解码策略，通过动态锚点回溯历史轨迹判定扩散LLM中跨块稳定token，实现早期解锁，在BBH上减少80%解码步数的同时提升3.67%性能。

**[CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)**

:   CoMeT 给已有 LLM 加一个"全局记忆 + FIFO 临时记忆"的双记忆插件，分块处理输入实现常数显存、线性时间复杂度，仅在 32k 上下文上微调就能在 1M token 内任意位置精确找回密码，并提出层级流水线并行让 16×80GB GPU 就能微调 128k 上下文。

**[CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)**

:   本文提出 CreditDecoding，一种无需训练的并行解码加速方法，通过累积 token 级历史证据（轨迹信用）来增强正确但置信度不足的 token，在 LLaDA-8B-Instruct 上实现最高 5.48 倍加速且准确率提升 0.48。

**[Lizard: An Efficient Linearization Framework for Large Language Models](lizard_an_efficient_linearization_framework_for_large_language_models.md)**

:   Lizard 用一个"Gated Linear Attention（全局压缩）+ Anchor Window Attention（局部精度）+ 可学习 gate 替代 RoPE"的混合 subquadratic 注意力替换预训练 Transformer 的 softmax attention，只用 0.04B token 蒸馏就能在 5-shot MMLU 上把现有 linearization 方法甩开 9.4–24.5 分，并配套一个 tensor-core 友好的训练算法把吞吐量提升 32%。

**[MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings](mtrouter_cost-aware_multi-turn_llm_routing_with_history-model_joint_embeddings.md)**

:   MTRouter把多轮Agent中的“每一轮该调用哪个LLM”建模为成本约束下的逐轮路由问题，通过历史-模型联合嵌入预测候选模型对最终任务结果的贡献，在ScienceWorld和HLE上同时提升任务表现并显著降低总调用成本。

**[Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)**

:   本文提出 MetaSD，一个将多个异构草稿器整合到推测解码中的统一框架，将草稿器选择建模为多臂赌博机问题，通过块散度（Block Divergence）奖励信号动态选择与目标 LLM 最对齐的草稿器，在黑盒和白盒配置下一致优于单草稿器方法。

**[Native Hybrid Attention for Efficient Sequence Modeling](native_hybrid_attention_for_efficient_sequence_modeling.md)**

:   本文提出 Native Hybrid Attention (NHA)，将线性 RNN 的长期记忆槽与滑动窗口的短期精确 token 拼接后通过单次 softmax 注意力统一处理，实现层内和层间混合的原生统一——无需额外融合参数即可动态分配长短期注意力权重，在 recall 密集和常识推理任务上超越 Transformer 和其他混合基线。

**[RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)**

:   RACER 提出了一种无需训练的推测解码方法，将基于检索的精确模式匹配与基于 logits 的未来预测统一起来，通过 copy-logit 策略构建 Logits Tree、LRU 驱逐的 AC 自动机构建 Retrieval Tree，在多个基准上实现了超过 2 倍的推理加速。

**[Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs](saber_an_efficient_sampling_with_adaptive_acceleration_and_backtracking_enhanced.md)**

:   本文提出 Saber，一个面向扩散语言模型（DLM）的免训练采样算法，通过自适应加速（根据已建立的上下文动态调整并行解码量）和回溯增强重遮蔽（撤销被新上下文证伪的 token）两种策略，在代码生成上平均提升 Pass@1 1.9% 的同时实现 251.4% 的推理加速。

**[Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning](small_data_big_noise_adversarial_training_for_robust_parameter-efficient_fine-tu.md)**

:   把对抗训练接进参数高效微调（PEFT），用一个统一的鲁棒优化框架 SDBN 在嵌入空间生成最坏扰动，并针对"破坏分词的字符噪声"和"生成式任务"各补一套离散不确定集，在小数据 + 噪声场景下显著提升 LoRA/Adapter/BitFit 的鲁棒性，且不增加任何可训练参数或显存。

**[SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)**

:   提出 SpecBound 自草稿推测解码框架，通过逐层温度退火抑制浅层虚假高置信度预测，并设计有界推测算法自适应控制草稿的深度和宽度，在保持输出无损的同时实现最高 2.33× 的推理加速。

**[Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)**

:   提出推测验证（Speculative Verification, SV），通过引入与草稿模型同等规模的伴随模型（companion model），利用草稿-伴随分布的相似性预测推测准确率，动态调整验证长度以最大化有效吞吐量，在大批量推理中实现相对标准推测解码平均1.4×、最高1.9×的加速。

**[StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)**

:   本文提出 StructKV，一个结构感知的 KV Cache 压缩框架，通过全局入度中心性（Global In-Degree Centrality）跨层累积注意力模式识别全局信息枢纽，动态枢纽层检测（Dynamic Pivot Detection）自适应定位最优压缩层，以及结构传播与解耦（Structural Propagation & Decoupling）分离计算预算和存储预算，在 LongBench 和 RULER 上以 60% prefill + 10% KV 实现了接近全上下文的性能。

**[Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)**

:   Tandem 让大模型只生成 Goal / Planning / Retrieval / Action 四类短思维线索，再由小模型用困惑度和熵判断线索是否足够并完成答案，在 MATH、GSM8K 和 HumanEval 上用约 60% 的计算成本达到或超过单独大模型推理效果。

**[Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)**

:   提出多层级任务画像引导的数据合成框架解决 LLM 路由的冷启动问题，并设计 TRouter——一种将任务类型作为隐变量的路由方法，通过变分推断建模查询-成本-性能关系，在冷启动和域内设置下均实现有效路由。

**[专业化的幻觉：揭示混合专家模型中的"常设委员会"](the_illusion_of_specialization_unveiling_the_domain-invariant_34standing_committ.md)**

:   通过引入 CommitteeAudit 框架，作者发现 MoE 模型中存在一个"常设委员会"——一个紧凑的、持久的专家组合，在不同领域始终被激活并占据大部分路由权重，这与广泛假设的领域特定专业化形成鲜明对比，揭示了稀疏计算内在的集中化结构。

**[阈值差分注意力：无 Sink、超稀疏且非分散的长上下文注意力](threshold_differential_attention_for_sink-free_ultra-sparse_and_non-dispersive_l.md)**

:   TDA 通过结合长度自适应阈值和差分抑制视图，实现无注意力 Sink、99% 精确稀疏、且性能竞争力的长上下文 Transformer 注意力。

**[TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs](tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_.md)**

:   TokenTiming 把草稿模型生成的 token 序列重新编码到目标 tokenizer 空间，再用动态时间规整构造多对多 token 对齐，从而让不同词表的现成小模型也能作为投机解码 draft model，并在多个 14B-70B 目标模型上取得最高 1.57x 的异构投机解码加速。

**[Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length](understanding_llm_performance_degradation_in_multi-instance_processing_the_roles.md)**

:   这篇论文系统评测 16 个 LLM 在 multi-instance processing 中的退化规律，发现性能下降不只是上下文变长造成的，实例数量本身对成功率的影响更强，尤其在 1,000 个以上实例时几乎所有模型都会崩溃且很少主动提醒用户。
