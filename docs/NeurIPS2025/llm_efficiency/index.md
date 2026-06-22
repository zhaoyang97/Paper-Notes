---
title: >-
  NeurIPS2025 LLM效率论文汇总 · 34篇论文解读
description: >-
  34篇NeurIPS2025的 LLM 效率方向论文解读，涵盖 LLM、布局/合成、对抗鲁棒、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "布局/合成"
  - "对抗鲁棒"
  - "对齐/RLHF"
item_list:
  - u: "3model_speculative_decoding/"
    t: "3-Model Speculative Decoding (PyramidSD)"
  - u: "a_unified_framework_for_establishing_the_universal_approxima/"
    t: "A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures"
  - u: "advancing_expert_specialization_for_better_moe/"
    t: "Advancing Expert Specialization for Better MoE"
  - u: "approximately_aligned_decoding/"
    t: "Approximately Aligned Decoding"
  - u: "constant_bit-size_transformers_are_turing_complete/"
    t: "Constant Bit-Size Transformers Are Turing Complete"
  - u: "critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag/"
    t: "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training"
  - u: "disc_dynamic_decomposition_improves_llm_inference_scaling/"
    t: "DISC: Dynamic Decomposition Improves LLM Inference Scaling"
  - u: "efficient_training-free_online_routing_for_high-volume_multi-llm_serving/"
    t: "Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving"
  - u: "flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe/"
    t: "FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training"
  - u: "from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in/"
    t: "From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers"
  - u: "hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac/"
    t: "Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access"
  - u: "harnessing_the_computation_redundancy_in_vits_to_boost_adversarial_transferabili/"
    t: "Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability"
  - u: "hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c/"
    t: "Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM"
  - u: "jet-nemotron_efficient_language_model_with_post_neural_architecture_search/"
    t: "Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search"
  - u: "l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod/"
    t: "L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models"
  - u: "let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e/"
    t: "Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads"
  - u: "long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l/"
    t: "Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs"
  - u: "loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges/"
    t: "LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?"
  - u: "moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe/"
    t: "MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE"
  - u: "mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite/"
    t: "Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures"
  - u: "omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d/"
    t: "OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding"
  - u: "on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks/"
    t: "On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks"
  - u: "scale-invariant_attention/"
    t: "Scale-invariant Attention"
  - u: "silent_tokens_loud_effects_padding_in_llms/"
    t: "Silent Tokens, Loud Effects: Padding in LLMs"
  - u: "skyladder_better_and_faster_pretraining_via_context_window_scheduling/"
    t: "SkyLadder: Better and Faster Pretraining via Context Window Scheduling"
  - u: "sparta_alignment_collectively_aligning_multiple_language_models_through_combat/"
    t: "SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat"
  - u: "technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context/"
    t: "Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context"
  - u: "tensor_product_attention_is_all_you_need/"
    t: "Tensor Product Attention Is All You Need"
  - u: "the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re/"
    t: "The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition"
  - u: "tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels/"
    t: "Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels"
item_total: 34
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🧠 NeurIPS2025** · **34** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (169)](../../ICLR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🧪 ICML2026 (48)](../../ICML2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **高频主题：** LLM ×5

**[3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)**

:   在标准的draft-target两模型推测解码的中间插入一个"qualifier"模型，构成三层金字塔式解码架构（PyramidSD），利用模型家族天然的熵梯度来分级过滤token，以模糊接受准则放宽匹配阈值，实现最高1.91×的速度提升（在RTX 4090上达到124 tok/s）。

**[A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures](a_unified_framework_for_establishing_the_universal_approxima.md)**

:   建立了统一的理论框架证明各类Transformer架构的万能逼近性(UAP)，核心条件仅两个——前馈层的非线性仿射不变性和注意力层的token可区分性——并利用解析性假设将后者简化为仅需检验两样本情况，成功覆盖softmax、RBF kernel、Performer、BigBird、Linformer等多种实用架构。

**[Advancing Expert Specialization for Better MoE](advancing_expert_specialization_for_better_moe.md)**

:   通过正交性损失（减少专家间投影重叠）和方差损失（增大路由分数差异）双目标优化，在不修改 MoE 架构的前提下将专家重叠减少 45%、路由方差提升 150%，11 个基准任务平均提升 23.79%，同时完全保持负载均衡。

**[Approximately Aligned Decoding](approximately_aligned_decoding.md)**

:   提出 Approximately Aligned Decoding (AprAD)，一种利用投机解码（speculative decoding）中的前缀选择算法来实现LLM受约束生成的方法——在遇到约束违反时，既不像约束生成那样仅回退一步（导致极端概率放大），也不像ASAp那样完全重新采样（计算成本过高），而是通过投机采样智能选择回退位置，在输出分布失真和计算效率之间取得良好平衡。

**[Constant Bit-Size Transformers Are Turing Complete](constant_bit-size_transformers_are_turing_complete.md)**

:   首次证明常数 bit 精度、固定参数数量的 Transformer（仅允许上下文窗口增长）是图灵完备的，并建立了精确的复杂度等价关系 WINDOW[s(n)] = SPACE[s(n)]，表明扩展上下文窗口——而非模型尺寸——已足以实现通用计算。

**[Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)**

:   提出 branched training 方法直接实证测量临界 batch size (CBS)，发现 CBS 在训练早期快速增长后趋于平稳且不依赖模型规模，据此设计 batch size warmup 策略以 43% 更少的梯度步数达到同等甚至更优的训练 loss。

**[DISC: Dynamic Decomposition Improves LLM Inference Scaling](disc_dynamic_decomposition_improves_llm_inference_scaling.md)**

:   DISC 提出了一种动态分解算法，在推理时根据每一步的 z-score（采样奖励的标准化最大值）自动、递归地调整推理步骤的粒度——困难步骤分更细、简单步骤一步跨过——可以即插即用地与贪心搜索、Beam Search、MCTS 结合，在 APPS、MATH、LiveCodeBench 上以更少的 token 预算达到更高的 pass@k。

**[Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)**

:   提出首个无需训练的在线 LLM 路由算法 PORT，通过近似最近邻搜索估计查询特征，并在少量初始查询上一次性优化对偶变量作为路由权重，在有限 token 预算下实现接近离线最优 ($1-o(1)$ 竞争比) 的路由性能，平均较基线提升 3.55× 性能、1.85× 成本效率和 4.25× 吞吐量。

**[FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)**

:   FlowMoE提出统一的流水线调度框架，将MHA计算、门控、专家计算和A2A通信纳入一体化流水线，并使用优先级驱动的all-reduce张量分块机制最大化通信与计算的重叠，在多种真实MoE模型上实现1.13×-1.82×加速、10-39%能耗降低和7-32%内存节省。

**[From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)**

:   通过严格的理论分析证明了预训练数据的多样性（由"max-sum ratio"刻画）决定了单层Transformer学到的是可泛化的induction head还是无法OOD泛化的位置捷径，并给出了使模型学会induction head的最优预训练分布。

**[Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)**

:   提出层次化稀疏注意力（HSA）及 RAMba 架构，通过两阶段 token-to-chunk 相关性学习与硬件对齐 kernel 设计，让 Mamba 获得高效长程随机访问能力，仅在 4K 上下文预训练即可在 64M passkey retrieval 上达到 100% 准确率。

**[Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability](harnessing_the_computation_redundancy_in_vits_to_boost_adversarial_transferabili.md)**

:   深入挖掘 ViT 中数据级和模型级的计算冗余，提出注意力稀疏化、注意力头置换、干净 token 正则化、Ghost MoE 多样化和鲁棒化 token 五种技术，结合在线学习策略动态选择操作，在 ImageNet-1K 上以 86.9% 平均 fooling rate 大幅超越所有基线。

**[Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)**

:   提出层次均衡打包（HBP）方法，通过多级打包分组、均衡批处理、自适应序列并行和稳定损失归一化，解决长短上下文混合 SFT 中的注意力计算不均衡和通信浪费问题，在 DeepSeek-V2 (236B) 上实现 2.4× 训练加速且性能无损。

**[Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)**

:   NVIDIA 提出 PostNAS 流水线——从预训练全注意力模型出发，冻结 MLP 权重，通过四步搜索（全注意力层放置→线性注意力块选择→新注意力块 JetBlock 设计→硬件感知超参搜索）得到混合架构 Jet-Nemotron，2B 模型在 MMLU-Pro 上超越 Qwen3-1.7B 同时生成吞吐提升 47×。

**[L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)**

:   L-MTP 在多token预测（MTP）基础上引入跳跃机制，预测非相邻位置的token（如位置1,3,5,7而非1,2,3,4），通过"后向查找"解码策略复用先前预测填补空隙，在3B-12B模型上实现22%推理加速的同时保持或提升任务性能。

**[Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)**

:   提出三种离散时间深度混合专家(MoE)生存分析架构，其中 Personalized MoE 通过让每个专家为每位患者生成定制化事件分布，同时实现出色的聚类、校准和预测精度。

**[Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)**

:   提出动态分层稀疏注意力 (DHSA)，通过自适应 chunk 分割 + chunk 级相似度预测 + 上采样到 token 级的分层框架，在不重训基座模型的前提下将密集注意力替换为稀疏注意力，在 Gemma2/3 上实现与密集注意力同等精度、20-60% prefill 延迟降低和 35% 峰值内存节省。

**[LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?](loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)**

:   构建覆盖法律/金融/游戏/代码四大真实领域、长度16K-2M token的长依赖推理基准LooGLE v2，设计10类领域特定任务共1,934个QA实例，评估10个LLM发现最强模型GPT-4.1仅59.2%，揭示当前LLM在真实长依赖场景下的根本不足。

**[MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)**

:   挑战"投机解码对MoE无效"的传统认知，理论与实验证明在中等batch size下MoE反而比稠密模型更受益于投机解码，提出target efficiency这一系统级指标来量化加速瓶颈，并构建了可靠的性能预测模型，在Qwen2-57B-A14B上实现最高2.29×加速。

**[Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)**

:   提出 Mozart 算法-硬件协同设计框架，通过专家聚类分配、细粒度流式调度和 3.5D 晶粒架构（NoP-Tree + 分层存储），在三个 MoE-LLM 上实现 1.9× 以上的训练加速。

**[OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)**

:   提出 OmniDraft 框架，通过在线 n-gram 缓存实现跨词表推测解码、混合蒸馏损失在线对齐草稿模型与目标模型、并结合自适应起草长度控制，使单个轻量 Llama-68M 模型可为 Vicuna-7B、Qwen2-7B、Llama3-8B 等不同目标模型提供推测解码加速（1.5-2x）。

**[On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)**

:   首次系统分析 MoE 在结构化复杂任务上的表达能力：证明浅层 MoE 可在低维流形上克服维度诅咒（近似速率由内在维度 $d$ 而非环境维度 $D$ 决定），深层 MoE 通过 $E$ 专家 × $L$ 层的分层组合可高效近似有 $E^L$ 段的分段函数，远超朴素上界 $LE$。

**[Scale-invariant Attention](scale-invariant_attention.md)**

:   借鉴自然图像的尺度不变性，提出对 attention logits 做位置相关的乘性缩放和加性偏移变换，使注意力在不同 token 范围上的总权重和稀疏度满足尺度不变性，从而实现从短序列训练到长序列推理的零样本泛化（4k→64k 仅需一个超参数 $\tau$）。

**[Silent Tokens, Loud Effects: Padding in LLMs](silent_tokens_loud_effects_padding_in_llms.md)**

:   系统性研究了padding token在未被正确掩码时对LLM的影响，发现即使少量padding也会漂移隐层表示、降低生成质量、不可预测地改变偏见，而128个padding token可将Llama-3.1-8B的有害提示攻击成功率从8%飙升到77.5%，本质上实现了jailbreak。

**[SkyLadder: Better and Faster Pretraining via Context Window Scheduling](skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)**

:   通过上下文窗口短到长的渐进式调度策略 SkyLadder，在固定计算量下实现更优的预训练效率（节省 22% 训练时间）和更好的模型性能（+3.7%），反驳了"长上下文=好性能"的业界信念。

**[SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat](sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)**

:   让多个LLM组成"斯巴达部落"互相竞技和互评，通过声誉加权的判断聚合生成偏好对，再用DPO迭代训练所有模型，在12个任务中的10个上超越Self-Rewarding等自对齐基线，平均提升7%。

**[Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context](technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)**

:   借鉴优化软件基准方法论，用性能比率精确量化ICL相对贝叶斯最优估计器的样本效率，发现存在"二分法"——少射下(≤15个演示)效率接近最优(仅多10%)而多射下(>40个演示)急剧恶化(多45%)，信息论分析证明这源于不可消除的非递减过剩风险，是ICL机制的内在限制。

**[Tensor Product Attention Is All You Need](tensor_product_attention_is_all_you_need.md)**

:   通过上下文张量积分解将 Q/K/V 表示为低秩因子的加权和，将 KV 缓存压缩至 1/10~1/16，同时在验证损失和下游任务精度上超越标准 MHA/MQA/GQA/MLA。

**[The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition](the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)**

:   通过理论分析和受控实验研究 sparse attention 的涌现机制，揭示涌现时间遵循关于序列长度和维度的幂律关系 $T_\epsilon \propto \sqrt{d} \cdot T$，并发现 in-context 和 cross-sample 两种数据重复策略都能加速涌现，为理解 LLM 能力涌现提供了统一的 sparse attention 视角。

**[Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)**

:   提出 TFLA（Tiled Flash Linear Attention）算法，通过二层序列并行化和 tiling 优化，实现高效的线性 RNN/mLSTM 内核，相比 FlashAttention 3 和 Mamba 2 获得显著墙钟加速（训练 >2x vs Mamba 2），同时保持等价的模型精度。

**[UMoE: Unifying Attention and FFN with Shared Experts](umoe_unifying_attention_and_ffn_with_shared_experts.md)**

:   通过重新表述多头注意力机制，揭示其与 FFN 共有的"两层矩阵乘法"结构，据此提出 UMoE 统一架构——在注意力和 FFN 层使用相同设计的专家并支持参数共享，在 Base(134M) 和 Large(1.1B) 模型上均优于现有 FFN-MoE 和 Attention-MoE 基线。

**[Unmasking COVID-19 Vulnerability in Nigeria: Mapping Risks Beyond Urban Hotspots](unmasking_covid-19_vulnerability_in_nigeria_mapping_risks_beyond_urban_hotspots.md)**

:   本文针对尼日利亚各州构建了一个综合 COVID-19 脆弱性风险评分体系,整合人口密度、贫困、医疗可及性和年龄风险四个维度,并通过 GIS 地图可视化热点区域,为公共卫生资源分配提供数据驱动的决策工具。

**[Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding](yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)**

:   提出 Yggdrasil，一个延迟最优的推测解码系统，通过 Equal-Growth Tree (EGT) 结构实现编译友好的动态草稿、延迟感知优化目标替代传统 AAL 指标、以及阶段调度运行时减少 CPU-GPU 协调开销，在 A100/A40 上实现了最高 3.98× 的端到端加速。

**[ZeroS: Zero-Sum Linear Attention for Efficient Transformers](zeros_zero-sum_linear_attention_for_efficient_transformers.md)**

:   通过移除 softmax 的零阶均匀项 $1/t$，构建零和权重的线性注意力机制 ZeroS，突破凸组合只能做加法混合的限制，支持单层内的差分/对比操作，在保持 $O(Nd^2)$ 线性复杂度的同时，在多个序列建模基准上匹配甚至超越标准 softmax 注意力。
