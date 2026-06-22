---
title: >-
  ICML2025 LLM效率论文汇总 · 12篇论文解读
description: >-
  12篇ICML2025的 LLM 效率方向论文解读，涵盖 LLM、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对齐/RLHF"
item_list:
  - u: "autonomy-of-experts_models/"
    t: "Autonomy-of-Experts Models (AoE)"
  - u: "cooperation_of_experts_fusing_heterogeneous_information_with_large_margin/"
    t: "Cooperation of Experts: Fusing Heterogeneous Information with Large Margin"
  - u: "curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling/"
    t: "Curse of High Dimensionality Issue in Transformer for Long-context Modeling"
  - u: "dssd_efficient_edge-device_llm_deployment_and_collaborative_inference_via_distri/"
    t: "DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding"
  - u: "easyinv_toward_fast_and_better_ddim_inversion/"
    t: "EasyInv: Toward Fast and Better DDIM Inversion"
  - u: "efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l/"
    t: "Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling"
  - u: "ladder-residual_parallelism-aware_architecture_for_accelerating_large_model_infe/"
    t: "Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference"
  - u: "long-short_alignment_for_effective_long-context_modeling_in_llms/"
    t: "Long-Short Alignment for Effective Long-Context Modeling in LLMs"
  - u: "mixture_of_lookup_experts/"
    t: "Mixture of Lookup Experts"
  - u: "moh_multi-head_attention_as_mixture-of-head_attention/"
    t: "MoH: Multi-Head Attention as Mixture-of-Head Attention"
  - u: "nextlong_toward_effective_long-context_training_without_long_documents/"
    t: "NExtLong: Toward Effective Long-Context Training without Long Documents"
  - u: "retraining-free_merging_of_sparse_moe_via_hierarchical_clustering/"
    t: "Retraining-Free Merging of Sparse MoE via Hierarchical Clustering"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🧪 ICML2025** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (169)](../../ICLR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🧪 ICML2026 (48)](../../ICML2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md)

**[Autonomy-of-Experts Models (AoE)](autonomy-of-experts_models.md)**

:   AoE 提出让 MoE 中的 expert 基于自身内部激活范数自主决定是否处理输入（而非由外部 router 决定），通过低秩权重分解降低预计算开销，在 700M-4B 参数语言模型预训练中超越传统 MoE。

**[Cooperation of Experts: Fusing Heterogeneous Information with Large Margin](cooperation_of_experts_fusing_heterogeneous_information_with_large_margin.md)**

:   提出 Cooperation of Experts (CoE) 框架，将异构信息编码为多重网络，通过两级专家设计与大间隔置信张量优化实现专家**协作**（而非竞争），在节点分类任务上全面超越现有 MoE 和多重网络方法。

**[Curse of High Dimensionality Issue in Transformer for Long-context Modeling](curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)**

:   本文从监督学习视角重新审视序列建模中的注意力冗余问题，提出了 Dynamic Group Attention (DGA) 机制，通过将不重要的 token 动态分组聚合来减少注意力计算中的冗余，在保持竞争性能的同时大幅降低推理延迟（LLaMA2-7B 在 16K 上下文下推理速度提升 2.42 倍）。

**[DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding](dssd_efficient_edge-device_llm_deployment_and_collaborative_inference_via_distri.md)**

:   提出分布式拆分推测解码（DSSD）框架，将推测解码的验证阶段拆分到设备端和边缘端，用一次下行传输（LLM的单个词表分布）替代多次上行传输（SLM的$\gamma$个词表分布），在保持推理质量不变的前提下大幅降低通信延迟。

**[EasyInv: Toward Fast and Better DDIM Inversion](easyinv_toward_fast_and_better_ddim_inversion.md)**

:   提出 EasyInv，通过在反演过程中周期性地将当前 latent 状态与前一步 latent 状态加权聚合（类卡尔曼滤波），增强初始 latent 的影响力、抑制噪声累积误差，在不需要迭代优化的前提下达到与迭代方法相当甚至更好的反演质量，同时推理速度提升约 3 倍。

**[Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)**

:   本文提出 Grouped Cross-Attention (GCA) 机制，将 chunk 级别的因果检索（causal retrieval）集成到注意力中实现端到端可学习的检索器，构建的 Differentiable Retrieval-based Transformer (DRT) 在 16M 上下文的 passkey 检索测试中达到近乎完美的准确率，实现了训练长度 1000 倍的长度泛化。

**[Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference](ladder-residual_parallelism-aware_architecture_for_accelerating_large_model_infe.md)**

:   本文提出 Ladder Residual，一种简单的架构修改——将每个模块的输入从上一层的输出改为上上层的输出（错位残差），使模块计算与 AllReduce 通信解耦，从而实现通信与计算的重叠，在 70B 模型 8 卡 TP 推理中实现 29% 的端到端加速，且模型性能与标准 Transformer 持平。

**[Long-Short Alignment for Effective Long-Context Modeling in LLMs](long-short_alignment_for_effective_long-context_modeling_in_llms.md)**

:   本文从模型输出分布的角度提出长度泛化的新视角——长短对齐 (Long-Short Alignment)，指出不同长度输入的输出分布一致性是长度泛化的关键因素，提出 Long-Short Misalignment 度量并将其作为训练正则项，在合成任务和自然语言任务上均显著提升长上下文建模能力。

**[Mixture of Lookup Experts](mixture_of_lookup_experts.md)**

:   提出 MoLE（Mixture of Lookup Experts），将 MoE 中的路由专家输入从中间特征改为 embedding token，使专家可在推理前被重参数化为查找表（LUT）并卸载到存储设备，从而在保持 MoE 级别性能的同时实现与 dense 模型相当的推理速度和显存占用。

**[MoH: Multi-Head Attention as Mixture-of-Head Attention](moh_multi-head_attention_as_mixture-of-head_attention.md)**

:   本文将多头注意力（MHA）重新表述为求和形式，借鉴 MoE 思想提出 Mixture-of-Head Attention（MoH），通过路由器为每个 token 动态选择最相关的注意力头子集，仅激活 50%~90% 的头即可匹配甚至超越标准 MHA 性能，并证明预训练模型（如 LLaMA3-8B）可通过 continue-tuning 转换为 MoH 模型。

**[NExtLong: Toward Effective Long-Context Training without Long Documents](nextlong_toward_effective_long-context_training_without_long_documents.md)**

:   本文提出 NExtLong 框架，通过将文档分割为 meta-chunk 并在 chunk 之间插入从预训练语料检索的硬负例干扰文本来合成长上下文训练数据，迫使模型区分长距离依赖信息和干扰内容，在 HELMET 和 RULER 基准上比此前最佳的长上下文合成方法 Quest 平均提升 7.33%。

**[Retraining-Free Merging of Sparse MoE via Hierarchical Clustering](retraining-free_merging_of_sparse_moe_via_hierarchical_clustering.md)**

:   提出 HC-SMoE，一种基于专家输出层次聚类的无需重训练专家合并框架，通过输出相似度度量和层次聚类实现 SMoE 模型的高效压缩，在 Qwen 和 Mixtral 上分别实现 25%-50% 的专家参数缩减并保持优越性能。
