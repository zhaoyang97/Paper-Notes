---
title: >-
  AAAI2026 LLM效率论文汇总 · 9篇论文解读
description: >-
  9篇AAAI2026的 LLM 效率方向论文解读，涵盖时序预测、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "LLM"
item_list:
  - u: "connectivity-guided_sparsification_of_2-fwl_gnns_preserving_full_expressivity_wi/"
    t: "Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity"
  - u: "harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex/"
    t: "Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models"
  - u: "hn-mvts_hypernetwork-based_multivariate_time_series_forecasting/"
    t: "HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting"
  - u: "how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-/"
    t: "How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts"
  - u: "intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora/"
    t: "InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE"
  - u: "judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti/"
    t: "Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction"
  - u: "moetta_test-time_adaptation_under_mixed_distribution_shifts_with_moe-layernorm/"
    t: "MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm"
  - u: "resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning/"
    t: "Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning"
  - u: "scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai/"
    t: "Scaling and Transferability of Annealing Strategies in Large Language Model Training"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🤖 AAAI2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (169)](../../ICLR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🧪 ICML2026 (48)](../../ICML2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

**[Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity](connectivity-guided_sparsification_of_2-fwl_gnns_preserving_full_expressivity_wi.md)**

:   Co-Sparsify 提出一种基于连通性感知的稀疏化框架，通过将 3-节点交互限制在双连通分量内、2-节点交互限制在连通分量内，消除可证明冗余的计算，在保持完整 2-FWL 表达力的同时显著提升效率，在合成子结构计数任务和 ZINC、QM9 等基准上取得 SOTA。

**[Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)**

:   首次系统研究长上下文语言模型中参数知识(parametric knowledge)对生成的影响，发现其影响随上下文长度增长而增强，且现有方法提升外部检索能力会抑制参数召回能力，据此提出Hybrid Needle-in-a-Haystack测试来同时评估两种能力。

**[HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)**

:   提出 HN-MVTS，利用超网络(HyperNetwork)为每个通道生成特定的最后一层权重，在通道独立(CI)和通道依赖(CD)之间取得平衡，作为即插即用模块可提升 DLinear、PatchTST、TSMixer 等多种主干模型的预测精度，且不增加推理时间。

**[How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)**

:   提出MASS框架，通过基于梯度的语义漂移检测自适应扩展MoE专家池，并结合Top-p置信度路由策略，在无需超参搜索的情况下自动发现最优专家数量，同时增强专家间的语义分化。

**[InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)**

:   提出 InterMoE，通过 Dynamic Temporal-Selective MoE 架构解决文本驱动的双人 3D 交互运动生成中的个体特征保持和语义忠实度问题：Synergistic Router 融合语义和运动学特征引导路由，Dynamic Temporal Selection 让专家动态选择关键时间帧，在 InterHuman 上 FID 降低 9%、InterX 上降低 22%。

**[Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)**

:   提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

**[MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm](moetta_test-time_adaptation_under_mixed_distribution_shifts_with_moe-layernorm.md)**

:   本文提出 MoETTA，一种将 LayerNorm 重参数化为多个结构解耦专家分支的测试时自适应框架，通过路由机制为不同域的样本选择不同的适应方向，解决了混合分布偏移下单一适应路径的局限性，并提出 potpourri/potpourri+ 两个更真实的评估基准，在所有设定下取得 SOTA。

**[Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)**

:   提出 MASS (Mask-Aware Sleep Staging) 框架，通过多层级 masking 策略和层次化 prompt learning 机制，仅用 **10% 的原始 EEG 信号**即可实现可靠的睡眠分期，为资源受限的可穿戴睡眠监测系统提供方案。

**[Scaling and Transferability of Annealing Strategies in Large Language Model Training](scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)**

:   提出模型无关的预测框架，分解训练损失为前向效应项（学习率积分S）、退火动量项（Adam-style动量积分M）和模型尺寸项N，证明退火策略可从小模型/小batch迁移到大模型/大batch，预测误差MAPE<2%。
