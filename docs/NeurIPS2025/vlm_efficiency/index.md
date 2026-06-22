---
title: >-
  NeurIPS2025 VLMEfficiency论文汇总 · 8篇论文解读
description: >-
  8篇NeurIPS2025的 VLM Efficiency 方向论文解读，涵盖多模态、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "模型压缩"
item_list:
  - u: "balanced_token_pruning_accelerating_vision_language_models_b/"
    t: "Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization"
  - u: "beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili/"
    t: "Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability"
  - u: "elasticmm_efficient_multimodal_llms_serving_with_elastic_multimodal_parallelism/"
    t: "ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism"
  - u: "flowcut_rethinking_redundancy_via_information_flow_for_effic/"
    t: "FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models"
  - u: "hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode/"
    t: "HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM"
  - u: "prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction/"
    t: "PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation"
  - u: "scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms/"
    t: "SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs"
  - u: "vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin/"
    t: "ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🧠 NeurIPS2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×6 · 模型压缩 ×2

**[Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)**

:   提出 Balanced Token Pruning (BTP)，通过联合考虑剪枝对当前层（局部）和后续层（全局）的影响，在浅层侧重多样性保留以维护下游表示质量、在深层侧重注意力选择以保持局部输出一致性，在 LLaVA/Qwen2.5-VL 等多个 LVLM 上仅保留 22% 视觉 token 即保持原模型 98% 性能。

**[Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)**

:   UAT（Unsupervised Adaptive Thresholding）为早退 DNN 设计了可靠性函数来评估中间层输出质量，并用多臂赌博机（MAB）算法在推理时动态学习最优退出阈值，实现 1.7-2.1× 加速且性能损失 <2%，同时对分布偏移鲁棒。

**[ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism](elasticmm_efficient_multimodal_llms_serving_with_elastic_multimodal_parallelism.md)**

:   提出弹性多模态并行（EMP）范式和 ElasticMM 系统，通过模态感知负载均衡和弹性分区调度将多模态推理的不同阶段解耦到独立实例，相比 vLLM TTFT 降低最高 4.2 倍、吞吐量提升 3.2-4.5 倍。

**[FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](flowcut_rethinking_redundancy_via_information_flow_for_effic.md)**

:   从信息流（Information Flow）视角重新理解VLM中视觉token冗余性的涌现机制，提出FlowCut框架通过层自适应剪枝比例、多标准融合评分和累积重要性跟踪实现与模型内在信息传播行为对齐的token剪枝，在LLaVA-1.5-7B上以88.9% token减少率超越SOTA 1.6%，LLaVA-NeXT-7B上以94.4%减少率超越4.3%。

**[HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM](hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)**

:   提出 Hawaii 框架，通过混合 LoRA 适配器（MoLA）和分层知识蒸馏（HKD），将多个视觉专家的知识蒸馏到单个视觉编码器中，在不增加推理成本的前提下显著提升 VLM 的视觉理解能力。

**[PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)**

:   PrefixKV 发现不同层 KV 缓存的重要性分布差异显著，将逐层缓存大小确定问题形式化为全局前缀配置搜索，通过二分搜索找到最优信息保留阈值使每层保持最大上下文信息，在 20% 压缩率下仅有 0.49 PPL 下降且提供 1.8× 推理加速。

**[SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)**

:   提出 SCOPE，一种联合建模显著性和覆盖率的视觉 Token 剪枝策略，通过迭代选择 SCOPE 得分最高的 Token 来保持语义完整性，在 9 倍 Token 缩减下保留 LLaVA-1.5 96% 的性能。

**[ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)**

:   针对VLM推测解码（speculative decoding）中草稿模型难以处理冗余视觉token的问题，提出ViSpec框架，通过视觉适配器压缩图像token+全局视觉特征注入+合成训练数据，首次在VLM推测解码中实现了显著加速（最高3.22×）。
