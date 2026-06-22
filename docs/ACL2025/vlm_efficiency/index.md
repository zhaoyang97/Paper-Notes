---
title: >-
  ACL2025 VLMEfficiency论文汇总 · 8篇论文解读
description: >-
  8篇ACL2025的 VLM Efficiency 方向论文解读，涵盖多模态、模型压缩、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "模型压缩"
  - "LLM"
item_list:
  - u: "effivlm_bench_vlm_acceleration/"
    t: "EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models"
  - u: "hierarchical_safety_realignment_lightweight_restoration_of_safety_in_pruned_larg/"
    t: "Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models"
  - u: "hotelmatch_llm_retrieval/"
    t: "HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval"
  - u: "madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l/"
    t: "MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference"
  - u: "omgm_orchestrate_multiple_granularities_and_modalities_for_efficient_multimodal_/"
    t: "OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval"
  - u: "redundancylens_revealing_and_exploiting_visual_token_processing_redundancy_for_e/"
    t: "RedundancyLens: Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-Only MLLMs"
  - u: "sophia_efficient_long_video/"
    t: "Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding"
  - u: "token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl/"
    t: "Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**💬 ACL2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×7 · 模型压缩 ×2 · LLM ×2

**[EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](effivlm_bench_vlm_acceleration.md)**

:   提出 EffiVLM-Bench 统一评估框架，从性能、泛化性、忠实度和效率四个维度系统评估 LVLM 免训练加速方法（token 压缩 + 参数压缩），覆盖 3 个前沿模型和 17 个基准任务，揭示各方法在不同压缩率下的 Pareto 最优权衡。

**[Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models](hierarchical_safety_realignment_lightweight_restoration_of_safety_in_pruned_larg.md)**

:   提出层次化安全重对齐方法HSR，通过先识别安全关键注意力头、再在这些头中定位并恢复被剪枝的安全关键神经元，以极低参数开销（万分之几）显著恢复被剪枝LVLM丢失的安全性能。

**[HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval](hotelmatch_llm_retrieval.md)**

:   提出 HotelMatch-LLM，用 SLM 编码 query + LLM 编码酒店文档的非对称架构，配合三目标多任务优化（检索对齐 + MLM地理预测 + 视觉设施识别）和 patch 级 mean pooling 多图处理，在旅行领域多模态检索任务上显著超过 MARVEL/VISTA 等 SOTA。

**[MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)**

:   本文提出MadaKV，一种模态感知的KV缓存逐出策略，通过模态偏好自适应（MPA）和层级压缩补偿（HCC）两个组件，在保持多模态长上下文任务性能的同时，显著降低KV缓存内存占用（80-95%）和解码延迟（1.3-1.5倍加速）。

**[OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval](omgm_orchestrate_multiple_granularities_and_modalities_for_efficient_multimodal_.md)**

:   提出OMGM——一个面向知识密集型视觉问答(KB-VQA)的多模态RAG系统，通过粗到细三步检索策略协调查询与知识库在不同粒度和模态间的匹配，在InfoSeek和E-VQA上取得SOTA检索性能和极具竞争力的问答结果。

**[RedundancyLens: Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-Only MLLMs](redundancylens_revealing_and_exploiting_visual_token_processing_redundancy_for_e.md)**

:   提出 RedundancyLens 框架，系统揭示了 decoder-only MLLM 中视觉 token 在自注意力和 FFN 操作上存在大量结构化、聚簇式冗余，并利用这一发现实现免训练推理加速，与现有 token 压缩方法正交且可组合。

**[Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding](sophia_efficient_long_video.md)**

:   提出Sophia模型处理小时级长视频：通过Shot-adaptive Frame Pruning（基于镜头分割的两阶段帧剪枝）精准选择查询相关帧，结合O(N)复杂度的Hierarchical Attention替代全注意力，在8个长视频benchmark中6个SOTA，且注意力FLOPs仅为InternVL2的1/8.5。

**[Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)**

:   通过大规模基准实验揭示了当前MLLM视觉token剪枝方法的多个根本性问题：精心设计的剪枝策略（FastV、SparseVLM）在多数基准上甚至不如随机选择和池化等朴素方法，原因在于注意力评分的位置偏差、对语言信息的误用、重要性与冗余性的失衡以及评估指标的不可靠。
