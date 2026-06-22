---
title: >-
  CVPR2025 LLM效率论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2025的 LLM 效率方向论文解读，涵盖压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "压缩/编码"
item_list:
  - u: "associative_transformer/"
    t: "Associative Transformer"
  - u: "efficient_data_driven_mixture-of-expert_extraction_from_trained_networks/"
    t: "Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks"
  - u: "locore_image_re-ranking_with_long-context_sequence_modeling/"
    t: "LOCORE: Image Re-ranking with Long-Context Sequence Modeling"
  - u: "moee_mixture_expert_extraction/"
    t: "Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks"
  - u: "spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training/"
    t: "Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**📷 CVPR2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (169)](../../ICLR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🧪 ICML2026 (48)](../../ICML2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md)

**[Associative Transformer](associative_transformer.md)**

:   提出 Associative Transformer (AiT)，通过在 Transformer 中引入可学习的显式记忆模块和 Hopfield 网络进行 token 重建，以更少的参数实现优于 ViT 的分类和关系推理性能。

**[Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks](efficient_data_driven_mixture-of-expert_extraction_from_trained_networks.md)**

:   提出一种从预训练 ViT 中自动提取 MoE（Mixture-of-Experts）变体的方法：先聚类 MLP 层的输出激活模式，再据此抽取对应的子网络作为专家，无需从头训练 MoE，在 ImageNet-1k 上仅需少量微调即可恢复 98% 原始性能，同时将 FLOPs 和模型大小分别减少 36% 和 32%。

**[LOCORE: Image Re-ranking with Long-Context Sequence Modeling](locore_image_re-ranking_with_long-context_sequence_modeling.md)**

:   提出 LoCoRe（Long-Context Re-ranker），首次实现基于局部描述子的列表级（list-wise）图像重排序，利用 Longformer 长上下文序列模型同时处理查询图像和整个候选列表的局部描述子，通过捕获候选图像间的传递关系显著提升重排序性能。

**[Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks](moee_mixture_expert_extraction.md)**

:   提出一种从预训练 ViT 中提取 MoE 变体的后训练方法，通过 HDBSCAN 聚类 MLP 隐层激活模式自动发现专家结构，无需重新训练即可在 ImageNet-1k 上减少 36% MACs 和 32% 参数的同时保留 98% 原始精度。

**[Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)**

:   本文提出 Spatial-TTT，通过测试时训练（TTT）机制将模型的部分参数（快速权重）作为紧凑非线性记忆，配合混合架构和空间预测机制，从无界视频流中持续积累和组织3D空间证据，在视频空间理解基准上达到 SOTA。
