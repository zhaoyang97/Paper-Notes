---
title: >-
  ICLR2026 VLMEfficiency论文汇总 · 5篇论文解读
description: >-
  5篇ICLR2026的 VLM Efficiency 方向论文解读，涵盖模型压缩、压缩/编码、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "模型压缩"
  - "压缩/编码"
  - "多模态"
item_list:
  - u: "hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p/"
    t: "HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit"
  - u: "index-preserving_lightweight_token_pruning_for_efficient_document_understanding_/"
    t: "Index-Preserving Lightweight Token Pruning for Efficient Document Understanding"
  - u: "ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr/"
    t: "IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning"
  - u: "mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_/"
    t: "Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models"
  - u: "ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_/"
    t: "PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🔬 ICLR2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (31)](../../CVPR2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [💬 ACL2026 (5)](../../ACL2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/vlm_efficiency/index.md)

🔥 **高频主题：** 模型压缩 ×3 · 压缩/编码 ×2 · 多模态 ×2

**[HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)**

:   提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。

**[Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)**

:   在 VLM 视觉编码器之前插入一个仅 203K 参数的二值 patch 分类器剔除文档背景 token，再用 $3 \times 3$ max-pooling 恢复碎片化文本区域并保留原始空间索引，在 Qwen2.5-VL 上实现 40-60% FLOPs 缩减且精度损失不超过 ~5%p。

**[IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)**

:   揭示了LVLM中RoPE位置编码隐式建立的视觉坐标系统（IVC tokens），提出一种训练免的、提示感知的视觉token剪枝策略，在保留IVC tokens和语义前景token的同时，削减约50%视觉token并维持≥99%原始性能。

**[Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)**

:   发现LVLM中KV Cache存在模态特异和注意力头特异的语义冗余，仅靠重要性选择会丢失语义覆盖，提出MixKV按头自适应混合重要性与多样性分数进行KV Cache压缩，在极端压缩下平均提升5.1%。

**[PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)**

:   提出PPE（Positional Preservation Embedding），利用RoPE各维度旋转独立性，将合并token内多个原始位置ID分块编码到不同维度段中，实现单个压缩token携带多个空间/时序位置信息。PPE是零参数、即插即用的通用算子，在55%压缩率下图像任务平均仅降3.6%、在90%压缩率下通过级联压缩仍保持可比性能。
