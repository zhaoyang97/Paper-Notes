---
title: >-
  ICLR2026 VLMEfficiency论文汇总 · 8篇论文解读
description: >-
  8篇ICLR2026的 VLM Efficiency 方向论文解读，涵盖多模态、模型压缩、LLM、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "模型压缩"
  - "LLM"
  - "压缩/编码"
item_list:
  - u: "bolt_decisionaligned_distillation_and_budget-aware_routing_for_constrained_multi/"
    t: "BOLT: Decision‑Aligned Distillation and Budget-Aware Routing for Constrained Multimodal QA on Robots"
  - u: "enhancing_visual_token_representations_for_video_large_language_models_via_train/"
    t: "Enhancing Visual Token Representations for Video Large Language Models via Training-free Spatial-Temporal Pooling and Gridding"
  - u: "fastflow_accelerating_the_generative_flow_matching_models_with_bandit_inference/"
    t: "FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference"
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
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🔬 ICLR2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×3 · 模型压缩 ×3 · LLM ×2 · 压缩/编码 ×2

**[BOLT: Decision‑Aligned Distillation and Budget-Aware Routing for Constrained Multimodal QA on Robots](bolt_decisionaligned_distillation_and_budget-aware_routing_for_constrained_multi.md)**

:   BOLT 把"机器人上的受限多选题问答"拆成训练期的**选项级决策蒸馏**（让 2B 小模型直接对齐 13B 教师在选项集上的偏好）和推理期的**预算感知路由**（只在便宜信号预示有正收益时才触发高分辨复评/同类检索/问题分解），用 2B 学生在 Robo2VLM-1 上做到 50.50% 准确率、反超 36.74% 的 13B 教师，同时把显存从 26.9GB 压到 3.8GB、能耗降 82.5%。

**[Enhancing Visual Token Representations for Video Large Language Models via Training-free Spatial-Temporal Pooling and Gridding](enhancing_visual_token_representations_for_video_large_language_models_via_train.md)**

:   针对视频大语言模型把成千上万视觉 token 压缩进有限上下文时丢失时空信息的问题，提出训练无关的 ST-GridPool：用「金字塔时序网格化」在不同时间尺度上聚合帧 token 注入多粒度运动信息，再用「基于范数的空间池化」依据 token 的 L2 范数加权保留高信息量区域，在 LLaVA-Video / LLaVA-OneVision 上即插即用、不需重训就稳定涨点。

**[FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference](fastflow_accelerating_the_generative_flow_matching_models_with_bandit_inference.md)**

:   FastFlow 是一个免训练、即插即用的流匹配（flow matching）推理加速框架：它用有限差分外推零成本地近似掉那些"几乎走直线"的冗余去噪步，并用一个多臂老虎机在线决定每次能安全跳几步，在图像/视频生成与编辑任务上拿到 2.6× 以上加速且基本不掉质量。

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
