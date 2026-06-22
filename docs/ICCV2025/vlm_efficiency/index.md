---
title: >-
  ICCV2025 VLMEfficiency论文汇总 · 11篇论文解读
description: >-
  11篇ICCV2025的 VLM Efficiency 方向论文解读，涵盖多模态、压缩/编码、模型压缩、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICCV2025"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "压缩/编码"
  - "模型压缩"
  - "LLM"
item_list:
  - u: "aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar/"
    t: "AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference"
  - u: "aircache_activating_inter_modal_relevancy_kv_cache_compression_for_efficient_large_vision_language_model/"
    t: "AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference"
  - u: "dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm/"
    t: "Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM"
  - u: "feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a/"
    t: "Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration"
  - u: "folder_accelerating_multi-modal_large_language_models_with_enhanced_performance/"
    t: "FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance"
  - u: "growing_a_twig_to_accelerate_large_vision-language_models/"
    t: "Growing a Twig to Accelerate Large Vision-Language Models"
  - u: "llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models/"
    t: "LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models"
  - u: "matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling/"
    t: "MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling"
  - u: "meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m/"
    t: "METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models"
  - u: "shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i/"
    t: "ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers"
  - u: "sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference/"
    t: "SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**📹 ICCV2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×10 · 压缩/编码 ×3 · 模型压缩 ×2 · LLM ×2

**[AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)**

:   提出 AirCache，一种面向 LVLM 的 KV Cache 压缩方法，通过精英观察窗口（Elite Observation Window）评估视觉 token 重要性，结合基于重要性分数分布强度与偏度的自适应层级预算分配，在仅保留 10% 视觉 KV Cache 时性能损失不超过 1%，解码延迟降低 29%-66%。

**[AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](aircache_activating_inter_modal_relevancy_kv_cache_compression_for_efficient_large_vision_language_model.md)**

:   提出AirCache，通过精英观测窗口（利用文本自注意力筛选关键文本token评估视觉token重要性）和自适应层间预算分配（基于重要性分数分布的强度和偏度），实现仅保留10%视觉KV缓存即可保持模型性能，解码延迟降低29%-66%。

**[Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM](dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm.md)**

:   提出 Dynamic-VLM，通过动态视觉Token压缩器根据视频长度灵活调整每帧Token数量，配合200万级高质量合成视频QA数据集，在 VideoMME 上比 LLaVA-OneVision 提升 2.7%，在 MuirBench 上提升 10.7%。

**[Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)**

:   揭示了 VLM 中早期视觉 token 剪枝存在系统性位置偏差（RoPE 导致倾向保留图像底部 token），并提出 FEATHER 方法通过去除 RoPE + 均匀采样 + 多阶段剪枝解决该问题，在定位任务上实现 5× 以上性能提升。

**[FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)**

:   提出 FOLDER——一种即插即用的视觉 token 压缩模块，通过系统分析信息损失的三个关键因素（压缩影响、传播效应、聚合方式），在视觉编码器的最后几层进行激进的 token 合并，实现最多 70% 的 token 削减，同时保持甚至提升模型性能。

**[Growing a Twig to Accelerate Large Vision-Language Models](growing_a_twig_to_accelerate_large_vision-language_models.md)**

:   提出 TwigVLM，通过在 VLM 早期层上"生长"一个轻量级 twig 模块，同时实现 twig 引导的视觉 token 剪枝（TTP，prefilling 加速）和自推测解码（SSD，decoding 加速），在 LLaVA-1.5-7B 上剪枝 88.9% 视觉 token 后保留 96% 精度，长回答生成速度提升 154%，在精度和速度上均大幅超越现有方法。

**[LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)**

:   利用视觉编码器中CLS token与空间token之间注意力分数的稀疏性，自适应地剪枝和合并视觉token，在仅保留5.5%视觉token的情况下维持LMM的可比性能。

**[MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)**

:   提出MaTVLM，将预训练VLM中部分Transformer层替换为Mamba-2层并通过单阶段知识蒸馏训练，在保持竞争性性能的同时实现3.6倍推理加速和27.5%显存降低。

**[METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)**

:   METEOR 提出首个面向多编码器 MLLM 的三阶段渐进式 token 剪枝框架：在编码阶段用特征秩分配各编码器的稀疏比例，在融合阶段通过协同剪枝消除跨编码器冗余，在解码阶段根据文本提示自适应调整剪枝比例，将视觉 token 减少 76% 而性能仅降 0.3%。

**[ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i.md)**

:   发现 MLLM 中存在显著的**层级冗余**——多数层对视觉 token 的变换贡献极小，据此提出 ShortV：在约 60% 的层中冻结视觉 token（跳过其注意力和 FFN 计算），在 LLaVA-NeXT-13B 上实现 50% FLOPs 减少，性能几乎无损。方法免训练，且与 token 剪枝方法正交可叠加。

**[SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)**

:   提出SparseVILA——首个解耦prefill和decode阶段视觉稀疏性的VLM推理加速框架：prefill阶段进行query-agnostic的冗余token剪枝，decode阶段进行query-aware的相关token检索，实现最高4.0×prefill加速、2.5×decode吞吐提升、2.6×端到端加速，同时在多轮对话场景中保持精度（现有方法因永久删除token而在多轮中急剧退化）。
