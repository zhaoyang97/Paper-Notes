---
title: >-
  AAAI2026 VLMEfficiency论文汇总 · 5篇论文解读
description: >-
  5篇AAAI2026的 VLM Efficiency 方向论文解读，涵盖多模态、压缩/编码、LLM、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "压缩/编码"
  - "LLM"
  - "对齐/RLHF"
item_list:
  - u: "em-kd_distilling_efficient_multimodal_large_language_model_w/"
    t: "EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens"
  - u: "filter_correlate_compress_training-free_token_reduction_for_/"
    t: "Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration"
  - u: "global_compression_commander_plug-and-play_inference_acceler/"
    t: "Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models"
  - u: "rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment/"
    t: "Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment"
  - u: "tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_/"
    t: "TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🤖 AAAI2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×3 · 压缩/编码 ×2

**[EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](em-kd_distilling_efficient_multimodal_large_language_model_w.md)**

:   提出EM-KD框架，通过Hungarian算法解决teacher-student间视觉token数量不平衡问题，结合视觉语义蒸馏(VSD)和视觉-语言亲和力蒸馏(VLAD)将vanilla teacher的知识迁移到高效student MLLM，在11个benchmark上以144 token/patch达到50.4均分，超越576 token的LLaVA-NeXT(49.4)同时推理速度提升近2倍。

**[Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](filter_correlate_compress_training-free_token_reduction_for_.md)**

:   提出FiCoCo三阶段框架（Filter-Correlate-Compress），通过集成视觉感知+语义感知冗余度量筛选丢弃token，利用token间相关性自适应回收信息，实现training-free的MLLM加速。在LLaVA-NeXT上达14.7×FLOPs压缩同时保留93.6%性能，在5种MLLM架构上全面超越FastV、SparseVLM等SOTA。

**[Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](global_compression_commander_plug-and-play_inference_acceler.md)**

:   提出GlobalCom²，一个**即插即用、无需训练**的token压缩框架，专为动态裁剪（dynamic cropping）结构的高分辨率VLM设计：利用全局缩略图（thumbnail）作为"指挥官"引导局部裁剪区域（crop）的差异化压缩，在压缩90%视觉token的同时保持>90%原始性能。

**[Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)**

:   揭示了 LVLM 中文本引导视觉token重要性评估的三种跨模态失配问题（因果、语义、空间），提出 VisionDrop——一个仅依赖视觉自注意力的免训练渐进式token剪枝框架，跨视觉编码器和 LLM 解码器多阶段压缩，在保留 5.6% token 时仍能维持 91%+ 原始性能。

**[TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)**

:   TinyChemVL 是一个仅4B参数的化学领域VLM，通过自适应token合并与剪枝策略将视觉token压缩至原来的1/16，并引入反应级别任务和基准ChemRxn-V，在分子和反应级别的视觉化学任务上达到SOTA性能，同时显著提升推理和训练速度。
