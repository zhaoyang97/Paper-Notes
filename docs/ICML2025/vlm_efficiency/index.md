---
title: >-
  ICML2025 VLMEfficiency论文汇总 · 3篇论文解读
description: >-
  3篇ICML2025的 VLM Efficiency 方向论文解读，涵盖多模态、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "模型压缩"
item_list:
  - u: "corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun/"
    t: "CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models"
  - u: "mminference_accelerating_pre-filling_for_long-context_vlms_via_modality-aware_pe/"
    t: "MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention"
  - u: "sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere/"
    t: "SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**🧪 ICML2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md)

🔥 **高频主题：** 多模态 ×2

**[CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)**

:   首次揭示 VLM 中 token 稀疏与神经元稀疏之间的内在关联——核心神经元与核心 token 相互决定、相互强化，并据此提出 CoreMatching 协同稀疏推理框架，在 pre-filling 和 decoding 两阶段同时实现加速，达到 5× FLOPs 降低和 10× 整体加速。

**[MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](mminference_accelerating_pre-filling_for_long-context_vlms_via_modality-aware_pe.md)**

:   本文提出 MMInference，通过“模态感知的置换稀疏注意力 + 头级离线模式搜索 + 在线动态索引 + 定制 GPU Kernel”，在不改模型不微调的前提下，将长上下文 VLM 的 prefill 阶段在 1M token 场景最高加速到 8.3x，同时尽量保持任务精度。

**[SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)**

:   SparseVLM 提出了首个文本引导的免训练视觉 token 稀疏化框架，通过选择与视觉相关的文本 token 作为"评分者"来评估视觉 token 的重要性，结合自适应剪枝比率和 token 回收机制，在 LLaVA 上仅保留 192 个 token（减少 66.7%）时维持 99.1% 的原始性能。
