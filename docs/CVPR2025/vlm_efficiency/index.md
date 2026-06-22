---
title: >-
  CVPR2025 VLMEfficiency论文汇总 · 3篇论文解读
description: >-
  3篇CVPR2025的 VLM Efficiency 方向论文解读，涵盖模型压缩、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "VLM Efficiency"
  - "论文解读"
  - "论文笔记"
  - "模型压缩"
  - "多模态"
item_list:
  - u: "coap_memory-efficient_training_with_correlation-aware_gradient_projection/"
    t: "COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection"
  - u: "mbq_modality-balanced_quantization_for_large_vision-language_models/"
    t: "MBQ: Modality-Balanced Quantization for Large Vision-Language Models"
  - u: "quantization_without_tears/"
    t: "Quantization without Tears"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ VLM Efficiency

**📷 CVPR2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/vlm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/vlm_efficiency/index.md) · [💬 ACL2026 (6)](../../ACL2026/vlm_efficiency/index.md) · [🧪 ICML2026 (4)](../../ICML2026/vlm_efficiency/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/vlm_efficiency/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/vlm_efficiency/index.md)

🔥 **高频主题：** 模型压缩 ×2

**[COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection](coap_memory-efficient_training_with_correlation-aware_gradient_projection.md)**

**[MBQ: Modality-Balanced Quantization for Large Vision-Language Models](mbq_modality-balanced_quantization_for_large_vision-language_models.md)**

:   发现大型VLM中视觉token和语言token对量化误差的敏感度差异超过10倍，提出MBQ方法在量化校准过程中引入基于梯度的模态平衡因子，在W3A16和W4A8设置下分别提升精度最高4.4%和11.6%，并实现1.4倍端到端加速。

**[Quantization without Tears](quantization_without_tears.md)**

:   提出 QwT（Quantization without Tears）方法，通过在量化网络的每个 block 后添加一个轻量级线性补偿层来弥补量化信息损失，该补偿层参数可通过闭式解在2分钟内求得，在视觉、语言、多模态等多种任务上均显著提升了 PTQ 精度。
