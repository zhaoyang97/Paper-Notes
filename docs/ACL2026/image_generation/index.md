---
title: >-
  ACL2026 图像生成论文汇总 · 5篇论文解读
description: >-
  5篇ACL2026的图像生成方向论文解读，涵盖 LLM、多模态、文生图、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "图像生成"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
  - "文生图"
  - "扩散模型"
item_list:
  - u: "anchor_llm-driven_subject_conditioning_for_text-to-image_synthesis/"
    t: "ANCHOR: LLM-driven Subject Conditioning for Text-to-Image Synthesis"
  - u: "from_ar_to_diffusion_efficiently_adapting_large_language_models_with_strictly_ca/"
    t: "From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons"
  - u: "mentor_efficient_multimodal-conditioned_tuning_for_autoregressive_vision_generat/"
    t: "MENTOR: Efficient Autoregressive Image Generation with Balanced Multimodal Control"
  - u: "multimodal_large_language_models_for_multi-subject_in-context_image_generation/"
    t: "Multimodal Large Language Models for Multi-Subject In-Context Image Generation"
  - u: "think_bright_diffuse_nice_enhancing_t2i-icl_via_inductive-bias_hint_instruction_/"
    t: "Think Bright, Diffuse Nice: Enhancing T2I-ICL via Inductive-Bias Hint Instruction and Query Contrastive Decoding"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**💬 ACL2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (490)](../../CVPR2026/image_generation/index.md) · [🔬 ICLR2026 (352)](../../ICLR2026/image_generation/index.md) · [🧪 ICML2026 (141)](../../ICML2026/image_generation/index.md) · [🤖 AAAI2026 (79)](../../AAAI2026/image_generation/index.md) · [🧠 NeurIPS2025 (221)](../../NeurIPS2025/image_generation/index.md) · [📹 ICCV2025 (213)](../../ICCV2025/image_generation/index.md)

🔥 **高频主题：** LLM ×3 · 多模态 ×2

**[ANCHOR: LLM-driven Subject Conditioning for Text-to-Image Synthesis](anchor_llm-driven_subject_conditioning_for_text-to-image_synthesis.md)**

:   这篇论文提出 ANCHOR 数据集，用 70K+ 来自 5 家新闻媒体的抽象 caption 暴露 T2I 模型在多主体、上下文推理和细粒度 grounding 上的失败，并提出 SAFE 用 LLM 抽取关键主体、在 embedding 层强化主体表示来提升图文一致性。

**[From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons](from_ar_to_diffusion_efficiently_adapting_large_language_models_with_strictly_ca.md)**

:   本文提出 FLUID，用严格因果注意力和熵感知 Elastic Horizon 把预训练自回归 LLM 高效适配为扩散式并行生成模型，在只用 2.7B 适配 tokens 的情况下取得接近强 AR 模型、优于现有扩散基线的推理和代码生成表现。

**[MENTOR: Efficient Autoregressive Image Generation with Balanced Multimodal Control](mentor_efficient_multimodal-conditioned_tuning_for_autoregressive_vision_generat.md)**

:   MENTOR 用统一自回归 decoder 和两阶段多模态训练，把参考图像与文本指令对齐到同一生成前缀中，在仅 3M 训练数据和 8 张 A100 约 1.5 天训练预算下，取得了较好的概念保持与 prompt following 平衡。

**[Multimodal Large Language Models for Multi-Subject In-Context Image Generation](multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)**

:   这篇论文提出 MUSIC，把多模态大语言模型的视觉推理能力引入多主体 in-context 图像生成，通过自动合成训练数据、视觉 CoT 和语义驱动空间布局规划，显著缓解多个参考主体同时生成时的主体遗漏、身份混淆和语义漂移问题。

**[Think Bright, Diffuse Nice: Enhancing T2I-ICL via Inductive-Bias Hint Instruction and Query Contrastive Decoding](think_bright_diffuse_nice_enhancing_t2i-icl_via_inductive-bias_hint_instruction_.md)**

:   这篇论文提出训练无关的 TBDN 框架，用 Hint Instruction 让 LVLM 更关注最终 query，用 Query Contrastive Decoding 抑制先验幻觉，再把更准确的文本描述交给扩散模型，在 CoBSAT 和 T2I Fast Mini-ImageNet 上显著提升文本到图像上下文学习性能。
