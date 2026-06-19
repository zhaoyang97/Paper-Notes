---
title: >-
  ICLR2026 文本生成论文汇总 · 3篇论文解读
description: >-
  3篇ICLR2026的文本生成方向论文解读，涵盖扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "文本生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
item_list:
  - u: "antislop_a_comprehensive_framework_for_identifying_and_eliminating_repetitive_pa/"
    t: "Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models"
  - u: "fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m/"
    t: "FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model"
  - u: "rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure/"
    t: "Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**🔬 ICLR2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (2)](../../ICML2026/nlp_generation/index.md) · [💬 ACL2026 (17)](../../ACL2026/nlp_generation/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/nlp_generation/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_generation/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_generation/index.md) · [💬 ACL2025 (27)](../../ACL2025/nlp_generation/index.md)

**[Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models](antislop_a_comprehensive_framework_for_identifying_and_eliminating_repetitive_pa.md)**

:   Antislop 把"LLM 生成里那些一眼能认出是 AI 的重复套话（slop）"当成可量化、可定位、可消除的对象：先用频率比统计画出模型专属的"slop 指纹"，再用一个推理期的回溯采样器精准压制这些模式，最后把采样器的拦截记录自动转成偏好数据，用新提出的 FTPO 微调把抑制能力永久焊进权重——在 GSM8K/MMLU/创意写作上几乎不掉点的前提下做到 90% 的 slop 削减。

**[FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)**

:   提出 FS-DFM（Few-Step Discrete Flow-Matching），通过步数感知训练和累积标量更新规则，将离散 flow-matching 语言模型的采样步数从 1024 步降低到 8 步，实现 128 倍加速，同时保持相当的困惑度和生成质量。

**[Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)**

:   从 proper scoring rules 框架出发，证明最高概率输出序列的负对数似然（MSP）是理论上合理的不确定性度量，并提出 G-NLL——仅用一次贪心解码就能逼近该度量，在多个场景下匹配或超越需要多次采样的 SOTA 方法。
