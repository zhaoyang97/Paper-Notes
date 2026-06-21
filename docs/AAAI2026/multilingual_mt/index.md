---
title: >-
  AAAI2026 多语言/翻译论文汇总 · 9篇论文解读
description: >-
  9篇AAAI2026的多语言/翻译方向论文解读，涵盖对齐/RLHF、LLM、推理、翻译、语音等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "多语言/翻译"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "LLM"
  - "推理"
  - "翻译"
  - "语音"
item_list:
  - u: "bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for/"
    t: "Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages"
  - u: "focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil/"
    t: "Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models"
  - u: "gloctm_cross-lingual_topic_modeling_via_a_global_context_space/"
    t: "GloCTM: Cross-Lingual Topic Modeling via a Global Context Space"
  - u: "how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per/"
    t: "How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective"
  - u: "midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu/"
    t: "MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis"
  - u: "mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_/"
    t: "Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering"
  - u: "nadir_differential_attention_flow_for_non-autoregressive_transliteration_in_indi/"
    t: "NADIR: Differential Attention Flow for Non-Autoregressive Transliteration in Indic Languages"
  - u: "vidia2std_a_parallel_corpus_and_methods_for_low-resource_vietnamese_dialect-to-s/"
    t: "ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation"
  - u: "x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no/"
    t: "X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**🤖 AAAI2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (8)](../../ICLR2026/multilingual_mt/index.md) · [💬 ACL2026 (63)](../../ACL2026/multilingual_mt/index.md) · [🧪 ICML2026 (3)](../../ICML2026/multilingual_mt/index.md) · [🧠 NeurIPS2025 (11)](../../NeurIPS2025/multilingual_mt/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/multilingual_mt/index.md) · [🧪 ICML2025 (1)](../../ICML2025/multilingual_mt/index.md)

🔥 **高频主题：** 对齐/RLHF ×2 · LLM ×2

**[Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)**

:   本文综合多项实证研究，揭示LLM安全机制在低资源语言和代码混合场景下的严重失效，并提出基于参数高效安全引导、文化驱动偏好数据和社区参与式对齐的资源感知蓝图。

**[Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)**

:   本文提出LAHIS方法，仅需一次前向-后向传播即可高效识别多语言LLM中的语言特异性和语言通用性注意力头，并展示了通过调控这些头来实现跨语言注意力转移、缓解非目标语言生成问题，以及仅用14-20个可训练参数就能提升多语言QA性能的能力。

**[GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)**

:   提出GloCTM，通过双路径VAE架构（局部语言路径+全局上下文路径）结合Polyglot Augmentation（跨语言近邻词扩充输入）、KL散度内部对齐、统一解码器结构对齐和CKA语义对齐四重机制，在3个跨语言数据集上全面超越现有方法的主题质量和跨语言对齐度。

**[How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)**

:   提出三元神经元分类（语言特定/语言相关/通用），将 LLM 多语言推理分为四阶段分析，发现多语言对齐通过增加语言相关神经元（减少语言特定神经元）来提升性能，且在未训练语言上也产生"自发多语言对齐"效应。

**[MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)**

:   提出 MIDB（多语言指令数据增强器），通过 36.8k 人类语言专家标注的修订样本训练一个统一模型，自动修复多语言合成指令数据中的内容错误、机器翻译缺陷和本地化不足问题，显著提升 16 种语言的指令数据质量和下游 LLM 的多语言/文化理解能力。

**[Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)**

:   通过激活转向（activation steering）技术缓解 LLM 中的内容效应偏见——模型将内容可信度与形式逻辑有效性混淆的问题，提出 K-CAST（基于 kNN 的条件激活转向）方法，在不响应静态转向的模型上实现高达 15% 的形式推理准确率提升。

**[NADIR: Differential Attention Flow for Non-Autoregressive Transliteration in Indic Languages](nadir_differential_attention_flow_for_non-autoregressive_transliteration_in_indi.md)**

:   提出 NADIR，一种结合差分 Transformer 和混合专家（MoE）的非自回归（NAR）多语言音译架构，在印度语言音译任务上实现了 13× 以上的推理加速，同时将 NAR 模型的幻觉错误（重复、替换、遗漏、插入）大幅降低，缩小了与自回归模型之间的精度差距。

**[ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation](vidia2std_a_parallel_corpus_and_methods_for_low-resource_vietnamese_dialect-to-s.md)**

:   ViDia2Std 构建了首个覆盖越南全部 63 个省份的手工标注越南语方言-标准语平行语料库（13,000+ 句对），并评估了多种 seq2seq 模型在方言归一化任务上的表现，证明方言归一化作为预处理步骤能显著提升机器翻译和情感分析等下游任务的性能。

**[X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework](x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)**

:   本文提出X-MuTeST框架，结合LLM语义推理和n-gram attention增强的两阶段训练方法，用于可解释的多语言仇恨言论检测，并提供了印地语和泰卢固语的首个token级人工标注理据基准数据集。
