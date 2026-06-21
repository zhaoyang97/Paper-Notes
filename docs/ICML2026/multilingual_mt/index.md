---
title: >-
  ICML2026 多语言/翻译论文汇总 · 3篇论文解读
description: >-
  3篇ICML2026的多语言/翻译方向论文解读，涵盖扩散模型、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "多语言/翻译"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "对抗鲁棒"
item_list:
  - u: "edit-based_refinement_for_parallel_masked_diffusion_language_models/"
    t: "Edit-Based Refinement for Parallel Masked Diffusion Language Models"
  - u: "optimizing_language_models_for_crosslingual_knowledge_consistency/"
    t: "Optimizing Language Models for Crosslingual Knowledge Consistency"
  - u: "toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages/"
    t: "Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**🧪 ICML2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (8)](../../ICLR2026/multilingual_mt/index.md) · [💬 ACL2026 (63)](../../ACL2026/multilingual_mt/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/multilingual_mt/index.md) · [🧠 NeurIPS2025 (11)](../../NeurIPS2025/multilingual_mt/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/multilingual_mt/index.md) · [🧪 ICML2025 (1)](../../ICML2025/multilingual_mt/index.md)

**[Edit-Based Refinement for Parallel Masked Diffusion Language Models](edit-based_refinement_for_parallel_masked_diffusion_language_models.md)**

:   ME-DLM 给 masked diffusion 语言模型（如 LLaDA）加一个"解码完再编辑修补"的轻量阶段：第一阶段照常 unmask 出粗稿，第二阶段用替换/删除/插入三种 token 级编辑做并行修正，监督信号来自 edit distance 的最短编辑脚本，在只用 1/8 扩散步数的情况下 HumanEval +11.6 / GSM8K +33.6 点反超 LLaDA-Instruct。

**[Optimizing Language Models for Crosslingual Knowledge Consistency](optimizing_language_models_for_crosslingual_knowledge_consistency.md)**

:   本文针对多语言 LLM 在不同语言间回答同一问题却给出冲突答案的问题，设计了一个**用"另一种语言下回答的对数似然"作为 reward 的 RL 目标**，证明其最优策略呈 product-of-experts 形式并在 $\gamma_1\gamma_2=\beta^2$ 时保证跨语言偏好一致；据此推导出无需 reward model、无需 online 采样的 **DCO（Direct Consistency Optimization）** 算法，在 9 个 LLM、3 个多语言 QA 基准、26 种语言上同时提升跨语言一致性（RankC）与回答准确率。

**[Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)**

:   LiRA 在冻结的多语言编码器与英文 LLM 之间插一层 "锚定 + 一致性正则" 的轻量微调模块，把低资源语言的句向量按 $\epsilon_1$（锚定误差）与 $\epsilon_2$（翻译 KL 距离）这两个理论可控的量约束到共享英文语义空间，从而在检索、排序与推理三类任务上同时拿到稳定提升。
