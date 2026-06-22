---
title: >-
  ICML2026 社会计算论文汇总 · 9篇论文解读
description: >-
  9篇ICML2026的社会计算方向论文解读，涵盖对齐/RLHF、多模态、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "多模态"
  - "LLM"
item_list:
  - u: "alignment_tampering_how_reinforcement_learning_from_human_feedback_is_exploited_/"
    t: "Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases"
  - u: "flips_instance-fingerprinting_for_llms_via_pseudo-random_sequences/"
    t: "FLIPS: Instance-Fingerprinting for LLMs via Pseudo-Random Sequences"
  - u: "ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti/"
    t: "IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection"
  - u: "mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod/"
    t: "MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News"
  - u: "objembed_towards_universal_multimodal_object_embeddings/"
    t: "ObjEmbed: Towards Universal Multimodal Object Embeddings"
  - u: "scope_selective_conformal_optimized_pairwise_llm_judging/"
    t: "SCOPE: Selective Conformal Optimized Pairwise LLM Judging"
  - u: "self-debias_self-correcting_for_debiasing_large_language_models/"
    t: "Self-Debias: Self-correcting for Debiasing Large Language Models"
  - u: "the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti/"
    t: "The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence"
  - u: "three_years_of_rchatgpt_societal_impact_evaluations_from_social_media_data/"
    t: "Three Years of r/ChatGPT: Societal Impact Evaluations from Social Media Data"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**🧪 ICML2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [🔬 ICLR2026 (17)](../../ICLR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/social_computing/index.md)

🔥 **高频主题：** 对齐/RLHF ×2 · 多模态 ×2 · LLM ×2

**[Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases](alignment_tampering_how_reinforcement_learning_from_human_feedback_is_exploited_.md)**

:   这篇论文提出 alignment tampering：当待对齐模型生成“高质量但带偏见”和“低质量但无偏见”的响应时，RLHF 的成对偏好标签会把质量与偏见混在一起，导致奖励模型、PPO/DPO 和 Best-of-N 采样进一步放大原本不想要的偏见。

**[FLIPS: Instance-Fingerprinting for LLMs via Pseudo-Random Sequences](flips_instance-fingerprinting_for_llms_via_pseudo-random_sequences.md)**

:   FLIPS 通过设计**伪随机种子序列**（仅模型所有者知晓种子）来生成模型独特"指纹响应"——攻击者即便微调或剪枝模型也无法消除指纹，黑盒查询场景下检测率 > 99%、误报率 < 1%。

**[IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)**

:   IDO 通过**显式建模模态间不一致性**作为可学习的分布优化目标——同时拉近真新闻的多模态嵌入并扩大假新闻的不一致，在 Weibo / Twitter / Fakeddit 上 F1 较 SOTA 提升 3-7%、对未见过的假新闻泛化能力显著提升。

**[MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News](mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod.md)**

:   MIND 通过**多视角理由生成 + 跨理由判别推理**为假新闻检测提供可解释 + 鲁棒的判别框架——同时利用 LLM 生成的事实核查、模态一致性、语义合理性 3 类理由，在 Weibo / Twitter / Fakeddit 上 F1 较 SOTA 提升 4-8%。

**[ObjEmbed: Towards Universal Multimodal Object Embeddings](objembed_towards_universal_multimodal_object_embeddings.md)**

:   ObjEmbed 训练一个**通用的对象嵌入模型**——通过结合检测、分割、检索、描述、分类等任务对齐多模态对象表示，在 OVD / OVS / Text2Image-Object / Open-Caption-Eval 等 11 项任务上单一嵌入超越或匹配任务特定 SOTA。

**[SCOPE: Selective Conformal Optimized Pairwise LLM Judging](scope_selective_conformal_optimized_pairwise_llm_judging.md)**

:   SCOPE 通过**双向偏好熵（BPE）**消除 LLM 评判中的位置偏差，结合**保形风险控制**实现有限样本 FDR 控制——在保持高覆盖率的前提下提供统计有效的风险界保证（覆盖率 0.583 时 FDR 仅 0.099 vs Vanilla 1.000 但 FDR 0.198）。

**[Self-Debias: Self-correcting for Debiasing Large Language Models](self-debias_self-correcting_for_debiasing_large_language_models.md)**

:   Self-Debias 把 LLM 的去偏问题重塑为「在自回归推理链上对概率质量做公平资源分配」：用轨迹级后缀边际作为资源单位，套 Jain 公平指数防止资源在易样本上塌缩，再配 cold-start SFT 与基于一致性过滤的在线自训练，仅用 20k 标注种子就让 Qwen3-8B 在 8 个 fairness/utility 基准上的平均分从 77.5 拉到 81.7，并把基础模型「自我纠错越纠越歪」的塌缩翻转成稳定 +0.4。

**[The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)**

:   本文用测度论框架把 InfoNCE 损失提升到表示分布上的确定性"种群能量"，证明 unimodal 情形是凸的且收敛到唯一 Gibbs 平衡，而对称多模态情形会出现持续的负对称 KL 耦合，从几何上必然产生 modality gap。

**[Three Years of r/ChatGPT: Societal Impact Evaluations from Social Media Data](three_years_of_rchatgpt_societal_impact_evaluations_from_social_media_data.md)**

:   把 r/ChatGPT 子版三年（2022-12 至 2025-11）共 13.7 万帖子用稀疏自编码器（SAE）拆成可解释特征，再用分段线性变点拟合追踪每个特征的时间轨迹，发现"情感性使用"（心理治疗、情感依恋）在 GPT-4o 发布后骤增；并提出在线监测算法 PuLSE，证明它本可在 2024 年 10 月就报警——比 OpenAI 公开承认这一影响早了半年。
