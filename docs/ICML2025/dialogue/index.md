---
title: >-
  ICML2025 对话系统论文汇总 · 2篇论文解读
description: >-
  2篇ICML2025的对话系统方向论文解读，涵盖 LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
item_list:
  - u: "investigating_non-transitivity_in_llm-as-a-judge/"
    t: "Investigating Non-Transitivity in LLM-as-a-Judge"
  - u: "position_uncertainty_quantification_needs_reassessment_for_large-language_model_/"
    t: "Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🧪 ICML2025** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (10)](../../ICLR2026/dialogue/index.md) · [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [🧪 ICML2026 (5)](../../ICML2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/dialogue/index.md) · [💬 ACL2025 (18)](../../ACL2025/dialogue/index.md)

**[Investigating Non-Transitivity in LLM-as-a-Judge](investigating_non-transitivity_in_llm-as-a-judge.md)**

:   揭示了 LLM-as-a-Judge 框架中评判偏好的**非传递性**问题（A>B, B>C 不能推出 A>C），证明固定基线模型的排名方式不可靠，提出基于循环赛 + Bradley-Terry 模型的排名方法及高效的 Swim 锦标赛策略。

**[Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents](position_uncertainty_quantification_needs_reassessment_for_large-language_model_.md)**

:   本文是一篇 Position Paper，通过梳理文献中 aleatoric 和 epistemic 不确定性的多种相互矛盾的定义，论证传统二分法在 LLM 交互场景中根本性失效，并提出 underspecification uncertainty（任务/上下文欠规范）、interactive learning（通过追问减少不确定性）和 output uncertainty（用自然语言而非标量表达不确定性）三个新研究方向。
