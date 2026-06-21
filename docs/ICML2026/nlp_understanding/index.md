---
title: >-
  ICML2026 NLP理解论文汇总 · 2篇论文解读
description: >-
  2篇ICML2026的 NLP 理解方向论文解读，收录 Causal Fine-Tuning under Laten、Controlling the Risk of Corrup等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "NLP 理解"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "causal_fine-tuning_under_latent_confounded_shift/"
    t: "Causal Fine-Tuning under Latent Confounded Shift"
  - u: "controlling_the_risk_of_corrupted_contexts_for_language_models_via_early-exiting/"
    t: "Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP 理解

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (2)](../../ICLR2026/nlp_understanding/index.md) · [💬 ACL2026 (34)](../../ACL2026/nlp_understanding/index.md) · [🤖 AAAI2026 (1)](../../AAAI2026/nlp_understanding/index.md) · [🧠 NeurIPS2025 (3)](../../NeurIPS2025/nlp_understanding/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_understanding/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_understanding/index.md)

**[Causal Fine-Tuning under Latent Confounded Shift](causal_fine-tuning_under_latent_confounded_shift.md)**

:   本文提出 Causal Fine-Tuning (CFT)：在标准 BERT 微调里嵌入一个 SCM 启发的"高级稳定特征 $C$ + 低级混杂敏感特征 $\Phi$"分解，并用 front-door 风格的 do-calculus 调整公式做预测，在文本伪相关注入攻击下显著优于 SFT/SWA/WISE 等单域泛化基线。

**[Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting](controlling_the_risk_of_corrupted_contexts_for_language_models_via_early-exiting.md)**

:   本文把"用户提供的损坏上下文会降低 LLM 性能"这个问题形式化为风险控制——以 zero-shot 表现作"安全基线"，结合动态 early-exit（在中间层就出预测避免后层 overthink 有害上下文）+ context-aware 损失 + 改进的 Learn-then-Test 框架（保留负损失值用风险变换而非裁剪），在 9 个任务上既保证风险 ≤ user-specified $\epsilon$，又获得 > 50% 的算力加速。
