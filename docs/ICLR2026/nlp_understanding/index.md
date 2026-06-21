---
title: >-
  ICLR2026 NLP理解论文汇总 · 2篇论文解读
description: >-
  2篇ICLR2026的 NLP 理解方向论文解读，涵盖 Agent、问答等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "NLP 理解"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "问答"
item_list:
  - u: "lane_label-aware_noise_elimination_for_fine-grained_text_classification/"
    t: "LANE: Label-Aware Noise Elimination for Fine-Grained Text Classification"
  - u: "whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh/"
    t: "What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📖 NLP 理解

**🔬 ICLR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (34)](../../ACL2026/nlp_understanding/index.md) · [🧪 ICML2026 (2)](../../ICML2026/nlp_understanding/index.md) · [🤖 AAAI2026 (1)](../../AAAI2026/nlp_understanding/index.md) · [🧠 NeurIPS2025 (3)](../../NeurIPS2025/nlp_understanding/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_understanding/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_understanding/index.md)

**[LANE: Label-Aware Noise Elimination for Fine-Grained Text Classification](lane_label-aware_noise_elimination_for_fine-grained_text_classification.md)**

:   LANE 把"识别错标样本"的经典 margin 指标升级为**标签感知边距 (Label-aware Margin)**——同样是负边距，若被错标的类别与模型预测类别语义相近（如把"愤怒"标成"恐惧"）就少惩罚，语义相远（把"信任"标成"恐惧"）才重罚，并据此对每个样本动态加权而非硬删除，在 10 个文本分类数据集上稳定超过 AUM/HMW 等强基线。

**[What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)**

:   提出 mean activation difference steering 方法和配套定量指标，在韵律诗生成和问答两个案例上跨 23 个开放模型（1B-32B）系统性证明：目标 token（韵脚/答案）的表示在序列早期位置已形成（前向规划），且因果性地影响中间 token 生成（后向规划）——隐式规划从 1B 模型即出现，是普遍机制而非大模型专属。
