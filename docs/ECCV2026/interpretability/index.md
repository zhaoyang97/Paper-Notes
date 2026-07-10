---
title: >-
  ECCV2026 可解释性论文汇总 · 2篇论文解读
description: >-
  2篇ECCV2026的可解释性方向论文解读，收录 Evaluating the Interpretabilit、On the Faithfulness of Post-Ho等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2026"
  - "可解释性"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "evaluating_the_interpretability_of_sparse_autoencoders_with_concept_annotations/"
    t: "Evaluating the Interpretability of Sparse Autoencoders with Concept Annotations"
  - u: "on_the_faithfulness_of_post-hoc_concept_bottleneck_models/"
    t: "On the Faithfulness of Post-Hoc Concept Bottleneck Models"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**🎞️ ECCV2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (34)](../../CVPR2026/interpretability/index.md) · [🔬 ICLR2026 (195)](../../ICLR2026/interpretability/index.md) · [💬 ACL2026 (63)](../../ACL2026/interpretability/index.md) · [🧪 ICML2026 (91)](../../ICML2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (80)](../../NeurIPS2025/interpretability/index.md)

**[Evaluating the Interpretability of Sparse Autoencoders with Concept Annotations](evaluating_the_interpretability_of_sparse_autoencoders_with_concept_annotations.md)**

:   本文提出一套基于人类概念标注的 SAE 可解释性评估框架，包含支持多对一匹配的 FBMP 算法和基于目标属性扰动的 TAPAScore 因果验证指标，并构建了 synCUB/synCOCO 两个合成扰动数据集；实验表明现有自动评估指标（FMS、MS、CKNNA）无法通过健全性检验，而本文的匹配指标和 TAPAScore 能可靠区分训练/未训练 SAE，且过完备度增加反而会降低扰动对齐质量。

**[On the Faithfulness of Post-Hoc Concept Bottleneck Models](on_the_faithfulness_of_post-hoc_concept_bottleneck_models.md)**

:   这篇论文系统分析了 post-hoc CBM 中概念投影（concept projection）的忠实性（faithfulness）问题，证明了分类器准确率不能作为瓶颈层质量的代理指标（随机投影也能达到竞争性能），识别出两个导致不忠实的根本原因——辅助数据集的协变量偏移和 VLM 代理标签的系统性错误——并提出了对应的诊断指标（$\mathcal{H}\Delta\mathcal{H}$-散度 和 误差-激活相关性）。
