---
title: >-
  ACL2025 优化/理论论文汇总 · 3篇论文解读
description: >-
  3篇ACL2025的优化/理论方向论文解读，涵盖对齐/RLHF、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "LLM"
item_list:
  - u: "aligned_but_blind_implicit_bias/"
    t: "Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race"
  - u: "ambik_dataset_of_ambiguous_tasks_in_kitchen_environment/"
    t: "AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment"
  - u: "scalebio_bilevel_data_reweighting/"
    t: "ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**💬 ACL2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/optimization/index.md) · [🔬 ICLR2026 (220)](../../ICLR2026/optimization/index.md) · [🧪 ICML2026 (88)](../../ICML2026/optimization/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md)

**[Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](aligned_but_blind_implicit_bias.md)**

:   揭示对齐训练的"种族盲视"副作用：对齐使 LLM 在歧义上下文中不再将 black/white 表征为种族概念，安全护栏因此无法激活，导致隐式偏见从 64.1% 飙升至 91.4%；反直觉地，在早期层注入种族感知激活（而非遗忘）可将隐式偏见从 97.3% 降至 42.4%。

**[AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)**

:   提出 AmbiK，一个专门用于厨房环境中歧义指令检测的纯文本数据集，包含 1000 对歧义/非歧义指令，按三种歧义类型（用户偏好/常识/安全）分类标注，并评估了多种基于 conformal prediction 的歧义检测方法，发现现有方法在该基准上表现很差。

**[ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](scalebio_bilevel_data_reweighting.md)**

:   ScaleBiO 提出基于罚函数重构的全一阶双层优化算法，首次将双层优化应用于 30B+ 参数 LLM 的数据源重加权，在 Qwen-2.5-32B 上实现 GSM8K +9%、MATH +5.8% 的提升。
