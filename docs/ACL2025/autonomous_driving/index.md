---
title: >-
  ACL2025 自动驾驶论文汇总 · 1篇论文解读
description: >-
  1篇ACL2025的自动驾驶方向论文解读，涵盖 LLM、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "自动驾驶"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "时序预测"
item_list:
  - u: "embracing_large_language_models_in_traffic_flow_forecasting/"
    t: "Embracing Large Language Models in Traffic Flow Forecasting"
item_total: 1
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**💬 ACL2025** · **1** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (157)](../../CVPR2026/autonomous_driving/index.md) · [🔬 ICLR2026 (50)](../../ICLR2026/autonomous_driving/index.md) · [🧪 ICML2026 (8)](../../ICML2026/autonomous_driving/index.md) · [🤖 AAAI2026 (56)](../../AAAI2026/autonomous_driving/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/autonomous_driving/index.md) · [📹 ICCV2025 (91)](../../ICCV2025/autonomous_driving/index.md)

**[Embracing Large Language Models in Traffic Flow Forecasting](embracing_large_language_models_in_traffic_flow_forecasting.md)**

:   提出 LEAF 框架，用图分支（pair-wise关系）和超图分支（non-pair-wise关系）的双分支预测器生成候选预测，再用冻结的 LLM 作为选择器（判别而非生成）挑选最优预测，通过 ranking loss 反馈优化预测器，在 PEMS 数据集上取得 SOTA。
