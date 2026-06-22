---
title: >-
  ACL2025 目标检测论文汇总 · 2篇论文解读
description: >-
  2篇ACL2025的目标检测方向论文解读，涵盖对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
item_list:
  - u: "anchored_answers_unravelling_positional_bias_in_gpt-2s_multiple-choice_questions/"
    t: "Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions"
  - u: "weed_out_then_harvest_dual_low-rank_adaptation_is_an_effective_noisy_label_detec/"
    t: "Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**💬 ACL2025** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (99)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/object_detection/index.md) · [🧪 ICML2026 (6)](../../ICML2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (27)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md)

**[Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions](anchored_answers_unravelling_positional_bias_in_gpt-2s_multiple-choice_questions.md)**

:   首次从失败案例角度对GPT-2系列在MCQ中的"锚定偏差"（始终选A）进行机械分析，通过Logit Lens定位到MLP中存储"A"偏好的特定值向量，用极简干预（更新值向量）将MCQ准确率平均提升70%+。

**[Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning](weed_out_then_harvest_dual_low-rank_adaptation_is_an_effective_noisy_label_detec.md)**

:   提出Delora框架，通过引入clean LoRA和noisy LoRA双模块构建噪声标签检测器，将样本选择与模型训练解耦，打破传统"小损失"方法中样本选择与训练互相影响的恶性循环。
