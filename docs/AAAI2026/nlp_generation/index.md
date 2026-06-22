---
title: >-
  AAAI2026 文本生成论文汇总 · 3篇论文解读
description: >-
  3篇AAAI2026的文本生成方向论文解读，涵盖布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "文本生成"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
item_list:
  - u: "automaldesc_large-scale_script_analysis_for_cyber_threat_research/"
    t: "AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research"
  - u: "c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation/"
    t: "C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation"
  - u: "structured_language_generation_model_loss_calibration_and_formatted_decoding_for/"
    t: "Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**🤖 AAAI2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (12)](../../ICLR2026/nlp_generation/index.md) · [💬 ACL2026 (17)](../../ACL2026/nlp_generation/index.md) · [🧪 ICML2026 (2)](../../ICML2026/nlp_generation/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_generation/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_generation/index.md) · [💬 ACL2025 (27)](../../ACL2025/nlp_generation/index.md)

**[AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research](automaldesc_large-scale_script_analysis_for_cyber_threat_research.md)**

:   提出 AutoMalDesc 自动化静态分析框架，通过迭代自步学习流水线——从 900 个专家标注种子样本出发，经 LoRA 微调 Llama-3.3-70B 生成伪标签，多阶段质量过滤后进行 V2 训练——实现 5 种脚本语言的恶意软件自动分类和行为描述，Batch 脚本检测准确率从 52.7% 提升到 82.4%。

**[C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)**

:   提出 C3TG 框架，通过两阶段方法实现多维度细粒度可控文本生成：生成阶段用加权 KL 散度融合属性分布调整 token 概率，优化阶段用能量函数（分类器分数 + 冲突惩罚项）结合 Feedback Agent 迭代重写，在 17 个属性子类上达到 90.4% 属性准确率且大幅降低毒性。

**[Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)**

:   提出 SLGM 框架，通过**结构化输入格式**、**格式损失**和**格式感知解码**三大组件，将生成式语言模型的结构化预测任务重构为分类问题，在不增加模型参数的前提下显著提升 <1B 模型在 NER、RE、SRL 等 5 类 13 个数据集上的结构预测性能。
