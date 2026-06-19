---
title: >-
  [论文解读] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model
description: >-
  [ACL 2025][多模态VLM][multilingual LVLM] 系统研究多语言LVLM训练策略中训练语言数量、语言数据分布和多语言OCR三个维度，发现可同时训练100种语言且仅需25-50%非英语数据，据此训练出覆盖100语言的Centurio模型达到SOTA。 问题定义：当前大多数LVLM主要使用英语数据训练…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "multilingual LVLM"
  - "视觉语言"
  - "training data distribution"
  - "OCR"
  - "language fidelity"
---

# Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model

**会议**: ACL 2025  
**arXiv**: [2501.05122](https://arxiv.org/abs/2501.05122)  
**代码**: [gregor-ge.github.io/Centurio](https://gregor-ge.github.io/Centurio)  
**领域**: 多模态VLM  
**关键词**: multilingual LVLM, vision-language, training data distribution, OCR, language fidelity  

## 一句话总结

系统研究多语言LVLM训练策略中训练语言数量、语言数据分布和多语言OCR三个维度，发现可同时训练100种语言且仅需25-50%非英语数据，据此训练出覆盖100语言的Centurio模型达到SOTA。

## 研究背景与动机

**问题定义：** 当前大多数LVLM主要使用英语数据训练，导致模型难以理解非英语输入、生成错误语言的输出、无法识别图像中非英语文字。如何在有限训练预算下，设计最优的多语言训练数据配比？

**现有方法局限：** 已有多语言LVLM工作（如Geigle et al., 2023; Sun et al., 2024; Maaz et al., 2024）在增加多语言数据时采用临时性（ad-hoc）策略，缺乏关于不同训练配比如何影响不同语言组表现的系统性洞察。

**核心动机：** 训练数据量总是受时间、计算资源和成本限制。在固定训练预算下，需要回答四个核心问题：(RQ1) 可以包含多少种训练语言而不降低英语性能？(RQ2-3) 预训练和指令微调中最优的语言分布是什么？(RQ4) 如何提升多语言图像文字理解能力？

## 方法详解

### 整体框架

采用LLaVA架构，以SigLIP SO400/384为图像编码器、Phi 3.5 (3.8B) 为LLM骨干，通过两层MLP对齐视觉和文本空间。训练分两阶段：
1. **预训练：** 在图像描述数据（ShareGPT4v，1.3M样本）上训练
2. **指令微调：** 在多样化视觉语言任务数据（LLaVA-Next改编，0.77M样本）上训练

使用NLLB开源机器翻译模型将英语数据翻译为其他语言，评估涵盖13个下游任务和43种语言。

### 关键设计

1. **渐进式语言扩展实验（RQ1）：** 从高资源语言组（T5: 6种）逐步扩展到T5-T4（24种）→ T5-T3（52种）→ T5-T2（69种）→ L100（99种），保持总数据量不变，观察性能变化
2. **语言分布搜索（RQ2-3）：** 固定语言数为100种，调整英语占比E从1%到90%，寻找最优平衡点。预训练和指令微调分别独立搜索
3. **多语言OCR增强（RQ4）：** 引入SMPQA（Synthetic Multilingual Plot QA）基准，覆盖11种语言和7种文字系统，使用Synthdog生成合成OCR训练数据

### 损失函数

标准的自回归语言建模损失（next-token prediction）。各阶段冻结图像编码器，仅更新MLP和LLM参数（LoRA）。OCR训练阶段额外解冻图像编码器。

## 实验

### RQ1: 训练语言数量（指令微调阶段，50%英语）

| 训练语言组 | T1 (最低资源) | T2 | T3 | T4 | T5 | en |
|-----------|------|------|------|------|------|------|
| English only | 14.4 | 30.4 | 24.4 | 23.6 | 28.5 | 53.6 |
| T5 (6种) | 16.5 | 31.0 | 26.3 | 26.7 | 34.0 | 53.7 |
| T5-T4 (24种) | 17.4 | 30.6 | 27.9 | 29.6 | 33.5 | 51.5 |
| L100 (99种) | 19.3 | 32.6 | 30.7 | 28.9 | 34.4 | 52.6 |

### RQ2: 指令微调中英语数据占比

| 英语占比 | T1 | T2 | T5 | en |
|---------|------|------|------|------|
| 1% | 19.1 | 30.3 | 31.7 | 48.9 |
| 25% | 19.7 | 35.5 | 33.0 | 50.3 |
| 50% | 19.3 | 32.6 | 34.4 | 52.6 |
| 90% | 15.9 | 31.2 | 34.1 | 54.8 |

### 关键发现

1. **无"多语言诅咒"：** 从7种扩展到100种训练语言，已包含语言的性能几乎不受影响，新增语言获得显著提升，语言保真度（language fidelity）从<1%提升至>95%
2. **少量多语言数据即可生效：** 25-50%的非英语数据即可大幅提升多语言能力，更多非英语数据有时反而降低性能
3. **预训练中多语言数据更重要：** 多语言预训练对低资源语言（T1/T2）提升尤为显著，英语数据比例从100%降至1%不会显著损害英语性能
4. **OCR数据对非拉丁文字帮助有限：** 合成OCR数据对拉丁字母语言效果显著，但对非拉丁文字（阿拉伯语、中文等）仍存在大量性能缺口，可能需要数量级更多的训练数据

## 亮点

- 迄今最系统的多语言LVLM训练策略研究，覆盖4个研究问题和100种语言
- 发现"语言暴露比数据量更重要"——少量多语言数据即可激活底层LLM的多语言能力
- 引入SMPQA基准填补多语言OCR评估空白
- 最终模型Centurio在14个任务和56种语言上达到SOTA，特别是在低资源语言上显著超越Qwen2-VL和InternVL 2.5
- 使用Llama 3作为额外骨干验证关键结论的泛化性

## 局限性

- 多语言训练数据通过机器翻译获得，中低资源语言翻译质量有限，实际效果可能被低估
- 图像编码器SigLIP对非拉丁文字的视觉表征能力有限，OCR增强效果受制于编码器
- 计算预算限制使得无法穷举所有组合（如预训练×指令微调的语言分布联合搜索）
- 仅在LLaVA架构上实验，结论对其他架构的泛化性未验证

## 相关工作

- **多语言LVLM：** PALO (Maaz et al., 2024) 支持10种语言、Pangea (Yue et al., 2024) 覆盖39种语言
- **视觉语言预训练：** LLaVA (Liu et al., 2023)、LLaVA-Next (Liu et al., 2024)
- **跨语言迁移：** Shaham et al. (2024)、Chen et al. (2024) 研究少语言训练+零样本迁移
- **多语言文字理解：** MTVQA (Tang et al., 2024)、Synthdog (Kim et al., 2022) 合成OCR数据

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning](singakids_a_multilingual_multimodal_dialogic_tutor_for_language_learning.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](chartcoder_chart_to_code.md)
- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions](logicqa_logical_anomaly_detection_with_vision_language_model_generated_questions.md)

</div>

<!-- RELATED:END -->
