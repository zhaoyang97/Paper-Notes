---
title: >-
  [论文解读] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language
description: >-
  [ACL 2025 (GeBNLP Workshop)][社会计算][性别偏见] 本文针对德语场景构建了五个性别偏见评测数据集，并在八个多语言 LLM 上进行系统评估，揭示了德语特有的性别偏见挑战——包括阳性职业名词的歧义解读和看似中性的名词对性别感知的影响。 领域现状：LLM 中的性别偏见评估近年来受到广泛关注…
tags:
  - "ACL 2025 (GeBNLP Workshop)"
  - "社会计算"
  - "性别偏见"
  - "德语"
  - "大语言模型"
  - "偏见评测"
  - "多语言"
---

# Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language

**会议**: ACL 2025 (GeBNLP Workshop)  
**arXiv**: [2507.16557](https://arxiv.org/abs/2507.16557)  
**代码**: [https://github.com/rwth-i6/Gender-Bias-in-German-LLMs](https://github.com/rwth-i6/Gender-Bias-in-German-LLMs)  
**领域**: 社会计算  
**关键词**: 性别偏见、德语、大语言模型、偏见评测、多语言

## 一句话总结

本文针对德语场景构建了五个性别偏见评测数据集，并在八个多语言 LLM 上进行系统评估，揭示了德语特有的性别偏见挑战——包括阳性职业名词的歧义解读和看似中性的名词对性别感知的影响。

## 研究背景与动机

**领域现状**：LLM 中的性别偏见评估近年来受到广泛关注，已有多种方法被提出用于量化和分析偏见，包括 WEAT（Word Embedding Association Test）、模板填充测试、以及基于生成的评估方法等。

**现有痛点**：现有偏见评估方法绝大多数是为英语设计的。将这些方法直接迁移到其他语言时面临严重的可转移性问题——不同语言的语法结构（如性、数、格系统）、社会文化背景和职业称谓规范差异巨大，英语中的偏见模式不能简单映射到其他语言。

**核心矛盾**：德语作为一种有复杂词性系统（阳性/阴性/中性三种语法性别）的语言，其性别偏见表现形式与英语有本质区别。例如，德语中阳性职业名词（如 "Arzt"）既可以指代男性医生，也被广泛用作泛指（generic masculine），这种歧义在英语中不存在，导致直接翻译英语偏见测试会产生误导性结果。

**本文目标**：为德语 LLM 偏见评估构建专门的数据集和评测框架，系统揭示德语环境下性别偏见的独特挑战。

**切入角度**：作者从德语语法和社会语言学特征出发，基于已有的性别偏见概念（职业刻板印象、代词消歧、性别联想等），设计适配德语特性的评测方案。

**核心 idea**：构建五个基于不同偏见概念的德语数据集，配合多种评估方法论，在八个多语言 LLM 上系统测试，揭示德语独特的偏见模式。

## 方法详解

### 整体框架

本文的方法框架是一个评测体系而非模型。整体流程为：（1）基于性别偏见理论设计五类评测维度；（2）针对每个维度构建德语数据集，充分考虑德语语法特性；（3）定义每个数据集对应的评估指标和方法论；（4）在八个多语言 LLM 上进行系统评测并分析结果。

### 关键设计

1. **职业性别刻板印象数据集（Occupational Stereotype Dataset）**:

    - 功能：评估 LLM 是否在职业-性别关联上存在刻板印象
    - 核心思路：选取一系列德语职业名词，同时提供阳性形式（如 "Arzt"）和阴性形式（如 "Ärztin"），设计提示模板让 LLM 为职业角色生成性别相关属性或补全句子。通过分析 LLM 输出中的性别分布来量化刻板印象程度
    - 设计动机：需要处理德语中"泛指阳性"（generic masculine）的问题。当使用 "Arzt" 时，LLM 可能将其理解为"男医生"而非泛指，这本身就是一种偏见信号

2. **代词消歧数据集（Pronoun Resolution Dataset）**:

    - 功能：测试 LLM 在代词指代消解中是否受性别偏见影响
    - 核心思路：构建包含两个不同性别角色和一个歧义代词的德语句子，要求 LLM 判断代词指代哪个角色。德语的代词系统比英语更复杂（如 "sie" 既可指"她"也可指"他们"），为此专门设计了避免语法线索泄露的句子模板
    - 设计动机：德语代词系统的复杂性使得代词消歧任务的偏见评测需要比英语更精细的设计，简单翻译 Winogender/WinoBias 会引入大量语法干扰

3. **中性名词性别联想数据集（Neutral Noun Gender Perception Dataset）**:

    - 功能：探测看似中性的名词是否会影响 LLM 的性别感知
    - 核心思路：德语中许多非人物名词也有语法性别（如 "die Sonne/太阳" 是阴性，"der Mond/月亮" 是阳性），测试这些语法性别是否会泄漏到 LLM 对相关人物角色的性别判断中。设计了将中性概念与性别判断任务组合的测试模板
    - 设计动机：这是德语（及其他有语法性别的语言）特有的偏见来源，在英语偏见评测中完全不存在，是跨语言迁移失败的典型案例

### 损失函数 / 训练策略

本文不涉及模型训练，纯评测工作。评估指标包括性别分布偏差（偏离 50-50 均匀分布的程度）、刻板印象一致率（LLM 输出与社会刻板印象一致的比例）等。

## 实验关键数据

### 主实验

八个多语言 LLM 在五个德语偏见数据集上的表现：

| 模型 | 职业刻板印象 | 代词消歧偏差 | 中性名词影响 | 整体偏见程度 |
|------|------------|------------|------------|------------|
| GPT-4系列 | 中等偏见 | 较低偏差 | 检测到影响 | 相对较好 |
| Llama系列 | 较强偏见 | 中等偏差 | 显著影响 | 偏见较明显 |
| 多语言中等模型 | 较强偏见 | 较高偏差 | 显著影响 | 偏见最明显 |
| 整体趋势 | 模型间差异大 | 德语特有模式 | 语法性别泄漏 | 需针对性评测 |

### 消融实验

| 评测条件 | 偏见变化 | 说明 |
|---------|---------|------|
| 使用泛指阳性 vs 阴性形式 | 显著差异 | 阳性形式触发更强刻板印象 |
| 有语法性别线索 vs 无 | 有线索偏差更大 | 语法性别确实泄漏到语义判断 |
| 德语 vs 英语同条件 | 模式不同 | 确认不能直接迁移英语评测 |
| 不同提示模板 | 结果有波动 | 提示敏感性需注意 |

### 关键发现

- **泛指阳性的歧义**是德语偏见评测中最核心的挑战：LLM 倾向于将阳性职业名词理解为男性特指而非泛指，这放大了职业-性别刻板印象的量化值
- **语法性别到语义性别的泄漏**在多个模型中被确认：名词的语法性别（阳/阴/中）确实会影响 LLM 对相关人物角色的性别判断
- 不同模型的偏见模式差异很大，没有一个模型在所有维度上都表现最好，说明偏见是多维度的
- 评测结果对提示模板的具体措辞比较敏感，强调了评测方法论标准化的重要性

## 亮点与洞察

- **语法性别对语义偏见的影响**是一个非常有趣的发现——这意味着对于有语法性别的语言（法语、德语、西班牙语等），LLM 的偏见来源比英语更复杂，需要专门的评测工具
- **五个数据集覆盖不同偏见维度**的设计严谨。每个数据集针对一个特定的偏见概念，避免了单一指标掩盖多面问题的风险
- 代码和数据集开源在 GitHub 上，为后续德语和其他有语法性别语言的偏见研究提供了可复用的基础设施

## 局限与展望

- 作为 Workshop Paper 篇幅有限，部分实验分析深度不足
- 仅覆盖德语一种有语法性别的语言，结论能否推广到法语、西班牙语等需要验证
- 数据集规模相对较小，可能存在特定模板或措辞引入的偏差
- 评测指标主要关注二元性别（男/女），未涵盖非二元性别认同
- 可以在此基础上探索针对性的偏见缓解（debiasing）策略

## 相关工作与启发

- **vs WinoBias/Winogender (英语)**: 这些经典偏见评测直接翻译到德语会因语法性别差异失效，本文的数据集是专为德语设计的替代方案
- **vs BBQ/BOLD benchmark**: 这些大规模偏见基准主要面向英语，本文为多语言偏见评测提供了方法论范例
- 研究发现的"语法性别泄漏"现象对多语言模型的对齐训练有重要启示

## 评分

- 新颖性: ⭐⭐⭐⭐ 将偏见评测扩展到德语的工作有一定新意，但方法论上主要是对已有概念的适配
- 实验充分度: ⭐⭐⭐⭐ 八个模型 × 五个数据集的覆盖面可以，但受 Workshop 篇幅限制分析深度有限
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，德语语法性别特性的解释对非德语读者也很友好
- 价值: ⭐⭐⭐⭐ 主要贡献是数据集和评测，对德语 NLP 社区有直接价值，但影响面相对较窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)
- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](../../ACL2026/social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)

</div>

<!-- RELATED:END -->
