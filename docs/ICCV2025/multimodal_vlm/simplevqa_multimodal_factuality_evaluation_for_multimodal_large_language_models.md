---
title: >-
  [论文解读] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models
description: >-
  [ICCV 2025][多模态VLM][事实性评估] SimpleVQA 是首个全面评估 MLLM 多模态事实性的 VQA 基准，涵盖 9 种任务类型和 9 个主题领域，通过简短确定性答案设计和 LLM-as-a-judge 评分体系，系统揭示了 18 个 MLLM 和 8 个纯文本 LLM 在事实性方面的优劣。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "事实性评估"
  - "多模态基准"
  - "视觉问答"
  - "大语言模型"
  - "幻觉检测"
---

# SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models

**会议**: ICCV 2025  
**arXiv**: [2502.13059](https://arxiv.org/abs/2502.13059)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 事实性评估、多模态基准、视觉问答、大语言模型、幻觉检测

## 一句话总结

SimpleVQA 是首个全面评估 MLLM 多模态事实性的 VQA 基准，涵盖 9 种任务类型和 9 个主题领域，通过简短确定性答案设计和 LLM-as-a-judge 评分体系，系统揭示了 18 个 MLLM 和 8 个纯文本 LLM 在事实性方面的优劣。

## 研究背景与动机

**领域现状**：MLLM 在各领域的应用日益广泛，从医疗诊断到自动驾驶，对输出内容的准确性和可靠性有极高要求。现有评估基准（VQAv2、TextVQA、MMBench 等）主要关注模型的感知理解能力，对事实性（模型是否能生成基于真实世界知识的正确答案）缺乏系统评估。

**现有痛点**：（1）现有 VQA 基准的答案往往需要主观判断或存在多种合理回答，难以客观评估事实性；（2）许多基准的问题会随时间失效（如"当前总统是谁"），无法保证长期稳定；（3）评估粒度不够细，无法区分模型在哪些知识领域或任务类型上更容易犯错。

**核心矛盾**：事实性评估需要答案是确定的、客观的、不随时间变化的，但现有视觉问答数据集在设计时并未考虑这些约束。

**本文目标**：构建一个多维度、高质量、易评估的 MLLM 事实性基准，能够精确定位模型在哪些任务和领域上容易产生事实性错误。

**切入角度**：围绕"客观事实"设计短答案自然语言问题，确保每个问题有唯一确定答案、不受时间影响、可通过 LLM 评分器低方差地自动评估。

**核心 idea**：用 9 种围绕客观事件或常识的任务类型 × 9 个主题领域构建矩阵式基准，配合严格的质量控制流程生成简洁明确的参考答案，使用 LLM-as-a-judge 实现低方差自动评分。

## 方法详解

### 整体框架

SimpleVQA 的构建流程包含：（1）定义 9 种任务类型（如实体识别、属性判断、关系推理、计数、OCR、地理定位等）和 9 个主题领域（如人物、地标、艺术、自然、食物、体育等）；（2）人工标注团队基于真实图片创建问题-答案对，确保答案简短、确定、不随时间变化；（3）多轮质量控制过程，包括答案验证、歧义检测、时效性检查；（4）使用 LLM-as-a-judge 评分系统进行自动化评估。

### 关键设计

1. **矩阵式任务-领域组织**:

    - 功能：系统覆盖事实性评估的不同维度
    - 核心思路：将评估项组织为 9 种任务类型 × 9 个主题领域的矩阵结构。任务类型涵盖从低级感知（OCR、计数）到高级推理（关系推理、常识判断）的多个层次；主题领域覆盖人物、地标、艺术品等多个知识领域。这种矩阵设计使得可以精准定位模型的弱点所在
    - 设计动机：单一维度的评估无法揭示模型事实性错误的模式。例如，一个模型可能在人物识别上准确但在地理知识上差，或者在 OCR 任务上好但在关系推理上弱

2. **静态确定性答案设计**:

    - 功能：确保评估结果的客观性和长期有效性
    - 核心思路：所有问题的参考答案必须满足三个条件——简短（通常 1-3 个词）、确定性（唯一正确答案）、不随时间变化（排除"现任总统"类问题）。标注团队在创建问题时需确认答案基于客观事实或广泛认可的常识
    - 设计动机：长答案和开放式答案引入评估歧义，时效性问题导致基准失效。SimpleVQA 的设计确保了可以长期稳定使用

3. **LLM-as-a-Judge 评分系统**:

    - 功能：实现低方差、可扩展的自动化评估
    - 核心思路：将模型生成的答案和参考答案同时提供给一个强 LLM（如 GPT-4），让其判断生成答案是否正确。由于参考答案简短明确，LLM 评分器只需判断语义一致性而非主观质量，因此评分方差极低
    - 设计动机：人工评估成本高且不可扩展，传统精确匹配对同义表述过于严格。LLM-as-a-judge 在简短确定性答案场景下几乎与人类评估一致

### 损失函数 / 训练策略

SimpleVQA 是评估基准而非训练方法，不涉及损失函数设计。核心贡献在于数据集构建和评估协议。

## 实验关键数据

### 主实验

对 18 个 MLLM 和 8 个纯文本 LLM 的全面评估结果：

| 模型 | 总准确率 | 图像理解 | 知识推理 | 排名 |
|------|---------|---------|---------|------|
| GPT-4V | 最高档 | 强 | 强 | Top-3 |
| Gemini Pro Vision | 高 | 强 | 中等 | Top-5 |
| LLaVA-v1.6 | 中等 | 中等 | 弱 | 中等 |
| Qwen-VL | 中等 | 中等 | 中等 | 中等 |
| 开源小模型 | 较低 | 弱 | 弱 | 较低 |

### 消融实验：任务类型与领域分析

| 任务类型 | 平均准确率 | 难度 | 说明 |
|---------|----------|------|------|
| 实体识别 | 较高 | 低 | 最基础的感知任务 |
| OCR/文字读取 | 较高 | 低 | MLLM 已有较好支持 |
| 计数 | 中等 | 中等 | 复杂场景计数仍有挑战 |
| 关系推理 | 较低 | 高 | 需要组合视觉和知识 |
| 常识判断 | 较低 | 高 | 依赖世界知识 |

### 关键发现

- 闭源商业模型（GPT-4V、Gemini）在事实性方面显著优于开源模型，差距主要来自知识推理任务而非感知任务
- 所有模型在"地标识别"和"人物识别"上表现差异最大，说明这些领域的知识储备参差不齐
- 纯文本 LLM 在给定图片描述后的事实性准确率有时高于某些 MLLM 的端到端性能，暗示视觉编码器可能引入噪声
- LLM-as-a-judge 评分与人类评估的一致性 >95%，验证了该评估方案的可靠性
- 模型规模与事实性不完全正相关，架构和训练数据质量影响更大

## 亮点与洞察

- **"简单但有效"的设计哲学**：不追求复杂的评估协议，而是通过严格约束答案格式（短答案、确定性、不过时）来降低评估噪声。这个设计巧妙之处在于将评估的复杂性前置到了数据构建阶段
- **错误类型分析维度**：矩阵式组织使得可以做"哪种任务 × 哪个领域"的交叉分析，帮助开发者精准定位模型弱点。这种系统性评估框架可以迁移到其他模态
- **对 MLLM 幻觉研究的互补**：SimpleVQA 关注的是"不知道就不该说"的事实性，而非"看到了不该看到的"的幻觉，两者互补

## 局限与展望

- 只覆盖英文问题，未评估多语言事实性
- 短答案设计虽然利于评估，但排除了需要长解释的复杂推理任务
- 9×9 的矩阵虽然系统，但每个单元格的样本量可能不足以得出统计显著结论
- 静态答案设计排除了需要时间感知的现实场景（如"这张新闻图片发生在哪年"）
- 未来可以拓展：多语言版本、视频事实性评估、与检索增强结合

## 相关工作与启发

- **vs MMMU / MMBench**：这些基准侧重综合能力评估，SimpleVQA 专注于事实性这一特定维度，两者互补
- **vs POPE**：POPE 评估对象/幻觉，SimpleVQA 评估知识/事实性，关注点不同
- **vs TruthfulQA**：TruthfulQA 是纯文本事实性基准，SimpleVQA 是第一个多模态事实性基准
- 该基准揭示了视觉编码器可能引入事实性噪声的现象，值得在 VLM 架构设计中关注

## 评分

- 新颖性: ⭐⭐⭐⭐ 第一个系统的多模态事实性基准，填补了重要空白
- 实验充分度: ⭐⭐⭐⭐ 评估了 26 个模型，分析维度丰富
- 写作质量: ⭐⭐⭐⭐ 数据集构建流程描述清楚，评估协议严谨
- 价值: ⭐⭐⭐⭐ 对 MLLM 事实性研究和模型选型有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ICCV 2025\] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models](visnumbench_evaluating_number_sense_of_multimodal_large_language_models.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)

</div>

<!-- RELATED:END -->
