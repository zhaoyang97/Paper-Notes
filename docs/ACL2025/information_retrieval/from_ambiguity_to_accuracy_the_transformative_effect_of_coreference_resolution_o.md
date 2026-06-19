---
title: >-
  [论文解读] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems
description: >-
  [ACL 2025][信息检索/RAG][共指消解] 本文系统研究了共指消解（coreference resolution）对 RAG 系统中文档检索和问答生成两阶段的影响，发现共指消解能一致性提升检索性能（尤其 mean pooling 模型受益最大），在 QA 任务中小模型的性能提升显著大于大模型，甚至使小模型达到大模型的基线水平。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "共指消解"
  - "RAG"
  - "检索增强生成"
  - "嵌入模型"
  - "问答系统"
---

# From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems

**会议**: ACL 2025  
**arXiv**: [2507.07847](https://arxiv.org/abs/2507.07847)  
**代码**: 无  
**领域**: 信息检索  
**关键词**: 共指消解, RAG, 检索增强生成, 嵌入模型, 问答系统

## 一句话总结

本文系统研究了共指消解（coreference resolution）对 RAG 系统中文档检索和问答生成两阶段的影响，发现共指消解能一致性提升检索性能（尤其 mean pooling 模型受益最大），在 QA 任务中小模型的性能提升显著大于大模型，甚至使小模型达到大模型的基线水平。

## 研究背景与动机

**领域现状**：RAG 已成为 NLP 中提升事实一致性、减少幻觉的核心框架，将外部文档检索与 LLM 生成相结合。然而 RAG 的有效性经常受到检索文档中共指复杂性的阻碍——文档中大量代词和缩写引入的歧义会干扰上下文学习。

**核心痛点**：共指复杂性从两个层面影响 RAG 系统：(1) **检索阶段**，代词和缩写使嵌入模型难以准确捕获文档语义，导致查询-文档匹配失败；(2) **生成阶段**，歧义性的引用关系破坏推理链，降低答案的事实准确性。这些检索错误在生成过程中被放大，最终损害用户对系统输出的信任。

**研究动机**：现有研究缺乏对共指消解如何分别影响 RAG 各核心组件（检索和上下文学习）的系统性实证分析。作者希望回答：(1) 不同 pooling 策略的嵌入模型在共指消解后表现差异如何？(2) 不同规模的 LLM 从共指消解中的受益程度是否不同？

## 方法详解

### 整体框架

本文并非提出新模型架构，而是一个**系统性实证研究**。核心流程为：

1. **共指消解预处理**：对每篇文档 $d_i$，使用 LLM 驱动的共指消解函数 $f_{\text{coref}}$ 生成消解后文档 $d_i' = f_{\text{coref}}(d_i)$，将代词/缩写替换为显式先行词（如 "GR" → "general relativity"，"it" → "the basketball"）
2. **检索实验**：在原始/消解后文档上分别评估多种嵌入模型的检索性能
3. **QA 实验**：在原始/消解后文档上分别评估多种 LLM 的问答性能

### 关键设计

**1. LLM 驱动的共指消解**

- 使用 gpt-4o-mini 实现共指消解函数，输入含未解决共指的文本，输出中将多个引用同一实体的表达显式链接
- 例子：文档中 "GR"、"it"、"Its" 分别被替换为 "general relativity"、"the basketball"、"The basketball's"
- 这种方法通过增加文本中的显式语义信息，提升嵌入模型对文档语义的捕获能力

**2. 检索实验中的模型架构与 Pooling 策略对比**

实验涵盖两大类架构共 8 个嵌入模型：

- **Encoder-based (4个)**：e5-large-v2、stella_en_400M_v5（Mean pooling）；gte-modernbert-base（[CLS] pooling）；bge-large-en-v1.5（[CLS] pooling）
- **Decoder-based (4个)**：NV-Embed-v2、LLM2Vec（Mean pooling）；gte-Qwen2-1.5B、Linq-Embed-Mistral（Last token pooling）

**3. QA 实验中的模型规模对比**

实验选取 7 个指令微调模型，覆盖 3 个系列各 2 个规模：
- Llama 3.2-3B / 3.1-8B
- Qwen2.5-3B / 7B
- gemma-2-2b / 9b
- Mistral-7B

### 损失函数 / 训练策略

本文为纯实证研究，不涉及模型训练。评估指标包括：
- **检索任务**：nDCG@k (k=1,3,5)，评估检索排序质量
- **QA 任务**：BELEBELE 和 BoolQ 使用 log likelihood 计算准确率，SQuAD2.0 使用 F1 分数

## 实验关键数据

### 主实验一：检索性能对比

| 模型 | Pooling | 原始 AVG@1 | +CR AVG@1 | 原始 AVG@5 | +CR AVG@5 |
|------|---------|-----------|-----------|-----------|-----------|
| e5-large-v2 | Mean | 0.809 | **0.814** | 0.810 | 0.809 |
| stella_en_400M_v5 | Mean | 0.785 | **0.790** | 0.803 | **0.804** |
| NV-Embed-v2 | Mean | 0.836 | **0.843** | 0.836 | 0.836 |
| LLM2Vec | Mean | 0.814 | **0.826** | 0.824 | **0.827** |
| gte-modernbert-base | [CLS] | 0.793 | **0.794** | 0.811 | 0.807 |
| bge-large-en-v1.5 | [CLS] | 0.776 | **0.777** | 0.799 | **0.800** |
| gte-Qwen2-1.5B | Last | 0.816 | 0.816 | 0.812 | **0.814** |
| Linq-Embed-Mistral | Last | 0.810 | **0.815** | 0.830 | **0.832** |

**关键发现**：Mean pooling 模型受益最大（LLM2Vec +0.012 @1），[CLS]/Last pooling 增益较小。

### 主实验二：QA 性能对比

| 模型 | 规模 | BoolQ 原始→+CR | BELEBELE 原始→+CR | SQuAD 原始→+CR |
|------|------|---------------|-------------------|---------------|
| Qwen2.5-3B | 3B | 0.780→0.780 | 0.780→**0.858** (+7.8%) | 0.297→**0.550** (+25.3%) |
| Qwen2.5-7B | 7B | 0.860→0.860 | 0.862→**0.902** (+4.0%) | 0.398→**0.798** (+40.0%) |
| gemma-2-2b | 2B | 0.801→0.802 | 0.263→**0.307** (+4.3%) | 0.519→**0.621** (+10.2%) |
| gemma-2-9b | 9B | 0.865→0.865 | 0.541→**0.547** (+0.6%) | 0.765→**0.842** (+7.8%) |
| Llama3.2-3B | 3B | 0.764→0.764 | 0.812→**0.839** (+2.7%) | 0.644→**0.689** (+4.5%) |
| Llama3.1-8B | 8B | 0.820→0.821 | 0.883→**0.913** (+3.0%) | 0.558→**0.783** (+22.4%) |

### 消融实验

本文的实验设计本身就是系统性消融：
- **Pooling 策略消融**：Mean pooling 在共指消解后获得最显著提升，因为其对所有 token 平等对待，代词替换为显式先行词后每个 token 携带更多语义信息
- **模型规模消融**：在 BELEBELE 上，Qwen2.5-3B 提升 +7.8% 远大于 7B 的 +4.0%；gemma-2-2b 提升 +4.3% 远大于 9b 的 +0.6%
- **跨规模对比**：消解后的 gemma-2-2b (F1=0.621) 和 Qwen2.5-3B (F1=0.550) 在 SQuAD 上可达到甚至超越未消解的大模型（Llama3.1-8B=0.558, Qwen2.5-7B=0.398）

### 关键发现

- **Mean pooling + 共指消解具有协同效应**：mean pooling 平等对待每个 token，消解后用显式实体替代代词，使每个 token 承载更丰富语义
- **小模型获益不成比例地大**：小模型处理指代歧义的内在能力有限，共指消解相当于"外部辅助"，大幅降低理解门槛
- **共指消解增加文档长度**：用先行词替换代词会增长文本，这进一步放大了 mean pooling 的优势（可更有效地整合不同长度文本的信息）

## 亮点与洞察

1. **问题视角新颖**：将共指消解重新定位为 RAG 系统的通用预处理步骤，而非传统的 NLP 子任务，提供了跨任务的统一视角
2. **Mean pooling 的机理解释**深入：不仅报告现象，还从 token 级语义承载量的角度给出合理解释
3. **实践价值高**：发现为资源受限场景（只能用小模型）提供了一个低成本的性能提升方案——先做共指消解再检索/QA
4. **小模型 + 共指消解 ≈ 大模型**这一发现有很强的工程启示

## 局限与展望

1. **共指消解本身的偏差**：使用 gpt-4o-mini 进行消解可能引入模型特有偏差，不一定与人类理解完全对齐
2. **领域泛化性不足**：虽使用 4 个数据集，但未覆盖专业技术文本（如医学、法律），消解在这些领域的效果未知
3. **生成灵活性的权衡**：显式引用虽提升清晰度，但可能限制语言模型生成自然多样回复的能力
4. **计算成本**：对每篇文档调用 LLM 进行共指消解带来额外开销，大规模语料库场景下可能不可行
5. **缺乏端到端评估**：检索和 QA 分开评估，未展示"共指消解 → 更好检索 → 更好 QA"的完整级联效果

## 相关工作与启发

- **共指消解方法**：从规则方法到端到端神经方法（Lee et al., 2017; Kantor & Globerson, 2019），再到 LLM 驱动消解
- **RAG 系统改进**：与 Dense X Retrieval（Chen et al., 2024）在提升检索粒度方面互补，与检索重排序研究正交
- **启发**：可将共指消解作为 RAG 的标准预处理流水线组件，尤其在部署小模型时；也可考虑在 chunking 之前做消解以提升 chunk 的自包含性

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 新颖性 | 6 | 方法层面新颖度有限（纯实证），但视角和发现有价值 |
| 技术深度 | 5 | 无新模型/算法，核心贡献在于全面的实证分析 |
| 实验充分性 | 8 | 8个嵌入模型 + 7个 LLM + 4个数据集，覆盖面广 |
| 写作质量 | 7 | 逻辑清晰，分析有深度，图1的例子很直观 |
| 实用价值 | 7 | 发现可直接落地为 RAG 预处理管线 |
| 总分 | 6.5 | 扎实的实证研究，发现有启发，但缺乏方法创新 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Distracting Effect: Understanding Irrelevant Passages in RAG](the_distracting_effect_understanding_irrelevant_passages_in_rag.md)
- [\[ACL 2025\] Investigating Language Preference of Multilingual RAG Systems](investigating_language_preference_of_multilingual_rag_systems.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[ACL 2025\] KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](knowshiftqa_rag_knowledge_shifts.md)
- [\[ACL 2025\] Contradiction Detection in RAG-Based Chatbots](contradiction_detection_in_rag-based_chatbots.md)

</div>

<!-- RELATED:END -->
