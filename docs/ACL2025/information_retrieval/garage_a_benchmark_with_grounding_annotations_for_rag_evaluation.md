---
title: >-
  [论文解读] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation
description: >-
  [ACL 2025][信息检索/RAG][RAG benchmark] GaRAGe 是一个包含 2366 个问题和超过 35K 条人工标注 grounding 段落的 RAG 基准，通过细粒度的 grounding 相关性标注，系统评估 LLM 在 RAG 场景下识别相关信息、拒绝回答和归因引用的能力。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "RAG benchmark"
  - "grounding annotation"
  - "factuality"
  - "deflection"
  - "attribution"
---

# GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation

**会议**: ACL 2025  
**arXiv**: [2506.07671](https://arxiv.org/abs/2506.07671)  
**代码**: [有](https://github.com/amazon-science/GaRAGe)  
**领域**: NLP / RAG 评估  
**关键词**: RAG benchmark, grounding annotation, factuality, deflection, attribution

## 一句话总结

GaRAGe 是一个包含 2366 个问题和超过 35K 条人工标注 grounding 段落的 RAG 基准，通过细粒度的 grounding 相关性标注，系统评估 LLM 在 RAG 场景下识别相关信息、拒绝回答和归因引用的能力。

## 研究背景与动机

RAG（检索增强生成）是当前 LLM 应用中最重要的范式之一，用户需要 LLM 能够从检索到的文档中准确定位相关信息并生成有据可查的回答。然而，现有 RAG 基准存在若干关键缺陷：

**评估维度混淆**：多数基准将查询生成和答案生成合并评估，无法单独衡量 LLM 的信息筛选能力

**grounding 质量未标注**：现有基准中检索到的段落要么全部相关（不现实），要么相关性未知（无法精准评估），缺少对每条 grounding 的人工相关性标注

**答案形式过于简单**：大量基准使用短答案或多选题，与真实用户需要长文本带引用的场景脱节

**来源单一**：很少有基准同时包含公开 Web 和私有知识库的混合 grounding，无法模拟企业 RAG 场景

GaRAGe 正是针对以上问题设计的：它结合了时间敏感性、多维度复杂性、公私混合来源和人工撰写的长文本答案，为 RAG 系统提供了迄今最全面的评测平台。

## 方法详解

### 整体框架

GaRAGe 的构建流程包含三大步骤：（1）多阶段复杂问题生成；（2）多源 grounding 段落收集；（3）人工标注和验证。

### 关键设计

1. **动态多类型问题生成**：采用 LLM 驱动的四步流水线——生成搜索计划→执行 Web 搜索→通过信息融合生成问题→过滤去重。问题涵盖时间敏感性（快变/慢变/静态）、复杂度（比较/多跳/后处理）、热度（头部/腰部/尾部）和领域类别四个维度。设计动机是模拟真实 RAG 场景中用户问题的多样性和挑战性。

2. **多源 grounding 收集**：对每个问题，先通过查询分解将复杂问题拆分为聚焦子查询，再分别从 Web 搜索引擎和私有知识库（Enron 邮件、ArXiv 摘要、AWS DevOps 指南、SEC 文件）检索文档。使用 STS 分类器过滤偏离原始问题意图的子查询，并通过交叉编码器进行文档重排。部分子集故意不做重排以引入更多噪声，增加难度。

3. **细粒度人工标注**：专业标注员对 2366 个问题进行四维度标注（时间敏感性/复杂度/热度/类别），对每条 grounding 段落标注相关性（回答问题/相关信息/过时/未知），并撰写带引用标记的长文本答案。427 个问题被标注为需要拒绝回答（deflection），模拟 grounding 不足的情况。

4. **评估指标体系**：

    - **Eligibility Score**：答案是否充分回应了用户请求
    - **Relevance-Aware Factuality (RAF)**：答案是否仅基于相关段落、且同时满足 eligibility
    - **Deflection Score**：在 grounding 不足时 LLM 是否正确拒绝回答（真阳性率）
    - **Attribution Score**：引用标记的精确率/召回率/F1

### 损失函数 / 训练策略

GaRAGe 本身是评估基准而非模型，不涉及训练。评估使用 GPT-4o 作为 judge 执行自动评分，温度设为 0.2。

## 实验关键数据

### 主实验（表格）

| 模型 | Eligibility | Factuality | RAF | Deflection TP |
|------|-----------|------------|-----|--------------|
| GPT-4o | **92.47** | 59.30 | 52.88 | **31.1** |
| Gemini 1.5 Flash | 84.88 | **70.50** | 59.43 | 27.2 |
| Nova Pro | 87.77 | 66.63 | **60.67** | 18.0 |
| Claude Sonnet | 86.07 | 64.67 | 48.91 | 25.3 |
| Qwen 32b | 90.50 | 61.00 | 52.90 | 21.5 |
| Mistral | 85.30 | 43.32 | 34.32 | 5.2 |

### Attribution 实验（表格）

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Claude Haiku | 49.9 | **71.9** | **58.9** |
| GPT-4o | **57.9** | 59.0 | 58.4 |
| Gemini 1.5 | 54.7 | 56.3 | 55.5 |
| Nova Pro | 56.9 | 49.6 | 53.0 |

### 关键发现

1. **所有模型在 RAF 上表现不佳**：最好的 Nova Pro 也仅达到 60.67%，说明 LLM 普遍倾向于"过度摘要"而非严格基于相关段落生成
2. **Deflection 能力极弱**：最好的 GPT-4o 真阳性率仅 31.1%，意味着近 70% 的场景下 LLM 在无可用 grounding 时仍会编造答案
3. **时间敏感问题更难**：快变问题的 RAF 比静慢问题低约 10%
4. **私有知识库场景显著更难**：相比 Web 问题，私有 KB 问题性能下降超过 10%
5. **grounding 噪声直接影响质量**：低相关比例（< 33%）的 grounding 比高相关比例（> 66%）导致 RAF 下降约 30%

## 亮点与洞察

- **核心贡献**是在每条 grounding 段落上提供相关性标注，首次实现了"是否基于相关信息"和"是否忠实于上下文"的区分评估——这在以往基准中是混淆的
- RAF 指标的提出具有重要意义：传统 factuality 只关心是否有依据，而 RAF 进一步要求依据必须是相关的、新鲜的
- 数据集同时覆盖 Web 和私有 KB 的混合检索场景，贴近企业部署的真实需求

## 局限与展望

1. 仅有英语数据，缺乏多语言支持
2. 评价使用 GPT-4o 作为 judge，可能存在对 GPT 系列模型的偏好
3. 标注的主观性（如话题热度）可能引入一定噪声
4. 数据集可能被后续 LLM 训练数据覆盖，影响长期评估有效性

## 相关工作与启发

- 与 CRAG、MultiHop RAG、Facts Grounding 等基准相比，GaRAGe 在 grounding 标注、答案全面性和多源支持方面更完整
- 对 RAG 系统开发者的关键启示：当前 LLM 在"区分相关与不相关 grounding"方面的能力远未达到可靠水平，简单拼接检索结果作为上下文的做法急需改进

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 细粒度 grounding 标注和 RAF 指标是重要贡献，弥补了现有评估的关键盲区
- **实验充分度**: ⭐⭐⭐⭐⭐ — 评估了 11 个模型、多个维度切片分析（时间敏感性/来源/噪声水平/热度），非常全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表设计优良，评估框架阐述得当
- **价值**: ⭐⭐⭐⭐⭐ — 对 RAG 社区具有直接且长期的实用价值，填补了评估空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
