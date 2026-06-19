---
title: >-
  [论文解读] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications
description: >-
  [ACL 2025][医疗NLP][RAG] 本文提出了 MedOmniKB 医学多源知识库和 Source Planning Optimisation (SPO) 方法，通过让专家模型探索多源检索计划并训练小模型学习源对齐，显著提升了医学多源检索规划能力，使 7B 小模型超越 72B 大模型。 领域现状：大语言模型在医学推…
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "RAG"
  - "medical QA"
  - "source planning"
  - "multi-source retrieval"
  - "knowledge base"
---

# Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications

**会议**: ACL 2025  
**arXiv**: [2501.02460](https://arxiv.org/abs/2501.02460)  
**代码**: [https://github.com/Jack-ZC8/Omni-RAG-Medical](https://github.com/Jack-ZC8/Omni-RAG-Medical)  
**领域**: 医学NLP  
**关键词**: RAG, medical QA, source planning, multi-source retrieval, knowledge base

## 一句话总结

本文提出了 MedOmniKB 医学多源知识库和 Source Planning Optimisation (SPO) 方法，通过让专家模型探索多源检索计划并训练小模型学习源对齐，显著提升了医学多源检索规划能力，使 7B 小模型超越 72B 大模型。

## 研究背景与动机

**领域现状**：大语言模型在医学推理、临床决策等任务中展现了良好前景，但其内部知识有限，容易产生幻觉。检索增强生成（RAG）技术通过引入外部知识来缓解这一问题，已成为医学问答的重要范式。

**现有痛点**：现有医学 RAG 方法通常将所有知识源统一处理，直接使用原始问题检索，未针对不同源的特性定制查询策略。虽然后续方法引入了 prompting 或反思策略来引导 LLM 利用多源，但由于对知识源内容缺乏真实感知，模型对源的期望与源实际内容之间存在错位。

**核心矛盾**：多源检索的核心难题是"源规划"（source planning）——即如何为每个知识源构建与其属性匹配的查询，而非千篇一律地使用相同问题检索所有源。现有方法要么忽略了这个问题，要么因感知不足而无法有效规划。

**本文目标** 1）缺乏足够丰富多样的医学知识库支持多源规划研究；2）如何让模型学会根据不同知识源的特性自动构建最优查询计划。

**切入角度**：作者观察到医学问题需要从教材、指南、研究文献、百科、知识图谱等不同类型的源获取信息，每种源的内容结构和检索方式截然不同。通过让大模型探索不同的检索计划并评估效果，可以生成高质量训练数据来教小模型学会源规划。

**核心 idea**：构建多体裁多结构医学知识库 MedOmniKB，并通过探索-评判-学习三阶段的 SPO 方法训练小模型实现高效多源检索规划。

## 方法详解

### 整体框架

输入是一个医学问题 $x$，系统需要从 5 个知识源 $K = \{K^i\}$（Book / Guideline / Research / Wiki / Graph）中检索相关文档，然后由 Reader 模型阅读检索到的文档来回答问题。核心是训练一个 Planner 模型 $\mathcal{M}_\theta$，为每个知识源生成定制化的查询，构成源规划 $P$。

### 关键设计

1. **MedOmniKB 多源医学知识库**:

    - 功能：提供丰富多样的多体裁、多结构医学知识检索基础
    - 核心思路：整合 5 类代表性知识源——Book（27.7k 文档，涵盖医学教材）、Guideline（45.7k 临床指南）、Research（25.3M PubMed 文献摘要）、Wiki（6.4M 维基百科条目）、Graph（UMLS+DrugBank 结构化知识图谱，含 1.7M 概念和 2.9M 关系）。文本类源切分为 ≤1000 字符的 chunk 并用 MedCPT 编码为向量存储于 Qdrant，图谱类通过概念查询获取定义和一跳关系后重排
    - 设计动机：此前的医学知识库要么体量不够大，要么缺少结构化知识图谱，无法充分研究源规划问题

2. **Planning Exploration + Planning Judging（探索与评判）**:

    - 功能：为训练数据自动生成正例和负例查询标注
    - 核心思路：用 Qwen2.5-72B 专家模型为训练集中的每个问题在每个源上生成 6 个候选查询（遵循源内多样性和跨源对齐两个原则），检索对应文档后，用同一专家模型以 LLM-as-a-judge 方式判断检索到的文档是否支持正确答案，将查询标记为正例 $q^{i,+}$ 或负例 $q^{i,-}$
    - 设计动机：通过 LLM-as-a-judge 获取高质量训练信号，比基于下游任务准确率或 rerank 分数更准确地评估查询质量

3. **Planning Learning（SFT + DPO 两阶段学习）**:

    - 功能：训练小模型学习源规划能力
    - 核心思路：从正例查询中为每个源选取至多 3 个构建正例计划 $P^+$，用 SFT 训练 $\mathcal{L}_{\text{SFT}} = -\mathbb{E} \log \mathcal{M}_\theta(P^+ | x)$；再用正负计划对进行 DPO 对齐 $\mathcal{L}_{\text{DPO}} = -\mathbb{E} \log \sigma(r_\theta(x, P^+) - r_\theta(x, P^-))$，其中 $r_\theta$ 表示基于 SFT 模型的隐式奖励
    - 设计动机：SFT 建立基础规划能力，DPO 进一步对齐多源特性；实验表明跳过 SFT 直接做 DPO 效果很差

### 损失函数 / 训练策略

- SFT 阶段：标准交叉熵损失，训练模型生成正例计划
- DPO 阶段：偏好对齐损失，正负计划对训练，$\beta$ 控制 KL 散度约束
- 训练基座：Qwen2.5-7B-Instruct
- 数据过滤：无任何正例查询的样本被过滤；每源至多 3 个正面查询以控制上下文长度

## 实验关键数据

### 主实验

| Reader | Planner/方法 | MedQA | MedMCQA | MMLU-Med | PubMedQA | 平均 |
|--------|-------------|-------|---------|----------|----------|------|
| Qwen2.5-7B | No Retrieval | 60.80 | 56.17 | 76.95 | 34.60 | 56.95 |
| Qwen2.5-7B | Original Question | 62.45 | 63.25 | 80.90 | 47.00 | 62.71 |
| Qwen2.5-7B | Prompting (72B) | 72.11 | 65.33 | 81.73 | 53.80 | 66.30 |
| Qwen2.5-7B | SeRTS (72B) | 70.70 | 66.83 | 82.55 | 55.60 | 67.06 |
| Qwen2.5-7B | **SPO (7B)** | **76.98** | **71.08** | **85.49** | **60.20** | **70.93** |

### 消融实验

| 配置 | MedQA | PubMedQA | SEER | 说明 |
|------|-------|----------|------|------|
| Full (SFT+DPO) | 76.98 | 60.20 | 61.90 | 完整模型 |
| SFT only | 74.08 | 59.20 | 58.50 | 去掉 DPO 后掉 ~3% |
| DPO only | 67.48 | 55.80 | 54.30 | 跳过 SFT 直接 DPO 效果很差 |
| Frozen (7B) | 64.10 | 53.20 | 53.50 | 未训练基线 |
| - Book | 70.38 | 52.60 | 56.80 | 去掉书籍源掉 8.57% |
| - Guideline | 72.35 | 56.40 | 52.10 | 去掉指南源在 SEER 上掉 15.83% |
| - Research | 71.72 | 35.40 | 51.20 | 去掉文献源在 PubMedQA 掉 41.2% |

### 关键发现

- SPO 训练后的 7B 模型在几乎所有配置下优于 72B 的冻结规划器（如 Prompting、Reflexion、SeRTS），体现了高效对齐的威力
- SFT 是关键基础：直接 DPO 仅获得有限提升（67.48 vs 64.10），而 SFT 能大幅提升（74.08 vs 64.10），DPO 在 SFT 之上进一步精调
- 不同源对不同任务贡献差异显著：Research 源对 PubMedQA 影响最大（去掉后从 60.2 降到 35.4），Guideline 对 SEER 影响最大
- 模型对未见过的源也有一定泛化能力（OOD corpus 实验），SPO 增强的源感知能力具有跨源迁移性

## 亮点与洞察

- **探索-评判-学习的训练范式**非常优雅：大模型负责探索和评判，小模型负责学习，实现了知识蒸馏式的源规划能力迁移。这一框架可推广到任何需要多源检索的场景
- **MedOmniKB 的多源设计**：将结构化知识图谱（UMLS+DrugBank）与非结构化文档统一纳入，是医学 RAG 领域较全面的知识库
- **7B 超 72B 的结论**很有实际价值：说明对于需要专业能力的任务，精细训练小模型比直接用大模型 prompting 更有效

## 局限与展望

- MedOmniKB 尚未覆盖所有医学知识资源（如影像数据、电子病历叙述等），可进一步扩展
- SPO 训练过程中的探索和评判步骤需要大量 72B 模型推理，成本较高
- 评测限于选择题和专家评分，缺乏真实医疗场景的用户满意度和治疗效果评估
- Graph 源（UMLS）的贡献相对较小，可能因为知识图谱的查询方式过于依赖精确术语匹配

## 相关工作与启发

- **vs MedRAG**: MedRAG 统一检索所有源不做规划，本文通过 SPO 实现差异化源规划，在 MedQA 上提升约 14 个百分点
- **vs Reflexion/SeRTS**: 反思式方法依赖模型自身反思能力，在多源场景下受限；SPO 通过显式训练信号避免了这个问题
- **vs RaFe Planning**: RaFe 用 rerank 分数作为训练信号，不够准确；SPO 的 LLM-as-judge 方式能更好地评估文档与答案的支持关系

## 评分

- 新颖性: ⭐⭐⭐⭐ 源规划问题的形式化定义和 SPO 三阶段范式较新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个数据集、多种 Reader、详尽的消融和 OOD 实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义明确
- 价值: ⭐⭐⭐⭐ 对医学 RAG 领域的多源规划提供了有价值的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [\[NeurIPS 2025\] Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID](../../NeurIPS2025/medical_nlp/demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](../../ACL2026/medical_nlp/sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ICML 2025\] On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains](../../ICML2025/medical_nlp/on_the_vulnerability_of_applying_retrieval-augmented_generation_within_knowledge.md)

</div>

<!-- RELATED:END -->
