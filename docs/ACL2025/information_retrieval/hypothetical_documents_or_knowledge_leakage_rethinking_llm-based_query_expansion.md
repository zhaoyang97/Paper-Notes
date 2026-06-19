---
title: >-
  [论文解读] Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion
description: >-
  [ACL2025][信息检索/RAG][查询扩展] 质疑 LLM-based 查询扩展（HyDE/Query2doc）的性能提升是否来自"假设性文档生成"，发现性能增益仅在 LLM 生成的文档包含与 gold evidence 语义一致的句子时才一致出现，揭示了 benchmark 中可能存在的知识泄露问题。
tags:
  - "ACL2025"
  - "信息检索/RAG"
  - "查询扩展"
  - "知识泄露"
  - "事实验证"
  - "HyDE"
  - "Query2doc"
  - "零样本检索"
---

# Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion

**会议**: ACL2025  
**arXiv**: [2504.14175](https://arxiv.org/abs/2504.14175)  
**代码**: 待确认  
**领域**: 信息检索  
**关键词**: 查询扩展, 知识泄露, 事实验证, HyDE, Query2doc, 零样本检索

## 一句话总结

质疑 LLM-based 查询扩展（HyDE/Query2doc）的性能提升是否来自"假设性文档生成"，发现性能增益仅在 LLM 生成的文档包含与 gold evidence 语义一致的句子时才一致出现，揭示了 benchmark 中可能存在的知识泄露问题。

## 研究背景与动机

**领域现状**: 零样本检索（zero-shot retrieval）是知识密集型应用的核心组件。近年来，HyDE 和 Query2doc 等 LLM-based 查询扩展（QE）方法在多个 benchmark 上取得了显著性能提升，被广泛采用。

**现有痛点**: 这些方法的核心假设是"LLM 生成的假设性文档虽然可能不准确，但能拉近 query 与目标文档的语义距离"——但该假设从未被严格验证。

**核心矛盾**: LLM 在海量语料上预训练，其中很可能包含 benchmark 的知识源（如 Wikipedia），那么 LLM 生成的到底是"假设性文档"还是"已记忆知识的复述"？如果是后者，检索任务退化为近乎精确匹配的 trivial task。

**本文目标**: 调查 LLM-based QE 方法的性能提升中有多少可归因于知识泄露（knowledge leakage），而非真正的假设性推理能力。

**切入角度**: 选择事实验证（fact verification）作为测试平台——该任务有明确的 gold evidence 可供比对，且是分类任务，便于清晰评估 QE 对端任务的影响。

**核心 idea**: 用 NLI 检测 LLM 生成文档中是否"蕴含"了 gold evidence 句子，将样本分为 matched/unmatched 两组对比性能，发现 QE 的效果仅在 matched 组上成立。

## 方法详解

### 整体框架

本文是一项实证分析研究，核心流程为：
1. 在三个 fact-verification benchmark 上运行两种主流 QE 方法（Query2doc + HyDE）
2. 用 NLI-based matching 算法检测 LLM 生成文档是否包含 gold evidence
3. 按 matched/unmatched 条件分割数据，分别评估检索和验证性能
4. 比较七种 LLM 在三个数据集上的一致性趋势

### 关键设计 1：NLI-based Matching 算法

- **功能**: 判断 LLM 为某个 claim 生成的扩展文档 $d$ 是否包含与 gold evidence 语义等价的句子。
- **为什么**: 如果 LLM 在扩展文档中"复述"了 gold evidence，那么检索性能的提升可能只是因为查询向量中已经嵌入了答案本身。
- **怎么做**:
  1. **句子切分**: 用 spaCy 对生成文档 $d$ 分句，去掉与 claim 高度重复的句子（ROUGE-2 > 0.95）；
  2. **NLI 标注**: 对所有 $(e_i, s_j) \in E \times S$ 对用 GPT-4o-mini 做 NLI 判断（entailment / contradiction / neutral）；
  3. **标签聚合**: 只要存在任一对 $(e_i, s_j)$ 被标为 entailment，则该 claim 标记为 matched（M），否则为 unmatched（¬M）。

### 关键设计 2：两种 QE 方法的实验设置

- **Query2doc**: 生成伪文档 $d$，将 $d$ 与 query 多份拷贝拼接为扩展查询 $q^+$，用 BM25 检索。$n=5$。
- **HyDE**: 生成假设文档 $d$，用 Contriever 分别编码 $q$ 和 $d$，向量平均后检索。$N=1$。

### 关键设计 3：评估策略

- **检索评估**: FEVER/SciFact 用 Recall@5 和 NDCG@5；AVeriTeC 用 METEOR 和 BERTScore（因 gold evidence 是人工改写的）
- **验证评估**: 用 GPT-4o-mini 对 top-5 检索证据做 verdict prediction，评估 macro F1
- **统计显著性**: 基于 8 次 LLM 生成重复实验，报告均值±标准误

### 损失函数 / 训练策略

本文无模型训练，纯分析性工作。所有 LLM 均以 zero-shot prompting 方式使用。

## 实验关键数据

### 主实验：QE 方法整体效果（Query2doc + GPT-4o-mini，k=5）

| 指标 | FEVER | SciFact | AVeriTeC |
|------|:-:|:-:|:-:|
| BM25 baseline Recall@5 | 31.0 | 51.2 | 17.8 (METEOR) |
| Query2doc Recall@5 | 36.4 | 55.1 | 19.1 (METEOR) |
| Query2doc F1 | 55.6 | 52.5 | 32.6 |

QE 在所有三个数据集上均显著优于 baseline（p < 0.001），七种 LLM 趋势一致。

### 核心分析：Matched vs. Unmatched 性能对比（GPT-4o-mini）

| 条件 | FEVER Recall@5 | SciFact Recall@5 | AVeriTeC METEOR |
|------|:-:|:-:|:-:|
| **Query2doc ALL** | 36.4 | 55.1 | 19.1 |
| Matched (M) | **40.5** | **63.3** | **21.6** |
| Unmatched (¬M) | 23.8 | 45.9 | 17.4 |
| **BM25 baseline** | **31.0** | **51.2** | **17.8** |

| 条件 | FEVER Recall@5 | SciFact Recall@5 | AVeriTeC METEOR |
|------|:-:|:-:|:-:|
| **HyDE ALL** | 37.3 | 61.2 | 18.7 |
| Matched (M) | **40.0** | **68.4** | **19.8** |
| Unmatched (¬M) | 23.4 | 50.8 | 16.4 |
| **Contriever baseline** | **26.8** | **55.1** | **17.6** |

### 知识泄露比例（匹配占比，Table 3 摘要）

| LLM | FEVER (Q2d/HyDE) | SciFact | AVeriTeC |
|-----|:-:|:-:|:-:|
| GPT-4o-mini | 75.8% / 83.5% | 52.8% / 59.1% | 40.4% / 68.0% |
| Llama-3.1-70b | 78.3% / 71.7% | 57.5% / 55.0% | 48.1% / 47.0% |

### 关键发现

1. **知识泄露普遍存在**: 在多数情况下，超过 40% 的 claim 的扩展文档包含与 gold evidence 语义一致的句子，FEVER 上高达 83.5%。
2. **性能提升来源于 matched 样本**: Matched 组性能显著高于整体和 unmatched 组（p < 0.001），unmatched 组在多数情况下甚至低于不使用 QE 的 baseline。
3. **趋势跨模型/数据集一致**: 七种 LLM × 三个数据集 × 两种 QE 方法，结论高度一致。
4. **实际应用警示**: 对于涉及新知识或小众知识的 claim，QE 方法可能不仅无效甚至有害。

## 亮点与洞察

1. **提出了一个极具价值的"反直觉"问题**: 挑战了 HyDE/Query2doc 被广泛接受的核心假设，学术勇气可嘉。
2. **方法论简洁有效**: NLI-based matching 算法简单直观，但能精准量化知识泄露程度。
3. **实验设计严谨**: 7 个 LLM × 3 个 benchmark × 2 种 QE × 8 次重复 × matched/unmatched 分层分析，覆盖全面。
4. **对社区的警示意义**: 提醒研究者在评估 LLM-based retrieval 方法时需考虑数据污染/知识泄露的影响，推动更公正的 benchmark 设计。

## 局限与展望

1. **因果关系未建立**: 仅观察到相关性（LLM 行为与泄露的关联），未证明"训练数据→生成"的因果链。
2. **NLI 判断质量**: 依赖 GPT-4o-mini 做 NLI 标注，本身可能引入偏差；虽有人工验证但规模有限。
3. **任务范围受限**: 仅在事实验证任务上验证，是否推广到 QA、对话检索等其他检索密集型任务未知。
4. **缺乏解决方案**: 本文以分析为主，未提出缓解知识泄露的具体方法。
5. **未探讨 QE 在"真正新知识"上的改造潜力**: 如果结合外部知识源扩展，QE 是否能恢复有效性？仅在 Discussion 中简要提及。

## 相关工作与启发

### vs. HyDE (Gao et al., 2023)
HyDE 假设生成的假设性文档即使有事实错误也能辅助检索。本文通过 NLI 分析证明 HyDE 的性能提升很大程度上依赖于 LLM 对 gold evidence 的记忆复述，而非假设性推理，动摇了 HyDE 的理论基础。

### vs. Data Contamination 研究（Deng et al., 2023; Xu et al., 2024）
已有数据污染研究通过困惑度、token 预测等方法检测 LLM 是否"见过"测试数据。本文首次将数据泄露问题引入 query expansion 领域，用 NLI 匹配作为检测手段，是数据污染研究在 IR 方向的自然延伸。

### vs. Query Expansion + 外部知识（Lei et al., 2024）
一些最新工作已开始引入外部知识源增强 QE。本文的发现为这类方法提供了强有力的动机——如果 LLM 内部知识是泄露所致，那么引入外部信息才是正途。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 对 QE 领域广泛接受的假设提出系统性挑战，研究问题极具洞察力
- **实验充分度**: ⭐⭐⭐⭐ — 7 个 LLM × 3 个数据集 × 2 种方法的全面覆盖，统计检验严格
- **写作质量**: ⭐⭐⭐⭐ — 问题阐述清晰，实验逻辑严密，Discussion 深入
- **价值**: ⭐⭐⭐⭐ — 对 IR 社区有重要警示意义，推动更严格的 benchmark 评估标准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Re-identification of De-identified Documents with Autoregressive Infilling](reidentification_deidentified.md)
- [\[ICLR 2026\] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](../../ICLR2026/information_retrieval/lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)
- [\[CVPR 2025\] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents](../../CVPR2025/information_retrieval/vdocrag_retrieval-augmented_generation_over_visually-rich_documents.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] Health-LLM: Personalized Retrieval-Augmented Disease Prediction System](health-llm_personalized_retrieval-augmented_disease_prediction_system.md)

</div>

<!-- RELATED:END -->
