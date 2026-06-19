---
title: >-
  [论文解读] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation
description: >-
  [ACL 2025][信息检索/RAG][RAG] 提出 Amber 框架，通过 Agent 协作式记忆更新器、自适应信息收集器和多粒度内容过滤器三个组件协同工作，在迭代式 RAG 范式中提升开放域问答的检索效率和答案质量。 检索增强生成（RAG）通过整合外部知识库来增强模型的响应准确性并缓解幻觉问题。然而…
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "RAG"
  - "自适应检索"
  - "记忆更新"
  - "多Agent协作"
  - "多粒度过滤"
---

# Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2504.05312](https://arxiv.org/abs/2504.05312)  
**代码**: 有 ([https://anonymous.4open.science/r/Amber-B203/](https://anonymous.4open.science/r/Amber-B203/))  
**领域**: NLP / 检索增强生成  
**关键词**: RAG, 自适应检索, 记忆更新, 多Agent协作, 多粒度过滤

## 一句话总结

提出 Amber 框架，通过 Agent 协作式记忆更新器、自适应信息收集器和多粒度内容过滤器三个组件协同工作，在迭代式 RAG 范式中提升开放域问答的检索效率和答案质量。

## 研究背景与动机

检索增强生成（RAG）通过整合外部知识库来增强模型的响应准确性并缓解幻觉问题。然而，现有 RAG 方法在开放域 QA 任务中存在三个核心问题：

**缺乏记忆机制**：每次检索独立进行，缺少对前序检索信息的汇总记忆，导致生成结果只反映某次检索的片段知识

**检索策略不自适应**：LLM 在使用检索片段进行推理时，不能主动评估信息的有效性，也无法根据已知信息决定何时停止检索或更新检索查询

**噪声干扰**：检索到的文本中有效部分很少，大量冗余信息引入噪声，遮蔽关键细节

这些问题在复杂的多跳 QA 和长文本 QA 任务中尤为突出，因为这些任务需要跨多个文档聚合和综合信息。

## 方法详解

### 整体框架

Amber 是一个自适应记忆更新的迭代式 RAG 框架，由三个核心组件协同工作：
- **Agent-based Memory Updater (AMU)**：通过多 Agent 协作方式整合和优化 LLM 的记忆
- **Adaptive Information Collector (AIC)**：作为主调度器控制整个 RAG 工作流，动态调整检索查询并决定何时停止
- **Multi-granular Content Filter (MCF)**：在检索过程中进行多粒度内容过滤以减少噪声

工作流程：给定查询 q，初始化空记忆 M₀ → 每轮迭代中检索 top-k 文本块 → MCF 过滤 → AMU 更新记忆 → AIC 评估是否充分 → 若不充分则生成新查询进入下一轮 → 最终用记忆 Mₜ 通过 ICL 生成答案。

### 关键设计

1. **Agent-based Memory Updater (AMU)**：

    - 由三个独立 Agent 组成对话式协作：**Reviewer**（评估者）、**Challenger**（挑战者）、**Refiner**（精炼者）
    - Reviewer 评估记忆更新的正确性和相关性
    - Challenger 识别潜在缺陷和被忽视的约束
    - Refiner 综合前两者的反馈进行具体修改
    - 设计动机：单一 Agent 更新容易遗漏信息或引入偏差，多 Agent 的对抗式审查确保记忆质量

2. **Adaptive Information Collector (AIC)**：

    - 每轮迭代三步：检索 top-k 文本块 → AMU 更新记忆 → 评估记忆是否足以回答查询
    - 若不充分则生成精炼查询 q_{t+1} = AIC(q, q_t, m_{t+1})
    - 设计动机：避免过度检索（浪费计算资源）和不足检索（信息不完整）

3. **Multi-granular Content Filter (MCF)**：

    - **两级过滤**：先在 chunk 级别判断整块是否与查询相关；再在 sentence 级别从相关 chunk 中筛选出关键句
    - 使用 STRINC、CXMI 指标和 GPT-4 生成训练数据，对 LLM 进行多任务学习微调
    - 设计动机：检索到的文本中噪声比例很高，直接使用会干扰记忆更新和最终回答

### 损失函数 / 训练策略

- MCF 通过多任务学习微调 LLM，同时训练 chunk 级别和 sentence 级别的过滤能力
- 迭代过程中使用 zero-shot ICL 生成最终答案
- 基础 LLM 使用 Qwen2-7b、Llama3-8b 和 GPT-3.5

## 实验关键数据

### 主实验（表格）

| 方法 | SQuAD (acc/f1) | NQ (acc/f1) | TriviaQA (acc/f1) | 2WikiMQA (acc/f1) | HotpotQA (acc/f1) | ASQA (str-em/str-hit) |
|------|--------------|-------------|-------------------|-------------------|-------------------|-----------------------|
| No Retrieval | 12.6/18.4 | 24.0/27.5 | 49.8/52.7 | 28.4/35.6 | 19.8/25.2 | 35.5/8.9 |
| Vanilla RAG (GPT-3.5) | 34.4/37.9 | 35.9/38.4 | 63.8/63.5 | 35.4/38.2 | 38.6/44.4 | 47.8/21.6 |
| Adaptive-RAG | 33.0/38.3 | 44.6/47.3 | 58.2/60.7 | 46.4/49.8 | 44.4/52.6 | 42.1/15.8 |
| **Amber (GPT-3.5)** | **35.8/39.1** | **47.4/52.0** | **66.8/66.1** | **46.7/46.0** | **47.4/53.6** | **51.3/26.3** |

### 消融实验（表格）

| 组件 | SQuAD acc | NQ acc | 2WikiMQA acc | HotpotQA acc | ASQA str-em |
|------|-----------|--------|-------------|-------------|-------------|
| Amber (完整) | 35.8 | 47.4 | 46.7 | 47.4 | 51.3 |
| - AMU | 下降 | 下降 | 下降 | 下降 | 下降 |
| - AIC | 下降 | 下降 | 下降 | 下降 | 下降 |
| - MCF | 下降 | 下降 | 下降 | 下降 | 下降 |

（消融实验验证了每个组件的有效性）

### 关键发现

1. **全面领先**：Amber 在所有六个数据集上均取得最优或次优性能，尤其在多跳 QA（2WikiMQA、HotpotQA）和长文本 QA（ASQA）上提升最为显著
2. **跨模型一致性**：无论使用 Qwen2-7b、Llama3-8b 还是 GPT-3.5 作为基础 LLM，Amber 框架都带来一致的性能提升
3. **多 Agent 协作的优势**：AMU 中的三 Agent 对话式协作显著优于单一 Agent 的记忆更新
4. **自适应停止的价值**：AIC 的自适应停止机制有效避免了过度检索带来的噪声和性能下降

## 亮点与洞察

- **记忆机制的引入**是核心创新：将 RAG 从"无状态检索"升级为"有状态的迭代式知识积累"
- 多 Agent 协作（Reviewer-Challenger-Refiner）的设计类似于学术同行评审流程，有助于确保记忆更新的质量
- 两级过滤（chunk → sentence）在保留信息的同时有效降噪，思路简洁有效
- AIC 基于当前记忆动态生成新查询的设计，解决了固定查询在多跳推理中信息不足的问题

## 局限与展望

- 多 Agent 对话的计算开销较大，每轮迭代需要多次 LLM 调用
- 迭代次数的控制依赖 AIC 的自适应判断，可能存在过早停止或过度迭代的风险
- MCF 的训练依赖 GPT-4 生成的标注数据，引入了对闭源模型的依赖
- 未在更大规模的 LLM（如 70B）上验证
- 检索器（Contriever）本身的召回率限制可能成为瓶颈

## 相关工作与启发

- 与 Self-RAG (Asai et al., 2023) 相比，Amber 不依赖自反馈标记而是使用显式的记忆管理
- 与 FLARE (Jiang et al., 2023) 的低置信度触发检索不同，Amber 通过结构化的 AIC 模块动态管理检索
- 启发：RAG 系统的发展方向是从"一次性检索-生成"走向"迭代式记忆积累"，记忆管理是未来 RAG 的核心机制

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 记忆更新的多Agent协作范式和两级内容过滤的设计有新意
- **实验充分度**: ⭐⭐⭐⭐ — 六个数据集覆盖单跳/多跳/长文本QA，三种基础LLM，与多种baseline对比
- **写作质量**: ⭐⭐⭐ — 框架描述较清楚但部分符号使用不够一致
- **价值**: ⭐⭐⭐⭐ — 对RAG领域的迭代式检索和记忆管理方向具有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)

</div>

<!-- RELATED:END -->
