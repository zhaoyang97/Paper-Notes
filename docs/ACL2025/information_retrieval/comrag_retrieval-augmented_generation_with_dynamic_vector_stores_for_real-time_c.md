---
title: >-
  [论文解读] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry
description: >-
  [ACL 2025][信息检索/RAG][检索增强生成] 提出ComRAG——一个面向工业实时社区问答的检索增强生成框架，通过静态知识向量库+高/低质量动态QA向量库：的三库架构和质心记忆机制：，在三个CQA数据集上获得向量相似度最高25.9%的提升，同时降低延迟8.7%-23.3%。 领域现状：社区问答(CQA)平台（如S…
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "检索增强生成"
  - "社区问答"
  - "动态向量存储"
  - "质心记忆机制"
  - "工业部署"
---

# ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry

**会议**: ACL 2025  
**arXiv**: [2506.21098](https://arxiv.org/abs/2506.21098)  
**代码**: 无  
**领域**: 信息检索  
**关键词**: 检索增强生成, 社区问答, 动态向量存储, 质心记忆机制, 工业部署

## 一句话总结

提出ComRAG——一个面向工业实时社区问答的检索增强生成框架，通过**静态知识向量库+高/低质量动态QA向量库**的三库架构和**质心记忆机制**，在三个CQA数据集上获得向量相似度最高25.9%的提升，同时降低延迟8.7%-23.3%。

## 研究背景与动机

**领域现状**：社区问答(CQA)平台（如Stack Overflow）是重要的知识库。现有方法分为检索式（从社区历史中选答案）和生成式（用LLM作"社区专家"直接回答）。

**现有痛点**：
   - **忽视外部领域知识**：纯CQA方法缺乏专业文档支撑
   - **静态视角**：不能应对问题的实时流入，无法利用动态积累的历史QA
   - **缺乏记忆机制**：历史QA不断增长但没有有效的存储管理策略

**核心矛盾**：实时CQA需要同时利用**静态领域知识**和**质量参差不齐的动态历史QA**，且必须控制存储增长以适应工业部署。

**本文目标**：(Q1) 如何融合静态知识和动态历史QA？(Q2) 如何管理快速增长的历史数据和参差不齐的回答质量？

**切入角度**：在RAG框架上扩展出三种向量存储（静态知识库 + 高质量CQA库 + 低质量CQA库），配合质心聚类的记忆管理。

**核心 idea**：用高/低质量分离的动态向量库+质心记忆实现高效实时社区问答。

## 方法详解

### 整体框架

ComRAG由三个向量库和三种查询路径组成：
- **静态知识向量库 $\mathcal{V}_{\text{kb}}$**：领域文档的embedding
- **高质量CQA向量库 $\mathcal{V}_{\text{high}}$**：评分≥$\gamma$ 的历史QA对
- **低质量CQA向量库 $\mathcal{V}_{\text{low}}$**：评分<$\gamma$ 的历史QA对

### 关键设计

1. **质心记忆机制(Centroid-Based Memory)**：

    - 将相似历史问题聚为 $m$ 个簇 $\{C_1, C_2, \dots, C_m\}$
    - 每个簇维护质心 $\mathbf{c} = \frac{1}{|C|}\sum_{q_i \in C} \text{Emb}(q_i)$
    - 新问题到来时：若与某簇质心相似度 $\geq \tau$，则归入该簇并更新质心；否则创建新簇
    - **同簇替换策略**：若新问题与簇内已有问题相似度 $> \delta$ 且新答案质量更高，则替换旧的
    - 设计动机：**控制存储增长**——只保留每簇的代表性问题，避免内存溢出

2. **三种查询路径**：

    - **路径①**：若高质量库中存在极相似问题（$\text{CosSim} \geq \delta$），直接复用历史答案
    - **路径②**：若有中等相似度的高质量历史QA（$\tau \leq \text{CosSim} < \delta$），用其作为参考生成答案
    - **路径③**：若高质量库无匹配，从静态知识库检索文档 + 从低质量库检索历史QA（作为**反面参考**，提示LLM避免类似错误）

3. **自适应温度调节**：

    - 根据检索到的历史答案评分的方差动态调整生成温度
    - 评分方差低（历史答案一致）→ 高温度鼓励探索
    - 评分方差高（历史答案参差）→ 低温度保证一致性
    - 公式：$T(\Delta) = |\exp(-k \cdot \min_{1 \leq i \leq l-1}(s_{i+1} - s_i))|_{[T_{min}, T_{max}]}$
    - 超参数：$k=250$，$T_{min}=0.7$，$T_{max}=1.2$

### 评分与更新

- 用BERT-Score作为答案质量评分器
- 每次生成答案后评分，按阈值 $\gamma$ 分入高质量或低质量CQA库

## 实验关键数据

### 主实验（三个数据集）

| 方法 | MSQA SIM | MSQA Avg Time | ProCQA SIM | ProCQA Avg Time | PolarDBQA BERT-Score |
|------|----------|---------------|------------|-----------------|---------------------|
| Raw LLM | 80.58 | 12.70s | 74.88 | 12.77s | 60.34 |
| Vanilla RAG | 80.73 | 13.86s | 75.59 | 16.97s | 64.78 |
| RAG+DPR | 80.50 | 14.08s | 74.83 | 13.79s | 66.55 |
| LLM+EXP | 76.70 | 20.23s | 67.78 | 22.69s | 67.00 |
| **ComRAG** | **94.70** | **11.60s** | **95.31** | **10.42s** | **67.39** |

- ComRAG在SIM指标上提升最高达**25.9%**（ProCQA: 74.88→95.31）
- 延迟降低8.7%-23.3%
- BERT-Score和BLEU/ROUGE-L均有竞争力

### 消融实验（PolarDBQA，10轮迭代）

- 去掉高质量CQA库：延迟增加4.9s，BERT-Score下降2.6
- 去掉质心记忆机制：延迟增加2.2s，BERT-Score下降0.5
- 去掉静态知识库和自适应温度：可直接复用答案的比例显著下降

### 效率与存储增长

- 随迭代进行，查询延迟持续下降：ProCQA从10.42s降到4.95s（-52.5%）
- BERT-Score随迭代持续提升：MSQA上累计提升2.25%
- 存储增长率从首轮的20.23%快速降至第10轮的2.06%（ProCQA）

### 关键发现

- **高质量CQA库贡献最大**：移除后对延迟和质量影响最大
- **质心记忆机制有效控制存储膨胀**：增长率 20.23% → 2.06%
- **低质量库作为反面教材有独特价值**：通过路径③引导LLM避开已知错误
- **系统随使用时间"变聪明"**：积累的历史QA不断提升回答效率和质量

## 亮点与洞察

- **三库架构的设计巧妙**：高质量用于复用/参考，低质量用于对比回避，静态库补充专业知识
- **质心记忆是工业可部署的核心**：不是简单的"全部存下来"，而是有选择地聚类替换
- **自适应温度调节**：虽然设计简单但直觉合理——历史答案一致时鼓励多样性，不一致时收紧
- **真正面向工业落地**：考虑了延迟、存储增长等实际部署问题

## 局限与展望

1. **阈值固定**：$\tau$、$\delta$、$\gamma$ 为手动设定的固定值，缺乏自适应调整
2. **低质量QA的处理太简单**：仅通过prompt告诉LLM"避免类似答案"，可以考虑更高级的过滤或修正机制
3. **路由策略是基于规则的**：三条查询路径的选择基于硬阈值，可引入学习式路由
4. **质心记忆不考虑时效性和使用频率**：冷门簇可能长期占用存储
5. **评测场景有限**：PolarDBQA是自建数据，且只用了一个LLM backbone (Qwen2.5-14B)

## 相关工作与启发

- **与标准RAG的区别**：RAG通常只有静态语料库；ComRAG的核心贡献是引入动态质量分级的历史QA管理
- **与LLM+EXP(MSQA原方法)比较**：ComRAG虽然框架更复杂但延迟反而更低，因为直接复用高质量答案跳过了生成步骤
- **启发**：对于任何需要持续回答相似问题的场景（客服、技术支持），"答案复用+质量分级"都是有价值的设计范式

## 评分

- **新颖性**: ⭐⭐⭐ — 高低质量分离和质心记忆有新意，但整体是RAG的工程化扩展
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集+消融+迭代评估+存储分析，覆盖全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，形式化完整
- **价值**: ⭐⭐⭐⭐ — 工业场景实用性强，阿里巴巴的PolarDB场景有真实落地背景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering](graf_graph_retrieval_augmented_by_facts_for_romanian_legal_multi-choice_question.md)
- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[NeurIPS 2025\] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](../../NeurIPS2025/information_retrieval/cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)

</div>

<!-- RELATED:END -->
