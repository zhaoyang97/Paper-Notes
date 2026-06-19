---
title: >-
  [论文解读] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval
description: >-
  [NeurIPS 2025][医疗NLP][RAG] 提出一种基于"需求差距"分析的知识库增强框架，通过叠加真实用户数据（论坛帖子）与现有心理健康资源库来识别内容空白，并用定向增强策略以最少的文档增量达到接近完整语料库的 RAG 检索质量。 心理健康领域的检索增强生成（RAG）系统依赖于专业策划的知识库…
tags:
  - "NeurIPS 2025"
  - "医疗NLP"
  - "RAG"
  - "knowledge base augmentation"
  - "mental health"
  - "gap analysis"
  - "corpus expansion"
---

# Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval

**会议**: NeurIPS 2025  
**arXiv**: [2509.13626](https://arxiv.org/abs/2509.13626)  
**代码**: 无  
**领域**: 医疗NLP  
**关键词**: RAG, knowledge base augmentation, mental health, gap analysis, corpus expansion

## 一句话总结

提出一种基于"需求差距"分析的知识库增强框架，通过叠加真实用户数据（论坛帖子）与现有心理健康资源库来识别内容空白，并用定向增强策略以最少的文档增量达到接近完整语料库的 RAG 检索质量。

## 研究背景与动机

心理健康领域的检索增强生成（RAG）系统依赖于专业策划的知识库，但存在以下核心问题：

**知识库扩展成本高**：心理健康资源由专业治疗师撰写和审核，内容生产周期长、成本高

**用户需求与内容供给错位**：用户在论坛等平台表达的关切往往使用非正式、口语化的语言，而知识库内容以正式的心理教育材料为主，存在主题覆盖和表达风格的双重鸿沟

**盲目扩展效率低**：不加区分地随机增加文档不仅耗费资源，且检索质量提升有限

核心洞察是：**内容缺口可以通过叠加自然用户需求数据来发现** —— 当某些话题被用户频繁提及但知识库中缺乏对应内容时，就产生了需要填补的空白。

## 方法详解

### 整体框架

框架分为四个阶段：(1) 数据收集与预处理；(2) 内容缺口分析；(3) 合成文档生成与语料库增强；(4) 通过多个 RAG 管线进行自动化评估。

### 关键设计

1. **主题分类体系**：基于临床 CLICC 框架（Clinician Index of Client Concerns）的46个主题，进一步扩展为368个子主题。使用 GPT-4o 对用户查询和知识库文档分别标注主题和子主题标签。

2. **覆盖差距度量（Coverage Gap）**：借鉴 TF-IDF 思想，将"词-文档"映射为"子主题-查询"。用户提及频率（demand）为 TF，知识库文档覆盖度（supply）为 IDF：

$$\text{Gap}(t) = \frac{\log(1 + f_p(t))}{\max_w \log(1 + f_p(w))} \cdot \left[\log\left(\frac{D + c}{df(t) + c}\right)\right]^\alpha$$

其中 $\alpha = 1.5$ 放大弱覆盖主题的影响。

3. **有用性差距度量（Usefulness Gap）**：使用 LLM-as-a-Judge（GPT-4o-mini）评估查询-文档对的上下文相关性（1-50）和实际帮助性（1-50），取每个子主题的平均分并反转，使高分对应大缺口。

4. **混合度量**：50% 覆盖差距 + 50% 有用性差距，敏感性分析显示不同权重下语料组成变化不超过24.2%。

5. **定向 vs 非定向扩充**：根据混合差距分数按比例分配各子主题的文档配额（Directed），并与随机抽样同等规模的文档集（Non-Directed）进行对照。

### 数据与评估管线

- **知识库**：新加坡心理健康平台 mindline.sg 的387篇策划文档
- **用户数据**：let's talk 论坛1223条匿名帖子（80%训练，20%测试）
- **参考语料库**：从7个全球心理健康平台生成7640篇合成文档（GPT-4o-mini，平均16.1秒/篇，$0.0007/篇）
- **四个 RAG 管线**：Baseline、Hierarchical、Reranking、Query Transformation

## 实验关键数据

### 主实验：达到参考语料库 ~95% 性能所需的最小文档增量

| RAG 管线 | Directed 增量 | Non-Directed 增量 | Directed 文档数 | Non-Directed 文档数 | 文档节省比 |
|---|---|---|---|---|---|
| Baseline | +318% | +763% | 1230 | 2954 | 58.4% |
| Hierarchical | +74% | +403% | 288 | 1560 | 81.5% |
| Reranking | +74% | +318% | 288 | 1230 | 76.5% |
| Query Transformation | +42% | +232% | 162 | 898 | 81.9% |

### 缺口分析结果

| 排名 | 主题 | 子主题 |
|---|---|---|
| 1 | Depression | 自我批判与低自我价值 |
| 2 | Relationship | 信任侵蚀与边界问题 |
| 3 | Anxiety | 健康焦虑与疾病恐惧 |
| 4 | Relationship | 依恋不安全与情感疏离 |
| 5 | Emotional dysregulation | 快速情绪波动与反应性 |
| 6 | Family | 父母冲突与家庭紧张 |
| 7 | Depression | 社交孤立与断联 |
| 8 | Depression | 快感缺失与退缩 |
| 9 | Anxiety | 社交评价恐惧与回避 |
| 10 | Anxiety | 睡眠障碍 |

### 关键发现

- **Query Transformation 管线效果最佳**：仅增加42%文档（162篇）即可达到95%参考性能，因为查询改写弥补了用户口语化表达与知识库正式语言之间的鸿沟
- **定向扩充全面优于随机扩充**：在所有管线上，Directed 方案以更少文档达到更好质量
- 聚合到主题层面，最欠缺的五大类别为：成瘾（非药物/酒精）、抑郁、焦虑、家庭、种族/民族/文化关切

## 亮点与洞察

- **TF-IDF 思想迁移到需求-供给分析**的创意非常巧妙，将信息检索中的经典方法论用于内容缺口检测
- **双维度缺口度量**（覆盖 + 有用性）比单一维度更全面，一个主题可能有文档但内容过于笼统或专业化
- 实验设计严谨，使用10组不同规模的 Directed 和 Non-Directed 语料进行系统对比
- 在高风险领域（心理健康）中，强调合成文档仅为概念验证，实际内容仍需专家撰写

## 局限与展望

- **人群偏差**：论坛用户偏年轻化（学业压力、身份认同），不代表知识库目标的全年龄段用户
- **缺乏人工评估**：所有评估依赖 LLM-as-a-Judge，未经临床专家验证，可能忽略治疗框架或情感适当性等专业维度
- **合成文档质量**：未经医学审核，仅作为可行性验证
- 混合权重的选择（50/50）虽然敏感性分析显示影响不大，但在不同领域可能需要调整
- 未探讨动态反馈循环，即随着知识库更新，用户需求可能也在变化

## 相关工作与启发

- 可与 RAG 系统的主动学习结合，实现"检测缺口 → 生成 → 人工审核 → 部署"的闭环
- 框架具有领域通用性，可应用于法律、教育等其他高风险垂直领域的知识库建设
- 对于知识库运营者，提供了基于数据驱动的内容优先级排序方法

## 评分

- **新颖性**: ⭐⭐⭐ — TF-IDF 迁移有创意，但整体方法论较为直接
- **实验充分度**: ⭐⭐⭐⭐ — 4个管线 × 22个语料配置 = 88组实验，系统性强
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分
- **实用价值**: ⭐⭐⭐⭐ — 对知识库运营有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2025\] Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](../../ACL2025/medical_nlp/clinical_coding_eight_recommendations.md)
- [\[ICML 2025\] On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains](../../ICML2025/medical_nlp/on_the_vulnerability_of_applying_retrieval-augmented_generation_within_knowledge.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](../../ACL2026/medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)
- [\[ICLR 2026\] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](../../ICLR2026/medical_nlp/biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)

</div>

<!-- RELATED:END -->
