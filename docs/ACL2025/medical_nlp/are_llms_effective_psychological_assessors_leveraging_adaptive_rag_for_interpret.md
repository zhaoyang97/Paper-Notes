---
title: >-
  [论文解读] Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice
description: >-
  [ACL 2025][医疗NLP][心理健康筛查] 本文提出了一种基于问卷引导的心理健康筛查框架，通过自适应RAG从用户Reddit帖子中检索相关内容，再用LLM代为填写标准化心理量表（如BDI-II），在无需训练数据的情况下匹配或超越有监督方法的性能，同时提供了临床可解释的评估结果。 领域现状：心理健康问题日益严重…
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "心理健康筛查"
  - "自适应RAG"
  - "心理量表"
  - "抑郁症检测"
  - "可解释AI"
---

# Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice

**会议**: ACL 2025  
**arXiv**: [2501.00982](https://arxiv.org/abs/2501.00982)  
**代码**: [https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment](https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment)  
**领域**: 医疗NLP  
**关键词**: 心理健康筛查, 自适应RAG, 心理量表, 抑郁症检测, 可解释AI

## 一句话总结
本文提出了一种基于问卷引导的心理健康筛查框架，通过自适应RAG从用户Reddit帖子中检索相关内容，再用LLM代为填写标准化心理量表（如BDI-II），在无需训练数据的情况下匹配或超越有监督方法的性能，同时提供了临床可解释的评估结果。

## 研究背景与动机

**领域现状**：心理健康问题日益严重，COVID-19后抑郁症患者增加了28%。社交媒体是人们表达情感和寻求支持的重要平台，已有不少NLP研究利用社交媒体内容进行心理健康分析。但主流方法大多采用端到端分类（直接从文本预测心理状态），这种黑盒方式缺乏临床可解释性。

**现有痛点**：（1）直接用LLM分类心理状态的零样本/少样本效果不佳，难以匹配有监督方法。（2）端到端分类无法解释"为什么认为该用户有抑郁倾向"，不符合心理学诊断实践。（3）心理健康评估的标准化工具（如BDI-II问卷）在临床中使用广泛，但计算方法几乎绕开了这些久经验证的工具。

**核心矛盾**：心理学家通过标准化问卷系统评估患者状态，每个问题都有明确的临床含义。但NLP方法跳过了这个结构化评估步骤，直接从文本到诊断，失去了可解释性和临床对齐性。

**本文目标**：让LLM像心理学家一样，通过分析用户的社交媒体帖子来"代为填写"标准化心理问卷，将复杂的诊断任务分解为逐题评估，实现可解释且准确的心理健康筛查。

**切入角度**：将心理健康预测重新定义为 $\sum_i f(\text{Text}, \text{Item}_i) \to Y$——对每个问卷题目检索相关帖子并生成回答，汇总得分得到最终评估。

**核心 idea**：用自适应RAG为每道心理问卷题目检索最相关的用户帖子，然后让LLM基于检索内容为用户"代答"该题，将诊断逻辑锚定在标准化临床工具上。

## 方法详解

### 整体框架
对于每个用户，系统执行以下流程：（1）将用户所有Reddit帖子向量化；（2）对问卷的每道题目及其选项向量化作为查询；（3）通过自适应检索为每道题找到最相关的帖子；（4）将检索到的帖子作为上下文，LLM为该题打分；（5）汇总所有题目得分，得到最终的严重程度评估。

### 关键设计

1. **自适应零样本检索策略（ABIDE-ZS）**:

    - 功能：为每道问卷题目动态确定检索帖子的最优数量
    - 核心思路：对每道题的4个选项分别计算与用户帖子的语义相似度。使用ABIDE算法自动确定最优的检索数量 $k^*$，该算法通过检测嵌入空间中语义一致性区域的边界来确定最佳邻域大小。不同题目和不同用户的 $k^*$ 可能不同——有的题目能找到很多相关帖子，有的只有少数
    - 设计动机：固定检索数量不合理——有的用户发帖多与某题目高度相关，有的则很少提及。自适应检索确保每道题获得恰好足够多的高质量上下文

2. **基于问卷结构的LLM打分**:

    - 功能：让LLM根据检索到的帖子为心理问卷的每道题目打分
    - 核心思路：为每道题构建prompt，包含题目描述、选项说明和检索到的用户帖子。LLM被要求分析帖子内容与各选项的匹配程度，输出最可能的选项得分（0-3）。支持直接打分和Chain-of-Thought两种策略，后者要求LLM先解释推理过程再给出分数
    - 设计动机：逐题打分将复杂的总体评估分解为可管理的子问题，每个子问题都有明确的临床定义，LLM更容易做出准确判断

3. **多量表扩展**:

    - 功能：将框架从抑郁症扩展到自残、饮食障碍、病理性赌博等心理状态
    - 核心思路：更换不同的标准化问卷（BDI-II→SHI、SCOFF、DSM-V赌博诊断量表），同一套检索+打分流程无需修改即可应用于不同心理状况的筛查
    - 设计动机：框架的核心是"问卷引导"而非"疾病特定"，证明该范式的通用性

### 损失函数 / 训练策略
完全无监督——不需要任何训练数据。LLM使用零样本推理，检索模型使用预训练的稠密检索器（测试了10种不同的检索模型）。

## 实验关键数据

### 主实验

| 方法 | eRisk 2019 (RMSE↓) | eRisk 2020 (RMSE↓) | 训练数据 |
|------|-------------------|-------------------|---------|
| SOTA有监督方法 | 8.21 | 10.45 | 需要 |
| GPT-4o-mini直接分类 | 12.35 | 13.82 | 不需要 |
| GPT-4o-mini + 直接提示 | 10.8 | 12.1 | 不需要 |
| **Ours (Claude-3.5 + aRAG)** | **7.89** | **9.95** | **不需要** |
| Ours (Qwen-2.5-70B + aRAG) | 8.15 | 10.28 | 不需要 |
| Ours (DeepSeek-V3 + aRAG) | 8.32 | 10.51 | 不需要 |

### 消融实验

| 配置 | RMSE (eRisk 2020) | 说明 |
|------|-------------------|------|
| aRAG + Claude-3.5 | 9.95 | 最优配置 |
| 固定k=5检索 | 11.23 | 自适应k优于固定k |
| 无检索（仅LLM） | 13.82 | 无上下文的LLM效果差 |
| 直接分类（不走问卷） | 12.10 | 问卷引导优于直接诊断 |
| CoT vs 直接打分 | 10.12 vs 9.95 | 直接打分略优 |
| 最佳检索模型: sf-e5 | 9.95 | 在10种检索器中最优 |
| 最差检索模型: contriever | 11.45 | 检索质量影响显著 |

### 关键发现
- 问卷引导方法显著优于直接让LLM做抑郁分类（RMSE降低约28%），证明了逐题评估的分解策略的有效性
- 自适应检索数量（ABIDE-ZS）平均每题检索9-20篇帖子，远少于用户全部帖子数，有效避免了信息过载
- 闭源模型（Claude-3.5）表现最好，但开源70B模型（Qwen-2.5）也能达到接近SOTA的水平
- 同一框架可无缝扩展到自残、饮食障碍等其他心理状况，展现了问卷引导范式的通用性
- 检索模型的选择影响很大——最佳和最差检索器之间差1.5个RMSE点

## 亮点与洞察
- "将LLM变成心理学家"的核心洞察非常深刻——不是直接让LLM做诊断，而是让它遵循心理学家的诊断流程（逐题评估→汇总得分→判断严重程度），大幅提升了可解释性和准确性。
- 无监督方法超越有监督方法是一个重要发现——这意味着标准化问卷的结构化知识可以弥补训练数据的缺失。
- 框架本身是"知识引导的AI"的绝佳范例——将领域专家知识（心理问卷）编码为推理结构，比让AI从头学习高效得多。

## 局限与展望
- Reddit用户群体可能不代表一般人群，数据可能存在自选偏差
- LLM根据帖子"代替"用户填问卷存在伦理问题——用户并未同意被评估
- 某些问卷题目（如关于自杀意念）的帖子可能非常稀少，影响检索质量
- 未来可以结合纵向分析（用户状态随时间变化），而不仅是横截面评估

## 相关工作与启发
- **vs MentalBERT (Ji et al., 2022)**: MentalBERT是有监督模型，需要训练数据且不可解释；本文无监督且可解释
- **vs Rosenman et al. (2024)**: 他们让LLM"扮演"受访者填问卷，本文是根据实际帖子内容填写，更加有据可依
- **vs eRisk系列竞赛**: 本文首次在完全无监督设置下匹配竞赛最佳结果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将心理问卷作为LLM推理的结构化框架是极具创意的设计
- 实验充分度: ⭐⭐⭐⭐⭐ 6种LLM、10种检索器、4种心理状况、两个基准数据集
- 写作质量: ⭐⭐⭐⭐ 研究问题定义精准，方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 对AI辅助心理健康筛查具有重要方法论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](../../ACL2026/medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_nlp/counselbench_llm_mental_health_qa.md)

</div>

<!-- RELATED:END -->
