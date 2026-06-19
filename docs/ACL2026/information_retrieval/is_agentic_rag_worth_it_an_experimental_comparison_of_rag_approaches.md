---
title: >-
  [论文解读] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches
description: >-
  [ACL 2026][信息检索/RAG][检索增强生成] 本文在四个数据集上从用户意图处理、查询重写、文档精炼和底层 LLM 选择四个维度系统对比了 Enhanced RAG 和 Agentic RAG，发现两者各有优势——Agentic RAG 在意图路由和查询重写上更灵活，Enhanced RAG 在文档重排上更有效，而 Agentic RAG 的成本高达 3.3 倍。
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "检索增强生成"
  - "Agentic RAG"
  - "Enhanced RAG"
  - "查询重写"
  - "成本分析"
---

# Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches

**会议**: ACL 2026  
**arXiv**: [2601.07711](https://arxiv.org/abs/2601.07711)  
**代码**: 无  
**领域**: 信息检索 / RAG  
**关键词**: 检索增强生成, Agentic RAG, Enhanced RAG, 查询重写, 成本分析

## 一句话总结

本文在四个数据集上从用户意图处理、查询重写、文档精炼和底层 LLM 选择四个维度系统对比了 Enhanced RAG 和 Agentic RAG，发现两者各有优势——Agentic RAG 在意图路由和查询重写上更灵活，Enhanced RAG 在文档重排上更有效，而 Agentic RAG 的成本高达 3.3 倍。

## 研究背景与动机

**领域现状**：RAG 已从研究概念发展为生产级语言系统的核心组件。基础 Naïve RAG（检索+生成）存在多个局限性，催生了两种改进范式：Enhanced RAG（在固定流水线中添加路由、重写、重排等模块）和 Agentic RAG（由 LLM 作为智能体自主编排整个流程）。

**现有痛点**：尽管这两种范式被快速采用，但缺乏系统的实证比较——在什么条件下应选择哪种方案？性能和成本的权衡如何？现有工作要么停留在理论定义层面，要么仅在单一设定下评测。

**核心矛盾**：Agentic RAG 提供了更大的灵活性（动态决策、迭代检索），但这种灵活性是否能转化为实际性能提升？额外的推理成本是否值得？

**本文目标**：在四个维度（用户意图处理、查询重写、文档列表精炼、底层 LLM 变更）上进行受控实验，量化两种范式的性能和成本差异。

**切入角度**：将 Naïve RAG 的四个已知缺陷映射为四个评估维度，分别设计 Enhanced 和 Agentic 的实现方案进行对比。

**核心 idea**：通过严格受控的实验设计，将"Enhanced vs Agentic RAG"这个模糊的架构选择问题转化为可量化的、维度化的决策指南。

## 方法详解

### 整体框架

本文不提新模型，而是把"Enhanced vs Agentic RAG"这个模糊的架构选择拆成可量化的受控实验。做法是把 Naïve RAG（检索+生成）的四个已知缺陷映射成四个评估维度——用户意图处理、查询重写、文档列表精炼、底层 LLM 选择——在每个维度上各自实现一套 Enhanced 方案（固定流水线里插模块）和一套 Agentic 方案（LLM 自主编排），喂入相同的查询与知识库，在四个数据集上对比检索/生成质量并核算 token 与时延成本，最后落成一份"哪个维度该选哪种范式"的决策指南。

### 关键设计

**1. 用户意图处理（User Intent Handling）：先判断查询要不要触发检索，避免无谓召回**

生产环境里大量查询其实不需要检索，盲目召回既费成本又引噪声。Enhanced 方案用 semantic-router 框架，基于预定义的有效/无效查询示例做语义相似度分类；Agentic 方案则去掉人工示例，由智能体自主判断是否调用 RAG 工具。两者在 500 个有效 + 500 个无效查询上对比，考察的正是"无需人工构造示例"的灵活性能否换来更准的路由——结果 Agentic 在窄域更优，但在 FEVER 这类宽泛领域会过度触发检索。

**2. 查询重写（Query Rewriting）：弥合简短查询与长文档之间的语义/格式鸿沟**

用户查询往往是一句短问题，知识库文档却是长篇文本，格式落差直接拖垮检索匹配。Enhanced 方案强制走 HyDE——让 LLM 先生成一个假设性回答段落再拿去检索；Agentic 方案则把"是否重写、如何重写"的自由度交还给智能体，按需自适应。两者统一用 NDCG@10 衡量检索质量，从而把"固定重写策略"与"自适应重写"放在同一标尺上比较。

**3. 文档列表精炼（Document List Refinement）：从初检结果里挑出最相关子集、压掉噪声**

初始检索难免混入噪声或次优文档，需要二次精炼。Enhanced 方案用 ELECTRA 交叉编码器对 top-20 文档显式重排；Agentic 方案则允许智能体触发多轮检索、迭代改善上下文。把两种"精炼哲学"放在一起跑后，一个关键观察浮出水面：Agentic 仅有 10% 的情况选择重新检索，且其中 53% 的文档与上一轮完全相同——说明智能体一旦决策就很少回头，迭代检索在这一维度上反而不如显式重排。

**4. 底层 LLM 选择（Underlying LLM）：考察更换生成模型对两种范式的影响模式是否一致**

生成模型在 Agentic RAG 里身兼二职——既要写最终答案，又要在每一步决定调不调工具、要不要重检索，因此一个直觉是：换用更弱/更便宜的 LLM 可能更伤 Agentic 而非 Enhanced。本文用同一查询与同一检索上下文，分别喂给 Qwen3 的 0.6B/4B/8B/32B 四档模型，以"大模型答案优于小模型的次数占比"（经 Selene-70B 做 LLM-as-a-Judge 配对评判）来量化模型规模的收益。结果是两种范式随规模提升的曲线几乎重合——换底层 LLM 带来的增益在 Enhanced 与 Agentic 上模式一致，说明这一维度并不构成两种范式的分水岭，选型时可独立于范式之外考虑。

### 损失函数 / 训练策略

本文不涉及模型训练。Enhanced RAG 各模块直接复用预训练模型（OpenAI embedding、ELECTRA 重排器），Agentic RAG 基于 PocketFlow 框架实现；生成与决策统一交给上面四档 Qwen3，最终答案质量一律用 LLM-as-a-Judge（Selene-70B）评判。

## 实验关键数据

### 主实验

**用户意图处理 (F1)**

| 设定 | FIQA | FEVER | CQA-EN |
|------|------|-------|--------|
| Naïve | 66.7 | 66.7 | 66.7 |
| Enhanced | 95.7 | **87.9** | 96.6 |
| Agentic | **98.8** | 64.6 | **99.8** |

**查询重写 (NDCG@10)**

| 设定 | FIQA | NQ | FEVER | CQAD | AVG |
|------|------|----|-------|------|-----|
| Naïve | 45.3 | 43.7 | 66.2 | 45.8 | 50.3 |
| Enhanced | 43.5 | 43.9 | 81.1 | 42.8 | 52.8 |
| Agentic | 43.2 | **51.7** | **83.1** | 44.3 | **55.6** |

**文档精炼 (NDCG@10)**

| 设定 | FIQA | CQA-EN | AVG |
|------|------|--------|-----|
| Naïve | 45.0 | 46.0 | 45.5 |
| Enhanced (w/ rewriting) | **51.0** | **48.0** | **49.5** |
| Agentic | 43.4 | 44.4 | 43.9 |

### 消融实验

**成本对比**

| 指标 | Agentic vs Enhanced |
|------|-------------------|
| 输入 token | 3.3× |
| 输出 token | 1.9× |
| 时间 | 1.5× |

### 关键发现

- Agentic RAG 在用户意图判断上表现更好（无需人工构造示例），但在宽泛领域（如 FEVER）会过度使用检索
- Agentic RAG 的自适应查询重写平均比 Enhanced 高 2.8 NDCG@10，在开放域查询（NQ）上优势达 +7.8
- Enhanced RAG 的显式重排远优于 Agentic 的迭代检索——智能体一旦做出决策就很少改变
- 更换底层 LLM 对两种范式的影响模式一致——性能随模型规模同比提升

## 亮点与洞察

- 将模糊的"Enhanced vs Agentic"选择问题拆解为四个独立维度，每个维度给出清晰的建议
- 最终建议是混合方案：用 Agentic 做意图路由和查询重写，用 Enhanced 的重排器做文档精炼
- 成本分析非常实用——Agentic RAG 高达 3.3 倍的 token 消耗提醒从业者需要认真考虑成本

## 局限与展望

- 仅评估单工具 Agentic RAG（只有检索工具），多工具智能体的表现可能不同
- 大规模数据集（NQ、FEVER）上的 Agentic 实验因耗时过长（>7天）而被排除
- 评估使用 LLM-as-a-Judge，存在评估偏差
- 未探索更复杂的 Agentic 策略（如反思、规划等）

## 相关工作与启发

- **vs CRAG/Self-RAG**: 属于 Enhanced RAG 的范畴，本文将其纳入统一框架对比
- **vs LLM Agent 框架**: 本文使用最简单的单工具 Agentic 设计以保持可比性
- **启发**：实际部署中应根据领域特性混合使用两种范式，而非全盘采用一种

## 评分

- 新颖性: ⭐⭐⭐ 问题重要但方法上无创新，主要是实验对比
- 实验充分度: ⭐⭐⭐⭐ 四个数据集×四个维度×成本分析，但部分实验因成本受限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实用导向强，适合工程参考
- 价值: ⭐⭐⭐⭐ 为 RAG 架构选择提供了急需的实证指南

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2026\] 生物医学 RAG 中检索何时无效：大规模实证研究](when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
