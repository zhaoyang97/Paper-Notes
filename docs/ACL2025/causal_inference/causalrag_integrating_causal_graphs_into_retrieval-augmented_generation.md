---
title: >-
  [论文解读] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
description: >-
  [ACL 2025][因果推理][检索增强生成] 提出 CausalRAG，将因果图集成到 RAG 的检索过程中——从文档构建文本图并识别因果关系，在查询时通过因果路径发现和因果摘要生成来检索上下文，在文档问答中显著提升上下文精度（92.86%）和检索召回率。 领域现状：RAG 通过检索外部知识来增强 LLM 的事实性…
tags:
  - "ACL 2025"
  - "因果推理"
  - "检索增强生成"
  - "因果图"
  - "知识图谱"
  - "文档问答"
  - "图索引"
---

# CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2503.19878](https://arxiv.org/abs/2503.19878)  
**代码**: [https://github.com/Pwnb/CausalRAG](https://github.com/Pwnb/CausalRAG)  
**领域**: RAG / 因果推理  
**关键词**: 检索增强生成, 因果图, 知识图谱, 文档问答, 图索引

## 一句话总结
提出 CausalRAG，将因果图集成到 RAG 的检索过程中——从文档构建文本图并识别因果关系，在查询时通过因果路径发现和因果摘要生成来检索上下文，在文档问答中显著提升上下文精度（92.86%）和检索召回率。

## 研究背景与动机

**领域现状**：RAG 通过检索外部知识来增强 LLM 的事实性。标准 RAG 基于文本分块+语义相似度检索，GraphRAG 则构建知识图谱辅助检索。

**现有痛点**：(1) 标准 RAG 的文本分块会破坏文档结构和因果链；(2) 语义相似度不等于因果相关性——相似词不意味着因果关系；(3) GraphRAG 在检索精度和召回率之间存在 tradeoff（Local 精度高但召回低，Global 反之）。

**核心矛盾**：很多复杂问题需要沿因果链推理（A 导致 B 导致 C），但语义相似度检索无法感知这种链式关系。

**本文目标**：如何让 RAG 系统理解并利用文档中的因果关系进行检索？

**切入角度**：在图索引阶段识别因果关系（用 LLM），在查询阶段发现相关因果路径，生成因果摘要作为检索上下文。

**核心 idea**：图索引 → 因果路径发现 → 因果摘要，三步实现因果感知的 RAG。

## 方法详解

### 整体框架
三个主要步骤：(1) **图索引**：从文档中提取实体和关系，构建文本图，用 LLM 识别因果关系；(2) **因果路径发现**：查询时找到与问题相关的因果路径；(3) **因果上下文检索**：沿因果路径生成因果摘要，作为 LLM 的生成上下文。

### 关键设计

1. **图索引与因果关系识别**:

    - 功能：从文档构建包含因果关系标注的知识图谱
    - 核心思路：用 LLM 提取实体和关系三元组，然后对每条关系用 LLM 判断是否为因果关系并标注方向
    - 设计动机：因果关系是文档中最有价值的推理链路，显式标注使检索可以优先沿因果路径

2. **因果路径发现**:

    - 功能：给定用户查询，在因果图中找到相关的因果路径
    - 核心思路：从查询实体出发，沿因果边扩展 k 跳，收集 s 个候选路径，用 LLM 选择最相关的因果路径
    - 设计动机：因果路径比孤立的三元组提供更完整的推理链，k 和 s 控制检索的深度和广度

3. **因果上下文检索与摘要**:

    - 功能：将因果路径转化为自然语言摘要，作为 LLM 的生成上下文
    - 核心思路：LLM 根据选中的因果路径生成结构化的因果摘要，保留因果链条的逻辑关系
    - 设计动机：相比直接拼接检索文本块，因果摘要更聚焦、更有逻辑性

### 损失函数 / 训练策略
- 无需训练——完全基于 LLM 的 in-context 能力做图构建、因果识别、路径选择和摘要生成
- 超参数 k（因果路径跳数）和 s（候选路径数）需要调优，最佳 k=3, s=3

## 实验关键数据

### 主实验（基于 OpenAlex 学术论文的 QA）

| 方法 | Answer Faithfulness↑ | Context Precision↑ | Context Recall↑ |
|------|---------------------|--------------------|-----------------|
| GraphRAG-Local | 78.18 | 89.18 | 41.54 |
| GraphRAG-Global | 55.27 | 66.67 | 47.22 |
| HippoRAG2 | 67.36 | 73.72 | 47.22 |
| **CausalRAG** | **78.00** | **92.86** | **49.46** |

### 消融实验（k 和 s 参数研究）

| k, s | 综合性能 |
|------|--------|
| k=1, s=1 | 0.534 |
| k=3, s=3 | 0.782 |
| k=5, s=5 | 0.824 |

### 关键发现
- **因果路径检索在精度和召回上同时优于 GraphRAG**: Context Precision 92.86（+3.68 vs GraphRAG-Local），Context Recall 49.46（+7.92 vs GraphRAG-Local）
- **文档越长，CausalRAG 优势越大**：从摘要（72.43）到全文（91.69），CausalRAG 在长文档上提升更显著
- **因果推理减少幻觉**：Answer Faithfulness 78.00，接近最佳的 GraphRAG-Local 78.18，但 precision 更高

## 亮点与洞察
- **"因果关系 > 语义相似度"用于检索**：这是核心洞察。很多 RAG 失败案例是因为检索到语义相关但因果无关的内容，CausalRAG 通过因果路径有效解决了这个问题
- **无需训练的即插即用方案**：完全基于 LLM 能力，不需要额外训练，易于部署
- **k 和 s 的渐进分析**：提供了检索深度/广度与性能的清晰 tradeoff 指导

## 局限与展望
- 依赖 LLM 内部知识做因果识别，在专业领域（医学/法律）可能不准确
- 因果路径识别需要额外 LLM 调用，引入计算开销
- 评估数据集规模较小（学术论文 QA），缺少大规模多领域验证
- 因果关系的识别质量直接影响下游性能，但缺少因果识别准确率的评估

## 相关工作与启发
- **vs GraphRAG**: GraphRAG 构建通用知识图谱，CausalRAG 专注因果子图，检索更有针对性
- **vs HippoRAG2**: HippoRAG2 模拟大脑记忆检索，CausalRAG 通过因果路径提供更结构化的推理
- **vs 标准 RAG**: 语义检索无法处理因果推理需求，CausalRAG 填补了这一空白

## 评分
- 新颖性: ⭐⭐⭐⭐ 因果图+RAG 的组合很有价值，但实现上主要依赖 LLM prompt
- 实验充分度: ⭐⭐⭐ 数据集小，评估指标有限
- 写作质量: ⭐⭐⭐⭐ 动机和方法阐述清晰
- 价值: ⭐⭐⭐⭐ 为 RAG 系统的因果感知检索提供了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](../../AAAI2026/causal_inference/i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[ACL 2025\] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)
- [\[NeurIPS 2025\] Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [\[ECCV 2024\] Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization](../../ECCV2024/causal_inference/integrating_markov_blanket_discovery_into_causal_representation_learning_for_dom.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)

</div>

<!-- RELATED:END -->
