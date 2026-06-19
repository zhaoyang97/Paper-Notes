---
title: >-
  [论文解读] Redundancy Principles for MLLMs Benchmarks
description: >-
  [多模态VLM] 本文从维度冗余、实例冗余和跨基准冗余三个层面系统量化了当前MLLM评测基准中的冗余现象，提出了基于性能排名相关性的冗余分析框架，为未来基准设计提供了原则性指导。 - 随着多模态大语言模型（MLLMs）的快速迭代，每年涌现出数百个评测基准（benchmark），但这些基准之间存在大量重叠和冗余 - 冗余导致评…
tags:
  - "多模态VLM"
---

# Redundancy Principles for MLLMs Benchmarks

| 属性 | 内容 |
|------|------|
| 标题 | Redundancy Principles for MLLMs Benchmarks |
| 会议 | ACL2025 |
| arXiv | [2501.13953](https://arxiv.org/abs/2501.13953) |
| 代码 | - |
| 领域 | Multimodal VLM / Benchmark Evaluation |
| 关键词 | MLLM, Benchmark Redundancy, Evaluation, Correlation Analysis, VLMEvalKit |

## 一句话总结

本文从维度冗余、实例冗余和跨基准冗余三个层面系统量化了当前MLLM评测基准中的冗余现象，提出了基于性能排名相关性的冗余分析框架，为未来基准设计提供了原则性指导。

## 研究背景与动机

- 随着多模态大语言模型（MLLMs）的快速迭代，每年涌现出数百个评测基准（benchmark），但这些基准之间存在大量重叠和冗余
- 冗余导致评估效率低下，重复测试相似能力而不能提供有价值的新洞察
- 过度强调某些任务类型可能扭曲研究优先级
- 需要一个系统化的框架来量化冗余程度，并为基准设计提供指导原则

**核心问题**：当前MLLM基准中到底存在多少冗余？如何科学地设计新基准以减少不必要的冗余？

## 方法详解

### 整体框架：Performance Correlation Redundancy Framework

核心假设：当评估相似的能力时，MLLMs的性能排名应表现出强烈的相关性；排名差异大则说明被评估的能力相对独立。

基于此假设，通过测量MLLM性能排名的相关性来量化冗余度。数据来源于VLMEvalKit，包含100+个MLLMs在20+基准上的评估结果。

### 1. 维度冗余（Dimensions Redundancy）

对于一个包含 $m$ 个维度的基准，每个维度 $X_i$ 的冗余度定义为：

$$\rho(X_i) = \frac{1}{m-1}\sum_{j \neq i} \text{CORR}(R_i, R_j)$$

其中 $R_i$ 是 $N$ 个MLLMs在维度 $X_i$ 上的排名。基准整体内部冗余度为所有维度冗余度的平均值：

$$\rho_{BI} = \frac{1}{m}\sum_{i=1}^{m}\rho(X_i)$$

### 2. 实例冗余（Instances Redundancy）

从总共 $M$ 个实例中随机采样 $A\%$，计算采样后的MLLM排名与全量排名的相关性：

$$\rho(A\%) = \frac{1}{T}\sum_{t=1}^{T}\text{CORR}(R_{A\%_t}, R_{GT})$$

重复采样 $T=100$ 次取平均以减少随机性影响。高相关性意味着仅需少量实例即可代表整个基准。

### 3. 跨基准冗余（Cross-Benchmark Redundancy）

对同一领域内的 $l$ 个基准，第 $i$ 个基准的冗余度为：

$$\rho(Y_i) = \frac{1}{l-1}\sum_{j \neq i}\text{CORR}(K_i, K_j)$$

高冗余度表示该基准可作为领域内的"锚点基准"（anchor benchmark），代表多个其他基准。

### 相关性度量

采用三种指标：SRCC（Spearman秩相关系数）衡量排名一致性、PLCC（Pearson线性相关系数）衡量线性关系、R²（决定系数）衡量拟合优度。

### Top-K 分析

聚焦于Top-K高性能MLLM进行分析，因为顶级模型的性能更受研究社区关注。

## 实验

### 1. 维度冗余分析（以MMBench为例）

**Top-50 MLLMs**：
- Image Emotion 和 Social Relation 表现出强冗余，评测的技能高度重叠
- Structuralized Image-Text Understanding 与 Spatial Relationship、Physical Property Reasoning、OCR 等多个维度冗余
- Celebrity Recognition 保持相对独立，因为这是知识型任务而非感知型任务
- Nature Relation 和 Spatial Relationship 冗余最高，因为它们是许多其他任务的基础技能

**Bottom-50 MLLMs**：
- 几乎所有维度都表现出显著更高的冗余度（SRCC/PLCC > 0.6）
- 原因：弱模型的基础能力普遍不足，一个维度的改进往往带动其他维度同步提升
- 关键启示：冗余分析应避免纳入全面表现不佳的模型

### 2. 实例冗余分析

| 发现 | 说明 |
|------|------|
| 大多数基准有50%+实例冗余 | 以0.95 SRCC/PLCC为阈值，至少一半实例是多余的 |
| 排名vs绝对性能预测 | 可靠排名所需实例远少于精确性能预测（后者需要90%+实例才能R²>0.95）|
| 模型能力影响冗余 | Top-50需要更多实例，Bottom-50用更少实例即可排名 |
| 基准间差异显著 | RealWorldQA冗余最低（需80%实例才饱和），其他基准远少于此 |

### 3. 跨基准冗余分析（Math领域）

对MathVista、MathVision、MathVerse、DynaMath进行分析：
- MathVista冗余最低，因为包含30-40%非数学任务（General VQA、图表理解等"噪声"）
- MathVerse和MathVision冗余最高，聚焦标准数学任务
- 去除MathVista中的非数学任务后，其冗余度显著提高，与其他数学基准更一致

## 亮点与洞察

1. **首次系统量化MLLM基准冗余**：从三个层面（维度、实例、跨基准）建立了完整的冗余分析方法论
2. **模型能力与冗余的关系**：强模型使基准冗余降低，弱模型使冗余增加——这一发现对基准设计有重要指导意义
3. **实用性原则**：提出了基准设计的具体原则——领域内广泛评估的基准应与同领域基准高冗余，专精基准则应低冗余
4. **提效明确**：大多数基准可以将实例数量减半而不影响MLLM排名，这对节省计算资源非常重要
5. **锚点基准概念**：高冗余基准可作为领域代表，无需运行所有基准即可获得领域性能估计

## 局限性

1. 核心假设（相似能力排名高度相关）在某些情况下可能不成立，因为类似任务可能因微妙差异而导致性能分化
2. 相关性指标（SRCC/PLCC/R²）可能无法充分捕捉模型在不同任务和条件下的完整复杂性
3. 冗余分析结果依赖于选择哪些MLLMs参与计算，不同选择可能导致不同结论
4. 分析基于统计学方法，对小众或特定任务的基准需要个案分析

## 相关工作

- **传统VQA基准**：GQA、VQA-V2、VizWiz、TextVQA等，问题简单，不适合评估现代MLLM
- **新一代VQA基准**：MMBench、MMVet、MMMU等，更灵活但迭代过快导致冗余
- **领域特定基准**：数学（MathVista、MathVerse）、OCR、医疗、遥感等
- **评估工具**：VLMEvalKit提供了统一的评估框架和开源数据

## 评分 ⭐⭐⭐⭐

**优点**：提出了一个非常实用且系统化的框架来分析基准冗余，实验全面覆盖20+基准和100+模型，结论对社区有直接指导价值。

**不足**：方法本身较直接（基于排名相关性），缺乏更深层的能力建模；分析依赖于现有评测数据，无法预测新兴基准的冗余情况。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)

</div>

<!-- RELATED:END -->
