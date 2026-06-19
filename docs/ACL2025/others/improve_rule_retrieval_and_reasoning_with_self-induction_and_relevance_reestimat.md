---
title: >-
  [论文解读] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate
description: >-
  [ACL 2025][规则检索] 针对规则检索中查询（具体实例化事实）与规则（抽象变量形式）之间的语义鸿沟，提出 SIAR（自归纳增强检索）和 R3（规则相关性重评估）两种方法，通过将查询映射到规则语义空间并重新评估规则相关性，显著提升规则检索和下游推理性能。 基于规则的推理是 LLM 的重要能力增强方式：先从经验中总结规则…
tags:
  - "ACL 2025"
  - "规则检索"
  - "语义对齐"
  - "自归纳"
  - "重排序"
  - "LLM推理"
---

# Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate

**会议**: ACL 2025  
**arXiv**: [2505.10870](https://arxiv.org/abs/2505.10870)  
**代码**: 无  
**领域**: 其他  
**关键词**: 规则检索, 语义对齐, 自归纳, 重排序, LLM推理

## 一句话总结

针对规则检索中查询（具体实例化事实）与规则（抽象变量形式）之间的语义鸿沟，提出 SIAR（自归纳增强检索）和 R3（规则相关性重评估）两种方法，通过将查询映射到规则语义空间并重新评估规则相关性，显著提升规则检索和下游推理性能。

## 研究背景与动机

基于规则的推理是 LLM 的重要能力增强方式：先从经验中总结规则，再检索相关规则辅助推理。然而，现有研究主要关注规则的生成和应用，**规则检索**这一关键中间环节被严重忽视。

规则检索与传统知识检索有本质区别：
- **传统检索**: 查询和目标段落通常共享关键词或语义相似性（如"美国总统是谁" → 包含"美国总统"的段落）
- **规则检索**: 查询是具体的实例化事实（"加州环境法规定必须回收"），而规则是抽象变量形式（"如果法规 Y 适用于地区 Z，则地区 Z 的人 X 必须遵守法规 Y"），两者之间存在巨大语义鸿沟

作者通过实验展示了这个问题的严重性：使用标准检索方法检索规则辅助推理，反而导致性能下降（相比不用规则）。但如果直接提供正确规则（golden rule），7B 模型可提升 31.54%，72B 模型可提升 23.67%。这说明**不是规则没用，而是检索不准**。

## 方法详解

### 整体框架

方法嵌入标准的 retrieve-then-reason 流程：
1. **检索前**: SIAR 利用 LLM 自归纳生成假设规则，用于查询增强
2. **检索后 / 推理前**: R3 利用 LLM 重新评估检索到的规则的相关性并重排序
3. **推理**: 用重排后的 top-k 规则辅助 LLM 推理

### 关键设计

#### 1. SIAR: 自归纳增强检索 (Self-Induction Augmented Retrieval)

**功能**: 在检索前，通过 LLM 的自归纳能力，从查询中生成一个假设性规则（self-induced rule, SI），然后用这个 SI 增强检索查询。

**核心思路**: 如果将查询集合和规则集合视为两个几乎不重叠的语义子空间——前者由实例化的具体事实构成，后者由抽象的概念知识构成——那么自归纳的作用就是将查询尽可能投射到规则子空间中、使查询能在检索时与具有相似底层逻辑的规则更好匹配。

**具体实现**: 使用 few-shot in-context learning 引导 LLM 对查询进行归纳抽象：
- 总结查询中的事实
- 抽象具体实体为变量
- 假设潜在的推理关系

**两种使用方式**:
- **SIAR (w/ SI)**: 仅用 SI 作为新查询
- **SIAR (w/ SI + input)**: 拼接 SI 和原始查询作为新查询

**设计动机**: 传统检索（无论稀疏还是密集）直接用原始查询匹配抽象规则，由于语义空间不匹配，效果很差。自归纳相当于做了一次"空间转换"，将查询从事实空间映射到规则空间。

#### 2. R3: 规则相关性重评估 (Rule Relevance ReEstimate)

**功能**: 对 SIAR 检索到的 top-n 规则列表，用 LLM 重新评估每条规则与原始查询的相关性，然后重排序并选取 top-k。

**核心思路**: 检索器只能评估语义相似度，无法判断规则是否真的能帮助推理。R3 通过让 LLM 评估两个维度来弥补：(1) 规则中的抽象知识能否被实例化为查询中的事实？(2) 规则是否有助于推理？

**具体实现**: 受 RankGPT 启发，直接 prompt LLM 输出重排后的规则列表（而非逐对比较），减少调用次数加速处理。

**设计动机**: SIAR 虽然能改善检索排名，但 LLM 的归纳能力有限，面对复杂查询仍可能产出不够准确的 SI。R3 提供了第二层保障，从应用层面（而非语义层面）评估规则质量。

### 方法特点

整个方法基于 prompting，**不需要任何训练**，通用性强，可以搭配不同的检索器（稀疏/密集/LLM检索器）和不同规模的 LLM 使用。

## 实验关键数据

### 主实验：检索性能（自然语言规则库）

| 方法 | CLUTRR R@1 | ULogic R@1 | CAIL2018 R@1 |
|------|-----------|-----------|-------------|
| Vanilla (sparse) | 6.67 | 68.91 | 25.30 |
| +SIAR (72B, SI+input) | 11.06 | 74.82 | 74.70 |
| +SIAR-R3 (72B, SI+input) | **14.31** | **92.17** | **86.14** |
| Vanilla (dense) | 2.10 | 30.36 | 9.04 |
| +SIAR (72B, SI) | 11.74 | 64.82 | 76.51 |
| +SIAR-R3 (72B, SI) | **14.03** | **88.19** | **81.32** |

### 主实验：推理性能

| 方法 | CLUTRR | ULogic | CAIL2018 |
|------|--------|--------|----------|
| Direct (无规则) | 38.36 | 93.01 | 80.12 |
| Golden Rule | 89.03 | 94.58 | 98.90 |
| Vanilla sparse retrieval | 37.60 | 93.13 | 73.49 |
| SIAR (sparse) | 49.14 | 94.21 | 86.14 |
| SIAR-R3 (sparse) | **51.71** | **95.90** | **86.75** |
| Vanilla dense retrieval | 30.53 | 90.00 | 72.89 |
| SIAR (dense) | 49.81 | 95.06 | 86.75 |
| SIAR-R3 (dense) | **51.05** | **95.78** | **84.94** |

（以上均为 Qwen2.5-72B-Instruct 的结果）

### 消融实验

| 因素 | 发现 |
|------|------|
| 开源 vs 闭源 LLM | Qwen2.5-72B 与 GPT-4o 表现相当，部分场景更优 |
| 72B vs 7B | 大模型的归纳和重排能力显著更强 |
| 稀疏 vs 密集检索 | 大多数场景稀疏检索更优（关键词匹配对规则更有效） |
| SI vs SI+input | 稀疏检索适合 SI+input，密集检索适合 SI only |
| 不同检索器 | BM25/BGE/BGE-Gemma2 均有显著提升 |
| 规则库翻倍 | 规则数量加倍后方法仍然有效 |

### 关键发现

1. **规则对推理的价值巨大但检索是瓶颈**: Golden rule 带来 23-31% 的提升，但 vanilla 检索反而降低性能
2. **SIAR 在密集检索上提升更大**: 密集检索 R@1 从 2.10 提升到 11.74（+9.64），说明密集检索受语义鸿沟影响更大
3. **R3 在简单数据集上更有效**: ULogic 和 CAIL2018 上 R3 带来巨大提升（最高 +43.25 R@1），但在 CLUTRR 上仅大模型获益
4. **稀疏检索通常优于密集检索**: 规则场景中很多概念在密集向量空间中表示不佳，关键词匹配反而更精准

## 亮点与洞察

- **问题定义清晰且重要**: 系统性地指出规则检索与传统知识检索的本质区别（具体 vs 抽象的语义鸿沟），是一个被忽视但关键的问题
- **理论直觉优雅**: "将查询从事实子空间映射到规则子空间"的思路简洁有力，类似于 HyDE 的思想但应用于规则场景
- **零训练、高通用性**: 纯 prompting 方法，可即插即用到任何 retrieve-then-reason 流程
- **分析深入**: 对不同检索器类型、规则格式、模型规模的交叉分析提供了丰富的实践指导

## 局限与展望

1. **规则库规模有限**: 最大仅 1048 条规则，与真实场景（如法律法规数万条）差距大
2. **依赖 LLM 归纳能力**: 小模型（7B）的自归纳质量明显不足，R3 也无法有效弥补
3. **计算成本**: SIAR 需要额外一次 LLM 推理，R3 需要再一次——对于大量查询场景可能不经济
4. **未探索规则组合**: 真实推理可能需要多条规则的链式组合，当前仅检索 top-k 独立规则
5. **可考虑训练轻量检索模型**: 用 SIAR 生成的 SI 作为训练信号，微调密集检索器以内化空间映射能力

## 相关工作与启发

- **HyDE (Gao et al.)**: 用 LLM 生成假设文档再检索，SIAR 可视为其在规则领域的变体
- **RankGPT**: R3 的重排思路受其启发，但评估标准从语义相关性扩展到推理有用性
- **ExpNote/Hypothesis Search**: 规则生成和应用的相关工作，本文聚焦被忽视的检索环节
- **启发**: 当检索目标与查询属于不同语义空间时，"先生成再检索"的范式具有普遍价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统研究规则检索问题，SIAR 的空间映射思路有理论美感
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多种检索器、多种模型规模的交叉实验覆盖全面
- **写作质量**: ⭐⭐⭐⭐ — Figure 1 清晰展示问题，方法描述简洁
- **价值**: ⭐⭐⭐⭐ — 指出重要问题并提供有效解决方案，但规则库规模限制了影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](tiser_timeline_self_reflection_temporal.md)
- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)
- [\[CVPR 2026\] NeuroRule: Bridging Vision and Logic with Differentiable Rule Induction](../../CVPR2026/others/neurorule_bridging_vision_and_logic_with_differentiable_rule_induction.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)

</div>

<!-- RELATED:END -->
