---
title: "If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?"
conference: "ACL 2025"
arxiv: "2502.11469"
code: "https://github.com/osekilab/TG-NAE"
domain: "others"
keywords: ["attention mechanism", "cognitive model", "memory retrieval", "Transformer Grammar", "psycholinguistics"]
---

# If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?

## 一句话总结

通过将 Transformer Grammar（TG）的注意力机制与人类阅读时间数据关联，首次证明在句法结构上操作的注意力比在 token 序列上操作的普通 Transformer 注意力能更好地预测人类阅读行为，揭示人类句子处理涉及"句法结构+词序列"的双重记忆表征。

## 研究背景与动机

1. 计算心理语言学近期发现注意力机制与人类记忆检索（cue-based retrieval）之间存在有趣的平行关系
2. 现有研究主要关注在 token 级表征上操作的普通 Transformer，忽视了句法结构的作用
3. 心理语言学长期研究表明，句法结构能解释 token 级因素无法解释的人类句子处理现象
4. 核心研究问题：如果注意力可以作为记忆检索的通用算法，那么它在句法结构上操作是否也能模拟人类记忆检索？
5. Transformer Grammar（TG）是一种在句法结构上操作注意力的模型，提供了天然的实验工具
6. 此前没有工作系统研究 TG 的注意力机制对人类阅读行为的预测能力

## 方法详解

### 整体框架

使用 Normalized Attention Entropy（NAE）作为连接模型和人类的 linking hypothesis，比较 TG 和 vanilla Transformer 的注意力对自定步速阅读时间（self-paced reading times）的预测能力。

### 关键设计

- **Transformer Grammar (TG)**: 一种句法语言模型，通过动作序列（开括号、终结符、闭括号）联合生成 token 序列和句法结构
- **COMPOSE 注意力**: TG 的核心创新——闭合短语时通过专用注意力机制生成短语向量表征，后续操作将其作为单一表征引用
- **STACK 注意力**: 在所有其他位置操作，注意力限制在栈上的元素（未闭合非终结符、未组合终结符和已闭合短语）
- **NAE 计算**: 对每个注意力头计算归一化注意力熵，除以最大熵并重归一化，衡量检索干扰程度
- **TG−comp 变体**: 去除 COMPOSE 注意力的 TG 变体，用于消融 COMPOSE 的贡献

### 实验设计

- **语言模型**: 16层8头 TG 和 Transformer（252M参数），在 BLLIP-lg（42M tokens）上训练
- **阅读时间数据**: Natural Stories 语料库（10个故事，10,245词，181名母语者的自定步速阅读时间）
- **统计分析**: 线性混合效应模型，控制词长、n-gram频率、surprisal、栈计数等基线因素
- **评估指标**: ΔLogLik（对数似然改善量），衡量 NAE 对阅读时间预测的贡献

## 实验关键数据

### 主实验：NAE 对阅读时间的预测贡献

| 模型 | ΔLogLik | 当前词效应(ms) | 溢出效应(ms) | 显著种子 |
|------|---------|---------------|-------------|---------|
| TG | 76.6 (±8.1) | 1.42 (±0.2) | 2.26 (±0.1) | 3/3 |
| Transformer | 42.8 (±9.5) | 1.32 (±0.2) | 1.46 (±0.2) | 3/3 |

### 独立性检验

- TG+Transformer 联合模型 > 仅 TG（p<0.001），说明 Transformer NAE 解释了 TG 无法捕获的方差
- TG+Transformer 联合模型 > 仅 Transformer（p<0.001），说明 TG NAE 有独立贡献

### COMPOSE 注意力消融

| 模型 | ΔLogLik |
|------|---------|
| TG | 46.1 (±9.1) |
| TG−comp | 18.1 (±9.3) |

- TG 显著优于 TG−comp（p<0.001），COMPOSE 注意力是 TG 优势的关键来源
- TG−comp 对 TG 没有额外贡献（p=0.478），说明 TG 已涵盖 TG−comp 的信息

### 词性分析发现

- TG 在动词类词性（VB, VBG, VBN, VBP）上显著优于 Transformer
- Transformer 在名词类词性（NN, NNP）上优于 TG
- 这与心理语言学文献一致：动词触发的检索依赖句法特征，名词触发的检索依赖语义特征

### 干扰效应 vs 衰减效应

- TG NAE 与 Category Locality Theory（衰减效应模型）的贡献相互独立（双向 p<0.001）
- 首次提供广覆盖证据：NAE 捕获的是干扰效应（interference）而非衰减效应（decay）

## 亮点与洞察

- **双重记忆表征假说**: 人类句子处理涉及两种记忆表征——基于句法结构的和基于词序列的，注意力是通用检索算法
- **COMPOSE 是关键**: TG 优势的核心来源是 COMPOSE 操作——将闭合短语压缩为单一表征，而非简单地考虑句法结构
- **动词 vs 名词的互补性**: TG 更擅长捕获动词触发的句法检索，Transformer 更擅长名词触发的语义检索
- **从计算层到算法层**: 将认知建模从 Marr 的计算层（surprisal 理论）推进到算法层（记忆表征和检索机制）
- **NLP工程→认知科学**: 注意力机制虽出自工程目的，却能作为认知科学中cue-based retrieval的计算实现

## 局限性

- NAE 计算方式（顶层取、头间求和、子词聚合）沿用前人工作，其他方案未探索
- 仅使用英语自定步速阅读数据，跨语言和其他认知量度（眼动、EEG、fMRI）的泛化性未验证
- 假设完美句法结构（"perfect oracle"），局部歧义和增量解析未纳入考虑
- 采用自顶向下解析策略，而心理语言学认为左角解析可能更符合人类句子处理

## 相关工作

- 注意力与记忆检索的平行关系（Ryu & Lewis, 2021; Oh & Schuler, 2022）
- Cue-based retrieval 理论（Van Dyke & Lewis, 2003）
- Transformer Grammar（Sartran et al., 2022）
- 句法语言模型与人类认知（Hale et al., 2018; Wolfman et al., 2024）
- Surprisal 理论和期望理论（Hale, 2001; Levy, 2008）

## 评分

- **新颖性**: ★★★★★ — 首次将 TG 注意力与人类记忆检索关联，提出双重记忆表征假说
- **技术深度**: ★★★★☆ — 实验设计严谨，统计方法规范，消融充分
- **实验充分性**: ★★★★☆ — 多角度分析覆盖了主效应、独立性、消融和词性分析，但仅限英语
- **实用价值**: ★★★☆☆ — 偏基础认知科学研究，对 NLP 工程的直接启示有限
