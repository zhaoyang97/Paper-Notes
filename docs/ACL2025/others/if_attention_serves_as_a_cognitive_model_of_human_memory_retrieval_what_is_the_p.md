---
title: >-
  [论文解读] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?
description: >-
  [ACL 2025][注意力机制] 本文探究 Transformer Grammar（TG）的注意力机制能否作为人类记忆检索的认知模型，通过 Normalized Attention Entropy（NAE）将模型与人类阅读时间关联，发现基于句法结构的注意力比基于 token 的注意力更能解释人类句子处理行为，且两者提供独立互补的贡献。
tags:
  - "ACL 2025"
  - "注意力机制"
  - "人类记忆检索"
  - "Transformer"
  - "句法结构"
  - "阅读时间预测"
---

# If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?

**会议**: ACL 2025  
**arXiv**: [2502.11469](https://arxiv.org/abs/2502.11469)  
**代码**: [有 (GitHub)](https://github.com/osekilab/TG-NAE)  
**领域**: 计算心理语言学 / NLP  
**关键词**: Attention, 人类记忆检索, Transformer Grammar, 句法结构, 阅读时间预测

## 一句话总结

本文探究 Transformer Grammar（TG）的注意力机制能否作为人类记忆检索的认知模型，通过 Normalized Attention Entropy（NAE）将模型与人类阅读时间关联，发现基于句法结构的注意力比基于 token 的注意力更能解释人类句子处理行为，且两者提供独立互补的贡献。

## 研究背景与动机

在计算心理语言学中，语言模型是否能充当人类句子处理的认知模型是一个核心问题。过去这一问题主要从**期望理论**（expectation-based theory）角度探讨——即下一个 token 的预测（surprisal）是否能模拟人类的预测处理。近年来，Transformer 中注意力机制的成功意外地为另一大类理论——**记忆理论**（memory-based theory）——打开了新路径。

研究者发现，注意力权重中的加权引用模式（weighted reference patterns）与人类在线句子理解中检索元素的方式存在引人注目的平行关系。特别是 **cue-based retrieval**（线索检索理论）认为，人类在处理句子时通过当前输入词提供的线索从工作记忆中检索先前元素，干扰项越多检索越困难。

然而，此前的工作都集中在**原生 Transformer**（操作于 token 级表示）上，忽略了一个重要事实：心理语言学研究早已表明，**句法结构**能够提供 token 级因素无法充分解释的人类句子处理解释。这就引出了本文的核心问题：**如果注意力能作为记忆检索的通用算法，那么操作于句法结构之上的注意力是否也能捕捉人类记忆检索？**

## 方法详解

### 整体框架

本文将 **Transformer Grammar（TG）** 的注意力机制与人类阅读时间数据关联起来。TG 是一种句法语言模型，它联合生成 token 序列和对应的句法结构。链接假设（linking hypothesis）采用 **Normalized Attention Entropy（NAE）**：注意力权重越分散（熵越高），说明检索时干扰越大，对应更长的阅读时间。

### 关键设计

1. **Transformer Grammar（TG）的 COMPOSE/STACK 机制**：TG 的核心创新在于处理闭合短语的方式。当生成闭括号 X) 时，通过 COMPOSE 注意力计算闭合短语的向量表示；后续的 STACK 操作引用该向量作为短语表示进行下一步预测。这意味着 TG 的注意力操作于句法结构单元上（闭合短语作为整体），而不是像原生 Transformer 一样操作于 token 序列上。

2. **NAE 的计算**：对每个词，取 TG 最顶层各注意力头的 NAE 值求和。NAE 通过对注意力权重重归一化后计算归一化熵得到，值域 [0,1]。对于 TG，只考虑词汇 token（terminal）触发的注意力，排除非词汇符号的注意力。

3. **TG-comp 变体**：为判断 TG 的优势是来源于简单考虑句法结构还是来自 COMPOSE 注意力（将闭合短语作为单一表示），作者构建了 TG-comp 变体——它处理动作序列中的每个动作为独立 token，不使用 COMPOSE 注意力。

### 训练策略

- **模型**：16 层 8 头 TG 和 Transformer（252M 参数），训练在 BLLIP-lg 语料上（42M tokens，1.8M 句子）
- **评估**：使用线性混合效应模型（Linear Mixed-Effects Model），以多种基线预测因子（词长、n-gram 频率、surprisal、stack count 等）加上 NAE 作为固定效应，通过 ΔLogLik（对数似然增量）评估 NAE 的贡献
- **阅读时间数据**：Natural Stories 语料库，含多种心理语言学上有趣的句法构造

## 实验关键数据

### 主实验：NAE 对阅读时间的预测贡献

| 模型 | ΔLogLik (↑) | NAE 效应量 (ms) | NAE_so 效应量 (ms) | 显著种子 |
|------|-------------|----------------|-------------------|---------|
| TG | 76.6 (±8.1) | 1.42 (±0.2)*** | 2.26 (±0.1)*** | 3/3 |
| Transformer | 42.8 (±9.5) | 1.32 (±0.2)*** | 1.46 (±0.2)*** | 3/3 |

### 消融实验：COMPOSE 注意力的贡献

| 模型 | ΔLogLik | NAE 显著性 | NAE_so 显著性 |
|------|---------|-----------|-------------|
| TG | 46.1 (±9.1) | ** (2/3) | *** (3/3) |
| TG-comp | 18.1 (±9.3) | ** (1/3) | *** (3/3) |

### 关键发现

1. **TG 的 NAE 贡献显著高于 Transformer**（76.6 vs 42.8 ΔLogLik），说明基于句法结构的记忆检索在人类句子处理中占据更主导的角色。
2. **两个模型提供独立贡献**：似然比检验表明，包含两者 NAE 的回归模型显著优于任何单一模型，说明人类使用双重记忆表示——句法结构和 token 序列。
3. **POS 分析揭示互补性**：TG 的 NAE 在动词（VB, VBG, VBN, VBP）上更好，Transformer 的 NAE 在名词（NN, NNP）上更好，符合"动词触发的检索依赖句法特征、名词触发的检索依赖语义特征"的假说。
4. **COMPOSE 注意力至关重要**：TG 大幅优于 TG-comp（46.1 vs 18.1），且 TG-comp 无法解释 TG 已捕捉的方差。COMPOSE 在动词处理上贡献最大。
5. **NAE 捕捉的是干扰效应而非衰减效应**：将 NAE 与 Category Locality Theory（CLT）同时加入模型后，两者贡献独立，确认 NAE 量化的是记忆检索中的干扰。

## 亮点与洞察

- **跨学科桥梁**：本文将 NLP 中的注意力机制与语言学中的句法结构理论结合，为人类记忆检索提供了一个广覆盖的候选实现方案。
- **从"计算层"走向"算法层"**：相比 surprisal 理论在 Marr 三层描述中最抽象的计算层，将注意力解读为记忆检索使研究下移到更具体的算法层。
- **双重记忆系统假说**的实证支持：一个基于句法结构、一个基于 token 序列，注意力是通用检索算法。

## 局限与展望

- NAE 的计算方式（仅顶层、跨头求和、子词求和）沿用旧方法，替代方案值得探索
- 仅在英语自步阅读时间语料上验证，跨语言和跨认知指标（如眼动、EEG/fMRI）的泛化性未知
- 使用"完美预言"句法结构，未处理人类实际遇到的局部歧义
- 采用 top-down 解析策略，而心理语言学证据更倾向 left-corner 策略

## 相关工作与启发

- Ryu and Lewis (2021) 首先提出将注意力熵作为 cue-based retrieval 的链接假设
- Oh and Schuler (2022) 将其推广到自然文本并提出 NAE
- Sartran et al. (2022) 的 Transformer Grammar 在语法判断和脑活动上优于原生 Transformer
- Category Locality Theory (Isono, 2024) 以句法短语间距离量化记忆衰减
- 本文整合这些线索，首次系统比较结构级 vs token 级注意力对人类记忆检索的建模

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 TG 的注意力首次解读为认知记忆检索模型，填补了句法结构级别的空白
- **实验充分度**: ⭐⭐⭐⭐ — 主实验 + 独立贡献测试 + POS 分析 + COMPOSE 消融 + 干扰 vs 衰减分离，分析链非常完整
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑清晰、动机铺垫自然、技术与认知理论的衔接流畅
- **价值**: ⭐⭐⭐⭐ — 为计算心理语言学和 NLP 的交叉研究提供了重要实证和理论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](developmentally-plausible_working_memory_shapes_a_critical_period_for_language_a.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ICML 2025\] Modern Methods in Associative Memory](../../ICML2025/others/modern_methods_in_associative_memory.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)
- [\[ACL 2025\] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents](in_prospect_and_retrospect_reflective_memory_management_for_long-term_personaliz.md)

</div>

<!-- RELATED:END -->
