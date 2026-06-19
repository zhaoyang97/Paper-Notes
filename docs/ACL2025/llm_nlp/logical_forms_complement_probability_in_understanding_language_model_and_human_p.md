---
title: >-
  [论文解读] Logical Forms Complement Probability in Understanding Language Model (and Human) Performance
description: >-
  [ACL 2025][LLM 其他][logical reasoning] 系统研究 LLM 在命题逻辑和模态逻辑推理上的能力，发现除了输入概率（perplexity）外，逻辑形式（modality、argument form）是预测 LLM 表现的重要互补因素，并通过人类行为数据对比揭示人机推理的异同。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "logical reasoning"
  - "modal logic"
  - "propositional logic"
  - "syllogism"
  - "LLM evaluation"
---

# Logical Forms Complement Probability in Understanding Language Model (and Human) Performance

**会议**: ACL 2025  
**arXiv**: [2502.09589](https://arxiv.org/abs/2502.09589)  
**代码**: —  
**领域**: 逻辑推理 / LLM 行为分析 / 认知科学  
**关键词**: logical reasoning, modal logic, propositional logic, syllogism, LLM evaluation  

## 一句话总结

系统研究 LLM 在命题逻辑和模态逻辑推理上的能力，发现除了输入概率（perplexity）外，逻辑形式（modality、argument form）是预测 LLM 表现的重要互补因素，并通过人类行为数据对比揭示人机推理的异同。

## 研究背景与动机

### 问题背景
随着 LLM 被广泛用于规划和决策，理解其逻辑推理能力变得至关重要。现有研究虽然表明 LLM 在逻辑推理上表现尚可，但缺乏对不同**逻辑形式**的细粒度分析——在自然语言呈现的多种论证形式中，LLM 是否表现一致？是否对某些形式有偏好？

### 核心研究问题
1. 输入概率（probability/perplexity）是否足以预测 LLM 在逻辑推理上的表现？
2. 逻辑形式（包括模态和论证形式）是否是额外的重要预测因子？
3. LLM 和人类在逻辑推理上的表现有何异同？

### 文献定位
- 与 Gonen et al. (2023) 和 McCoy et al. (2024) 的概率假说大致一致，但**补充了逻辑形式这一关键维度**
- 与 Eisape et al. (2024) 对范畴三段论的研究互补，本文聚焦**假言三段论和选言三段论**
- 首次在 LLM benchmark 中引入**模态逻辑**（必然性与可能性）

## 方法详解

### 数据集构建

**逻辑形式体系**：基于命题逻辑和正规模态逻辑（alethic modal logic），使用以下核心算子：
- $\neg$（否定）、$\Box$（必然，must）、$\Diamond$（可能，may）
- $\vee$（析取，or）、$\wedge$（合取，and）、$\to$（蕴含，if...then）

**涉及的逻辑形式**（4 种有效推理 + 4 种谬误 × 3 种模态 = 24 种）：

有效推理（ground truth = Yes）：
- $\vee^L$：选言三段论（左否定前件）
- $\to^L$：假言三段论-肯定前件（Modus Ponens）
- $\vee^R$：选言三段论（右否定后件）
- $\to^R$：假言三段论-否定后件（Modus Tollens）

谬误（ground truth = No）：
- $\vee^L_\nvdash$：肯定析取项
- $\to^L_\nvdash$：肯定后件（Affirming the Consequent）
- $\vee^R_\nvdash$：肯定析取项
- $\to^R_\nvdash$：否定前件（Denying the Antecedent）

**模态**：每组变量可在三种模态下操作：$\Box$（必然）、$\Diamond$（可能）、$\varnothing$（无模态/命题逻辑）

### 知识偏见控制

关键设计——避免解释内容的常识偏见影响推理：
- 生成 204 个动词短语 + 200 个常用人名组合成主-动-宾三元组
- 每个解释中两个变量的内容**独立**
- 随机生成 1000 个解释，应用到 24 种逻辑形式，共 **24,000 个问题**

### 翻译为自然语言
将逻辑形式转化为 Yes/No 问题格式，例如：
> Consider the following statements: Jane is watching a show or John is reading a book. Jane isn't watching a show. Question: Based on these statements, can we infer that John is reading a book?

### 评估指标
采用**基于概率的软准确率**（soft accuracy），避免贪心解码低估模型能力：

$$\hat{p} = \frac{p(\text{No}|s)\mathbb{1}[y=\text{No}] + p(\text{Yes}|s)\mathbb{1}[y=\text{Yes}]}{p(\text{No}|s) + p(\text{Yes}|s)}$$

## 实验

### 评估模型
10 个开源模型：mistral-7b/8x7b、llama-2-7b/13b/70b、llama-3-8b/70b、yi-34b、phi-2、phi-3-mini
商业模型参考：OpenAI o1、Gemini-1.5-Pro

### 主实验结果

| 模型 | 总体准确率 | 无模态 | 必然(□) | 可能(◇) | MP | MT | 肯定后件 |
|------|----------|-------|---------|---------|----|----|---------|
| llama-3-70b | 0.714 | 0.745 | 0.554 | **0.843** | 0.773 | **0.515** | 0.661 |
| mistral-8x7b | **0.724** | 0.698 | 0.601 | 0.874 | **0.873** | 0.023 | 0.648 |
| phi-3-mini | 0.690 | 0.657 | 0.536 | 0.877 | **0.974** | 0.475 | 0.462 |
| OpenAI o1 | 0.926 | 1.000 | 0.773 | 1.000 | 1.000 | 0.895 | 1.000 |
| 人类 | 0.595 | 0.589 | 0.566 | 0.640 | **0.901** | 0.628 | **0.225** |

### 关键发现

**发现一：模态影响显著**
- 所有模型在**可能性（◇）模态**下表现最好，必然性（□）最差
- 统计检验高度显著（p < 0.001）

**发现二：论证形式差异显著**
- 大多数模型在**Modus Tollens（否定后件）** 上表现最差（有效推理中）
- 在**肯定后件**谬误上表现最差（谬误中）
- Modus Ponens 是最容易的形式
- 这些模式在线性混合效应模型中均显著（p < 0.001）

**发现三：Perplexity 不是可靠的预测指标**
- Perplexity 与准确率的相关性存在但很弱（ρ = -0.09）
- 线性混合效应模型：marginal R² = 0.342，conditional R² = 0.543
- 控制实验（用无意义词替换解释）显示 perplexity 差异巨大（~30 vs ~10）但性能差异取决于逻辑形式

**发现四：LLM 的模态偏好**
- 在可能性模态下，LLM 倾向于给出肯定答案（Yes）
- 在必然性模态下，LLM 倾向于给出否定答案（No）

### 人机对比
- **共同点**：人类和 LLM 都在 Modus Ponens 上表现最好
- **差异**：人类在"肯定后件"谬误上表现极差（0.225），而某些 LLM 表现尚可
- 某些 LLM 对特定逻辑形式的偏好缺乏人类直觉支持

### 统计分析方法
使用**线性混合效应模型**：
$$\text{Acc}_\text{soft} \sim \text{Modality} + \text{ArgForm} + \text{Perplexity} + (1 + \text{Perplexity} | \text{LLM})$$

固定效应：模态、论证形式、困惑度；随机效应：模型特异性偏差

## 亮点与洞察

1. **逻辑形式作为预测因子的发现**：首次系统性地证明逻辑形式（而非仅概率）是预测 LLM 推理表现的关键因素
2. **模态逻辑的引入**：首个在 LLM benchmark 中系统纳入必然性/可能性模态的工作
3. **严格的实验控制**：消除知识偏见的数据构建方法值得借鉴
4. **软准确率指标**：避免贪心解码低估模型能力的评估方法
5. **人机对比**：提供了新的行为数据，揭示 LLM 推理与人类推理的本质异同
6. **无意义词控制实验**：巧妙验证 perplexity 的不可靠性

## 局限性

1. 仅考虑原子级别的命题和模态逻辑推理，未涉及多步复杂推理
2. 使用 zero-shot 设置，添加 few-shot 示例可能改变绝对性能和模式
3. 人类实验样本量有限，可能存在人群偏差
4. 商业模型（o1、Gemini）因 API 限制只能用贪心解码评估，不直接可比
5. 未探索模态逻辑以外的其他逻辑系统（如时态逻辑、认知逻辑）

## 相关工作

- **逻辑推理 Benchmark**：ProofWriter、PrOntoQA 等合成数据集，但未涉及模态逻辑
- **LLM 逻辑推理**：Clark et al. (2021)、Hahn et al. (2021) 等训练/微调方法，本文聚焦 prompting 评估
- **人类逻辑推理**：Eisape et al. (2024) 的范畴三段论研究、Lampinen et al. (2024) 的内容效应
- **概率假说**：Gonen et al. (2023)、McCoy et al. (2024) 提出 LLM 在高概率输入上表现更好

## 评分

⭐⭐⭐⭐ — 研究设计严谨（受控实验 + 混合效应模型），发现具有启发性（逻辑形式 > 概率）。数据集构建的去偏方法和人机对比增加了研究深度。不足在于限于原子推理和模态逻辑的特定子集。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)
- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](collaborative_performance_prediction_for_large_language_models.md)
- [\[ACL 2025\] HumT DumT: Measuring and Controlling Human-like Language in LLMs](humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)

</div>

<!-- RELATED:END -->
