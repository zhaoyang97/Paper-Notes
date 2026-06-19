---
title: >-
  [论文解读] SGIC: A Self-Guided Iterative Calibration Framework for RAG
description: >-
  [ACL 2025][信息检索/RAG][自校准] SGIC 利用 LLM 的 token 级不确定性分数（文档相关性不确定性 + 答案置信度不确定性）作为自校准的引导信号，通过迭代将前一轮答案及其不确定性分数注入提示中触发上下文推理，在 HotpotQA 上将 Llama2-7B 的 EM 从 69.1% 提升到 77.2%（+8.1%），对 GPT-4o 也有 2.8% 的提升。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "自校准"
  - "不确定性估计"
  - "迭代推理"
  - "文档相关性"
  - "多跳QA"
---

# SGIC: A Self-Guided Iterative Calibration Framework for RAG

**会议**: ACL 2025  
**arXiv**: [2506.16172](https://arxiv.org/abs/2506.16172)  
**代码**: 无  
**领域**: RAG / 检索增强生成  
**关键词**: 自校准, 不确定性估计, 迭代推理, 文档相关性, 多跳QA

## 一句话总结

SGIC 利用 LLM 的 token 级不确定性分数（文档相关性不确定性 + 答案置信度不确定性）作为自校准的引导信号，通过迭代将前一轮答案及其不确定性分数注入提示中触发上下文推理，在 HotpotQA 上将 Llama2-7B 的 EM 从 69.1% 提升到 77.2%（+8.1%），对 GPT-4o 也有 2.8% 的提升。

## 研究背景与动机

**领域现状**：RAG 通过检索外部文档来增强 LLM 的生成能力。现有工作主要关注如何检索更相关的文档或设计更好的指令模板，但较少关注 LLM 的自校准能力——即利用自身的上下文推理能力迭代改进生成答案。

**现有痛点**：（1）即使检索到相关文档，LLM 对多跳问题的首次回答也经常出错，但现有 RAG 方法很少提供"第二次机会"；（2）自校准方法（如 Self-Refine、CoT 迭代）通常依赖复杂的提示设计或多模型辩论，成本高昂；（3）LLM 对文档相关性的判断和答案的置信度可以通过 token 概率来量化，但这一信号尚未被系统性地用于指导校准。

**核心矛盾**：LLM 在 RAG 场景中的不确定性信号（文档相关性、答案置信度）是天然可用的自监督信号，但现有方法未将其转化为校准的引导信号。

**本文目标** 将 LLM 的不确定性分数（token 概率乘积）作为显式信号注入迭代自校准提示中，让 LLM 基于前述答案的不确定性进行有针对性的上下文推理和答案修正。

**切入角度**：作者观察到 LLM 对正确答案和错误答案、相关文档和不相关文档的不确定性分数分布存在清晰的分离（如图 1 所示），这为利用不确定性作为校准信号提供了统计基础。

**核心 idea**：用 token 概率乘积计算文档和答案的不确定性分数，将这些分数和前轮答案一起注入提示进行迭代自校准，同时构建校准训练集微调开源 LLM。

## 方法详解

### 整体框架

SGIC 的推理流程为：（1）给定所有检索文档，生成首轮答案并计算答案不确定性分数 $s'_{ans}$；（2）对每个文档单独生成答案并计算文档不确定性分数 $s'_{doc}$；（3）将文档（带不确定性标注）+ 前轮答案（带置信度）重新组织为新提示，LLM 进行上下文推理生成校准答案；（4）迭代 K 轮。对开源 LLM，额外构建包含不确定性信息的训练集进行微调。

### 关键设计

1. **双维度不确定性估计**:

    - 功能：分别量化每个文档的相关性和生成答案的可靠性
    - 核心思路：答案不确定性 $s_{ans} = \prod_{i=1}^m p_i$（所有 token 最高概率的乘积），归一化到 [0, 100]。文档不确定性 $s_{doc} = 1 - \prod p_i$（用单个文档生成答案的概率乘积的补数），表示该文档的信息不充分程度。高 $s_{ans}$ 表示答案可信，高 $s_{doc}$ 表示文档不相关
    - 设计动机：实证显示（图 1）正确答案和错误答案的不确定性分布有清晰分离，文档相关与不相关同样可区分，为校准提供了可靠的引导信号

2. **迭代自校准推理**:

    - 功能：利用前轮答案和不确定性分数作为上下文线索进行多轮答案改进
    - 核心思路：第 $k$ 轮的提示包含：原始文档（每个带 $s'_{doc}$ 标注）+ 前 $k-1$ 轮的答案和对应 $s'_{ans}$（如 "Round 1: Lord High... (Uncertainty: 73), Round 2: United States... (Uncertainty: 51)"）。LLM 通过对比前轮答案的不确定性变化趋势进行上下文推理
    - 设计动机：不确定性分数的数值变化为 LLM 提供了明确的校准方向信号——如果前轮不确定性高则需要更大改动，如果已经低则倾向保持

3. **校准训练集构建（开源 LLM）**:

    - 功能：让小型开源 LLM 学会有效利用不确定性分数
    - 核心思路：对训练集执行推理生成首轮答案和不确定性，然后将正确答案作为校准后的目标标注，构建含不确定性信号的训练样本进行微调（全参或 LoRA）
    - 设计动机：大型闭源模型（GPT-4o）天然具备利用数值提示的能力，但小型开源模型（Llama2-7B）需要微调才能理解不确定性分数的含义

### 损失函数 / 训练策略

使用标准的因果语言建模损失微调开源 LLM。Llama2-7B 使用 LoRA，Phi-3.5 使用全参微调。

## 实验关键数据

### 主实验

| 模型 | 数据集 | EM (基线) | EM (SGIC) | 提升 |
|------|--------|----------|----------|------|
| Llama2-7B-Chat (LoRA) | HotpotQA | 69.1 | **77.2** | +8.1 |
| Llama2-7B-Chat (LoRA) | NQ | 74.7 | **79.0** | +4.3 |
| Phi-3.5-mini (Full) | HotpotQA | 42.8 | **55.3** | +12.5 |
| GPT-4o | HotpotQA | 73.7 | **76.5** | +2.8 |
| GPT-4o | NQ | 63.3 | **65.2** | +1.9 |
| GPT-4o-mini | HotpotQA | 69.2 | **74.1** | +4.9 |

### 消融实验

不确定性组件消融（Llama2-7B, HotpotQA EM）：

| 配置 | EM | 说明 |
|------|-----|------|
| 仅校准（无不确定性） | 71.8 | 基础校准有效 |
| + 答案不确定性 | 76.2 | +4.4，答案置信度信号关键 |
| + 文档不确定性 | **77.2** | +1.0，文档相关性信号补充 |
| Oracle 文档 | 75.3 | 上界参考 |

按问题类型（Llama2-7B, HotpotQA）：

| 问题类型 | EM (基线) | EM (SGIC) | 提升 |
|---------|----------|----------|------|
| Bridge (多跳桥接) | 65.0 | **75.7** | +10.7 |
| Comparison (比较) | 69.6 | **83.1** | +13.5 |

### 关键发现

- **迭代校准有效**：多轮校准持续改善性能，K=5 轮趋于饱和
- **答案不确定性贡献最大**：在消融中，加入答案不确定性带来 +4.4% EM，文档不确定性额外带来 +1.0%
- **比较题获益最大**：比较类问题从 69.6% 提升到 83.1%（+13.5%），可能因为这类题目的校准信号更明确
- **闭源模型也能获益**：GPT-4o 这样的强模型也有 +2.8% 提升，说明显式不确定性信号对任何 LLM 都有辅助价值
- **小模型获益大**：Phi-3.5-mini 提升 12.5%，说明能力不足的模型从校准引导中受益更多

## 亮点与洞察

- **不确定性分数作为显式校准信号**：将 LLM 天然具备的不确定性信息显式化为数值标注注入提示，简单直觉但效果显著。这种"self-aware calibration"思路可以推广到任何需要迭代改进的 LLM 任务
- **零额外模型的自校准**：相比多模型辩论等方法，SGIC 仅使用单个 LLM 就实现有效校准，降低成本
- **训练集构建策略巧妙**：通过自己的不确定性+正确答案构建校准训练样本，让开源小模型也能学会利用不确定性信号

## 局限与展望

- **不确定性估计的准确性是瓶颈**：token 概率乘积是一种粗糙的不确定性度量，更精确的估计方法可能进一步提升效果
- **仅在 2 个多跳 QA 基准上验证**：缺少在摘要、翻译等其他 RAG 任务上的验证
- **迭代轮数增加推理成本**：K=5 轮校准意味着 5 倍推理开销
- **开源模型需要微调**：SGIC 对开源模型需要额外的微调步骤才能有效利用不确定性分数

## 相关工作与启发

- **vs Self-RAG**：Self-RAG 通过微调让模型生成特殊标记判断检索需求，SGIC 用不确定性分数引导迭代校准，两者互补
- **vs SeaKR**：SeaKR 用内部状态不确定性做检索决策，SGIC 用输出概率不确定性做答案校准，视角不同但思路相似
- **vs Self-Refine/CoT迭代**：这些方法缺少量化的校准信号，SGIC 通过不确定性分数提供了明确的改进方向

## 评分
- 新颖性: ⭐⭐⭐ 将不确定性分数注入校准提示的思路自然但增量创新有限
- 实验充分度: ⭐⭐⭐⭐ 覆盖闭源/开源 LLM、两个基准、详细消融和问题类型分析
- 写作质量: ⭐⭐⭐ 方法描述清晰但不确定性估计的理论动机可以更深入
- 价值: ⭐⭐⭐⭐ 简单有效的 RAG 改进方法，对开源和闭源模型都适用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](../../ACL2026/information_retrieval/an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)

</div>

<!-- RELATED:END -->
