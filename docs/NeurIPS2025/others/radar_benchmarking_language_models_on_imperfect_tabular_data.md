---
title: >-
  [论文解读] Radar: Benchmarking Language Models on Imperfect Tabular Data
description: >-
  [NeurIPS 2025][表格推理] 提出 Radar 基准，通过对真实表格数据注入五类数据工件（缺失值、错误值、异常值、格式不一致、逻辑不一致），系统评估语言模型在不完美表格数据上的数据感知推理能力，揭示即使是前沿模型在引入数据工件后性能也大幅下降。 语言模型越来越多地被部署为自主数据分析代理，用于对表格数据进行趋势汇…
tags:
  - "NeurIPS 2025"
  - "表格推理"
  - "数据感知"
  - "数据工件"
  - "语言模型基准"
  - "鲁棒性评估"
---

# Radar: Benchmarking Language Models on Imperfect Tabular Data

**会议**: NeurIPS 2025  
**arXiv**: [2506.08249](https://arxiv.org/abs/2506.08249)  
**代码**: [GitHub](https://github.com/kenqgu/RADAR) / [HuggingFace](https://huggingface.co/datasets/kenqgu/RADAR)  
**领域**: 其他  
**关键词**: 表格推理, 数据感知, 数据工件, 语言模型基准, 鲁棒性评估

## 一句话总结

提出 Radar 基准，通过对真实表格数据注入五类数据工件（缺失值、错误值、异常值、格式不一致、逻辑不一致），系统评估语言模型在不完美表格数据上的数据感知推理能力，揭示即使是前沿模型在引入数据工件后性能也大幅下降。

## 研究背景与动机

语言模型越来越多地被部署为自主数据分析代理，用于对表格数据进行趋势汇总、关系识别和数据操作。然而，一个关键问题被忽视了：模型是否具备**数据感知能力**——即识别、推理和正确处理缺失值、异常值、逻辑不一致等数据工件的能力。

现实世界的表格数据普遍存在数据质量问题。在高风险场景中（如医疗记录中错误记录的心率 220 bpm），如果模型不能识别这些工件，可能导致有害或误导性的结论。现有表格推理基准大多假设数据是"干净"的，即使有些工作研究了结构扰动（如行列打乱），也只是测试模型是否依赖位置启发式，而不评估模型对数据质量问题的语义理解能力。

此外，现实数据通常包含数百或数千行，而现有基准通常使用小表格且不支持表格大小的控制。Radar 同时填补了这两个空白：系统评估数据工件处理能力，并控制表格大小以研究推理性能如何随输入复杂度变化。

## 方法详解

### 整体框架

Radar 的核心思想是：给定干净源表 $T$，通过程序化扰动函数 $g:(T, Q) \mapsto (T_p, T_r)$ 生成含工件的表 $T_p$ 和对应的恢复表 $T_r$。评估时模型接收扰动表 $T_p$ 和查询 $Q$，正确答案定义为 $A = f(Q, T_r)$，即模型需要自主识别并处理数据工件才能得到正确答案。直接在 $T_p$ 上计算会得到错误结果。

### 关键设计

1. **五类数据工件**：

    - **Missing Data**：用空值替换有效单元格
    - **Bad Values**：注入明显错误/占位值（如 -1, 9999, TEST, #REF!）
    - **Outliers**：插入极端不合理数值（如静息心率 220 bpm）
    - **Inconsistent Formatting**：同一数据的不同表示（如 "22 lbs" vs "22 pounds" vs "weight = 22"）
    - **Inconsistent Logic**：跨字段矛盾（如结束时间早于开始时间、BMI 与身高体重不匹配）

2. **程序化扰动框架**：每个扰动函数针对特定查询设计，确保直接在扰动表上计算会得到错误结果。扰动影响不超过 10% 的行，保持数据分布的整体完整性。答案函数 $f$ 和扰动函数 $g$ 均可跨不同大小的表格运行，只要核心字段 $\mathcal{C}$ 存在。

3. **多维度表格大小控制**：以 token 数衡量表格大小 $\tau \in \{2K, 4K, 8K, 16K\}$，同时控制列数 $c \in \{5, 10, 20\}$。对于给定的 $(\tau, c)$ 组合，选择行数 $R = \arg\min_r |tok(T_s, r, c) - \tau|$。这使得可以在控制语义和复杂度不变的条件下系统研究表格大小对推理性能的影响。

### 数据集构建

由 12 位数据科学专家众包收集 53 个任务，来自 27 个源表格，跨 9 个应用领域。专家编写了 260 个扰动函数，经多轮代码审查和交叉验证。最终 Radar 包含 2980 个任务实例。提供两个子集：Radar-T（53 个任务，标准化为 10 列 / 8K token）和 Radar-S（10 个任务，完整的大小变化配置）。

### 评估设置

评估两种基线：（1）直接提示（Direct Prompting），模型直接文本推理；（2）代码代理（Code Agent），模型配备 Python shell 工具。在系统提示中明确指示模型注意五类数据工件，但不针对特定表实例。使用精确匹配（EM）作为主指标。

## 实验关键数据

### 主实验

| 模型 | Clean (Direct) | Artifact Avg (Direct) | Clean (Code) | Artifact Avg (Code) |
|------|------|------|------|------|
| StructLM (表格微调) | 2.3 | 0.8 | - | - |
| TableGPT2 (表格微调) | 0.8 | 1.1 | 35.0 | 6.9 |
| Gemma3 27B | 1.9 | 2.3 | 75.5 | 13.6 |
| DeepSeek-V3 | 1.9 | 3.5 | 96.2 | 37.6 |
| GPT-4.1 | 17.0 | 14.3 | 98.1 | 48.6 |
| Gemini 2.5 Flash Thinking | 39.6 | 19.9 | 88.7 | 43.3 |
| DeepSeek-R1 | 34.0 | 26.3 | 84.9 | 52.8 |
| o3-mini (high) | 73.6 | 35.6 | 75.5 | 44.9 |
| Gemini 2.5 Pro | 71.7 | 50.9 | 84.9 | 62.3 |
| **o4-mini (high)** | **83.0** | **53.9** | **100** | **61.8** |

o4-mini 在 Code Agent 下干净表格达到 100% 精确匹配，但引入数据工件后下降约 **59%**。

### 按工件类型分析

| 工件类型 | o4-mini (Direct) | o4-mini (Code) | Gemini 2.5 Pro (D) | Gemini 2.5 Pro (C) |
|------|------|------|------|------|
| Missing | 49.1 | 50.9 | 50.9 | 73.6 |
| Bad Values | 58.5 | 54.7 | 56.6 | 54.7 |
| Outliers | 56.2 | 83.3 | 47.9 | 64.6 |
| Formatting | 73.6 | 79.2 | 56.6 | 73.6 |
| **Logic** | **32.1** | **41.1** | **42.9** | **44.6** |

**逻辑不一致是最难处理的工件类型**，所有模型在此类别上表现最差。

### 关键发现

- **代码执行不是万能药**：虽然代码代理在干净表格上几乎完美，但面对数据工件仍有显著性能差距，特别是对逻辑不一致
- **表格大小影响**：直接提示下性能随 token 数增加而持续下降，到 16K token 时几乎降为零；代码代理性能基本不受表格大小影响
- **宽表 vs 窄表**：在相同 token 数下，模型在列多行少的宽表上表现更好——因为模型倾向于逐行检查，行数增加线性增加计算量
- **值推导 vs 行丢弃**：代码代理在需要丢弃行的任务上提升明显，但在需要跨列推导值的任务上改善有限

## 亮点与洞察

- 揭示了一个被忽视但极其重要的问题：LLM 在干净数据上的表现远不能代表其在真实数据上的能力
- 程序化扰动框架设计精巧，支持自动生成大规模可验证的评估实例
- 逻辑不一致是最难检测的工件类型，因为需要多列/多行的交叉推理
- 对构建可靠数据科学代理有直接指导意义：需要专门的数据质量检测模块

## 局限与展望

- 目前只支持五类独立的扰动，未研究多种工件同时存在的情况
- 要求唯一正确的纠正操作，排除了多种合理修正共存的场景
- 扰动函数由专家手工编写，扩展到新领域需要额外人力
- 未探索模型在被明确告知"数据可能有问题"的情况下是否能做得更好

## 相关工作与启发

- 对比 ROBUT、BIG-Bench Extra Hard 等只做结构扰动的工作，Radar 要求语义层面的数据理解
- 与 BLADE、DABench 等数据分析基准互补：它们评估分析能力，Radar 评估数据质量感知
- 启示：未来的 LLM 代理应将数据验证和清洗作为分析流程的第一步

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个系统评估 LLM 数据感知能力的表格基准
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 11 个模型、两种范式、多维度分析
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，框架描述详细
- **价值**: ⭐⭐⭐⭐⭐ 对 LLM-as-data-analyst 的可靠性评估有重要意义，揭示了关键短板

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](../../ACL2025/others/generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ICML 2026\] GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data](../../ICML2026/others/gotabpfn_from_feature_ordering_to_compact_tokenization_for_tabular_foundation_mo.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](../../ICLR2026/others/tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[NeurIPS 2025\] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)

</div>

<!-- RELATED:END -->
