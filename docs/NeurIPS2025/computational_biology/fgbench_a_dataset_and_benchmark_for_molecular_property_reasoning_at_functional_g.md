---
title: >-
  [论文解读] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models
description: >-
  [NeurIPS 2025][计算生物][功能基团] 本文提出 FGBench，一个包含 625K 分子性质推理问题的数据集，专注于功能基团（functional group）级别的推理评估，通过三个维度（单功能基团影响、多功能基团交互、分子比较）系统揭示了当前 LLM 在细粒度化学推理能力上的严重不足。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "功能基团"
  - "分子性质推理"
  - "化学基准"
  - "结构-活性关系"
  - "LLM推理"
---

# FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2508.01055](https://arxiv.org/abs/2508.01055)  
**代码**: [https://github.com/xuanliugit/FGBench](https://github.com/xuanliugit/FGBench)  
**领域**: 计算生物  
**关键词**: 功能基团, 分子性质推理, 化学基准, 结构-活性关系, LLM推理

## 一句话总结

本文提出 FGBench，一个包含 625K 分子性质推理问题的数据集，专注于功能基团（functional group）级别的推理评估，通过三个维度（单功能基团影响、多功能基团交互、分子比较）系统揭示了当前 LLM 在细粒度化学推理能力上的严重不足。

## 研究背景与动机

**领域现状**：LLM 在化学领域应用日益广泛，包括分子性质预测、分子描述生成和分子生成。但现有数据集（如 MoleculeNet）主要关注分子整体层面的性质预测，提供的是分子级别的标签，缺乏更细粒度的结构信息。

**现有痛点**：(1) 功能基团是决定分子物理化学性质的核心结构单元（如羟基 -OH 赋予极性和氢键能力、羧基 -COOH 参与酯化反应），但现有数据库缺少功能基团与分子性质之间的显式关联；(2) 现有功能基团标注方法（如 CheckMol）基于直接模式匹配，在两个功能基团重叠时会失败，也无法直接判断两分子间的功能基团差异；(3) 没有专门评估 LLM 在功能基团级别推理能力的基准。

**核心矛盾**：化学家在预测分子性质时通常遵循"三步推理"——关联相似分子、观察功能基团差异、基于先验知识推断性质变化，但 LLM 缺乏这种细粒度推理能力，也没有合适的训练和评估资源。

**本文目标** (1) 如何构建可靠的功能基团级分子配对数据集？(2) LLM 在不同粒度的功能基团推理任务上的表现如何？(3) 化学专用微调是否能提升功能基团级推理能力？

**切入角度**：模仿化学家的推理流程，通过比较结构相似但功能基团不同的分子对来构建 QA 对。采用"验证-重构"策略确保数据质量。

**核心 idea**：通过验证-重构策略构建功能基团级分子配对数据集，系统评估 LLM 在单/多功能基团影响和分子比较三个维度上的推理能力，揭示当前模型的严重不足。

## 方法详解

### 整体框架

FGBench 构建流程：从 MoleculeNet 的 10 个数据集（ESOL, Lipophilicity, FreeSolv, HIV, BACE, BBBP, Tox21, SIDER, ClinTox, QM9）出发 → 规范化 SMILES → 基于 Tanimoto 相似度（512-bit Morgan 指纹）筛选相似分子对（阈值 > 0.7）→ 使用 AccFG 工具提取功能基团差异及精确位置 → 验证-重构策略验证正确性 → 基于模板生成 QA 对。最终产出 42,967 个分子配对和 625,936 个 QA 对，覆盖 245 种功能基团和 8 种分子性质。

### 关键设计

1. **验证-重构策略（Validation-by-Reconstruction）**:

    - 功能：确保功能基团差异标注的正确性和分子编辑操作的化学有效性
    - 核心思路：对于分子对 $(M_1, M_2)$ 及其功能基团差异 $(FG_1, FG_2)$，从 $M_1$ 上移除 $FG_1$ 并在相同位置替换为 $FG_2$，检验重构后分子是否与 $M_2$ 一致且化学有效。这一过程同时生成构建 QA 对所需的信息：带原子编号的分子 SMILES、带原子编号的功能基团 SMILES、以及功能基团在分子上的连接位置
    - 设计动机：直接的功能基团模式匹配在复杂结构中容易出错（尤其是重叠、异构等情况）。重构过程提供了端到端的正确性验证，且可泛化到其他分子性质数据集

2. **三维度推理任务**:

    - 功能：从不同粒度全面评估 LLM 的功能基团推理能力
    - 核心思路：
        - **维度1 - 单功能基团影响**：筛选仅有一种功能基团差异的分子对，评估模型对单一功能基团效应的理解（如删除羟基对溶解度的影响）
        - **维度2 - 多功能基团交互**：保留多功能基团差异的分子对，评估模型理解多个功能基团叠加/交互效应的能力
        - **维度3 - 分子比较**：直接给出两个完整分子进行性质比较，不提供功能基团编辑信息，作为对照基线
    - 设计动机：从简单到复杂逐层评估。维度3 作为对照可揭示模型是否仅依靠已记忆的分子级知识，而非功能基团级推理

3. **布尔与数值双类别 QA 对**:

    - 功能：分别评估定性判断和定量推理能力
    - 核心思路：布尔 QA 问功能基团修改是否改变性质方向（如从活性变非活性）；数值 QA 问确切数值变化量（如溶解度差异）。每个 QA 包含带原子编号的 SMILES、性质名称和初始性质值、详细的功能基团编辑指令
    - 设计动机：趋势判断和精确预测是两种本质不同的推理能力

### 评估设计

从 625K QA 中精选 7,146 个样本构成评估子集（每个任务最多 25 对），跨维度和类别均衡分布。评估 9 个模型（2 个闭源、4 个通用开源、3 个化学专用），分类任务用 ACC，回归用 RMSE。化学专用模型使用单独的答案解析器以适应其有限的指令遵循能力。

## 实验关键数据

### 主实验

| 模型 | Single-Bool | Inter-Bool | Comp-Bool | Single-Value RMSE |
|------|------------|-----------|----------|------------------|
| o3-mini | **0.687** | **0.693** | **0.703** | 101.886 |
| GPT-4o | 0.667 | 0.488 | 0.614 | 77.990 |
| Llama-3.1 70B | 0.683 | 0.530 | 0.456 | 84.119 |
| Llama-3.1 8B | 0.548 | 0.547 | 0.474 | 162.351 |
| Qwen2.5-7B | 0.590 | 0.396 | 0.664 | 63.511 |
| ChemLLM-7B | 0.233 | 0.235 | 0.250 | 209.584 |
| nach0 | 0.606 | 0.543 | 0.041 | 104.534 |
| LlaSMol-Mistral-7B | 0.387 | 0.298 | 0.239 | 266.720 |

### 消融实验

| 分析维度 | 发现 |
|---------|------|
| 单 FG → 多 FG 交互 | 大多数模型准确率显著下降（GPT-4o: 0.667→0.488；Llama 70B: 0.683→0.530） |
| 化学专用 vs 通用模型 | ChemLLM-7B（0.233）远不如同规模 Llama-3.1 8B（0.548） |
| nach0 偏差分析 | 在 Single-Bool 中 97.7% 预测为 False，高 ACC 来自类别偏差 |
| Qwen2.5 异常 | Comp-Bool 0.664 vs Inter-Bool 0.396，疑似已见过 MoleculeNet 分子但缺乏 FG 知识 |
| 推理模型优势 | o3-mini 在 4/6 任务上最优，推理能力对化学推理重要 |

### 关键发现

- **LLM 在功能基团级推理严重不足**：最佳模型 o3-mini 在 Single-Bool 仅 68.7%，多功能基团交互更差
- **化学微调适得其反**：ChemLLM 尽管在分子级数据上广泛训练，但在 FGBench 上远不如通用模型——分子级知识无法迁移到功能基团级
- **推理能力是关键**：o3-mini 的推理训练使其在化学性质推理上持续领先，尤其在需要多步推理的多 FG 交互任务上
- **失败案例分析**：o3-mini 推理苯腈 vs 苯的 logD 时能正确识别极性（部分正确推理）但弄反了方向（事实性错误），暗示检索增强可能是改进方向

## 亮点与洞察

- **填补评估空白**：首个专门针对功能基团级分子性质推理的数据集（625K QA，245 种功能基团），规模和全面性均优秀
- **验证-重构策略的可泛化性**：数据构建管线可直接应用于其他分子性质数据集
- **揭示化学微调的困境**：分子级训练不仅不能迁移到功能基团级，还可能损害基座模型的通用推理能力
- **细粒度标注的多模态潜力**：功能基团的精确位置标注使数据集可直接用于分子图-文本多模态学习

## 局限与展望

- 未考虑位置异构（邻/间/对位取代）、碳链异构和立体异构等更细粒度的结构差异
- 仅基于 MoleculeNet 的 10 个数据集，性质种类有限
- 评估为零样本设置，未探索 few-shot 或微调后的性能
- 缺少 3D 分子结构信息，某些性质（如手性对药理活性影响）需要 3D 感知
- SMILES 作为输入格式本身不直观，可能限制 LLM 的化学推理能力

## 相关工作与启发

- MoleculeNet 是经典的分子性质基准，但仅提供分子级标签，FGBench 向下扩展到功能基团级
- AccFG 工具解决了功能基团重叠标注问题，是 FGBench 构建的关键依赖
- SciBench 和 MolPuzzle 分别评估化学数学推理和分子结构解析，FGBench 聚焦结构-性质关系推理
- 功能基团在分子表示预训练中已被广泛使用（Li et al. 2023, Nguyen et al. 2024），但作为 token 而非推理评估目标

## 评分

⭐⭐⭐⭐ (4/5)

数据构建方法扎实（验证-重构策略），任务设计合理（三维度 × 两类别），规模可观（625K QA）。揭示了化学微调 LLM 的泛化困境这一重要发现。作为 benchmark 论文，模型评估充分。局限在于属于相对常规的 benchmark 贡献，方法论创新有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[ACL 2026\] BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](../../ACL2026/computational_biology/biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/computational_biology/tracing_pharmacological_knowledge_in_large_language_models.md)

</div>

<!-- RELATED:END -->
