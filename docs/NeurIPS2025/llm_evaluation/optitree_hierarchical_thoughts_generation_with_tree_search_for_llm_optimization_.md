---
title: >-
  [论文解读] OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling
description: >-
  [NeurIPS 2025][LLM评测][运筹优化建模] 提出 OptiTree，通过构建建模树（modeling tree）组织运筹优化问题的层次化分类与建模思维，利用树搜索将复杂问题自适应分解为更简单的子问题序列，显著提升 LLM 在优化建模任务上的准确率（在多个困难基准上提升超过 10%）。
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "运筹优化建模"
  - "LLM推理"
  - "树搜索"
  - "子问题分解"
  - "层次化思维"
---

# OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2510.22192](https://arxiv.org/abs/2510.22192)  
**代码**: [GitHub](https://github.com/MIRALab-USTC/OptiTree/tree/main)  
**领域**: LLM评测  
**关键词**: 运筹优化建模, LLM推理, 树搜索, 子问题分解, 层次化思维

## 一句话总结

提出 OptiTree，通过构建建模树（modeling tree）组织运筹优化问题的层次化分类与建模思维，利用树搜索将复杂问题自适应分解为更简单的子问题序列，显著提升 LLM 在优化建模任务上的准确率（在多个困难基准上提升超过 10%）。

## 研究背景与动机

运筹优化（Operations Research, OR）建模是将自然语言描述的实际问题转换为数学优化模型的过程，传统上依赖大量人工专业知识。近年来，研究者利用 LLM 自动化这一过程，主要分为两类方法：

**基于提示的方法**（CoE, OptiMUS, MCTS）：将建模任务分解为固定步骤——依次生成变量、约束和目标函数

**微调方法**（ORLM, LLMOPT）：在大量建模数据上训练专用 LLM

**现有方法的关键问题**：

- 固定步骤分解不考虑问题复杂度，对于复杂问题效果差
- 作者分析发现，在 Medium 和 Hard 问题上，**超过 70% 的错误来自变量定义错误**
- 缺乏对问题之间结构化关系的利用

**核心观察**（三个动机）：

**观察 1**：现有固定步骤分解在复杂问题上失败率高，变量定义是主要瓶颈

**观察 2**：69% 的复杂 OR 问题包含标准 OR 问题作为子问题，且 LLM 能以较高准确率识别这些子问题

**观察 3**：基于子问题分解的朴素方法已能提升建模性能

## 方法详解

### 整体框架

OptiTree 的核心思想是：**将复杂 OR 问题自适应分解为一系列更简单的子问题，并利用子问题的建模经验（modeling thoughts）指导原问题的建模**。

整体流程包含三个阶段：
1. **建模树构建**：从数据集中自动构建层次化树结构
2. **树搜索与思维检索**：给定新问题，搜索树找到合适子问题链
3. **全局建模思维合成**：整合子问题的层次化思维，生成最终模型

### 子问题定义与识别

给定一个 OR 问题 $\mathcal{P}$，其优化模型为：

$$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.} \quad g_i(\boldsymbol{x}; \beta_i) \leq 0, \quad i=1,\ldots,N$$

如果将变量分为两组 $\boldsymbol{x} = (\tilde{\boldsymbol{x}}_1, \tilde{\boldsymbol{x}}_2)$，子问题 $\tilde{\mathcal{P}}$ 定义为只包含部分变量和约束的简化问题：

$$\min_{\tilde{\boldsymbol{x}}_1} f_1(\tilde{\boldsymbol{x}}_1) \quad \text{s.t.} \quad g_{i_k,1}(\tilde{\boldsymbol{x}}_1; \beta_{i_k,1}) \leq 0$$

**识别方式**：通过 LLM 将问题描述提炼为原子级陈述思维（statement thoughts）$\mathcal{C}_{\mathcal{P}} = \{c_1, c_2, \ldots, c_{n_\mathcal{P}}\}$，通过语义包含关系 $\mathcal{C}_{\tilde{\mathcal{P}}} \subseteq_{\mathcal{S}} \mathcal{C}_{\mathcal{P}}$ 判断子问题关系。

### 建模树结构

**建模树**（Modeling Tree）是一棵层次化树结构：

- **根节点**：表示抽象的组合优化问题类
- **每个节点**包含：问题类别名称、陈述思维 $\mathcal{C}_{\mathcal{P}}$、建模思维 $\mathcal{T}(\mathcal{P})$
- **父子关系**：父节点是子节点的子问题（$\mathcal{P}_j \subseteq_{\mathcal{S}} \mathcal{P}_i$）
- **子节点继承**父节点的基本变量和约束，并添加专门化组件
- **保序性质**：祖先节点一定是后代节点的子问题

### 树搜索与全局建模思维构建

给定问题 $\mathcal{P}$，从根节点开始逐层搜索：

$$\mathcal{P}^{(1)} = \underset{\mathcal{P}_t^{(0)}}{\text{argmax}} \; \mathbb{I}(\mathcal{P}_t^{(0)} \subseteq_{\mathcal{S}} \mathcal{P}) \cdot \text{Sim}_{\text{LLM}}(\mathcal{C}_{\mathcal{P}_t^{(0)}}, \mathcal{C}_{\mathcal{P}})$$

得到子问题链 $\mathcal{P}^{(1)} \subseteq_{\mathcal{S}} \mathcal{P}^{(2)} \subseteq_{\mathcal{S}} \cdots \subseteq_{\mathcal{S}} \mathcal{P}^{(M)}$，取最大子问题 $\mathcal{P}^{(M)}$ 的建模思维 $\mathcal{T}(\mathcal{P}^{(M)})$ 与问题描述结合，合成全局建模思维 $\mathcal{T}(\mathcal{P})$。

### 建模树的动态构建与更新

使用 OR-Instruct 3K 数据集中的 400 个问题自动构建：
1. 对每个新问题执行树搜索，找到最大子问题
2. 如果建模结果正确——树已覆盖此类问题
3. 如果错误——执行**节点扩展**，将新问题插入树中
4. 扩展时维护保序性质：检查新问题与现有兄弟节点的子问题关系

## 实验关键数据

### 主实验

在 7 个基准数据集上的建模准确率对比（基于 DeepSeek-V3）：

| 方法 | NL4Opt | MAMO EasyLP | MAMO ComplexLP | ComplexOR | IndustryOR | OptiBench | OptMATH |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CoT | 74.0 | 82.9 | 40.7 | 52.6 | 35.0 | 53.1 | 21.1 |
| CoE | 79.2 | 85.9 | 43.1 | 63.2 | 33.0 | 55.2 | 24.1 |
| OptiMUS | 80.6 | 87.1 | 45.2 | 79.0 | 36.0 | 58.8 | 32.5 |
| MCTS | 89.6 | 88.0 | 51.6 | 79.0 | 46.0 | 67.9 | 38.6 |
| DeepSeek-R1 | 86.1 | 79.5 | 57.3 | 68.4 | 38.0 | 70.2 | 33.1 |
| OpenAI-o1 | 87.1 | 87.6 | 54.5 | 73.6 | 40.0 | 71.5 | 34.9 |
| **OptiTree** | **98.3** | **96.9** | **81.5** | **84.2** | **54.0** | **74.7** | **52.4** |

### 消融实验

| 变体 | MAMO ComplexLP | IndustryOR |
|:---|:---:|:---:|
| OptiTree（完整） | 81.5 | 54.0 |
| w/o Tree Search | 下降明显 | 下降明显 |
| w/o Modeling Thoughts | 显著下降 | 显著下降 |
| depth=1 | 低于完整版 | 低于完整版 |
| depth=3 | 接近完整版 | 接近完整版 |

效率对比（秒/问题，IndustryOR）：

| 方法 | 总时间 |
|:---|:---:|
| CoE | 81.8 |
| OptiMUS | 57.8 |
| MCTS | 124.6 |
| OptiTree | **19.9**（搜索 8.4 + 建模 11.5） |

### 关键发现

1. **困难数据集提升最大**：在 MAMO ComplexLP 上从 MCTS 的 51.6% 提升到 81.5%（+30%），OptMATH 上从 38.6% 提升到 52.4%（+14%）
2. **超越推理 LLM**：OptiTree + DeepSeek-V3 超越 DeepSeek-R1 和 OpenAI-o1
3. **子问题覆盖率高**：平均 88% 的问题能找到子问题，手工验证准确率高
4. **效率优势明显**：OptiTree 的推理时间仅为 MCTS 的约 1/6
5. **仅需 400 个问题**即可构建泛化良好的建模树

## 亮点与洞察

1. **子问题分解的洞察非常精彩**：将 OR 建模从"生成变量→约束→目标"的扁平流程，转变为"找到结构相似的简单问题→增量建模"的层次化流程，这与人类专家的建模思路高度一致
2. **建模树的设计很优雅**：既是知识组织结构，又是搜索空间，将 RAG 和树搜索自然结合
3. **陈述思维的引入有效缓解 LLM 幻觉**：将问题转化为原子级语义描述再比较，比直接比较自然语言描述更可靠
4. **建模树自动构建**：不需要人工策划，从数据驱动地构建和更新，保证了可扩展性
5. **搜索空间极大缩减**：从指数级的变量/约束空间缩减到有限子问题集合

## 局限与展望

1. **建模树的质量依赖于构建数据集**：如果数据集覆盖的 OR 问题类型有限，树的泛化能力受限
2. **只关注提示方法**：未与微调方法结合，理论上 OptiTree 的思维可以作为微调数据的增强
3. **子问题识别依赖 LLM 能力**：对于 LLM 不熟悉的新型 OR 问题类型，识别准确率可能下降
4. **树的深度与广度的权衡**：过深的树可能引入累积误差，过浅则分解不充分
5. **未考虑多目标优化等更复杂场景**

## 相关工作与启发

- **Buffer-of-Thoughts (BoT)**：思维缓冲的概念与本文建模思维的存储和检索高度相关
- **RAG + 树搜索的结合**：为其他需要结构化知识检索的 LLM 应用提供了范式参考
- **增量建模策略**：可以推广到数学证明、代码生成等需要逐步构建的任务

## 评分

- ⭐⭐⭐⭐ (4/5)
- **创新性** ⭐⭐⭐⭐⭐：子问题分解 + 建模树 + 树搜索的组合非常新颖，抓住了 OR 建模的结构化特性
- **实验充分性** ⭐⭐⭐⭐⭐：7 个数据集、多种 LLM 骨干、详细的消融和效率分析
- **写作质量** ⭐⭐⭐⭐：动机清晰，三个观察层层递进
- **实用价值** ⭐⭐⭐⭐：对 OR 从业者和 LLM 应用落地有直接参考价值
- **理论深度** ⭐⭐⭐：子问题定义有形式化，但保序性证明较简单

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[ACL 2025\] HPSS: Heuristic Prompting Strategy Search for LLM Evaluators](../../ACL2025/llm_evaluation/hpss_heuristic_prompting_strategy_search_for_llm_evaluators.md)
- [\[AAAI 2026\] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](../../AAAI2026/llm_evaluation/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)
- [\[ACL 2025\] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](../../ACL2025/llm_evaluation/realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)
- [\[NeurIPS 2025\] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)

</div>

<!-- RELATED:END -->
