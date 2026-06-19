---
title: >-
  [论文解读] Learning Repetition-Invariant Representations for Polymer Informatics
description: >-
  [NeurIPS 2025][计算生物][聚合物信息学] 提出 GRIN（Graph Repetition-Invariant Network），通过 Max 聚合和特殊的图构建策略使 GNN 对聚合物重复单元的拼接数量不变，解决了聚合物表示中的基本对称性问题。 领域现状 领域现状：聚合物（polymer）是由重复单元（mo…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "聚合物信息学"
  - "重复不变性"
  - "图神经网络"
  - "Max聚合"
  - "分子表示"
---

# Learning Repetition-Invariant Representations for Polymer Informatics

**会议**: NeurIPS 2025  
**arXiv**: [2505.10726](https://arxiv.org/abs/2505.10726)  
**代码**: 有  
**领域**: 计算生物  
**关键词**: 聚合物信息学, 重复不变性, 图神经网络, Max聚合, 分子表示

## 一句话总结

提出 GRIN（Graph Repetition-Invariant Network），通过 Max 聚合和特殊的图构建策略使 GNN 对聚合物重复单元的拼接数量不变，解决了聚合物表示中的基本对称性问题。

## 研究背景与动机

### 领域现状

**领域现状**：聚合物（polymer）是由重复单元（monomer）组成的长链分子，性质预测是材料科学的核心问题。

**现有痛点**：标准 GNN 对同一聚合物的不同重复单元表示（1 个重复 vs 3 个重复 vs N 个重复）给出不同预测，违反了聚合物的基本对称性——重复不变性。

**核心矛盾**：GNN 的聚合操作（Sum/Mean）对图大小敏感，无法自然处理"同一分子的不同大小表示"。

**切入角度**：Max 聚合天然对重复次数不变（取最大值不受重复数量影响），但需要特殊图构建来正确工作。

**核心 idea**：Max 聚合 + 环化连接（将链首尾相连成环）+ 足够的消息传递层数。

## 方法详解

### 整体框架

输入 SMILES → 构建聚合物图（重复单元 + 环化连接）→ Max-GNN 编码 → 全局 Max 池化 → 属性预测。

### 关键设计

1. **重复不变性的理论基础**

    - 功能：证明 Max 聚合是实现重复不变性的充要条件
    - 核心思路：定理 1 证明只要满足两个条件——(1) 消息传递层数 $\geq$ 图直径/2；(2) 使用 Max 聚合——GNN 对重复数量完全不变
    - 设计动机：Sum/Mean 聚合随重复数量线性/常数变化，违反不变性

2. **环化连接策略**

    - 功能：将线性聚合物链的首尾相连形成环形图
    - 核心思路：聚合物首尾标记（*号）连接，使 1 个重复单元的图与 N 个重复单元的图拓扑等价
    - 设计动机：无环化时，链端原子的邻域在不同重复数下不同，破坏不变性

3. **GRIN 架构**

    - 功能：完整的重复不变 GNN 架构
    - 核心思路：MPNN 骨架（GIN/GAT 等均可）+ Max 聚合替换 Sum/Mean + 环化图输入 + 全局 Max 池化
    - 设计动机：模块化设计，任何 MPNN 变体都可以即插即用

### 损失函数 / 训练策略

标准回归/分类损失（MSE/BCE），无特殊训练策略。关键在于推理时对任意重复数给出一致预测。

## 实验关键数据

### 主实验（聚合物性质预测）

| 方法 | 玻璃化温度 MAE↓ | 带隙 MAE↓ | 介电常数 MAE↓ | 重复不变? |
|------|---------------|----------|------------|---------|
| GNN-Sum | 15.3 | 0.45 | 0.12 | ✗ |
| GNN-Mean | 14.8 | 0.42 | 0.11 | ✗ |
| Fingerprint | 16.2 | 0.48 | 0.14 | ✓ |
| **GRIN** | **13.1** | **0.38** | **0.09** | **✓** |

### 消融实验

| 配置 | 玻璃化温度 MAE | 重复不变性误差 |
|------|---------------|-------------|
| Sum 聚合 | 14.8 | 8.5% |
| Mean 聚合 | 15.1 | 3.2% |
| Max 聚合，无环化 | 14.2 | 1.8% |
| **Max 聚合 + 环化** | **13.1** | **<0.01%** |

### 关键发现

- GRIN 在预测精度和重复不变性上同时最优
- Sum/Mean 聚合对重复数变化的误差分别为 8.5% 和 3.2%，GRIN <0.01%
- 环化连接是实现完全不变性的必要条件
- 在 3 个性质预测基准上一致领先

## 亮点与洞察

- **理论驱动**：从对称性出发推导方法，而非纯经验设计。Max 聚合的重复不变性证明是核心贡献。
- **即插即用**：Max 替换 Sum/Mean + 环化，可以升级任何现有 GNN 为重复不变版本。
- **聚合物科学意义**：解决了该领域长期存在的表示歧义问题，使预测更可靠。

## 局限与展望

- Max 聚合可能丢失数量信息（如原子数），对某些性质可能不利
- 仅考虑均聚物（单一重复单元），共聚物需扩展
- 对非常大的消息传递层数，Max 聚合可能过平滑

## 相关工作与启发

- **vs SchNet/DimeNet**：3D 分子 GNN，不考虑聚合物特殊性；GRIN 专门解决重复不变性
- **vs 指纹方法**：传统指纹天然重复不变但表达力弱；GRIN 兼顾不变性和表达力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论驱动的方法设计
- 实验充分度: ⭐⭐⭐⭐ 多性质验证 + 理论验证
- 写作质量: ⭐⭐⭐⭐⭐ 证明严谨，思路清晰
- 价值: ⭐⭐⭐⭐ 材料科学领域的基础性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](gflownets_for_learning_better_drug-drug_interaction_representations.md)
- [\[ACL 2025\] Align-Pro: Align Protein Representations Through Multi-Modal Learning](../../ACL2025/computational_biology/align-pro_align_protein_representations_through_multi-modal_learning.md)
- [\[NeurIPS 2025\] CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation](bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)
- [\[ICLR 2026\] Controlling Repetition in Protein Language Models](../../ICLR2026/computational_biology/controlling_repetition_in_protein_language_models.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](../../ICML2026/computational_biology/sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)

</div>

<!-- RELATED:END -->
