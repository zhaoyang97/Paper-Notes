---
title: >-
  [论文解读] Generating Synthetic Relational Tabular Data via Structural Causal Models
description: >-
  [ACL 2025][synthetic data] 本文扩展了 TabPFN 的基于结构因果模型（SCM）的合成数据生成方法，提出了一个能够生成多表关联（relational）合成表格数据的框架，通过耦合节点和隐因果关系实现跨表依赖建模。 合成表格数据生成在近年来受到越来越多关注，尤其在基础模型训练中扮演关键角色…
tags:
  - "ACL 2025"
  - "synthetic data"
  - "relational tables"
  - "structural causal models"
  - "DAG"
  - "tabular foundation model"
---

# Generating Synthetic Relational Tabular Data via Structural Causal Models

**会议**: ACL 2025  
**arXiv**: [2507.03528](https://arxiv.org/abs/2507.03528)  
**代码**: 无  
**领域**: 数据生成 / 表格数据  
**关键词**: synthetic data, relational tables, structural causal models, DAG, tabular foundation model

## 一句话总结

本文扩展了 TabPFN 的基于结构因果模型（SCM）的合成数据生成方法，提出了一个能够生成多表关联（relational）合成表格数据的框架，通过耦合节点和隐因果关系实现跨表依赖建模。

## 研究背景与动机

合成表格数据生成在近年来受到越来越多关注，尤其在基础模型训练中扮演关键角色。TabPFN 利用大量基于 SCM 生成的合成数据集取得了突破性成果，证明了合成数据对表格基础模型的重要性。然而，当前的生成方法存在一个核心局限：

**现实中的表格数据绝大多数是关系型的**——由多张相互关联的表组成（如数据库中的主表和从表），但现有方法仅能生成单张独立表

**基于学习的方法**（如 Synthetic Data Vault、GNN+扩散模型）需要真实世界数据集作为基础提取统计和关系模式，而可访问的真实关系型数据集稀缺

**TabPFN 的 SCM 方法**虽然独立于真实数据、能自然建模因果依赖，但被限制在单表场景

本文的动机明确：将 SCM 方法从单表扩展到关系型多表，使得可以无限量生成具有复杂跨表因果依赖的合成数据，用于训练关系型表格基础模型。

## 方法详解

### 整体框架

方法基于 SCM 的有向无环图（DAG），分三步执行：（1）结构采样——构建图拓扑；（2）预采样——校准噪声尺度和分类边界；（3）主采样——传播数据并读出表格。在此基础上扩展到关系数据生成，通过耦合节点连接多个 DAG。

### 关键设计

1. **结构采样（Structure Sampling）**：使用 Barabási-Albert 模型采样有向图，去除孤立节点和反向边后得到 DAG。根节点数据用多维向量初始化（从正态/伽马分布采样），每个节点定义传播函数 $g_i: \mathbb{R}^{|\text{pa}(i)| \cdot n} \to \mathbb{R}^n$（单层全连接神经网络 + 随机非线性激活函数如 ReLU、logabs）。**与 TabPFN 的关键区别**：分类特征的生成不嵌入传播函数中，而是在读出阶段才离散化——这保证了后续节点接收到连续向量而非被类别数限制的状态。读出通过 pooling 函数（norm、mean、median 等）将 n 维向量投影为标量。

2. **预采样（Pre-Sampling）**：为了让噪声尺度与数据分布匹配，先进行一次无噪声的"预跑"：独立采样根节点数据并传播到全图，计算每个节点的 10% 和 90% 分位数 $q_{0.1}(i), q_{0.9}(i)$。主采样中，噪声按分位数差缩放：
    $x_i = g_i(x_{\text{pa}(i)}) + (q_{0.9}(i) - q_{0.1}(i)) \varepsilon_i$
   这确保传播信号 $g_i$ 仍是主要信息源，噪声是适度扰动。对分类节点，使用 k-means 将预采样数据聚类为 K 个类别，定义分类 pooling 函数为最近质心分配。

3. **关系数据扩展（核心贡献）**：独立采样两个 DAG $\mathcal{G}_{\text{main}}$ 和 $\mathcal{G}_{\text{add}}$，引入耦合节点 C 连接二者（C 由 $\mathcal{G}_{\text{add}}$ 的汇节点导致，指向 $\mathcal{G}_{\text{main}}$ 的特征节点）。为建模隐因果影响，额外将 $\mathcal{G}_{\text{add}}$ 的特征节点连接到 $\mathcal{G}_{\text{main}}$ 的目标节点（黄色边）。数据分两次采样：一次在合并图上（生成主表），一次仅在 $\mathcal{G}_{\text{add}}$+C 上（生成附加表），两表通过 C 列关联，样本量可不同。

4. **分类节点 C 的设计**：耦合节点使用较多类别数（如 175 类），模拟外键列，使两表之间的关联更贴近真实数据库的外键约束。

### 损失函数 / 训练策略

本文是数据生成方法，不涉及模型训练或损失函数。生成过程完全基于随机采样和确定性传播，所有参数（图结构、分布参数、激活函数）均随机初始化。

## 实验关键数据

### 主实验（表格）

基于 Fig. 2 的示例图生成两张关联表（主表 100,000 行、附加表 500 行），使用 EmbDI 嵌入方法 + k-NN 预测，对比仅用主表 vs. 联合两表：

| 目标节点 | 任务类型 | 仅主表 | 主表+附加表 | 趋势 |
|---------|---------|--------|-----------|-----|
| M4 | 分类 (AUC) | ~0.55 | ~0.65 | 联合更好 |
| M6 | 回归 (RMSE) | ~0.45 | ~0.35 | 联合更好 |

（具体数值随嵌入维度变化，在高嵌入维度下联合方法持续优于单表方法）

### 消融实验——参数影响分析

| 参数 | 影响 |
|------|-----|
| 隐维度 n | n 越大，从特征预测目标越难（信息压缩损失更大） |
| 激活函数类型 | 影响数据复杂度和非线性关系的多样性 |
| 噪声比例（分位数选择） | 控制信号 vs 噪声的平衡 |
| 类别数 K | 影响分类任务的难度和类间可分性 |

### 关键发现

1. **跨表依赖有效性验证**：附加表包含的隐因果信息确实影响主表的目标列，且这些信息无法仅从主表的 C 列推断——这正是真实关系型数据的核心特征
2. 在嵌入维度足够高时，联合两表的预测质量稳定优于仅用主表
3. 隐因果连接（黄色边）是关键：没有它们，C 列传播的信息就足以表示所有关系，附加表将不提供额外信息

## 亮点与洞察

- **方法简洁但目标明确**：通过耦合节点和隐因果边将单表 SCM 框架自然扩展至多表，概念清晰、实现直接
- **对 TabPFN 原框架的改进值得注意**：将分类特征生成从传播函数中分离出来，在读出阶段才离散化，这避免了类别数限制后续节点的信息流——这一设计修改对单表场景也适用
- **可扩展性强**：同样的方法可以串联更多 DAG 生成三张及以上的关联表

## 局限与展望

1. **实验规模不足**：仅在一个示例数据集上验证，缺乏大规模系统性实验和参数敏感性分析
2. **数据类型单一**：仅支持数值和分类特征，未涵盖文本、图像等多模态数据
3. **关系结构简单**：仅验证了两张表的关系，三张及以上表的交叉连接、循环依赖等复杂场景未探索
4. **下游任务评估有限**：仅用 EmbDI + k-NN 的简单基线验证，未测试在 tabular foundation model 训练中的实际效果
5. 未与现有关系数据生成方法（如 Synthetic Data Vault、Hudovernik 2024）进行直接对比

## 相关工作与启发

- 直接建立在 TabPFN (Hollmann et al. 2025) 的 SCM 数据生成框架之上
- Synthetic Data Vault (Patki et al. 2016) 是首个基于学习的关系数据生成方法，但依赖真实数据
- Hudovernik (2024) 结合 GNN 嵌入和扩散模型处理关系结构，但同样需要真实数据
- 本文的优势在于完全独立于真实数据、可大规模生成，适合基础模型预训练

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 扩展方向明确且合理，但核心思想是 TabPFN 框架的自然延伸，增量较小
- **实验充分度**: ⭐⭐⭐ — 仅一个示例数据集、数据量化指标不够系统，属于 short paper 级别的验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构紧凑、概念阐述清楚，算法伪代码和图示都很直观
- **价值**: ⭐⭐⭐⭐ — 对 tabular foundation model 社区有潜在价值，但需要更充分的实验来证明实际影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](../../ICLR2026/others/tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](../../NeurIPS2025/others/radar_benchmarking_language_models_on_imperfect_tabular_data.md)
- [\[ACL 2025\] Causal Estimation of Tokenisation Bias](causal_tokenisation_bias.md)

</div>

<!-- RELATED:END -->
