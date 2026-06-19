---
title: >-
  [论文解读] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers
description: >-
  [AAAI 2026][位置编码] Tab-PET 提出从表格特征间关联关系中估计图结构，利用图拉普拉斯特征向量构造位置编码（PE）注入 Tabular Transformer，理论和实验均证明 PE 可降低嵌入的有效秩从而提升泛化，在 50 个数据集上为 TabTransformer / SAINT / FT-Transformer 带来一致改进，且 Spearman 关联图效果最佳。
tags:
  - "AAAI 2026"
  - "位置编码"
  - "表格数据"
  - "图拉普拉斯"
  - "Transformer"
  - "有效秩"
---

# Tab-PET: Graph-Based Positional Encodings for Tabular Transformers

**会议**: AAAI 2026  
**arXiv**: [2511.13338](https://arxiv.org/abs/2511.13338)  
**代码**: [https://github.com/kentridgeai/Tab-PET](https://github.com/kentridgeai/Tab-PET)  
**领域**: 表格数据学习 / Transformer  
**关键词**: 位置编码, 表格数据, 图拉普拉斯, Transformer, 有效秩

## 一句话总结

Tab-PET 提出从表格特征间关联关系中估计图结构，利用图拉普拉斯特征向量构造位置编码（PE）注入 Tabular Transformer，理论和实验均证明 PE 可降低嵌入的有效秩从而提升泛化，在 50 个数据集上为 TabTransformer / SAINT / FT-Transformer 带来一致改进，且 Spearman 关联图效果最佳。

## 研究背景与动机

**领域现状**：表格数据是机器学习最常见的数据形式之一，GBDTs（XGBoost、CatBoost）长期主导。近年 TabTransformer、SAINT、FT-Transformer 等 Transformer 架构在表格领域取得不错进展，但整体仍未稳定超越 GBDTs。

**现有痛点**：图像有空间局部性、文本有序列顺序，Transformer 在这些模态可利用位置编码（PE）注入归纳偏置。而表格数据**特征顺序任意、缺乏天然结构先验**，现有 Tabular Transformer 一律不使用 PE，学界共识是"表格数据无结构，PE 无用"。

**核心矛盾**：表格数据面临 (a) 样本稀缺、(b) 高维异构特征、(c) 无结构先验三重困难，Self-attention 在无 PE 时将所有特征视为完全等价的无序集合，无法利用特征间潜在的关联结构来简化学习任务。

**本文目标**：能否为表格 Transformer 构造有意义的位置编码？PE 能否真的提升泛化性能？如果能，应该从什么角度构造 PE？

**切入角度**：作者从**有效秩（effective rank）** 的理论分析出发，发现 PE 可以降低 CLS token 输出嵌入的有效秩（内在维度），这相当于降低了学习问题的维度从而提升泛化。当 PE 与数据实际结构对齐时，有效秩下降更显著。

**核心 idea**：从特征关联图中提取拉普拉斯特征向量作为固定 PE，注入 Tabular Transformer，利用 PE 降低嵌入有效秩的性质来强化泛化。

## 方法详解

### 整体框架

Tab-PET 的 pipeline 分四步：  
**输入** → (1) 数据预处理（分类特征 one-hot 编码 + 连续特征标准化） → (2) 特征级图估计（每个特征是一个节点，边权反映特征间关联） → (3) 从图拉普拉斯中提取特征向量构造 PE → (4) PE 与原始 embedding 拼接后送入 Transformer 层 → **输出**预测。

整个流程不修改 Transformer 内部结构，只在 embedding 层增加了 PE 拼接，属于即插即用的增强方案。

### 关键设计

1. **图估计（Graph Estimation）**

    - 功能：在特征维度上构造图，每个特征是节点，边权刻画特征间的统计依赖或因果关系。
    - 核心思路：探索两类图估计范式——
        - **因果图**：假设线性结构方程模型 $\mathbf{x} = \mathbf{W}\mathbf{x} + \boldsymbol{\epsilon}$，用 LiNGAM（利用非高斯性识别因果方向）或 NOTEARS（连续优化 + 无环约束）学习有向无环图。
        - **关联图**：直接用成对统计量 $w_{ij} = \rho(x_i, x_j)$ 构造边权，$\rho$ 可选 Pearson 相关、Spearman 秩相关或互信息（Chow-Liu 算法保证树结构 DAG）。
    - 设计动机：特征间关联结构在表格数据中是隐式存在的（如金融数据中收入和消费高度相关），图估计把这种隐式结构显式化。实验证明关联图优于因果图——因果图太稀疏（图熵低），关联图更稠密能捕获更丰富的特征依赖。

2. **位置编码构造（PE Creation）**

    - 功能：从估计出的图的拉普拉斯矩阵中提取特征向量，作为每个特征的位置编码。
    - 核心思路：先对邻接矩阵做对称化 $\mathbf{A}_{\text{sym}} = \frac{1}{2}(\mathbf{A} + \mathbf{A}^\top)$，计算图拉普拉斯 $\mathbf{L} = \mathbf{D} - \mathbf{A}_{\text{sym}}$，取前 $k$ 个和后 $k$ 个特征向量（排除常值第一个），标准化后拼接为 PE 矩阵 $\mathbf{P} = [\mathbf{e}_2, \dots, \mathbf{e}_{k+1}, \mathbf{e}_{d-k+1}, \dots, \mathbf{e}_d]$，再乘以缩放因子 $\mathbf{P}' = \alpha \cdot \mathbf{P}$。
    - 设计动机：低频特征向量捕获图的全局结构（相似特征获得相似编码），高频特征向量捕获局部差异（区分密切相关节点间的细微差别），两者结合在同质和异质图上均有效。$k$ 通过基于谱间隙的自适应算法自动选择，$\alpha$ 在验证集上从 9 个候选值中贪心搜索。

3. **位置编码集成（PE Integration）**

    - 功能：将 PE 与 Transformer 的特征 embedding 拼接。
    - 核心思路：对每个特征 $x_i$，其原始嵌入 $\mathbf{z}_i$ 与缩放后的 PE $\mathbf{p}_i'$ 做拼接 $\mathbf{z}_i' = [\mathbf{z}_i; \mathbf{p}_i'] \in \mathbb{R}^{n+2k}$，然后送入 self-attention 层。对于类别特征的多个 one-hot 节点，取其 PE 的均值作为该特征的统一编码。
    - 设计动机：拼接而非相加，保留了原始 embedding 信息不被 PE 覆盖，同时让模型可以通过 attention 自行学习如何利用位置信息。

### 理论动机：PE 与有效秩

作者从理论上证明了 PE 降低有效秩的能力。有效秩定义为 CLS 嵌入矩阵奇异值分布的 Shannon 熵的指数：

$$r_{\text{eff}}(\mathbf{X}) = \exp\left(-\sum_{i=1}^{r} \tilde{\sigma}_i \log \tilde{\sigma}_i\right)$$

- **定理 1（随机输入）**：即使特征独立同分布，PE 也能降低有效秩，上界约为 $r_{\text{eff}} \approx 1 + d/C_\alpha$，其中 $C_\alpha$ 随 PE 缩放因子 $\alpha$ 指数增长。无 PE 时 $\tau=0$，有效秩显著更大。
- **定理 2（结构化输入）**：当 PE 与数据结构对齐（相似特征分配相同 PE）时，有效秩上界进一步从 $1 + d/(2C_\alpha)$ 降至 $1 + 1/C_\alpha$，降幅巨大。

这意味着**PE 是一种隐式的维度约简工具**，对齐数据结构的 PE 效果更强。

## 实验关键数据

### 主实验：图估计方法对比

| 图估计方法 | 类型 | 分类提升(%) | 回归提升(%) | 额外耗时(min) |
|-----------|------|-----------|-----------|-------------|
| NOTEARS | 因果 | 1.36 | 3.64 | 76.83 |
| LiNGAM | 因果 | 1.41 | 3.97 | 10.96 |
| Pearson | 关联 | 1.61 | 4.16 | 0.78 |
| **Spearman** | **关联** | **1.72** | **4.34** | **0.79** |
| Chow-Liu | 关联(树) | 1.17 | 4.29 | 0.38 |

Spearman 关联图在分类和回归上均取得最高提升，且计算开销仅约 0.79 分钟。

### Tab-PET vs 可学习 PE

| 方法 | 分类平均提升(%) | 回归平均提升(%) | 分类胜率(%) | 回归胜率(%) |
|------|---------------|---------------|-----------|-----------|
| Learnable PE | 0.04 | 0.62 | 12 | 8 |
| **Tab-PET** | **1.72** | **4.34** | **88** | **92** |

Tab-PET 在 88%/92% 的数据集上优于可学习 PE，说明在小数据场景下固定的结构化 PE 远优于从头学习的 PE。

### 排名对比（50 数据集平均排名，越低越好）

| 模型 | 分类排名 | 回归排名 |
|------|---------|---------|
| XGBoost | 3.40 | 5.20 |
| CatBoost | 3.76 | 2.96 |
| TabTransformer | 7.33 | 7.14 |
| TabTransformer+PET | 5.33 | 5.71 |
| SAINT | 4.52 | 3.64 |
| SAINT+PET | **3.28** | **2.84** |
| FT-Transformer | 4.44 | 4.08 |
| FT-Transformer+PET | **2.44** | **2.88** |

FT-Transformer+PET 在分类上 rank 2.44 排名第一，SAINT+PET 在回归上 rank 2.84 排名第一，均超过 XGBoost 和 CatBoost。

### 关键发现

- **关联图 >> 因果图**：Spearman/Pearson 产生高熵稠密图，因果方法产生低熵稀疏图。稠密图的 PE 包含更丰富的结构信息，带来更大增益。
- **$\alpha$ 的最优范围**：合成实验表明 $\alpha$ 过大（如 10）反而降低性能——PE 信号过强会压制原始特征内容。
- **固定 PE > 可学习 PE**：表格数据集通常较小，可学习 PE 容易过拟合，固定的图结构 PE 更稳健（胜率 88-92% vs 8-12%）。
- **有效秩实证验证**：在 15 个真实数据集上观测到 Tab-PET 的有效秩随 $\alpha$ 指数衰减，且显著低于随机 PE，与理论预测完全吻合。
- **所有 Transformer 架构均获统计显著提升**（Wilcoxon 检验 $p < 0.05$），Tab-PET 不依赖特定架构。

## 亮点与洞察

- **挑战"表格无结构"共识**：学界普遍认为 PE 对表格 Transformer 无用，本文用理论+50 数据集的实验彻底推翻了这一认知，打开了表格 Transformer 的新优化维度。
- **有效秩理论框架精巧**：将 PE 的效果归结为"降低嵌入有效秩 → 降低学习维度 → 提升泛化"这一清晰链条，提供了可量化的理解工具。与随机矩阵理论结合的分析方式可迁移到其他领域的 PE 分析。
- **即插即用且计算廉价**：不修改 Transformer 结构，只需多算一轮 Spearman 相关（<1 min）+ 拉普拉斯特征分解，就能稳定提升性能。这种"免费午餐"式的增强思路可推广到任何使用 Transformer 处理无序输入的场景（如点云、集合学习）。
- **图熵作为图估计质量指标**：发现高图熵关联图 > 低图熵因果图，为图估计方法选择提供了简单实用的诊断标准。

## 局限与展望

- **仅限线性因果模型**：因果图估计假设线性 SEM，对非线性因果关系可能欠拟合。可尝试用非线性因果发现方法（如 CGNN、DAG-GNN）。
- **图估计依赖训练集**：PE 从训练数据估计，测试集分布漂移时图结构可能失效。缺乏对分布偏移鲁棒性的讨论。
- **one-hot 编码导致维度膨胀**：高基数类别特征 one-hot 后节点数暴增，图拉普拉斯计算和特征分解开销可能不可忽略。
- **未与非 Transformer 深度模型比较**：如 TabNet、NODE 等也是表格深度学习的重要基线，缺乏对比。
- **$\alpha$ 和 $k$ 需要验证集调参**：虽然自动 $k$ 选择减少了调参负担，但 $\alpha$ 仍需从 9 个候选值中搜索，增加训练成本。
- **改进思路**：可以探索动态 PE（随训练迭代更新图结构）、针对特征子集的局部 PE、与 attention 权重联合优化的自适应 PE 等方向。

## 相关工作与启发

- **vs FT-Transformer (Gorishniy et al. 2021)**：FT-Transformer 为每个特征创建可学习 embedding 但不使用 PE，Tab-PET 在其基础上仅增加 PE 拼接就将排名从 4.44 提升到 2.44（分类），证明 PE 是 FT-Transformer 遗漏的重要组件。
- **vs SAINT (Somepalli et al. 2021)**：SAINT 使用行间+行内双重 attention 且明确声称 PE 对表格无用。Tab-PET 直接在 SAINT 上加 PE 获得显著提升，反驳了 SAINT 的论断。
- **vs 图 Transformer 中的 PE (Dwivedi & Bresson 2020, Ito et al. 2025)**：Tab-PET 借鉴了图 Transformer 的拉普拉斯特征向量 PE 思路，但创新地用于没有显式图结构的表格数据——先估计图再提取 PE，属于"无中生有"地创造结构先验。
- **vs GBDTs (XGBoost, CatBoost)**：Tab-PET 使 Transformer 在 50 数据集的平均排名上首次超过 GBDTs，这是表格深度学习领域的里程碑式结果。

## 评分

- 新颖性: ⭐⭐⭐⭐ 挑战"表格数据不需要 PE"的广泛共识，有效秩理论分析是优雅的新视角，但图拉普拉斯 PE 本身在图领域已有先例
- 实验充分度: ⭐⭐⭐⭐⭐ 50 个数据集、3种 Transformer 架构、5种图估计方法、合成+真实实验、消融/统计显著性检验一应俱全
- 写作质量: ⭐⭐⭐⭐ 理论和实验叙述清晰，Figure 1 的框架图直观易懂，但部分定理的假设条件较强
- 价值: ⭐⭐⭐⭐ 提供了即插即用的增强方案且几乎零额外计算开销，对表格 Transformer 从业者有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Neural Graph Navigation for Intelligent Subgraph Matching](neural_graph_navigation_for_intelligent_subgraph_matching.md)
- [\[AAAI 2026\] Structure-Aware Encodings of Argumentation Properties for Clique-width](structure-aware_encodings_of_argumentation_properties_for_clique-width.md)
- [\[ICLR 2026\] The Counting Power of Transformers](../../ICLR2026/others/the_counting_power_of_transformers.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](../../ICLR2026/others/harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)

</div>

<!-- RELATED:END -->
