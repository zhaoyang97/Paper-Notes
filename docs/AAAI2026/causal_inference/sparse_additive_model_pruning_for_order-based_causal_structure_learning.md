---
title: >-
  [论文解读] Sparse Additive Model Pruning for Order-Based Causal Structure Learning
description: >-
  [AAAI2026][因果推理][causal discovery] 提出 SARTRE 框架，利用随机化树嵌入与组稀疏回归学习稀疏加性模型，替代 CAM-pruning 中基于假设检验的冗余边修剪，在基于拓扑序的因果结构学习中实现显著加速且精度不降。 因果结构学习旨在从观测数据中恢复变量间的因果有向无环图（DAG）…
tags:
  - "AAAI2026"
  - "因果推理"
  - "causal discovery"
  - "sparse additive model"
  - "剪枝"
  - "group lasso"
  - "randomized tree embedding"
---

# Sparse Additive Model Pruning for Order-Based Causal Structure Learning

**会议**: AAAI2026  
**arXiv**: [2602.15306](https://arxiv.org/abs/2602.15306)  
**代码**: 待确认  
**领域**: 因果推理  
**关键词**: causal discovery, sparse additive model, DAG pruning, group lasso, randomized tree embedding

## 一句话总结

提出 SARTRE 框架，利用随机化树嵌入与组稀疏回归学习稀疏加性模型，替代 CAM-pruning 中基于假设检验的冗余边修剪，在基于拓扑序的因果结构学习中实现显著加速且精度不降。

## 背景与动机

因果结构学习旨在从观测数据中恢复变量间的因果有向无环图（DAG）。基于拓扑序的方法（order-based approach）是主流框架之一，分为两步：

1. **排序步**：估计 DAG 的拓扑序，如 SCORE 算法利用 score matching 逐步识别叶节点
2. **修剪步**：从拓扑序诱导的全连接 DAG 中移除伪边，识别每个变量的真正父节点

现有方法大多采用 CAM-pruning 进行修剪，即对每个变量拟合广义加性模型（GAM），再通过假设检验逐一判断候选父节点是否冗余。然而该方法存在两个关键瓶颈：

- **计算代价高**：需要对所有变量反复拟合 GAM，在高维场景下成为整个算法的瓶颈
- **多重检验问题**：对每个变量-候选父节点对重复假设检验，可能因多重检验导致估计精度下降

## 核心问题

如何设计一种高效且准确的修剪方法，避免 CAM-pruning 中反复拟合 GAM 和多重假设检验的开销，同时保持甚至提升因果图估计的准确性？

## 方法详解

### SARTRE 框架总览

SARTRE（Sparse Additive Randomized TRee Ensemble）的核心思想是学习稀疏加性模型来回归每个变量对其候选父节点的关系，从而直接通过系数是否为零来修剪冗余边，无需假设检验。框架分为两个阶段：

### 阶段一：随机化树嵌入

对每个变量 $X_j$，构建一个完全随机化的决策树集成 $h_j$。这些树以无监督方式随机选择分裂点，无需目标变量，因此生成速度极快。每棵树的叶节点对应 $X_j$ 值域上的一个区间 $r_{j,k}$，收集所有叶节点区间得到区间集合 $R_j = \{r_{j,1}, \dots, r_{j,l_j}\}$。

基于区间集合定义指示函数嵌入：

$$\phi_j(X_j) = (\mathbb{I}[X_j \in r_{j,1}], \dots, \mathbb{I}[X_j \in r_{j,l_j}]) \in \{0,1\}^{l_j}$$

**关键优势**：区间集合 $R_j$ 的生成与目标变量无关，因此只需为每个变量生成一次嵌入，可在所有以 $X_j$ 为候选父节点的回归问题中复用。

### 阶段二：组稀疏回归

对每个变量 $X_i$，用加性模型回归其候选父节点：

$$\hat{f}_i(X_{\hat{pa}(i)}) = \sum_{j \in \hat{pa}(i)} g_{i,j}(X_j), \quad g_{i,j}(X_j) = \boldsymbol{\beta}_{i,j}^\top \phi_j(X_j)$$

将所有候选父节点的嵌入和系数向量拼接后，通过 group lasso 回归优化：

$$\min_{\boldsymbol{\beta}_i} \sum_{m=1}^{n} (x_{m,i} - \boldsymbol{\beta}_i^\top \Phi_i(\boldsymbol{x}_m))^2 + \lambda \sum_{j \in \hat{pa}(i)} \|\boldsymbol{\beta}_{i,j}\|_2$$

group lasso 惩罚对每组（对应一个候选父节点）的系数施加 $\ell_2$ 范数惩罚，鼓励整组系数同时变为零。当 $\boldsymbol{\beta}_{i,j} = \boldsymbol{0}$ 时，形状函数 $g_{i,j}(X_j) = 0$ 对所有 $X_j$ 成立，即可直接移除 $X_j \to X_i$ 这条边。

### 理论保证

论文证明了表示能力命题（Proposition 1）：基于区间指示函数的形状函数可以逼近任意连续函数到任意精度。虽然结构比平滑样条简单，但作为分段常数函数具有万能逼近的理论基础。

## 实验关键数据

### 合成数据（ER/SF 图，$n=2000$）

| 设置 | 方法 | SHD | SID | 时间(s) |
|------|------|-----|-----|---------|
| ER1, $d=50$ | SCORE | ~30 | ~250 | ~120 |
| ER1, $d=50$ | DAS | ~28 | ~240 | ~80 |
| ER1, $d=50$ | SARTRE | **~20** | **~200** | **~10** |

- 稀疏图（ER1/SF1）上 SARTRE 的 SHD 和 SID **优于**所有基线
- 稠密图（ER4/SF4）上精度**可比**，但速度快一个数量级

### 高维实验（$d \in \{64, 128, 256, 512\}$，使用真实拓扑序）

SARTRE 在所有维度上均快于 DAS，且 SHD 和 SID 更优。

### 真实数据集

| 数据集 | 方法 | SHD | SID | 时间(s) |
|--------|------|-----|-----|---------|
| Sachs | SCORE | 43.2 | 102.4 | 14.7 |
| Sachs | DAS | 27.6 | 70.3 | 6.42 |
| Sachs | **SARTRE** | **22.7** | **58.0** | **3.62** |
| fMRI | SCORE | 19.6 | 71.8 | 11.4 |
| fMRI | DAS | 11.6 | 58.6 | 4.75 |
| fMRI | **SARTRE** | 12.9 | 60.0 | **3.18** |

## 亮点

1. **思路简洁高效**：将 GAM + 假设检验替换为随机化树嵌入 + group lasso，减少两个计算瓶颈
2. **嵌入复用设计**：区间集合以无监督方式生成，跨所有目标变量复用，避免重复拟合
3. **无需假设检验**：通过 group lasso 的组稀疏性直接判断边是否冗余，消除多重检验问题
4. **即插即用**：可与任意拓扑序估计算法（SCORE、CAM、CaPS 等）组合
5. **高维友好**：在 $d=512$ 的实验中仍表现出色，优于 DAS

## 局限与展望

- **超参数选择**：正则化参数 $\lambda$ 和区间数量 $l_j$ 需预先设定，缺乏自适应调参方法
- **理论保证不足**：仅证明了表示能力，未给出修剪正确性的理论保证
- **假设较强**：假设加性噪声模型（ANM）和非线性可微链接函数，未处理潜在混杂因子
- **稠密图精度略逊**：在稠密 DAG（ER4）且样本量大时，CAM-pruning 的检验方法可能更精确
- 未来可探索自适应调参机制和存在隐变量场景下的鲁棒性

## 与相关工作的对比

| 方法 | 修剪策略 | 是否需要假设检验 | 计算复杂度 | 高维表现 |
|------|----------|-----------------|-----------|---------|
| CAM-pruning | GAM + p-value 检验 | 是 | 高（反复拟合 GAM） | 差 |
| DAS | 先用 score matching 筛选再 CAM-pruning | 是 | 中 | 一般 |
| **SARTRE** | 随机树嵌入 + group lasso | **否** | **低** | **好** |

与 EBM（Lou et al. 2013）等基于树的 GAM 学习框架不同，SARTRE 显式地鼓励组稀疏性以实现变量选择，是专为因果修剪设计的。

## 启发与关联

- **随机树嵌入 + 线性模型**是一种值得借鉴的通用范式：用无监督方式生成非线性特征表示，再用有理论保证的凸优化求解，兼顾效率和可解释性
- Group lasso 的组稀疏性在因果发现中的应用可以推广到其他需要结构化变量选择的场景
- 该方法的"嵌入复用"思想可启发其他需要重复拟合模型的流水线进行类似的计算共享优化

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将随机树嵌入与组稀疏回归结合用于因果修剪，思路新颖且自然
- 实验充分度: ⭐⭐⭐⭐ — 合成/真实数据覆盖全面，含高维实验和消融分析
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、图示直观、算法描述完整
- 价值: ⭐⭐⭐⭐ — 对基于拓扑序的因果发现流水线具有直接实用价值，且 SARTRE 框架可独立用于非线性变量选择

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](../../NeurIPS2025/causal_inference/differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[ICLR 2026\] Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning](../../ICLR2026/causal_inference/beyond_dags_a_latent_partial_causal_model_for_multimodal_learning.md)
- [\[ICML 2025\] Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants](../../ICML2025/causal_inference/causal_effect_identification_in_lvlingam_from_higher-order_cumulants.md)

</div>

<!-- RELATED:END -->
