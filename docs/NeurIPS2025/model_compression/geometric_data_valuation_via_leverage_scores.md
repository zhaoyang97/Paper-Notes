---
title: >-
  [论文解读] Geometric Data Valuation via Leverage Scores
description: >-
  [NeurIPS 2025][模型压缩][数据估值] 提出基于**统计杠杆分数（leverage scores）**的几何数据估值方法，作为 Data Shapley 值的高效代理，满足对称性、效率性和虚拟玩家等公理，并通过 ridge leverage 扩展解决维度饱和问题，提供 $O(\varepsilon)$ 近似最优的理论保证。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "数据估值"
  - "Shapley值"
  - "杠杆分数"
  - "主动学习"
  - "实验设计"
---

# Geometric Data Valuation via Leverage Scores

**会议**: NeurIPS 2025  
**arXiv**: [2511.02100](https://arxiv.org/abs/2511.02100)  
**代码**: [GitHub](https://github.com/rodrgo/geosh)  
**领域**: 模型压缩 / LLM效率  
**关键词**: 数据估值, Shapley值, 杠杆分数, 主动学习, 实验设计

## 一句话总结

提出基于**统计杠杆分数（leverage scores）**的几何数据估值方法，作为 Data Shapley 值的高效代理，满足对称性、效率性和虚拟玩家等公理，并通过 ridge leverage 扩展解决维度饱和问题，提供 $O(\varepsilon)$ 近似最优的理论保证。

## 研究背景与动机

数据估值（Data Valuation）旨在量化训练集中每个数据点对模型性能的贡献，在数据清洗、子集选择、联邦学习激励分配和数据市场定价等场景中具有重要意义。Data Shapley 是理论上最严谨的方法，但其组合爆炸的计算复杂度（需要遍历所有子集的边际贡献）使其在大规模场景中不可行。

现有方法的痛点：
- 计算昂贵，需要多次模型重训练
- 需要访问模型权重和梯度
- 主要聚焦于预训练阶段的数据质量控制，忽略了数据获取决策中的不确定性

本文从**几何和数值线性代数**的角度出发，提出一种模型无关的数据估值方法。

## 方法详解

### 整体框架

核心思想：**leverage score 衡量每个数据点在特征空间中的结构重要性**——高杠杆点跨越独特的方向因而有价值，低杠杆点往往是冗余的。

给定数据矩阵 $X \in \mathbb{R}^{n \times d}$，第 $i$ 个数据点的杠杆分数定义为投影矩阵 $H = X(X^TX)^{-1}X^T$ 的第 $i$ 个对角元素：

$$\ell_i = x_i^T (X^TX)^{-1} x_i$$

归一化得到估值函数：

$$\pi_i = \frac{\ell_i}{\sum_{j=1}^n \ell_j}$$

### 关键设计

**定理 1（Shapley 公理满足性）**：当 $\text{rank}(X) = d$ 时，归一化杠杆分数 $\pi_i$ 在效用函数 $U(S) = \text{span}\{x_i: i \in S\}$ 下满足：
- **对称性**：等价贡献的数据点获得相等估值
- **效率性**：所有估值之和为 1（因为 $\text{Tr}(H) = d$，归一化后求和为 1）
- **虚拟玩家**：不增加子空间维度的数据点估值为 0

**维度饱和问题**：普通杠杆分数在子空间达到满秩 $d$ 后，所有新数据点的边际价值为零。这在实际中是一个严重限制。

**Ridge Leverage 扩展**——引入正则化解决饱和：

$$\ell_i^{(\lambda)} = x_i^T (X^TX + \lambda I)^{-1} x_i$$

Ridge leverage 的关键性质：
- 对任意 $\lambda > 0$，$\ell_i^{(\lambda)} \in (0,1)$，统计维度 $k_\lambda = \sum_i \ell_i^{(\lambda)}$ 严格位于 $(0, d)$ 之间
- 即使 $\text{rank}(X_S) = d$，新增数据点仍保留正值（因为正则化收缩了逆矩阵但不消除边际增益）

**与经典实验设计的联系**：
- D-最优性边际增益：$\log\det(A + xx^T) - \log\det(A) = \log(1 + \ell^{(\lambda)}(x)) > 0$
- A-最优性边际增益：$\text{Tr}((A+xx^T)^{-1}) - \text{Tr}(A^{-1}) = -\|A^{-1}x\|_2^2 / (1+\ell^{(\lambda)}(x)) < 0$

两个经典准则的边际增益都是 ridge leverage 的单调函数。

**定理 3（$\varepsilon$-近似保证）**：按 ridge leverage 概率采样 $m \geq C \frac{k_\lambda + \log(2d/\delta)}{\varepsilon^2}$ 个样本，以概率 $\geq 1-\delta$ 有：

$$\|\hat{\theta} - \theta^\star\|_A \leq 4\varepsilon \|\theta_{\text{lin}}\|_A, \quad R(\hat{\theta}) - R(\theta^\star) \leq 8\varepsilon^2 \|\theta_{\text{lin}}\|_A^2$$

即采样子集训练的模型参数和预测风险与全数据模型 $O(\varepsilon)$-接近。

### 损失函数 / 训练策略

本文方法不涉及传统意义上的训练策略——估值通过闭式计算获得。在主动学习实验中，ridge leverage 用于选择最有信息量的样本进行标注，无需梯度或反向传播。

## 实验关键数据

### 主实验（主动学习实验）

**MNIST 上的主动学习对比（3层 MLP，784→256→64→10）**：
- 初始 100 个标注样本，40 轮主动学习，每轮选 5 个样本

| 方法 | 最终测试准确率 |
|------|-------------|
| Ridge Leverage（自适应 $\lambda$） | **0.846 ± 0.006** |
| K-center | ~0.82 |
| Margin | ~0.83 |
| Entropy | ~0.83 |
| EGL | ~0.81 |
| Random | ~0.80 |

Ridge leverage 在预训练阶段结束后即展现明显优势，且跨试验变异性最低。

**理论 vs 实验验证**：

| 保证类型 | 具体结论 |
|---------|---------|
| 矩阵近似 | $(1-\varepsilon)A \preceq A_S \preceq (1+\varepsilon)A$ |
| 参数近似 | $\|\hat{\theta} - \theta^\star\|_A \leq 4\varepsilon\|\theta_{\text{lin}}\|_A$ |
| 风险近似 | $R(\hat{\theta}) - R(\theta^\star) \leq 8\varepsilon^2\|\theta_{\text{lin}}\|_A^2$ |

### 消融实验

论文为 workshop 论文，未包含系统消融。但通过理论分析了不同 $\lambda$ 选择的影响：
- $\lambda = 0$：退化为普通杠杆分数，存在维度饱和
- $\lambda > 0$：打破饱和，自适应设置 $\lambda = 0.01 \times \text{Tr}(X^TX)/64$ 效果最好

### 关键发现

1. **无需梯度的数据选择**：Ridge leverage 不需要模型梯度或反向传播，仅依赖数据的几何结构
2. **公理性保证**：满足 Shapley 的对称性和效率性（ridge 版不满足虚拟玩家公理，但这是合理的——冗余数据仍有降低方差的价值）
3. **与实验设计的深层联系**：A/D-最优性准则的边际增益都由 leverage score 控制

## 亮点与洞察

- **视角新颖**：将数值线性代数中的经典概念（leverage scores）重新解释为数据估值工具，桥接 NLA 和运筹学
- **计算高效**：$O(nd^2)$ 复杂度计算所有 leverage scores，远低于 Shapley 值的指数复杂度
- **理论优雅**：$\varepsilon$-近似定理基于矩阵 Chernoff 集中不等式，证明紧凑
- **模型无关**：估值仅依赖数据矩阵，不需要指定或训练任何模型

## 局限与展望

1. **Workshop 论文**：实验规模有限（仅 MNIST），未在大规模数据集或 LLM 场景验证
2. **线性假设**：理论保证基于 ridge 回归（线性模型），非线性场景需进一步研究
3. **不满足线性公理**：归一化杠杆分数不满足 Shapley 的线性公理，限制了多任务聚合
4. **嵌入空间依赖**：主动学习实验中需要预训练模型的倒数第二层嵌入，方法的有效性依赖于嵌入质量
5. 未与更先进的主动学习方法（BALD、BatchBALD、LESS、TRAK）对比

## 相关工作与启发

- **Data Shapley**：本文的理论对标物，leverage scores 提供了高效但略弱的替代
- **A/D-最优实验设计**：ridge leverage 自然联系到最优实验设计的经典准则
- **随机数值线性代数**：leverage score 采样在随机化最小二乘中有广泛应用
- **启发**：可将 ridge leverage 扩展到特征空间（如 kernel leverage scores）以处理非线性问题，或用于 LLM 的训练数据筛选

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 几何视角独特，联系优雅
- **技术深度**: ⭐⭐⭐⭐ — 理论证明严谨，公理验证完整
- **实验充分度**: ⭐⭐ — Workshop 论文实验较少，仅 MNIST
- **实用性**: ⭐⭐⭐ — 计算高效但适用场景待拓展
- **总体**: ⭐⭐⭐⭐

## 背景与动机

## 核心问题

## 方法详解

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Predictive Data Selection: The Data That Predicts Is the Data That Teaches](../../ICML2025/model_compression/predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [\[ICML 2026\] Beyond Temperature: Hyperfitting as a Late-Stage Geometric Expansion](../../ICML2026/model_compression/beyond_temperature_hyperfitting_as_a_late-stage_geometric_expansion.md)
- [\[ICML 2026\] The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models](../../ICML2026/model_compression/the_shape_of_addition_geometric_structures_of_arithmetic_in_large_language_model.md)
- [\[NeurIPS 2025\] Weight Weaving: Parameter Pooling for Data-Free Model Merging](weight_weaving_parameter_pooling_for_data-free_model_merging.md)
- [\[NeurIPS 2025\] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)

</div>

<!-- RELATED:END -->
