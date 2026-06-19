---
title: >-
  [论文解读] Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner
description: >-
  [NeurIPS 2025][优化/理论][Shampoo优化器] 通过将Shampoo预条件矩阵分解为特征值和特征基两部分，揭示了学习率嫁接(grafting)实质上是弥补特征值的陈旧性和缩放偏差，并提出了特征值校正和自适应特征基更新频率来替代这些启发式技巧。 Shampoo是一种基于Kronecker因子分解的预条件随机…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "Shampoo优化器"
  - "预条件矩阵"
  - "Kronecker因子"
  - "学习率嫁接"
  - "特征基自适应更新"
---

# Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner

**会议**: NeurIPS 2025  
**arXiv**: [2506.03595](https://arxiv.org/abs/2506.03595)  
**代码**: 暂无  
**领域**: 优化  
**关键词**: Shampoo优化器, 预条件矩阵, Kronecker因子, 学习率嫁接, 特征基自适应更新

## 一句话总结

通过将Shampoo预条件矩阵分解为特征值和特征基两部分，揭示了学习率嫁接(grafting)实质上是弥补特征值的陈旧性和缩放偏差，并提出了特征值校正和自适应特征基更新频率来替代这些启发式技巧。

## 研究背景与动机

Shampoo是一种基于Kronecker因子分解的预条件随机梯度优化器，近期在AlgoPerf神经网络训练竞赛的外部调参赛道中获胜，引发了对非对角预条件方法的新一轮关注。

**核心痛点**：Shampoo的成功严重依赖两个关键启发式技巧：(1) **学习率嫁接**——用Adam的逐层更新幅度重新缩放Shampoo的更新方向；(2) **陈旧预条件**——每100步才重新计算预条件矩阵的逆根。这两个技巧增加了算法复杂度、引入额外超参数，且缺乏理论解释。实验表明，不使用嫁接的Shampoo在多种超参数下都表现不佳，甚至差于AdamW。

**已有工作的不足**：虽然SOAP提出了特征值校正版Shampoo，效果良好，但缺乏系统性地理解嫁接扮演了什么角色、为什么需要它、以及如何原理性地替代它。

**核心idea**：将预条件矩阵的更新解耦为特征值更新和特征基更新两部分。嫁接实质上是在弥补Shampoo特征值的陈旧性和因Kronecker结构导致的缩放偏差。通过直接校正特征值，可以消除对嫁接的需求；通过自适应判断特征基的更新频率，可以提升计算效率。

## 方法详解

### 整体框架

Shampoo将全矩阵Adam的 $mn \times mn$ 预条件矩阵近似为两个较小矩阵的Kronecker积：$\mathbf{C}_t^{\text{Shampoo}} = (\mathbf{R}_t \otimes \mathbf{L}_t)^{1/2}$，其中 $\mathbf{L}_t \in \mathbb{R}^{m \times m}$, $\mathbf{R}_t \in \mathbb{R}^{n \times n}$。本文通过特征分解视角分析这个近似的误差来源。

### 关键设计

1. **特征值与特征基的解耦分析**：Shampoo的更新可以分解为三步：(a) 用特征基将梯度旋转到新坐标系 $\tilde{\mathbf{G}}_t = \mathbf{Q}_{\mathbf{L}_t}^\top \mathbf{G}_t \mathbf{Q}_{\mathbf{R}_t}$；(b) 对旋转后的梯度进行坐标级缩放；(c) 旋转回原坐标系。通过Lemma 1证明，更新幅度完全由特征值决定（特征基不影响幅度），且Shampoo的幅度界有额外的维度依赖因子 $m^{-p/2} n^{-p/2}$：

$$m^{-p/2}n^{-p/2}\lambda_{\max}^{-p}\|\mathbf{G}\|_F \leq \|\mathbf{U}^{\text{Shampoo}}\|_F \leq m^{-p/2}n^{-p/2}\lambda_{\min}^{-p}\|\mathbf{G}\|_F$$

而全矩阵Adam和EShampoo的幅度界不含这些维度因子。这解释了为什么嫁接对Shampoo如此关键——它弥补了层间因维度差异导致的更新幅度不一致。

2. **特征值校正（EShampoo）**：不将特征值限制为Kronecker积结构，而是在旋转坐标系下独立维护一个密集缩放矩阵 $\mathbf{D}_t = \beta_2 \mathbf{D}_{t-1} + (1 - \beta_2) \tilde{\mathbf{G}}_t^{\odot 2}$，其中 $\tilde{\mathbf{G}}_t^{\odot 2}$ 是变换后梯度的逐元素平方。这使得每步都能以 $O(mn)$ 的代价更新"特征值"，完全消除嫁接的需要。实验验证了三个预测：(a) Shampoo2（更紧的Kronecker近似）配合 $S^{-1}$ 缩放可替代嫁接；(b) 特征值校正可匹配或超越嫁接版Shampoo；(c) Adam的最优学习率可直接迁移到EShampoo。

3. **自适应特征基更新频率**：提出基于QR算法终止准则的自适应判据。对每个Kronecker因子 $\mathbf{L}_t$，计算近似特征值矩阵 $\hat{\Lambda}_{\mathbf{L}_t} = \hat{\mathbf{Q}}_{\mathbf{L}_{t-1}}^\top \mathbf{L}_t \hat{\mathbf{Q}}_{\mathbf{L}_{t-1}}$，如果其非对角元素的相对误差低于阈值 $\tau$：

$$\frac{\|\hat{\Lambda}_{\mathbf{L}_t} - \text{diag}(\hat{\Lambda}_{\mathbf{L}_t})\|_F}{\|\hat{\Lambda}_{\mathbf{L}_t}\|_F} \leq \tau$$

则跳过特征基更新，复用旧的特征基。这样不同层、不同Kronecker因子可以有独立的更新频率。

### 损失函数 / 训练策略

标准神经网络训练目标 $\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$。EShampoo在每步更新特征值校正矩阵 $\mathbf{D}_t$（与Adam grafting相同的内存开销），以固定或自适应频率更新特征基（通过torch.linalg.eigh或warm-started QR算法）。

## 实验关键数据

### 主实验（AlgoPerf子集工作负载）

| 工作负载 | Shampoo变体 | 达标率 | 步数 | 时间(min) |
|---------|-----------|-------|------|----------|
| FastMRI | Adam grafting (F=100) | 4/5 | 4301±109 | 13.96±0.44 |
| FastMRI | EShampoo (F=100) | 5/5 | 2536±66 | 10.44±0.21 |
| FastMRI | EShampoo (τ=0.1, F=50) | 5/5 | 2468±145 | 10.81±0.72 |
| ImageNet ViT | Adam grafting (F=100) | 1/1 | 79907 | 894.27 |
| ImageNet ViT | EShampoo (F=100) | 1/1 | 76226 | 894.85 |
| OGBG | Adam grafting (F=100) | 2/5 | 12574±708 | 39.20±1.88 |
| OGBG | EShampoo (F=100) | 3/5 | 8320±1203 | 33.02±4.05 |
| OGBG | EShampoo (τ=0.1, F=50) | 5/5 | 7117±328 | 27.55±3.49 |

### 消融实验（Imagewoof ViT / ConvNeXt V2）

| 配置 | 效果 | 说明 |
|------|------|------|
| Shampoo (F=1, 无嫁接) | ViT可匹配嫁接版, ConvNeXt V2不行 | 验证维度差异导致的缩放问题 |
| Shampoo2 + $S^{-1}$ (F=1) | 匹配或超越嫁接版 | 更紧的Frobenius近似 |
| EShampoo (F=100) | 匹配或超越嫁接版 | 特征值校正消除嫁接需求 |
| τ=0.99 (仅首次计算特征基) | 显著优于AdamW (0.12 vs 0.3) | 单次特征基计算即带来巨大收益 |
| 1D参数用Adam + 2D参数用EShampoo | 性能等价于全EShampoo | 1D参数的特征基更新完全不必要 |
| 2D参数用Adam + 1D参数用EShampoo | 接近纯Adam的性能 | 收益完全来自2D参数的特征基 |

### 关键发现

- **特征基更新的重要性集中在训练早期**：前10%迭代使用高精度特征基（$\tau=0.01$）对最终loss影响远大于后90%
- **不同参数类型的特征基变化速率不同**：bias和LayerNorm参数的特征基变化速度约为权重矩阵的4倍
- **自适应频率相比固定F=100可节省约20%的wall-clock时间**

## 亮点与洞察

- **解耦视角的深刻性**：将嫁接这一黑箱启发式技巧还原为特征值校正问题，既有理论解释（维度依赖的幅度界），又有实证验证
- **实用性强**：EShampoo只需在Shampoo基础上加一个 $mn$ 维缓冲区（与嫁接开销相同），就能消除嫁接并获得更好性能
- **AdamW→EShampoo的超参数迁移**：这对实际部署至关重要，降低了采用新优化器的调参成本
- **"冻结特征基仍强于AdamW"**的发现令人惊讶且有实践价值

## 局限与展望

- 主要实验在小/中等规模模型上进行，缺少大语言模型上的充分验证（仅有3.24亿参数的Llama 3分析）
- 自适应更新频率在某些工作负载（如ImageNet ViT）上略微变差，说明问题相关的最优策略尚待探索
- 尚未解决理论中一个开放问题：如何将近似质量纳入Shampoo的遗憾界
- 对非AdaGrad家族方法（如K-FAC、TNT）的适用性讨论较少

## 相关工作与启发

- 本文与SOAP关系密切，但提供了更深入的理论分析和系统性实验
- 特征基自适应更新的QR准则借鉴了数值线性代数领域的矩阵分解理论
- 对Kronecker因子方法进行"纯化"的思路可推广到所有基于Kronecker近似的优化器

## 评分

- 新颖性: ⭐⭐⭐⭐ 解耦视角有新意，但特征值校正本身已在SOAP中出现
- 实验充分度: ⭐⭐⭐⭐ 覆盖了AlgoPerf工作负载、多种模型架构，消融详细
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导循序渐进，实验验证与理论预测对应清晰
- 价值: ⭐⭐⭐⭐⭐ 对理解和改进Shampoo类优化器有重要推动作用，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo](../../ICML2026/optimization/foam_frequency_and_operator_error-based_adaptive_damping_method_for_reducing_sta.md)
- [\[NeurIPS 2025\] From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications](from_average-iterate_to_last-iterate_convergence_in_games_a_reduction_and_its_ap.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
