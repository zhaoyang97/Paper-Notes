---
title: >-
  [论文解读] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization
description: >-
  [NeurIPS 2025][优化/理论][minimax optimization] 提出Ada-Minimax和Ada-BiO两个自适应算法，通过将动量归一化技术与新型在线噪声估计策略结合，首次在无需预知梯度噪声水平的情况下，为非凸-强凹极小极大和非凸-强凸双层优化达到sharp收敛率Õ(1/√T + √σ̄/T^{1/4})。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "minimax optimization"
  - "bilevel optimization"
  - "adaptive algorithm"
  - "momentum normalization"
  - "convergence rate"
---

# Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2509.15399](https://arxiv.org/abs/2509.15399)  
**代码**: 有（随论文附录提交）  
**领域**: 优化理论 / 层次化优化  
**关键词**: minimax optimization, bilevel optimization, adaptive algorithm, momentum normalization, convergence rate

## 一句话总结
提出Ada-Minimax和Ada-BiO两个自适应算法，通过将动量归一化技术与新型在线噪声估计策略结合，首次在无需预知梯度噪声水平的情况下，为非凸-强凹极小极大和非凸-强凸双层优化达到sharp收敛率Õ(1/√T + √σ̄/T^{1/4})。

## 研究背景与动机

**领域现状**：层次化优化（minimax和bilevel）是机器学习的核心问题形式——对抗学习、AUC最大化建模为minimax问题，元学习和超参数优化建模为bilevel问题。已有大量算法提出并建立了收敛保证。**现有痛点**：所有现有方法的收敛率都依赖于预先知道随机梯度噪声的大小σ̄来设定步长等超参数。当噪声水平未知或在训练中变化时，这些方法无法自动适应——在低噪声环境下步长设得保守导致收敛慢，在高噪声环境下步长过大导致不收敛。**核心矛盾**：标准的AdaGrad类自适应方法在单层优化中已经解决了噪声自适应问题，但直接应用到层次化优化中会引入复杂的随机性依赖问题——上下层变量的AdaGrad步长相互耦合，难以分析。**切入角度**：不使用标准AdaGrad，而是采用归一化SGD+动量(NSGDM)处理上层变量，配合AdaGrad-Norm处理下层变量，关键创新在于动量参数本身也自适应调整。

## 方法详解

### 整体框架
Ada-Minimax（Algorithm 1）和Ada-BiO（Algorithm 2）共享相同的算法框架：上层变量通过归一化SGD+动量更新（x_{t+1} = x_t - η_{x,t}·m_t/‖m_t‖），下层变量通过AdaGrad-Norm梯度下降/上升更新。两个算法的核心区别仅在于梯度估计方式——Ada-Minimax直接计算随机梯度，Ada-BiO使用Neumann级数近似超梯度。

### 关键设计

1. **自适应动量归一化(Adaptive NSGDM)**:

    - 功能：上层变量的更新，自动适应噪声水平
    - 核心思路：动量更新 m_t = β_t·m_{t-1} + (1-β_t)·g_{x,t}，归一化步 x_{t+1} = x_t - η_{x,t}·m_t/‖m_t‖。关键在于动量参数α_t = 1-β_t 自适应设定为 α_t = α/√(α² + Σ‖g_{x,k} - g̃_{x,k}‖²)，其中分母的差分项在线估计梯度方差
    - 设计动机：归一化梯度使步长与梯度幅度无关，动量参数根据估计的噪声水平动态调整——噪声大时α_t小(更多动量平滑)，噪声小时α_t大(更快响应)

2. **上下层协调的自适应步长**:

    - 功能：同时为上层和下层变量设定自适应学习率
    - 核心思路：上层步长 η_{x,t} = η_x·α'_t/t，其中α'_t = α/√(α² + Σ‖g_{x,k} - g̃_{x,k}‖² + ‖g_{y,k}‖²)同时包含上下层信息；下层步长 η_{y,t} = η_y/√(γ² + Σ‖g_{y,k}‖²)是标准AdaGrad-Norm形式
    - 设计动机：α'_t同时包含上下层梯度信息，有效控制ratio η_{x,t}/η_{y,t}，确保上下层更新速度协调（Lemma 5.7的关键）

3. **双独立梯度采样的噪声估计**:

    - 功能：在线估计梯度噪声水平
    - 核心思路：每步采两个独立随机梯度g_{x,t}和g̃_{x,t}，用‖g_{x,t} - g̃_{x,t}‖²作为方差的无偏代理量。累积和Σ‖g_{x,k} - g̃_{x,k}‖²在高概率下被控制在[σ̄²t, 4σ̄²t]之间（Lemma 5.5）
    - 设计动机：直接估计方差需要知道真实梯度，而双采样差分仅需额外一次梯度查询即可得到方差的在线估计

### 损失函数 / 训练策略
Ada-Minimax适用于 min_x max_y f(x,y)（f对y强凹），Ada-BiO适用于 min_x f(x,y*(x)) s.t. y*(x) = argmin_y g(x,y)（g对y强凸）。Ada-BiO使用Neumann级数近似超梯度∇Φ(x)，需要额外的二阶信息查询。

## 实验关键数据

### 主实验

| 任务 | 数据集 | 指标 | Ada-Minimax/BiO | TiAda/TFBO | SGDA/PDSM |
|------|--------|------|-----------------|------------|-----------|
| 合成minimax (σ=0) | 1维函数 | ‖∇Φ‖ | **最快收敛** | 可比 | 较慢 |
| 合成minimax (σ=100) | 1维函数 | ‖∇Φ‖ | **收敛** | 不收敛 | 不收敛 |
| Deep AUC最大化 | Sentiment140 | 训练AUC | **比PDSM高20%** | 低 | 次优 |
| Deep AUC最大化 | Sentiment140 | 测试AUC | **比PDSM高2%** | 低 | 次优 |
| 超参数优化 | TREC | 训练/测试ACC | **快速收敛** | TFBO不收敛 | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| α ∈ [0.1, 2.0] | AUC几乎不变 | 对初始动量参数鲁棒 |
| η_y ∈ [0.001, 0.05] | 初始阶段有差异，最终一致 | 下层学习率鲁棒 |
| η_x ∈ [0.001, 0.01] | 影响早期训练动态 | η_x=0.05时最终性能下降 |
| γ ∈ [0.01, 2.0] | 一致性好 | 初始化参数鲁棒 |

### 关键发现
- Ada-Minimax在高噪声(σ=100)下仍能收敛，而TiAda即使充分调参也无法收敛
- 在Deep AUC最大化上，Ada-Minimax无论按epoch还是按wall-clock time都是收敛最快的
- 超参数鲁棒性很强：4个主要超参数(α, η_x, η_y, γ)在宽范围内变化对最终性能影响很小
- 实验验证了噪声下界假设：在AUC任务上经验噪声范围为[0.21, 210.71]

## 亮点与洞察
- 理论首创性：首个为随机层次化优化提供自适应且sharp收敛保证的工作，率先实现了无需噪声先验知识的自动速率插值
- 分析框架通用：先建立单层自适应NSGDM的分析框架（Theorem 5.6），再扩展到minimax和bilevel，思路清晰可迁移
- 超参数鲁棒：四个主要超参数在数量级范围内变化对性能影响极小，极大降低了调参负担
- 实用变体设计：将理论算法中的双采样差分替换为相邻迭代梯度差（计算更省），不影响实际效果

## 局限与展望
- 需要随机梯度噪声具有非零下界的假设（Assumption 3.2(ii)），noiseless情况虽也覆盖但该假设在某些场景不自然
- 仅考虑非凸-强凹/强凸设定，更一般的非凸-非凸情况待探索
- 主要是理论贡献，深度学习实验规模有限（单个数据集+简单模型）
- 每步需要两次独立梯度采样（虽然实际变体可用相邻迭代近似）

## 相关工作与启发
- **TiAda**：唯一尝试自适应的minimax方法，但收敛率对噪声非sharp（不显式依赖σ̄）
- **AdaGrad-Norm**：单层优化的自适应基础，本文将其扩展到层次化优化的下层
- **Cutkosky & Mehta (2020)**：归一化SGD+动量的原始方法，但用固定动量参数；本文关键创新是让动量参数也自适应

## 评分
- 新颖性: ⭐⭐⭐⭐ 自适应+sharp的双重保证在层次化优化中是首次，动量归一化+自适应参数的结合具有原创性
- 实验充分度: ⭐⭐⭐ 合成实验清晰验证理论，但深度学习实验仅1个数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨完整，证明结构层次分明
- 价值: ⭐⭐⭐⭐ 对优化理论社区重要贡献，实用变体的超参数鲁棒性对实践有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings](sharper_convergence_rates_for_nonconvex_optimisation_via_reduction_mappings.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)

</div>

<!-- RELATED:END -->
