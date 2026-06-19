---
title: >-
  [论文解读] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds
description: >-
  [NeurIPS 2025][黎曼优化] 提出 Riemannian Online to NonConvex (RO2NC) 算法及其零阶版本 ZO-RO2NC，首次为黎曼流形上完全非光滑非凸随机优化建立了 $O(\delta^{-1}\epsilon^{-3})$ 的有限时间样本复杂度保证，匹配欧几里德最优结果。
tags:
  - "NeurIPS 2025"
  - "黎曼优化"
  - "非光滑非凸"
  - "Goldstein 稳定性"
  - "零阶优化"
  - "有限时间分析"
---

# Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds

**会议**: NeurIPS 2025  
**arXiv**: [2510.21468](https://arxiv.org/abs/2510.21468)  
**代码**: 无  
**领域**: 其他  
**关键词**: 黎曼优化, 非光滑非凸, Goldstein 稳定性, 零阶优化, 有限时间分析

## 一句话总结
提出 Riemannian Online to NonConvex (RO2NC) 算法及其零阶版本 ZO-RO2NC，首次为黎曼流形上完全非光滑非凸随机优化建立了 $O(\delta^{-1}\epsilon^{-3})$ 的有限时间样本复杂度保证，匹配欧几里德最优结果。

## 研究背景与动机

**领域现状**：黎曼优化广泛应用于深度学习（正交约束）、PCA、字典学习、低秩矩阵补全等涉及流形结构搜索空间的问题。现有黎曼优化算法主要针对光滑目标函数，包括梯度下降、无投影方法、加速方法等。

**现有痛点**：
   - 非光滑非凸函数的 $\epsilon$-驻点不可求解（NP-hard），需引入替代准则
   - 欧几里德空间中 Goldstein 稳定性近年才被研究（2020-），且已达到最优 $O(\delta^{-1}\epsilon^{-3})$
   - 黎曼设定下，非光滑非凸优化的有限时间分析完全空白——仅有渐近收敛结果

**核心矛盾**：流形几何（不同点的切空间不同、平行移动引入失真、retraction 非线性）使得欧几里德分析工具无法直接迁移

**切入角度**：将欧几里德空间的 O2NC 算法（基于在线学习到非凸优化的转换）适配到黎曼流形，利用 retraction 保证可行性，用平行移动/投影处理不同切空间的向量运算

**核心idea**：通过巧妙选择基准动作 $u_t$（使用平行移动将梯度传输到公共切空间）和分析流形曲率引入的误差项，将欧几里德最优复杂度延伸到黎曼设定

## 方法详解

### 整体框架
RO2NC 基于双层循环结构：外层 epoch $k=1,...,K$，内层迭代 $t=0,...,T-1$。每个 epoch 中，算法通过在线学习子程序生成动作 $\Delta_t$，更新 $x_{t+1} = \text{Retr}_{x_t}(\Delta_t)$，在随机中间点 $w_t = \text{Retr}_{x_t}(s_t\Delta_t)$（$s_t \sim \text{unif}[0,1]$）处计算梯度作为反馈。输出为随机选择的某个 epoch 的中间点。

### 关键设计

1. **Goldstein 稳定性的黎曼推广**:

    - 功能：定义黎曼流形上的 $(\delta,\epsilon)$-驻点概念
    - 核心思路：Riemannian $\delta$-subdifferential 定义为 $\partial_\delta f(x) := \text{cl conv}\{P_{y,x}^g(\partial f(y)): y \in \text{cl } B(x,\delta)\}$，需要平行移动将不同切空间的次微分集搬到同一切空间。点 $x$ 是 $(\delta,\epsilon)$-驻点当 $\|\text{grad } f(x)\|_\delta \leq \epsilon$
    - 设计动机：平行移动保持向量长度（等距性），适合定义距离相关的概念；若用投影则失去等距性

2. **RO2NC 的平行移动版本**:

    - 功能：使用平行移动更新动作 $\Delta_t$ 和反馈梯度 $g_t$
    - 核心思路：$\Delta_{t+1} = \text{clip}(P_{x_t,x_{t+1}}^g(\Delta_t) - \eta g_t')$，其中 $g_t' = P_{w_t,x_{t+1}}^g(g_t)$ 是梯度的平行移动版本。clip 将动作限制在切空间球 $\mathbb{B}_{T_{x_{t+1}}\mathcal{M}}(D)$ 内
    - 设计动机：平行移动保证分析中可以复用在线优化的遗憾界；关键创新在于基准动作的选择：$u_t = \mathcal{P}_{S_t}^s\big(-D \frac{\sum_\tau (\mathcal{P}_{S_{\tau+1}}^s)^{-1} \circ P_{w_\tau,x_{\tau+1}}^g(\nabla_\tau)}{\|\cdot\|}\big)$，将所有梯度传输到公共切空间 $T_{x_0}\mathcal{M}$

3. **RO2NC 的投影版本**:

    - 功能：用计算更高效的投影替代平行移动
    - 核心思路：$\Delta_{t+1} = \text{Proj}_{T_{x_{t+1}}\mathcal{M}}(\Delta_t - \eta g_t)$，在环境空间计算后投影回切空间
    - 设计动机：投影在实现上更高效（无需计算测地线），虽失去等距性但通过 Lemma 2.8 控制失真：$\|P_{0,t}^\gamma(v) - \text{Proj}(v)\| \leq C\|v\| \cdot \text{length}(\gamma)$

4. **零阶 ZO-RO2NC**:

    - 功能：在梯度不可用时用函数值查询估计梯度
    - 核心思路：黎曼梯度估计器 $g_\delta(x) = \frac{d}{2\delta}(F(\text{Exp}_x(\delta u), \nu) - F(\text{Exp}_x(-\delta u), \nu))u$，其中 $u$ 从切空间单位球均匀采样。定义辅助函数 $h_\delta(x) = \int f \circ \text{Exp}_x(u) dp_x(u)$ 建立梯度估计与 Goldstein 次微分的关系
    - 设计动机：在切空间采样（而非流形上）避免流形体积计算；通过 Lemma 4.1 控制梯度估计与次微分集的距离

### 训练策略
- 步长 $\eta = D/G\sqrt{T}$，裁剪参数 $D = \delta/T$
- 外层 $K$ 个 epoch，内层 $T$ 步迭代，总样本 $N = KT$
- 输出为随机选择的 $\bar{w}_k = w_{k,\lfloor T/2 \rfloor}$

## 实验关键数据

### 主实验 — 稀疏 PCA (球面 $\mathbb{S}^{n-1}$)

目标函数：$\min_{x \in \mathbb{S}^{n-1}} \{-x^\top A x + \mu \|x\|_1\}$

| 设置 | 算法 | 收敛性 |
|------|------|--------|
| 一阶 + 平行移动 | RO2NC (PT) | 梯度范数随 epoch 单调下降 |
| 一阶 + 投影 | RO2NC (Proj) | 与平行移动版本表现相当 |
| 零阶 | ZO-RO2NC | 收敛但速度慢于一阶，对参数更敏感 |

### 理论复杂度对比

| 参考文献 | 方法 | 设置 | 目标 | 收敛率 |
|---------|------|------|------|--------|
| grohs2016 | 次梯度 | 确定性 | 非光滑 | 渐近 |
| chen2024 | 近端梯度 | 确定性 | 复合(Stiefel) | $O(\epsilon^{-2})$ |
| wang2022 | SPIDER | 随机 | 复合(Stiefel) | $O(\epsilon^{-3})$ |
| **本文** | **RO2NC** | **随机** | **完全非光滑** | $O(\delta^{-1}\epsilon^{-3})$ |

### 关键发现
- 投影版本与平行移动版本达到相同复杂度 $O(\delta^{-1}\epsilon^{-3})$，实际表现也相当
- 零阶版本保持相同复杂度，但常数因子更大，对超参数更敏感
- 流形曲率通过相关常数进入收敛界，但不影响阶数

## 亮点与洞察
- **首个完全非光滑黎曼有限时间保证**：此前黎曼非光滑非凸优化仅有渐近结果，本文填补了重要空白
- **匹配欧几里德最优**：$O(\delta^{-1}\epsilon^{-3})$ 复杂度匹配 Cutkosky & Mehta 在欧几里德空间的最优结果
- **基准动作的巧妙选择**：通过平行移动组合构建时间变化的基准 $u_t$，使得可以复用在线优化的遗憾分析
- **零阶估计器的切空间采样**：避免流形体积计算，大幅降低实现复杂度

## 局限与展望
- Goldstein 稳定性定义依赖平行移动选择，其他传输映射可能给出不同结构
- 实验仅在球面上验证，缺乏 Stiefel/Grassmann 等更复杂流形的实验
- 曲率效应的更深入分析和自适应利用曲率信息的算法设计是开放问题
- 理论要求 retraction 曲线满足一、二阶导数界，排除了一些非标准 retraction

## 相关工作与启发
- **vs O2NC (Cutkosky & Mehta)**：欧几里德版本，本文是其黎曼推广，额外处理切空间不一致和曲率失真
- **vs 复合目标方法**：Chen/Wang/Li/Deng 等处理 $f(x)+h(x)$（$f$ 光滑），本文处理完全非光滑
- **vs 零阶黎曼方法**：Li 等的零阶方法仅处理光滑目标，本文首次处理非光滑

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次黎曼完全非光滑非凸有限时间分析，理论贡献突出
- 实验充分度: ⭐⭐⭐ 仅球面稀疏PCA一个实验，验证充分性有限
- 写作质量: ⭐⭐⭐⭐ 技术分析严谨，核心创新点(基准选择、曲率控制)阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 填补了黎曼非光滑优化理论的重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts](../../ICML2026/learning_theory/geometric_and_stochastic_analysis_of_discontinuities_in_sparse_mixture-of-expert.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)
- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](../../ICLR2026/learning_theory/lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[ICML 2026\] Finite-Width Neural Tangent Kernels from Feynman Diagrams](../../ICML2026/learning_theory/finite-width_neural_tangent_kernels_from_feynman_diagrams.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](../../ICML2026/learning_theory/on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)

</div>

<!-- RELATED:END -->
