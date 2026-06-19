---
title: >-
  [论文解读] Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems
description: >-
  [AAAI2026][物理/科学计算][sparse feedback control] 提出随机稀疏反馈控制框架：控制器在每个时间步仅访问状态向量的随机子集，通过 LMI 联合设计反馈增益矩阵和 Bernoulli 稀疏化参数，在保证渐近均方稳定性（AMSS）的同时最小化所需传感器数量，实验中仅用 0.3% 的状态分量即可达到与全状态反馈可比的性能。
tags:
  - "AAAI2026"
  - "物理/科学计算"
  - "sparse feedback control"
  - "randomized sparsification"
  - "asymptotic mean-square stability"
  - "LMI"
  - "large-scale systems"
---

# Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems

**会议**: AAAI2026  
**arXiv**: [2511.13870](https://arxiv.org/abs/2511.13870)  
**代码**: 无  
**领域**: 科学计算  
**关键词**: sparse feedback control, randomized sparsification, asymptotic mean-square stability, LMI, large-scale systems  

## 一句话总结

提出随机稀疏反馈控制框架：控制器在每个时间步仅访问状态向量的随机子集，通过 LMI 联合设计反馈增益矩阵和 Bernoulli 稀疏化参数，在保证渐近均方稳定性（AMSS）的同时最小化所需传感器数量，实验中仅用 0.3% 的状态分量即可达到与全状态反馈可比的性能。

## 背景与动机

经典控制理论假设控制器可以在每个时间步访问完整的状态向量（或输出），但在高维系统或资源受限环境中，测量全状态向量代价高昂。大规模系统（如电力网络）的传感器部署、数据采集和通信带宽都受到严格限制。

现有方法主要从确定性稀疏的角度处理这一问题：

- 通过控制增益矩阵 $K$ 的行/列稀疏化来减少执行器/传感器使用
- 使用 $\ell_1$ 范数最小化获得固定的稀疏结构
- 开环控制策略在确定性稀疏下的稳定性分析

这些方法的共同局限在于：稀疏结构是固定的、确定性的，缺乏对随机化稀疏策略的系统性分析。本文首次研究**随机选择状态子集**作为反馈的控制系统稳定性问题。

## 核心问题

给定离散时间线性系统 $x(k+1) = Ax(k) + Bu(k)$，控制器在每个时间步 $k$ 仅能访问状态向量 $x(k)$ 的随机子集，需要解决：

1. **稳定性保证**：在随机稀疏化下，闭环系统何时能实现渐近均方稳定（AMSS），即 $\lim_{k\to\infty} \mathbb{E}\|x(k)\|_2^2 = 0$？
2. **最小化传感需求**：如何联合设计反馈增益 $K$ 和稀疏化策略，使得所需的期望活跃传感器数量最小？
3. **异构稀疏化**：不同状态分量的测量成本不同时，如何为每个分量分配不同的采样概率？

## 方法详解

### 随机稀疏化策略

在每个时间步 $k$，稀疏化矩阵为对角矩阵：

$$\mathcal{C}(k) = \text{Diag}\left(\frac{c_1(k)}{p_1}, \ldots, \frac{c_n(k)}{p_n}\right)$$

其中 $c_i(k) \sim \text{Ber}(p_i)$ 为独立 Bernoulli 随机变量。除以 $p_i$ 的缩放保证了稀疏化的无偏性：$\mathbb{E}[\mathcal{C}(k)] = I_n$。控制输入为 $u(k) = K\mathcal{C}(k)x(k)$，闭环动态为：

$$x(k+1) = (A + BK\mathcal{C}(k))x(k)$$

### 场景一：均匀稀疏化

所有分量使用相同的 Bernoulli 参数 $p$。核心结果（Proposition 1）给出上界：

$$\mathbb{E}(\|x(k+1)\|_2^2) \leq f(p) \cdot \mathbb{E}(\|x(k)\|_2^2)$$

其中 $f(p) = \|D^\top D + \frac{1-p}{p} \cdot \text{Diag}(L^\top L)\|$，$D = A + BK$，$L = BK$。当 $f(p) < 1$ 时系统 AMSS。

**关键定理（Theorem 1）**：设 $K_\gamma$ 为 LMI 的解，$s_{\max} = \max_i \sum_k l_{k,i}^2$，则当 $p > p_{K_\gamma} = \frac{1}{1 + \alpha_\gamma}$（其中 $\alpha_\gamma = \frac{1 - \|D_\gamma\|^2}{s_{\max}}$）时系统 AMSS。

### 场景二：自适应稀疏化

每个状态分量使用独立的概率 $p_i$。Theorem 2 给出分量级条件：

$$p_i > \frac{1}{1 + \frac{1 - \|D_\gamma\|^2}{s_i}}$$

其中 $s_i = \sum_k l_{k,i}^2$ 反映第 $i$ 个状态分量对控制输入的影响程度。$s_i$ 小的分量可以更频繁地丢弃，$s_i$ 大的分量需要更高采样率。

### LMI 求解框架

存在性条件基于两个假设：(1) $B$ 列满秩；(2) $(I_n - B(B^\top B)^{-1}B^\top)A$ 的最大奇异值 $a_n < 1$。Assumption 2 的直觉是：$A$ 在 $B$ 的像空间正交补方向上的作用必须有界，确保反馈可以通过 $BK$ 充分修正系统动态。

**Algorithm 1**（均匀稀疏化）：
1. 计算 $a_n$，离散化 $[a_n, 1]$
2. 对每个 $\gamma$，解 LMI 得到 $K_\gamma$，计算 $p_{K_\gamma}$
3. 返回最小 $p^\star$ 及对应的 $K$

**Algorithm 2**（自适应稀疏化）：在 Algorithm 1 基础上，为每个 $\gamma$ 计算分量级 $p_{i,K_\gamma}$，最小化加权期望稀疏度 $ES = \sum_i w_i p_i$。

### 观测器扩展

通过分离原理，当 $(A,C)$ 可观/可检测时，Luenberger 观测器重构的状态可以替代真实状态进行随机稀疏化，理论保证不变。

## 实验关键数据

### Grid-Forming Converter（$n=3$）

| 指标 | 数值 |
|------|------|
| 均匀稀疏化 $p^\star$ | 0.79 |
| 自适应稀疏化 $\mathbf{p}^\star$ | [0.026, 0.026, 0.794] |
| 期望传感器数 | 均匀: 2.37/3; 自适应: 0.846/3 |

自适应策略识别出前两个状态分量对控制影响小（$p_i = 0.026$），仅第三个分量需要高采样率。

### 大规模电力系统（$n=1000$，状态维度 2000）

| 指标 | 数值 |
|------|------|
| 均匀稀疏化 $p^\star$ | 0.0026 |
| 期望活跃传感器数 | $\approx 5.2$ / 2000 (**0.26%**) |
| 确定性方法最少测量数 | 37（且为 NP-hard 问题） |

Monte Carlo 模拟验证：$p \geq p^\star$ 时所有轨迹均方收敛至零；$p < p^\star$ 时系统发散。更高的稀疏化（更小 $p$）确实导致更慢的收敛速率，与理论分析（Remark 1）一致。

## 亮点

1. **首次研究随机稀疏反馈**：此前所有稀疏控制工作均为确定性方法，本文开辟了随机稀疏化的新方向
2. **极端稀疏化效果**：1000 节点系统中，仅需 0.3% 状态测量即可保证 AMSS，远优于确定性方法的 37 个测量
3. **理论完整性**：从期望稳定性→均方稳定性→几乎必然稳定性，形成完整的稳定性层次分析
4. **自适应机制**：分量级概率分配使得高影响状态高采样、低影响状态低采样，自然适配异构传感成本
5. **计算可行性**：基于 LMI 的算法可用标准凸优化求解器（MOSEK）高效求解

## 局限与展望

1. **保守性**：基于谱范数不等式的上界分析可能过于保守，$p^\star$ 可能不是紧的下界
2. **无噪声假设**：当前分析未考虑过程噪声和测量噪声，实际系统不可避免存在干扰
3. **线性系统限制**：理论框架仅适用于线性系统，非线性系统的推广需要新的工具
4. **Assumption 2 的强度**：要求 $a_n < 1$ 比经典的可镇定条件更强，限制了适用范围
5. **离散化精度**：Algorithm 1/2 通过网格搜索 $\gamma \in [a_n, 1]$，精度取决于步长 $\delta$
6. **缺乏实物实验**：仅有数值仿真，未在真实物理系统上验证

## 与相关工作的对比

| 方法 | 稀疏类型 | 反馈类型 | 稳定性 | 自适应 |
|------|---------|---------|--------|--------|
| $\ell_1$ 范数方法 | 确定性（固定零结构） | 闭环 | 渐近稳定 | ✗ |
| 行基数约束 | 确定性（混合整数） | 闭环 | 渐近稳定 | ✗ |
| $s$-稀疏控制 | 确定性 | 开环 | 渐近稳定 | ✗ |
| 传感器调度 | 确定性/周期性 | 估计器 | 估计收敛 | ✗ |
| **本文** | **随机（Bernoulli）** | **闭环** | **AMSS** | **✓** |

核心区别：本文是第一个联合设计时变随机传感器选择与控制增益以保证全局均方渐近稳定性的框架。

## 启发与关联

- **分布式学习的启发**：稀疏化策略直接受梯度稀疏化（如 Wangni et al. 的 top-k 稀疏化）启发，体现了优化与控制的深层联系
- **大规模系统适用性**：对智能电网、交通网络等超高维系统有直接应用价值
- **与压缩感知的联系**：随机测量 + 稳定性保证的范式与压缩感知中的随机采样保 RIP 条件有结构相似性
- **通信受限控制**：可自然推广到网络化控制系统中通信带宽受限的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将随机稀疏化引入反馈控制的稳定性分析，问题设置新颖
- 实验充分度: ⭐⭐⭐ — 数值仿真充分验证理论，但缺乏真实系统实验
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，结构清晰，符号定义完整
- 价值: ⭐⭐⭐⭐ — 开辟新方向，对大规模物联网控制系统有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval](../../ICML2026/physics/mathbbr2k_is_theoretically_large_enough_for_embedding-based_top-k_retrieval.md)
- [\[ICLR 2026\] Feedback-driven Recurrent Quantum Neural Network Universality](../../ICLR2026/physics/feedback-driven_recurrent_quantum_neural_network_universality.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](../../NeurIPS2025/physics/why_is_attention_sparse_in_particle_transformer.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/physics/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](../../ICML2025/physics/erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)

</div>

<!-- RELATED:END -->
