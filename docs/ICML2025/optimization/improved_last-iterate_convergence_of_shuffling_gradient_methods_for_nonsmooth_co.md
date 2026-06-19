---
title: >-
  [论文解读] Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization
description: >-
  [ICML 2025][优化/理论][shuffling gradient] 首次证明 Random Reshuffle（RR）和 Single Shuffle（SS）在非光滑（强）凸有限和优化中的 last-iterate 收敛率严格优于 Proximal GD，RR 达到 $\tilde{O}(GD_\star / (n^{1/4}\sqrt{K}))$，近似匹配下界 $\Omega(1/(n^{1/4}\sqrt{K}))$。
tags:
  - "ICML 2025"
  - "优化/理论"
  - "shuffling gradient"
  - "Random Reshuffle"
  - "last-iterate convergence"
  - "nonsmooth convex"
  - "suffix average"
---

# Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization

**会议**: ICML 2025  
**arXiv**: [2505.23056](https://arxiv.org/abs/2505.23056)  
**代码**: 无  
**领域**: 优化理论  
**关键词**: shuffling gradient, Random Reshuffle, last-iterate convergence, nonsmooth convex, suffix average

## 一句话总结

首次证明 Random Reshuffle（RR）和 Single Shuffle（SS）在非光滑（强）凸有限和优化中的 last-iterate 收敛率严格优于 Proximal GD，RR 达到 $\tilde{O}(GD_\star / (n^{1/4}\sqrt{K}))$，近似匹配下界 $\Omega(1/(n^{1/4}\sqrt{K}))$。

## 研究背景与动机

**领域现状**：Shuffling 梯度方法（RR/SS/IG）是求解有限和优化 $\min_x F(x) = f(x) + \psi(x)$（其中 $f = \frac{1}{n}\sum_{i=1}^n f_i$）最广泛使用的实践算法。不同于 SGD 随机采样单个分量，shuffling 方法在每个 epoch 中按排列顺序遍历所有 $n$ 个分量。在光滑凸优化中已证明 RR/SS 的 last-iterate 达到最优收敛率。

**现有痛点**：Liu & Zhou (2024b) 建立了首个 last-iterate 收敛结果，但其非光滑情况下的界为 $O(GD_\star/\sqrt{K})$——这对任意 shuffling 策略成立，与 Proximal GD 相当，完全无法体现 RR/SS 中排列随机性带来的加速优势。Koren et al. (2022) 证明了 average iterate 在 RR 下的 $O(1/(n^{1/4}\sqrt{K}))$ 速率，但 last-iterate 的改进一直是开放问题。

**核心矛盾**：非光滑性使得标准分析技术（依赖梯度 Lipschitz 连续性）失效，而排列结构引入的梯度间相关性使分析复杂度远超独立采样的 SGD。

**本文目标**：回答 Liu & Zhou (2024b) 提出的开放问题——在非光滑（强）凸优化中，RR/SS 的 last-iterate 是否可以证明比 Proximal GD 更快？

**切入角度**：利用 RR/SS 排列结构的精细分析，在步级（而非仅 epoch 级）建立递推不等式，提取出排列随机性带来的额外 $n^{-1/4}$ 或 $n^{-1/2}$ 加速因子。

**核心 idea**：RR 的 last-iterate 达到 $\tilde{O}(GD_\star/(n^{1/4}\sqrt{K}))$——首次证明非光滑下 shuffling 比 Proximal GD 严格更快。

## 方法详解

### 整体框架

本文研究的算法是 General Proximal Gradient Method（Algorithm 1）：在每步 $t$，选择索引 $\mathsf{i}(t) \in [n]$，执行 $x_{t+1} = \arg\min_x \psi(x) + \langle \nabla f_{\mathsf{i}(t)}(x_t), x\rangle + \|x - x_t\|^2/(2\eta_t)$。索引的生成方式决定了 RR（每 epoch 重新随机排列）、SS（全程用同一排列）或 IG（固定排列）。与现有 work 不同，Algorithm 1 在每步都执行近端更新（而非仅 epoch 末），且适用于任意 $T \in \mathbb{N}$（不要求 $T = Kn$）。

### 关键设计

1. **RR 的一般凸 last-iterate 收敛（Theorem 4.2）**:

    - 功能：建立 RR 下 last-iterate 的首个加速收敛率
    - 核心思路：设 $T = Kn$，步长 $\eta_t = \eta/\sqrt{t}$，则 $\mathbb{E}[F(x_{Kn+1}) - F(x_\star)] = \tilde{O}(G_{f,2} D_\star / (n^{1/4}\sqrt{K}))$。当 $K = \Omega(\log n)$ 时对数因子可消除。关键技术创新在于逐步分析函数值变化，利用排列结构中"每 epoch 恰好遍历所有 $n$ 个分量"的性质来提取 $n^{-1/4}$ 加速因子
    - 设计动机：相比 Liu & Zhou (2024b) 的 $O(1/\sqrt{K})$，本文的 $\tilde{O}(1/(n^{1/4}\sqrt{K}))$ 快了 $\Theta(n^{1/4})$ 倍，首次体现了 RR 排列随机性在非光滑情况下的加速作用

2. **RR 的 suffix average 最优性（Corollary 4.3）**:

    - 功能：作为 Theorem 4.2 的推论，首次证明 suffix average（最后一个 epoch 的平均）也达到 $\tilde{O}(1/(n^{1/4}\sqrt{K}))$
    - 核心思路：由 last-iterate 的界直接蕴含最后 $n$ 个迭代点的平均值的界。这一结果近似匹配 Koren et al. (2022) 的下界 $\Omega(1/(n^{1/4}\sqrt{K}))$，填补了上下界之间的空白
    - 设计动机：Koren et al. 仅对 average iterate 证明了上界但缺少对应的 last-iterate 和 suffix average 结果。本文从 last-iterate 出发同时解决了两个问题

3. **SS 的 last-iterate 收敛（Theorems 4.5, 4.6）**:

    - 功能：对 Single Shuffle 建立 last-iterate 收敛率
    - 核心思路：一般凸下：$\tilde{O}(GD_\star/(n^{1/4}K^{1/4}) \lor GD_\star/\sqrt{n})$。在约束优化特殊情况下（$\psi = I_{\mathcal{C}}$）可改进为 $\tilde{O}(GD_\star/(n^{1/4}K^{1/4}) \land GD_\star/\sqrt{K})$，对任意 $K$ 都优于 Liu & Zhou。强凸下：$\tilde{O}(\mu D_\star^2/(n^2K^2) + G^2/(\mu\sqrt{nK}) + G^2/(\mu n))$
    - 设计动机：SS 只在开始随机选一次排列，其随机性弱于 RR（每 epoch 重新排列）。SS 的收敛率确实弱于 RR（$K^{1/4}$ vs $\sqrt{K}$），但仍严格优于 Proximal GD，揭示了"哪怕单次排列"也带来加速

### 损失函数 / 训练策略

本文是纯理论工作。假设条件为：每个 $f_i$ 凸且 $G_i$-Lipschitz（仅在 $\text{dom}\psi$ 上），$\psi$ proper 闭凸且可能 $\mu$-强凸（$\mu \ge 0$）。步长选择为 $\eta_t = \eta/\sqrt{t}$（一般凸）或 $\eta_t = \eta/t$（强凸），其中 $\eta$ 根据 $G, D_\star, n$ 最优选取。论文还提出了一个保证 last-iterate 收敛的一般性充分条件，不限于 shuffling 策略。

## 实验关键数据

### 主要收敛率对比（$T = Kn$, 一般凸 $\mu=0$）

| 方法 | 采样 | 收敛率 | 输出 |
|------|------|--------|------|
| Proximal GD | ANY | $O(GD_\star/\sqrt{K})$ | $x_{Kn+1}$ |
| Koren et al. 2022 | RR | $O(GD_\star/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}^{\text{avg}}$ |
| **本文 Thm 4.2** | RR | $\tilde{O}(GD_\star/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}$ |
| **本文 Cor 4.3** | RR | $\tilde{O}(GD_\star/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}^{\text{suffix}}$ |
| **本文 Thm 4.5** | SS | $\tilde{O}(GD_\star/(n^{1/4}K^{1/4}) \lor GD_\star/\sqrt{n})$ | $x_{Kn+1}$ |
| 下界 (Koren 2022) | RR/SS | $\Omega(1/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}^{\text{suffix}}$ |

### 强凸 $\mu > 0$ 收敛率

| 方法 | 采样 | 收敛率 |
|------|------|--------|
| Proximal GD / Liu&Zhou | ANY | $\tilde{O}(\mu D_\star^2/K^2 + G^2/(\mu K))$ |
| **本文 Thm 4.4** | RR | $\tilde{O}(\mu D_\star^2/(n^2K^2) + G^2/(\mu\sqrt{n}K))$ |
| **本文 Thm 4.7** | SS | $\tilde{O}(\mu D_\star^2/(n^2K^2) + G^2/(\mu\sqrt{nK}) + G^2/(\mu n))$ |
| 下界 | ANY | $\Omega(1/(nK))$ |

### 关键发现

- RR 在一般凸下相比 Proximal GD 加速因子为 $\Theta(n^{1/4})$，强凸下为 $\Theta(n^{1/2})$——$n$ 越大加速越明显，精确反映了每 epoch 遍历 $n$ 个分量的优势
- Last-iterate 的界直接蕴含 suffix average 的最优性——说明 last-iterate 分析已足够精细，不需要额外的 averaging 操作
- SS 弱于 RR 但仍优于 GD，说明"仅一次随机排列"也能提供加速
- 对数因子在 $K = \Omega(\log n)$（通常成立）时可消除

## 亮点与洞察

- **非光滑下 shuffling 严格优势的首次证明**：终结了"non-smooth 场景下 shuffling 是否优于 GD"的理论争论。加速来源于排列保证每 epoch 均匀覆盖所有分量，减少了梯度估计的方差
- **Last-iterate 蕴含 suffix average 最优性**：这一优雅的理论结果表明 last-iterate 分析可以"一石二鸟"，实践中直接使用最后一个迭代点即可，无需额外存储或计算历史平均
- **分析技术的突破**：逐步（而非逐 epoch）分析函数值变化的递推技术，以及处理排列依赖性的新方法，可推广到其他非光滑有限和优化的分析中

## 局限性

- 强凸 RR 的速率 $\tilde{O}(G^2/(\mu\sqrt{n}K))$ 是否最优仍为开放问题——下界仅为 $\Omega(1/(nK))$
- IG（确定性排列）在非光滑情况下是否优于 GD 未涉及
- 假设 $f_i$ 凸且 Lipschitz 限制了对非凸深度学习的直接适用性
- 步长选择依赖 $G$, $D_\star$, $n$ 等参数的先验知识，自适应步长分析是自然后续

## 相关工作与启发

- **vs Liu & Zhou (2024b)**：光滑时已最优但非光滑仅达 GD 水平（$O(1/\sqrt{K})$）；本文将非光滑 last-iterate 改进到 $\tilde{O}(1/(n^{1/4}\sqrt{K}))$，填补了最后一块拼图
- **vs Koren et al. (2022)**：仅对 average iterate 建立上界和对 suffix average 建立下界；本文首次证明 last-iterate 也达到近最优速率，同时蕴含 suffix average 的匹配上界
- **vs Mishchenko et al. (2020)**：经典光滑强凸 shuffling 分析；本文推广到非光滑设置

## 评分

- 新颖性: ⭐⭐⭐⭐ 非光滑情况下shuffling last-iterate优于GD的首次证明，解决重要开放问题
- 实验充分度: ⭐⭐ 纯理论工作，无实验验证
- 写作质量: ⭐⭐⭐⭐⭐ Table 1 结果总结极清晰，定理陈述严谨
- 价值: ⭐⭐⭐⭐ 为shuffling gradient理论填补关键空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications](../../NeurIPS2025/optimization/from_average-iterate_to_last-iterate_convergence_in_games_a_reduction_and_its_ap.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[ICML 2025\] Enhancing Parallelism in Decentralized Stochastic Convex Optimization](enhancing_parallelism_in_decentralized_stochastic_convex_optimization.md)

</div>

<!-- RELATED:END -->
