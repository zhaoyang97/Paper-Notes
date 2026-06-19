---
title: >-
  [论文解读] Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework
description: >-
  [ECCV2024][优化/理论][tensor completion] 提出基于可学习张量核范数的多目标张量恢复框架 (MOTC)，通过引入可学习酉矩阵替代固定变换来解决 t-SVD 方法在非光滑张量数据上的性能退化问题，并通过多目标优化有效利用张量各维度的低秩性。 基于张量奇异值分解 (t-SVD) 的张量恢复方法近年来…
tags:
  - "ECCV2024"
  - "优化/理论"
  - "tensor completion"
  - "tensor SVD"
  - "multi-objective optimization"
  - "learnable tensor nuclear norm"
  - "non-smooth tensor recovery"
---

# Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework

**会议**: ECCV2024  
**arXiv**: [2311.13958](https://arxiv.org/abs/2311.13958)  
**代码**: [jzheng20/MOTC](https://github.com/jzheng20/MOTC)  
**领域**: 优化  
**关键词**: tensor completion, tensor SVD, multi-objective optimization, learnable tensor nuclear norm, non-smooth tensor recovery

## 一句话总结

提出基于可学习张量核范数的多目标张量恢复框架 (MOTC)，通过引入可学习酉矩阵替代固定变换来解决 t-SVD 方法在非光滑张量数据上的性能退化问题，并通过多目标优化有效利用张量各维度的低秩性。

## 背景与动机

基于张量奇异值分解 (t-SVD) 的张量恢复方法近年来在彩色图像和视频等视觉数据处理中取得显著成功。这类方法的核心思路是：对张量数据沿某一维度施加固定的可逆变换（如 DFT 或 DCT），然后分析变换后各前片的低秩性来探索张量的全局低秩结构。

然而，现有 t-SVD 方法在面对**非光滑变化**的张量数据时性能严重退化，主要体现在两个场景：

1. **无序图像序列**：分类任务中样本顺序通常是随机的，张量切片的排列变化（Slice Permutation Variability, SPV）会显著影响 t-SVD 结果
2. **内容快速变化的张量数据**：如帧间变化剧烈的视频、不同场景拼接的图像序列

这些问题的根源在于：固定的可逆变换（DFT/DCT）使得 t-SVD 对张量切片的排列顺序和非光滑变化极为敏感。此外，将 t-SVD 推广到高阶张量时，传统加权求和方法（如 WSTNN）需要引入 $h(h-1)/2$ 个权重参数，调参困难。

## 核心问题

1. 如何让 t-SVD 方法在面对非光滑/无序张量数据时仍然有效？
2. 如何高效地将 t-SVD 推广到高阶张量，同时探索不同维度间的相关性，避免大量权重参数？

## 方法详解

### 1. 可学习张量核范数 (Learnable Tensor Nuclear Norm)

传统 t-SVD 方法假设张量可分解为 $\mathcal{M} = \mathcal{Z} \times_{k_3} \hat{U}_{k_3}^T \cdots \times_{k_h} \hat{U}_{k_h}^T$，其中变换矩阵 $\hat{U}$ 是预设的（如 DFT 矩阵）。当张量切片无序时，信息会坍缩到高频切片中，导致低秩假设失效。

本文的关键创新是将固定变换替换为**可学习的酉矩阵** $\{U_{k_n}\}$，提出带可学习张量秩的补全模型 (TC-SL)：

$$\min_{\mathcal{X}, U_{k_n}} \text{rank}_{[k_1,k_2]}(\mathcal{X} \times_{k_{s+1}} U_{k_{s+1}} \cdots \times_{k_h} U_{k_h}) \quad \text{s.t. } \Psi_{\mathbb{I}}(\mathcal{M}) = \Psi_{\mathbb{I}}(\mathcal{X}), \; U_{k_n}^T U_{k_n} = I$$

由于张量秩函数是离散的（NP-hard 问题），利用张量核范数作为最紧凸包络进行松弛，得到最终的 TC-SL 模型：

$$\min_{\mathcal{X}, \tilde{\mathcal{U}}} \|\mathcal{X}\|_{*, \tilde{\mathcal{U}}}^{[k_1,k_2]} \quad \text{s.t. } \Psi_{\mathbb{I}}(\mathcal{M}) = \Psi_{\mathbb{I}}(\mathcal{X})$$

### 2. 多目标张量恢复框架 (MOTC)

TC-SL 依赖特定的 $(k_1, k_2)$ 选择，仅考虑单一模式的低秩性。为同时利用所有维度的相关性，提出多目标模型：

$$\min_{\mathcal{X}, \tilde{\mathcal{U}}_{(k_1,k_2)}} \left[\|\mathcal{X}\|_{*, \tilde{\mathcal{U}}_{(k_1,k_2)}}^{[k_1,k_2]}\right]_{1 \leq k_1 < k_2 \leq h} \quad \text{s.t. } \Psi_{\mathbb{I}}(\mathcal{M}) = \Psi_{\mathbb{I}}(\mathcal{X})$$

每个目标函数考察不同模式对 $(k_1, k_2)$ 的低秩性，无需引入额外权重参数。

### 3. 交替近端乘子法 (APMM)

提出 Alternating Proximal Multiplier Method 求解 TC-SL：

- **更新 $\mathcal{Z}$**：通过张量奇异值阈值操作求解
- **更新 $U_{k_n}$**：通过 SVD 分解 $\mu \mathcal{A}_{(k_n)}\mathcal{B}_{(k_n)}^T + \eta U_{k_n}^{(t)}$ 得到闭式解 $U_{k_n} = UV^T$
- **更新 $\mathcal{E}$**：投影到补集上的闭式解

每次迭代复杂度为 $\mathcal{O}((h-s)(h+I_{(1)}-1)I_{(1)} \prod I_k + h I_{(1)} \prod I_k)$。

### 4. MOTC 的求解

交替执行两步：
- 用 APMM 学习各 $(k_1, k_2)$ 对应的最优变换 $\hat{\mathcal{U}}_{(k_1,k_2)}$
- 用 NSGA-II（多目标遗传算法）求解多目标优化问题，最终取 Pareto 前沿个体的平均作为输出

两步均可并行化加速。

## 实验关键数据

### 图像修复 — 不同场景图像序列 (BSD 数据集)

| 采样率 | TNN-DCT | WSTNN | TC-SL | **MOTC** |
|--------|---------|-------|-------|----------|
| 0.3    | 23.25   | 25.75 | 26.32 | **27.53** |
| 0.5    | 27.25   | 31.07 | 31.55 | **33.25** |
| 0.7    | 32.04   | 37.11 | 38.33 | **39.97** |
| 平均   | 27.51   | 31.31 | 32.06 | **33.58** |

### 图像修复 — 随机打乱序列 (采样率 0.3)

| 数据集   | WSTNN | HTNN-DCT | TC-SL | **MOTC** |
|----------|-------|----------|-------|----------|
| CIFAR10  | 25.02 | 22.12    | 24.63 | **26.46** |
| CIFAR100 | 24.76 | 21.71    | 24.50 | **26.16** |
| LFW      | 34.33 | 30.15    | 31.57 | **35.67** |
| GTF      | 26.66 | 22.11    | 32.10 | **33.56** |
| 平均     | 27.69 | 24.02    | 28.20 | **30.46** |

### 视频修复 — 快速变化帧 (HMDB51, 采样率 0.3)

| 方法 | TNN-DCT | WSTNN | HTNN-DCT | TC-SL | **MOTC** |
|------|---------|-------|----------|-------|----------|
| 平均 PSNR | 30.01 | 35.28 | 27.71 | 36.82 | **38.74** |

MOTC 在视频修复中超越第三名方法 **3.5 dB 以上**，部分视频改进达 **5-10 dB**。TC-SL 相比仅考虑单维低秩的方法提升约 **6.5 dB**。

## 亮点

1. **问题定义精准**：清晰识别 t-SVD 方法在非光滑数据上的根本缺陷（SPV 和非光滑变化），并给出统一的解决方案
2. **可学习变换替代固定变换**：引入可学习酉矩阵是一个自然且优雅的设计，既解决了 SPV 问题，又能适应数据的内在特性
3. **多目标框架避免权重调优**：用多目标优化替代加权求和，不需设定 $h(h-1)/2$ 个权重参数
4. **收敛性保证**：证明了 APMM 收敛到 KKT 点，收敛速率为 $\mathcal{O}(1/\eta^{(t)})$
5. **实验效果显著**：视频修复上 MOTC 提升 3.5+ dB，在部分困难样本上提升 5-10 dB

## 局限与展望

1. **计算开销大**：MOTC 需要对所有 $(k_1,k_2)$ 模式对分别运行 APMM，再加上 NSGA-II 迭代，大规模数据上效率堪忧
2. **NSGA-II 的启发式性质**：多目标优化阶段依赖遗传算法，结果取 Pareto 前沿平均值，缺乏理论最优性保证
3. **仅考虑张量补全任务**：虽然文中提到框架可扩展到去噪、聚类等任务，但未提供实验验证
4. **酉矩阵的初始化**：初始变换选择（如 DFT/DCT）可能影响收敛质量，未讨论对初始化的敏感性
5. **缺少深度学习方法对比**：未与基于深度学习的图像/视频修复方法进行比较

## 与相关工作的对比

| 方法 | 核心思路 | 处理非光滑 | 高阶张量 | 权重调优 |
|------|---------|-----------|---------|---------|
| TNN-DCT/DFT | 固定变换 + 3阶 t-SVD | 否 | 否 | 无 |
| WSTNN | 加权所有模式展开的 TNN | 否 | 是 | 需 $h(h-1)/2$ 个 |
| HTNN-DCT | 高阶 t-SVD + 固定 DCT | 否 | 是（单维） | 无 |
| TC-SL (本文) | 可学习酉矩阵 + t-SVD | **是** | 是（单维） | 无 |
| MOTC (本文) | 多目标 + 可学习核范数 | **是** | **是（多维）** | **无** |

与 HTNN-DCT 的直接对比最能说明可学习变换的价值：TC-SL 在随机打乱数据上平均超出 4+ dB。MOTC 在此基础上进一步利用跨维度相关性，再提升 2+ dB。

## 启发与关联

- **可学习变换的思路**可推广到其他基于固定变换的张量方法（如张量去噪、张量分解），只要将固定变换替换为可学习酉矩阵并用类似的近端方法优化
- **多目标优化替代加权求和**的思路值得在其他需要多个正则项的优化问题中借鉴，避免繁琐的权重调参
- 对于视频理解中帧序列不连续的场景（如视频摘要、关键帧提取后的补全），本文方法具有直接的应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ — 可学习张量核范数和多目标框架的结合是新颖且自然的
- 实验充分度: ⭐⭐⭐⭐ — 多数据集多场景验证，改进明显；缺深度学习对比
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，动机阐述到位
- 价值: ⭐⭐⭐⭐ — 为 t-SVD 方法在非光滑场景的应用提供了有效解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](../../ICML2026/optimization/multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](../../ICML2026/optimization/safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[AAAI 2026\] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](../../AAAI2026/optimization/motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)
- [\[ICML 2026\] Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization](../../ICML2026/optimization/accelerated_multiple_wasserstein_gradient_flows_for_multi-objective_distribution.md)

</div>

<!-- RELATED:END -->
