---
title: >-
  [论文解读] Understanding Sharpness Dynamics in NN Training with a Minimalist Example: The Effects of Dataset Difficulty, Depth, Stochasticity, and More
description: >-
  [ICML2025][优化/理论][sharpness dynamics] 提出用"每层单神经元的深度线性网络"作为极简模型，系统性地研究 progressive sharpening 和 edge of stability 现象，引入 dataset difficulty $Q$ 概念并推导了 sharpness 在全局最优处的上下界，理论分析了数据规模、网络深度、batch size 和学习率对 sharpness 动态的影响机制。
tags:
  - "ICML2025"
  - "优化/理论"
  - "sharpness dynamics"
  - "progressive sharpening"
  - "edge of stability"
  - "deep linear networks"
  - "dataset difficulty"
---

# Understanding Sharpness Dynamics in NN Training with a Minimalist Example: The Effects of Dataset Difficulty, Depth, Stochasticity, and More

**会议**: ICML2025  
**arXiv**: [2506.06940](https://arxiv.org/abs/2506.06940)  
**代码**: 未提供  
**领域**: 优化  
**关键词**: sharpness dynamics, progressive sharpening, edge of stability, deep linear networks, dataset difficulty

## 一句话总结

提出用"每层单神经元的深度线性网络"作为极简模型，系统性地研究 progressive sharpening 和 edge of stability 现象，引入 dataset difficulty $Q$ 概念并推导了 sharpness 在全局最优处的上下界，理论分析了数据规模、网络深度、batch size 和学习率对 sharpness 动态的影响机制。

## 研究背景与动机

- **Progressive sharpening**：用 GD 训练深度网络时，loss Hessian 的最大特征值（sharpness）会逐渐增大，最终在 $2/\eta$ 附近振荡（edge of stability, EoS）。
- Cohen et al. (2021) 通过大量实验发现 sharpness 动态受数据集大小、网络深度、batch size、学习率等因素影响，但其背后机理缺乏理论解释。
- 已有理论工作（Wang et al. 2022, Agarwala et al. 2023, Marion & Chizat 2024）或只给上界、或只限于特定区间、或局限于合成数据，无法全面量化各因素的影响。
- **核心动机**：构建一个足够简单（可严格分析）又能忠实复现实际训练 sharpness 行为的最小模型。

## 方法详解

### 极简模型设计

每层只有一个神经元的深度线性网络（depth $D \geq 2$）：

$$f(x;\theta) = (x^\top u) \prod_{i=1}^{D-1} v_i$$

其中 $u \in \mathbb{R}^d$ 为第一层权重，$v_i \in \mathbb{R}$ 为后续各层标量权重，总参数量 $p = d + D - 1$。使用 MSE loss：

$$L(\theta) = \frac{1}{2N} \|Xu \prod_{i=1}^{D-1} v_i - y\|^2$$

Sharpness 定义为 $S(\theta) = \lambda_{\max}(\nabla^2 L(\theta))$。

### 核心概念：Dataset Difficulty

对数据矩阵 $X$ 的 SVD 分解 $X = \sum_{i=1}^r \sigma_i e_i w_i^\top$，标签在左奇异向量上的投影系数 $d_i = e_i^\top y$，定义：

$$Q := \sum_{i=1}^{r} \frac{d_i^2}{\sigma_i^2}$$

直觉上 $Q$ 刻画了模型完美拟合数据所需的"总距离"——每个方向上待拟合的标签分量 $d_i$ 越大、对应奇异值 $\sigma_i$ 越小，拟合越困难。$Q$ 仅依赖数据集，与架构和优化器无关。

### Sharpness 界的推导

**两层情形（$D=2$）**：定义层不平衡度 $C(\theta) = \|\Pi_W u\|^2 - v_1^2$，则全局最优点 $\theta^\star$ 处：

$$\frac{1}{N}\left[\sigma_1^2 (v_1^\star)^2 + \frac{d_1^2}{(v_1^\star)^2}\right] \leq S(\theta^\star) \leq \frac{1}{N}\left[\sigma_1^2 (v_1^\star)^2 + \frac{\sum_i d_i^2}{(v_1^\star)^2}\right]$$

其中 $(v_1^\star)^2 = \frac{\sqrt{C(\theta^\star)^2 + 4Q} - C(\theta^\star)}{2}$，sharpness 随 $Q$ 增大、随 $C(\theta^\star)$ 减小。

**一般深度（$D \geq 2$，balanced 条件下）**：

$$\frac{1}{N}\left[\sigma_1^2 Q^{\frac{D-1}{D}} + (D-1) d_1^2 Q^{-\frac{1}{D}}\right] \leq S(\theta^\star) \leq \frac{1}{N}\left[\sigma_1^2 Q^{\frac{D-1}{D}} + (D-1) \sum_i d_i^2 \cdot Q^{-\frac{1}{D}}\right]$$

主导项定义为 **predicted sharpness**：$\hat{S}_D = \frac{\sigma_1^2}{N} Q^{\frac{D-1}{D}}$。

### 优化器对 Sharpness 的影响

- **Gradient Flow**：层不平衡度 $C(\theta(t))$ 为守恒量（$D=2$）或保持 balanced（$D>2$），因此收敛 sharpness 可由初始化直接预测。
- **GD/SGD**：$C$ 逐步增大。Theorem 5.9 精确给出 GD 和 SGD 各一步后 $C$ 的增量公式，证明 SGD 比 GD 增大 $C$ 更多（$\Psi_2 \geq \Psi_1$, $\Omega_2 \geq \Omega_1$），且该额外增量与 $\frac{N-B}{B}$ 成正比——小 batch 和大学习率使 $C$ 增长更快，导致最终 sharpness 更小，即 progressive sharpening 程度降低。

## 实验关键数据

### 极简模型复现 Phenomenon 1

| 因素 | 效果 | 实验设置 |
|------|------|----------|
| 数据集大小 ↑ | sharpness ↑ | CIFAR10 2-label, $N$ 从 100 到 1000 |
| 网络深度 ↑ | sharpness ↑ | $D$ 从 2 到 5 |
| Batch size ↑ | sharpness ↑ | SGD, $B$ 从 10 到 $N$ |
| 学习率 ↑（小 $B$） | sharpness ↓ | SGD, $\eta$ 从 0.01 到 0.3 |

### Dataset Difficulty $Q$ 的数值验证

| 数据集 | $N=100$ | $N=300$ | $N=1000$ |
|--------|---------|---------|----------|
| CIFAR10 | 0.22 | 1.70 | 44.44 |
| SVHN | 1.16 | 21.13 | 859.4 |
| Google Speech | 0.26 | 1.67 | 26.34 |

$Q$ 随 $N$ 增大而急剧增长，与 progressive sharpening 趋势一致。

### Predicted Sharpness 与实际 Sharpness 的相关性

- 5 层线性网络（width 2048）在 CIFAR10 上：$\hat{S}_D$ 与 $S(\theta(\infty))$ 相关系数达 **0.99**。
- 4 层 tanh 网络（width 1024）在 SVHN 上：相关系数 **0.81**。
- 即使在非线性、宽网络、非 balanced 初始化等偏离理论假设的场景下，预测依然有效。

### Edge of Stability 的复现

极简模型成功复现了：(1) sharpness 上升至 $2/\eta$ 后振荡；(2) loss 的非单调下降（含 spike）；(3) 振荡幅度随时间衰减——已有工作的极简模型多无法同时复现这三个特征。

## 亮点与洞察

- **dataset difficulty $Q$**：一个纯数据依赖的标量，统一量化了数据规模对 sharpness 的影响，且在实际非线性网络中仍有强预测力。
- **极简模型的高保真性**：仅 $d + D - 1$ 个参数就能复现 progressive sharpening、EoS、loss spike、振荡衰减等实际训练的代表性行为。
- **层不平衡度 $C$ 的分析**：揭示了 SGD 噪声通过增大 $C$ 来抑制 sharpness 增长的定量机制，将 batch size / 学习率的效应归结为对 $C$ 增量的调控。
- **双向界**：同时给出 sharpness 的上界和下界（已有工作多只给上界），且数值上界和下界非常紧。

## 局限与展望

- 理论分析局限于**线性激活**和**每层单神经元**，虽然实验表明结论对非线性网络有一定迁移性，但缺乏严格保证。
- EoS 阶段振荡衰减的理论分析尚未完成，且作者发现该行为与**数值精度**高度相关（高精度可能导致 loss 爆炸）。
- 未讨论**交叉熵 loss** 下的行为（因 margin maximization 导致 sharpness 后期下降，对比不直接）。
- 对**网络宽度**的影响未能给出统一解释（MSE 和 CE loss 下趋势不一致）。
- SGD 的分析仅限于 $D=2$，深层 SGD 的 $C$ 演化规律有待扩展。

## 相关工作与启发

- **Cohen et al. (2021)**：formalize progressive sharpening 和 EoS 的实验基准，本文在其实验发现的基础上构建理论。
- **Damian et al. (2023)**：self-stabilization 机制解释 EoS，但假设 progressive sharpening 已经发生。
- **Marion & Chizat (2024)**：deep linear network 中 sharpness 上界分析，本文给出更紧的双向界。
- **Wang et al. (2022)**：output-layer norm 代理 sharpness，仅限特定区间。
- **Agarwala & Pennington (2024)**：二次回归模型中 SGD 的 sharpness 分析，本文在更一般设定下得到类似结论。

## 评分

- 新颖性: ⭐⭐⭐⭐ — dataset difficulty 概念新颖，极简模型的高保真度令人印象深刻
- 实验充分度: ⭐⭐⭐⭐ — 在多数据集、多架构、多优化器上系统性验证，scatter plot 相关性分析有说服力
- 写作质量: ⭐⭐⭐⭐⭐ — 组织清晰，理论与实验穿插得当，图表精简
- 价值: ⭐⭐⭐⭐ — 为理解 sharpness 动态提供了实用的理论工具和简洁的分析框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)
- [\[ICML 2025\] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias](how_transformers_learn_regular_language_recognition_a_theoretical_study_on_train.md)
- [\[ICML 2025\] Understanding Mode Connectivity via Parameter Space Symmetry](understanding_mode_connectivity_via_parameter_space_symmetry.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)

</div>

<!-- RELATED:END -->
