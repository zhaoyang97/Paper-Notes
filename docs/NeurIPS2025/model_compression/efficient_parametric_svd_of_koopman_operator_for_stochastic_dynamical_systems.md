---
title: >-
  [论文解读] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems
description: >-
  [NEURIPS2025][模型压缩][Koopman operator] 提出基于 low-rank approximation (LoRA) 的目标函数来学习随机动力系统 Koopman 算子的 top-k 奇异函数，完全避免了 VAMPnet/DPNet 中数值不稳定的矩阵分解操作，且梯度天然无偏。
tags:
  - "NEURIPS2025"
  - "模型压缩"
  - "Koopman operator"
  - "singular value decomposition"
  - "dynamical systems"
  - "low-rank approximation"
  - "deep learning"
---

# Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems

**会议**: NEURIPS2025  
**arXiv**: [2507.07222](https://arxiv.org/abs/2507.07222)  
**代码**: [MinchanJeong/NeuralKoopmanSVD](https://github.com/MinchanJeong/NeuralKoopmanSVD)  
**领域**: 其他  
**关键词**: Koopman operator, singular value decomposition, dynamical systems, low-rank approximation, deep learning

## 一句话总结

提出基于 low-rank approximation (LoRA) 的目标函数来学习随机动力系统 Koopman 算子的 top-k 奇异函数，完全避免了 VAMPnet/DPNet 中数值不稳定的矩阵分解操作，且梯度天然无偏。

## 背景与动机

Koopman 算子理论通过将非线性动力系统提升到无穷维函数空间，使得线性算子的谱分析技术可以直接应用。近年来，Dynamic Mode Decomposition (DMD) 展示了从轨迹数据中以数据驱动方式识别系统主导模式的能力。在此基础上，VAMPnet 和 DPNet 等深度学习方法被提出来学习 Koopman 算子的主导奇异子空间。

然而，这些方法在目标函数计算过程中需要对经验二阶矩矩阵执行 SVD 或矩阵求逆等数值不稳定操作。这带来两个核心问题：

1. **梯度偏差**：通过 SVD/矩阵求逆进行反向传播会引入有偏梯度估计，特别是当矩阵病态（特征值间距小）时，梯度会爆炸
2. **可扩展性差**：mini-batch 估计的二阶矩矩阵可能是秩亏的，导致求逆不稳定；VAMPnet 需要额外引入正则化参数 $\lambda$ 来缓解

## 核心问题

如何设计一个概念简单、数值稳定、且易于与现代深度学习流水线集成的方法，来学习随机动力系统 Koopman 算子的 top-k 奇异函数？

## 方法详解

### 问题设定

考虑随机离散时间动力系统 $\mathbf{x}_{t+1} = \xi(F(\mathbf{x}_t), \epsilon_t)$，其中 Koopman 算子为条件期望算子：

$$(\mathcal{K}g)(\mathbf{x}) = \mathbb{E}_{p(\mathbf{x}'|\mathbf{x})}[g(\mathbf{x}')]$$

目标是从 $N$ 条独立轨迹中收集的转移对 $\{(\mathbf{x}_t, \mathbf{x}_{t+1})\}$ 学习 $\mathcal{K}$ 的 top-k 奇异子空间。

### LoRA 目标函数

核心思想是直接最小化 low-rank 近似误差 $\|\mathcal{K} - \sum_{i=1}^k f_i \otimes g_i\|_{\text{HS}}^2$，化简后得到极其简洁的目标：

$$\mathcal{L}_{\text{lora}}(\mathbf{f}, \mathbf{g}) = -2 \operatorname{tr}(\mathsf{T}[\mathbf{f}, \mathbf{g}]) + \operatorname{tr}(\mathsf{M}_{\rho_0}[\mathbf{f}] \cdot \mathsf{M}_{\rho_1}[\mathbf{g}])$$

其中 $\mathsf{T}[\mathbf{f}, \mathbf{g}]$ 是联合二阶矩矩阵，$\mathsf{M}_{\rho_0}[\mathbf{f}]$ 和 $\mathsf{M}_{\rho_1}[\mathbf{g}]$ 分别是当前和未来状态的二阶矩矩阵。

**关键优势**：目标函数完全是二阶矩矩阵的多项式，不涉及任何矩阵求逆或 SVD 操作，因此：
- 梯度可以自然地通过 mini-batch 进行无偏估计
- 无需正则化参数调优
- 易于集成到标准深度学习训练框架

由 Eckart-Young-Mirsky 定理保证，该目标在全局最优处精确刻画 $\mathcal{K}$ 的奇异子空间。

### Nesting 技术：学习有序奇异函数

为了学习按奇异值大小排序的奇异函数，引入 nesting 技术：

- **Joint nesting**：同时优化所有维度 $\mathcal{L}_{\text{lora}}^{\text{joint}} = \sum_{i=1}^k \alpha_i \mathcal{L}_{\text{lora}}(\mathbf{f}_{1:i}, \mathbf{g}_{1:i})$，均匀权重最优
- **Sequential nesting**：逐步更新第 $i$ 对函数 $(f_i, g_i)$，利用归纳论证保证收敛

两种 nesting 几乎不增加额外计算开销，且实验中观察到 nesting 能持续提升下游任务表现。

### 推断方法

学习完奇异子空间后，提供两种推断路径：

**方法 1 (CCA + LoRA)**：对学到的基函数先白化，再做 SVD 对齐，得到有序奇异函数。利用有限秩近似进行多步前向/后向预测。

**方法 2 (EDMD)**：将 Koopman 算子投影到学到的基函数张成的子空间上，通过最小二乘回归得到近似 Koopman 矩阵，再进行多步预测。

### 连续时间推广

对于可逆连续时间动力系统（如 overdamped Langevin dynamics），目标函数简化为 $\mathcal{L}_{\text{lora}}^{\text{sa}}(\mathbf{f}) = -2\operatorname{tr}(\mathsf{M}_{\rho_0}[\mathbf{f}, \mathcal{L}\mathbf{f}]) + \|\mathsf{M}_{\rho_0}[\mathbf{f}]\|_F^2$，无需额外的 lagged encoder。

## 实验关键数据

### Ordered MNIST

合成实验，数字按 0→1→2→3→4→0 循环转移。LoRA 及其变体在多步预测 RMSE 上持续优于 VAMPnet-1、DPNet 和 DPNet-relaxed，尤其 joint nesting 效果最佳。预测步数范围 $t \in \{-15, \ldots, 15\}$。

### 1D Langevin Dynamics

连续时间可逆过程。LoRA$_{\text{seq}}$ 能可靠地恢复前 8 个特征函数及特征值，与真值高度吻合。相比之下，DPNet 目标在相同配置下未能收敛。

### Chignolin 分子动力学

高维真实数据（人工微型蛋白质折叠动力学）。关键结果（VAMP-E 分数，越高越好）：

| 方法 | Low-rank (k=16) | High-rank (k=64) |
|------|-----------------|-------------------|
| DPNet-relaxed | 7.36±0.40 | 6.97±0.31 |
| VAMPnet-1 | 9.54±0.31 | 19.71±0.59 |
| LoRA | 10.27±0.31 | 37.74±0.95 |
| LoRA$_{\text{jnt}}$ | 10.74±0.35 | 38.50±0.83 |
| LoRA$_{\text{seq}}$ | **12.29±0.07** | **37.33±1.66** |

注意：VAMPnet-2 和 DPNet 在此实验中发散；VAMPnet-1 对 PyTorch 版本和数据划分敏感，某些设置下也会发散。LoRA 变体是唯一在所有设置下一致收敛的方法。

## 亮点

1. **极简目标函数**：LoRA 目标仅是二阶矩矩阵的多项式，完全规避了矩阵求逆和 SVD，概念清晰且实现简单
2. **无偏梯度**：与 VAMPnet/DPNet 不同，mini-batch 梯度天然无偏，对大规模系统友好
3. **无需正则化**：不需要 VAMPnet 的 $\lambda$ 调优或 DPNet 的 metric distortion loss
4. **收敛一致性**：在所有实验中均可靠收敛，是唯一在 chignolin 高维实验中不发散的方法类
5. **完整理论保证**：有 Eckart-Young-Mirsky 定理的全局最优性保证，以及 $O(N^{-1/2})$ 的样本复杂度

## 局限与展望

1. 学到的动力学质量受限于数据的时间分辨率，可能无法恢复最慢的物理过程
2. 对高度非正规（non-normal）或混沌系统中鲁棒识别 coherent structures 仍是开放问题
3. 仅适用于随机动力系统（确定性系统的 Koopman 算子不紧致）
4. LoRA 目标函数的优化景观（landscape）理论分析尚不完善
5. 未探索针对 Koopman 分析的专用神经网络架构

## 与相关工作的对比

| 方面 | VAMPnet | DPNet | 本文 LoRA |
|------|---------|-------|-----------|
| 目标函数 | Schatten 范数 + 矩阵求逆 | VAMP-2 + metric distortion | 二阶矩多项式 |
| 数值稳定性 | 需要正则化 $\lambda$ | 需要 relaxation | 天然稳定 |
| 梯度无偏性 | 有偏 | 有偏 | 无偏 |
| 正则化参数 | $\lambda$ (交叉验证) | $\gamma$ | 无需 |
| 高维可扩展性 | 差（k=64 退化） | 发散 | 保持稳定 |

## 启发与关联

- LoRA 思想从矩阵低秩近似推广到算子低秩近似，这一思路可能对其他无穷维算子的谱学习问题有借鉴意义
- 目标函数设计的核心启示：避免在训练目标中引入需要反向传播的不可微/不稳定操作，用等价的多项式形式替代
- Nesting 技术提供了一种不增加计算开销的方式来引导模型关注最重要的信号，可能对其他多目标学习场景有参考价值
- 与分子动力学的结合表明，该方法在科学计算领域有实际应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ (LoRA 目标函数虽简单但巧妙，Koopman 领域首次系统性应用)
- 实验充分度: ⭐⭐⭐⭐ (合成 + Langevin + 分子动力学覆盖面广，消融充分)
- 写作质量: ⭐⭐⭐⭐⭐ (理论推导严谨，记号清晰，动机阐述流畅)
- 价值: ⭐⭐⭐⭐ (为 Koopman 算子学习提供了实用且可靠的新基线)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization](tighter_cmi-based_generalization_bounds_via_stochastic_projection_and_quantizati.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](../../CVPR2025/model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[ICLR 2026\] Textual Equilibrium Propagation for Deep Compound AI Systems](../../ICLR2026/model_compression/textual_equilibrium_propagation_for_deep_compound_ai_systems.md)
- [\[AAAI 2026\] Parametric Pareto Set Learning for Expensive Multi-Objective Optimization](../../AAAI2026/model_compression/parametric_pareto_set_learning_for_expensive_multi-objective_optimization.md)
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](../../ICLR2026/model_compression/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)

</div>

<!-- RELATED:END -->
