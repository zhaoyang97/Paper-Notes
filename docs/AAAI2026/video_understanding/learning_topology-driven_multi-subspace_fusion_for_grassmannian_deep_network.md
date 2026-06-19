---
title: >-
  [论文解读] Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks
description: >-
  [AAAI 2026][视频理解][Grassmannian] 提出拓扑驱动的 Grassmann 流形多子空间融合网络 GMSF-Net，通过自适应多子空间构建和基于 Fréchet 均值的子空间交互机制，将欧氏空间中多通道交互的思想成功迁移到非欧几何域，在 3D 动作识别、EEG 分类和图任务上取得 SOTA 性能。
tags:
  - "AAAI 2026"
  - "视频理解"
  - "Grassmannian"
  - "流形学习"
  - "子空间融合"
  - "3D动作识别"
  - "黎曼神经网络"
---

# Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks

**会议**: AAAI 2026  
**arXiv**: [2511.08628](https://arxiv.org/abs/2511.08628)  
**代码**: [GitHub](https://github.com/Xua-Yu/GMSF-Net)  
**领域**: 视频理解 / 流形学习  
**关键词**: Grassmannian, 流形学习, 子空间融合, 3D动作识别, 黎曼神经网络

## 一句话总结

提出拓扑驱动的 Grassmann 流形多子空间融合网络 GMSF-Net，通过自适应多子空间构建和基于 Fréchet 均值的子空间交互机制，将欧氏空间中多通道交互的思想成功迁移到非欧几何域，在 3D 动作识别、EEG 分类和图任务上取得 SOTA 性能。

## 研究背景与动机

### 核心问题

Grassmann 流形是几何表示学习的有力工具，能将高维数据建模为低维子空间。然而现有方法存在两个关键局限：

**静态单子空间表示**：GrNet、GDLNet 等方法对输入数据仅使用一个固定的正交子空间建模，难以捕捉数据中的局部几何变化和多模态分布，导致表达能力受限

**缺乏子空间交互**：欧氏空间中深度学习的成功很大程度归功于多通道交互与非线性激活（如 LeNet-5 的多通道卷积），但 Grassmann 流形上的多子空间交互一直被忽视

### 关键挑战

- 如何在 Grassmann 流形上进行有效的子空间交互？
- 如何设计可堆叠的深层架构以扩展容量？
- 如何保证黎曼流形上的优化收敛性？

## 方法详解

### 整体架构：GMSF-Net

GMSF-Net 由三个核心模块组成：自适应多子空间编码器（AMSE）、多子空间交互块和黎曼批归一化。

### 1. 自适应多子空间构建（AdaMSC）

这是 AMSE 的核心，灵感来自 Kolmogorov-Arnold 表示定理。流程如下：

1. **提取帧级特征**并计算协方差矩阵 $X \in \mathbb{R}^{n \times n}$，建模特征沿时间维度的统计依赖
2. **Schmidt 正交化**：将 $X$ 映射为一组低维正交子空间 $\mathcal{S} = \{S_1, S_2, \ldots, S_k\}$，其中 $S_i^\top S_j = 0$（$i \neq j$），每个 $S_j \in \mathcal{G}(n,1)$
3. **可学习权重选择**：为每个新子空间 $S'_{m'}$ 初始化可学习权重 $\mathcal{W}^{(m')}$，经 Softmax 归一化后选取 top-$p$ 个最重要的原子子空间
4. **加权组合**：$S'_{m'} = [\tilde{w}_{j_1}^{(m')} S_{j_1}, \tilde{w}_{j_2}^{(m')} S_{j_2}, \ldots, \tilde{w}_{j_p}^{(m')} S_{j_p}]$

这种设计使得新子空间由不同的关键原子组成，能够自适应地调整以适应不同任务。

### 2. 拓扑驱动的收敛性分析

论文从拓扑学角度严格证明了多子空间构建的收敛性：

- 使用投影度量 $d_p(X_1, X_2) = 2^{-1/2} \|X_1 X_1^T - X_2 X_2^T\|_F$ 定义 Grassmann 流形上的距离
- 通过黎曼梯度下降迭代更新子空间
- 证明在投影度量诱导的拓扑下，子空间序列收敛到稳定子空间 $S^*$：$d(S'(t), S^*) \to 0$

### 3. 多子空间交互块

包含两个子组件：

**Grassmann 多子空间表示（GMSR）**：将同一可学习映射矩阵 $W_c$（约束在 Stiefel 流形上）应用于所有输入子空间，生成不同几何框架下的表示：$X_{GMSR}^{c,m'} = W_c^T S'_{m'}$，之后通过 QR 或 SVD 分解保持正交性。

**Grassmann 子空间交互（GSI）**：使用 Fréchet 均值融合多个子空间：

- 当 $m'=2$ 时，存在闭式解（测地线插值）
- 当 $m'>2$ 时，使用 Karcher 流算法在切空间中迭代优化

### 4. 优化策略

- **黎曼批归一化**：将统计特征映射到 SPD 流形并归一化，增强判别性
- **互信息正则化**：最大化不同子空间间的信息互补性
- **总损失**：$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_C$

## 实验

### 实验一：3D 动作识别（FPHA 数据集）

| 方法 | 准确率(%) | 模型大小(MB) | FLOPs(M) |
|------|-----------|-------------|----------|
| GrNet | 78.79±1.82 | 6.73 | 38.60 |
| SPDNet | 87.65±1.02 | 13.60 | 1595.50 |
| MATT | 87.70±0.68 | 1.83 | 142.07 |
| SPDNetBN | 89.33±0.49 | 13.63 | 1902.97 |
| GDLNet | 87.60±0.69 | 1.83 | 33.69 |
| **GMSF-Net-1Block** | **90.43±0.74** | **1.20** | 48.42 |
| **GMSF-Net-3Blocks** | **91.22±0.53** | **1.30** | 81.07 |

GMSF-Net-3Blocks 比 GrNet 提升 12.43%，同时模型更小（1.30MB vs 6.73MB）。

### 实验二：EEG 信号分类（MAMEM-SSVEP-II 数据集）

| 方法 | 准确率(%) | 模型大小(MB) |
|------|-----------|-------------|
| EEGNet | 53.72±7.23 | 0.075 |
| SCCNet | 62.11±7.70 | 0.55 |
| SPDNet | 62.30±3.12 | 2.81 |
| GrNet | 61.23±3.56 | 1.95 |
| MATT | 65.19±3.14 | 1.97 |
| GDLNet | 65.52±2.86 | 1.95 |
| **GMSF-Net-3Blocks** | **66.87±1.46** | **1.94** |

GMSF-Net 比 GrNet 提升 5.64%，比 GDLNet 提升 1.35%，且标准差更小（1.46 vs 2.86），稳定性显著增强。

### 消融实验

| 配置 | HDM05 | FPHA | SSVEP |
|------|-------|------|-------|
| 自适应子空间 + 交互 | **63.64%** | **90.43%** | **66.74%** |
| 自适应子空间（无交互） | 56.49% | 80.68% | 59.83% |
| 随机子空间 + 交互 | 50.29% | 72.47% | 56.01% |
| 固定子空间 + 交互 | 53.04% | 83.06% | 66.05% |

消融实验清晰表明：(1) 自适应机制显著优于随机/固定子空间；(2) 交互机制在高质量子空间上才能发挥作用，随机子空间的交互反而引入噪声。

## 亮点与创新

1. **首次在黎曼神经网络中引入 Grassmann 子空间深层交互**，成功将欧氏空间多通道交互哲学迁移至非欧几何域
2. **拓扑驱动的理论保证**：基于投影度量拓扑严格证明了自适应子空间的收敛性，提供了可靠的理论基础
3. **显著降低模型开销**：在 FPHA 上用 1.30MB 模型超越了 13.63MB 的 SPDNetBN，效率优势明显
4. **可学习的子空间选择机制**（类比 KAN 思想）使子空间构建完全数据驱动

## 局限性

1. **任务限制**：主要在中小规模数据集（HDM05、FPHA、SSVEP）上验证，尚未在大规模视频理解任务上测试
2. **Grassmann 假设依赖**：在不符合理想子空间结构的数据集上（如 PubMed）性能提升有限，模型对数据几何结构有较强假设
3. **计算复杂度**：Karcher 流迭代求解 Fréchet 均值的计算开销随子空间数和维度增加而增大
4. **块数量饱和**：从 1Block 到 3Blocks 性能提升逐渐递减，可堆叠架构的深度扩展潜力有限

## 相关工作

- **GrNet** (Huang et al., 2018)：首个 Grassmann 流形深度网络，设计了 FRMap、OrthMap、ProjMap 等层，但仅使用单一静态子空间
- **GDLNet** (Wang et al., 2024)：在 Grassmann 流形上引入自注意力机制建立子空间间依赖，但仍受限于单子空间建模
- **SPDNet/SPDNetBN**：基于 SPD 流形的深度网络，模型较大且计算开销高
- **MATT** (Pan et al., 2022)：基于流形注意力的方法

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |

**综合评分**: ⭐⭐⭐⭐ (4/5)

> 理论扎实、创新点清晰，成功将多通道交互思想迁移至 Grassmann 流形，但应用场景偏学术化，大规模实用性有待验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)
- [\[CVPR 2026\] Gamba: Mamba-based Graph Convolutional Network with Dynamic Graph Topology Learning for Action Recognition](../../CVPR2026/video_understanding/gamba_mamba-based_graph_convolutional_network_with_dynamic_graph_topology_learni.md)
- [\[ECCV 2024\] Bayesian Evidential Deep Learning for Online Action Detection](../../ECCV2024/video_understanding/bayesian_evidential_deep_learning_for_online_action_detection.md)
- [\[CVPR 2026\] SDTrack: A Baseline for Event-based Tracking via Spiking Neural Networks](../../CVPR2026/video_understanding/sdtrack_a_baseline_for_event-based_tracking_via_spiking_neural_networks.md)
- [\[ICML 2026\] Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection](../../ICML2026/video_understanding/privacy-aware_video_anomaly_detection_through_orthogonal_subspace_projection.md)

</div>

<!-- RELATED:END -->
