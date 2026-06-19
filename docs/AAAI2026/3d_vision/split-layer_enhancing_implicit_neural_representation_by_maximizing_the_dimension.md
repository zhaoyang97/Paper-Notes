---
title: >-
  [论文解读] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space
description: >-
  [AAAI 2026][3D视觉][隐式神经表示] 提出 Split-Layer，将 MLP 全连接层拆分为多个并行分支并用 Hadamard 积整合输出，在不增加参数和计算的前提下将特征空间维度从 $C$ 指数级扩展到 $\binom{C/\sqrt{N}+N-1}{N}$，显著提升隐式神经表示（INR）的表征能力。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "隐式神经表示"
  - "特征空间"
  - "MLP重构"
  - "Hadamard积"
  - "多任务"
---

# Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space

**会议**: AAAI 2026  
**arXiv**: [2511.10142](https://arxiv.org/abs/2511.10142)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 隐式神经表示, 特征空间, MLP重构, Hadamard积, 多任务

## 一句话总结

提出 Split-Layer，将 MLP 全连接层拆分为多个并行分支并用 Hadamard 积整合输出，在不增加参数和计算的前提下将特征空间维度从 $C$ 指数级扩展到 $\binom{C/\sqrt{N}+N-1}{N}$，显著提升隐式神经表示（INR）的表征能力。

## 研究背景与动机

### 问题定义

隐式神经表示（INR）使用神经网络将信号建模为连续函数，在逆问题求解中有广泛应用。然而，INR 的表征能力受限于 MLP 架构中特征空间的维度。具体而言：

- **全连接层的局限性**：对于宽度为 $C$ 的全连接层，其输出构成的特征空间是一个 $C$ 维欧氏空间。每个输出元素是输入元素的线性组合，特征空间维度与层宽度成线性关系。
- **扩展代价高昂**：将层宽度线性增加（如增加到 $2C$），特征空间维度线性增长，但参数量却二次增长（从 $C^2$ 到 $4C^2$），计算代价不可承受。

### 已有方法的不足

现有提升 INR 表征能力的方法主要分两类：

**坐标嵌入类**：如 Fourier 位置编码（PEMLP）、哈希表（InstantNGP）等，将低维坐标映射到高维流形

**特殊激活函数类**：如 SIREN（周期正弦）、WIRE（小波）、FINER（变周期）等

然而，这些方法本质上是引入**学习偏置（learning bias）**，使某些特征更容易被学习，而非从根本上扩大可学习特征的范围。它们没有从模型结构层面扩展特征空间的维度。

### 核心动机

作者认为问题根源在于 MLP 的全连接机制——特征空间维度与神经元数量呈**线性关系**。如果能重新组织连接机制，改用"分裂"（split）风格，就能在保持相同参数量的同时，将特征空间维度**指数级扩展**。

## 方法详解

### 整体框架

Split-Layer 的核心思想非常简洁：将全连接层拆分为 $N$ 个并行分支，每个分支有独立的权重矩阵，然后通过 **Hadamard 积（逐元素乘法）**整合各分支的输出。这种操作将线性组合升级为**高阶多项式**，从而构建了远超原始 MLP 的高维特征空间。

### 关键设计

#### 1. **Split-Layer 结构**：将全连接层拆分为多分支 Hadamard 积形式

**核心思路**：原始全连接层的第 $l$ 层输出为：

$$z_i^l = \sum_{j=1}^{C} w_{ij} z_j^{l-1}$$

这是 $C$ 个线性独立元素的线性组合，特征空间维度为 $C$。

Split-Layer 将该层拆分为 $N$ 个分支，每个分支的权重矩阵 $\mathbf{W}_n^l \in \mathbb{R}^{C/\sqrt{N} \times C/\sqrt{N}}$，输出通过 Hadamard 积整合：

$$z_i^l = \prod_{n=1}^{N} \left(\sum_{j=1}^{C/\sqrt{N}} w_{ij}^n z_j^{l-1}\right)$$

展开后得到：

$$z_i^l = \sum_{(j_1,j_2,...,j_N)} \left(\prod_{n=1}^{N} w_{ij_n}^n\right) \left(z_{j_1}^{l-1} z_{j_2}^{l-1} \cdots z_{j_N}^{l-1}\right)$$

这形成了 **$N$ 阶齐次多项式**，其中不同项 $z_{j_1}^{l-1} z_{j_2}^{l-1} \cdots z_{j_N}^{l-1}$ 相互线性独立。

**设计动机**：通过多项式乘积，将特征空间从线性空间扩展为多项式空间。不同项的总数等于从 $C/\sqrt{N}$ 个元素中可重复地选取 $N$ 个的组合数：

$$\text{特征空间维度} = \binom{C/\sqrt{N}+N-1}{N}$$

当 $C=256, N=2$ 时，原始特征空间为 256 维，而 Split-Layer 将其扩展到 $\binom{181+1}{2} = 16,471$ 维，扩展了约 64 倍，但参数量保持不变（每个分支 $(C/\sqrt{N})^2$ 个参数，$N$ 个分支总共 $C^2$）。

#### 2. **最优拆分数的选择**：平衡特征空间维度与权重矩阵自由度

**核心思路**：增大拆分数 $N$ 会增大特征空间维度，但也会缩小每个分支的权重矩阵尺寸，减少其探索特征组合的自由度。经验发现最优拆分数为：

$$N^* \approx (0.17C)^{2/3}$$

**设计动机**：这是一个在特征空间维度扩展和权重矩阵表达能力之间的最佳平衡点。作者通过在不同网络宽度 $C$ 下的 2D 图像拟合实验验证了该公式的鲁棒性——最优结果总出现在理论曲面附近。

#### 3. **通用性设计**：作为即插即用模块适配不同 INR 骨干

**核心思路**：Split-Layer 替换 INR 中所有隐藏全连接层，适用于各种 INR 架构（ReLU MLP、SIREN、Gauss、PEMLP、WIRE、FINER）。

**设计动机**：Split-Layer 是一种模型结构层面的改进，与输入编码方式和激活函数正交，因此可以与现有各种增强 INR 表征能力的方法叠加使用。

### Neural Tangent Kernel 视角的验证

从 NTK 角度看，Split-MLP 的 NTK 特征值分布更均匀，从 $[10^{-3}, 10^{0}]$ 扩展到 $[10^{-2}, 10^{2}]$，意味着对高频成分的收敛性能更好。这从理论上进一步验证了 Split-Layer 的有效性。

### 损失函数 / 训练策略

- 所有任务使用标准损失函数（如 L2 距离、交叉熵损失），无需特殊设计
- 权重初始化：SIREN 和 FINER 使用各自的特定初始化方案，其余使用默认 LeCun 初始化
- 优化器：Adam，训练轮次因任务而异
- 实验中统一设置 $N=2$（即 2-split），已能取得优秀效果

## 实验关键数据

### 主实验

Split-Layer 在 6 种 INR 骨干、4 个任务上进行了全面评估。

| 任务 | 骨干 | Baseline | Split | 提升 |
|------|------|----------|-------|------|
| 2D 图像拟合 (PSNR↑) | ReLU | 21.24 | 30.89 | **+45.43%** |
| 2D 图像拟合 (PSNR↑) | SIREN | 38.52 | 39.25 | +1.90% |
| 2D 图像拟合 (PSNR↑) | PEMLP | 29.60 | 40.78 | **+37.77%** |
| 2D 图像拟合 (PSNR↑) | Gauss | 31.74 | 40.84 | **+28.67%** |
| CT 重建 (PSNR↑) | SIREN | 18.32 | 29.11 | **+58.90%** |
| CT 重建 (PSNR↑) | PEMLP | 28.11 | 32.29 | +14.87% |
| 3D 形状表示 (CD↓) | ReLU | 1.00e-4 | 2.01e-5 | **+79.90%** |
| 3D 形状表示 (CD↓) | Gauss | 2.19e-5 | 5.33e-6 | **+75.66%** |

**5D 新视角合成（NeRF 场景，PSNR↑）**：

| 方法 | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship | 平均 |
|------|-------|-------|-------|--------|------|-----------|-----|------|------|
| NeRF | 31.37 | 24.50 | 28.90 | 34.94 | 30.71 | 28.60 | 28.99 | 27.27 | 29.41 |
| Split-NeRF | 31.78 | 24.81 | 29.34 | 35.33 | 31.76 | 28.87 | 31.85 | 27.83 | **30.20** |
| DINER | 34.49 | 25.43 | 33.28 | 36.45 | 34.82 | 29.58 | 33.43 | 29.25 | 32.09 |
| Split-DINER | **34.85** | 25.47 | **33.39** | **36.92** | **35.14** | 29.59 | **34.01** | **29.49** | **32.36** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| $N=2$ (默认) | 最佳或接近最佳 | 实用性与性能的最佳平衡 |
| 不同 $C$ 下的最优 $N$ | 与 $(0.17C)^{2/3}$ 曲面吻合 | 验证最优拆分公式的鲁棒性 |
| 特征可视化 | Split 后特征更多样化 | 9 个特征 → 45/84 个不同特征基 |
| NTK 特征值分布 | Split-MLP 更均匀 | 高频收敛性能更好 |

### 关键发现

1. **ReLU 和 SIREN 获益最大**：ReLU 在图像拟合任务上提升 45%，SIREN 在 CT 重建上提升 59%，说明原始骨干表征能力越弱，Split-Layer 带来的提升越显著
2. **Split-PEMLP 表现突出**：在图像拟合和形状表示任务中均达到最佳性能
3. **零额外成本**：Split-Layer 在不增加参数和计算量的前提下实现了显著提升
4. **通用性强**：在 6 种骨干 × 4 个任务的所有组合中均有提升

## 亮点与洞察

1. **理论优雅**：从特征空间维度的角度分析 INR 表征能力，将问题归结为组合数学中的可重复组合问题，理论推导简洁清晰
2. **实现简单**：仅需将全连接层替换为多分支 Hadamard 积结构，无需修改训练流程或损失函数
3. **正交于现有方法**：Split-Layer 与位置编码、激活函数等方法互不冲突，可叠加使用
4. **NTK 视角验证**：从神经切线核的角度进一步解释了性能提升的理论基础

## 局限与展望

1. **最优拆分公式为经验公式**：$N^* \approx (0.17C)^{2/3}$ 缺乏严格理论推导，仅通过实验拟合得到
2. **Hadamard 积可能带来训练不稳定性**：高阶多项式可能导致梯度爆炸/消失问题，论文未深入分析
3. **仅验证了 INR 场景**：未在更广泛的深度学习任务（如分类、检测）中验证 Split-Layer 的通用性
4. **未探索与 Hash Grid 等方法的结合**：如 InstantNGP 等方法叠加 Split-Layer 的效果值得探索

## 相关工作与启发

- 与 MFN（Multiplicative Filter Networks）的联系：MFN 也使用了乘法操作来组合多分支输出，但 Split-Layer 给出了更清晰的理论分析（特征空间维度计算）
- Hilbert 核或高斯核可能提供进一步扩展特征空间的途径（作者在结论中提及）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 从全连接层结构角度重新思考 INR 表征能力，思路新颖
- **实验充分度**: ⭐⭐⭐⭐ — 4 个任务 × 6 个骨干的全面评估，但缺乏计算效率的定量对比
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导清晰，图示直观
- **价值**: ⭐⭐⭐⭐ — 即插即用的通用 INR 增强模块，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](../../ICCV2025/3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)
- [\[CVPR 2026\] Content-Aware Frequency Encoding for Implicit Neural Representations with Fourier-Chebyshev Features](../../CVPR2026/3d_vision/content-aware_frequency_encoding_for_implicit_neural_representations_with_fourie.md)
- [\[CVPR 2026\] NTK-Guided Implicit Neural Teaching](../../CVPR2026/3d_vision/ntk-guided_implicit_neural_teaching.md)

</div>

<!-- RELATED:END -->
