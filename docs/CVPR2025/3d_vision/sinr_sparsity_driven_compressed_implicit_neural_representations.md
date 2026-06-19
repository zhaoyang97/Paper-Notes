---
title: >-
  [论文解读] SiNR: Sparsity Driven Compressed Implicit Neural Representations
description: >-
  [CVPR 2025][3D视觉][隐式神经表示] 发现INR权重空间呈高斯分布这一关键性质，基于压缩感知理论用随机感知矩阵将权重向量转换为高维稀疏编码，实现了不依赖量化方案的基础性INR压缩，可与任何现有INR压缩方法叠加使用。 领域现状：隐式神经表示（INR）用MLP将坐标映射到信号值（如像素颜色、占用值）…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "隐式神经表示"
  - "信号压缩"
  - "稀疏编码"
  - "压缩感知"
  - "INR"
---

# SiNR: Sparsity Driven Compressed Implicit Neural Representations

**会议**: CVPR 2025  
**arXiv**: [2503.19576](https://arxiv.org/abs/2503.19576)  
**代码**: [https://dsgrad.github.io/SINR](https://dsgrad.github.io/SINR)  
**领域**: 3D视觉  
**关键词**: 隐式神经表示, 信号压缩, 稀疏编码, 压缩感知, INR

## 一句话总结

发现INR权重空间呈高斯分布这一关键性质，基于压缩感知理论用随机感知矩阵将权重向量转换为高维稀疏编码，实现了不依赖量化方案的基础性INR压缩，可与任何现有INR压缩方法叠加使用。

## 研究背景与动机

**领域现状**：隐式神经表示（INR）用MLP将坐标映射到信号值（如像素颜色、占用值），是一种统一的跨模态数据表示方式。现有INR压缩方法如COIN、COIN++、INRIC主要采用两种策略：直接对训练好的INR做量化+熵编码；或通过可学习变换在INR之上导出潜在编码再压缩。

**现有痛点**：现有方法的性能严重依赖量化方案和熵编码的选择，且没有方法试图在量化和熵编码之前从根本上压缩INR本身——即发现并利用INR权重空间中内在的可压缩性。

**核心矛盾**：自然信号在变换域中天然稀疏（如DCT域），INR本质上是将信号编码到MLP权重中的一种域变换。既然信号可压缩，那对应的权重空间也应该存在可利用的压缩性，但如何发现这种隐藏的稀疏性？

**本文目标**：在量化和熵编码之前，找到一种基础性的INR压缩方法，可独立于具体的压缩pipeline使用。

**切入角度**：作者观察到一个关键现象——训练好的INR的权重向量近似服从高斯分布，且这一规律在图像、占用场、NeRF等不同模态中都成立。根据中心极限定理（CLT），高斯分布的随机变量可由任意随机变量的有限线性组合产生，这意味着可以用随机矩阵作为感知矩阵来发现稀疏表示。

**核心 idea**：将每个权重向量$\mathbf{w}$分解为$\mathbf{w} = \mathbf{A}\mathbf{x}$，其中$\mathbf{A}$是随机高斯感知矩阵（由种子控制），$\mathbf{x}$是高维稀疏向量。由于$\mathbf{A}$可由种子重建，只需传输$\mathbf{x}$的非零值和索引即可恢复原始权重。

## 方法详解

### 整体框架

SINR作为一个前处理模块插入到INR压缩流程中：训练好INR后，先用SINR将权重向量转换为稀疏编码（只存非零值和索引），然后再进行量化和熵编码。解码端通过种子重建感知矩阵，从稀疏编码恢复权重，再正常推理。

### 关键设计

1. **权重空间的高斯分布发现**:

    - 功能：为使用压缩感知提供理论基础
    - 核心思路：通过实验验证，INR隐藏层的权重分布在图像、占用场、NeRF等不同数据模态下都趋近高斯分布。这不依赖于具体的激活函数（Sinusoidal、Gaussian、WIRE等），是INR的一个内在特性
    - 设计动机：高斯分布特性使得CLT可以直接应用——既然$w_i = \sum_j A_{ij} x_j$可以产生高斯分布的$w_i$，那么存在一个稀疏的$\mathbf{x}$使得$\mathbf{w} = \mathbf{A}\mathbf{x}$成立

2. **基于随机感知矩阵的稀疏编码**:

    - 功能：将$k_1$维权重向量压缩为$2s$个元素（$s$个非零值 + $s$个索引）
    - 核心思路：给定权重向量$\mathbf{w} \in \mathbb{R}^{k_1}$，构造随机感知矩阵$\mathbf{A} \in \mathbb{R}^{k_1 \times k_2}$（$k_2 > k_1$），通过$L_1$最小化（OMP算法）求解$\min \|\mathbf{x}\|_1$ s.t. $\mathbf{w} = \mathbf{A}\mathbf{x}$，约束$2s < k_1$保证压缩效果。感知矩阵$\mathbf{A}$由一个随机种子控制，收发双方共享种子即可，无需传输矩阵本身
    - 设计动机：传统字典学习需要学习/传输字典$\mathbf{A}$，而CLT告诉我们随机矩阵就够用了。这完全消除了对字典的依赖，是本文最核心的理论贡献

3. **对Tiny INR的特殊处理**:

    - 功能：处理隐藏层神经元数$k < 50$的小型INR
    - 核心思路：当$k$很小时，满足$2s < k$的稀疏表示难以有效压缩。解决方案是将$k \times k$的权重矩阵展平为$k^2 \times 1$的向量，由于$k^2 \gg k$，在这个更高维度上做稀疏编码效果更好
    - 设计动机：保证SINR对各种规模的INR都适用，包括参数量很少的轻量级模型

### 损失函数 / 训练策略

SINR本身不涉及训练，是一个后处理算法。INR的训练使用标准的信号重建损失（如MSE）。SINR在训练完成后应用，使用OMP（正交匹配追踪）算法求解L1最小化问题。量化使用16位均匀量化器（65536级），熵编码使用Brotli编码。稀疏度$s$的选择不依赖于具体信号，只依赖于隐藏层神经元数量。

## 实验关键数据

### 主实验

图像编码（KODAK数据集）：

| 配置(h,m) | 方法 | ~30dB PSNR所需bpp |
|-----------|------|-------------------|
| (3,128) | COIN (baseline) | ~3.7 bpp |
| (3,128) | INRIC | ~2.0 bpp |
| (3,128) | **SINR** | **~1.7 bpp** |

COIN++集成：达到~24.2dB PSNR，COIN++需>1.5 bpp，加SINR后<1.0 bpp。

### 消融实验

| 实验 | 关键发现 |
|------|---------|
| C1: 变化神经元数(32→128) | 神经元越多，SINR压缩收益越大 |
| C2: 变化层数(3→7)固定64 | 层数增加时压缩收益相对有限 |
| C3/C4: Meta-learning | SINR在元学习框架下同样有效 |
| C5: COIN++ | 可压缩modulation参数 |
| 占用场 | SINR取得最小文件和最高IoU |

### 关键发现

- 隐藏层神经元越多，INR学到的信号表示越鲁棒，权重空间中隐藏的可压缩性越强——更大的模型反而SINR压缩比更高
- SINR的最优稀疏度$s$不依赖于具体信号内容，只取决于隐藏层维度，参数选择非常简单
- 占用场数据模态比图像更可压缩，说明不同模态信号在INR权重空间中的冗余程度不同
- SINR与COIN++兼容，能同时压缩基网络参数和调制参数

## 亮点与洞察

- **权重高斯分布的发现**堪称"第一性原理级"的洞察——INR将信号编码到权重后，权重呈高斯分布是CLT的必然结果，这为整个压缩方案提供了坚实的理论基础
- **感知矩阵无需学习/传输**是极其优雅的设计——利用CLT证明随机矩阵即可，只需共享种子，完全消除了字典传输的开销
- **模态无关的通用压缩**：同一套方法对图像、3D占用场、NeRF都有效，真正体现了INR作为统一数据表示的优势

## 局限与展望

- OMP算法在高维度时计算开销较大，限制了对大规模INR的实时压缩
- 未讨论在训练中引入稀疏约束的可能——联合训练+压缩可能获得更好的率失真性能
- 量化和熵编码仍使用简单方案，与更先进的学习型量化结合可能进一步提升
- 偏置向量未做压缩（因为太小），在极端压缩场景下偏置也值得考虑

## 相关工作与启发

- **vs COIN/COIN++**: 这些方法直接量化INR参数或其调制，未探索权重空间的内在可压缩性。SINR在它们之前插入一步基础压缩，两者互补
- **vs INRIC**: INRIC用元学习提高INR泛化能力进而间接改善压缩，SINR则直接发现权重冗余。SINR可以直接叠加在INRIC之上获得双重收益
- **vs SHACIRA**: SHACIRA用特征网格重参数化做压缩，是架构层面的设计；SINR是权重层面的通用后处理，不依赖架构

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 权重高斯分布→CLT→随机感知矩阵的推理链非常优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖图像/占用场/NeRF多模态，多种配置组合
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，动机铺垫到位
- 价值: ⭐⭐⭐⭐ 作为通用的INR前处理压缩模块具有很强的实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)
- [\[ECCV 2024\] Lagrangian Hashing for Compressed Neural Field Representations](../../ECCV2024/3d_vision/lagrangian_hashing_for_compressed_neural_field_representations.md)
- [\[CVPR 2026\] Task-Driven Implicit Representations for Automated Design of LiDAR Systems](../../CVPR2026/3d_vision/task-driven_implicit_representations_for_automated_design_of_lidar_systems.md)
- [\[ICCV 2025\] LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression](../../ICCV2025/3d_vision/linr-pcgc_lossless_implicit_neural_representations_for_point_cloud_geometry_comp.md)
- [\[CVPR 2026\] Content-Aware Frequency Encoding for Implicit Neural Representations with Fourier-Chebyshev Features](../../CVPR2026/3d_vision/content-aware_frequency_encoding_for_implicit_neural_representations_with_fourie.md)

</div>

<!-- RELATED:END -->
