---
title: >-
  [论文解读] Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data
description: >-
  [ECCV 2024][隐式神经表示] 提出超像素引导的隐式神经表示（S-INR），用广义超像素替代像素作为INR的基本单元，通过专属注意力MLP和共享字典矩阵两个模块，充分挖掘广义超像素内部和之间的语义信息，在图像重建/补全/去噪以及点数据恢复等任务上超越现有INR方法。 隐式神经表示（INR）使用基于坐标的MLP将空间坐…
tags:
  - "ECCV 2024"
  - "隐式神经表示"
  - "超像素"
  - "多维数据恢复"
  - "注意力机制"
  - "字典学习"
---

# Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data

**会议**: ECCV 2024  
**arXiv**: [2411.11356](https://arxiv.org/abs/2411.11356)  
**代码**: 无  
**领域**: 其他  
**关键词**: 隐式神经表示, 超像素, 多维数据恢复, 注意力机制, 字典学习

## 一句话总结

提出超像素引导的隐式神经表示（S-INR），用广义超像素替代像素作为INR的基本单元，通过专属注意力MLP和共享字典矩阵两个模块，充分挖掘广义超像素内部和之间的语义信息，在图像重建/补全/去噪以及点数据恢复等任务上超越现有INR方法。

## 研究背景与动机

隐式神经表示（INR）使用基于坐标的MLP将空间坐标映射到对应值（如像素强度、占据值），已在图像、视频、3D形状等多维数据表示中取得成功。然而，现有INR方法存在一个根本性问题：

**忽略了数据的固有语义信息**。标准INR以单个像素为基本单元，逐像素独立进行坐标→值映射，既没有利用像素间的局部语义关联，也没有捕获跨区域的结构共性。

为解决这一问题，作者提出一个自然的问题：**能否在INR框架下开发一种高效利用数据固有语义信息的新表示方法？**

核心思路：用包含丰富语义信息的**广义超像素**替代像素作为INR的基本单元。

## 方法详解

### 整体框架

S-INR由三个关键组件构成：
1. **广义超像素分割算法（GSSA）**：将数据分割为语义一致的区域
2. **专属注意力MLP（$\Psi_{\theta_k}$）**：为每个超像素使用独立的MLP捕获内部语义
3. **共享字典矩阵（$\mathbf{D}$）**：跨所有超像素共享，捕获超像素间的共性

数学表达：
$$\hat{\mathbf{o}}^k = \mathbf{D}(\Psi_{\theta_k}(\mathbf{x}^k)), \quad k = 1, \ldots, K$$

### 关键设计

**1. 广义超像素定义**

不同于传统超像素仅适用于图像，广义超像素扩展到任意点数据（如3D表面、气象数据）。需满足两个条件：
- **不相交性**：各超像素之间无重叠
- **空间连通性**：每个超像素内的数据点在空间上连续

**2. 广义超像素分割算法（GSSA）**

基于k-means++变体，在聚类时同时考虑特征相似性和空间坐标距离：
$$m_{ik} = \begin{cases} 1 & \text{if } k = \arg\min_k \|\mathbf{o}_i - \boldsymbol{\mu}_k\|^2 + \alpha \|\mathbf{x}_i - \mathbf{x}_{\boldsymbol{\mu}_k}\|^2 \\ 0 & \text{otherwise} \end{cases}$$

权重 $\alpha$ 控制空间连通性的强度，确保满足广义超像素的两个条件。

**3. 专属注意力MLP**

在每个超像素的MLP中插入自注意力模块，增强特征维度间的表达能力：
$$\psi_l^k(\mathbf{z}_{l+1}^k) = \eta(\mathbf{U}_l^k(\delta(\mathbf{V}_l^k(\tau(\mathbf{z}_{l+1}^k))))) \otimes \mathbf{z}_{l+1}^k$$

其中 $\tau$ 为通道平均池化，$\delta$ 为ReLU，$\eta$ 为sigmoid，$\otimes$ 为逐通道乘积。这种通道注意力机制让模型能自适应地强调对当前超像素最重要的特征维度。

**4. 共享字典矩阵**

$\mathbf{D} \in \mathbb{R}^{s \times r}$ 在所有超像素间共享，类似字典学习中的编码矩阵但设为可学习参数。它将每个超像素MLP输出的 $r$ 维系数映射到 $s$ 维输出，通过共享实现跨超像素的信息传递和共性捕获。

### 损失函数 / 训练策略

针对三种任务设计不同损失：

1. **数据重建**：$\sum_k \sum_i \|\mathbf{o}_i^k - \mathbf{D}(\Psi_{\theta_k}(\mathbf{x}_i^k))\|^2$
2. **数据补全**：仅在观测位置上计算损失 $\|\cdot\|_\Omega^2$
3. **数据去噪**：与重建相同，利用INR的隐式正则化过滤噪声

使用Adam优化器，模型为无监督（仅需观测数据），无需额外训练集。

超参数：隐层大小35，5层，$K \in \{15, 25, 50\}$，$\alpha \in \{1, 5, 20\}$，$\omega_0 \in \{30, 150, 300\}$。

## 实验关键数据

### 主实验（表格）

**图像重建**：

| 方法 | Kodim PSNR↑ | Kodim SSIM↑ | Pavia PSNR↑ | Pavia SSIM↑ |
|------|-------------|-------------|-------------|-------------|
| **S-INR** | **36.077** | **0.965** | **39.102** | **0.949** |
| WIRE | 33.199 | 0.918 | 38.455 | 0.941 |
| SIREN | 33.052 | 0.932 | 37.727 | 0.937 |
| Fourier | 32.101 | 0.899 | 37.982 | 0.935 |
| Gauss | 30.188 | 0.862 | 36.413 | 0.923 |
| DIP | 30.154 | 0.882 | 36.283 | 0.919 |

### 点数据恢复（表格）

**3D表面补全和气象数据补全（NRMSE↓, R-Square↑）**：

| 方法 | 气象(63°N) NRMSE↓ | 气象(63°N) R²↑ | 3D Scene1 NRMSE↓ | 3D Scene1 R²↑ |
|------|-------------------|----------------|------------------|---------------|
| **S-INR** | **0.058** | **0.900** | **0.074** | **0.944** |
| SIREN | 0.078 | 0.818 | 0.109 | 0.878 |
| KNR | 0.072 | 0.849 | 0.112 | 0.868 |
| RF | 0.076 | 0.829 | 0.107 | 0.880 |
| DT | 0.101 | 0.698 | 0.171 | 0.703 |

### 关键发现

1. S-INR在图像重建上比次优方法WIRE高出约3 dB PSNR
2. 在图像补全任务中（2.5%采样率），S-INR达到29.068 dB，比WIRE高约1 dB
3. 在图像去噪中，S-INR无需任何显式正则化即可有效去噪，归功于超像素的隐式结构约束
4. 在3D表面补全任务中，S-INR的R²达到0.944，显著优于SIREN的0.878
5. 气象数据补全验证了广义超像素在非图像数据上的有效性
6. 传统回归方法（KNR、RF）在结构复杂的数据上表现不如S-INR

## 亮点与洞察

- **从像素到超像素的范式转移**：这一简单而强大的思路为INR引入了结构先验
- **广义超像素的定义**超越了传统图像超像素的边界，使其适用于气象、3D点云等多种数据类型
- **个性+共性**的架构设计本质上是局部-全局信息的平衡——专属MLP捕获局部特征，共享字典捕获全局结构
- 注意力机制的引入增强了特征维度间的交互，特别是对多光谱图像等高维数据有效

## 局限与展望

- 每个超像素需要一个独立MLP，当 $K$ 较大时参数量显著增加
- GSSA是预处理步骤，对含噪/不完整的观测数据的超像素分割质量存疑
- 超像素数量 $K$ 作为关键超参数需要手动调节
- 仅在中等分辨率数据上验证（256×256/512×768），高分辨率数据的可扩展性未知
- 未与最近的基于hash编码的INR方法（如InstantNGP）对比
- 共享字典矩阵的大小 $r$ 需要针对不同数据类型分别调参

## 相关工作与启发

- SIREN的正弦激活函数被S-INR沿用作为基础组件
- 与Neural Dictionary等工作的思路类似，但S-INR在字典前加入了超像素感知的注意力MLP
- 超像素的思想在传统图像处理中已有广泛应用（如SLIC），本文将其引入神经表示领域
- 共享字典矩阵的设计借鉴了字典学习和稀疏编码的思想

## 评分

- **创新性**: ★★★★☆ — 超像素+INR的结合新颖，广义超像素定义有意义
- **实用性**: ★★★☆☆ — 参数效率待改进，超参数较多
- **实验完整性**: ★★★★★ — 图像+点数据+多任务的全面验证
- **写作质量**: ★★★★☆ — 数学定义严谨，结构清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Functional Transform-Based Low-Rank Tensor Factorization for Multi-Dimensional Data Recovery](functional_transform-based_low-rank_tensor_factorization_for_multi-dimensional_d.md)
- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](../../CVPR2025/others/evos_efficient_implicit_neural_training_via_evolutionary_selector.md)
- [\[ICML 2025\] DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](../../ICML2025/others/dsp_dynamic_sequence_parallelism_for_multi-dimensional_transformers.md)
- [\[CVPR 2026\] Adaptive Data Augmentation with Multi-armed Bandit: Sample-Efficient Embedding Calibration for Implicit Pattern Recognition](../../CVPR2026/others/adaptive_data_augmentation_with_multi-armed_bandit_sample-efficient_embedding_ca.md)
- [\[ECCV 2024\] CLR-GAN: Improving GANs Stability and Quality via Consistent Latent Representation and Reconstruction](clr-gan_improving_gans_stability_and_quality_via_consistent_latent_representatio.md)

</div>

<!-- RELATED:END -->
