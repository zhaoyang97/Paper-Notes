---
title: >-
  [论文解读] SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯泼溅] 提出 SOGS，在基于锚点的 3D-GS 中引入二阶锚点（利用协方差矩阵捕获特征维度间相关性进行特征增强）和选择性梯度损失，在将锚点特征维度从 32 降至 12-16 的同时实现了超越 Scaffold-GS 的渲染质量。 基于锚点的 3D-GS（如 Scaffold-GS…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯泼溅"
  - "锚点特征增强"
  - "二阶统计"
  - "模型压缩"
  - "选择性梯度损失"
---

# SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.07476](https://arxiv.org/abs/2503.07476)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 锚点特征增强, 二阶统计, 模型压缩, 选择性梯度损失

## 一句话总结

提出 SOGS，在基于锚点的 3D-GS 中引入二阶锚点（利用协方差矩阵捕获特征维度间相关性进行特征增强）和选择性梯度损失，在将锚点特征维度从 32 降至 12-16 的同时实现了超越 Scaffold-GS 的渲染质量。

## 研究背景与动机

基于锚点的 3D-GS（如 Scaffold-GS）利用锚点特征通过 MLP 预测高斯属性，有效减少了高斯冗余。但面临一个核心困境：

1. **特征维度与模型大小的矛盾**：大锚点特征（如 32 维）提升渲染质量但显著增大模型（因锚点数量庞大）；缩小特征维度则降低高斯属性预测质量，导致纹理和几何伪影
2. **HAC 等压缩方法的局限**：它们压缩的是存储大小而非训练/渲染时的实际模型大小
3. **像素级损失对纹理不敏感**：L1 损失难以引导模型关注细节纹理和结构

本文的核心思路：特征维度间的相关性（二阶统计量）可以增强每个锚点的表征能力，用更少的特征维度实现更高质量的渲染。

## 方法详解

### 整体框架

SOGS 基于 Scaffold-GS，每个锚点存储特征 $\mathbf{f}^a \in \mathbb{R}^D$（$D$ 从 32 降至 12-16）。通过计算所有锚点特征的协方差矩阵和相关矩阵，提取 top-$M$ 个主成分（特征协变模式），与每个锚点的原始特征结合，产生增强的二阶特征用于高斯属性预测。加上选择性梯度损失聚焦难渲染区域。

### 关键设计

**1. 二阶锚点（Second-Order Anchor）**

- **功能**：通过特征维度间的协变关系增强锚点表征，补偿缩小特征维度带来的信息损失
- **核心思路**：将所有 $N$ 个锚点的 $D$ 维特征视为 $N$ 个样本、$D$ 个变量，计算协方差矩阵 $\Sigma \in \mathbb{R}^{D \times D}$，再标准化为相关矩阵 $\mathbf{R}$（消除尺度影响）。对 $\mathbf{R}$ 做特征分解，选取 top-$M$（$M=2$）个特征向量 $\mathbf{P}$ 作为全局共享的主要协变模式。对每个锚点：$\mathbf{f}^t_i = F_i([\mathbf{P}_i, \mathbf{f}^a])$，将增强特征与原始特征拼接预测高斯属性
- **设计动机**：纹理和结构不仅由单个特征定义，还由特征间的相互依赖关系决定。二阶统计量捕获这些依赖，实现特征增强而不增加锚点的存储维度

**2. 选择性梯度损失（Selective Gradient Loss）**

- **功能**：引导模型自适应地关注难渲染的纹理和结构区域
- **核心思路**：用 Sobel 算子提取渲染图和 GT 的梯度图，计算梯度差异 $l_x, l_y$ 作为损失。关键创新是引入动态区域选择——用梯度差异的绝对值 $w_x = |G'_x - G_x|$ 作为权重图，使损失 $\mathcal{L}_s = w_x \cdot l_x + w_y \cdot l_y$ 聚焦于渲染误差最大的区域
- **设计动机**：梯度图的大部分区域是平坦低梯度区，朴素梯度损失会被这些无信息区域主导。加权机制确保模型持续聚焦于关键纹理区域，并随训练动态调整关注区域

**3. 协方差计算与相关矩阵构建**

- **功能**：将原始协方差标准化，消除不同特征维度尺度差异的影响
- **核心思路**：$R_{uv} = \frac{\Sigma_{uv}}{\sigma_u \sigma_v}$，确保主成分反映的是真实的特征间相关性而非方差大小
- **设计动机**：两个方差大的变量即使关系弱也会产生大协方差，标准化后的相关矩阵能更准确地捕获特征间的协变模式

### 损失函数

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{vol} \mathcal{L}_{vol} + \lambda_s \mathcal{L}_s$$

其中 $\lambda_1 = 0.8$，$\lambda_{SSIM} = 0.2$，$\lambda_{vol} = 0.01$，$\lambda_s = 0.01$。

## 实验关键数据

### 主实验：与 Scaffold-GS 对比

| 数据集 | 方法 | SSIM↑ | PSNR↑ | LPIPS↓ | 锚点维度 |
|--------|------|-------|-------|--------|---------|
| Mip-NeRF360 | Scaffold-GS | 0.806 | 27.50 | 0.252 | 32 dim |
| Mip-NeRF360 | **SOGS** | **0.815** | **27.85** | **0.221** | **16 dim** |
| T&T | Scaffold-GS | 0.853 | 23.96 | 0.177 | 32 dim |
| T&T | **SOGS** | **0.855** | **24.14** | **0.176** | **12 dim** |
| Deep Blending | Scaffold-GS | 0.906 | 30.21 | 0.254 | 32 dim |
| Deep Blending | **SOGS** | **0.907** | **30.29** | **0.252** | **12 dim** |
| BungeeNeRF | Scaffold-GS | 0.865 | 26.62 | 0.241 | 32 dim |
| BungeeNeRF | **SOGS** | **0.880** | **27.06** | **0.171** | **16 dim** |

### 消融实验（BungeeNeRF，锚点维度 32）

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Base (Scaffold-GS) | 26.62 | 0.865 | 0.241 |
| Base + SOA | 27.25 | 0.879 | 0.208 |
| Base + SOA + SGL (SOGS) | **27.39** | **0.887** | **0.161** |

### 关键发现

- 锚点维度从 32 降至 12-16，渲染质量反而提升（PSNR +0.35, LPIPS -0.031）
- 二阶锚点贡献最大（PSNR +0.63, LPIPS -0.033），选择性梯度损失进一步提升 LPIPS
- 特征维度 >16 后收益递减，12-16 是模型大小与性能的最佳平衡点
- 在大规模场景 BungeeNeRF 上提升更显著（LPIPS 从 0.241 降至 0.171）

## 亮点与洞察

1. **统计学思路引入 3D-GS**：用 PCA/协方差的经典统计方法增强神经场特征，思路新颖且理论基础扎实
2. **模型更小质量更好**：打破了"大特征=高质量"的惯性认知，证明特征间的相关性可以补偿维度缩减
3. **选择性梯度损失实用性强**：基于 Sobel 的简单方案有效提升纹理渲染质量

## 局限与展望

- PCA 分解带来额外计算开销，虽然仍保持实时渲染但训练时间略增
- top-$M$ 的选择需要调参（论文中 $M=2$）
- 特征增强的 MLP 引入了额外参数
- 未来可探索自适应确定 $M$ 和特征维度 $D$

## 相关工作与启发

- **Scaffold-GS**：基础方法，SOGS 在其锚点特征上进行二阶增强
- **HAC/ContextGS**：压缩 Scaffold-GS 的存储大小，但不减少实际模型大小
- **3D-GS**：原始方法，高斯冗余严重

## 评分

⭐⭐⭐⭐ — 用经典统计学方法解决 3D-GS 的特征效率问题，思路清晰优雅。在多个基准上以更小模型取得更好性能，消融实验充分验证了每个组件的贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)
- [\[CVPR 2025\] ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](escape_equivariant_shape_completion_via_anchor_point_encoding.md)
- [\[CVPR 2025\] Steepest Descent Density Control for Compact 3D Gaussian Splatting](steepest_descent_density_control_for_compact_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](dronesplat_3d_gaussian_splatting_for_robust_3d_reconstruction_from_in-the-wild_d.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
