---
title: >-
  [论文解读] Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting
description: >-
  [NeurIPS 2025][3D视觉][平面3D重建] 提出Plana3R，一个无需位姿和平面标注的前馈框架，从未配对的双视角图像中预测稀疏3D平面基元和度量尺度相对位姿，实现室内场景的零样本度量平面3D重建。 室内环境是人类生活的主要场所，创建其数字孪生对很多应用至关重要。室内场景天然富含平面结构（地板、墙壁、桌面等）…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "平面3D重建"
  - "度量重建"
  - "平面splatting"
  - "室内场景"
  - "前馈模型"
---

# Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2510.18714](https://arxiv.org/abs/2510.18714)  
**代码**: [项目页面](https://lck666666.github.io/plana3r/)  
**领域**: 3D视觉  
**关键词**: 平面3D重建, 度量重建, 平面splatting, 室内场景, 前馈模型

## 一句话总结

提出Plana3R，一个无需位姿和平面标注的前馈框架，从未配对的双视角图像中预测稀疏3D平面基元和度量尺度相对位姿，实现室内场景的零样本度量平面3D重建。

## 研究背景与动机

室内环境是人类生活的主要场所，创建其数字孪生对很多应用至关重要。室内场景天然富含平面结构（地板、墙壁、桌面等），因此平面基元是一种非常适合室内3D重建的紧凑表示。

**现有方法的两个关键限制**：

**标注依赖**：前馈平面重建方法（如SparsePlanes、NOPE-SAC）需要精确的平面掩码和3D平面标注进行训练，这种密集标注稀缺且制备复杂，严重限制了可用数据量和模型泛化能力。

**位姿依赖**：逐场景优化方法（如PlanarSplatting）需要精确配准的多视角稠密图像，在稀疏无位姿场景下无法使用。

**本文的核心insight**：室内环境的尺寸遵循人体尺度分布，平面3D表示天然具备预测度量3D几何的潜力。通过利用PlanarSplatting的可微平面渲染技术，可以仅用深度图和法线图作为监督（远比平面标注易获取），训练Transformer前馈模型直接预测稀疏平面基元和度量位姿。

## 方法详解

### 整体框架

输入为同一场景的两张无位姿图像 $I^1, I^2$ 及其内参 $\mathbf{K}^1, \mathbf{K}^2$。网络 $\mathcal{F}$ 输出一组3D平面基元（深度 $d_\pi$、半径 $\mathbf{r}_\pi$、四元数 $\mathbf{q}_\pi$）和6-DoF度量相对位姿 $P_{\text{rel}}$。通过PlanarSplatting的CUDA可微渲染器将平面基元渲染为深度图和法线图，与GT进行比较实现梯度反传。

### 关键设计

1. **基于ViT的编码-解码架构**：采用Siamese ViT编码器提取特征 $F^i \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times D_{\text{enc}}}$，然后通过带交叉注意力的Transformer解码器生成低分辨率嵌入 $G_{\text{low}}^i$。位姿头从拼接的双视角特征回归相对位姿。编码器和解码器用DUSt3R的预训练权重初始化。

2. **层次化基元预测架构（HPPA）**：从 $G_{\text{low}}$ 用三个回归头预测低分辨率（$\frac{H}{16} \times \frac{W}{16}$）平面基元。通过反卷积网络上采样得到 $G_{\text{high}}$，用同一组回归头预测高分辨率（$\frac{H}{8} \times \frac{W}{8}$）基元。关键问题是哪些区域用低分辨率、哪些用高分辨率。本文用一个简单启发式：计算低分辨率法线图 $\mathbf{N}_{\text{low}}^{\text{patch}}$ 的梯度幅值，梯度超过阈值 $g_{\text{th}}=0.5$ 的区域切换为高分辨率基元。法线变化大的区域需要更多小平面拟合，变化小的区域用少量大平面即可。

3. **无需平面标注的监督**：利用PlanarSplatting的CUDA可微渲染器将平面基元渲染为全分辨率深度图和法线图，直接与GT深度/法线图比较。法线图GT用Metric3Dv2生成伪标签。这使得模型可在只有深度和法线标注的大规模双视角数据集上训练，无需任何平面级标注。

4. **平面合并**：预测的平面基元通过法线和距离的阈值合并为语义一致的大平面，实现平面级实例分割——这是一种自然涌现的能力，无需额外训练。

### 损失函数 / 训练策略

**三类损失**：

- **Patch损失**（热身阶段）：直接在patch分辨率监督深度和法线：$\mathcal{L}_*^{\text{patch}} = \alpha_1\|1 - (\mathbf{N}_*^{\text{patch}})^\top\mathbf{N}_*^{\text{r.gt}}\|_1 + \alpha_1\|\mathbf{N}_*^{\text{patch}} - \mathbf{N}_*^{\text{r.gt}}\|_1 + \alpha_2\|\mathbf{D}_*^{\text{patch}} - \mathbf{D}_*^{\text{r.gt}}\|_1$

- **渲染损失**：通过可微渲染在全分辨率上监督：$\mathcal{L}_*^{\text{render}} = \beta_1\|1 - (\mathbf{N}_*^{\text{render}})^\top\mathbf{N}^{\text{gt}}\|_1 + \beta_1\|\mathbf{N}_*^{\text{render}} - \mathbf{N}^{\text{gt}}\|_1 + \beta_2\|\mathbf{D}_*^{\text{render}} - \mathbf{D}^{\text{gt}}\|_1$

- **位姿损失**：$\mathcal{L}^{\text{pose}} = \gamma_1\|\mathbf{t}^{\text{gt}} - \mathbf{t}\|_1 + \gamma_2\|\mathbf{q}^{\text{gt}} - \frac{\mathbf{q}}{\|\mathbf{q}\|}\|_1 + \gamma_3(1 - \frac{\mathbf{t} \cdot \mathbf{t}^{\text{gt}}}{\|\mathbf{t}\|\|\mathbf{t}^{\text{gt}}\|})$

**训练配置**：用4个数据集约400万图像对训练，先1 epoch热身仅用patch和位姿损失，再10 epoch用全部三种损失。输入分辨率 $512 \times 384$。总训练量256 GPU-days（H20 GPU）。

## 实验关键数据

### 主实验（双视角重建与位姿估计）

| 方法 | ScanNetV2 Chamfer↓ | ScanNetV2 F-score↑ | ScanNetV2 Trans Med(m)↓ | ScanNetV2 Rot Med(°)↓ |
|------|-------------------|-------------------|------------------------|----------------------|
| SparsePlanes | - | - | 0.56 | 15.46 |
| NOPE-SAC | 0.26 | 61.86 | 0.41 | 8.27 |
| MASt3R | 0.21 | 74.92 | 0.11 | 2.17 |
| **Plana3R** | **0.11** | **92.52** | **0.07** | **2.01** |

在Matterport3D上（零样本，未见过训练数据），Plana3R的F-score达56.63，超过在该数据集上训练的NOPE-SAC（54.96）。

### 消融实验（单目深度估计 NYUv2）

| 方法 | Rel↓ | RMSE↓ | δ₁↑ |
|------|------|-------|-----|
| PlaneRecTR | 0.157 | 0.547 | 74.2 |
| MASt3R | 0.152 | 0.51 | 83.0 |
| **Plana3R** | **0.132** | **0.463** | **86.4** |

Plana3R在从未见过的NYUv2上实现零样本度量深度估计，超越MASt3R。

### 关键发现

- 稀疏平面基元表示在结构化室内环境中比稠密点云更紧凑高效，同时保持高精度
- 平面合并后自然涌现的平面分割能力在Replica数据集上优于需要平面标注训练的PlaneRecTR
- 层次化基元预测通过梯度阈值自适应选择分辨率，实现768-3072之间的基元数量灵活调节
- 多视角重建通过逐对推理即可扩展到8帧以上输入

## 亮点与洞察

- 用可微平面渲染替代显式平面标注的监督方式非常优雅，大幅降低了数据需求
- 层次化基元预测的启发式（法线梯度阈值）简单有效，避免了额外学习开销
- 度量尺度的直接预测受益于室内场景的人体尺度先验，这一观察很有见地
- 平面分割作为紧凑表示的副产品自然涌现，体现了好表示自带上游能力的理念

## 局限与展望

- 当前仅支持双视角逐对推理，多视角需要多次前向传播
- 平面表示对非平面区域（曲面、复杂物体）建模能力有限
- 对法线伪标签（Metric3Dv2）的质量敏感
- 梯度阈值 $g_{\text{th}}$ 需要手动设置，未自适应学习

## 相关工作与启发

- **DUSt3R / MASt3R**：前馈双视角3D重建基础模型，但用稠密点云表示
- **PlanarSplatting**：本文核心依赖的可微平面渲染组件，提供CUDA加速的平面基元渲染
- **SparsePlanes / NOPE-SAC**：此前的双视角平面重建方法，需要平面标注
- 启发：利用领域结构先验（如室内平面性）选择更紧凑的表示，可在少量基元上超越稠密方法

## 评分

- **新颖性**: ⭐⭐⭐⭐ 无标注平面重建+层次化基元预测+度量尺度预测的组合创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 跨4个数据集、5个任务的全面评估，零样本泛化令人印象深刻
- **写作质量**: ⭐⭐⭐⭐ 模块化描述清晰，公式规范
- **价值**: ⭐⭐⭐⭐ 为室内3D重建提供了一种更紧凑、更有语义的替代方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)
- [\[NeurIPS 2025\] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Any4D: Unified Feed-Forward Metric 4D Reconstruction](../../CVPR2026/3d_vision/any4d_unified_feed-forward_metric_4d_reconstruction.md)
- [\[CVPR 2026\] AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend](../../CVPR2026/3d_vision/amb3r_accurate_feed-forward_metric-scale_3d_reconstruction_with_backend.md)
- [\[CVPR 2026\] REArtGS++: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting](../../CVPR2026/3d_vision/reartgs_generalizable_articulation_reconstruction_with_temporal_geometry_constra.md)

</div>

<!-- RELATED:END -->
