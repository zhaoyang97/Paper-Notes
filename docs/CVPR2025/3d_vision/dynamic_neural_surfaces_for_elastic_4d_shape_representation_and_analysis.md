---
title: >-
  [论文解读] Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis
description: >-
  [CVPR 2025][3D视觉][4D形状分析] 本文提出 Dynamic Spherical Neural Surfaces (D-SNS)，用 MLP 将 genus-0 的 4D 表面建模为时空连续函数，然后在 SRNF/SRVF 空间中直接完成时空配准、测地线计算和均值估计，无需离散化，在 4D 人体和面部数据集上超越了 4D Atlas。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "4D形状分析"
  - "神经表面表示"
  - "时空配准"
  - "黎曼几何"
  - "弹性度量"
---

# Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis

**会议**: CVPR 2025  
**arXiv**: [2503.03132](https://arxiv.org/abs/2503.03132)  
**代码**: [https://4d-dsns.github.io/DSNS/](https://4d-dsns.github.io/DSNS/)  
**领域**: 人体理解 / 3D视觉  
**关键词**: 4D形状分析, 神经表面表示, 时空配准, 黎曼几何, 弹性度量

## 一句话总结
本文提出 Dynamic Spherical Neural Surfaces (D-SNS)，用 MLP 将 genus-0 的 4D 表面建模为时空连续函数，然后在 SRNF/SRVF 空间中直接完成时空配准、测地线计算和均值估计，无需离散化，在 4D 人体和面部数据集上超越了 4D Atlas。

## 研究背景与动机

**领域现状**：4D 形状分析（即随时间形变的 3D 表面统计分析）是计算机视觉与图形学中的重要课题。传统方法将 4D 表面离散化为三角网格序列，在离散点上计算配准、测地线和统计量。SMPL 等参数化模型虽能处理人体运动，但属于类别特定模型，无法跨类别分析。

**现有痛点**：传统离散化方法的性能依赖网格采样质量和分辨率。当时间采样变稀疏时，离散方法的精度明显下降。此外，先离散再分析的流程可能导致次优解。弹性度量下的非线性优化在离散域上计算量极大。

**核心矛盾**：4D 形状分析需要同时解决空间配准（不同参数化）和时间配准（不同变形速度）两个问题，而现有方法必须先离散化才能进行这些操作，这两步之间产生信息损失。

**本文目标**：将 4D 表面表示为空间和时间上的连续函数，在连续域上直接完成时空配准、测地线和统计量计算。

**切入角度**：作者借鉴神经隐式表示的思路（如 NeRF、NeuS），将球面参数化的 genus-0 表面用 MLP 拟合为连续映射。关键观察是——对于球面参数化的表面，SRNF 映射可以将复杂的弹性度量简化为 $\mathbb{L}^2$ 度量。

**核心 idea**：用 MLP 将 4D 表面编码为 $\mathbb{S}^2 \times [0,1] \to [-1,1]^3$ 的连续函数（D-SNS），然后通过 SRNF/SRVF 空间在连续域上完成全部形状分析任务。

## 方法详解

### 整体框架
输入是离散的 4D 表面（genus-0 三角网格序列），经球面参数化后用 MLP 拟合为连续的 D-SNS 表示。然后在 SRNF 空间中进行空间配准，在 SRVF 空间中进行时间配准，最终可计算测地线和 Karcher 均值。所有分析任务直接在神经表示上完成，仅在可视化时才离散化。

### 关键设计

1. **Dynamic Spherical Neural Surfaces (D-SNS)**:

    - 功能：将离散 4D 表面编码为时空连续函数
    - 核心思路：MLP $F_\Theta: \mathbb{S}^2 \times [0,1] \to [-1,1]^3$ 将球面坐标 $s$ 和时间 $t$ 映射到 3D 表面点。网络由 6 个残差块组成，每块含两个 1024 节点的层，使用 SoftPlus 激活函数确保光滑性。对空间域和时间域均施加位置编码。训练时最小化预测点与真实点的 MSE，batch size=80K 点。
    - 设计动机：连续表示使得法线场、切向场等微分量可通过自动微分直接计算，无需离散近似。残差连接显著提升了对粗细几何细节的表示能力。

2. **空间配准（SRNF 空间）**:

    - 功能：消除两个 3D 表面之间的参数化差异
    - 核心思路：将表面映射到 Square Root Normal Fields (SRNF) 空间，其中弹性度量变为 $\mathbb{L}^2$ 度量。空间微分同胚 $\gamma: \mathbb{S}^2 \to \mathbb{S}^2$ 用球谐基的加权和表示，旋转 $O \in SO(3)$ 用 SVD 求解。两者交替优化直至收敛，整个过程冻结 D-SNS 权重，仅优化微分同胚参数。
    - 设计动机：SRNF 将非线性弹性度量线性化，使得配准优化变得高效；球谐基天然保证了微分同胚的光滑性。

3. **时间配准（SRVF 空间）**:

    - 功能：对齐两个 4D 表面的变形速度差异
    - 核心思路：先用 PCA 将 4D 表面降维为低维曲线，再映射到 Square Root Velocity Fields (SRVF) 空间。时间微分同胚 $\zeta: [0,1] \to [0,1]$ 用一个小型 MLP（2 个残差块，32 神经元）实现，输出经 Sigmoid 约束在 $[0,1]$。通过正则项 $L_M = \int_0^1 \max(0, -\partial\zeta/\partial t)$ 强制单调性。
    - 设计动机：直接在 4D 表面上做时间配准维度太高且度量非线性。PCA 降维 + SRVF 映射双重简化使问题变为求解低维欧氏空间中的曲线弹性配准。

### 损失函数 / 训练策略
- D-SNS 表示学习：MSE 损失，训练 50K epochs，每 epoch 随机采样表面点
- 空间配准：SRNF 空间的 $\mathbb{L}^2$ 距离，500 次迭代
- 时间配准：$L = \|q_1 - q_2 \circ \zeta\|^2 + \lambda L_M$，训练 3000 epochs，每 200 epochs 更换时间采样点
- 均值计算：Karcher 均值在 SRVF 空间中通过 Eqn.15 联合优化得到

## 实验关键数据

### 主实验：表示精度

| 数据集 | 均值误差 ($\times 10^{-6}$) | 中值误差 ($\times 10^{-6}$) | 标准差 ($\times 10^{-6}$) |
|--------|------|------|------|
| DFAUST | 1.60 | 0.52 | 1.52 |
| CAPE | 1.51 | 0.68 | 1.27 |
| COMA | 2.29 | 1.68 | 1.66 |
| VOCA | 0.89 | 0.51 | 0.79 |

### 时空配准对比（测地距离，越小越好）

| 方法 | CAPE | DFAUST | COMA | VOCA |
|------|------|--------|------|------|
| 4D Atlas (Mean/Std/Med) | 1.29/0.48/1.40 | 2.38/1.12/1.87 | 0.06/0.02/0.06 | 0.50/0.11/0.41 |
| Ours (Mean/Std/Med) | 0.36/0.17/0.32 | 0.77/0.20/0.72 | 0.03/0.02/0.02 | 0.10/0.03/0.10 |

### 关键发现
- D-SNS 表示误差在 $10^{-6}$ 量级，逐点误差 <0.01，忠实还原复杂 4D 表面
- 仅用 30 个时间采样训练的 D-SNS 也能高质量插值缺失帧
- 当时间采样从 50 降到 25 时，4D Atlas 配准误差明显上升，而本文方法保持稳定——连续表示对采样密度不敏感
- 在所有四个数据集上，本文时空配准的测地距离均显著低于 4D Atlas

## 亮点与洞察
- **SRNF/SRVF 的巧妙利用**：将非线性弹性度量在不同空间中线性化，使得空间和时间配准分别变为欧氏空间的优化问题。这种"找个好的表示空间让问题变简单"的思路非常经典。
- **连续表示的优势**：D-SNS 使所有微分量可解析计算，配准不依赖离散分辨率。这为"先连续、后分析"的范式提供了有力证据。
- **可迁移性**：MLP 作为连续函数逼近器 + 弹性度量空间分析的组合，可以迁移到任何需要统计分析形变物体的任务，如医学图像中器官的纵向分析。

## 局限与展望
- **仅支持 genus-0 表面**：依赖球面参数化，无法处理有洞或高亏格的物体
- **计算效率**：每个 4D 表面需要单独训练一个 D-SNS（2-3小时），分析大规模数据集时代价高。作者提到未来可探索类似 DeepSDF 的条件化表示
- **未考虑纹理/外观**：仅处理几何形状，不涉及外观信息
- **数据集受限**：所有数据集均为已配准的三角网格，未验证从原始扫描数据直接建模的能力

## 相关工作与启发
- **vs 4D Atlas**：4D Atlas 在离散信号上操作，配准精度依赖时间采样密度；本文在连续函数上操作，对采样稀疏更鲁棒，四个数据集上全面超越
- **vs SMPL/SMPL-X**：SMPL 通过骨骼关节显式建模关节运动，是类别特定模型；本文方法是通用的 genus-0 表面分析框架，不限类别
- **vs Neural Surface Maps**：Morreale et al. 的 Neural Surface Maps 用 MLP 表示静态表面映射，本文将其推广到动态 4D 表面并引入完整的统计分析框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 将神经表示与黎曼几何统计分析结合的思路有一定新意，但各组件（MLP拟合、SRNF/SRVF）均为已有技术
- 实验充分度: ⭐⭐⭐⭐ 四个数据集上的定量和定性评估较全面，但仅与 4D Atlas 一个方法对比
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，但全文较长（22页），信息密度可提高
- 价值: ⭐⭐⭐⭐ 为 4D 形状分析提供了坚实的连续化框架，但应用场景相对小众

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] 4Deform: Neural Surface Deformation for Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)
- [\[CVPR 2025\] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes](volumetric_surfaces_representing_fuzzy_geometries_with_layered_meshes.md)
- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)
- [\[CVPR 2025\] 3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping](3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)

</div>

<!-- RELATED:END -->
