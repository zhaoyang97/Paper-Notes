---
title: >-
  [论文解读] Radiant Foam: Real-Time Differentiable Ray Tracing
description: >-
  [ICCV 2025][3D视觉][可微渲染] 提出 Radiant Foam，一种基于体积网格（tetrahedral mesh）光线追踪的新型可微场景表示，在不依赖光栅化的前提下达到了与 Gaussian Splatting 相当的渲染速度和质量，同时天然支持反射、折射等光传输现象。 领域现状：可微场景表示正朝着更高效、…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "可微渲染"
  - "光线追踪"
  - "体素网格"
  - "场景表示"
  - "实时渲染"
---

# Radiant Foam: Real-Time Differentiable Ray Tracing

**会议**: ICCV 2025  
**arXiv**: [2502.01157](https://arxiv.org/abs/2502.01157)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 可微渲染, 光线追踪, 体素网格, 场景表示, 实时渲染

## 一句话总结

提出 Radiant Foam，一种基于体积网格（tetrahedral mesh）光线追踪的新型可微场景表示，在不依赖光栅化的前提下达到了与 Gaussian Splatting 相当的渲染速度和质量，同时天然支持反射、折射等光传输现象。

## 研究背景与动机

**领域现状**：可微场景表示正朝着更高效、实时的方向演进。近年来 3D Gaussian Splatting (3DGS) 凭借光栅化的高效性迅速流行，成为辐射场方法的主流替代品。光栅化相比传统基于光线的渲染（ray-based rendering）具有显著速度优势，因其可充分利用 GPU 光栅化硬件。

**现有痛点**：光栅化的效率建立在大量近似之上——它假设每个 splat 是平面的、独立排序的，并以 tile-based 方式处理。这些近似使得反射、折射等光传输现象的实现变得极其困难。虽然基于 NeRF 的方法天然支持任意光线路径，但其体积采样计算量太大，无法实时；而基于硬件 RT core 的 Gaussian 光线追踪方法则依赖特定硬件 API（如 OptiX），可移植性差。

**核心矛盾**：渲染效率与物理正确性之间存在 trade-off——光栅化高效但物理近似严重，光线追踪物理正确但实时性不足。

**本文目标**：设计一种既保持光线追踪物理灵活性、又达到实时渲染速度的场景表示方法。

**切入角度**：作者注意到计算机图形学中存在一种被忽视但高度高效的体积网格光线追踪算法——基于四面体网格的逐单元遍历（cell-by-cell traversal），其复杂度与光线穿过的单元数成正比而非总单元数。

**核心 idea**：用四面体网格（tetrahedral mesh）作为辐射场的空间结构，每个四面体顶点存储辐射场属性，利用经典的四面体网格光线追踪实现高效、无近似的可微渲染。

## 方法详解

### 整体框架

Radiant Foam 的整体 pipeline 如下：输入是多视角图像及对应相机参数，输出是场景的新视角渲染图。场景被表示为一个 3D Delaunay 四面体网格，网格顶点携带密度和球谐系数（SH）等属性。渲染时，对每条射线，方法沿四面体网格逐单元遍历，在每个单元内通过重心坐标插值获取连续的辐射场值，最终通过体积渲染积分得到像素颜色。整个过程完全可微，支持端到端训练。

### 关键设计

1. **四面体网格场景表示（Tetrahedral Mesh Representation）**:

    - 功能：用 3D Delaunay 四面体剖分作为辐射场的空间结构
    - 核心思路：将一组 3D 点（初始化自 SfM 或随机）构建 Delaunay 三角剖分，形成四面体网格。每个顶点存储密度 $\sigma$ 和球谐系数。在四面体内部，通过重心坐标对顶点属性做线性插值，获得空间中任意点的辐射场值。这种表示保证了空间划分的完备性和连续性。
    - 设计动机：相比 Gaussian Splatting 的离散 splat 集合，四面体网格提供了连续的空间划分，避免了 splat 重叠和排序问题；相比 NeRF 的均匀/分层采样，四面体网格的自适应大小自然实现了空间自适应分辨率。

2. **高效四面体光线遍历（Efficient Tetrahedral Ray Traversal）**:

    - 功能：在四面体网格中高效追踪光线路径
    - 核心思路：利用 Lagae & Dutré (2008) 提出的四面体网格光线遍历算法。对于每条射线，首先找到其进入的第一个四面体，然后逐个穿过相邻四面体（通过共享面的邻接关系），直到射线离开场景。每个遍历步骤只需一次面-射线相交测试（计算光线与三角面的交点），复杂度为 $O(k)$，其中 $k$ 是光线穿过的四面体数量，而非总数量。这比基于 BVH 的方法更轻量，也无需硬件 RT core。
    - 设计动机：四面体网格的连续性保证了从一个单元到下一个单元的遍历是确定性的（每个面恰好被两个四面体共享），避免了传统 BVH 中的复杂遍历逻辑。这使得 GPU shader 编程实现简单高效，无需特殊硬件支持。

3. **可微体积渲染与自适应密度控制**:

    - 功能：支持端到端可微渲染并自适应调整场景分辨率
    - 核心思路：体积渲染积分在每个四面体内是解析可计算的——因为密度在四面体内线性变化（重心插值），透射率 $T$ 和颜色积分都有闭式解。反向传播时，梯度可沿遍历路径直接传播到各顶点属性。训练过程中通过增删顶点实现自适应密度控制：在误差大的区域增加顶点（分裂）、在贡献低的区域删除顶点（剪枝），动态更新 Delaunay 网格。
    - 设计动机：闭式积分避免了逐步采样的离散化误差，而自适应控制使得模型可以将更多表示能力分配给复杂区域，同时保持稀疏区域的高效性。

### 损失函数 / 训练策略

训练采用标准的光度重建损失，包括 L1 损失和 D-SSIM 损失的加权组合：$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$，其中 $\lambda=0.2$。训练过程中周期性地进行自适应控制——每隔一定迭代执行顶点分裂/剪枝，并重建 Delaunay 网格。优化器使用 Adam，分别为位置和球谐系数设置不同学习率。

## 实验关键数据

### 主实验

在 Mip-NeRF360、Tanks&Temples、Deep Blending 等标准数据集上与 3DGS 及变体对比：

| 数据集 | 指标 | Radiant Foam | 3DGS | Mini-Splatting | 提升 |
|--------|------|-------------|------|----------------|------|
| Mip-NeRF360 | PSNR(dB) | 27.35 | 27.21 | 27.36 | +0.14 vs 3DGS |
| Mip-NeRF360 | SSIM | 0.813 | 0.815 | 0.819 | 持平 |
| Mip-NeRF360 | FPS | 148 | 159 | 165 | ~93% of 3DGS |
| Tanks&Temples | PSNR(dB) | 23.72 | 23.14 | 23.58 | +0.58 vs 3DGS |
| Deep Blending | PSNR(dB) | 29.65 | 29.41 | 29.23 | +0.24 vs 3DGS |

### 消融实验

| 配置 | PSNR(dB) | FPS | 说明 |
|------|---------|-----|------|
| Full model | 27.35 | 148 | 完整模型 |
| w/o 自适应控制 | 26.42 | 112 | 固定点数，质量和速度均下降 |
| w/o SH（改用低阶颜色） | 26.81 | 162 | 速度略提升但质量显著下降 |
| 用均匀采样替代逐单元遍历 | 26.15 | 42 | 速度骤降，证明遍历算法关键 |
| 增加初始点数 (2x) | 27.41 | 125 | 质量微提但速度下降 |

### 关键发现

- 四面体逐单元遍历是速度的核心保障，替换为均匀采样后 FPS 从 148 降至 42
- Radiant Foam 在反射/折射场景中的优势更明显，因光线追踪天然支持二次射线
- 自适应密度控制对复杂场景（室外360°）效果最大，对简单场景提升有限
- 无需硬件 RT core 是重要的实用优势，使方法可在任何现代 GPU 上运行

## 亮点与洞察

- **重新发掘经典算法**：四面体网格遍历是图形学的经典方法，但在 NeRF/3DGS 时代被忽视。作者将其与可微渲染结合，巧妙地在光线追踪和实时渲染之间找到平衡点。
- **无需特殊硬件**：不同于依赖 OptiX/RT core 的方法，Radiant Foam 仅用标准 GPU shader 实现，可移植性极强。这对跨平台部署和嵌入式应用非常有价值。
- **连续场表示**：四面体网格提供的连续空间划分可以迁移到需要连续密度场的其他任务，如物理仿真中的流体/烟雾渲染、医学成像中的体数据可视化。

## 局限与展望

- 四面体网格的构建和更新（Delaunay 重建）有一定计算开销，尤其在大规模场景中可能成为瓶颈
- 当前四面体内用线性插值，对高频纹理细节的建模能力可能弱于 3DGS 的各向异性 Gaussian
- 作者未展示大规模城市场景的实验结果，场景可扩展性存疑
- 可以探索高阶插值（如二次重心坐标）来提升细节表达能力
- 与 3DGS 的混合方案——近处用 Gaussian splatting、远处用四面体网格——可能是一个有趣的方向

## 相关工作与启发

- **vs 3D Gaussian Splatting**: 3DGS 通过光栅化实现极致速度但牺牲了光传输灵活性；Radiant Foam 通过高效光线追踪保持灵活性的同时接近 3DGS 速度。在需要反射/折射的场景（如透明物体、镜面）Radiant Foam 有天然优势。
- **vs 3DGS-RT (NVIDIA)**: 同为光线追踪 Gaussian 方案，但 3DGS-RT 依赖 NVIDIA OptiX RT core 硬件加速，移植性差；Radiant Foam 纯软件实现更通用。
- **vs Instant-NGP/Zip-NeRF**: 基于哈希网格的隐式方法渲染速度仍慢于 Radiant Foam，且同样面临光传输建模困难。

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心贡献是将经典四面体遍历与现代可微渲染结合，思路独特但算法本身不是新的
- 实验充分度: ⭐⭐⭐⭐ 标准benchmark全面对比，但缺少反射/折射场景的针对性实验
- 写作质量: ⭐⭐⭐⭐ 清晰地解释了动机和方法，图示直观
- 价值: ⭐⭐⭐⭐ 为可微渲染提供了光栅化之外的实时替代方案，具有实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Lens Component Deletion based on Differentiable Ray Tracing](../../CVPR2026/3d_vision/lens_component_deletion_based_on_differentiable_ray_tracing.md)
- [\[CVPR 2026\] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting](../../CVPR2026/3d_vision/stochastic_ray_tracing_for_the_reconstruction_of_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](../../CVPR2025/3d_vision/irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)

</div>

<!-- RELATED:END -->
