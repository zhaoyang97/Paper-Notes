---
title: >-
  [论文解读] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 本文提出CarGS，通过发现高斯基元对渲染和重建任务的贡献冲突根源在于协方差，设计了轻量残差结构Lite-Geo来自适应解耦两个任务的几何贡献，并引入法线+SDF双引导的致密化策略，在统一模型中同时实现SOTA的渲染质量和重建精度，且存储仅为双模型方法的9%。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "表面重建"
  - "新视角合成"
  - "贡献自适应正则化"
  - "统一框架"
---

# Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization

**会议**: CVPR 2025  
**arXiv**: [2503.00881](https://arxiv.org/abs/2503.00881)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 表面重建, 新视角合成, 贡献自适应正则化, 统一框架

## 一句话总结

本文提出CarGS，通过发现高斯基元对渲染和重建任务的贡献冲突根源在于协方差，设计了轻量残差结构Lite-Geo来自适应解耦两个任务的几何贡献，并引入法线+SDF双引导的致密化策略，在统一模型中同时实现SOTA的渲染质量和重建精度，且存储仅为双模型方法的9%。

## 研究背景与动机

**领域现状**：从多视角图像同时实现高质量渲染和精确3D表面重建是计算机视觉与图形学的核心挑战。3D高斯溅射（3DGS）以其高质量渲染和快速推理获得了广泛关注，但由于高斯点云的非结构化特性，保证精确几何重建仍很困难。

**现有痛点**：现有方法主要分两类：(1) 基于基元的方法（如SuGaR、2DGS、PGSR）对高斯基元施加几何正则化，但渲染和重建目标之间存在内在冲突，添加几何正则化会损害渲染质量；(2) 双模型方法（如GSDF）分别用3DGS做渲染、SDF做重建，计算和存储开销极大（1.7GB存储、3小时训练、非实时推理）。

**核心矛盾**：渲染需要一些远离表面的"浮动"高斯来捕捉丰富纹理，而重建要求高斯紧贴表面。两个任务对高斯贡献权重的需求相反——渲染倾向于增大远离表面的高斯权重，重建倾向于抑制它们。作者通过实验发现，丢弃50%高CD值（远离表面）的高斯后，重建质量F1几乎不变，但渲染PSNR急剧下降，直观证明了贡献冲突的存在。

**本文目标** 在统一模型内自适应调整每个高斯基元对渲染和重建任务的贡献，既保持渲染质量不降，又实现高精度重建。

**切入角度**：作者通过实验分析发现，协方差（而非不透明度）是造成贡献冲突的关键属性。因此可以为几何重建预测一组额外的协方差参数，与渲染协方差解耦，实现自适应贡献。

**核心 idea**：通过轻量残差MLP为每个高斯基元学习专用于几何的协方差偏移量，在渲染协方差基础上做残差预测，从而在统一模型中解耦两个任务的贡献需求。

## 方法详解

### 整体框架

CarGS基于Scaffold-GS的锚点框架。输入多视角图像，通过SfM获取初始点云和锚点。每个锚点通过MLP预测k个神经高斯的属性（颜色、位置偏移、协方差、不透明度）。核心创新是引入Lite-Geo残差模块来为几何预测额外的协方差，以及法线+SDF引导的几何致密化策略。最终通过TSDF融合从深度图提取网格。

### 关键设计

1. **Lite-Geo 残差结构（贡献自适应正则化）**:

    - 功能：为几何重建预测与渲染解耦的协方差参数，同时继承渲染任务中的隐式深度信息
    - 核心思路：在渲染MLP输出的协方差（scale和rotation）基础上，增加一个Geo-MLP预测残差偏移量。几何协方差 $y = M_\Sigma^{rgb}(\mathbf{f}, \theta; \phi_1) + \Delta y$，其中 $\Delta y = M_\Sigma^{geo}(\mathbf{f}, \theta; \phi_2)$，$\phi_2 = \lambda \phi_1$ 用渲染MLP参数的缩放初始化。这种残差设计让几何协方差继承了渲染优化过程中学到的隐式深度信息，避免单独几何MLP的过拟合问题。
    - 设计动机：直接用独立MLP预测几何协方差会严重过拟合，因为几何正则化（平面约束+跨视角约束）只提供连续性和一致性约束，缺乏准确的深度监督。渲染任务通过光度损失隐含了深度信息，残差结构可以有效利用这些信息。

2. **几何引导的致密化策略**:

    - 功能：利用法线和SDF双重线索引导高斯基元在高频细节区域生长
    - 核心思路：在传统基于梯度的致密化基础上，增加几何引导项。通过SDF值 $s = D(x) - Z(x)$ 和法线差异 $n = N_d(x) - N(x)$ 计算致密化准则 $\epsilon_g = \nabla_g + \omega_g(\zeta(s) \cdot (\omega_n n \text{ if } \mu(s) < \theta \text{ else } 1))$，其中 $\zeta(s)$ 是高斯函数，使距离零水平集越近的点越倾向于生成新高斯。法线差异大的区域（高频细节区域）也被优先致密化。
    - 设计动机：现有几何引导致密化（如GSDF）仅使用SDF值，无法充分捕捉高频细节区域。引入法线线索可以在表面附近但结构复杂的区域增加高斯密度，改善渲染锐度和重建精度。

3. **属性贡献的深入分析**:

    - 功能：识别造成渲染-重建冲突的核心属性
    - 核心思路：通过梯度detach实验，分别阻断不透明度和协方差在几何损失中的梯度。实验发现：阻断不透明度梯度对重建质量影响极小（F1仍为0.64 vs 0.65），而阻断协方差梯度导致重建质量剧烈下降（F1从0.65降至0.46）。这证明协方差是几何正则化影响渲染-重建贡献的主因。
    - 设计动机：只有精确定位了冲突根源，才能设计最小化干预的解决方案——只需为协方差增加几何分支，其余属性保持共享。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_c + \alpha \mathcal{L}_{plane} + \beta \mathcal{L}_{cross}$，其中 $\mathcal{L}_c$ 是RGB重建损失（L1+D-SSIM），$\mathcal{L}_{plane}$ 是平面约束（深度导出法线与渲染法线对齐），$\mathcal{L}_{cross}$ 是跨视角一致性约束（前向-后向投影误差）。$\alpha=0.01$，$\beta=0.2$。训练先用纯渲染目标预训练若干epoch，再加入Lite-Geo模块和几何损失联合优化。

## 实验关键数据

### 主实验

| 方法 | TnT F1↑ | TnT PSNR↑ | Mip360 PSNR↑ | 存储 | 训练时间 | FPS |
|------|---------|-----------|-------------|------|---------|-----|
| NeuS | 0.38 | 23.71 | - | 1.4GB | >24h | <1 |
| Neuralangelo | 0.50 | 25.43 | - | 4.2GB | >24h | <1 |
| GSDF | 0.46 | 26.05 | - | 1.7GB | 3h | <1 |
| PGSR | 0.60 | 26.12 | 27.43 | 0.42GB | 1.2h | 103 |
| **CarGS** | **0.65** | **26.41** | **27.68** | **0.16GB** | 1.2h | 90 |

CarGS在TnT数据集上F1达到0.65（超过PGSR的0.60），同时PSNR 26.41也是最高。存储仅为PGSR的38%、GSDF的9%。

### 消融实验

| 配置 | F1↑ | PSNR↑ | 说明 |
|------|-----|-------|------|
| (a) 直接加几何损失 | 0.38 | 25.72 | 基础方法重建质量差 |
| (b) + Geo-MLP | 0.44 | 25.75 | 重建提升但有过拟合 |
| (c) + Lite-Geo残差 | 0.62 | 26.13 | 残差结构大幅提升两个任务 |
| (d) + Geo-Densify | **0.65** | **26.41** | 几何致密化进一步增益 |

### 关键发现

- Lite-Geo残差结构是最关键的设计，从(a)到(c)的F1提升为0.38→0.62（+63%），PSNR从25.72→26.13。
- 几何致密化策略在Lite-Geo基础上进一步带来F1 0.62→0.65、PSNR 26.13→26.41的提升。
- 协方差而非不透明度是贡献冲突的根源：detach协方差梯度导致F1从0.65暴跌到0.46。
- 在Mip-NeRF360数据集上，CarGS在所有GS方法中渲染质量最高（PSNR 27.68），同时不牺牲重建精度。

## 亮点与洞察

- **协方差是冲突根源的分析非常精炼**：通过简单的梯度detach实验就精准定位了问题，避免了对整个框架的过度改造。这种通过控制变量实验找关键因子的分析方法值得学习。
- **残差结构的双重巧妙**：(1)继承渲染协方差中的隐式深度信息，避免几何分支的过拟合；(2)通过缩放参数$\lambda$初始化，保证初始预测接近渲染协方差，收敛更稳定。
- **存储效率惊人**：0.16GB vs GSDF的1.7GB，10倍压缩的同时性能更好。这得益于统一模型设计+锚点框架的紧凑性。

## 局限与展望

- 法线引导致密化依赖深度图导出的法线质量，在深度不连续或遮挡边界处可能不可靠。
- 在室外大规模场景（如完整TnT Courthouse场景）的重建质量仍有提升空间（F1=0.17）。
- Lite-Geo的残差权重$\lambda$和几何致密化的超参$\omega_g$、$\omega_n$需要手动调参。
- 目前仅支持静态场景，扩展到动态场景的贡献自适应更具挑战。

## 相关工作与启发

- **vs PGSR**: PGSR是典型的基元正则化方法，直接用几何损失约束共享的高斯属性，导致渲染-重建冲突。CarGS通过Lite-Geo解耦协方差解决了这个问题，F1从0.60提升到0.65，PSNR也有提升。
- **vs GSDF**: GSDF使用双模型（3DGS+SDF）分别处理两个任务，虽然效果好但计算和存储开销巨大。CarGS在统一模型中达到更好的效果，训练时间减少60%，存储减少91%，且支持实时渲染。
- **vs 2DGS**: 2DGS通过2D高斯基元改善几何对齐，但也面临渲染质量下降的问题。CarGS的自适应贡献策略提供了一个更通用的解决思路。

## 评分

- 新颖性: ⭐⭐⭐⭐ 贡献自适应的idea新颖，通过协方差解耦的切入角度精准
- 实验充分度: ⭐⭐⭐⭐ TnT和Mip-NeRF360上有充分比较，消融设计到位
- 写作质量: ⭐⭐⭐⭐ 分析部分逻辑清晰，图表设计好
- 价值: ⭐⭐⭐⭐ 为统一渲染+重建建立了强基线，贡献解耦的思想可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction](../../CVPR2026/3d_vision/plug-and-play_pde_optimization_for_3d_gaussian_splatting_toward_high-quality_ren.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2025\] SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)

</div>

<!-- RELATED:END -->
