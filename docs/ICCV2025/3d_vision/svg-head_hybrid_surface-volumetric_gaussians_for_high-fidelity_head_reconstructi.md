---
title: >-
  [论文解读] SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing
description: >-
  [ICCV 2025][3D视觉][头部重建] 提出SVG-Head，通过表面高斯(显式纹理图)和体积高斯(非朗伯区域补充建模)的混合表示，首次实现高保真高斯头部化身的实时外观编辑。 创建高保真且可编辑的头部化身面临核心困难： NeRF隐式表示：难以编辑 3DGS颜色绑定：每个高斯独立存储颜色，缺乏全局外观解耦…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "头部重建"
  - "3D高斯"
  - "纹理编辑"
  - "FLAME网格"
  - "表面-体积混合"
---

# SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing

**会议**: ICCV 2025  
**arXiv**: [2508.09597](https://arxiv.org/abs/2508.09597)  
**代码**: [项目页](https://heyy-sun.github.io/SVG-Head/)  
**领域**: 3D视觉  
**关键词**: 头部重建, 3D高斯, 纹理编辑, FLAME网格, 表面-体积混合

## 一句话总结

提出SVG-Head，通过表面高斯(显式纹理图)和体积高斯(非朗伯区域补充建模)的混合表示，首次实现高保真高斯头部化身的实时外观编辑。

## 研究背景与动机

创建高保真且可编辑的头部化身面临核心困难：

**NeRF隐式表示**：难以编辑

**3DGS颜色绑定**：每个高斯独立存储颜色，缺乏全局外观解耦，无法实时编辑纹理

**现有编辑方法局限**：GaussianAvatar-Editor需要text-to-image模型辅助，MeGA需要数分钟优化

SVG-Head的关键洞察：通过表面高斯将全局外观解耦为可学习纹理图，实现实时编辑。

## 方法详解

### 表面高斯 (surf-GS)

- 约束在FLAME网格面上，通过重心坐标定位
- **Mesh-aware Gaussian UV映射**：将3D位置映射到FLAME UV空间
    - 将ray-Gaussian交点投影到关联三角面上
    - 重心插值获取UV坐标，简化为单一仿射变换：
  $$\phi(I(\mathbf{r}_p, \mathcal{G}_i)) = \phi(\boldsymbol{\mu}_i) + T(\boldsymbol{\mu}_i)(I(\mathbf{r}_p, \mathcal{G}_i) - \boldsymbol{\mu}_i)$$

- **一致UV坐标**：约束高斯中心在网格面上 $\boldsymbol{\mu}_i = \xi_A \mathbf{v}_A + \xi_B \mathbf{v}_B + \xi_C \mathbf{v}_C$，旋转对齐法线，确保每个像素有唯一UV

- **动态纹理**：$\mathcal{T} = \mathcal{T}_{\text{diff}} + \mathcal{T}_{\text{dy}}$，其中动态纹理由表情参数 $\psi$ 条件化的卷积网络生成

### 体积高斯 (vol-GS)

表面高斯受限于FLAME表面，对嘴唇、头发等非朗伯区域建模不足。体积高斯允许：
- 在网格附近自由移动
- 独立存储颜色（不从纹理采样）
- 保持FLAME绑定以支持动画

### 微分混合渲染

统一两种高斯的颜色计算：

$$\mathcal{C}(\mathcal{G}_i, \mathbf{r}_p) = \begin{cases} \mathbf{c}_i^{\text{SH}} & \text{if } v_i = 1 \text{ (vol-GS)} \\ h(\phi(I(\mathbf{p}, \mathcal{T}_{\text{dy}} + \mathcal{T}_{\text{diff}})) + \mathbf{c}_i^{\text{SH}_{\text{res}}} & \text{if } v_i = 0 \text{ (surf-GS)} \end{cases}$$

### 分层优化策略

**第一阶段**：仅优化表面高斯和FLAME参数
- 光度损失 $\mathcal{L}_{\text{rgb}}$（L1 + D-SSIM）
- 漫反射光度损失 $\mathcal{L}_{\text{rgb}}^{\text{diff}}$（正则化纹理图）
- 缩放损失 $\mathcal{L}_{\text{scale}}$

**第二阶段**：联合优化两种高斯
- 冻结大部分surf-GS参数（仅优化 $o^{(s)}$ 和 $\mathcal{T}_{\text{dy}}$）
- 添加位置正则 $\mathcal{L}_{\text{pos}}$ 和alpha正则 $\mathcal{L}_a = \|\mathbf{A_s} - 1\|_2^2$

## 实验

### NeRSemble数据集

| 方法 | 可编辑 | NVS-PSNR↑ | NVS-SSIM↑ | NVS-LPIPS↓ |
|------|--------|-----------|-----------|------------|
| PointAvatar | ✗ | 25.8 | 0.893 | 0.097 |
| Gaussian Head Avatar | ✗ | 29.5 | 0.894 | 0.084 |
| GaussianAvatars | ✗ | 31.6 | 0.938 | 0.065 |
| MeGA | 慢 | **32.0** | **0.940** | **0.062** |
| **SVG-Head** | **实时** | 31.8 | 0.939 | 0.063 |

SVG-Head在可编辑方法中metrics最优，与不可编辑方法的MeGA持平，同时支持实时编辑。

### 编辑能力对比

| 方法 | 编辑方式 | 时间 |
|------|---------|------|
| MeGA | 优化 | 分钟~小时 |
| GaussianAvatar-Editor | text-to-image | 非实时 |
| **SVG-Head** | **直接纹理编辑** | **实时** |

## 亮点与洞察

1. **首个显式纹理图的高斯头部化身**：实现了3DGS头部化身的实时外观编辑
2. **UV一致性保证**：约束高斯在面上+旋转对齐法线，解决纹理模糊问题
3. **互补式设计**：surf-GS负责全局外观编辑，vol-GS补充复杂区域
4. **分层优化避免耦合**：先独立优化surf-GS获得清晰纹理，再联合优化

## 局限性

- 纹理分辨率受UV图大小限制
- 体积高斯部分不支持通过纹理编辑
- 依赖FLAME跟踪质量
- 对极端表情的动态纹理生成可能不够精确

## 相关工作

- GaussianAvatars: FLAME绑定3DGS
- MeGA: 网格+高斯混合但编辑慢
- Texture-GS: 静态场景纹理高斯

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (首个支持实时纹理编辑的高斯头部)
- 技术深度: ⭐⭐⭐⭐⭐ (UV映射+混合渲染+分层优化)
- 实验充分度: ⭐⭐⭐⭐ (重建+编辑+消融)
- 实用价值: ⭐⭐⭐⭐⭐ (实时编辑是杀手功能)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)
- [\[ICCV 2025\] HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity](hineus_high-fidelity_neural_surface_mitigating_low-texture_and_reflective_ambigu.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](../../CVPR2025/3d_vision/hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2026\] PercHead: Perceptual Head Model for Single-Image 3D Head Reconstruction & Editing](../../CVPR2026/3d_vision/perchead_perceptual_head_model_for_single-image_3d_head_reconstruction_editing.md)
- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)

</div>

<!-- RELATED:END -->
