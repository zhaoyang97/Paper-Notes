---
title: >-
  [论文解读] Deformable Radial Kernel Splatting
description: >-
  [CVPR 2025][3D视觉][高斯泼溅] 提出可变形径向核 (DRK) 来泛化传统高斯泼溅，通过可学习的径向基函数、$L_1$/$L_2$ 范数混合和边缘锐化机制，用更少的图元实现更高质量的3D场景渲染。 3D Gaussian Splatting (3DGS) 虽然取得了巨大成功，但高斯核存在三个内在限制： 1. 径…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "高斯泼溅"
  - "可变形核"
  - "新视图合成"
  - "3D场景表示"
  - "光栅化"
---

# Deformable Radial Kernel Splatting

**会议**: CVPR 2025  
**arXiv**: [2412.11752](https://arxiv.org/abs/2412.11752)  
**代码**: [https://yihua7.github.io/DRK-web/](https://yihua7.github.io/DRK-web/)  
**领域**: 3D视觉  
**关键词**: 高斯泼溅, 可变形核, 新视图合成, 3D场景表示, 光栅化

## 一句话总结

提出可变形径向核 (DRK) 来泛化传统高斯泼溅，通过可学习的径向基函数、$L_1$/$L_2$ 范数混合和边缘锐化机制，用更少的图元实现更高质量的3D场景渲染。

## 研究背景与动机

3D Gaussian Splatting (3DGS) 虽然取得了巨大成功，但高斯核存在三个内在限制：
1. **径向对称性约束**：投影到屏幕空间总是椭圆，无法高效表示矩形、三角形等多样形状
2. **平滑边界约束**：$L_2$ 范数仅产生圆锥曲线边界，难以表示直线边缘
3. **尺度-锐度耦合**：高斯分布中衰减率和空间范围通过协方差矩阵耦合，锐利特征需要窄分布，难以同时捕获急剧过渡和大空间范围

这导致即使是简单的三角形、矩形等基本形状，也需要数千个高斯图元来近似，造成过度参数化。

## 方法详解

### 整体框架

DRK 是一种基于2D平面的新型图元，扩展了传统 2DGS。每个 DRK 由参数集 $\Theta=\{\mu, q, s_k, \theta_k, \eta, \tau, o, sh\}$ 定义，其中 $\{s_k, \theta_k\}_{k=1}^K$ 控制形状，$\eta$ 控制边界曲率，$\tau$ 控制锐度。渲染管线基于3DGS，增加了多边形裁剪和缓存排序策略。

### 关键设计

1. **可学习径向基函数**:
    - 功能：通过 $K$ 个控制点定义核形状，突破高斯的径向对称限制
    - 核心思路：每个控制点用极坐标 $(s_k, \theta_k)$ 表示，$s_k$ 为径向长度，$\theta_k$ 为极角。对任意点 $(u,v)$，用余弦权重在相邻径向基间平滑插值：$\alpha = o \cdot \exp(-\frac{r_2^2}{2}(\frac{1+\cos(\Delta\theta_k)}{2s_k^2} + \frac{1-\cos(\Delta\theta_k)}{2s_{k+1}^2}))$
    - 设计动机：当 $K=4$ 且角度为 $k\pi/2$、相对轴尺度相同时，可退化为标准 2D 高斯，保证向后兼容

2. **$L_1$/$L_2$ 范数混合**:
    - 功能：实现从曲线到直线边界的连续控制
    - 核心思路：引入混合权重 $\eta \in (0,1)$，完整核函数为 $\alpha = o \cdot \exp(-\frac{1}{2}(\eta r_1^2 + (1-\eta)\frac{r_2^2}{\bar{s}^2}))$。$L_1$ 范数通过相邻端点逆变换计算，其菱形单位球映射到端点间的直线段
    - 设计动机：$L_2$ 范数只能产生圆锥曲线，$L_1$ 范数可产生直线边界，混合两者可灵活表示人造环境中常见的线性边缘

3. **边缘锐化函数**:
    - 功能：解耦核的空间范围和边缘锐度
    - 核心思路：引入分段线性映射函数 $\Psi(g)$，由锐化系数 $\tau \in (-1,1)$ 控制，将密度值重新映射向0或1，产生更锐利的边缘过渡同时保持空间范围。最终 $\alpha = o \cdot \Psi(g)$
    - 设计动机：高斯核中锐利特征需要窄分布，无法同时表达大范围和锐利边缘；$\Psi$ 将两者解耦

### 渲染优化

- **低通滤波**: 采用视角相关的低通滤波器 $\tilde{\alpha} = \max(\alpha, o \cdot \exp(\cdot))$，按视角余弦缩放滤波器大小，防止过小图元过拟合单个训练视角
- **多边形裁剪**: 将径向基端点投影为多边形，精确判断 tile 是否与核相交，比传统 AABB 方法更高效
- **缓存排序**: 用射线-平面交点距离 $r_t$ 替代中心深度排序，维护8元素排序数组，解决多核重叠时的排序不一致和 popping 问题

## 实验关键数据

### 主实验（DiverseScene 数据集）

| 方法 | PSNR↑ | LPIPS↓ | SSIM↑ | 图元数↓ |
|------|------|----------|------|------|
| 2D-GS | 33.92 | 0.0881 | 0.9514 | 359K |
| 3D-GS | 34.41 | 0.0861 | 0.9621 | 336K |
| 3D-HGS | 35.68 | 0.0637 | 0.9521 | 373K |
| GES | 35.05 | 0.0804 | 0.9634 | 330K |
| DRK | **37.58** | **0.0564** | **0.9752** | 260K |
| DRK (S2) | 35.03 | 0.0823 | 0.9637 | **42K** |

### 消融实验

| 配置 | PSNR | LPIPS | 说明 |
|------|---------|------|------|
| DRK (S2) | 35.03 | 0.0823 | 极稀疏，仅42K图元 |
| DRK (S1) | 36.62 | 0.0668 | 中等密度，109K图元 |
| DRK (Full) | 37.58 | 0.0564 | 完整模型，260K图元 |

### 关键发现

- DRK 在所有渲染质量指标上全面超越 3DGS、2DGS、3D-HGS、GES
- 极稀疏版 DRK(S2) 仅用 **42K** 图元（3DGS 的 1/8）即可达到与 GES 可比的质量
- 模型大小从 79.7MB (3DGS) 可降至 12.3MB (DRK-S2)
- 在 Mip-NeRF360 无界场景上，DRK 在感知质量 (LPIPS, SSIM) 上优势显著
- 单个 DRK 图元可灵活建模矩形、三角形、椭圆等多种形状，而这需要数百个高斯

## 亮点与洞察

- **从特殊到一般的泛化思路**: 证明 2D 高斯是 DRK 的一个特例（$K=4$, 对称角度），使新方法自然兼容已有工作
- **$L_1$/$L_2$ 混合的优雅设计**: 用一个标量 $\eta$ 连续控制从曲线到直线边界的过渡，同时保持可微性
- **DiverseScene 数据集贡献**: 新建的、涵盖纹理/几何/高光/大场景的评测集，填补了场景多样性评测的空白
- **效率与质量的帕累托前沿**: 不同稀疏度的 DRK 变体构成了一条优于现有方法的帕累托前沿

## 局限与展望

- 完整 DRK 的渲染速度 (77.5 FPS) 低于 3DGS (247 FPS)，核运算更复杂
- 在无界场景远处区域可能过拟合，监督信号不足
- 径向基数量 $K$ 的选择是超参数，可考虑自适应调整
- 未探索 DRK 在动态场景、生成任务等下游应用中的潜力

## 相关工作与启发

- GES 通过调整指数值控制锐度但保持旋转对称；DisC-GS 和 3D-HGS 用切割技术处理不连续，但仍被高斯平滑性约束——DRK 从根本上解决了形状限制
- 启发：3D 表示中，核函数的设计空间远比高斯大，可学习的形状参数化是一个有前景的方向
- 与并行工作 3D-CS（使用光滑凸形状）有关联，但 DRK 更灵活

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 根本性地泛化了高斯核，数学推导优雅，向后兼容性证明巧妙
- 实验充分度: ⭐⭐⭐⭐ 自建数据集+Mip-NeRF360，不同稀疏度变体分析到位，但消融仍可更深入
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，图示出色，特别是形状对比可视化非常直观
- 价值: ⭐⭐⭐⭐⭐ 开辟了超越高斯核的新方向，对3D表示学习有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PRaDA: Projective Radial Distortion Averaging](prada_projective_radial_distortion_averaging.md)
- [\[CVPR 2025\] RDD: Robust Feature Detector and Descriptor Using Deformable Transformer](rdd_robust_feature_detector_and_descriptor_using_deformable_transformer.md)
- [\[ECCV 2024\] Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](../../ECCV2024/3d_vision/per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DepthSplat: Connecting Gaussian Splatting and Depth](depthsplat_connecting_gaussian_splatting_and_depth.md)
- [\[CVPR 2026\] Hermite Radial Basis Function for Surface Reconstruction via Differentiable Rendering](../../CVPR2026/3d_vision/hermite_radial_basis_function_for_surface_reconstruction_via_differentiable_rend.md)

</div>

<!-- RELATED:END -->
