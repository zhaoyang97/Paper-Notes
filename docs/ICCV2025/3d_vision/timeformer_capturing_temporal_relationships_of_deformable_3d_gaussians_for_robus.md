---
title: >-
  [论文解读] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction
description: >-
  [ICCV 2025][3D视觉][动态场景重建] 提出TimeFormer模块,通过跨时间Transformer编码器隐式学习可变形3D Gaussian的时序关系,并设计双流优化策略在训练时迁移运动知识,推理时无额外开销。 动态场景重建中,现有方法通过变形场将3DGS扩展到动态场景,但存在关键问题:时序关系建模不足：…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "动态场景重建"
  - "3D Gaussian Splatting"
  - "Transformer"
  - "时序关系建模"
  - "即插即用"
---

# TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2411.11941](https://arxiv.org/abs/2411.11941)  
**代码**: [项目页面](https://patrickddj.github.io/TimeFormer/)  
**领域**: 3D视觉  
**关键词**: 动态场景重建, 3D Gaussian Splatting, Transformer, 时序关系建模, 即插即用

## 一句话总结

提出TimeFormer模块,通过跨时间Transformer编码器隐式学习可变形3D Gaussian的时序关系,并设计双流优化策略在训练时迁移运动知识,推理时无额外开销。

## 研究背景与动机

动态场景重建中,现有方法通过变形场将3DGS扩展到动态场景,但存在关键问题:**时序关系建模不足**。

现有变形场的设计(MLP、时空平面、多项式、傅里叶级数等)从各个独立时间戳学习运动模式,忽略了跨时间的内在关系:
- 一些方法引入运动流正则化,但仅关注**相邻帧**的局部时序关系
- 对于**剧烈运动、极端几何形状、反射表面**等复杂场景,局部视角难以有效建模

核心问题在于:变形场将每个时间戳视为独立输入,缺乏对整个时间序列的全局视角。这导致在面对复杂运动模式时重建质量显著下降。

## 方法详解

### 整体框架

TimeFormer是一个**即插即用模块**,无需修改基础变形3DGS方法的架构。包含:
1. **跨时间Transformer编码器** — 建模多时间戳间的隐式运动模式
2. **双流优化策略** — 共享权重实现运动知识迁移,推理时可移除TimeFormer

### 跨时间Transformer编码器

将不同时间戳视为BatchFormer中的"特殊batch":

设 $\mathcal{T}_s = \{t_i\}_{i=0}^{B-1}$ 为随机采样的时间戳,$\mathcal{G} \in \mathbb{R}^{N \times (3+C)}$ 为规范空间中的Gaussian。

将位置复制B份得到 $\mathcal{G}_c \in \mathbb{R}^{B \times N \times 3}$,与时间戳拼接后经位置编码:

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

经 $M$ 层Transformer编码后,用小型MLP投影为位置偏移:

$$\mathcal{O} = MLP(F_{M-1}), \quad \mathcal{G}_t = \mathcal{G}_c + \mathcal{O}$$

### 梯度分析

无TimeFormer时: $\frac{\partial \Delta\mu_i}{\partial \mu} = \frac{\partial \mathcal{D}(\mu, t_i)}{\partial \mu}$

加入TimeFormer后: $\frac{\partial \Delta\mu_i}{\partial \mu} = \frac{\partial \mathcal{D}(a, t_i)}{\partial a} \cdot (1 + \frac{\partial \mathcal{P}(\mu, \mathcal{T}_s)}{\partial \mu})$

额外的梯度项 $\frac{\partial \mathcal{P}(\mu, \mathcal{T}_s)}{\partial \mu}$ 使得当前状态可被任何过去或未来状态影响,从全局视角捕获运动模式。

### 双流优化策略

原始分支和TimeFormer分支共享变形场权重:

$$\mathcal{L}_c = \|Splatting(\mathcal{D}(\mathcal{G}_c, \mathcal{T}), \mathcal{V}) - \mathcal{I}_{gt}\|_1$$
$$\mathcal{L}_t = \|Splatting(\mathcal{D}(\mathcal{G}_t, \mathcal{T}), \mathcal{V}) - \mathcal{I}_{gt}\|_1$$
$$\mathcal{L} = \lambda_c \mathcal{L}_c + \lambda_t \mathcal{L}_t$$

其中 $\lambda_c > \lambda_t$,防止TimeFormer分支过拟合导致推理质量下降。推理时移除TimeFormer分支,保持原始渲染速度。

## 实验

### 多视角动态场景 (Neural 3D Video数据集)

| 方法 | Sear Steak PSNR | Flame Salmon PSNR | Coffee Martini PSNR | Mean PSNR | Mean SSIM |
|------|----------------|-------------------|---------------------|-----------|-----------|
| K-Plane | 32.52 | 30.44 | 29.99 | 31.63 | 0.960 |
| GS4D | 32.92 | 26.39 | 25.23 | 30.07 | 0.936 |
| 4D-Rotor | 32.86 | 28.25 | 27.95 | 31.06 | 0.938 |

### 即插即用效果验证

TimeFormer可应用于多种不同的变形3DGS骨干网络,均带来一致的质量提升和渲染速度提升。

### 关键发现

1. **TimeFormer引导更高效的规范空间分布** — 隐式跨时间关系学习促使具有相似变化的Gaussian在优化过程中自动聚合,加速渲染
2. **推理时FPS反而提升** — 虽然TimeFormer仅在训练时使用,但更好的规范空间分布使推理速度更快
3. **对复杂运动的提升最显著** — 在咖啡马提尼等包含反射面和流体的场景中改进幅度最大

## 亮点与洞察

1. **即插即用设计** — 无需修改基础方法架构,可无缝集成到现有变形3DGS方法中
2. **零推理开销** — 通过权重共享的双流策略,训练时学到的运动知识在推理时无额外计算成本
3. **从学习视角建模运动** — 区别于显式运动流/光流监督,TimeFormer从RGB监督中自动提取运动模式
4. **梯度流的理论分析** — 清晰解释了TimeFormer如何通过额外梯度项实现跨时间信息传播

## 局限性

- 随机采样时间戳可能不总是覆盖最关键的运动变化
- Transformer的自注意力在大量时间戳上的计算开销较高
- 需要额外的训练时间用于双流优化

## 相关工作

- **变形场设计**: D-NeRF, Deformable-3DGS (MLP/K-Plane/多项式)
- **运动建模**: 光流监督 (MD-Splatting, D3DG), 基于邻近帧 (DN-4DGS)
- **4D表示**: GS4D, 4D-Rotor

## 评分

- 新颖性: ⭐⭐⭐⭐ (跨时间Transformer + 零推理开销的设计思路新颖)
- 技术深度: ⭐⭐⭐⭐ (梯度分析清晰,双流策略设计合理)
- 实验充分度: ⭐⭐⭐⭐ (多数据集多骨干验证)
- 实用价值: ⭐⭐⭐⭐⭐ (即插即用,实际部署友好)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[CVPR 2025\] RDD: Robust Feature Detector and Descriptor Using Deformable Transformer](../../CVPR2025/3d_vision/rdd_robust_feature_detector_and_descriptor_using_deformable_transformer.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions](stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
