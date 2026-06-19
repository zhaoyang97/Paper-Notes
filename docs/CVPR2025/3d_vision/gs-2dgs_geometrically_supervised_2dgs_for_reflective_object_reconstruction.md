---
title: >-
  [论文解读] GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction
description: >-
  [CVPR 2025][3D视觉][2D Gaussian Splatting] 在 2DGS 基础上引入基础模型（Marigold + Depth Pro）的深度/法线伪标签监督和延迟着色（Deferred Shading）的物理渲染管线，在反射物体重建上显著超越 GS 方法、媲美 SDF 方法且快了一个数量级。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "2D Gaussian Splatting"
  - "reflective object"
  - "PBR"
  - "deferred shading"
  - "foundation model"
  - "normal estimation"
---

# GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction

**会议**: CVPR 2025  
**arXiv**: [2506.13110](https://arxiv.org/abs/2506.13110)  
**代码**: [GitHub](https://github.com/hirotong/GS2DGS)  
**领域**: 3D视觉  
**关键词**: 2D Gaussian Splatting, reflective object, PBR, deferred shading, foundation model, normal estimation

## 一句话总结

在 2DGS 基础上引入基础模型（Marigold + Depth Pro）的深度/法线伪标签监督和延迟着色（Deferred Shading）的物理渲染管线，在反射物体重建上显著超越 GS 方法、媲美 SDF 方法且快了一个数量级。

## 研究背景与动机

**领域现状**: 高反射物体的 3D 建模是长期难题——镜面反射强烈依赖视角，违反了多视图一致性假设。SDF 方法（NeRO、TensoSDF）可获得高质量 mesh 但训练耗时数小时；3DGS 渲染快但表面提取噪声大。

**现有痛点**:
- GS 方法（GShader、GS-IR、R3DG）只解决了渲染/relighting 的部分问题，但几何重建仍然粗糙
- 反射物体外观由表面属性（材质+几何）和环境光照共同决定，是严重的不适定问题
- 单纯结合 PBR 和几何约束不够——反射表面的法线/深度从多视图立体无法可靠估计

**核心矛盾**: 需要同时解决几何估计和材质/光照分解，但反射面使传统多视图方法失效。

**本文切入角度**: 利用基础模型（Foundation Model）从单视图预测法线和深度——这些模型基于海量数据训练，不依赖多视图一致性，因此对反射表面不敏感。

## 方法详解

### 整体框架

两阶段训练流程：
1. **Stage 1（30K iter）**: 在原始 2DGS 基础上加入基础模型的法线损失 $\mathcal{L}_n$ 和深度损失 $\mathcal{L}_d$，优化几何
2. **Stage 2（10K iter）**: 启用 PBR 管线，为每个 2D Gaussian 分配 albedo、metallic、roughness 参数，联合优化几何+材质+环境光照

### 关键设计

**1. 基础模型几何监督**
- **功能**: 使用 Marigold（法线估计）和 Depth Pro（深度估计）对每张输入图像预测伪 GT 法线 $\tilde{N}$ 和深度 $\tilde{D}$
- **法线损失**: $\mathcal{L}_n = \|\hat{N} - \tilde{N}\|_1 + (1 - \hat{N}^T \tilde{N})$（L1 + cosine 组合）
- **深度损失**: Scale-invariant depth loss，通过最小二乘求解 scale $\omega$ 和 shift $b$ 对齐渲染深度与预测深度
- **设计动机**: 基础模型从单视图推理几何，基于海量训练数据的先验知识，不受反射表面影响

**2. 延迟着色（Deferred Shading）**
- **功能**: 将渲染分为几何 pass（渲染 G-buffer：深度、法线、PBR 参数）和着色 pass（基于 G-buffer 做 PBR 着色）
- **核心思路**: 
    - 前向着色（Forward Shading）对每个 Gaussian 独立计算辐射并 alpha-blending → 射线上多个 Gaussian 法线不同导致着色不准
    - 延迟着色只在最终合成的表面点着色一次 → 法线和位置更准确
- **设计动机**: 对反射物体，精确的着色点法线方向至关重要；延迟着色也减少了计算量（只着色一次而非逐 Gaussian）

**3. PBR + 环境光照解耦**
- **功能**: 基于 Cook-Torrance BRDF 将渲染方程分解为 diffuse + specular 项
- **核心思路**: 
    - 每个 2D Gaussian 学习 albedo $\mathbf{a}$、metallic $m$、roughness $\rho$ 三个 PBR 参数
    - 可训练 HDR cube map 表示环境光照
    - Specular 项使用 split-sum 近似
- **设计动机**: PBR 分解使得材质和光照可分离，支持 relighting

### 损失函数

$$\mathcal{L} = \mathcal{L}_{GS} + \lambda_n \mathcal{L}_n + \lambda_d \mathcal{L}_d + \lambda_{light} \mathcal{L}_{light} + \lambda_{pbr} \mathcal{L}_{pbr}$$

- $\mathcal{L}_{GS}$: 2DGS 原始损失（RGB 重建 + 法线一致性）
- $\mathcal{L}_{light}$: 自然光正则化 $\|\mathbf{L} - \bar{\mathbf{L}}\|^2$（三通道均值约束）
- $\mathcal{L}_{pbr}$: PBR 参数平滑正则 $\|\nabla \mathbf{X}\| \exp(-\|\nabla \mathbf{C}_{gt}\|)$
- 权重: $\lambda_n=0.5$, $\lambda_d=0.05$, $\lambda_{light}=0.002$

## 实验关键数据

### 主实验（Glossy Blender 重建质量 Chamfer-L1↓）

| 方法 | 类型 | 平均 Chamfer-L1 | 训练时间 |
|---|---|---|---|
| NeRO | SDF | 0.0042 | 12h |
| TensoSDF | SDF | 0.0106 | 6h |
| GShader | GS | 0.0169 | 0.5h |
| R3DG | GS | 0.0303 | 1h |
| GS-IR | GS | 0.0553 | 0.5h |
| **GS-2DGS (Ours)** | **GS** | **0.0068** | **0.7h** |

GS 方法中最优，且接近 SDF SOTA（NeRO 0.0042），训练速度快 17 倍。

### Relighting 质量（Glossy Blender PSNR↑/SSIM↑）

| 方法 | PSNR | SSIM | FPS |
|---|---|---|---|
| GShader | 14.96 | 0.811 | 50 |
| GS-IR | 17.11 | 0.811 | 214 |
| R3DG | 19.19 | 0.837 | 1.5 |
| **Ours** | **19.56** | **0.856** | **160** |

### 消融实验

| 配置 | Chamfer-L1↓ | PSNR↑ |
|---|---|---|
| 2DGS baseline | 0.0481 | 26.23 |
| + Geometric supervision | 0.0084 | 25.52 |
| + PBR | 0.0074 | 25.86 |
| + Deferred Shading (Full) | **0.0068** | **26.76** |

### 关键发现

1. **几何监督贡献最大**: Chamfer-L1 从 0.0481 降至 0.0084（减少 82%），是性能提升的核心
2. **延迟着色改善环境光估计**: 对比 forward shading，deferred shading 的环境光估计更准确，PSNR 提升 0.9 dB
3. **PBR 兼顾重建与渲染**: 加入 PBR 后 Chamfer-L1 继续下降且 PSNR 回升
4. **实际训练效率**: 总 40K iter（约 42 分钟），远快于 SDF 方法的 6-12 小时

## 亮点与洞察

- 利用基础模型弥补反射面的几何估计盲区，是一个优雅且通用的思路
- 延迟着色首次被引入 2DGS 用于反射物体，理论分析清晰
- 两阶段训练策略平衡了几何精度和材质分解
- 在 GS 方法中首次接近 SDF 方法的重建质量，同时保持实时渲染

## 局限与展望

- 相比 SDF（NeRO）仍有小幅差距，GS 缺乏固有的几何平滑先验
- 基础模型预测的深度/法线仍有误差，可能引入幻觉
- 未处理透明物体或次表面散射
- 仅在物体级数据集上验证，未扩展到场景级
- 依赖两个外部基础模型增加了预处理时间

## 相关工作与启发

- NeRO 和 TensoSDF 奠定了 SDF+PBR 的反射物体重建基线
- R3DG 提出 point-based ray tracing 但速度慢（1.5 FPS）
- Marigold/Depth Pro 等基础模型的涌现为几何监督提供了新信号源
- 启发：foundation model 作为伪 GT 提供者将在更多 3D 任务中发挥作用

## 评分

⭐⭐⭐⭐ — 方法设计合理、效果显著，GS 反射物体重建的重要进步；基础模型+延迟着色的组合有效解决了核心问题，工程价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GS-ASM: 2DGS-Supervised Active Stereo Matching](../../CVPR2026/3d_vision/gs-asm_2dgs-supervised_active_stereo_matching.md)
- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[CVPR 2025\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)
- [\[CVPR 2025\] SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [\[CVPR 2025\] COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting](cob-gs_clear_object_boundaries_in_3dgs_segmentation_based_on_boundary-adaptive_g.md)

</div>

<!-- RELATED:END -->
