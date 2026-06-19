---
title: >-
  [论文解读] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images
description: >-
  [ICCV 2025][3D视觉][语义3DGS] 提出SpatialSplat,通过双场语义表示和选择性Gaussian机制,从稀疏无位姿图像前馈生成紧凑的语义3D Gaussian,将表示参数量减少60%同时超越SOTA方法。 语义感知3D重建从2D图像获取语义3D结构,是机器人、自动驾驶和VR/AR的基础技术…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "语义3DGS"
  - "前馈重建"
  - "无位姿"
  - "双场架构"
  - "Gaussian选择"
---

# SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images

**会议**: ICCV 2025  
**arXiv**: [2505.23044](https://arxiv.org/abs/2505.23044)  
**代码**: [GitHub](https://github.com/shengyu724/SpatialSplat)  
**领域**: 3D视觉  
**关键词**: 语义3DGS, 前馈重建, 无位姿, 双场架构, Gaussian选择

## 一句话总结

提出SpatialSplat,通过双场语义表示和选择性Gaussian机制,从稀疏无位姿图像前馈生成紧凑的语义3D Gaussian,将表示参数量减少60%同时超越SOTA方法。

## 研究背景与动机

语义感知3D重建从2D图像获取语义3D结构,是机器人、自动驾驶和VR/AR的基础技术。现有前馈3DGS方法在引入语义时面临两个核心问题:

**逐像素Gaussian预测的冗余** — 重叠区域产生大量冗余基元,带来不必要的内存开销

**高维语义特征的压缩损失** — 512维+语言特征必须压缩到64-128维才能附加到每个基元,导致**不可逆信息损失**

现有方法(如LSM)简单地将压缩特征附加到每个像素级Gaussian,既不高效也不准确。

### 关键观察

1. 冗余基元共享相似的几何和外观,可直接从图像特征识别(无需几何先验)

**逐基元语义并非必要** — 同一实例内的Gaussian具有高度语义一致性,粗粒度语义+细粒度实例信息就足够

## 方法详解

### 双场语义表示

将密集语义特征场分解为两个组件:

**细粒度实例感知辐射场** $\mathcal{F}_I$:
- 每个Gaussian附带低维实例特征 $\boldsymbol{f}_I \in \mathbb{R}^N$ 和重要性分数 $\boldsymbol{\beta}$
- 由2D基础模型(SAM等)引导学习

**粗粒度语义特征场** $\mathcal{F}_S$:
- 以 $S$ 倍降采样的分辨率预测,基元数量大幅减少
- 保留**未压缩**的语义特征 $\boldsymbol{f}_S \in \mathbb{R}^M$
- 少量基元即可编码完整语义(因同实例内语义一致)

### 选择性Gaussian机制 (SGM)

为每个基元预测重要性分数 $\beta_i$,乘以不透明度修改alpha blending:

$$\boldsymbol{c} = \sum_{i=1}^n \boldsymbol{c}_i \boldsymbol{\alpha}_i \boldsymbol{\beta}_i \prod_{j=1}^{i-1}(1 - \boldsymbol{\alpha}_j \boldsymbol{\beta}_j)$$

使用类Leaky ReLU的阈值处理:
$$\beta_i = \begin{cases} \beta_i & \text{if } \beta_i > \tau \\ \beta_i \times 10^{-3} & \text{if } \beta_i < \tau \end{cases}$$

BCE损失+L1正则推动 $\beta_i$ 向0或1二值化:
$$\mathcal{L}_I = \mathcal{L}_{BCE}(\boldsymbol{S}, \hat{\boldsymbol{S}}) + \frac{1}{\|\boldsymbol{S}\|}\sum_{\beta_i \in \boldsymbol{S}} \beta_i$$

### 3D几何预测

纯ViT编码器-解码器,不需要几何先验。通过注入相机内参解决尺度歧义(无需深度监督)。

## 实验

### ScanNet语义3D重建

| 方法 | 前馈 | Source mIoU↑ | Target mIoU↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------------|-------------|-------|-------|--------|
| L-Seg | ✗ | 0.5541 | 0.5558 | N/A | N/A | N/A |
| NeRF-DFF | ✗ | 0.5381 | 0.5137 | 22.49 | 0.765 | 0.283 |
| Feature-3DGS | ✗ | 0.4992 | 0.3223 | 17.96 | 0.581 | 0.489 |
| NoPoSplat | ✔ | N/A | N/A | 25.70 | 0.816 | 0.188 |
| LSM | ✔ | 0.5141 | 0.5104 | 24.12 | 0.796 | 0.253 |
| SpatialSplat-Lite | ✔ | 0.5272 | 0.5265 | 25.45 | 0.803 | 0.204 |
| **SpatialSplat** | ✔ | **0.5593** | **0.5587** | **25.46** | **0.805** | **0.205** |

### 参数效率

SpatialSplat仅使用baseline 40%的表示参数,同时在所有指标上超越。

### 关键发现

1. 双场架构以40%的参数达到更优的语义分割和渲染质量
2. 选择性Gaussian机制有效识别并剔除冗余基元,无需几何先验
3. 粗粒度未压缩语义 > 细粒度压缩语义,证明了"不压缩但少量"优于"压缩但所有"的策略
4. 首次同时学习语义和实例先验的前馈3DGS框架

## 亮点与洞察

1. **语义表示的解耦设计** — "粗语义+细实例"的分解思路新颖且高效
2. **不压缩的反直觉选择** — 证明保留完整语义特征用少量基元比压缩后广撒网更有效
3. **从图像识别冗余** — 绕过了需要精确相机外参才能检测重叠的限制
4. **无3D监督** — 完全从2D基础模型引导学习

## 局限性

- 粗粒度语义场的降采样率S需预设
- 对实例边界的分割精度受2D基础模型质量制约
- 重要性分数的阈值τ需调参

## 相关工作

- **前馈3DGS**: pixelSplat, MVSplat, NoPoSplat
- **特征场蒸馏**: LERF, LangSplat, Feature-3DGS, LSM
- **紧凑3DGS**: Scaffold-GS, HAC, LightGaussian

## 评分

- 新颖性: ⭐⭐⭐⭐ (双场架构+选择性Gaussian)
- 技术深度: ⭐⭐⭐⭐ (SGM设计+损失函数完整)
- 实验充分度: ⭐⭐⭐⭐ (与多类方法全面对比)
- 实用价值: ⭐⭐⭐⭐⭐ (60%参数减少,实际部署价值高)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sparfels: Fast Reconstruction from Sparse Unposed Imagery](sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)
- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[CVPR 2026\] Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images](../../CVPR2026/3d_vision/uni3r_unified_3d_reconstruction_and_semantic_understanding_via_generalizable_gau.md)
- [\[CVPR 2025\] ERUPT: Efficient Rendering with Unposed Patch Transformer](../../CVPR2025/3d_vision/erupt_efficient_rendering_with_unposed_patch_transformer.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)

</div>

<!-- RELATED:END -->
