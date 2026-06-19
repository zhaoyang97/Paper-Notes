---
title: >-
  [论文解读] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出SPFSplat,首个在训练和推理时均不需要真值位姿的自监督3DGS框架,通过共享ViT骨干同时预测Gaussian基元和相机位姿,在极端视角变化下超越需要位姿的SOTA方法。 现有稀疏视角NVS方法按位姿依赖分为三类: 需要位姿： pix…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "无位姿训练"
  - "自监督"
  - "稀疏视角"
  - "位姿估计"
---

# No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views

**会议**: ICCV 2025  
**arXiv**: [2508.01171](https://arxiv.org/abs/2508.01171)  
**代码**: [项目页面](https://ranrhuang.github.io/spfsplat/)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 无位姿训练, 自监督, 稀疏视角, 位姿估计

## 一句话总结

提出SPFSplat,首个在训练和推理时均不需要真值位姿的自监督3DGS框架,通过共享ViT骨干同时预测Gaussian基元和相机位姿,在极端视角变化下超越需要位姿的SOTA方法。

## 研究背景与动机

现有稀疏视角NVS方法按位姿依赖分为三类:

**需要位姿**: pixelSplat, MVSplat — 依赖SfM,稀疏视角下不可靠

**监督无位姿**: NoPoSplat, Splatt3R — 推理无位姿但训练需要真值位姿,限制了数据可扩展性

**自监督无位姿**: PF3plat, SelfSplat — 使用独立模块做位姿和重建,导致特征不一致和反馈循环不稳定

核心挑战:**渲染损失内在地耦合了3D几何和相机位姿的学习**。位姿误差降低重建质量,重建质量差又进一步影响位姿估计,形成不稳定的反馈循环。

现有自监督方法使用**独立模块**做位姿和场景重建,在不同的特征空间工作,无法共享几何知识,导致训练不稳定。

## 方法详解

### 共享骨干架构

基于ViT的编码器-解码器:
- **编码器**: 共享权重,独立处理每个视角
- **解码器**: 交叉注意力聚合多视角信息
- **Gaussian预测头**: 两个DPT头预测中心位置和其他属性
- **位姿头**: 基于同一解码器,轻量3层MLP输出10维位姿表示

位姿表示: 4维齐次平移 + 6维旋转(两个未归一化坐标轴,通过叉积构造完整旋转矩阵)。

### 训练时的双分支设计

- **Context-only分支**: 仅从上下文视角预测Gaussian(推理时使用)
- **Context-with-target分支**: 包含上下文和目标视角,估计目标位姿(仅训练时使用)

关键:Gaussian重建和目标位姿预测**解耦**,防止目标视角信息泄露到3D表示中。

### 重投影损失

$$\mathcal{L}_{reproj} = \sum_{v=1}^N \sum_{j=1}^{H \times W} \|\mathbf{p}_j^v - \pi(\boldsymbol{K}^v, \boldsymbol{P}^{v \to 1}, \boldsymbol{\mu}_j^{v \to 1})\|$$

对两个分支的位姿都应用重投影损失,强制Gaussian中心与像素对齐。

**为什么不用渲染损失替代?** 将上下文视角也纳入渲染损失会导致训练崩溃——模型优先优化第一视角(其渲染不依赖位姿),抑制其他视角的Gaussian。

### 总损失

$$\mathcal{L} = \mathcal{L}_{render} + \mathcal{L}_{reproj}$$

其中 $\mathcal{L}_{render} = \|I^t - \hat{I^t}\|_2 + \gamma\text{LPIPS}(I^t, \hat{I^t})$

## 实验

### RealEstate10K新视角合成

| 方法 | 类别 | Small PSNR | Medium PSNR | Large PSNR | Avg PSNR |
|------|------|----------|-----------|----------|---------|
| pixelSplat | 需位姿 | 20.28 | 23.73 | 27.15 | 23.86 |
| MVSplat | 需位姿 | 20.37 | 23.81 | 27.47 | 24.01 |
| NoPoSplat | 监督无位姿 | 22.51 | 24.90 | 27.41 | 25.03 |
| SelfSplat | 自监督 | 14.83 | 18.86 | 23.34 | 19.15 |
| PF3plat | 自监督 | 18.36 | 20.95 | 23.49 | 21.04 |
| **SPFSplat** | **自监督** | **22.90** | **25.10** | **27.65** | **25.27** |

### 位姿估计性能

SPFSplat无需位姿监督即超越了依赖几何先验的SOTA方法。

### 关键发现

1. **首次自监督方法超越需要位姿的方法** — SPFSplat的Avg PSNR(25.27)超过NoPoSplat(25.03)和MVSplat(24.01)
2. 在**大视角变化**(Small类别)下优势尤为显著
3. 推理速度与NoPoSplat相当(0.042s),远快于PF3plat(1.171s)
4. 重投影损失是训练稳定性的关键

## 亮点与洞察

1. **真正的无位姿** — 训练和推理均无需任何真值位姿,极大扩展了可训练数据范围
2. **共享骨干的互增强** — 位姿估计受益于场景几何,Gaussian预测受益于准确对齐,形成正向循环
3. **重投影Loss的关键作用** — 解决了纯渲染Loss的训练崩溃问题
4. **规范空间的位姿无关性** — 在第一视角坐标系预测Gaussian,减少位姿误差对几何的影响

## 局限性

- 需要已知相机内参
- 目前支持稀疏视角(2-3视角)输入
- 大基线下虽然优势明显但绝对质量仍有提升空间

## 相关工作

- **需要位姿**: pixelSplat, MVSplat, GRM
- **监督无位姿**: NoPoSplat, Splatt3R, LEAP, PF-LRM
- **自监督无位姿**: PF3plat, SelfSplat, Nope-NeRF
- **SfM**: DUSt3R, MASt3R, VGGSfM

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (首个超越有位姿方法的无位姿自监督方法)
- 技术深度: ⭐⭐⭐⭐⭐ (反馈循环分析+重投影损失设计精妙)
- 实验充分度: ⭐⭐⭐⭐⭐ (与三类方法全面对比)
- 实用价值: ⭐⭐⭐⭐⭐ (真正无位姿,可扩展到无标注数据)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)
- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](../../CVPR2025/3d_vision/selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[ICCV 2025\] S3E: Self-Supervised State Estimation for Radar-Inertial System](s3e_self-supervised_state_estimation_for_radar-inertial_system.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)

</div>

<!-- RELATED:END -->
