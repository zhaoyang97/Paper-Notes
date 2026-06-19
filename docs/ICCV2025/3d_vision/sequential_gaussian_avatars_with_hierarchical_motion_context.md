---
title: >-
  [论文解读] Sequential Gaussian Avatars with Hierarchical Motion Context
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 提出 SeqAvatar，利用显式3DGS表示结合层次化运动上下文（粗粒度骨骼运动 + 细粒度逐点速度）建模人体化身的运动相关外观变化，并通过时空多尺度采样增强运动条件的鲁棒性，在多个数据集上取得SOTA渲染质量同时保持实时渲染速度。 基于3DGS的可动画人体化身重建近年来…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "人体化身"
  - "非刚性形变"
  - "运动上下文"
  - "SMPL"
---

# Sequential Gaussian Avatars with Hierarchical Motion Context

**会议**: ICCV 2025  
**arXiv**: [2411.16768](https://arxiv.org/abs/2411.16768)  
**代码**: [Project Page](https://zezeaaa.github.io/projects/SeqAvatar/)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 人体化身, 非刚性形变, 运动上下文, SMPL

## 一句话总结

提出 SeqAvatar，利用显式3DGS表示结合层次化运动上下文（粗粒度骨骼运动 + 细粒度逐点速度）建模人体化身的运动相关外观变化，并通过时空多尺度采样增强运动条件的鲁棒性，在多个数据集上取得SOTA渲染质量同时保持实时渲染速度。

## 研究背景与动机

基于3DGS的可动画人体化身重建近年来取得了显著进展，但仍面临一个核心问题：**非刚性形变建模不充分**。具体表现为：

**姿态到外观的一对多映射**：同一人体姿态在不同运动状态下可能对应不同的外观（如裙摆的惯性摆动），现有方法仅使用当前帧的空间姿态信息，无法区分这种歧义。

**局部细节缺失**：当前方法主要依赖全局骨骼信息预测形变，对远离骨骼的区域（如飘逸衣物、头发）无法建模精细运动。

**已有序列建模的局限**：Dyco等基于NeRF的方法虽然尝试使用人体姿态残差建模运动序列，但姿态序列的全局性特征限制了其捕捉更细粒度运动细节的能力，且无法充分利用3DGS的显式点特性。

核心洞察：**3DGS的显式点表示使得逐点运动建模成为可能**——可以为每个高斯基元计算独立的速度向量，从而捕捉骨骼运动之外的局部细节变化。

## 方法详解

### 整体框架

SeqAvatar 在标准SMPL+LBS+3DGS管线基础上引入层次化运动上下文条件。流程：(1) 从SMPL模板顶点初始化标准空间高斯 → (2) 构建粗粒度骨骼运动条件 $f_{\Delta\mathcal{P}}$ 和细粒度逐点速度条件 $f_\mathcal{V}$ → (3) MLP预测非刚性形变 → (4) LBS刚性变换到观测空间 → (5) 高斯溅射渲染。

### 关键设计

1. **粗粒度骨骼运动条件 (Coarse Skeleton Motion)**：

    - 对于目标帧 $t$，采样等间隔历史帧序列 $\mathcal{T} = \{t-s, t-2s, ..., t-Ls\}$
    - 计算相邻帧间的姿态差异（轴角形式）：$\Delta\mathcal{P} = \{\Delta P^t = \delta(P^t, P^{t-s}) | t \in \mathcal{T}\}$，其中 $P \in \mathbb{R}^{K \times 3}$ 为身体姿态
    - 通过MLP编码为固定维度嵌入：$f_{\Delta\mathcal{P}} = \mathcal{E}_{\Delta\mathcal{P}}(\Delta\mathcal{P}) \in \mathbb{R}^{32}$
    - **设计动机**：相比直接使用当前帧姿态，姿态差异序列能捕捉运动的时间动态，从而区分相同姿态下的不同外观状态

2. **细粒度逐点速度条件 (Fine Vertex Motion)**：

    - 每个高斯基元的逐点速度无法直接计算（高斯位置在优化中不断变化，且非刚性变换存在循环依赖）
    - 解决方案：构建**运动模板场** $\mathcal{F}_\mathbf{V} = \{\mathbf{V}_i\}_{i=1}^{N}$，存储每个SMPL模板顶点的速度
    - SMPL顶点速度计算：先将模板顶点 $\mathbf{T}$ 通过标准LBS变换到观测空间 $\mathbf{T_o}^t = \mathbf{LBS}(\mathbf{T}, \mathbf{B}^t, \mathbf{W})$，然后计算 $\mathbf{V}^t = (\mathbf{T_o}^t - \mathbf{T_o}^{t-s}) / s$
    - 每个高斯基元的速度从运动模板场中通过KNN采样获取
    - **关键优势**：利用3DGS的显式点表示，为每个点提供独立的局部运动信息，捕捉骨骼运动无法覆盖的区域（如飘动的裙摆）

3. **时空多尺度采样 (Spatio-Temporal Multi-Scale Sampling, STMS)**：

   **空间维度**：对每个高斯基元 $\mathcal{G}_i$，采样 $\tau$ 个最近邻模板顶点的速度作为输入，学习局部区域的运动嵌入：
    $e_i^t = \mathcal{E}_{knn}(\{\mathbf{V}_j^t\}), \quad j \in \mathbf{KNN}(\mathbf{T}, \mathbf{x}_i)$

   **时间维度**：使用递增间隔的多尺度序列采样，同时捕获总体运动趋势和帧间细节：
    $\mathcal{S} = \{s = s_0 + i\Delta s\}_{i=0}^{i=m}$

   将多尺度采样的骨骼和逐点运动条件分别拼接后输入编码器：
    $f_{\Delta\mathcal{P}} = \mathcal{E}_{\Delta\mathcal{P}}(\{\Delta\mathcal{P}_s\}), \quad f_\mathcal{V} = \mathcal{E}_\mathcal{V}(\{\mathcal{V}_s\}), \quad s \in \mathcal{S}$

   **设计动机**：小间隔捕获细粒度帧间变化，大间隔捕获整体运动趋势，两者互补提升泛化能力

4. **非刚性形变预测**：结合所有运动条件，使用MLP预测每个高斯的位置、缩放和旋转偏移：
    $\delta\mathbf{x}, \delta\mathbf{s}, \delta\mathbf{r} = \mathcal{E}_{non-rigid}(\mathbf{x}, P, f_{\Delta\mathcal{P}}, f_\mathcal{V})$

   然后更新标准空间高斯：$\mathbf{x'} = \mathbf{x} + \delta\mathbf{x}$，$\mathbf{s'} = \mathbf{s} + \delta\mathbf{s}$，$\mathbf{r'} = \mathbf{r} \cdot \delta\mathbf{r}$

### 损失函数 / 训练策略

综合损失函数：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{color} + \lambda_2 \mathcal{L}_{ssim} + \lambda_3 \mathcal{L}_{lpips} + \mathcal{L}_{mask}$$

- $\mathcal{L}_{color}$：L1颜色损失
- $\mathcal{L}_{ssim}$：SSIM结构相似性损失
- $\mathcal{L}_{lpips}$：LPIPS感知损失
- $\mathcal{L}_{mask}$：渲染alpha与人体mask的L2损失

附加正则化约束：$\mathcal{L}_{isopos}$ 和 $\mathcal{L}_{isocov}$ 控制高斯基元的位置和协方差。同时使用姿态精修MLP $\mathcal{E}_{pose}$ 优化SMPL姿态估计。

LBS权重通过学习偏移更新：$\omega_k(\mathbf{x}) = \omega_k^{SMPL}(\mathbf{x}) + \mathcal{E}_{lbs}(\mathbf{x})$

## 实验关键数据

### 主实验

DNA-Rendering数据集（6个场景平均）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS*↓ |
|------|-------|-------|---------|
| 3DGS-Avatar | 28.63 | 0.9565 | 41.43 |
| GART | 28.99 | 0.9597 | 44.55 |
| GauHuman | 29.55 | 0.9600 | 40.96 |
| **SeqAvatar** | **32.05** | **0.9711** | **30.91** |

I3D-Human数据集（Novel View，4个场景平均）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS*↓ | FPS |
|------|-------|-------|---------|-----|
| 3DGS-Avatar | 30.86 | 0.9608 | 34.07 | 实时 |
| Dyco (NeRF) | 31.06 | 0.9607 | 30.71 | ~0.7 |
| GauHuman | 30.13 | 0.9562 | 45.37 | 实时 |
| **SeqAvatar** | **32.24** | **0.9664** | **29.78** | ~45 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS*↓ | 说明 |
|------|-------|-------|---------|------|
| (a) Baseline（无非刚性形变） | 29.76 | 0.9569 | 38.35 | 仅LBS |
| (b) +普通非刚性MLP+当前姿态 | 31.05 | 0.9617 | 34.35 | +位姿条件 |
| (c) +$\Delta\mathcal{P}$骨骼运动 | 31.89 | 0.9645 | 32.17 | +粗粒度时序 |
| (d) +$\mathcal{V}$逐点速度 | 32.01 | 0.9651 | 31.23 | +细粒度运动 |
| **(e) +STMS (完整)** | **32.24** | **0.9664** | **29.78** | 完整方法 |

### 关键发现

- 骨骼运动条件 $\Delta\mathcal{P}$ 带来最大的性能跳跃（31.05→31.89 PSNR），说明时序运动信息对非刚性形变至关重要
- 逐点速度条件 $\mathcal{V}$ 进一步提升局部区域细节（31.89→32.01），尤其在远离骨骼的飘逸衣物区域
- STMS多尺度采样提供额外0.23dB PSNR提升，增强了泛化能力
- SeqAvatar保持实时渲染（~45 FPS on I3D-Human），比NeRF-based Dyco（~0.7 FPS）快约60倍
- 在DNA-Rendering上比最佳3DGS方法GauHuman高出约2.5dB PSNR
- 分布外姿态动画（用一个序列训练，渲染另一序列的未见姿态）也表现良好

## 亮点与洞察

- **充分利用3DGS显式表示的优势**：逐点速度是3DGS特有的能力，NeRF因隐式表示无法自然实现
- **运动模板场设计精妙**：通过SMPL模板顶点速度间接提供高斯基元速度，避免了循环依赖和优化不稳定
- **多尺度时间采样**：类似于卷积中的多尺度感受野概念，在时间维度捕获不同频率的运动信息
- **性能-速度权衡优秀**：在渲染质量超越Dyco（NeRF）的同时保持60倍速度优势

## 局限与展望

- 高斯表示可能在渲染中引入轻微模糊，NeRF的光线积分相比更锐利
- 局部速度线索来自粗糙的SMPL模型而非密集表面追踪，可能限制精细服装形变的精度
- 依赖SMPL初始化和姿态估计的准确性
- 未在单目视频输入设置下验证（实验均使用多视角输入）
- 长序列训练的效率和内存消耗值得进一步优化

## 相关工作与启发

- 与Dyco（NeRF+姿态序列）的核心区别：SeqAvatar除了姿态残差外还利用了逐点速度，且基于3DGS实现实时渲染
- 与3DGS-Avatar/GauHuman等3DGS方法不同，SeqAvatar引入了时序运动信息
- 运动模板场的思路可推广到其他需要逐点运动建模的3DGS动态场景重建任务
- 多尺度时间采样策略可作为通用模块用于其他以运动序列为条件的生成模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 层次化运动上下文设计新颖，运动模板场巧妙解决了循环依赖
- 实验充分度: ⭐⭐⭐⭐⭐ DNA-Rendering/I3D-Human/ZJU-MoCap三个数据集，Novel View/Novel Pose全面评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观，公式推导完整
- 价值: ⭐⭐⭐⭐ 为3DGS人体化身建模提供了有效的运动条件增强方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/3d_vision/motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] Hierarchical Material Recognition from Local Appearance](hierarchical_material_recognition_from_local_appearance.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](../../CVPR2025/3d_vision/sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)

</div>

<!-- RELATED:END -->
