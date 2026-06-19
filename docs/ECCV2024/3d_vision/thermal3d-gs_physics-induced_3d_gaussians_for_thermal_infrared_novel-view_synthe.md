---
title: >-
  [论文解读] Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis
description: >-
  [ECCV 2024][3D视觉][热红外成像] 提出Thermal3D-GS，通过神经网络建模大气传输效应和热传导物理过程，并引入温度一致性约束，实现热红外图像的高质量新视角合成，创建了首个大规模热红外新视角合成数据集TI-NSD。 热红外成像具有全天候成像和强穿透能力，在夜间和恶劣天气场景中极具优势…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "热红外成像"
  - "新视角合成"
  - "3D高斯溅射"
  - "物理建模"
  - "大气传输"
---

# Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis

**会议**: ECCV 2024  
**arXiv**: [2409.08042](https://arxiv.org/abs/2409.08042)  
**代码**: [项目页面](https://github.com/Thermal3DGS)  
**领域**: 3D视觉  
**关键词**: 热红外成像, 新视角合成, 3D高斯溅射, 物理建模, 大气传输

## 一句话总结

提出Thermal3D-GS，通过神经网络建模大气传输效应和热传导物理过程，并引入温度一致性约束，实现热红外图像的高质量新视角合成，创建了首个大规模热红外新视角合成数据集TI-NSD。

## 研究背景与动机

热红外成像具有全天候成像和强穿透能力，在夜间和恶劣天气场景中极具优势。然而直接将可见光新视角合成方法应用于热红外图像会产生两个特有问题：(1) **漂浮物(floaters)**——大气传输效应导致同一物体在不同视角的辐射衰减不同，3D-GS学习到错误的补偿高斯；(2) **边缘模糊**——物体间热传导使边界温度梯度变化，多帧平均导致边缘信息丢失。

## 方法详解

### 整体框架

基于3D-GS框架，增加两个物理驱动模块：(1) Atmospheric Transmission Field (ATF)——建模大气衰减；(2) Thermal Conduction Module (TCM)——建模热传导对边缘的影响。另引入温度不连续性损失约束。

### 关键设计

**大气传输场(ATF)**: 基于Bouguer-Lambert-Beer定律$I = I_0 e^{\mu(\lambda)d}$建模辐射衰减。使用MLP网络（深度8，隐层256）以位置编码后的高斯位置$\gamma(x)$和拍摄时间$\gamma(t)$为输入，预测吸收系数$\mu_{abs}$、散射系数$\mu_{sca}$和传播距离$d$：

$$SH = SH_0 \cdot e^{(\mu_{abs} + \mu_{sca})d}$$

将衰减效应与几何解耦，使3D-GS可以独立学习无衰减的几何结构。

**热传导模块(TCM)**: 基于二维温度场热传导方程$\frac{\partial u}{\partial t} = \alpha \Delta u$，其中$\Delta$是2D拉普拉斯算子，$\alpha = k/(c\rho)$是热扩散率。由于$\alpha$在像素间非均匀，使用3层卷积网络融合输入图像和其二阶梯度特征来模拟像素级的$\alpha$，通过残差加法机制修复热传导导致的热量损失。

**温度不连续性损失**: 真实温度场通常平滑连续，角点意味着学习异常。使用Harris角点检测的响应函数作为权重，强调异常区域的L1损失：

$$\mathcal{L}_{dis} = \frac{R}{R_{max}} \max(1 - \frac{i}{iter_t}, 0) \mathcal{L}_1$$

其中$iter_t = 5000$作为衰减阈值。

### 损失函数

$$\mathcal{L}_{total} = \lambda_{dis}\mathcal{L}_{dis} + \lambda\mathcal{L}_{D-SSIM} + (1-\lambda_{dis}-\lambda)\mathcal{L}_1$$

其中$\lambda_{dis} = \lambda = 0.2$。

## 实验关键数据

### TI-NSD数据集定量对比

20个场景平均结果：

| 方法 | 室内PSNR↑ | 室外PSNR↑ | 无人机PSNR↑ | 平均PSNR↑ | 平均SSIM↑ | 平均LPIPS↓ |
|------|-----------|-----------|-------------|-----------|-----------|-----------|
| Plenoxels | 22.13 | 22.15 | 25.56 | 23.28 | 0.805 | 0.390 |
| INGP-Base | 26.99 | 26.00 | 20.86 | 24.62 | 0.811 | 0.332 |
| INGP-Big | 27.46 | 26.45 | 20.82 | 24.91 | 0.812 | 0.323 |
| 3D-GS (30k) | 32.98 | 28.89 | 34.51 | 32.01 | 0.936 | 0.206 |
| **Ours (30k)** | **36.01** | **32.60** | **36.74** | **35.04** | **0.955** | **0.187** |

### 消融实验

| 方法 | 室内PSNR↑ | 室外PSNR↑ | 无人机PSNR↑ | 平均PSNR↑ |
|------|-----------|-----------|-------------|-----------|
| 3D-GS | 32.98 | 28.89 | 34.51 | 32.01 |
| 3D-GS + ATF | 35.12 | 31.53 | 36.65 | - |
| **3D-GS + ATF + TCM + Loss** | **36.01** | **32.60** | **36.74** | **35.04** |

### 关键发现

- 相比3D-GS基线平均提升3.03 dB PSNR（35.04 vs 32.01），改善显著
- ATF模块贡献最大（约2 dB），有效消除了漂浮物问题
- TCM在高低温物体边界区域效果明显，可恢复被热传导模糊的清晰边缘
- 无人机场景因飞行速度快导致运动模糊，本方法仍能保持优秀重建质量

## 亮点与洞察

1. **首个热红外新视角合成专用方法**，将物理成像过程（大气传输、热传导）引入3D-GS框架
2. TI-NSD数据集（20场景，6664帧，涵盖室内/室外/无人机）填补了该领域的数据集空白
3. 物理公式驱动的网络设计——不直接求解物理方程（多解问题），而是用物理方程指导神经网络结构设计

## 局限性

- 假设每个高斯共享均匀的衰减系数，可能不适用于衰减变化剧烈的场景
- TCM仅在2D图像空间操作，未在3D空间建模热传导
- 温度不连续损失依赖Harris角点检测的准确性

## 相关工作与启发

本文开创性地将热红外物理特性与3D重建相结合。物理驱动的退化建模思路（将成像效应与几何解耦）可推广到其他特殊成像模态（如SAR、医学影像）的新视角合成。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[CVPR 2026\] Thermal is Always Wild: Characterizing and Addressing Challenges in Thermal-Only Novel View Synthesis](../../CVPR2026/3d_vision/thermal_is_always_wild_characterizing_and_addressing_challenges_in_thermal-only_.md)
- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->
