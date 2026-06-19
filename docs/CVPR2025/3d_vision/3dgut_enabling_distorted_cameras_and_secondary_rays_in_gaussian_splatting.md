---
title: "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
authors: "Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic"
affiliations: "NVIDIA, University of Toronto"
venue: "CVPR 2025"
date: 2024-12-17
tags: ["3D Gaussian Splatting", "Unscented Transform", "rolling shutter", "secondary rays", "differentiable rendering"]
arxiv: "2412.12507"
code: "https://github.com/nv-tlabs/3dgrut"
---

# 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting

## 研究背景与动机

3D Gaussian Splatting (3DGS) 在新视角合成中取得了显著成功，但其核心的 EWA splatting 管线存在根本性限制。EWA splatting 要求将3D高斯投影到2D平面上，这一过程依赖于**仿射近似**（一阶泰勒展开），假设投影在高斯支撑范围内是局部线性的。这一假设在以下场景中严重不成立：

**畸变相机模型**：鱼眼镜头、滚动快门（rolling shutter）等非针孔相机模型引入非线性畸变，仿射近似导致明显伪影

**二次光线**：反射和折射等效果需要追踪二次光线，这些光线的方向与原始投影方向不同，无法用单一仿射变换描述

**广角镜头**：视场角超过120°时，边缘区域的透视失真使得线性近似误差急剧增大

现有方法尝试通过后处理或特定场景的修补来解决这些问题，但缺乏统一的数学框架。本文提出使用**Unscented Transform (UT)** 替代 EWA 中的仿射近似，从根本上解决这一限制。

## 方法详解

### Unscented Transform 原理

Unscented Transform 是一种用于非线性变换下概率分布传播的技术。其核心思想是通过一组精心选择的 **sigma points** 来捕获分布的统计特性，而非对变换函数本身进行线性化。

对于一个 $n$ 维高斯分布 $\mathcal{N}(oldsymbol{\mu}, oldsymbol{\Sigma})$，UT 选取 $2n+1$ 个 sigma points：

$$oldsymbol{\chi}_0 = oldsymbol{\mu}, \quad oldsymbol{\chi}_i = oldsymbol{\mu} + \sqrt{(n+\lambda)} \cdot oldsymbol{L}_i, \quad oldsymbol{\chi}_{n+i} = oldsymbol{\mu} - \sqrt{(n+\lambda)} \cdot oldsymbol{L}_i$$

其中 $oldsymbol{L}$ 是 $oldsymbol{\Sigma}$ 的 Cholesky 分解，$\lambda = lpha^2(n+\kappa) - n$ 是缩放参数。

### 3DGUT Splatting 管线

本文将 UT 集成到 3DGS 的 splatting 管线中，形成 **3DGUT**：

| 步骤 | EWA Splatting | 3DGUT (本文) |
|------|--------------|-------------|
| 投影方式 | 仿射近似（雅可比矩阵） | Unscented Transform (sigma points) |
| 相机模型 | 仅支持针孔 | 任意可微相机模型 |
| 二次光线 | 不支持 | 原生支持 |
| 计算开销 | 低 | 稍高（7个sigma点 vs 1次矩阵乘法） |
| 精度 | 一阶近似 | 二阶精度 |

#### Sigma Points 生成

对每个3D高斯 $\mathcal{G}(oldsymbol{\mu}_{3D}, oldsymbol{\Sigma}_{3D})$，生成7个sigma points（3D空间中 $n=3$）：
- 中心点 $oldsymbol{\chi}_0 = oldsymbol{\mu}_{3D}$
- 沿协方差矩阵主轴方向的6个采样点

#### 非线性投影

将所有sigma points通过完整的非线性相机投影函数 $\mathbf{h}(\cdot)$：

$$oldsymbol{\mathcal{Y}}_i = \mathbf{h}(oldsymbol{\chi}_i)$$

这里 $\mathbf{h}$ 可以是任意可微的投影函数，包括包含畸变系数的鱼眼模型或rolling shutter模型。

#### 2D高斯恢复

从投影后的sigma points恢复2D高斯参数：

$$oldsymbol{\mu}_{2D} = \sum_i w_i^{(m)} oldsymbol{\mathcal{Y}}_i, \quad oldsymbol{\Sigma}_{2D} = \sum_i w_i^{(c)} (oldsymbol{\mathcal{Y}}_i - oldsymbol{\mu}_{2D})(oldsymbol{\mathcal{Y}}_i - oldsymbol{\mu}_{2D})^T$$

### 滚动快门支持

Rolling shutter 相机的每行像素在不同时刻曝光，导致运动物体出现果冻效应。3DGUT 将时间维度纳入投影函数：

$$\mathbf{h}_{RS}(oldsymbol{x}, t) = \pi(oldsymbol{T}(t) \cdot oldsymbol{x})$$

其中 $oldsymbol{T}(t)$ 是随时间变化的相机姿态。UT 可以自然地处理这种时空耦合的非线性投影。

### 二次光线追踪

对于反射和折射效果，本文在每个高斯上附加法线信息，通过光线-高斯交点计算反射/折射方向，生成二次光线。二次光线再次与场景中的高斯进行交互，UT 同样适用于这一过程。

## 实验结果

### 标准场景对比

| 方法 | Mip-NeRF360 PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|
| 3DGS (原始) | 27.21 | 0.815 | 0.214 |
| Mip-Splatting | 27.79 | 0.827 | 0.203 |
| 3DGUT (针孔) | 27.83 | 0.829 | 0.199 |
| 3DGUT (畸变) | **28.14** | **0.838** | **0.188** |

### Rolling Shutter 场景

在 TUM-RS 和 Unreal-RS 数据集上，3DGUT 在处理滚动快门场景时显著优于忽略快门效应的基线方法，PSNR 提升 2-4 dB。

### 反射/折射场景

在包含镜面反射和玻璃折射的合成场景中，3DGUT 是首个能够在3DGS框架内直接处理二次光线效果的方法，避免了以往需要额外环境贴图或多层表示的复杂设计。

### 消融实验

| 组件 | PSNR | 说明 |
|------|------|------|
| Full 3DGUT | 28.14 | 完整模型 |
| w/o UT (回退EWA) | 27.21 | 退化为标准3DGS |
| UT + 针孔 | 27.83 | 仅UT，不含畸变 |
| UT + 畸变 | 28.14 | 完整畸变模型 |

## 总结与展望

3DGUT 通过引入 Unscented Transform 替代传统的 EWA splatting 中的仿射近似，提供了一个统一且优雅的框架来处理畸变相机模型和二次光线效果。这一方法的核心优势在于**不需要对投影函数做任何线性化假设**，只需要投影函数本身是可微的即可。虽然计算开销略有增加（约1.3×），但获得的灵活性和精度提升使其在实际应用中具有显著价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [\[CVPR 2026\] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](../../CVPR2026/3d_vision/directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)
- [\[ICLR 2026\] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](../../ICLR2026/3d_vision/3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)

</div>

<!-- RELATED:END -->
