---
title: >-
  [论文解读] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction
description: >-
  [AAAI 2026][3D视觉][动态场景重建] 提出 Sparse4DGS，首个面向稀疏帧输入的4D动态场景重建方法，通过纹理感知的变形正则化（TADR）和纹理感知的规范优化（TACO）两大核心模块，引导高斯分布聚焦纹理丰富区域，在仅5-30帧稀疏输入下实现高质量动态新视角合成。 动态高斯溅射方法在4D场景重建中取得了显…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "动态场景重建"
  - "4D高斯溅射"
  - "稀疏帧"
  - "纹理感知"
  - "随机梯度朗之万动力学"
---

# Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2511.07122](https://arxiv.org/abs/2511.07122)  
**代码**: [项目页面](https://ChangyueShi.github.io/Sparse4DGS)  
**领域**: 3D视觉  
**关键词**: 动态场景重建, 4D高斯溅射, 稀疏帧, 纹理感知, 随机梯度朗之万动力学

## 一句话总结

提出 Sparse4DGS，首个面向稀疏帧输入的4D动态场景重建方法，通过纹理感知的变形正则化（TADR）和纹理感知的规范优化（TACO）两大核心模块，引导高斯分布聚焦纹理丰富区域，在仅5-30帧稀疏输入下实现高质量动态新视角合成。

## 研究背景与动机

动态高斯溅射方法在4D场景重建中取得了显著进展，但现有方法如 Deformable3DGS 和 4DGaussians 严重依赖密集帧视频序列（通常需要数百帧）。在真实世界中，由于设备限制（如低帧率摄像头），往往只能获取稀疏帧。

作者发现，当输入帧数从密集降为稀疏时，现有方法在**纹理丰富区域**出现严重退化。这是因为：
1. **变形空间退化**：稀疏输入提供的时间约束不足，导致变形网络在高频纹理区域无法准确建模几何变化
2. **规范空间退化**：规范高斯场缺乏足够的监督信号，在纹理复杂区域容易出现几何坍缩

核心直觉是：稀疏帧输入本质上提供了有限的信息，此时高频纹理信号成为丰富细节和动态线索的主要来源。因此，应当引导高斯关注纹理丰富区域，从而更好地建模底层结构。

## 方法详解

### 整体框架

Sparse4DGS 基于规范高斯场+变形网络的动态重建范式。输入稀疏帧序列后：
1. 使用 Sobel 算子提取每帧的2D纹理强度（TI）图
2. 使用单目深度估计器（DPT）获取深度图
3. 将纹理强度嵌入3D高斯属性中
4. 通过 TADR 正则化变形网络
5. 通过 TACO 优化规范高斯场

### 关键设计

#### 1. **纹理强度高斯场（TI Gaussian Field）**：将纹理丰富度信息嵌入3D高斯

首先通过 Sobel 算子计算每个输入 RGB 图像的水平和垂直梯度图 $TI_x$ 和 $TI_y$，然后得到逐像素梯度幅值作为纹理强度的显式度量：

$$TI_{gt}(i,j) = \sqrt{TI_x(i,j)^2 + TI_y(i,j)^2}$$

为了在3D空间表示纹理丰富度，为每个高斯引入新属性 $TI$，通过可微光栅化器渲染成纹理图 $TI_{render}$。

**关键创新**：使用皮尔逊相关系数（PCC）而非常规 L1 损失来对齐渲染纹理图与真值纹理图。这是因为 Sobel 算子独立应用于每张图像会导致空间不一致性，而 PCC 关注相对变化率，能有效缓解这一问题：

$$L_{tex} = 1 - \text{PCC}(TI_{gt}, TI_{render})$$

#### 2. **纹理感知变形正则化（TADR）**：约束变形网络的几何结构

TADR 的核心思想是利用深度图的纹理一致性来约束变形场。传统方法直接对比渲染深度和单目深度的图像级 PCC，但这无法捕获局部深度变化。

TADR 的做法是：
- 先用 Sobel 对渲染深度 $D_{render}$ 和 DPT 深度 $D_{dpt}$ 分别提取纹理强度图
- 然后对这两个深度纹理图计算 PCC 损失

$$L_{tadr} = 1 - \text{PCC}(TI_{gt}^{depth}, TI_{render}^{depth})$$

这种"纹理化"的深度对齐方式更关注局部深度变化的一致性，而非全局深度分布。

#### 3. **纹理感知规范优化（TACO）**：重构规范高斯的梯度下降过程

TACO 基于随机梯度朗之万动力学（SGLD），在每次迭代中引入基于纹理强度的噪声项，驱动高斯向纹理丰富区域收敛：

$$g = g - \alpha_g \cdot \nabla_g \mathbb{E}[L(g;I)] + \alpha_{noise} \cdot (\epsilon_{tex} + \epsilon_o)$$

其中纹理噪声项为：
$$\epsilon_{tex} = \sigma(-k(TI - t)) \cdot \sum \eta$$

当高斯到达纹理丰富区域时，$TI$ 值趋近于1，$\epsilon_{tex}$ 趋近于0，噪声自然停止。这意味着噪声会持续扰动优化过程，直到高斯收敛到纹理丰富区域。$\epsilon_o$ 则用于减少低不透明度的模糊高斯（floaters）。

### 损失函数 / 训练策略

总训练损失为：
$$L = L_{rgb} + \lambda_1 \cdot L_{tex} + \lambda_2 \cdot L_{tadr}$$

其中 $L_{rgb}$ 为标准的 MSE + SSIM 损失。最优超参数为 $\lambda_1 = \lambda_2 = 0.01$。

训练过程使用 TACO 替代标准 SGD 更新规范高斯参数。此方法适用于从5 FPS到30 FPS的不同帧率视频。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Sparse4DGS | Deformable3DGS | 4DGaussians | CoRGS | 提升 |
|--------|------|------------|----------------|-------------|-------|------|
| NeRF-Synthetic (20帧) | PSNR↑ | **25.31** | 22.65 | 22.47 | 20.15 | +2.66 |
| NeRF-Synthetic (20帧) | SSIM↑ | **0.944** | 0.927 | 0.931 | 0.920 | +0.013 |
| NeRF-DS (20帧) | PSNR↑ | **22.34** | 20.81 | 19.70 | 19.86 | +1.53 |
| NeRF-DS (20帧) | LPIPS↓ | **0.233** | 0.301 | 0.350 | 0.319 | -0.068 |
| HyperNeRF (30帧) | PSNR↑ | **23.91** | 22.41 | 20.64 | 20.50 | +1.50 |
| iPhone-4D (30FPS) | PSNR↑ | **29.81** | 27.01 | 28.79 | 21.58 | +1.02 |
| iPhone-4D (5FPS) | PSNR↑ | **27.51** | 21.12 | 16.37 | 16.81 | +6.39 |

在所有数据集上均大幅领先，尤其在极端稀疏的5FPS场景中，PSNR提升超过6dB。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| Baseline（无TADR+TACO） | 20.81 | 0.753 | 0.301 | 基线方法 |
| w/o TADR | 21.89 | 0.792 | 0.245 | 去除变形正则化，PSNR降0.45 |
| w/o TACO | 21.33 | 0.773 | 0.271 | 去除规范优化，PSNR降1.01 |
| **完整方法** | **22.34** | **0.801** | **0.233** | TACO贡献更大 |
| TACO w/o $\epsilon_o$ | 21.81 | 0.792 | 0.246 | 去除不透明度噪声项 |
| TACO w/o $\epsilon_{tex}$ | 21.57 | 0.783 | 0.260 | 去除纹理噪声项 |
| $L_{tex}$ w/o PCC | 21.71 | 0.789 | 0.245 | PCC换L1，降0.6 |
| w/o texture-aware depth | 21.46 | 0.775 | 0.277 | 常规深度正则化 |

### 关键发现

1. TACO 的贡献大于 TADR（1.01 vs 0.45 PSNR提升），说明规范空间优化是稀疏帧重建的瓶颈
2. PCC 损失相比 L1 损失在纹理嵌入和深度对齐中均有显著优势
3. 纹理感知的深度损失相比直接深度 PCC 对齐提升0.88 PSNR
4. 在5FPS极端稀疏场景下优势最为显著（+6.39 PSNR）

## 亮点与洞察

1. **问题定义新颖**：首次定义并系统研究稀疏帧4D动态场景重建问题
2. **纹理驱动的优化策略**：观察到稀疏帧下退化集中在纹理丰富区域，并基于此设计了完整的解决方案
3. **SGLD框架的创新应用**：将随机梯度朗之万动力学引入动态高斯优化，纹理引导的噪声项设计优雅且有效
4. **PCC替代L1**：在存在空间不一致性的场景中，PCC作为相关性度量比L1更鲁棒
5. **真实场景验证**：提出 iPhone-4D 数据集，展示了在手机拍摄视频上的实际应用潜力

## 局限与展望

1. 当场景中纹理信息极度匮乏时（如纯色墙壁），方法效果可能受限
2. 依赖 DPT 单目深度估计器的精度，预训练深度模型的误差会传播
3. iPhone-4D 数据集规模较小（仅4个场景），验证范围有限
4. 未探索极短序列（如2-3帧）的情况
5. TACO 的噪声超参数可能需要针对不同场景调优

## 相关工作与启发

- **动态高斯溅射**：Deformable3DGS、4DGaussians 提供了标准的规范场+变形网络框架
- **少样本高斯溅射**：DNGaussian 提出深度正则化，CoRGS 改进训练过程，FSGS 解决稀疏初始化
- **SGLD在3DGS中的应用**：Kheradmand et al. 首次将SGLD引入高斯溅射优化
- **启发**：纹理引导的优化思路可推广到其他稀疏输入的3D重建任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个稀疏帧动态重建，纹理感知策略新颖
- **实验充分度**: ⭐⭐⭐⭐ — 四个数据集，详尽消融实验
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法推导严谨
- **实用价值**: ⭐⭐⭐⭐ — 对低帧率视频的动态重建有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos](dynamic_gaussian_scene_reconstruction_from_unsynchronized_videos.md)
- [\[ICLR 2026\] Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements](../../ICLR2026/3d_vision/implicit_4d_gaussian_splatting_for_fast_motion_with_large_inter-frame_displaceme.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[ICLR 2026\] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)

</div>

<!-- RELATED:END -->
