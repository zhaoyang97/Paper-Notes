---
title: >-
  [论文解读] Textured Gaussians for Enhanced 3D Scene Appearance Modeling
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] Textured Gaussians 将传统图形学中的纹理映射和Alpha映射引入 3DGS，为每个高斯体分配独立的 2D RGBA 纹理图，使单个高斯体能表达空间变化的颜色和透明度，大幅提升了3DGS的表达能力——在相同高斯数量下提升渲染质量，在 1% 高斯数量下 PSNR 提升近 2dB。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "纹理映射"
  - "新视角合成"
  - "外观建模"
  - "Alpha映射"
---

# Textured Gaussians for Enhanced 3D Scene Appearance Modeling

**会议**: CVPR 2025  
**arXiv**: [2411.18625](https://arxiv.org/abs/2411.18625)  
**代码**: [https://textured-gaussians.github.io](https://textured-gaussians.github.io)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 纹理映射, 新视角合成, 外观建模, Alpha映射

## 一句话总结
Textured Gaussians 将传统图形学中的纹理映射和Alpha映射引入 3DGS，为每个高斯体分配独立的 2D RGBA 纹理图，使单个高斯体能表达空间变化的颜色和透明度，大幅提升了3DGS的表达能力——在相同高斯数量下提升渲染质量，在 1% 高斯数量下 PSNR 提升近 2dB。

## 研究背景与动机

**领域现状**：3D Gaussian Splatting (3DGS) 已成为新视角合成的 SOTA 方法，凭借高质量渲染、快速训练和推理、显式表示等优势，被广泛应用于表面重建、场景编辑、人体建模等任务。

**现有痛点**：3DGS 的单个高斯体有两个根本性限制：(1) **颜色单一**——同一高斯体覆盖的所有像素被着色为相同颜色（仅有高斯衰减因子的缩放），无法表达空间变化的纹理细节；(2) **形状受限**——每个高斯体只能表示椭球形状，无法表达复杂的几何结构。这意味着 3DGS 需要大量高斯体来拟合高频纹理和精细几何，导致内存和计算开销大。

**核心矛盾**：单个高斯原语的表达能力与整体模型效率之间的 trade-off。要想高质量渲染就需要密集的高斯体，但高斯体数量过多会导致训练和渲染效率下降。

**本文目标**：在不改变3DGS整体框架的前提下，大幅提升单个高斯体的表达能力，使其能用更少的高斯体达到同等或更好的渲染质量。

**切入角度**：作者从传统网格渲染（mesh rendering）中获得灵感——网格表面通过纹理映射来表达复杂外观，每个三角面片可以对应纹理图上的一个区域。类比地，每个高斯体也可以拥有自己的纹理图。

**核心 idea**：给每个3D高斯体分配一个固定分辨率的 2D 纹理图（支持 Alpha-only / RGB / RGBA），通过射线-高斯相交和 UV 映射来查询纹理值，实现空间变化的颜色和透明度。

## 方法详解

### 整体框架
Textured Gaussians 建立在 3DGS 之上，渲染pipeline为：(1) 从相机中心向每个像素发射射线；(2) 对射线与场景中3D高斯体求交，得到交点；(3) 在交点处通过 UV 映射查询该高斯体的纹理图，获取纹理颜色 $\mathbf{c}^{tex}$ 和纹理 alpha 值 $\alpha^{tex}$；(4) 将纹理颜色与 SH 基础颜色叠加，纹理alpha与高斯opacity相乘，按前后顺序进行alpha合成得到最终像素颜色。

### 关键设计

1. **射线-高斯相交与 UV 映射**:

    - 功能：将像素射线映射到高斯体的局部纹理坐标
    - 核心思路：每个高斯体由三个主轴定义，取两个最大尺度的轴构成一个平面 $\mathcal{P}$，法线方向为最小尺度轴。对于像素 $\mathbf{p}$，从相机原点 $\mathbf{o}$ 发射射线，与平面 $\mathcal{P}$ 求交得到 3D 交点 $\mathbf{x}$。然后将交点投影到高斯的两个主轴方向上，归一化到 $[0, \mathcal{T}-1]$ 的纹理坐标 $(u, v)$：$u = \frac{m \cdot \sigma_1 + (\mathbf{x} - \boldsymbol{\mu}) \cdot \mathbf{r}_1}{2m \cdot \sigma_1} \cdot (\mathcal{T}-1)$，其中 $\sigma_1, \sigma_2$ 是尺度，$\mathbf{r}_1, \mathbf{r}_2$ 是轴方向，$m=3$ 是范围倍数
    - 设计动机：这种 UV 映射方式将纹理自然地铺在高斯体的"表面"上，且与高斯体的几何变换（旋转、缩放）完全兼容

2. **广义高斯外观模型**:

    - 功能：统一描述原始3DGS和Textured Gaussians的渲染方程
    - 核心思路：将像素最终颜色的渲染方程扩展为：$\mathbf{c}_{final}(\mathbf{p}) = \sum_{i=1}^K \mathbf{c}_i(\mathbf{p}) \cdot \alpha_i(\mathbf{p}) \cdot \prod_{j=1}^{i-1}(1-\alpha_j(\mathbf{p}))$，其中颜色 $\mathbf{c}_i(\mathbf{p}) = \mathbf{c}_i^{base} + \mathbf{c}_i^{tex}(u,v)$（SH基础颜色+纹理颜色），alpha值 $\alpha_i(\mathbf{p}) = \alpha_i^{tex}(u,v) \cdot \mathcal{G}_i(\mathbf{x}) \cdot o_i$（纹理alpha × 高斯值 × 不透明度）。当 $\mathbf{c}^{tex}=0, \alpha^{tex}=1$ 时退化为原始3DGS
    - 设计动机：RGB纹理让高斯体表达高频颜色变化；alpha纹理让高斯体突破椭球形状限制——通过空间变化的透明度可以"雕刻"出任意形状

3. **两阶段优化策略**:

    - 功能：稳定训练，避免联合优化的病态问题
    - 核心思路：**Stage 1**（30K iterations）：按标准3DGS流程优化所有高斯属性（位置、旋转、尺度、SH系数、不透明度），包括自适应密度控制（ADC）。**Stage 2**（30K iterations）：用Stage 1的结果初始化所有高斯属性，冻结ADC，联合优化高斯属性和纹理图参数。纹理RGB初始化为近零值（$25/255$），alpha通道初始化为1
    - 设计动机：联合优化所有参数是高度不适定问题（高斯位置和纹理内容存在大量歧义），两阶段策略让Stage 1先确定好几何布局，Stage 2在此基础上学习纹理细节

### 损失函数 / 训练策略
使用标准的加权光度损失：$\mathcal{L} = \lambda \mathcal{L}_1 + (1-\lambda) \mathcal{L}_{SSIM}$，$\lambda=0.8$。纹理图的学习率设为 0.001，对所有数据集统一使用，不需要逐数据集调参。所有实验在 NVIDIA H100 GPU 集群上完成。

## 实验关键数据

### 主实验

在 5 个标准基准数据集上评估 PSNR/SSIM/LPIPS，与 3DGS*（改进版3DGS实现）在相同高斯数量下比较。

| 数据集 | 3DGS* (PSNR/SSIM/LPIPS) | RGBA Textured (PSNR/SSIM/LPIPS) | 1% GS 3DGS* | 1% GS Ours |
|--------|--------------------------|----------------------------------|-------------|------------|
| Blender | 33.09/0.967/0.044 | **33.31/0.969/0.038** | 26.89/0.916/0.117 | **28.02/0.934/0.085** |
| Mip-NeRF 360 | 27.28/0.832/0.187 | **27.43/0.838/0.176** | 22.37/0.629/0.477 | **23.75/0.707/0.337** |
| DTU | 33.54/0.970/0.055 | **33.68/0.972/0.050** | 30.88/0.932/0.158 | **32.41/0.963/0.070** |
| Tanks & Temples | 24.18/0.854/0.175 | **24.39/0.860/0.163** | 19.90/0.674/0.441 | **21.08/0.738/0.311** |
| Deep Blending | 28.04/0.894/0.271 | **28.52/0.902/0.239** | 23.97/0.817/0.434 | **24.88/0.845/0.371** |

### 消融实验（纹理类型 ablation）

| 纹理类型 | Blender (PSNR) | Mip-NeRF 360 | DTU | T&T | DB |
|----------|---------------|-------------|-----|-----|-----|
| 3DGS* (无纹理) | 33.09 | 27.28 | 33.54 | 24.18 | 28.04 |
| Alpha-only | **33.22** | 27.32 | 33.51 | 24.27 | **28.36** |
| RGB-only | 33.18 | 27.30 | 33.48 | 24.22 | 28.30 |
| RGBA | **33.31** | **27.43** | **33.68** | **24.39** | **28.52** |

### 关键发现
- **Alpha-only 纹理的效果出人意料地好**：仅用 alpha 通道（模型大小为 RGBA 的 1/4）就能超过 RGB-only 纹理并接近 RGBA。这是因为 alpha 纹理让高斯体突破椭球限制，通过空间变化的不透明度实现复杂形状，而 RGB 纹理仍被限制在椭球形状内
- 高斯数量越少，Textured Gaussians 的优势越大：在 1% 高斯数下 PSNR 提升近 2dB，在 100% 高斯下提升 0.2-0.5dB
- 相同模型大小下（通过减少高斯数来补偿纹理开销），alpha-only 纹理通常表现最好，说明存在高斯参数与纹理参数之间的最优分配比例
- 纹理分辨率和高斯数量之间存在 sweet spot：固定模型大小时，最佳性能不是最大纹理分辨率，也不是最多高斯数，而是两者的平衡点

## 亮点与洞察
- **alpha 纹理比 RGB 纹理更重要的发现违反直觉但意义深远**：它揭示了 3DGS 的瓶颈不在于颜色表达，而在于形状表达。椭球形状的限制比单一颜色的限制对渲染质量的影响更大。这一洞察可以指导未来 3DGS 改进的方向
- **纹理映射+高斯溅射的结合非常自然**：射线-高斯相交本身就是 Textured Gaussians 需要的操作，额外计算开销极小，推理速度几乎不变
- **两阶段训练策略可推广到其他 3DGS 扩展方法**：任何给高斯体增加额外属性的方法都可以采用"先优化几何，再优化添加属性"的策略

## 局限与展望
- 当前使用 2D 漫反射纹理，无法建模空间变化的镜面反射颜色；未来可扩展为 3D 体积纹理或 5D 辐射场
- 两阶段训练使训练时间约为标准 3DGS 的 2 倍
- 与同期工作 GStex 相比，GStex 基于 2D Gaussian Splatting 且不支持 alpha 通道，表达能力受限
- 纹理分辨率是固定的，不同大小的高斯体使用相同分辨率可能不够灵活——大高斯需要更高分辨率的纹理

## 相关工作与启发
- **vs 3DGS (原始)**: 3DGS 每个高斯只有一个颜色和椭球形状，Textured Gaussians 通过纹理映射大幅提升单体表达能力
- **vs Textured-GS (Huang & Gong)**: 该方法通过修改视角方向计算让颜色在高斯内平滑变化，但由于 SH 表示的平滑性，无法重建高频纹理。本文的纹理图没有这个限制
- **vs Texture-GS (Xu et al.)**: 该方法使用全局纹理图 + 学习的 UV 映射，受限于球面参数化，无法处理复杂几何和大场景。本文的每高斯独立纹理更灵活
- **vs GStex (Rong et al., concurrent)**: GStex 基于 2D 高斯盘分布纹理元素，但缺少 alpha 通道，无法表达复杂形状

## 评分
- 新颖性: ⭐⭐⭐⭐ 将纹理映射引入3DGS的想法自然但有效，alpha纹理的发现是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集+自定义数据集，多种高斯数量/纹理分辨率/纹理类型的全面消融
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述清晰，公式推导完整，消融实验系统化
- 价值: ⭐⭐⭐⭐ 提升单高斯体表达能力的方向有实际意义，尤其对内存受限的场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](../../ICCV2025/3d_vision/can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[CVPR 2025\] RNG: Relightable Neural Gaussians](rng_relightable_neural_gaussians.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)

</div>

<!-- RELATED:END -->
