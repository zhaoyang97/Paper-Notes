---
title: >-
  [论文解读] Hardware-Rasterized Ray-Based Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯泼溅] 本文提出了首个基于硬件光栅化的射线型3D高斯泼溅（RayGS）渲染方案 VKRayGS，通过严谨的数学推导在3D空间中构建最小包围四边形，实现了约40倍的渲染加速，同时保持了RayGS的高质量渲染效果，并额外提出了RayGS的MIP抗锯齿方案。 1. 领域现状：3D高斯泼溅…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯泼溅"
  - "硬件光栅化"
  - "射线追踪"
  - "新视角合成"
  - "VR渲染"
---

# Hardware-Rasterized Ray-Based Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.18682](https://arxiv.org/abs/2503.18682)  
**代码**: [https://github.com/facebookresearch/vkraygs](https://github.com/facebookresearch/vkraygs)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 硬件光栅化, 射线追踪, 新视角合成, VR渲染

## 一句话总结
本文提出了首个基于硬件光栅化的射线型3D高斯泼溅（RayGS）渲染方案 VKRayGS，通过严谨的数学推导在3D空间中构建最小包围四边形，实现了约40倍的渲染加速，同时保持了RayGS的高质量渲染效果，并额外提出了RayGS的MIP抗锯齿方案。

## 研究背景与动机

1. **领域现状**：3D高斯泼溅（3DGS）通过泼溅方式渲染场景，速度快但存在投影近似误差。RayGS通过射线-高斯精确交叉计算消除了这一误差，获得了更高质量的渲染。

2. **现有痛点**：RayGS质量更好但计算开销显著增加，目前主流实现（如GOF）基于CUDA软件光栅化，帧率太低无法满足VR等高帧率应用需求。

3. **核心矛盾**：标准3DGS的support在图像平面上是椭圆，容易用硬件光栅化。但RayGS的support可能是半双曲线，无法简单在图像平面用四边形包围。

4. **本文目标** 如何将RayGS映射到标准硬件光栅化管线（vertex shader + fragment shader），同时保持渲染质量？

5. **切入角度**：放弃在图像平面求解包围四边形，转而在3D空间中寻找包围RayGS primitive support的四边形。证明了support边界上的最大密度点集构成一个3D空间中的2D椭圆，与单位圆同构。

6. **核心 idea**：通过证明RayGS primitive的support边界在3D空间中形成椭圆，建立与单位圆的同构映射，从而在3D空间高效计算最小包围四边形并映射到硬件光栅化管线。

## 方法详解

### 整体框架
输入为一组已训练的RayGS高斯原语集合，输出为给定视角的渲染图像。渲染管线复用标准Vulkan硬件光栅化架构，核心修改在vertex shader（计算每个高斯原语的3D包围四边形顶点）和fragment shader（计算每个像素的RayGS散度/不透明度）。后端的alpha混合逻辑与标准GS硬件实现相同。

### 关键设计

1. **3D空间椭圆-单位圆同构映射（Vertex Shader核心）**:

    - 功能：为每个高斯原语计算3D空间中的最小面积包围四边形
    - 核心思路：对于给定的高斯原语（中心 $\boldsymbol{\mu}$，协方差 $\Sigma$），RayGS的support边界上各射线的最大密度点形成集合 $\mathcal{E}$。通过三步变换证明 $\mathcal{E}$ 与单位圆 $\mathbb{S}_1$ 同构：先做 $\mathbf{S}^{-1}\mathbf{R}_p^\top$ 变换将椭球标准化到单位球面，再旋转使正则化后的中心对齐z轴，最后取x-y子向量归一化得到单位圆上的点。逆映射 $\Phi^{-1}$ 允许将任何包围单位圆的2D四边形映射回3D空间。为了最小化面积，通过求解 $2 \times 2$ 矩阵 $\mathbf{B} = \mathbf{Q}_{0:2}^\top \mathbf{Q}_{0:2}$ 的特征向量找到椭圆主轴方向，构建轴对齐的最紧包围矩形。
    - 设计动机：在无限多个有效3D四边形中选择面积最小的，可以最小化fragment shader处理的无效像素数。2×2矩阵的特征分解有闭式解，计算开销极小。

2. **Fragment Shader中的RayGS散度计算**:

    - 功能：利用硬件插值能力高效计算每个像素的RayGS渲染不透明度
    - 核心思路：对于任何穿过四边形的射线 $\boldsymbol{x}$，存在插值系数 $\boldsymbol{\alpha}$ 使得 RayGS散度可表示为 $\mathcal{D}_{\text{ray}} = \{c^{-2} + \|\mathbf{Z}_{\text{ray}}\boldsymbol{\alpha}\|^{-2}\}^{-1}$。因此只需在顶点处指定 $\mathbf{Z}_{\text{ray}}$，由硬件自动插值，再在fragment shader中做一次点积和简单标量运算即可。与标准GS的fragment shader（仅需点积）相比，额外开销只是一次求逆和求和。
    - 设计动机：通过推导闭式关系，将复杂的RayGS散度计算归结为硬件可自动完成的线性插值+一个简单后处理运算，最大化利用GPU并行能力。

3. **RayGS的MIP抗锯齿方案**:

    - 功能：解决训练和测试分辨率不一致时的锯齿伪影
    - 核心思路：对于归一化射线，将3D高斯分布在射线方向正交平面上边缘化，得到2D高斯。用像素大小对应的各向同性2D高斯做平滑，近似像素区域的积分。最终不透明度中使用膨胀后的协方差 $\hat{\Sigma} = \Sigma + \sigma_x^2 \tau^2(\mathbf{x})\mathbf{I}$，并引入不透明度调制因子 $\sqrt{|\Sigma|c^2 / (|\hat{\Sigma}|\hat{c}^2)}$。为了高效实现，将像素依赖的项近似为常数。
    - 设计动机：VR应用中用户可以自由移动，训练/测试分辨率差距大，不处理MIP会导致明显的锯齿和闪烁。

### 损失函数 / 训练策略
本文是推理时（渲染器）的工作，不涉及新的训练损失。直接使用GOF等方法已训练好的RayGS模型进行渲染。MIP方案在训练时使用也可通过修改不透明度实现。

## 实验关键数据

### 主实验

| 场景 | 指标 | VKRayGS | GOF (CUDA) | 加速比 |
|------|------|---------|------------|--------|
| MipNeRF360 (9场景均值) | FPS↑ | ~232 | ~5.2 | ~40× |
| bicycle | FPS↑ | 177 | 4 | 44× |
| bonsai | FPS↑ | 341 | 6 | 57× |
| MipNeRF360 均值 | PSNR↑ | ~27.2 | ~27.3 | -0.4% |
| MipNeRF360 均值 | LPIPS↓ | ~0.223 | ~0.237 | +5.9% |

### 消融实验

| 对比配置 | FPS | PSNR | 说明 |
|---------|-----|------|------|
| GOF (CUDA RayGS) | ~5 | 27.3 | 基线RayGS渲染器 |
| VKRayGS (本文) | ~232 | 27.2 | 40×加速，质量几无损失 |
| VKGS (标准GS硬件化) | 更快 | 稍低 | 标准GS质量不如RayGS |
| GS CUDA (INRIA) | 中 | 稍低 | CUDA vs Vulkan：Vulkan约2×快 |

### 关键发现
- 平均40×加速是巨大的突破，直接将RayGS从"不可用"提升到VR级别的实时渲染
- 质量差异微小（PSNR降约0.1dB），且部分场景LPIPS反而更好，说明差异来自实现细节而非方法本身
- RTX2080这种中端GPU即可实现实时RayGS渲染（大部分场景>170 FPS）
- 近裁剪面需要禁用，否则四边形被裁剪会产生可见不连续

## 亮点与洞察
- **数学推导的优雅性**：从3D椭圆到单位圆的同构映射推导严谨且几何直觉清晰，这是一个将复杂几何问题化简到2×2矩阵特征分解的典范
- **工程价值极高**：40×加速且保持质量，直接解锁VR/MR场景中使用RayGS，实用性非常突出
- **MIP方案的理论正确性**：相比MIP-Splatting的启发式方案，本文从边缘化3D高斯分布出发推导，理论上更严谨

## 局限与展望
- 当前方案需要禁用近裁剪面，某些极端场景可能产生渲染问题
- 依赖Vulkan API，虽然理论上可移植到OpenGL，但跨平台兼容性仍需验证
- MIP方案中将像素依赖项近似为常数，对于极薄但很长的高斯原语精度可能下降
- 当前只处理推理端，如果能将硬件光栅化扩展到可微渲染用于训练，价值更大

## 相关工作与启发
- **vs GOF (CUDA)**: 同样的RayGS模型，渲染速度提升40×，证明了硬件光栅化相对CUDA软件光栅化的巨大优势
- **vs VKGS**: VKGS是标准GS的硬件光栅化实现，VKRayGS在其上增加了RayGS支持，质量提升的同时速度相当
- **vs 3DGS原版**: RayGS消除了投影近似误差，结合本文的快速渲染器，实现了质量和速度的双赢

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个RayGS硬件光栅化方案，数学推导新颖
- 实验充分度: ⭐⭐⭐⭐ 在标准benchmark上全面评估速度和质量
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，配图直观
- 价值: ⭐⭐⭐⭐⭐ 40×加速直接解锁VR应用场景，工程价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)
- [\[CVPR 2026\] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting](../../CVPR2026/3d_vision/stochastic_ray_tracing_for_the_reconstruction_of_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)
- [\[CVPR 2025\] Geometry Field Splatting with Gaussian Surfels](geometry_field_splatting_with_gaussian_surfels.md)

</div>

<!-- RELATED:END -->
