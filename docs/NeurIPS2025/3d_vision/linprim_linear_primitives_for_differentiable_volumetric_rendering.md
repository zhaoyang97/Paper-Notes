---
title: >-
  [论文解读] LinPrim: Linear Primitives for Differentiable Volumetric Rendering
description: >-
  [NeurIPS 2025][3D视觉][新视角合成] 提出 LinPrim，用线性基元（八面体和四面体）替代3D高斯核作为新视角合成的场景表示，通过可微光栅化pipeline实现端到端优化，在真实数据集上以更少的基元数量达到与3DGS可比的重建质量，同时保持实时渲染能力。 新视角合成（NVS）是3D视觉的核心任务…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "新视角合成"
  - "可微渲染"
  - "体积渲染"
  - "多面体基元"
  - "3D高斯溅射"
---

# LinPrim: Linear Primitives for Differentiable Volumetric Rendering

**会议**: NeurIPS 2025  
**arXiv**: [2501.16312](https://arxiv.org/abs/2501.16312)  
**代码**: [GitHub](https://nicolasvonluetzow.github.io/LinPrim/)  
**领域**: 3D视觉  
**关键词**: 新视角合成, 可微渲染, 体积渲染, 多面体基元, 3D高斯溅射

## 一句话总结

提出 LinPrim，用线性基元（八面体和四面体）替代3D高斯核作为新视角合成的场景表示，通过可微光栅化pipeline实现端到端优化，在真实数据集上以更少的基元数量达到与3DGS可比的重建质量，同时保持实时渲染能力。

## 研究背景与动机

新视角合成（NVS）是3D视觉的核心任务，当前最强方法主要基于NeRF（隐式但计算昂贵）和3D Gaussian Splatting（显式且实时）。**核心矛盾**：3DGS使用高斯核作为基元，虽然效果好但高斯核是无界的、非均匀密度的，不够直觉化且难以直接对接传统三角形网格的3D处理工具链。研究问题是：简单的有界多面体基元能否在NVS中取得与高斯核相当的效果？本文的核心idea是用透明的八面体/四面体作为场景基元，利用光线-三角形求交的可微性来实现梯度优化，探索3D表示的设计空间。

## 方法详解

### 整体框架

LinPrim 建立在3DGS的pipeline之上，替换了核心的场景表示和渲染过程：

- **输入**：多视角RGB图像 + SfM点云
- **场景表示**：透明八面体/四面体集合，每个基元由位置、旋转、顶点距离、不透明度、球谐系数描述
- **渲染**：可微GPU光栅化器：预处理→全局排序+分块→逐像素光线-三角形求交→Alpha混合
- **优化**：L1+SSIM损失，ADAM优化器，梯度反向传播到所有基元参数

### 关键设计

1. **八面体基元设计**:

    - 限制顶点位于相对中心的坐标轴上，对称地用一个距离描述对向顶点对
    - 每个基元11个浮点数：3（位置）+ 4（旋转四元数）+ 3（三对轴距离）+ 1（不透明度），与标准高斯核参数量相同
    - 加上48个球谐系数描述视角相关的颜色
    - 强制对称性确保中心是几何重心，简化包围盒计算

2. **四面体基元设计**:

    - 四个等间距基向量描述从中心到顶点的方向，优化各自距离
    - 由于缺乏对称性，需要12个浮点数（比八面体多1个）
    - 性能略逊但验证了框架的通用性

3. **可微渲染pipeline**:

    - **预处理**：构建多面体→相机坐标变换→射影空间近似→计算屏幕包围盒
    - **光栅化**：使用 Möller-Trumbore 算法（MTIA）进行光线-三角形求交，完全可微
    - **不透明度计算**：凸多面体与像素光线最多两个交点，密度基于交点间距离：$\sigma(\alpha) = -\frac{\log(1 - 0.99 \cdot \alpha)}{2 \cdot \min(d_x, d_y, d_z)}$
    - **混合**：前到后Alpha混合，累积不透明度达0.999时截断

4. **抗锯齿**:

    - 3D平滑滤波器：限制基元最小尺寸，确保至少在一个训练视角的一个像素可见
    - 2D类Mip滤波器：扩展投影包围盒，增大基元在屏幕空间的最小尺寸（近似方法）

5. **种群控制（Population Control）**:

    - 从SfM点初始化，随机旋转确保均匀覆盖
    - 剪枝：移除过透明、过大或屏幕占比过高的基元
    - 克隆/分裂：高位置梯度基元，小的克隆、大的分裂
    - 分裂采样：按各轴距离为标准差从PDF采样新位置（八面体对称可行）
    - 兼容GS-MCMC的种群控制策略

### 损失函数 / 训练策略

使用3DGS的标准损失：$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{SSIM}$。ADAM优化器，30k迭代训练。学习率根据场景尺度（训练相机最大距离）自适应调整。所有场景使用一致的超参数。

## 实验关键数据

### 主实验

| 方法 | ScanNet++ PSNR | ScanNet++ 基元数 | MipNeRF360 PSNR | MipNeRF360 基元数 |
|------|---------------|-----------------|-----------------|------------------|
| 3DGS | 24.09 | 738k | 27.43 | 3.32M |
| Mip-Splatting | 24.12 | 977k | 27.79 | 4.17M |
| 3D Convex Splatting | 24.26 | 440k | 27.22 | 1.02M |
| **LinPrim (Octa)** | **24.04** | **255k** | 26.63 | **1.79M** |
| GS-MCMC (255k) | 24.50 | 255k | -- | -- |
| LinPrim+MCMC (255k) | **24.41** | 255k | -- | -- |
| LinPrim+MCMC (738k) | **24.55** | 738k | 27.04 | 3.32M |

### 消融实验

| 配置 | ScanNet++ PSNR | MipNeRF360 PSNR | 说明 |
|------|---------------|-----------------|------|
| 八面体 | 24.04 / 0.849 SSIM | 26.63 / 0.803 SSIM | 默认方案 |
| 四面体 | 24.05 / 0.848 SSIM | 25.96 / 0.790 SSIM | ScanNet++持平，MipNeRF360劣 |
| LinPrim+MCMC vs GS-MCMC (同数量) | 24.41 vs 24.50 | -- | 基元类型差距小 |
| 默认种群控制 vs MCMC控制 | 24.04→24.41 | 26.63→27.04 | MCMC改善显著 |

### 关键发现

1. **用更少基元达到相当质量**：LinPrim在ScanNet++上用255k基元接近3DGS的738k基元效果，说明有界多面体的表达效率更高
2. **八面体优于四面体**：对称性更好带来更稳定的优化，在MipNeRF360上差距明显（26.63 vs 25.96）
3. **兼容MCMC种群控制**：证明LinPrim不是特定于3DGS种群策略的，可以受益于更先进的方法
4. **有界基元的优劣势**：在反射/透明区域能产生更清晰的深度估计，但在观察不充分区域会出现"分段状"硬边
5. **小场景更适合**：密集覆盖的较小场景效果最佳，保持几何和视觉清晰度

## 亮点与洞察

- 科学价值大于工程价值：系统探索了多面体基元在NVS中的可行性，拓展了3D表示的设计空间
- 八面体11个参数与高斯核相同，参数效率优（更少基元、接近质量），暗示高斯核可能过度参数化
- 光线-三角形求交天然可微且成熟（MTIA），理论上可直接利用已有的三角形渲染硬件加速
- 从SfM初始化后用随机旋转确保覆盖均匀性是一个细节但重要的设计

## 局限与展望

- 在MipNeRF360大场景上PSNR低于3DGS约0.8dB，差距较明显
- 有界基元在观察稀疏区域会产生硬边伪影（不像高斯平滑过渡）
- 当前渲染效率低于优化后的高斯光栅化器，实际FPS有差距
- 种群控制策略原本为高斯设计，多面体专用策略可能显著提升性能
- 四面体由于不对称性需要特殊处理，当前pipeline未充分优化

## 相关工作与启发

- 与3D Convex Splatting（NeurIPS 2024）最相似，但LinPrim使用更简单的均匀密度多面体而非平滑凸体
- 与EVER（均匀密度椭球）的关系：LinPrim进一步将基元简化为多面体
- 与Beyond Gaussians（线性核）的区别：后者仍是椭圆非均匀密度，LinPrim是有界均匀
- 启示：NVS基元设计空间远大于高斯核，有界/均匀密度的简单基元值得更多探索

## 评分

- 新颖性: ⭐⭐⭐⭐ 多面体基元在NVS中的首次系统探索，但核心pipeline仍借鉴3DGS
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、两种基元、MCMC对比，消融较全
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法推导严谨，讨论诚实
- 价值: ⭐⭐⭐⭐ 拓展了3D表示设计空间，但性能尚未超越3DGS

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MeshSplatting: Differentiable Rendering with Opaque Meshes](../../CVPR2026/3d_vision/meshsplatting_differentiable_rendering_with_opaque_meshes.md)
- [\[CVPR 2026\] D-Prism: Differentiable Primitives for Structured Dynamic Modeling](../../CVPR2026/3d_vision/d-prism_differentiable_primitives_for_structured_dynamic_modeling.md)
- [\[CVPR 2026\] Hermite Radial Basis Function for Surface Reconstruction via Differentiable Rendering](../../CVPR2026/3d_vision/hermite_radial_basis_function_for_surface_reconstruction_via_differentiable_rend.md)
- [\[NeurIPS 2025\] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)
- [\[CVPR 2026\] Voxify3D: Pixel Art Meets Volumetric Rendering](../../CVPR2026/3d_vision/voxify3d_pixel_art_meets_volumetric_rendering.md)

</div>

<!-- RELATED:END -->
