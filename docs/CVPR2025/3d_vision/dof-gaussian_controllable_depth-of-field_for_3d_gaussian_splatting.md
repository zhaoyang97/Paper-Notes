---
title: >-
  [论文解读] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯] 提出 DoF-Gaussian，为 3D 高斯表示引入基于几何光学的可学习透镜成像模型，通过逐场景深度先验调整和离焦-对焦自适应策略，实现从浅景深（散焦模糊）输入图像重建清晰 3D 场景，并支持可控景深渲染（重对焦、光圈调节、散焦形状变换等交互应用）。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯"
  - "景深控制"
  - "散焦去模糊"
  - "透镜成像模型"
  - "深度先验"
  - "弥散圆"
---

# DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.00746](https://arxiv.org/abs/2503.00746)  
**代码**: [https://dof-gaussian.github.io/](https://dof-gaussian.github.io/)  
**领域**: 3D视觉 / 新视角合成  
**关键词**: 3D高斯, 景深控制, 散焦去模糊, 透镜成像模型, 深度先验, 弥散圆

## 一句话总结

提出 DoF-Gaussian，为 3D 高斯表示引入基于几何光学的可学习透镜成像模型，通过逐场景深度先验调整和离焦-对焦自适应策略，实现从浅景深（散焦模糊）输入图像重建清晰 3D 场景，并支持可控景深渲染（重对焦、光圈调节、散焦形状变换等交互应用）。

## 研究背景与动机

**领域现状**：3DGS 及其变体在新视角合成上取得了巨大成功，但它们基于针孔成像假设，要求输入图像全部对焦清晰。然而真实世界照片常常包含浅景深效果（散景模糊），这在日常摄影中非常普遍。

**现有痛点**：
- 3DGS 系列方法在处理散焦输入时性能显著下降，因为散景模糊破坏了几何精度。
- 现有去散焦方法（BAGS、Deblurring 3DGS）使用模糊估计网络但缺乏透镜物理模型，无法实现可控景深——不能重对焦或调整散景效果。
- 基于 NeRF 的方法（DoF-NeRF、LensNeRF）虽有透镜模型，但训练慢、渲染低效。
- 现有评估数据集仅评估去模糊能力，不评估重对焦和相机参数学习精度。

**核心矛盾**：散焦输入需要显式建模弥散圆（Circle of Confusion, CoC）来恢复清晰场景，但理想光学 CoC 与真实 DSLR 相机 CoC 之间存在固有差异，直接建模会引入系统误差。

**本文目标** 如何在 3DGS 框架中高效地从散焦输入恢复清晰场景，同时支持可控景深渲染？

## 方法详解

### 整体框架

DoF-Gaussian 在 Mip-Splatting 基础上构建，整体流程为：(1) 对浅景深输入图像运行 SfM 获取稀疏深度；(2) 用稀疏深度微调单目深度网络得到逐场景深度先验；(3) 3DGS 优化过程中引入可学习透镜参数（光圈 $\mathcal{A}$、对焦距离 $\mathcal{F}$），通过 CUDA 加速的透镜成像算法将清晰渲染结果模拟为散焦图像，与输入图像比较训练；(4) 推理时设光圈为 0 即可得到全清晰图像，或自由调节光圈和对焦距离实现各种景深效果。

### 关键设计

1. **基于几何光学的透镜成像模型**:
    - 功能：将针孔模型替换为薄透镜模型，显式建模弥散圆
    - 核心思路：空间点 $P$ 距透镜距离为 $d$，其在像平面上的弥散圆直径为 $r(d) = \mathcal{A}|1/\mathcal{F} - 1/d|$。当 $d = \mathcal{F}$（在焦面上）时 $r = 0$，成像清晰。用可微分的 tanh 函数替代理想的阶跃 CoC 函数以保证梯度传播，并实现为 CUDA 并行的前向/后向传播算法。
    - 设计动机：有了显式透镜模型，可以学习每张图像的真实光圈和对焦距离，进而精确建模散焦效果并支持推理时可控渲染。

2. **逐场景深度先验调整**:
    - 功能：为散焦输入提供准确的场景几何指导
    - 核心思路：散焦图像直接使用单目深度网络预测的深度不够准确。本文先用 COLMAP 的 SfM 得到稀疏但鲁棒的深度点，然后用 silog 损失 $\mathcal{L}_{silog}$ 微调深度网络使其适配当前场景的尺度和布局，得到逐场景的深度先验 $D_{pred}$，用 L2 损失约束 3DGS 渲染深度。
    - 设计动机：散焦区域的 SfM 点云仍然可靠（因为特征匹配在清晰区域进行），以此为锚点微调深度网络可以显著改善几何精度。

3. **离焦-对焦自适应（Defocus-to-Focus Adaptation）**:
    - 功能：弥补理想 CoC 与真实 CoC 之间的差异
    - 核心思路：训练分两阶段——前 $t$ 步用全图均匀权重建模散焦效果以学习准确的透镜参数；之后用 sigmoid 权重函数 $\Psi(x) = 1/(1+e^{-a(x-b)})$（$x = |1/\mathcal{F} - 1/d|$）重新加权损失，让焦区（$x$ 小的区域）获得更高权重，同时逐像素缩放光圈 $\mathcal{A}' = \mathcal{A} \cdot \Psi$。
    - 设计动机：学到透镜参数后，我们知道哪些区域是清晰的，此时转向重点优化清晰区域可以补偿理想 CoC 无法精确匹配真实散焦的系统偏差。

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \Psi \odot (\mathcal{L}_{rec} + w_d \mathcal{L}_{depth}) + w_n \mathcal{L}_{normal}$

- $\mathcal{L}_{rec} = (1-\lambda)\mathcal{L}_1(I, C^*) + \lambda \mathcal{L}_{D-SSIM}(I, C^*)$：散焦渲染 vs 输入图像
- $\mathcal{L}_{depth} = \|D - D_{pred}\|_2$：渲染深度 vs 逐场景深度先验（$w_d = 0.01$）
- $\mathcal{L}_{normal}$：法线一致性损失（$w_n = 0.05$）
- 训练 30000 迭代，$t = 10000$ 时启动 defocus-to-focus 自适应

## 实验关键数据

### 散焦去模糊（Deblur-NeRF 数据集）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 可控景深 |
|------|-------|-------|--------|----------|
| Deblur-NeRF | 23.47 | 0.720 | 0.121 | ✗ |
| DoF-NeRF | 22.70 | 0.682 | 0.185 | ✓ |
| DP-NeRF | 23.67 | 0.730 | 0.108 | ✗ |
| BAGS | 23.95 | 0.754 | 0.094 | ✗ |
| Deblurring 3DGS | 23.71 | 0.747 | 0.107 | ✗ |
| **DoF-Gaussian** | **23.97** | **0.756** | **0.093** | **✓** |

DoF-Gaussian 在去模糊指标上全面最优或次优，且是唯一基于 3DGS 的可控景深方法。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Baseline (无透镜) | 21.31 | 0.636 | 0.239 |
| + 透镜模型 | 23.05 | 0.728 | 0.109 |
| + 透镜 + 自适应 | 23.59 | 0.742 | 0.104 |
| + 透镜 + 深度先验 | 23.42 | 0.738 | 0.098 |
| **Full model** | **23.97** | **0.756** | **0.093** |

三个组件逐步累加带来 +2.66 dB PSNR 提升，其中透镜模型贡献最大（+1.74 dB）。

### 关键发现

- 在合成数据集上，DoF-Gaussian 的透镜参数学习误差显著低于 DoF-NeRF（$\delta_\mathcal{A}$: 0.068 vs 0.196, $\delta_\mathcal{F}$: 0.079 vs 0.256），验证了透镜参数可被准确学习。
- 在全清晰输入测试中，DoF-Gaussian 性能与 Mip-Splatting 持平甚至略优（PSNR 27.81 vs 27.05），说明透镜模型在一般输入下不会引入退化。
- 逐场景深度先验相比不微调或稀疏深度监督方案分别高 0.44 和 0.53 dB PSNR。

## 亮点与洞察

- **物理驱动的优雅设计**：用几何光学原理替代黑盒去模糊网络，一个透镜公式 $r(d) = \mathcal{A}|1/\mathcal{F} - 1/d|$ 同时解决了去模糊和可控景深两个问题。
- **Defocus-to-Focus 自适应策略巧妙**：承认理想 CoC 与真实 CoC 的差异而非强行拟合，转而利用学到的焦面信息重新分配优化权重，是一种实用的近似方案。
- **丰富的交互应用**：重对焦、光圈调节、CoC 形状变换（圆→五边形/六边形）、动态景深视频等，推理时仅需调参即可，是 3DGS 走向电影级渲染的有意义探索。
- 保持了 3DGS 的实时渲染优势，相比 NeRF-based 方案大幅提升训练和渲染效率。

## 局限与展望

- CUDA 实现的 CoC 模拟在大光圈时计算量显著增加（每个像素需遍历更大邻域）。
- 仅建模理想圆形 CoC + 后处理变形，未直接建模真实镜头的像差和非均匀散焦。
- 深度先验依赖 COLMAP 的 SfM 质量，对纹理稀疏或大面积散焦场景可能退化。
- 当前每张图像独立学习光圈和对焦距离，未建模相机参数在序列中的连续性。
- 未讨论与 HDR/运动模糊等其他非理想条件的联合建模。

## 相关工作与启发

- **vs BAGS**: BAGS 用模糊估计网络建模散焦但无透镜模型，不支持重对焦；DoF-Gaussian 的物理模型既去模糊又支持可控景深。
- **vs DoF-NeRF**: 最相似工作，但基于 NeRF 效率低；DoF-Gaussian 基于 3DGS 大幅提升效率，且透镜参数学习更精确。
- **vs Deblurring 3DGS**: 仅用缩放因子建模模糊，物理建模粗糙；DoF-Gaussian 的透镜模型更精确。
- **启发**：建模真实世界物理成像原理（而非黑盒学习）是处理非理想输入的有效路径，可推广到运动模糊、低光照等场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在3DGS中引入完整透镜模型实现可控景深
- 实验充分度: ⭐⭐⭐⭐ 真实+合成数据集，消融全面，交互应用展示丰富
- 写作质量: ⭐⭐⭐⭐ 物理推导清晰，图示与算法伪代码规范
- 价值: ⭐⭐⭐⭐ 填补了3DGS可控景深的空白，对摄影和电影渲染有直接应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Geometry Field Splatting with Gaussian Surfels](geometry_field_splatting_with_gaussian_surfels.md)
- [\[CVPR 2025\] DepthSplat: Connecting Gaussian Splatting and Depth](depthsplat_connecting_gaussian_splatting_and_depth.md)
- [\[CVPR 2025\] DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds](dashgaussian_optimizing_3d_gaussian_splatting_in_200_seconds.md)
- [\[CVPR 2025\] 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
