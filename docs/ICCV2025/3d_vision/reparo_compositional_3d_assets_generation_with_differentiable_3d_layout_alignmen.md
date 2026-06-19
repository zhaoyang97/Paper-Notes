---
title: >-
  [论文解读] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment
description: >-
  [ICCV 2025][3D视觉][组合式3D生成] 提出REPARO，通过先分别重建单个物体3D网格再利用基于最优传输的可微渲染进行布局对齐，实现从单张图像生成多物体组合式3D资产。 现有image-to-3D生成模型在多物体场景中面临根本性困难： 数据集偏差：3D训练数据大多为居中对齐的单物体，预处理会重新居中输入…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "组合式3D生成"
  - "可微渲染"
  - "最优传输"
  - "布局对齐"
  - "多物体场景"
---

# REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment

**会议**: ICCV 2025  
**arXiv**: [2405.18525](https://arxiv.org/abs/2405.18525)  
**代码**: [项目页](https://reparo-3d.github.io/)  
**领域**: 3D视觉  
**关键词**: 组合式3D生成, 可微渲染, 最优传输, 布局对齐, 多物体场景

## 一句话总结

提出REPARO，通过先分别重建单个物体3D网格再利用基于最优传输的可微渲染进行布局对齐，实现从单张图像生成多物体组合式3D资产。

## 研究背景与动机

现有image-to-3D生成模型在多物体场景中面临根本性困难：

**数据集偏差**：3D训练数据大多为居中对齐的单物体，预处理会重新居中输入，引入固有的位置偏差

**遮挡处理困难**：被遮挡物体被错误表示为融合实体，导致生成的资产错误合并

**整体式网格表示**：输出为单一网格，用户必须通过容易出错的后处理来分割各物体

REPARO的核心思路：分而治之——先利用现有模型的单物体生成优势分别重建，再通过可微渲染优化布局。

## 方法详解

### 整体框架

两阶段流水线：
1. **单物体重建**：从输入图像提取各物体 → 补全被遮挡部分 → 使用off-the-shelf模型生成3D资产
2. **布局对齐**：将所有物体放入统一坐标系 → 通过可微渲染优化空间排列

### 单物体提取与重建

- 使用SAM分割各物体获取二值mask
- 被遮挡物体通过Stable-Diffusion-based inpainting补全
- 调整图像使物体居中（适配模型的中心偏差）
- 使用DreamGaussian或TripoSR生成3D资产

### 基于最优传输的远程外观损失

传统像素级 $L_2$ 损失在渲染图和参考图无重叠区域时梯度为零，优化陷入局部极小。引入最优传输建立全局对应：

代价函数综合RGB颜色、深度和位置：

$$c_{ij} = \alpha \cdot \|I_i - I_j^{ref}\|_2 + \beta \cdot \|D_i - D_j^{ref}\|_2 + \gamma \cdot \|p_i - p_j\|_2$$

通过Sinkhorn散度求解运输矩阵 $T$，建立一对一映射 $\sigma(\cdot)$，损失为：

$$L_a(I, I^{ref}) = \frac{1}{N} \sum_i^N c_{i\sigma(i)}$$

梯度传播：

$$\frac{\partial L_a}{\partial \theta} = \frac{\partial L_a}{\partial I} \cdot \frac{\partial I}{\partial \theta} + \frac{\partial L_a}{\partial D} \cdot \frac{\partial F_D}{\partial I} \cdot \frac{\partial I}{\partial \theta} + \frac{\partial L_a}{\partial p} \cdot \frac{\partial p}{\partial \theta}$$

### 高级语义损失

利用冻结的DINO-v2提取特征，对齐渲染图与参考图的语义关系：

$$L_s(I, I^{ref}) = \frac{1}{K} \sum_i^K \|f_i - f_i^{ref}\|_2$$

### 总损失

$$L(I, I^{ref}) = \lambda L_a(I, I^{ref}) + (1-\lambda) L_s(I, I^{ref})$$

优化参数为各物体的平移 $t$ 和缩放 $s$，不包含旋转（image-to-3D模型已保持朝向一致）。

## 实验

### 主实验 - 组合式3D资产生成

| 方法 | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|-------|--------|
| DreamGaussian | 0.807 | 13.28 | 0.802 | 0.240 |
| TripoSR | 0.795 | 17.25 | 0.863 | 0.218 |
| Wonder3D | 0.801 | 13.69 | 0.807 | 0.238 |
| REPARO♣ | **0.833** | 17.28 | 0.826 | 0.234 |
| REPARO♠ | 0.822 | **17.75** | **0.865** | **0.216** |

REPARO在CLIP得分上显著提升，验证了组合式方法对语义一致性的改善。

### 资源消耗

| 阶段 | 显存 | 时间 |
|------|------|------|
| SAM分割 | 6GB | <1s |
| Inpainting | 8GB | 20s |
| 单物体生成(TripoSR) | 6GB | <1s |
| 布局对齐 | 6GB | 90s |
| **总计(TripoSR)** | **≤8GB** | **120s** |

REPARO在显存≤8GB的约束下完成全流程，具有较好的实用性。

## 亮点与洞察

1. **问题分解策略精巧**：充分利用现有单物体模型的优势，避免了多物体联合生成的固有困难
2. **最优传输解决梯度消失**：OT损失提供远程对应关系，解决了标准L2损失在非重叠区域梯度为零的问题
3. **多模态代价函数**：同时考虑RGB、深度和位置三种信号增强对齐鲁棒性
4. **即插即用**：可结合任意image-to-3D模型使用

## 局限性

- 依赖SAM和inpainting模型的质量，补全效果影响单物体重建
- 对严重遮挡（大面积被遮挡）的物体补全能力有限
- 未优化旋转参数，假设image-to-3D模型输出朝向与输入一致
- Sinkhorn算法在大图像上的计算开销较大

## 相关工作

- DreamFusion, DreamGaussian: 单物体3D生成
- TripoSR, Zero-1-to-3: Image-to-3D方法
- DROT: 最优传输用于可微渲染

## 评分

- 新颖性: ⭐⭐⭐⭐ (OT+可微渲染的布局对齐很新颖)
- 技术深度: ⭐⭐⭐⭐ (损失函数设计精巧)
- 实验充分度: ⭐⭐⭐ (定量实验可更丰富)
- 实用价值: ⭐⭐⭐⭐ (实际多物体场景有需求)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](adahuman_animatable_detailed_3d_human_generation_with_compositional_multiview_di.md)
- [\[ECCV 2024\] ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance](../../ECCV2024/3d_vision/comboverse_compositional_3d_assets_creation_using_spatially-aware_diffusion_guid.md)
- [\[ICCV 2025\] LACONIC: A 3D Layout Adapter for Controllable Image Creation](laconic_a_3d_layout_adapter_for_controllable_image_creation.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)

</div>

<!-- RELATED:END -->
