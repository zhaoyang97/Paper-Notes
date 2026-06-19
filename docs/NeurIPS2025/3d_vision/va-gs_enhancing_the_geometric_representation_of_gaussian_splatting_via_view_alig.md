---
title: >-
  [论文解读] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment
description: >-
  [NeurIPS 2025][3D视觉][3D高斯溅射] 通过引入边缘感知图像监督、可见性感知的多视图光度对齐、法线约束和深度图像特征对齐四种视图对齐（View Alignment）策略，显著提升3D高斯溅射的几何表示精度，在表面重建和新视图合成上取得SOTA。 3D高斯溅射（3DGS）凭借其实时渲染和高质量视图合成能力…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "表面重建"
  - "多视图对齐"
  - "法线一致性"
  - "几何表示"
---

# VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2510.11473](https://arxiv.org/abs/2510.11473)  
**代码**: [GitHub](https://github.com/LeoQLi/VA-GS)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 表面重建, 多视图对齐, 法线一致性, 几何表示

## 一句话总结

通过引入边缘感知图像监督、可见性感知的多视图光度对齐、法线约束和深度图像特征对齐四种视图对齐（View Alignment）策略，显著提升3D高斯溅射的几何表示精度，在表面重建和新视图合成上取得SOTA。

## 研究背景与动机

3D高斯溅射（3DGS）凭借其实时渲染和高质量视图合成能力，迅速成为场景表示的主流方法。然而，3DGS在精确表面重建方面仍然存在显著缺陷。核心矛盾在于：高斯基元的**离散无结构特性**导致仅靠RGB渲染损失进行优化时，几何精度不足，尤其在以下两个场景中问题突出：

**光照引起的伪影**：阴影和高光会扭曲光度损失，导致重建几何偏移

**边界模糊**：物体边缘处的高斯基元方向不确定性导致表面边界不清晰

现有方法如SuGaR通过密度场提取网格但计算昂贵、2DGS用2D面片但在无界场景中失败、GS-Pull引入SDF但表面过于光滑、PGSR用局部平面假设但无法解决光照问题。

本文的核心idea是：将表面重建问题转化为**视图对齐**问题——通过单视图内的边缘和法线对齐，以及多视图间的光度和特征对齐，从多个互补角度约束高斯基元的几何属性，从而在不引入额外隐式表示的前提下实现高精度表面重建。

## 方法详解

### 整体框架

给定一组带位姿的RGB图像，目标是学习一组3D高斯函数及其属性（颜色/不透明度/位置/形状）来表示3D场景几何。方法在标准3DGS的基础上引入五个损失函数：$\mathcal{L}_I$（边缘感知图像重建）、$\mathcal{L}_{nc}$（法线一致性）、$\mathcal{L}_{ns}$（法线平滑）、$\mathcal{L}_p$（多视图光度对齐）和$\mathcal{L}_f$（多视图特征对齐）。

### 关键设计

1. **边缘感知图像重建（$\mathcal{L}_I$）**：在标准L1+D-SSIM损失基础上，增加图像梯度项$\beta_2 L_1(\nabla\tilde{I} - \nabla I)$来监督边缘信息。设计动机是原始颜色损失过度平滑高频区域，添加梯度约束可以保留尖锐结构和边界细节。

2. **边缘感知法线一致性（$\mathcal{L}_{nc}$）**：对齐高斯基元法线$\tilde{N}$和深度图梯度法线$\hat{N}$，使用边缘权重$\delta = (1-\nabla I)^2$在边缘区域降低损失贡献。设计动机是边缘处高斯法线方向模糊，强行对齐会引入错误监督，因此在边缘处降权。

3. **法线平滑约束（$\mathcal{L}_{ns}$）**：约束相邻像素法线差异，使用阈值$\tau$和ReLU门控机制区分真实几何边缘和噪声。解决无纹理区域的法线噪声以及光照变化引入的伪边缘。

4. **可见性感知的多视图光度对齐（$\mathcal{L}_p$）**：借鉴传统MVS方法，通过单应性矩阵$H_{rs}$将参考视图像素投射到源视图，计算NCC光度一致性。关键创新包括：

    - **可见性项**$\upsilon_{rs}$：判断投射点是否在源视图视野内
    - **遮挡权重**$\omega$：通过**重投影误差**$\varphi$过滤被遮挡或几何误差大的像素，$\omega = 1/\exp(\varphi)$（当$\varphi < 1$）

5. **多视图特征对齐（$\mathcal{L}_f$）**：使用预训练网络提取深度图像特征，计算参考视图和源视图对应位置的特征余弦相似度。设计动机是图像级损失对噪声、模糊和光照变化敏感，高维特征空间更鲁棒。

### 损失函数 / 训练策略

最终损失：$\mathcal{L} = \mathcal{L}_I + \lambda_1\mathcal{L}_{nc} + \lambda_2\mathcal{L}_{ns} + \lambda_3\mathcal{L}_p + \lambda_4\mathcal{L}_f$

训练分阶段进行：
- 前7000步：仅用颜色损失预训练，获得粗略几何初始化
- 加入边缘项和法线对齐
- 加入多视图光度对齐（8000步）
- 加入多视图特征对齐（5000步）
- 新视图合成再训练10000步

## 实验关键数据

### 主实验：DTU表面重建（Chamfer距离↓）

| 方法 | Mean CD | 训练时间 |
|------|---------|---------|
| 3DGS | 1.96 | 3.4m |
| 2DGS | 0.80 | 5.8m |
| GS-Pull | 0.75 | 5.6m |
| PGSR | 0.53 | 15m |
| GausSurf | 0.52 | - |
| **VA-GS (Ours)** | **0.49** | 15.5m |

### TNT数据集重建（F1-score↑）

| 方法 | Barn | Truck | Mean |
|------|------|-------|------|
| 3DGS | 0.13 | 0.19 | 0.09 |
| PGSR | 0.66 | 0.66 | 0.52 |
| GausSurf | 0.50 | 0.65 | 0.50 |
| **VA-GS (Ours)** | **0.71** | 0.64 | **0.54** |

### 消融实验（TNT F1-score）

| 配置 | Precision | Recall | F1 | 说明 |
|------|-----------|--------|-----|------|
| 仅$\mathcal{L}_I$ | 0.09 | 0.23 | 0.13 | 无几何约束，最差 |
| 去掉$\mathcal{L}_{nc}+\mathcal{L}_{ns}$ | 0.40 | 0.57 | 0.46 | 法线约束关键 |
| 去掉$\mathcal{L}_p+\mathcal{L}_f$ | 0.33 | 0.40 | 0.36 | 多视图对齐不可或缺 |
| 加scale loss（平面化） | 0.51 | 0.60 | 0.54 | 无额外收益 |
| 完整方法 | 0.51 | 0.60 | **0.54** | 各模块互补 |

### 关键发现

- 将3D高斯压成平面盘（scale loss）对本方法无增益，甚至降低Mip-NeRF 360的渲染质量——说明保留完整3D高斯表示更优
- 多视图源视图数N=3是最佳平衡点，N=4无额外收益但增加计算成本
- 多视图对齐（光度+特征）单独去掉任一影响不大，但全部去掉F1从0.54暴跌至0.36

## 亮点与洞察

1. **不依赖任何外部几何表示**（如SDF、mesh），纯粹通过约束高斯基元本身实现高质量重建
2. 可见性+遮挡权重的设计非常工程化但有效，重投影误差作为遮挡判据简洁优雅
3. 边缘处降权法线损失的策略（$\delta=(1-\nabla I)^2$）直接有效地解决了边界法线歧义

## 局限与展望

- 训练速度较慢（15.5m vs 3DGS的3.4m）——多视图对齐的计算开销
- 法线平滑约束可能在高曲率区域过度平滑
- 深度图像特征依赖预训练模型质量

## 相关工作与启发

- 多视图光度一致性是解决单视图监督不足的经典思路，但关键在于处理遮挡和可见性
- 特征级对齐对光照变化的鲁棒性值得在其他任务（如SLAM、SfM）中借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ 各模块设计扎实，将经典MVS思想与3DGS有机结合
- 实验充分度: ⭐⭐⭐⭐⭐ DTU/TNT/Mip-NeRF360全面覆盖，消融详尽
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，但部分符号较多
- 价值: ⭐⭐⭐⭐ 对3DGS表面重建有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions](sceneforge_enhancing_3d-text_alignment_with_structured_scene_compositions.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](../../CVPR2026/3d_vision/dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)

</div>

<!-- RELATED:END -->
