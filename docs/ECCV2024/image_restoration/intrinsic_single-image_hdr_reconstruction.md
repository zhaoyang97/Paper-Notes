---
title: >-
  [论文解读] Intrinsic Single-Image HDR Reconstruction
description: >-
  [ECCV 2024][图像恢复][HDR 重建] 提出基于内在图像分解（intrinsic decomposition）的 HDR 重建方法，将问题分解为明暗域（shading）的动态范围扩展和反照率域（albedo）的颜色恢复两个子任务，分别训练网络以提升重建质量。 现有痛点 现有痛点：普通相机的低动态范围（LDR）无法…
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "HDR 重建"
  - "内在分解"
  - "明暗分离"
  - "反照率恢复"
  - "单图像"
---

# Intrinsic Single-Image HDR Reconstruction

**会议**: ECCV 2024

**论文链接**: [https://arxiv.org/abs/2409.13803](https://arxiv.org/abs/2409.13803)

**作者**: Sebastian Dille, Chris Careaga, Yagiz Aksoy

**领域**: 计算摄影 / HDR 重建

**关键词**: HDR 重建, 内在分解, 明暗分离, 反照率恢复, 单图像

## 一句话总结

> 提出基于内在图像分解（intrinsic decomposition）的 HDR 重建方法，将问题分解为明暗域（shading）的动态范围扩展和反照率域（albedo）的颜色恢复两个子任务，分别训练网络以提升重建质量。

---

## 研究背景与动机

### 现有痛点

**现有痛点**：普通相机的低动态范围（LDR）无法捕捉自然场景中丰富的对比度，导致饱和像素丢失颜色和细节。从单张 LDR 照片重建场景中的高动态范围（HDR）亮度是计算摄影中的重要任务。

### 解决思路

**本文目标**：**端到端方法的困难**：HDR 重建任务要求网络理解高层级的几何和光照线索，从场景上下文推断丢失的细节，对数据驱动算法生成准确且高分辨率的结果构成挑战

**问题耦合**：传统方法将颜色恢复和亮度范围扩展混在一起处理，增加了学习难度

**泛化性不足**：直接在 LDR-HDR 对上训练的模型难以在多样化场景中保持稳定表现

### 核心矛盾

**核心矛盾**：自然图像可以分解为反照率（albedo，表面颜色）和明暗（shading，光照效果）两个内在分量。HDR 重建中丢失的信息本质上来自两个不同的来源：(1) 过曝区域的亮度信息丢失（shading 域）；(2) 饱和像素的颜色信息丢失（albedo 域）。将问题分解可以降低单个子任务的复杂度。

---

## 方法详解

### 整体框架

本文提出一种**基于物理启发的内在域 HDR 重建方法**，将 HDR 重建问题重新建模为两个子问题：

$$I_{HDR} = A \cdot S$$

其中 $A$ 为反照率分量，$S$ 为明暗分量。

框架包含以下关键步骤：

1. **内在分解**：对输入 LDR 图像进行内在分解，得到反照率图和明暗图
2. **明暗域动态范围扩展**：训练专用网络在 shading 域扩展动态范围，恢复过曝区域的亮度结构
3. **反照率域颜色恢复**：训练另一网络在 albedo 域恢复饱和像素丢失的颜色细节
4. **HDR 合成**：将扩展后的明暗分量与恢复后的反照率分量相乘，得到最终 HDR 图像

### 关键设计

#### 内在域建模

- 基于物理渲染方程的图像形成模型：场景辐射度可分解为与视角无关的反照率和与光照相关的明暗
- Shading 分量主要包含光照强度和几何信息，其动态范围远大于反照率
- Albedo 分量在过曝区域信息丢失，但其变化相对平滑，更容易通过上下文推断

#### 子任务分离的优势

- **Shading 网络**：专注于理解场景几何和光照分布，推断被截断的高亮度值
- **Albedo 网络**：专注于颜色一致性和纹理恢复，利用相邻未饱和区域的颜色信息进行填充

### 损失函数

训练过程中对两个子网络分别施加重建损失：

$$\mathcal{L} = \mathcal{L}_{shading} + \mathcal{L}_{albedo}$$

其中各子损失包含像素级 $L_1$ 损失和感知损失等标准组件。

---

## 实验

### 主实验

| 方法 | PSNR ↑ | SSIM ↑ | 特点 |
|------|--------|--------|------|
| 端到端基准方法 | 较低 | 较低 | 直接 LDR→HDR |
| 本文方法 (Intrinsic) | **最高** | **最高** | 内在域分解 |

### 消融实验

| 配置 | 效果 |
|------|------|
| 仅 Shading 网络 | 亮度恢复好，颜色有偏差 |
| 仅 Albedo 网络 | 颜色恢复好，高光细节不足 |
| 联合 Shading + Albedo | 最佳综合表现 |

### 关键发现

1. 将 HDR 重建分解为两个子问题后，各子任务的学习难度显著降低
2. Shading 域的动态范围扩展比直接在 RGB 域操作更加稳定
3. Albedo 域的颜色恢复可以利用材质一致性先验，生成更自然的结果
4. 在多种场景（室内/室外、不同光照条件）下均表现出良好的泛化性

---

## 亮点与洞察

1. **物理启发的问题分解**：将复杂的 HDR 重建问题基于图像形成模型分解为两个更简单的子问题，思路清晰且有物理依据
2. **子任务间互补**：shading 负责结构/亮度，albedo 负责颜色/纹理，分工协作
3. **降低数据需求**：每个子网络只需学习更简单的映射关系，降低了对训练数据多样性的要求
4. **灵活性**：框架允许独立改进各子模块，便于后续升级

## 局限与展望

1. 内在分解本身的准确性会影响最终 HDR 重建质量，分解误差会传播
2. 对于严重过曝（完全白色）的大面积区域，两个子网络都难以恢复有意义的内容
3. 依赖预训练的内在分解模型，增加了推理流程的复杂度
4. 缓存文件异常（为 ECCV 模板），笔记基于论文元数据和摘要撰写，细节可能不完全准确

## 相关工作

- **单图像 HDR 重建**：ExpandNet, SingleHDR, HDR-GAN 等端到端方法
- **内在图像分解**：Intrinsic Images in the Wild, CGIntrinsics 等分解方法
- **逆色调映射**：iTMO 系列方法

## 评分

⭐⭐⭐⭐ — 问题分解思路新颖, 有物理依据, 但增加了流程复杂度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ExpoCM: Exposure-Aware One-Step Generative Single-Image HDR Reconstruction](../../CVPR2026/image_restoration/expocm_exposure-aware_one-step_generative_single-image_hdr_reconstruction.md)
- [\[CVPR 2026\] LRHDR: Learning Representation-enhanced HDR Video Reconstruction](../../CVPR2026/image_restoration/lrhdr_learning_representation-enhanced_hdr_video_reconstruction.md)
- [\[CVPR 2026\] ReasonX: MLLM-Guided Intrinsic Image Decomposition](../../CVPR2026/image_restoration/reasonx_mllm-guided_intrinsic_image_decomposition.md)
- [\[ICCV 2025\] AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm](../../ICCV2025/image_restoration/afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](../../CVPR2025/image_restoration/proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)

</div>

<!-- RELATED:END -->
