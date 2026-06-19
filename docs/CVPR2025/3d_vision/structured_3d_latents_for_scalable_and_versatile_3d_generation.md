---
title: >-
  [论文解读] Structured 3D Latents for Scalable and Versatile 3D Generation
description: >-
  [CVPR 2025][3D视觉][3D生成] 提出 Structured LATents (SLat/TRELLIS)，一种统一的 3D 隐空间表示，将稀疏 3D 网格与 DINOv2 多视图特征融合，支持解码为辐射场/3D 高斯/网格等多种格式，在 500K 3D 资产上训练高达 2B 参数的整流流 Transformer，约 10 秒生成高质量 3D 资产并支持灵活局部编辑。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D生成"
  - "结构化隐空间"
  - "稀疏体素"
  - "多格式解码"
  - "整流流"
---

# Structured 3D Latents for Scalable and Versatile 3D Generation

**会议**: CVPR 2025  
**arXiv**: [2412.01506](https://arxiv.org/abs/2412.01506)  
**代码**: [GitHub (TRELLIS)](https://github.com/Microsoft/TRELLIS)  
**领域**: 3D视觉/3D生成  
**关键词**: 3D生成, 结构化隐空间, 稀疏体素, 多格式解码, 整流流

## 一句话总结

提出 Structured LATents (SLat/TRELLIS)，一种统一的 3D 隐空间表示，将稀疏 3D 网格与 DINOv2 多视图特征融合，支持解码为辐射场/3D 高斯/网格等多种格式，在 500K 3D 资产上训练高达 2B 参数的整流流 Transformer，约 10 秒生成高质量 3D 资产并支持灵活局部编辑。

## 研究背景与动机

- **3D 表示的碎片化**：网格、点云、辐射场、3D 高斯各有优劣——辐射场和高斯渲染外观出色但几何提取困难，网格几何精确但外观建模弱。缺乏统一的生成范式。
- **现有隐空间方案的不足**：Triplane 等维度多格式解码困难；基于点云/体素的方案无法同时捕获精细几何和外观；需要昂贵的 3D fitting 预处理。
- **Fitting-free 的需求**：之前方法需要先将 3D 数据预拟合到特定表示（如先拟合 3DGS 再编码），过程耗时且有损。
- **规模化瓶颈**：现有 3D 生成模型大多数百 M 参数、数万训练样本，远不及 2D 生成模型的规模。
- **编辑灵活性**：大多数 3D 生成方法不支持生成后编辑（局部修改、格式灵活切换）。

## 方法详解

### 整体框架

**编码**：对 3D 资产渲染密集多视图图像 → DINOv2 提取特征图 → 投影聚合到稀疏活跃体素上 → Sparse VAE 编码为结构化隐变量 $\boldsymbol{z} = \{(\boldsymbol{z}_i, \boldsymbol{p}_i)\}_{i=1}^L$。**解码**：不同解码器分别将 SLat 解码为 3D 高斯 / 辐射场（Strivec CP 分解）/ 网格（FlexiCubes）。**生成**：两阶段管线——(1) 整流流 Transformer 生成稀疏结构 $\{\boldsymbol{p}_i\}$；(2) 整流流 Transformer 生成局部隐变量 $\{\boldsymbol{z}_i\}$。

### 关键设计

**设计一：DINOv2 视觉特征聚合的稀疏隐空间**
- **功能**：无需 3D fitting 即可编码完整的几何+外观信息
- **核心思路**：在 $64^3$ 3D 网格上定义活跃体素（表面交叉处，$L \ll N^3$，平均 20K），对 3D 资产渲染 150 张多视图图像，用预训练 DINOv2 提取特征图，每个体素投影到多视图特征图上检索对应位置特征并取平均得 $\boldsymbol{f}_i$。活跃体素提供粗结构，DINOv2 特征提供精细几何和外观。
- **设计动机**：DINOv2 已被证明有强 3D 感知能力，直接利用其特征可避免训练专用 3D 编码器；稀疏体素 + 丰富特征 = 结构 + 细节的解耦。

**设计二：多格式解码器间特征共享**
- **功能**：同一隐空间支持解码为高斯/辐射场/网格
- **核心思路**：先用 3DGS 解码器端到端训练编码器+解码器（高保真+高效），冻结编码器后再分别训练其他格式的解码器。所有解码器共享相同的 Transformer 主干结构，仅输出层不同：高斯输出位移+颜色+尺度+不透明度+旋转（每体素 $K$ 个），辐射场输出 CP 分解向量，网格输出 FlexiCubes 参数+SDF 值。
- **设计动机**：证明了用高斯作为代理任务学到的隐空间可迁移到其他表示——SLat 确实是表示无关的。

**设计三：两阶段稀疏整流流 Transformer 生成**
- **功能**：高效生成稀疏结构 + 精细隐变量
- **核心思路**：阶段 1 的结构生成器 $\mathcal{G}_S$：先用 3D 卷积 VAE 将二值活跃网格压缩为低分辨率特征网格，再用 Transformer 去噪（dense grid + CFM loss）。阶段 2 的隐变量生成器 $\mathcal{G}_L$：用稀疏卷积将体素打包 ($2^3$ 区域)，Transformer 去噪后上采样恢复。两阶段均使用 adaLN 注入时间步，cross-attention 注入文本/图像条件。训练到 2B 参数。
- **设计动机**：结构和隐变量解耦让每阶段任务更简单；稀疏卷积利用了 3D 数据的天然稀疏性（$L \ll N^3$）大幅降低计算。

### 损失函数

VAE 训练：3DGS 解码用 $\mathcal{L}_1$ + D-SSIM + LPIPS 渲染损失 + KL 正则。辐射场用类似渲染损失。网格用深度/法线图渲染损失。生成模型用 CFM 目标 $\mathcal{L}_{CFM} = \mathbb{E}\|\boldsymbol{v}_\theta(\boldsymbol{x}, t) - (\boldsymbol{\epsilon} - \boldsymbol{x}_0)\|^2$。

## 实验关键数据

### 重建保真度对比

| 方法 | PSNR ↑ | LPIPS ↓ | CD ↓ | F-score ↑ |
|------|--------|---------|------|----------|
| LN3Diff | 26.44 | 0.076 | 0.0299 | 0.9649 |
| 3DTopia-XL | 25.34 | 0.074 | 0.0128 | 0.9939 |
| CLAY | - | - | 0.0124 | 0.9976 |
| **Ours (SLat)** | **32.74** | **0.025** | **0.0083** | **0.9999** |

### 生成性能约 10 秒/物体

- 在 Toys4k 基准上全面超越 Shap-E、LN3Diff、CLAY、3DTopia-XL 等方法
- 支持文本/图像条件生成
- 支持 tuning-free 局部编辑（删除、添加、替换）

### 关键发现

1. SLat 重建 PSNR 32.74 远超其他隐空间方法（LN3Diff 26.44），证明 DINOv2 特征聚合极其有效
2. F-score 0.9999——几乎无损的几何重建
3. 用高斯训练的编码器可直接迁移到辐射场和网格解码器，验证了 SLat 的表示无关性
4. 500K 数据 + 2B 参数的扩大规模带来了明显的质量提升

## 亮点与洞察

- **Fitting-free 训练**：完全避免了耗时的 3D 预拟合过程，直接从渲染图像编码
- **表示无关的统一隐空间**：首次实现一个隐空间解码到高斯/辐射场/网格三种格式并均保持高质量
- **灵活编辑**支持细节变化（保持结构改变外观）和区域编辑（局部重生成），无需额外训练
- **DINOv2 的 3D 感知能力**被充分证明——降低了 3D 生成对专用 3D 编码器的需求

## 局限与展望

- 两阶段生成存在误差累积——结构生成不佳会影响隐变量生成
- 当前训练数据和条件主要覆盖物体级别，场景级别生成待探索
- 活跃体素分辨率 $64^3$ 可能限制超细节几何的表达

## 相关工作与启发

- SLat 的稀疏体素 + 视觉特征架构可扩展到场景级 3D 生成
- DINOv2 作为通用 3D 编码器的应用潜力有待进一步挖掘
- 整流流在 3D 生成中的成功证明了其在 2D 以外领域的适用性

## 评分

⭐⭐⭐⭐⭐ — 里程碑式的工作，首次实现了高质量、多格式、可编辑的统一 3D 生成。设计优雅（稀疏体素+DINOv2）、规模化（500K 数据/2B 参数）、开源（代码+模型+数据全公开），对 3D 生成社区有深远影响。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Native and Compact Structured Latents for 3D Generation](../../CVPR2026/3d_vision/native_and_compact_structured_latents_for_3d_generation.md)
- [\[CVPR 2025\] ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion](shapeshifter_3d_variations_using_multiscale_and_sparse_point-voxel_diffusion.md)
- [\[CVPR 2025\] SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE](sar3d_autoregressive_3d_object_generation_and_understanding_via_multi-scale_3d_v.md)
- [\[ICCV 2025\] From One to More: Contextual Part Latents for 3D Generation](../../ICCV2025/3d_vision/from_one_to_more_contextual_part_latents_for_3d_generation.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
