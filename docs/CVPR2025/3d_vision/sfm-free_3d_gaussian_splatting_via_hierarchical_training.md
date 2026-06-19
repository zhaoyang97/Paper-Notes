---
title: >-
  [论文解读] SfM-Free 3D Gaussian Splatting via Hierarchical Training
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 提出无需SfM预处理的3DGS方法（SFGS），通过层次化训练策略合并多个局部3DGS模型为统一场景表示，并利用视频帧插值改善相机位姿估计，在Tanks and Temples上PSNR提升2.25dB。 标准3DGS依赖SfM预处理来获取相机位姿和稀疏点云进行初始化…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "无SfM重建"
  - "层次化训练"
  - "视频帧插值"
  - "新视角合成"
---

# SfM-Free 3D Gaussian Splatting via Hierarchical Training

**会议**: CVPR 2025  
**arXiv**: [2412.01553](https://arxiv.org/abs/2412.01553)  
**代码**: [GitHub](https://github.com/jibo27/3DGS_Hierarchical_Training)  
**领域**: 3D Vision  
**关键词**: 3D高斯溅射, 无SfM重建, 层次化训练, 视频帧插值, 新视角合成

## 一句话总结

提出无需SfM预处理的3DGS方法（SFGS），通过层次化训练策略合并多个局部3DGS模型为统一场景表示，并利用视频帧插值改善相机位姿估计，在Tanks and Temples上PSNR提升2.25dB。

## 研究背景与动机

标准3DGS依赖SfM预处理来获取相机位姿和稀疏点云进行初始化，但SfM存在以下问题：
- **耗时长**：对大规模场景需要大量时间进行特征匹配和三角化
- **鲁棒性不足**：在重复纹理、无纹理区域或特征提取错误时容易失败
- **不可微分**：限制了端到端学习的可能性

现有的无SfM方法如CF-3DGS通过仿射变换估计相对位姿，但存在两个核心缺陷：
1. **位姿估计精度不足**：大相机运动时，帧间重叠减少导致位姿累积误差增大
2. **高斯覆盖稀疏**：仅用第一帧深度图初始化点云，场景中未覆盖区域缺少高斯，标准自适应密度控制难以在这些稀疏区域进行有效增密

本文核心洞察在于：将复杂场景分段训练、分层合并，并用视频帧插值平滑大运动估计。

## 方法详解

### 整体框架

输入为带有小相机运动的视频序列 $\mathcal{I}=\{I_i\}_{i=1}^N$，方法包含三个阶段：
1. **相机位姿估计**：通过连续帧对的相对位姿堆叠获得全局位姿，利用VFI模型插值中间帧减少大运动的估计误差
2. **层次化训练**：将视频分割为多个重叠片段，每个片段训练一个base 3DGS模型，然后通过重要性评分裁剪+合并策略迭代合并为统一模型
3. **多源监督**：合并后使用原始帧、base模型伪视图和VFI插值帧进行多源训练以减少过拟合

### 关键设计1：视频帧插值辅助位姿估计

- **功能**：解决大相机运动下相对位姿估计不准确的问题
- **核心思路**：利用预训练的视频帧插值模型（如EMA-VFI）在相邻帧 $I_i$ 和 $I_{i+1}$ 之间生成中间帧 $I_{i+0.5}$，将一步大运动分解为两步小运动。相对位姿变为 $T_{i \to i+1} = T_{i \to i+0.5} \odot T_{i+0.5 \to i+1}$
- **设计动机**：大相机运动导致帧间重叠度低，单帧3DGS模型渲染目标帧时会产生大量伪影。插值帧缩小了单步运动幅度，显著减少渲染伪影，提升位姿估计精度。该策略在Tanks and Temples上带来0.35dB的PSNR提升

### 关键设计2：层次化训练与合并策略

- **功能**：解决单帧初始化导致场景远端区域高斯覆盖不足的核心问题
- **核心思路**：设定层次级别 $L$，将视频均匀分为 $2^L$ 个重叠片段。每个片段独立训练一个base 3DGS模型，然后迭代两两合并直到得到统一模型。合并前对每个模型的高斯计算重要性分数（基于渲染梯度敏感度），保留top $\gamma$% 的高斯，再取并集
- **设计动机**：标准自适应密度控制依靠累积梯度来clone/split高斯，但在稀疏区域梯度太小无法触发增密。本方法将合并过程本身视为一种增密：丢弃不重要的高斯，同时从其他模型引入重要高斯填补空白区域。实验中 $L=2$、$\gamma=50\%$ 效果最佳，该策略在Tanks and Temples上带来1.19-1.58dB提升

### 关键设计3：多源监督训练

- **功能**：防止合并后模型在有限训练帧上过拟合
- **核心思路**：合并后使用三类数据训练：(1) 原始训练帧；(2) base 3DGS模型在虚拟中间视角渲染的伪视图（通过SE(3)空间插值获得相机位姿）；(3) VFI生成的插值帧。训练时50%概率选择伪视图或插值帧
- **设计动机**：合并后的3DGS需要精细训练以融合来自不同模型的高斯，但仅用原始帧训练容易过拟合。伪视图和插值帧提供了更丰富的视角监督，提升泛化能力

### 损失函数

标准3DGS光度损失：$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$，适用于所有训练/伪/插值帧。

## 实验关键数据

### 主实验：Tanks and Temples新视角合成

| 方法 | Church | Barn | Museum | Family | Horse | Ballroom | Francis | Ignatius | **Mean PSNR** |
|------|--------|------|--------|--------|-------|----------|---------|----------|---------------|
| BARF | 23.17 | 25.28 | 23.58 | 23.04 | 24.09 | 20.66 | 25.85 | 21.78 | 23.42 |
| Nope-NeRF | 25.17 | 26.35 | 26.77 | 26.01 | 27.64 | 25.33 | 29.48 | 23.96 | 26.34 |
| CF-3DGS | 30.23 | 31.23 | 29.91 | 31.27 | 33.94 | 32.47 | 32.72 | 28.43 | 31.28 |
| **Ours** | **31.34** | **34.95** | **31.59** | **34.71** | **35.82** | **34.12** | **34.09** | **31.64** | **33.53** |

PSNR平均提升 **+2.25dB**，最大增益3.72dB（Barn场景）。

### 消融实验：各组件贡献

| 设置 | PSNR | 提升 |
|------|------|------|
| 基线（无层次/无VFI） | ~31.28 | - |
| + 视频帧插值 | +0.35dB | VFI改善位姿估计 |
| + 层次化训练(L=2) | +1.19~1.58dB | 解决高斯覆盖不足 |
| + 多源监督 | 额外提升 | 减少过拟合 |
| 渐进式 vs 层次化 | 两者均≥1.32dB | 层次化略优 |

### 关键发现

- 在CO3D-V2数据集上平均PSNR提升1.74dB，最佳场景提升3.90dB
- 即使不知道相机内参，方法仍比SOTA高0.89dB PSNR
- 层次化训练生成的高斯分布更均匀，有效覆盖场景各区域

## 亮点与洞察

1. **合并即增密**：将模型合并重新解读为一种高层级的高斯增密策略，直接解决了标准密度控制在稀疏区域失效的问题
2. **VFI的创新应用**：利用2D视频生成模型辅助3D重建中的位姿估计，跨领域方法迁移思路值得借鉴
3. **简单有效的重要性裁剪**：基于渲染梯度的重要性评分简单直观，剪枝+取并集的合并策略避免了复杂的3D高斯对应关系计算

## 局限与展望

- 依赖视频输入假设相邻帧间的运动较小，不适用于图像集合输入
- 层次化训练增加了总训练时间（需训练多个base模型后合并）
- 位姿估计仍通过仿射变换近似，理论上不够严谨
- 未来可结合更强的单目深度估计器或DUSt3R等方法进一步提升

## 相关工作与启发

- **CF-3DGS**：本文的直接基线，通过仿射变换进行无SfM的3DGS训练
- **InstantSplat / COGS**：面向稀疏视图的无SfM方法，与本文面向视频的设定互补
- **3DGS压缩**：重要性评分借鉴了LightGaussian等压缩方法的参数敏感度分析

## 评分

⭐⭐⭐⭐ — 方法设计直觉清晰，层次化训练是解决无SfM高斯覆盖不足的优雅方案，实验提升显著(+2.25dB)。局限在于增加训练开销且限定视频输入。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)
- [\[CVPR 2025\] Hash3D: Training-free Acceleration for 3D Generation](hash3d_training-free_acceleration_for_3d_generation.md)
- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2025\] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
