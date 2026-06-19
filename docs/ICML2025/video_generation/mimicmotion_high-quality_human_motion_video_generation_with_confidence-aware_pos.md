---
title: >-
  [论文解读] MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance
description: >-
  [ICML 2025][视频生成][pose-guided video generation] 基于 Stable Video Diffusion 构建姿态引导人体视频生成框架，通过将姿态估计置信度编码进引导信号、对高置信手部区域放大训练损失、以及位置感知的渐进式潜变量融合三项设计，在 TikTok 数据集上 FID-VID 达 9.3（前最优 12.4），同时支持任意长度平滑视频生成。
tags:
  - "ICML 2025"
  - "视频生成"
  - "pose-guided video generation"
  - "confidence-aware"
  - "progressive latent fusion"
  - "hand region enhancement"
  - "SVD"
---

# MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance

**会议**: ICML 2025  
**arXiv**: [2406.19680](https://arxiv.org/abs/2406.19680)  
**代码**: [https://tencent.github.io/MimicMotion](https://tencent.github.io/MimicMotion)  
**领域**: 图像/视频生成  
**关键词**: pose-guided video generation, confidence-aware, progressive latent fusion, hand region enhancement, SVD

## 一句话总结

基于 Stable Video Diffusion 构建姿态引导人体视频生成框架，通过将姿态估计置信度编码进引导信号、对高置信手部区域放大训练损失、以及位置感知的渐进式潜变量融合三项设计，在 TikTok 数据集上 FID-VID 达 9.3（前最优 12.4），同时支持任意长度平滑视频生成。

## 研究背景与动机

**领域现状**：姿态引导的人体视频生成（pose-guided human motion video generation）是视频生成的重要子方向。当前方法如 AnimateAnyone、MagicAnimate、MagicPose 等，以参考图像和姿态序列为条件，利用扩散模型生成对应的人体运动视频。基础模型通常建立在 Stable Video Diffusion（SVD）等预训练模型之上。

**现有痛点**：现有方法存在三个核心问题：(1) 手部畸变严重——尤其在大幅运动场景中，手指变形、位置错乱非常常见；(2) 质量与平滑的矛盾——为了保证帧间时序平滑往往牺牲图像细节，导致帧模糊；(3) 姿态估计噪声——DWPose 等检测器在动态场景中的检测精度有限，存在重复检测、遮挡误检等问题，导致训练过拟合噪声样本。

**核心矛盾**：姿态估计的不确定性同时影响了训练和推理两个阶段。训练时噪声姿态导致模型学习到错误的运动映射；推理时不准确的姿态引导直接产生畸变输出。此外，受计算限制，现有方法一次只能生成十几帧，无法原生支持长视频。

**本文目标** (1) 如何在姿态估计不准确时仍保持高质量生成？(2) 如何针对性改善手部等关键区域的生成质量？(3) 如何生成任意长度的平滑视频？

**切入角度**：作者观察到姿态估计器（DWPose）对每个关键点天然输出置信度分数，但以往方法完全忽略了这个信号，仅用固定阈值做简单过滤。如果将置信度直接编码进姿态引导信号，模型就能自动区分可靠与不可靠的姿态，从而在训练和推理阶段都减轻噪声姿态的负面影响。

**核心 idea**：将姿态估计置信度编码为引导信号的亮度，让模型自适应地信任高置信姿态关键点，同时用区域损失放大和渐进式潜变量融合分别解决手部畸变和长视频拼接问题。

## 方法详解

### 整体框架

MimicMotion 基于预训练的 Stable Video Diffusion（SVD）构建。输入包括一张参考图像 $I_{\text{ref}}$ 和一个姿态序列。参考图像走两条路径：通过 CLIP 提取跨注意力特征注入 U-Net 各层，同时通过冻结的 VAE 编码器得到潜变量表示并沿时间维度复制后与视频帧特征在通道维度拼接。姿态序列由多层卷积构成的 PoseNet 提取特征，逐元素加到 U-Net 第一个卷积层的输出上（而非每层都加，以避免干扰预训练模型的时空交互层）。整个去噪过程在潜变量空间进行，最终通过带时间层的 VAE 解码器重建视频帧。

### 关键设计

1. **置信度感知的姿态引导（Confidence-Aware Pose Guidance）**:

    - 功能：将姿态估计的不确定性信息编码进引导信号，使模型能区分可靠与不可靠的姿态
    - 核心思路：不使用传统的固定阈值过滤关键点，而是将每个关键点和肢体连接的颜色值乘以其置信度分数。高置信度的关键点在姿态引导图中颜色更亮（更显著），低置信度的关键点颜色更暗（接近黑色）。这样模型在训练时自动降低对低置信姿态的关注，在推理时也能处理不确定的姿态输入。例如当 DWPose 产生重复检测或遮挡误检时，低置信度会使错误关键点在引导图中几乎不可见
    - 设计动机：姿态估计在动态视频中天然存在不确定性（自遮挡、运动模糊等），置信度分数是现成的但被忽略的信号。通过连续的置信度加权而非二值过滤，保留了信息的渐变性，比硬阈值更鲁棒

2. **手部区域损失放大（Hand Region Enhancement）**:

    - 功能：针对性改善手部生成质量，减少手指畸变
    - 核心思路：基于手部关键点的置信度分数识别可靠的手部区域。当手部所有关键点的置信度都超过阈值时，构建包围手部关键点的矩形边界框，在训练损失中对该区域的损失值放大 10 倍（$w_{\text{hand}}=10$）。这使得模型在训练时偏向关注高质量的手部样本，学习更准确的手部生成
    - 设计动机：手部是视频生成中最容易畸变的区域（手指细节多、自由度高），同时也是人类观看时最关注的区域。通过与置信度结合，确保只对高质量手部区域加权，避免强化错误样本

3. **渐进式潜变量融合（Progressive Latent Fusion）**:

    - 功能：生成任意长度的平滑视频，消除段边界的闪烁和突变
    - 核心思路：将长姿态序列分割为固定长度 $N$ 的视频段，相邻段之间有 $C$ 帧重叠（$C \ll N$）。在每个去噪步中，各段独立去噪后对重叠区域进行位置感知的加权融合。融合权重 $\lambda_{\text{fusion}} = 1/(C+1)$，靠近当前段中心的帧权重更高，靠近段边界的帧权重更低。具体对于第 $i$ 段的第 $j$ 帧（$j \leq C$），融合公式为 $\mathbf{z}_i^j \leftarrow j\lambda_{\text{fusion}}\mathbf{z}_i^j + (1 - j\lambda_{\text{fusion}})\mathbf{z}_{i-1}^{N-C+j}$。该方法是 training-free 的，仅在推理阶段应用
    - 设计动机：直接的 MultiDiffusion 式平均融合对所有重叠帧权重相同，导致段边界处出现突变（背景先清晰→模糊→再清晰）。渐进式融合通过位置感知的权重渐变确保段间平滑过渡

### 损失函数

基础损失沿用扩散模型的标准 MSE 去噪损失 $\mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t; \mathbf{c}, t)\|_2^2]$。在此基础上，手部区域损失放大将可靠手部区域内的损失值乘以放大系数 10，使得该区域对梯度更新的贡献更大。训练在 8 块 A100 GPU 上进行 20 个 epoch，学习率 $10^{-5}$，每个 clip 16 帧，收集了 4,436 个舞蹈视频作为训练数据。

## 实验关键数据

### 主实验表格：与现有方法在 TikTok 测试集上的定量对比

| 方法 | FID-VID↓ | FVD↓ | SSIM↑ | PSNR↑ |
|------|----------|------|-------|-------|
| MagicAnimate | 16.2 | 848 | 0.740 | 17.5 |
| MagicPose | 13.3 | 916 | 0.776 | 18.8 |
| Moore-AnimateAnyone | 12.4 | 728 | 0.758 | 18.7 |
| MuseV | 14.6 | 754 | 0.766 | 17.6 |
| **MimicMotion (本文)** | **9.3** | **594** | **0.795** | **20.1** |

MimicMotion 在所有四个指标上全面领先，FID-VID 从 12.4 降至 9.3（降幅 25%），FVD 从 728 降至 594（降幅 18%）。

### 消融实验表格：各组件的贡献

| 手部增强 | 置信度感知 | 渐进融合 | FID-VID↓ | FVD↓ | SSIM↑ | PSNR↑ |
|---------|----------|---------|----------|------|-------|-------|
| ✗ | ✗ | ✗ | 14.6 | 776 | 0.760 | 18.0 |
| ✓ | ✗ | ✗ | 15.0 | 678 | 0.758 | 17.9 |
| ✓ | ✓ | ✗ | 12.2 | 623 | 0.787 | 18.4 |
| ✓ | ✓ | ✓ | **9.3** | **594** | **0.795** | **20.1** |

### 关键发现

- 置信度感知姿态引导的加入使 FVD 从 678 降至 623，证实其对时序平滑的核心作用
- 手部区域增强单独使用效果有限（FID-VID 甚至微增至 15.0），但与置信度感知结合后效果显著（FID-VID 从 14.6 降至 12.2），说明两者存在协同效应——只有置信度可靠时才应放大手部损失
- 渐进式融合在 FVD 上进一步从 623 降至 594，同时 PSNR 从 18.4 跃升至 20.1，证明其对长视频时序连贯性和帧质量的双重提升
- 用户研究中 75.5%–100% 的参与者偏好 MimicMotion 的结果，即使对比图像质量较高的 MuseV 仍有 75.5% 偏好度

## 亮点与洞察

- 置信度信号的利用非常巧妙：姿态估计器天然输出置信度，但此前无人将其编码进引导信号，属于"免费的午餐"
- 手部区域损失放大设计简单工程友好，仅需一个置信度阈值和放大系数两个超参数
- 渐进式融合是 training-free 的推理策略，不影响模型训练，可即插即用
- 三个贡献精确对应三个实际痛点（姿态噪声→置信度感知、手部畸变→区域增强、长视频→渐进融合），问题驱动而非技术驱动

## 局限性

- 依赖 DWPose 姿态检测器，非人形角色或极端动作场景效果受限
- 仅支持 2D 姿态骨架引导，不支持 SMPL/DensePose 等 3D 参数化表示
- 训练数据仅 4,436 个舞蹈视频，场景多样性受限
- 渐进融合在相邻段风格大幅变化时可能仍存在过渡伪影

## 相关工作与启发

- **vs AnimateAnyone/MagicAnimate**：这些方法忽略姿态估计的不确定性，MimicMotion 的置信度感知设计是关键差异化因素
- **vs MultiDiffusion/Lumiere**：MultiDiffusion 用等权平均融合重叠帧，Lumiere 沿用此策略仍有段边界突变；MimicMotion 的位置感知渐进融合根本性解决了这一问题
- **启发**：将检测器副产品（置信度）作为条件信号的思路可推广到其他条件生成任务（如深度引导、分割引导等）

## 评分

⭐⭐⭐⭐ 问题驱动、设计精准、工程实用，每个组件都有清晰的消融验证。置信度感知引导是简单但有效的创新。受限于仅在舞蹈视频场景验证，泛化性有待证明。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](../../CVPR2025/video_generation/osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[NeurIPS 2025\] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](../../NeurIPS2025/video_generation/foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)
- [\[CVPR 2026\] 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation](../../CVPR2026/video_generation/3d-aware_implicit_motion_control_for_view-adaptive_human_video_generation.md)
- [\[CVPR 2025\] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion](../../CVPR2025/video_generation/posetraj_pose-aware_trajectory_control_in_video_diffusion.md)

</div>

<!-- RELATED:END -->
