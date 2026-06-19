---
title: >-
  [论文解读] SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization
description: >-
  [CVPR 2025][自动驾驶][点云处理] 提出 SuperPC，首个将点云补全、上采样、去噪和着色四个任务统一在单一条件扩散模型中的框架，通过三级条件（raw/local/global）和空间混合融合策略（SMF）有效融合图像与点云模态。 - 点云处理涵盖补全、上采样、去噪和着色四大任务，在自动驾驶、3D 重建等领域至…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "点云处理"
  - "扩散模型"
  - "多任务学习"
  - "多模态融合"
  - "三维重建"
---

# SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization

**会议**: CVPR 2025  
**arXiv**: [2503.14558](https://arxiv.org/abs/2503.14558)  
**代码**: [项目主页](https://sairlab.org/superpc/)  
**领域**: 自动驾驶 / 点云处理  
**关键词**: 点云处理, 扩散模型, 多任务学习, 多模态融合, 三维重建

## 一句话总结

提出 SuperPC，首个将点云补全、上采样、去噪和着色四个任务统一在单一条件扩散模型中的框架，通过三级条件（raw/local/global）和空间混合融合策略（SMF）有效融合图像与点云模态。

## 研究背景与动机

- 点云处理涵盖补全、上采样、去噪和着色四大任务，在自动驾驶、3D 重建等领域至关重要
- 现有方法通常独立处理各任务，使用不同模型分别解决，忽略了缺陷之间的相互关联性
- 串联应用独立模型会导致误差累积：例如补全模型的误差传递影响上采样质量，上采样产生的误差模式导致去噪效果不佳
- 已有扩散模型方法仅处理单一任务，且仅保留输入的部分信息（全局 or 局部 or 原始），无法满足四种任务的不同需求
- 多数方法限于单一模态输入（仅点云或仅图像），仅能处理简单对象级点云（如 ShapeNet），无法应对复杂场景级数据
- 需要一个统一模型同时利用图像和点云两种模态的信息，在不同抽象层级引导扩散过程

## 方法详解

### 整体框架

SuperPC 采用条件扩散概率模型（CDPM），以输入的稀疏/不完整/含噪点云和对应图像为条件，通过反向扩散过程生成高质量、稠密、带彩色的点云。框架核心是三级条件（TLC）机制，分别从原始（raw）、局部（local）和全局（global）三个层级提取和融合图像与点云的信息，在扩散的每一步引导噪声预测网络 $\epsilon_\theta$。损失函数为标准噪声预测 MSE 损失扩展至三级条件形式。

### 关键设计

**1. 原始模块与双空间早期融合 (Raw Module & Dual-Spatial Early Fusion)**
- **功能**: 在扩散每一步中保留并融合图像和点云的原始纹理与空间信息
- **核心思路**: 包含两个分支——图像特征投影分支使用 PyTorch3D 的点栅格化将 2D 图像特征投影到部分去噪点云上（$P_t(N,3) \to P_t'(N, 3+C_1)$）；点云特征插值分支使用基于逆距离加权的 k 近邻插值将输入点云特征对齐到目标点云上
- **设计动机**: 通过两种空间操作实现"双空间早期融合"，既保留了图像的稠密颜色/纹理信息，又保留了输入点云的原始几何结构，为扩散过程提供最直接的原始级引导

**2. 局部模块与注意力深度融合 (Local Module & Attention-Based Deep Fusion)**
- **功能**: 提取并融合图像和点云的局部对象级特征，提供细节丰富的局部条件
- **核心思路**: 分别使用 ResNet 和 PointNet++ 作为图像和点云编码器，通过特征金字塔网络（FPN）获取多尺度特征，然后通过跨注意力机制融合两种模态的中间特征表示：$\text{Attention}(Q, K, V) = \text{softmax}(QK/\sqrt{T}) \cdot V$，其中 $Q$ 来自图像特征，$K, V$ 来自点云特征
- **设计动机**: 跨注意力能够对齐图像语义特征与点云空间特征，生成统一的局部融合特征图，捕获多尺度语义信息，弥补原始模块在高层语义表示上的不足

**3. 全局模块 (Global Module)**
- **功能**: 生成一维全局结构潜在码，确保高层语义一致性
- **核心思路**: 通过 PointNet++ 层和最大池化层将局部融合特征图压缩为一维全局潜在码，强调高层特征并降低维度
- **设计动机**: 仅依赖局部或原始条件容易使模型陷入局部最优或过拟合单一任务，全局条件提供整体形状和语义约束，增强模型对复杂场景和多任务的鲁棒性

### 损失函数

采用标准条件扩散噪声预测损失，扩展为三级条件形式：

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, c_{\text{raw}}, c_{\text{local}}, c_{\text{global}}, t)\|^2\right]$$

其中 $c_{\text{raw}}$、$c_{\text{local}}$、$c_{\text{global}}$ 分别为三级条件。

## 实验关键数据

### 主实验：ShapeNet 四任务对比

| 方法 | 补全 DCD↓ | 补全 F1↑ | 上采样 DCD↓ | 上采样 F1↑ | 去噪 DCD↓ | 去噪 F1↑ |
|------|-----------|----------|-------------|------------|-----------|----------|
| AdaPoinTr | 0.462 | 0.423 | ✗ | ✗ | 0.562 | 0.405 |
| GradPU | ✗ | ✗ | 0.298 | 0.589 | 0.533 | 0.412 |
| ScoreDenoise | ✗ | ✗ | 0.346 | 0.537 | 0.291 | 0.812 |
| SOTA 组合 | 0.462 | 0.423 | 0.281 | 0.659 | 0.280 | 0.856 |
| **SuperPC** | **0.387** | **0.557** | 0.293 | 0.631 | 0.285 | 0.837 |

*SuperPC 在补全任务上大幅领先，组合任务 DCD 从 0.509 降至 0.476*

### 场景级基准：KITTI-360

| 方法 | 补全 DCD↓ | 上采样 DCD↓ | 去噪 DCD↓ | 组合 DCD↓ |
|------|-----------|-------------|-----------|-----------|
| SOTA 组合 | 0.649 | 0.597 | 0.369 | 0.725 |
| **SuperPC** | **0.632** | **0.577** | **0.327** | **0.681** |

*在真实场景 KITTI-360 上，SuperPC 在所有任务上均优于 SOTA 独立模型的组合*

### 关键发现

- SuperPC 是唯一能同时执行四个任务的单一模型，其他方法均需多模型串联
- 在 ShapeNet→TartanAir（对象到场景）和 TartanAir→KITTI-360（仿真到真实）的泛化实验中均保持优势
- 早期融合和深度融合单独使用效果不佳，SMF 策略的松耦合组合显著优于两者
- 支持任意实数上采样倍率（如 1.3425×），而非仅限整数倍

## 亮点与洞察

1. **统一思维的价值**: 将四个相关任务统一到一个模型中，利用任务间的相互补充关系，避免了串联误差累积
2. **三级条件设计精巧**: raw/local/global 三个层级分别覆盖像素级细节、物体级语义和全局结构，形成完整的信息层次
3. **新基准贡献**: 设计了对象级和场景级共三个新基准，以及对象到场景、仿真到真实两个泛化评估赛道

## 局限与展望

- 着色任务的定量评估相对有限，论文主要聚焦于前三个任务
- 对于极度稀疏或遮挡严重的场景，两种模态的信息都可能不足
- 未来可探索更多模态（如语义标签）的融入，以及在更大规模场景中的扩展

## 相关工作与启发

- 与 LiDiff 等双任务方法相比，SuperPC 覆盖更多任务且性能更强
- 多级条件的设计思路可推广到其他3D生成任务（如场景生成、形状编辑）
- 空间混合融合策略为多模态3D处理提供了一种灵活有效的融合范式

## 评分

⭐⭐⭐⭐ — 首次将四个核心点云处理任务统一到单一扩散模型，方法设计层次清晰，三级条件机制合理有效。实验覆盖对象级和场景级多个基准，泛化能力值得肯定。但着色评估较薄弱，且场景级提升幅度相对有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Unlocking Generalization Power in LiDAR Point Cloud Registration](unlocking_generalization_power_in_lidar_point_cloud_registration.md)
- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)

</div>

<!-- RELATED:END -->
