---
title: >-
  [论文解读] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting
description: >-
  [ICCV 2025][3D视觉][空间智能] 提出一个可扩展的数据生成管线，通过集成深度估计、相机标定和尺度校准，将单视图2D图像自动转换为包含点云、相机位姿、深度图的尺度真实3D表示，生成了约200万场景的COCO-3D和Objects365-v2-3D数据集，显著提升多种3D任务性能。 空间智能（Spatial Int…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "空间智能"
  - "2D-to-3D"
  - "深度估计"
  - "尺度标定"
  - "点云数据集"
---

# Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting

**会议**: ICCV 2025  
**arXiv**: [2507.18678](https://arxiv.org/abs/2507.18678)  
**代码**: [Project Page](https://ZhangGongjie.github.io/TowardsSSI-page/)  
**领域**: 3D视觉  
**关键词**: 空间智能, 2D-to-3D, 深度估计, 尺度标定, 点云数据集

## 一句话总结

提出一个可扩展的数据生成管线，通过集成深度估计、相机标定和尺度校准，将单视图2D图像自动转换为包含点云、相机位姿、深度图的尺度真实3D表示，生成了约200万场景的COCO-3D和Objects365-v2-3D数据集，显著提升多种3D任务性能。

## 研究背景与动机

空间智能（Spatial Intelligence）——感知、推理和交互3D环境的能力——被视为AI的下一波突破方向。然而，该领域的发展严重受限于大规模3D数据集的稀缺。目前获取3D数据的三条路径各有局限：

**仿真生成**：游戏引擎（如NVIDIA Isaac Gym）可以快速廉价地生成数据，但存在显著的sim-to-real gap，简化的几何和物理模型无法捕捉真实世界的复杂性。

**AI生成3D资产**：当前方法主要限于单物体生成，场景级生成仍具挑战；生成的场景常出现不成比例的元素、卡通化外观和不合理的物体布局。

**传感器采集**：LiDAR和RGB-D相机提供高保真3D数据，但采集和标注成本高昂。现有数据集规模小（ScanNet仅1503个场景）且通常局限于特定领域（室内）。

与此同时，2D图像数据集（COCO、Objects365-v2等）涵盖了海量、多样、标注丰富的图像，但其在推进空间智能方面的潜力尚未被充分挖掘。本文的核心洞察是：**可以利用现有丰富的2D图像数据，通过深度估计和相机参数预测，自动生成高质量的3D训练数据**。

## 方法详解

### 整体框架

数据生成管线分4个步骤：(1) 相对深度估计 → (2) 度量深度估计 → (3) 尺度标定 → (4) 相机参数预测 + 3D投影。核心思想是利用相对深度的细粒度几何精度和度量深度的全局尺度信息进行互补。

### 关键设计

1. **双深度估计与尺度标定**：这是本文方法论的核心创新。

    - **相对深度估计**：使用MoGe模型，先估计3D点云再推导相对深度图 $d_r$。MoGe通过多尺度局部几何损失保证了局部几何精度，但缺乏尺度信息。
    - **度量深度估计**：使用Metric3D v2，将焦距作为输入，端到端预测度量深度 $d_m$。该模型在多种室内外场景上联合训练，减少对单一数据集深度分布的过拟合。
    - **尺度标定**：在有效点集 $\mathcal{V}$ 上计算缩放因子 $s$，将相对深度转换为尺度标定深度 $d_{sc}$：

$$s = \frac{\frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} d_{m,i}}{\frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} d_{r,i}}, \quad d_{sc,i} = s \cdot d_{r,i}$$

   这样得到的深度图同时具备精细的几何细节和正确的全局尺度。

2. **相机参数预测**：对于缺少真值相机参数的野外图像，分两步估计：

    - **内参**：采用WildCamera预测焦距和主点，具备尺度感知和裁剪检测能力
    - **外参**：使用PerspectiveFields推断相机姿态——提供逐像素的上方向向量和纬度值，据此构建旋转矩阵使重建点云与标准3D坐标系（z轴向上）对齐

3. **3D标注生成**：利用尺度标定深度 $d_{sc}$ 和相机参数 $[K, R|T]$，将每个有效像素 $(u_i, v_i)$ 投影到3D空间：

$$\mathbf{P}_i^{\text{cam}} = d_{sc,i} \cdot K^{-1} \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix}, \quad \mathbf{P}_i^{\text{world}} = R \cdot \mathbf{P}_i^{\text{cam}} + T$$

   对分割标注直接投影到3D；对框标注用区域内最大最小深度构建3D框。Objects365-v2仅有框标注时先用SAM生成掩码再投影。

### 损失函数 / 训练策略

本文核心贡献是**数据生成**而非新的训练方法。使用统一超参数设置训练所有模型和数据集，避免针对特定数据集微调超参数带来的偏差。具体下游任务训练时：
- 实例分割：Uni3D + Mask3D
- 语义分割：SpUNet / PTv2 / Uni3D + 2层MLP
- 参考实例分割：TGNN
- QA和稠密描述：LL3DA

## 实验关键数据

### 主实验

| 任务 | 指标 | 仅ScanNet | COCO-3D预训练+ScanNet | 提升 |
|------|------|-----------|----------------------|------|
| 实例分割 | mAP | 24.30% | **28.64%** | +4.34 |
| 语义分割(SpUNet) | mIoU | 31.09% | **62.48%** | +31.39 |
| 语义分割(PTv2) | mIoU | 51.04% | **55.81%** | +4.77 |
| 语义分割(Uni3D) | mIoU | 52.14% | **55.83%** | +3.69 |
| 参考实例分割 | mIoU | 26.10% | **32.47%** | +6.37 |
| 3D QA(ScanQA) | CIDEr | 75.67 | **79.11** | +3.44 |

### 消融实验

| 模型 | 预训练数据 | ScanNet mIoU | ScanNet mAcc | ScanNet allAcc |
|------|-----------|-------------|-------------|---------------|
| SpUNet | 无 | 31.09 | 36.54 | 68.63 |
| SpUNet | COCO-3D | **62.48** (+31.39) | **70.38** (+33.84) | **84.89** (+16.26) |
| PTv2 | 无 | 51.04 | 58.73 | 78.17 |
| PTv2 | COCO-3D | **55.81** (+4.77) | **63.19** (+4.46) | **80.62** (+2.45) |
| Uni3D | 无 | 52.14 | 59.06 | 79.05 |
| Uni3D | COCO-3D | **55.83** (+3.69) | **66.10** (+7.04) | **81.31** (+2.26) |

零样本泛化也表现出色：仅在COCO-3D上训练的模型可直接在ScanNet、S3DIS、Matterport3D和Structured3D上进行推理，如Toilet类mAP超60%。

### 关键发现

- SpUNet在COCO-3D预训练后mIoU提升31.39%，说明合成数据能极大补充真实数据的不足
- 即使合成3D数据仅捕获部分视角点云，仍能有效泛化到ScanNet等完整视角数据集
- 统一超参数设置虽然牺牲了单数据集峰值性能，但更好地反映了数据本身的价值
- 高度分布分析显示：合成数据中物体高度符合真实世界分布（如人类0.5-2.0m），验证了管线的可靠性

## 亮点与洞察

- **方法论清晰优雅**：双深度估计 + 尺度标定的思路简单有效，相对深度给几何细节、度量深度给全局尺度
- **规模空前**：COCO-3D训练集117,183场景，远超ScanNet(1,503)和Structured3D(3,500)
- **统一超参数设置**：所有实验使用相同超参数，避免了刷数据集的嫌疑，更真实地反映合成数据的价值
- **开源数据集贡献大**：COCO-3D和Objects365-v2-3D覆盖300+类别、200万场景，可广泛用于空间智能研究

## 局限与展望

- 单视图重建的点云仅捕获部分视角几何，存在遮挡和不完整问题
- 深度估计在室外大尺度场景（尤其含人类的场景）仍有不足
- 当前仅使用单帧2D图像，若能利用视频的多帧一致性可进一步提升3D重建质量
- 与真实传感器数据的分布差异（domain gap）仍然存在
- 相机参数预测本身存在误差，可能在某些极端场景下导致3D投影失准

## 相关工作与启发

- SpatialRGPT和SpatialBot也尝试从2D生成3D数据，但缺乏精细几何或相机参数，适用范围有限
- 本文的双深度估计互补策略可推广到其他需要同时保证局部精度和全局一致性的3D重建任务
- 实验证明"2D-to-3D lifting"可作为推进空间智能的基础范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 双深度估计+尺度标定思路简洁有效，数据集规模空前
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖实例分割、语义分割、参考分割、QA、稠密描述等多任务，零样本泛化验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，实验设计公正，统一超参数设置立意端正
- 价值: ⭐⭐⭐⭐⭐ 开源的大规模3D数据集将直接推动空间智能领域发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[CVPR 2026\] Lifting Unlabeled Internet-level Data for 3D Scene Understanding](../../CVPR2026/3d_vision/lifting_unlabeled_internet-level_data_for_3d_scene_understanding.md)
- [\[CVPR 2026\] Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images](../../CVPR2026/3d_vision/learning_3d_representations_for_spatial_intelligence_from_unposed_multi-view_ima.md)
- [\[NeurIPS 2025\] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](../../NeurIPS2025/3d_vision/trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)

</div>

<!-- RELATED:END -->
