---
title: >-
  [论文解读] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction
description: >-
  [CVPR 2025][自动驾驶][3D占用预测] 提出 GaussianWorld，将 3D 占用预测重新定义为以当前传感器输入为条件的 4D 占用预测问题，通过将场景演化分解为自车运动对齐、动态物体运动和新区域补全三个因素，在 3D 高斯空间中用世界模型显式建模场景变化，在 nuScenes 上不增加额外计算量的前提下将单帧方法的 mIoU 提升超过 2%。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "3D占用预测"
  - "世界模型"
  - "3D高斯表示"
  - "流式推理"
  - "时序融合"
---

# GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction

**会议**: CVPR 2025  
**arXiv**: [2412.10373](https://arxiv.org/abs/2412.10373)  
**代码**: [https://github.com/zuosc19/GaussianWorld](https://github.com/zuosc19/GaussianWorld)  
**领域**: 自动驾驶 / 3D视觉  
**关键词**: 3D占用预测, 世界模型, 3D高斯表示, 流式推理, 时序融合

## 一句话总结

提出 GaussianWorld，将 3D 占用预测重新定义为以当前传感器输入为条件的 4D 占用预测问题，通过将场景演化分解为自车运动对齐、动态物体运动和新区域补全三个因素，在 3D 高斯空间中用世界模型显式建模场景变化，在 nuScenes 上不增加额外计算量的前提下将单帧方法的 mIoU 提升超过 2%。

## 研究背景与动机

**领域现状**：视觉为中心的 3D 占用预测是自动驾驶中描述场景精细结构与语义的关键任务。利用时序信息来增强感知已成为共识，主流方法遵循"感知-变换-融合"的三阶段流程：对每一帧独立提取 BEV 或体积特征，然后根据自车轨迹对齐多帧特征，最后融合得到当前时刻的占用预测。

**现有痛点**：这类方法直接融合多帧表征，完全忽略了驾驶场景演化固有的连续性和简洁性。相邻帧的场景表征本应高度相关——场景变化通常仅来自自车运动和少量动态物体的位移，但直接多帧融合丢失了这一强先验，既增加了模型负担，又降低了效率（需要额外存储和处理多帧特征）。

**核心矛盾**：传统时序建模方法将每一帧视为独立快照来融合，而忽视了帧间变化本身所蕴含的物理规律（静态不动、动态有序移动）。这种"暴力融合"既不高效也不充分。

**本文目标** 如何在不增加计算开销的前提下，充分利用驾驶场景的演化先验来提升 3D 占用预测精度？

**切入角度**：作者观察到驾驶场景的帧间变化可被优雅地分解为三个正交因素：(1) 自车运动引起的全局位移，(2) 动态物体的局部运动，(3) 新进入视野的区域需要感知。这三个因素彼此独立且易于建模。

**核心 idea**：用 3D 高斯作为显式场景表示，通过世界模型在高斯空间中预测场景演化，将占用预测转化为 4D 预测问题，以流式方式实现零额外计算开销的时序增强。

## 方法详解

### 整体框架

GaussianWorld 采用流式推理范式：给定上一帧的 3D 高斯表示 $\mathbf{g}^{T-1}$ 和当前帧 RGB 图像 $\mathbf{x}^T$，模型预测当前帧的 3D 高斯 $\mathbf{g}^T$，再由高斯表示得到占用预测结果。整体流程包括：(1) 将历史高斯根据自车运动做全局对齐；(2) 在新观测区域补充随机初始化的高斯；(3) 用统一的高斯世界层同时建模历史高斯的演化和新高斯的感知；(4) 精炼后的高斯预测当前占用。输入为 6 个环视相机图像，输出为 200×200×16 的语义占用网格。

### 关键设计

1. **场景演化三因素分解**:

    - 功能：将复杂的帧间场景变化拆解为三个独立且可学习的因素
    - 核心思路：以自车为中心的场景演化主要源自三方面——自车运动导致全局坐标偏移、动态物体（车辆、行人等）的局部位移、以及自车前进后新暴露区域的出现。对于 (1)，直接利用自车轨迹的仿射变换矩阵 $\mathbf{M}_{ego}$ 对所有历史高斯的位置做全局变换；对于 (2)，根据语义概率将高斯分为动态集合 $\{g_D\}$ 和静态集合 $\{g_S\}$，仅更新动态高斯的位置；对于 (3)，在新区域均匀采样随机高斯来补全。
    - 设计动机：这种分解引入了强物理先验，大大降低了模型需要学习的变化复杂度——静态物体在对齐后几乎不需要更新，动态物体只需学习局部位移，新区域则独立感知。

2. **统一精炼块（Unified Refinement Block）**:

    - 功能：用统一的网络结构同时处理历史高斯的演化和新高斯的感知
    - 核心思路：将运动层 $M_{ove}$ 和感知层 $P_{er}$ 合并为统一的演化层 $E_{vol}$，二者共享编码器 $E_{nc}$（自编码 + 与 RGB 特征的交叉注意力）和精炼模块 $R_{ef}$，区别仅在于更新哪些属性——新高斯更新全部属性（位置、尺度、旋转、语义、时序特征），历史动态高斯仅更新位置。堆叠 $n_e$ 个演化层和 $n_r$ 个精炼层迭代优化。
    - 设计动机：共享参数使模型保持简洁高效，不增加额外计算量。演化层和精炼层分工明确，前者建模物理运动，后者修正与真实世界的偏差。

3. **3D 高斯场景表示**:

    - 功能：为场景提供显式、连续、可移动的 3D 表示
    - 核心思路：用稀疏的 3D 语义高斯描述场景，每个高斯包含位置 $\mathbf{p}$、尺度 $\mathbf{s}$、旋转 $\mathbf{r}$、语义概率 $\mathbf{c}$ 和时序特征 $\mathbf{f}$。相比 BEV/体素等隐式表示，3D 高斯的显式位置属性使得自车对齐和动态物体运动可以直接通过移动高斯位置实现，无需复杂的特征插值。
    - 设计动机：传统 BEV 特征难以直接建模物体的连续运动，而高斯的显式坐标天然支持仿射变换和局部平移。

### 损失函数 / 训练策略

训练使用交叉熵损失和 Lovász-Softmax 损失的组合。采用渐进式流式训练策略：先在单帧任务上预训练，然后以流式方式微调。训练初期序列较短，逐步延长，并以概率 $p$ 随机丢弃上一帧的高斯（simulating 初始帧），随着训练推进逐渐降低 $p$，使模型逐步适应长序列预测。

## 实验关键数据

### 主实验

在 nuScenes 验证集上的 3D 语义占用预测结果：

| 方法 | 时序帧数 | mIoU | IoU | 延迟 | 显存 |
|------|---------|------|-----|------|------|
| GaussianFormer-B (单帧) | 0 | 19.73 | 30.68 | 225ms | 6958M |
| GaussianFormer-T (融合) | 3 | 20.42 | 31.34 | 382ms | 10019M |
| 3D Gaussian Fusion | 3 | 20.24 | 32.27 | 379ms | 9993M |
| **GaussianWorld (本文)** | **1** | **21.87** | **33.02** | **228ms** | **7030M** |

GaussianWorld 仅用 1 帧历史（流式），mIoU 超过使用 3 帧融合的方法 1.45 个点，且延迟几乎与单帧持平（228ms vs 225ms），显存增量极小（72M）。

### 消融实验

| 配置 | mIoU | IoU | 说明 |
|------|------|-----|------|
| Full model | 21.87 | 33.02 | 三因素全部使用 |
| w/o ego motion | 18.47 | 28.88 | 不做自车对齐，掉 3.4 |
| w/o dynamics | 21.17 | 32.49 | 不建模动态物体运动 |
| w/o completion | × | × | 不补全新区域，训练崩溃 |

### 关键发现

- 自车运动对齐是最关键的因素（去掉掉 3.4 mIoU），这说明全局坐标对齐是时序建模的基础。
- 新区域补全不可或缺，去掉后训练直接崩溃，因为新区域没有高斯表示会导致占用预测出现大面积空白。
- 流式世界模型方案在延迟和显存上几乎无开销（仅多 3ms 和 72M），而传统多帧融合方法需要额外 50%+ 的延迟和 40%+ 的显存。

## 亮点与洞察

- **零额外计算的时序增强**：通过世界模型范式，只需上一帧的高斯表示（而非多帧输入），将时序建模的计算开销压缩到几乎为零，这是一种优雅的效率-性能权衡。
- **物理驱动的分解设计**：三因素分解不仅引入了有效先验（降低学习难度），更巧妙的是将连续的场景变化解耦为全局刚体变换 + 局部运动 + 新增感知，与自动驾驶场景物理高度吻合。
- 世界模型用于感知增强的思路可迁移到 BEV 感知、点云分割等任务——只要场景在时间上连续演化，就可以用"预测+修正"替代"多帧融合"。

## 局限与展望

- 仅使用 1 帧历史，对极度遮挡或剧烈运动的场景可能信息不足，是否可以选择性保留关键帧的高斯？
- 动态/静态分类依赖语义概率的硬划分，对语义歧义的物体（如停着的车）可能误判。
- 目前仅在 nuScenes 一个数据集上验证，对于更复杂的城市场景（如 Waymo）的泛化性有待验证。
- 新区域的随机初始化高斯缺乏几何先验，是否可以利用深度估计来初始化？

## 相关工作与启发

- **vs GaussianFormer**: GaussianFormer 提供了 3D 高斯用于占用预测的基础表示，GaussianWorld 在其基础上增加了世界模型范式和流式推理，将 mIoU 提升 2+%。
- **vs CVT-Occ**: CVT-Occ 做体积特征的时序融合但不考虑帧间关联性，GaussianWorld 显式建模关联性且更高效。
- **vs StreamPETR**: StreamPETR 以物体查询做隐式时序建模，但不适用于密集占用预测；GaussianWorld 的显式高斯表示天然支持密集任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ 世界模型用于感知增强 + 三因素分解是有意义的新范式
- 实验充分度: ⭐⭐⭐⭐ 消融全面，时序对比方法丰富，但仅一个数据集稍显不足
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示精美，公式推导连贯
- 价值: ⭐⭐⭐⭐ 零额外计算的时序增强思路对实际部署有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](../../ICCV2025/autonomous_driving/sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](../../ECCV2024/autonomous_driving/occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](gdfusion_temporal_fusion_occupancy.md)

</div>

<!-- RELATED:END -->
