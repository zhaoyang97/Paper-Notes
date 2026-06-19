---
title: >-
  [论文解读] SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion
description: >-
  [CVPR 2025][遥感][语义场景补全] 首次将卫星影像引入 3D 语义场景补全(SSC)任务，提出双分支框架 SGFormer，通过地面视角引导的卫星特征校正和自适应融合策略，有效解决因视觉遮挡导致的场景补全不完整问题。 3D 语义场景补全(SSC)旨在预测场景中每个体素的占据状态和语义类别，是自动驾驶和机器人导航的…
tags:
  - "CVPR 2025"
  - "遥感"
  - "语义场景补全"
  - "卫星影像融合"
  - "鸟瞰图"
  - "可变形注意力"
  - "自适应权重"
---

# SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion

**会议**: CVPR 2025  
**arXiv**: [2503.16825](https://arxiv.org/abs/2503.16825)  
**代码**: [GitHub](https://github.com/gxytcrc/SGFormer)  
**领域**: 遥感  
**关键词**: 语义场景补全, 卫星影像融合, 鸟瞰图, 可变形注意力, 自适应权重

## 一句话总结

首次将卫星影像引入 3D 语义场景补全(SSC)任务，提出双分支框架 SGFormer，通过地面视角引导的卫星特征校正和自适应融合策略，有效解决因视觉遮挡导致的场景补全不完整问题。

## 研究背景与动机

3D 语义场景补全(SSC)旨在预测场景中每个体素的占据状态和语义类别，是自动驾驶和机器人导航的关键任务。现有基于相机的方法虽有进展，但面临以下根本瓶颈：

- **3D-2D 投影的非唯一对应**：多个 3D 体素对应到 2D 图像的重叠区域，导致语义歧义和径向伪影
- **视觉遮挡问题严重**：地面相机视角有限，被遮挡区域缺乏全局视角信息，难以恢复完整场景
- **远距离区域预测困难**：纯地面方法缺乏 long-range 的全局透视，影响规划和决策
- **深度估计不确定性**：即使有深度信息，也只能处理可见区域

卫星影像具有以下优势：低成本且广泛可用、BEV 视角自然适合城市场景水平布局、覆盖范围广可有效补充地面视角盲区。但将卫星影像引入 SSC 面临对齐和信息质量两大挑战。

## 方法详解

### 整体框架

SGFormer 采用双分支架构：地面分支使用 EfficientNet-B7 提取特征并通过深度引导将 2D 特征变换到 3D 体素空间；卫星分支使用 ResNet-50 提取特征并变换到 BEV 空间。两分支特征通过自适应融合模块合并后，经语义头输出体素级类别预测。

### 关键设计1: 地面视角引导的卫星特征校正 — 解决卫星-地面视角错位

**功能**: 利用地面分支的压缩特征引导卫星分支的特征学习，解决卫星图像定位噪声和俯视遮挡导致的特征错位问题。

**核心思路**: 首先将地面分支的 3D 体素特征 $\mathbf{F}_g^{3D}$ 通过 max pooling 沿 z 轴压缩为 BEV 特征 $\mathbf{F}_g^{BEV}$，然后与可学习的 BEV 查询 $\mathbf{Q}_{bev}$ 组合形成混合特征 $v_{hybrid}$，通过可变形自注意力机制预热 BEV 查询：

$$\mathbf{Q}_{bev} = \text{DA}(\mathbf{Q}_{bev}, \mathbf{p}_{bev}, v_{hybrid})$$

预热后的查询再通过可变形交叉注意力从卫星特征 $\mathbf{F}_s^{2D}$ 中查询信息。

**设计动机**: 卫星图像的定位噪声和俯视遮挡会导致特征级的不一致。通过地面视角信息预热 BEV 查询，使偏移层在交叉注意力中预测更合适的采样偏移，实现更精准的卫星特征提取。

### 关键设计2: 自适应双路径融合模块 — 平衡卫星和地面特征贡献

**功能**: 动态预测通道域和空间域权重，在不同区域和尺度上自适应地平衡两个视角的贡献。

**核心思路**: 设计通道注意力路径和空间注意力路径。通道路径在 3D 空间计算通道权重 $\mathbf{W}_c \in \mathbb{R}^{D}$，空间路径在 BEV 空间计算空间权重 $\mathbf{W}_s \in \mathbb{R}^{H \times W}$。两者与 MLP 输出组合得到最终融合权重：

$$\mathbf{W}_a = \text{MLP}(\mathbf{F}'_c) \oplus C(\mathbf{F}'_c) \oplus S(\mathbf{F}'_c)$$

融合特征为 $\mathbf{F}_f = \mathbf{W}_a \cdot \mathbf{F}'_g + (1 - \mathbf{W}_a) \cdot \mathbf{F}'_s$。

**设计动机**: 卫星视角擅长大范围场景布局（道路、建筑），而地面视角擅长小物体和动态物体的细节。自适应权重允许网络在不同区域和不同语义类别上灵活选择最佳信息源。还引入概率网络识别有价值的体素以增强学习效率。

### 关键设计3: 不确定性引导的特征精炼 — 聚焦高不确定性区域

**功能**: 只对高不确定性体素进行精细化处理，提高效率。

**核心思路**: 先将融合特征投影为粗糙语义预测 $\mathbf{L}_{coarse}$，计算每个体素的熵值，选择 top-k 高熵体素通过可变形交叉注意力从地面特征 $\mathbf{F}_g^{2D}$ 重新采样特征。

**设计动机**: 避免对所有体素做密集精炼操作，将计算资源集中在最需要的区域。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem} + \mathcal{L}_{ce} + \lambda_{bev} \mathcal{L}_{bev} + \lambda_{co} \mathcal{L}_{co}$$

包括加权交叉熵损失、场景类亲和力损失（几何和语义）、BEV 辅助损失和粗糙预测损失。

## 实验关键数据

### 主实验: SemanticKITTI 语义场景补全

| 方法 | IoU (SC) | mIoU (SSC) | 输入 |
|------|----------|------------|------|
| MonoScene | 34.16 | 6.06 | 单目 |
| TPVFormer | 34.25 | 11.26 | 多相机 |
| VoxFormer | 44.15 | 12.35 | 单目+深度 |
| SurroundOcc | 34.72 | 11.86 | 多相机 |
| **SGFormer** | **45.67** | **13.72** | 单目+卫星 |

### SSCBench-KITTI-360 数据集

| 方法 | IoU | mIoU |
|------|-----|------|
| VoxFormer | 42.95 | 12.20 |
| OccFormer | 40.83 | 13.46 |
| **SGFormer** | **44.32** | **14.58** |

### 消融实验

| 设置 | IoU | mIoU |
|------|-----|------|
| 仅地面分支 | 44.15 | 12.35 |
| + 卫星分支（无校正） | 44.78 | 13.01 |
| + 地面引导校正 | 45.12 | 13.39 |
| + 自适应融合 | **45.67** | **13.72** |

### 关键发现

- 加入卫星分支后 IoU 提升约 **1.5 个点**，mIoU 提升约 **1.4 个点**
- 地面引导校正策略对 mIoU 贡献 **0.38 个点**，说明对齐问题确实重要
- 卫星影像在大物体（道路、建筑）类别上提升最明显，在小物体上改善有限
- 自适应融合比简单拼接或固定权重融合效果更好

## 亮点与洞察

1. **首次将卫星影像引入 SSC 任务**，开辟了全新的研究方向，卫星-地面两类正交视角天然互补
2. **地面引导的卫星特征校正**设计巧妙，利用已有地面信息预热 BEV 查询以缓解对齐问题
3. **自适应双路径融合**在通道和空间两个维度权衡两个视角，比固定融合策略更灵活
4. 卫星影像作为低成本、轻量级且广泛可用的补充信息源，实际部署价值很高

## 局限与展望

- 卫星影像通常是预采集的，无法反映动态场景变化（如临时停放的车辆）
- 卫星图像分辨率有限，对小物体（如交通标志、行人）的帮助不大
- GPS 定位噪声导致的卫星图像对齐问题仍需进一步解决
- 仅在 KITTI 系列数据集上验证，更大规模和多样化场景的泛化性有待测试
- 未来可探索卫星影像的时序更新机制

## 相关工作与启发

- **VoxFormer**: 使用深度初始化稀疏 proposal 的 SSC 方法，SGFormer 地面分支基于此
- **BEVFormer**: 利用 Transformer 聚合多视角信息到 BEV 空间的框架
- **SG-BEV / SNAP**: 将卫星图像引入 BEV 分割和 2D 地图构建的先驱工作

## 评分

⭐⭐⭐⭐ — 首次引入卫星影像到 SSC 任务的开创性工作，问题定义清晰，技术路线合理。双分支设计和自适应融合策略有效缓解了跨视角融合的挑战。虽然性能提升幅度有限（mIoU ~1.4），但开辟了一个有前景的新方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Semantic-Adaptive Diffusion for Dynamic Spatiotemporal Fusion](../../CVPR2026/remote_sensing/semantic-adaptive_diffusion_for_dynamic_spatiotemporal_fusion.md)
- [\[CVPR 2025\] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)
- [\[CVPR 2025\] Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes](dense_dispersed_structured_light_for_hyperspectral_3d_imaging_of_dynamic_scenes.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)
- [\[CVPR 2025\] MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting](mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d.md)

</div>

<!-- RELATED:END -->
