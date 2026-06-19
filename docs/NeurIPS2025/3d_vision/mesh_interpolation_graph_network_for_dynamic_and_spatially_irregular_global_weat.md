---
title: >-
  [论文解读] Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting
description: >-
  [NeurIPS 2025][3D视觉][图神经网络] 提出 MIGN 框架，通过网格插值策略将不规则气象站数据映射到规则 HEALPix 网格上进行消息传递，并引入参数化球谐函数位置编码增强空间泛化能力，在全球天气预报任务中显著超越现有方法。 现有数据驱动的天气预报模型主要聚焦于 规则网格数据：（如 ERA5）…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "图神经网络"
  - "全球天气预报"
  - "网格插值"
  - "球谐函数"
  - "空间泛化"
---

# Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting

**会议**: NeurIPS 2025  
**arXiv**: [2509.20911](https://arxiv.org/abs/2509.20911)  
**代码**: [有](https://github.com/compasszzn/MIGN)  
**领域**: 3D Vision / 时空预测  
**关键词**: 图神经网络, 全球天气预报, 网格插值, 球谐函数, 空间泛化

## 一句话总结

提出 MIGN 框架，通过网格插值策略将不规则气象站数据映射到规则 HEALPix 网格上进行消息传递，并引入参数化球谐函数位置编码增强空间泛化能力，在全球天气预报任务中显著超越现有方法。

## 研究背景与动机

现有数据驱动的天气预报模型主要聚焦于 **规则网格数据**（如 ERA5），如 FourCastNet、Pangu、GraphCast 等。然而，实际气象站观测数据具有两个核心挑战：

**空间不规则性**：全球气象站分布极不均匀，大多数集中在北美和西欧，不同经纬度密度差异巨大。现有 GNN 方法在同时学习高密度和低密度区域时面临困难。

**动态分布**：气象站的数量和位置随时间变化（新建/废弃），使用固定站点集训练的模型容易过拟合，无法对未见位置进行预测。

现有 GNN 方法（如 DyGrAE、STAR 等）大多只关注区域性预报（如欧洲、北美），忽略了全球天气系统的耦合性，且通常假设站点集固定不变。**本文首次系统研究了空间不规则且动态变化的全球站点天气预报问题**。

## 方法详解

### 整体框架

MIGN 采用 **编码器-处理器-解码器** 架构：
- **编码器**：将不规则站点信息通过消息传递映射到规则 HEALPix 网格节点上
- **处理器**：在规则网格上进行标准 GNN 消息传递
- **解码器**：将网格上的预测结果反向映射回目标站点

### 关键设计

1. **网格插值策略（Mesh Interpolation）**

   **功能**：将空间不规则的气象站数据对齐到规则的球面网格上。

   **核心思路**：使用 HEALPix 将球面均匀划分为等面积像素，构建层级细化的均匀网格。编码器通过消息传递从站点节点向最近的网格节点聚合信息：

    $\mathbf{h}_{v_h}^{(E)} = \text{AGG}^{(E)}(\{\mathbf{m}_{v_s \to v_h}^{(E)} : v_s \in \mathcal{N}(v_h)\})$

   其中消息由站点节点的隐藏状态经 MLP 生成。网格节点特征初始化为零，通过消息传递实现"插值"。

   **设计动机**：在规则网格上，空间邻接关系清晰，每个节点位置固定，可直接使用标准 GNN/CNN/Transformer。这避免了不规则图上因密度差异导致的学习不平衡问题。

2. **球谐函数位置编码（Spherical Harmonics Location Embedding）**

   **功能**：为球面上的每个位置提供参数化的连续位置编码。

   **核心思路**：将位置信息建模为球面函数 $f(\lambda, \phi)$，用实球谐函数基展开：

    $SH(\lambda, \phi) = \bigoplus_{n \geq 0} \bigoplus_{m=-n}^{n} w_n^m Y_n^m(\lambda, \phi)$

   其中 $w_n^m$ 为可学习系数，在所有节点间共享。最终节点表示为 $\mathbf{h} = [x; SH(\lambda, \phi)]$。

   **设计动机**：非参数化位置编码（如直接使用坐标）提供的位置信息有限，难以泛化到未见区域。球谐函数天然适合球面数据分析，不同阶次捕获不同空间尺度的变化模式。且不同天气变量在同一区域的变化模式不同，因此为每种变量学习独立的位置编码。

### 损失函数 / 训练策略

采用 MSE 损失：$\mathcal{L}_{train} = \sum_{s \in \mathcal{D}_{train}} \|\hat{Y}_s^{t+1} - Y_s^{t+1}\|^2$

使用 Adam 优化器，batch size=4，隐藏维度 64，学习率 0.001，2 层 GNN。网格细化级别为 3（768 个网格节点），10-近邻构图，球谐函数最高阶数 N=2。

## 实验关键数据

### 主实验

在 NOAA GSOD 全球观测数据集上评估 6 个气象变量，对比 13 个时空基线模型。

| 变量 | 指标 | MIGN | 最强基线 | 提升 |
|------|------|------|----------|------|
| MAX TEMP | MSE | 8.47 | 9.74 (STGCN) | 13% |
| MIN TEMP | MSE | 8.01 | 9.44 (STGCN) | 15% |
| DEWP | MSE | 7.92 | 9.25 (STGCN/DyGrAE) | 15% |
| SLP | MSE | 20.09 | 23.83 (DualCast) | 15% |
| WDSP | MSE | 8.38 | 8.60 (STGCN) | 3% |
| MXSPD | MSE | 19.73 | 20.25 (HD-TTS) | 3% |

### 消融实验

| 配置 | 说明 |
|------|------|
| 完整 MIGN | 最佳性能 |
| 去除球谐函数编码 | 性能显著下降，验证位置编码的重要性 |
| 去除网格插值 | 性能下降，验证规则化对齐的必要性 |
| 多步预测（3天输入→4天输出） | MIGN 在所有步骤均超越基线 |

### 关键发现

- **空间泛化**：随机选取一半站点训练，在未见站点上测试。DyGrAE 和 STAR 在欧洲和北美区域 MAE 较高，而 MIGN 在这些区域误差显著更低，展现出对未见位置的强泛化能力。
- **区域鲁棒性**：多数方法难以同时在密集和稀疏观测区域表现良好，MIGN 在不同区域模式下均产生更稳定的结果。

## 亮点与洞察

- 将传统地球科学中的网格插值思想与 GNN 消息传递巧妙结合，是一种**物理先验驱动的网络设计**
- 球谐函数编码是连续的、可微的，且天然适合球面几何，比简单的坐标编码优雅得多
- 框架的灵活性很强：编码器/处理器/解码器中的 GNN 可替换为任意现有方法

## 局限与展望

- 目前仅预测单一时间步（次日），更长期的预报需要进一步研究
- HEALPix 网格的分辨率是固定的，可能需要自适应多分辨率网格
- 未考虑站点间的时间序列依赖（如使用 RNN/Transformer 处理时间维度）
- 与 ERA5 等再分析数据的融合尚未探索

## 相关工作与启发

- GraphCast 也使用了网格结构，但针对规则网格数据且边是静态的；MIGN 的核心贡献在于处理不规则动态站点数据
- 球谐函数在地球科学中有广泛应用（磁场、重力场），本文将其引入为 GNN 的可学习位置编码，这一思路可迁移到其他球面数据任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 网格插值+球谐编码的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 13个基线对比、泛化实验、消融实验均充分
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 对实际天气站网络预报有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] UGM2N: An Unsupervised and Generalizable Mesh Movement Network via M-Uniform Loss](ugm2n_an_unsupervised_and_generalizable_mesh_movement_network_via_m-uniform_loss.md)
- [\[ICCV 2025\] Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion](../../ICCV2025/3d_vision/global_motion_corresponder_for_3d_point-based_scene_interpolation_under_large_mo.md)
- [\[CVPR 2026\] Global-Aware Edge Prioritization for Pose Graph Initialization](../../CVPR2026/3d_vision/global-aware_edge_prioritization_for_pose_graph_initialization.md)
- [\[ICCV 2025\] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection](../../ICCV2025/3d_vision/ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)
- [\[CVPR 2026\] Point4Cast: Streaming Dynamic Scene Reconstruction and Forecasting](../../CVPR2026/3d_vision/point4cast_streaming_dynamic_scene_reconstruction_and_forecasting.md)

</div>

<!-- RELATED:END -->
