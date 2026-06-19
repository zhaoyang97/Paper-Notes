---
title: >-
  [论文解读] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video
description: >-
  [CVPR 2025][3D视觉][4D Gaussian Splatting] 提出 4DGC，一个率失真感知的 4D 高斯压缩框架，通过运动感知动态高斯建模（多分辨率运动网格+稀疏补偿高斯）和端到端压缩（可微量化+隐式熵模型），在 3DGStream 基础上实现 16 倍压缩且不损失渲染质量。 领域现状 领域现状：1.…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "4D Gaussian Splatting"
  - "video compression"
  - "rate-distortion"
  - "free-viewpoint video"
  - "streamable"
---

# 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video

**会议**: CVPR 2025  
**arXiv**: [2503.18421](https://arxiv.org/abs/2503.18421)  
**代码**: 暂无  
**领域**: 3D视觉  
**关键词**: 4D Gaussian Splatting, video compression, rate-distortion, free-viewpoint video, streamable

## 一句话总结
提出 4DGC，一个率失真感知的 4D 高斯压缩框架，通过运动感知动态高斯建模（多分辨率运动网格+稀疏补偿高斯）和端到端压缩（可微量化+隐式熵模型），在 3DGStream 基础上实现 16 倍压缩且不损失渲染质量。

## 研究背景与动机

### 领域现状

**领域现状**：1. **领域现状**：3D 高斯溅射（3DGS）能实现高质量的自由视点视频（FVV）渲染，但每帧需要存储大量高斯属性（位置、颜色、协方差等），导致存储和传输成本极高。

2. **现有痛点**：(1) 现有方法将高斯表示和压缩分开处理，忽略率失真权衡。(2) 帧间冗余未被充分利用——相邻帧的高斯属性高度相似。(3) 静态 3DGS 压缩方法无法直接扩展到动态场景。

3. **核心矛盾**：高质量 FVV 渲染需要大量高斯参数，但流式传输要求极低的比特率。需要在表示设计阶段就考虑压缩效率。

4. **本文要解决什么？** 设计一个端到端的 4D 高斯压缩方案，在表示和压缩两个层面同时优化率失真性能。

5. **切入角度**：利用运动网格捕获帧间刚性运动（大部分场景变化），仅对残差部分用稀疏补偿高斯表示，大幅减少需要编码的信息量。

6. **核心idea一句话**：运动网格建模帧间运动 + 稀疏补偿处理新区域 + 端到端率失真优化压缩。

### 解决思路

**本文目标**：### 整体框架
4DGC 包含两个核心模块：(1) 运动感知动态高斯建模——用多分辨率运动网格估计帧间运动，稀疏补偿高斯处理新出现的区域；(2) 端到端压缩——可微量化各属性并用隐式熵模型估计码率，联合优化渲染质量和比特率。


## 方法详解

### 整体框架
4DGC 包含两个核心模块：(1) 运动感知动态高斯建模——用多分辨率运动网格估计帧间运动，稀疏补偿高斯处理新出现的区域；(2) 端到端压缩——可微量化各属性并用隐式熵模型估计码率，联合优化渲染质量和比特率。

### 关键设计

1. **多分辨率运动网格**
    - 做什么：估计帧间的刚性运动（平移+旋转）
    - 核心思路：$\Delta\boldsymbol{\mu}_t = \Phi_{\mu}(\bigcup_{l=1}^L \text{interp}(\mathbf{P}_{t-1}^l, \mathbf{M}_t^l))$
    - 设计动机：运动网格是连续的低维表示，比逐高斯存储运动向量高效得多

2. **稀疏补偿高斯**
    - 做什么：为新出现区域或快速变化区域添加额外高斯
    - 两种触发条件：梯度变化（$|\nabla| > \tau_g$）和快速变换（$|\Delta\mu| > \tau_\mu$）
    - 最终表示：$\hat{\mathbf{G}}_t = \hat{\mathbf{G}}_{t-1}(\cdot) + \Delta\hat{\mathbf{G}}_t$

3. **端到端率失真压缩**
    - 可微量化：直接在训练中量化高斯属性
    - 隐式熵模型：估计每个属性的码率
    - RD 损失：$\mathcal{L} = \mathcal{L}_{render} + \lambda \cdot R$

### 损失函数
- 渲染损失 + λ × 比特率，λ 控制压缩率

## 实验关键数据

### 主实验

| 方法 | 压缩比 | PSNR | 存储 |
|------|--------|------|------|
| 3DGStream | 1× | 基准 | 大 |
| 4DGC | **16×** | ≈基准 | 1/16 |

### 消融实验

| 组件 | 效果 |
|------|------|
| w/o 运动网格 | 压缩效率显著下降 |
| w/o 补偿高斯 | 新区域渲染质量差 |
| w/o 端到端训练 | RD 性能次优 |
| Full | 最佳 RD 权衡 |

### 关键发现
- 运动网格捕获了绝大部分帧间变化，补偿高斯仅占少量
- 隐式熵模型比传统熵编码更适合高斯属性的分布
- 16× 压缩下渲染质量几乎无损

## 亮点与洞察
- **表示与压缩联合优化**：不是先建模再压缩，而是让表示本身就适合压缩
- 运动网格的设计利用了动态场景中"大部分区域是刚性运动"的先验
- 端到端可微框架使得 RD 优化成为可能

## 局限与展望 / 可改进方向
- 非刚性运动（如布料、液体）可能需要更复杂的运动建模
- 流式延迟未充分评估

- 与其他压缩/检测方法的组合可能产生更好效果
- 更大规模数据集（如更长视频序列、更多变化场景）上的评估尚需进一步开展
- 不同应用场景（移动端、服务器端）的部署优化值得探索
- 方法的理论分析可以进一步深入

## 相关工作与启发
- **vs 3DGStream**: 无压缩设计，存储成本是 4DGC 的 16 倍
- **vs Compact3D**: 仅处理静态场景的压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ 表示与压缩联合设计思路好
- 实验充分度: ⭐⭐⭐⭐ RD 曲线对比完整
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰
- 价值: ⭐⭐⭐⭐ FVV 流式传输的核心问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](../../NeurIPS2025/3d_vision/motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)
- [\[CVPR 2025\] GIFStream: 4D Gaussian-based Immersive Video with Feature Stream](gifstream_4d_gaussian-based_immersive_video_with_feature_stream.md)
- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)
- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](../../AAAI2026/3d_vision/streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)

</div>

<!-- RELATED:END -->
