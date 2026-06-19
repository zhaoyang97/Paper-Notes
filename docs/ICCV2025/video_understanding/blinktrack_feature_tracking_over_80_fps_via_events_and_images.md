---
title: >-
  [论文解读] BlinkTrack: Feature Tracking over 80 FPS via Events and Images
description: >-
  [ICCV 2025][视频理解][事件相机] 提出 BlinkTrack，将可微卡尔曼滤波引入学习框架，有效解决事件相机和传统相机异步数据的关联与不确定性感知融合，实现超过 80 FPS 的高帧率特征跟踪，并在遮挡场景中显著优于现有方法。 特征跟踪是计算机视觉的基石任务（SfM、SLAM、物体跟踪等）…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "事件相机"
  - "特征跟踪"
  - "卡尔曼滤波"
  - "多模态融合"
  - "高帧率跟踪"
---

# BlinkTrack: Feature Tracking over 80 FPS via Events and Images

**会议**: ICCV 2025  
**arXiv**: [2409.17981](https://arxiv.org/abs/2409.17981)  
**代码**: [GitHub](https://github.com/ColieShen/BlinkTrack)  
**领域**: 视频理解  
**关键词**: 事件相机, 特征跟踪, 卡尔曼滤波, 多模态融合, 高帧率跟踪

## 一句话总结

提出 BlinkTrack，将可微卡尔曼滤波引入学习框架，有效解决事件相机和传统相机异步数据的关联与不确定性感知融合，实现超过 80 FPS 的高帧率特征跟踪，并在遮挡场景中显著优于现有方法。

## 研究背景与动机

特征跟踪是计算机视觉的基石任务（SfM、SLAM、物体跟踪等）。事件相机以极高时间分辨率异步检测场景变化，适合高频跟踪和极端光照条件，但存在关键局限：**无法捕获精细纹理信息**，在慢速运动场景中空间信号稀疏，导致误差累积和跟踪丢失。

融合事件数据和传统图像帧是一个有吸引力的方向，但面临三大挑战：

**数据关联**：事件相机异步检测变化（如 100Hz），传统相机固定帧率（如 30Hz），两者不同步

**不确定性感知融合**：两种相机的不确定性分布不同——事件相机在低纹理区域不可靠，传统相机在运动模糊/极端光照下不可靠，需自适应加权

**运行效率**：高帧率跟踪要求最小计算开销，注意力机制等方法虽有效但太重

**核心洞察**：卡尔曼滤波天然支持异步数据融合和不确定性建模，但传统方法需手动调参。BlinkTrack 将其扩展为可学习框架——网络学习预测不确定性，卡尔曼滤波根据不确定性自动加权融合，实现端到端训练。

## 方法详解

### 整体框架

BlinkTrack 包含三个核心组件：
- **事件模块**：从事件数据生成初始跟踪预测和不确定性
- **图像模块**：从图像帧生成重定位预测和不确定性
- **卡尔曼滤波器**：接收两个模块的异步预测，基于时间戳和不确定性进行最优融合

### 关键设计

1. **事件模块**：采用参考图像 patch 与事件 patch 匹配的范式：

    - **金字塔特征编码器**：两个浅层 U-Net 分别编码参考 patch $P_{evt_{ref}}$（从灰度图裁剪）和事件 patch $P_{evt_j}$（SBT-Max 预处理），构建 2 层特征金字塔增强多尺度感知，计算相关图 $C_{evt_j}$
    - **双 LSTM 位移预测器**：特征 LSTM（ConvLSTM）传播历史特征信息，位移 LSTM 聚合位移特征并通过门控机制生成最终位移 $\Delta\hat{p}$
    - **不确定性预测器**：5 层 CNN 将特征映射为 2 类得分（certain/uncertain），Softmax 归一化到 [0,1]，再通过抛物线函数映射到 [0,10]，放入协方差矩阵 $\hat{\Sigma}_{evt_j} \in \mathbb{R}^{2 \times 2}$

2. **图像模块**：轻量级设计（>50 FPS），灵感来自 PIPs：

    - **金字塔编码器**：类 RAFT 结构将全帧编码为 1/8 分辨率的特征图 $F_{img}$
    - **多尺度相关金字塔**：4 级平均池化 → 与参考特征相关 → 采样 7×7 patch 拼接
    - **MLP-Mixer 预测头**：聚合参考特征、目标特征、相关图、位置编码预测位移和不确定性
    - 支持**迭代精炼**：用预测位置重新采样特征和相关图，逐步逼近真实位置

3. **可微卡尔曼滤波器**：采用匀速运动模型，状态 $x = (x, y, v_x, v_y)^T$：

    - **预测步**：基于运动模型和时间间隔传播状态和不确定性
    - **更新步**：接收任一模块的观测 $(\Delta\hat{p}, \hat{\Sigma})$，通过卡尔曼增益 $K_k = P_{k|k-1} H^T S_k^{-1}$ 融合
    - 高不确定性时更依赖内部匀速模型；低不确定性时更依赖当前观测
    - 全过程可微，不确定性通过 ground truth 间接监督

### 损失函数 / 训练策略

分阶段训练事件和图像模块：
- **事件模块训练**：先不用不确定性预测器和卡尔曼滤波训练编码器和位移预测器（$\mathcal{L}_{\hat{disp}}$），再冻结编码器和位移预测器，训练不确定性预测器（$\mathcal{L}_{\tilde{disp}} + w_1 \mathcal{L}_{vis}$）
- **图像模块训练**：从一开始就启用卡尔曼滤波和不确定性训练（$\mathcal{L}_{image} = \mathcal{L}_{\tilde{disp}} + w_2 \mathcal{L}_{\hat{disp}} + w_3 \mathcal{L}_{uncert}$）
- 可见性损失 $\mathcal{L}_{vis} = \text{CrossEntropy}(1-\hat{\sigma}, g)$ 作为不确定性的代理监督
- 训练数据为自建 MultiTrack 数据集（高帧率合成 + 遮挡）

数据集贡献（MultiTrack, EC-occ, EDS-occ）：合成管线可生成彩色图像、事件数据、遮挡轨迹和可见性标签，通过 FILM 插帧 + DVS-Voltmeter 合成事件。

## 实验关键数据

### 主实验（表格）

**EC 和 EDS 数据集（Feature Age / Expected FA）**：

| 方法 | 数据 | EC FA↑ | EC Exp FA↑ | EDS FA↑ | EDS Exp FA↑ |
|------|------|--------|-----------|---------|------------|
| EKLT | E | 0.811 | 0.775 | 0.325 | 0.205 |
| Deep-EV-Tracker | E | 0.795 | 0.787 | 0.549 | 0.451 |
| Ours (E) | E | 0.833 | 0.819 | 0.568 | 0.474 |
| D-Tracker + KLT | E+I | 0.735 | 0.730 | 0.594 | 0.503 |
| Ours (E+I w. K) | E+I | **0.851** | **0.845** | **0.653** | **0.550** |

### 消融实验（表格）

| 组件 | EC Exp FA↑ | EC-occ↑ | EDS Exp FA↑ | EDS-occ↑ | 参数量 |
|------|-----------|---------|------------|---------|-------|
| Full (金字塔+双LSTM) | **0.819** | **0.522** | **0.474** | **0.343** | 33.1M |
| 无金字塔 | 0.789 | 0.439 | 0.395 | 0.283 | 29.3M |
| 仅特征LSTM | 0.794 | 0.473 | 0.428 | 0.337 | 32.7M |
| 仅位移LSTM | 0.617 | 0.332 | 0.282 | 0.186 | 31.9M |
| 无LSTM | 0.568 | 0.338 | 0.294 | 0.225 | 31.6M |

遮挡数据实验：卡尔曼滤波在遮挡场景提升最显著——E+I+K 在 EC-occ 遮挡点上 δ_occ 从 11.4 提升到 28.5（+149%）。

### 关键发现

- 事件模块单独已优于 Deep-EV-Tracker，金字塔特征编码器贡献显著
- 朴素融合（替换初始点）甚至可能降低性能，而卡尔曼融合带来稳定大幅提升
- 运行效率：事件模块 <9ms/帧，卡尔曼滤波 <1ms，整体 >80 FPS（多模态）、>100 FPS（仅预处理事件）
- 不确定性可视化表明模型在遮挡时正确输出高不确定性
- 在极端光照（DSEC 数据集）下，CoTracker 完全失败而 BlinkTrack 仍可运行
- MultiTrack 数据集比 MultiFlow 提供更大位移范围和遮挡，更好释放学习潜力

## 亮点与洞察

- 将经典卡尔曼滤波与深度学习优雅结合：网络负责预测不确定性，滤波器负责最优融合
- 架构设计充分考虑效率：事件模块用 patch 级特征避免全帧编码；图像模块用轻量编码器实现 >50 FPS 重定位
- 两个模块通过卡尔曼滤波低耦合连接，支持模块化替换和异步推理
- 自建数据集填补了事件相机特征跟踪在遮挡标注方面的空白

## 局限与展望

- 网络容量和感受野可能限制当前性能上限
- MultiTrack 训练数据未建模运动模糊，在 EDS 上图像模块+卡尔曼略有性能损失
- 匀速运动假设较简化，扩展卡尔曼滤波（EKF）可能处理更复杂运动
- 两个模块分开训练，联合训练可能进一步提升（但需更多计算资源）

## 相关工作与启发

- 相比 Deep-EV-Tracker 的朴素融合（替换初始点），卡尔曼融合方法优势明显
- 相比 FF-KDT 将事件对齐到图像帧率再融合，BlinkTrack 保持事件的原生高时间分辨率
- 可微卡尔曼滤波的思路可推广到其他异步多模态融合场景（雷达+相机、IMU+视觉）

## 评分

- 新颖性: ⭐⭐⭐⭐ （可微卡尔曼滤波用于异步多模态特征跟踪的设计很自然优雅）
- 实验充分度: ⭐⭐⭐⭐⭐ （多数据集 + 遮挡 + 消融 + 运行时分析 + 长期稳定性）
- 写作质量: ⭐⭐⭐⭐ （方法描述详尽，补充材料完整）
- 价值: ⭐⭐⭐⭐ （事件相机跟踪方向的实用进展，>80 FPS 达到实际部署要求）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Towards Efficient General Feature Prediction in Masked Skeleton Modeling](towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [\[CVPR 2026\] TAPFormer: Robust Arbitrary Point Tracking via Transient Asynchronous Fusion of Frames and Events](../../CVPR2026/video_understanding/ttapformer_robust_arbitrary_point_tracking_via_transient_asynchronous_fusion_of_.md)
- [\[CVPR 2025\] Localizing Events in Videos with Multimodal Queries](../../CVPR2025/video_understanding/localizing_events_in_videos_with_multimodal_queries.md)
- [\[ECCV 2024\] Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking](../../ECCV2024/video_understanding/exploring_the_feature_extraction_and_relation_modeling_for_light-weight_transfor.md)
- [\[CVPR 2026\] FPS-Bench: A Benchmark for High Frame-Rate Video Understanding](../../CVPR2026/video_understanding/fps-bench_a_benchmark_for_high_frame-rate_video_understanding.md)

</div>

<!-- RELATED:END -->
