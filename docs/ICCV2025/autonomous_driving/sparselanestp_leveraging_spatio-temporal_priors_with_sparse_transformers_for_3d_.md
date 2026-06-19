---
title: >-
  [论文解读] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection
description: >-
  [ICCV 2025][自动驾驶][3D车道线检测] 提出 SparseLaneSTP，将车道线几何先验（平行性、连续性）和时序信息融合进稀疏 Transformer 架构，通过 Catmull-Rom 样条表示、时空注意力机制和时序正则化，在多个 3D 车道线检测基准上取得 SOTA。 3D 车道线检测是自动驾驶中的核心任…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "3D车道线检测"
  - "Transformer"
  - "时空先验"
  - "Catmull-Rom样条"
  - "时序融合"
---

# SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection

**会议**: ICCV 2025  
**arXiv**: [2601.04968](https://arxiv.org/abs/2601.04968)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 3D车道线检测, 稀疏Transformer, 时空先验, Catmull-Rom样条, 时序融合

## 一句话总结

提出 SparseLaneSTP，将车道线几何先验（平行性、连续性）和时序信息融合进稀疏 Transformer 架构，通过 Catmull-Rom 样条表示、时空注意力机制和时序正则化，在多个 3D 车道线检测基准上取得 SOTA。

## 研究背景与动机

3D 车道线检测是自动驾驶中的核心任务，需要在车辆坐标系中直接输出 3D 车道标线。现有方法存在以下问题：

**BEV 方法的固有缺陷**：传统方法先将前视图（FV）特征转换为鸟瞰图（BEV）表示，再进行车道线估计。无论使用 IPM（假设平坦地面）还是学习式映射，BEV 变换都会引入误差，导致特征与真实 3D 路面错位，后续难以补偿。

**稀疏方法忽略先验**：近年来基于 DETR 的稀疏检测方法（如 LATR）避免了 BEV 表示的误差，但完全忽略了车道线的几何先验（如平行性、连续性）。这些先验在 LaneCPP 等 BEV 方法中已被证明有效，但尚未成功适配到稀疏架构。

**时序信息未被利用**：车道线标记是静态的，历史帧的观测在遮挡、能见度差等场景下具有消歧义的巨大潜力。然而，现有方法几乎没有有效利用时序融合。

作者的核心洞察：稀疏 Transformer 中的查询设计天然适合以对象为中心的时序传播范式（如 StreamPETR），而车道线的静态特性使得时序一致性正则化成为可能。

## 方法详解

### 整体框架

SparseLaneSTP 的输入为单帧 RGB 图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$，输出 $N$ 条 3D 车道线。框架包含：CNN backbone 提取特征 → 车道线实例分割初始化查询 → Transformer 解码器（含时空注意力 STA + 可变形交叉注意力 DCA）→ 预测层输出样条控制点和分类概率。

### 关键设计

1. **Catmull-Rom (CR) 样条连续表示**：

    - **核心思路**：在稀疏查询设计中，控制点的 3D 位置作为模型内部状态，应直接对应车道线上的精确位置。B-Spline 的控制点不在曲线上（见论文 Fig. 2），不适合稀疏设计。CR 样条的曲线必然经过控制点，天然匹配。
    - **公式**：第 $i$ 条车道线表示为 $\mathbf{f}_i(s) = [s^3, s^2, s, 1] \cdot \mathbf{M}_{CR} \cdot \mathbf{P}_i$，其中 $s \in [0,1]$，$\mathbf{P}_i \in \mathbb{R}^{M \times 4}$ 包含 3D 空间坐标和可见性分量。
    - **设计动机**：连续表示几乎不需要后处理，训练时可利用全部稠密 GT。纵向 $y$ 分量预定义为均匀分布以避免过参数化。

2. **时空注意力机制 (STA)**：

    - **核心思路**：替代全局自注意力中大量冗余的查询交互，基于车道线结构先验设计三种注意力：
        - **同车道注意力 (SLA)**：限制交互在同一车道线的相邻控制点之间，捕获内在连续性。
        - **平行邻居注意力 (PNA)**：促进相邻平行车道线之间的交互，学习几何约束。
        - **时序交叉注意力 (TCA)**：利用记忆队列中历史帧传播的查询，与当前查询交互。
    - **记忆队列**：采用 FIFO 策略存储过去 $T$ 帧的查询嵌入和控制点，历史控制点根据自车位姿变换传播到当前帧坐标系。
    - **设计动机**：减少冗余交互，聚焦车道线特有的空间关系；时序交叉注意力可在遮挡/低可见度场景下利用历史观测消除歧义。

3. **时空正则化**：

    - **空间正则化** $\mathcal{L}_{spatial}$：继承自 LaneCPP，鼓励车道平行性、路面光滑性和抑制过度曲率。
    - **时序一致性正则化** $\mathcal{L}_{temp}$：基于时序预测的指数滑动平均，约束当前帧预测与历史平均预测的一致性。使用可见性加权的 L1 损失：$\mathcal{L}_{temp} = \frac{1}{N} \sum_i \int_s \bar{\mathbf{f}}_{v,i}^{(t)}(s) \cdot \| \mathbf{f}_{3D,i}(s) - \bar{\mathbf{f}}_{3D,i}^{(t)}(s) \|_1$。
    - **设计动机**：车道线是静态标记，帧间预测应保持一致。时序正则化防止预测漂移和逐渐消失。

### 损失函数 / 训练策略

- 回归损失：L1 损失用于 $x$ 和 $z$ 分量
- 可见性损失：二元交叉熵
- 分类损失：Focal Loss
- 附加正则化：$\mathcal{L}_{spatial} + \mathcal{L}_{temp}$
- 预测层采用层间共享权重的 MLP，使用 sigmoid 归一化到 $[0,1]$ 后缩放到目标范围
- 辅助任务：车道线实例分割用于初始化查询

## 实验关键数据

### 主实验

**OpenLane 数据集对比（Table 4）：**

| 方法 | Backbone | F1(%)↑ | X-err near(m)↓ | X-err far(m)↓ | Z-err near(m)↓ | Z-err far(m)↓ |
|------|----------|--------|----------------|---------------|----------------|---------------|
| PersFormer | EfficientNet-B7 | 50.5 | 0.485 | 0.553 | 0.364 | 0.431 |
| LATR | ResNet-50 | 61.9 | 0.219 | 0.259 | 0.075 | 0.104 |
| PVALane | Swin-B | 63.4 | 0.226 | 0.257 | 0.093 | 0.119 |
| GroupLane | ConvNext-B | 64.1 | 0.320 | 0.441 | 0.233 | 0.402 |
| **SparseLaneSTP** | **ResNet-50** | **66.1** | **0.203** | **0.240** | **0.066** | **0.092** |

**ONCE-3DLanes 数据集（Table 6）：**

| 方法 | F1(%)↑ | Precision(%)↑ | Recall(%)↑ | CD(m)↓ |
|------|--------|--------------|-----------|--------|
| LATR | 80.59 | 86.12 | 75.73 | 0.052 |
| GroupLane | 80.73 | 82.56 | 78.90 | 0.053 |
| **SparseLaneSTP** | **82.75** | **86.47** | **79.33** | **0.048** |

### 消融实验

**各贡献的增量效果（Table 1，OpenLane，2层解码器）：**

| 配置 | F1(%)↑ | 增量 |
|------|--------|------|
| Baseline（离散 + 全局注意力） | 61.8 | — |
| + CR 样条连续表示 | 62.9 | +1.1 |
| + 时空注意力 (STA) | 65.0 | +2.1 |
| + 正则化 | 65.3 | +0.3 |

**注意力组合消融（Table 2）：**

| 注意力类型 | F1(%)↑ |
|-----------|--------|
| 全局自注意力 | 62.9 |
| SLA + PNA | 63.8 |
| SLA + PNA + TCA | **65.0** |

### 关键发现

- 时序交叉注意力 (TCA) 带来最大增益 (+1.2%)，验证了车道线场景下时序信息的重要性
- 最佳时序窗口 $T=3$ 帧：过少缺乏时序上下文，过多引入冗余
- 2 层解码器 + STA 的轻量模型即可达到 65.3% F1 / 16.5 FPS，超过 6 层 LATR 的 61.9% / 12.1 FPS，时序集成仅增加 9% 开销

## 亮点与洞察

- **CR 样条替代 B-Spline 的洞察极为精准**：控制点在曲线上这一性质与稀疏查询的 3D 位置语义完美匹配
- **注意力的结构化设计**比全局注意力更有效，说明在几何结构明确的任务中引入归纳偏置优于通用方案
- 时序信息对遮挡/低可见度场景的恢复效果显著（定性结果 Fig. 6 中非时序模型丢失检测而时序模型保持稳定）
- 额外贡献了高质量 3D 车道线数据集（250m 范围，含遮挡标注），基于自标注管线

## 局限与展望

- 仅使用单目前视摄像头，未探索多摄像头环视场景
- 自标注数据集依赖 2D 检测器和视觉里程计的质量
- 未利用车道线的语义信息（实线/虚线/停止线等）辅助时序匹配
- 可扩展到 3D 车道线跟踪以更充分利用时序信息（作者已提及为未来方向）

## 相关工作与启发

- 继承 LaneCPP 的空间先验思想并成功适配到稀疏架构，证明了先验知识跨架构迁移的可行性
- 时序融合借鉴 StreamPETR / Sparse4Dv2 的对象中心查询传播范式，比 BEV 时序融合更高效
- 启发：结构化注意力 + 领域先验正则化的组合可推广到其他具有几何约束的检测任务（如道路边缘检测、护栏检测）

## 评分

- **新颖性**: ⭐⭐⭐⭐ CR 样条适配稀疏查询的洞察新颖，时空注意力设计合理但思路不算全新
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集 + 自建数据集，消融详尽，效率分析完整
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式规范，图表丰富
- **价值**: ⭐⭐⭐⭐ 在 3D 车道线检测具有实际工程价值，证明时序融合对车道线检测的重要性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[CVPR 2025\] Rethinking Lanes and Points in Complex Scenarios for Monocular 3D Lane Detection](../../CVPR2025/autonomous_driving/rethinking_lanes_and_points_in_complex_scenarios_for_monocular_3d_lane_detection.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[CVPR 2026\] STUR3D: Spatio-Temporal Unified Representation Learning for 3D Object Detection](../../CVPR2026/autonomous_driving/stur3d_spatio-temporal_unified_representation_learning_for_3d_object_detection.md)
- [\[ECCV 2024\] Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](../../ECCV2024/autonomous_driving/equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)

</div>

<!-- RELATED:END -->
