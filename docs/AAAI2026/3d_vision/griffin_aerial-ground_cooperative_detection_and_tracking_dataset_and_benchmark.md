---
title: >-
  [论文解读] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark
description: >-
  [AAAI 2026][3D视觉][空地协同感知] 提出 Griffin，一个空地协同（AGC）3D感知数据集和基准框架，包含250+动态场景（37K+帧），通过CARLA-AirSim联合仿真实现真实无人机动力学、变化巡航高度（20-60m）和遮挡感知标注，并提供系统化的鲁棒性评估协议。 领域现状 协同感知已成为克服单车系…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "空地协同感知"
  - "无人机-车辆协作"
  - "3D目标检测"
  - "多目标跟踪"
  - "协同感知数据集"
---

# Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark

**会议**: AAAI 2026  
**arXiv**: [2503.06983](https://arxiv.org/abs/2503.06983)  
**代码**: [https://github.com/wang-jh18-SVM/Griffin](https://github.com/wang-jh18-SVM/Griffin)  
**领域**: 3D视觉 / 协同感知  
**关键词**: 空地协同感知, 无人机-车辆协作, 3D目标检测, 多目标跟踪, 协同感知数据集

## 一句话总结

提出 Griffin，一个空地协同（AGC）3D感知数据集和基准框架，包含250+动态场景（37K+帧），通过CARLA-AirSim联合仿真实现真实无人机动力学、变化巡航高度（20-60m）和遮挡感知标注，并提供系统化的鲁棒性评估协议。

## 研究背景与动机

### 领域现状

协同感知已成为克服单车系统限制（遮挡、有限视野）的重要方向。主要模式包括：
- **V2V（车车协同）**：OPV2V、V2V4Real等
- **V2I（车路协同）**：DAIR-V2X、V2X-Seq等
- **UAV协同**：CoPerception-UAVs、UAV3D等

### 核心矛盾

V2V/V2I系统需要大规模基础设施投资和车联网普及，经济门槛高。**空地协同（AGC）**——将无人机与地面车辆配对——是更灵活、经济的替代方案，可按需部署，提供无遮挡的鸟瞰视角。但AGC感知研究受制于**缺乏高质量公共数据集和基准**。

### 现有数据集的不足

| 问题 | 受影响的数据集 |
|------|-------------|
| 理想化通信和定位（无噪声） | UAV3D, AeroCollab3D, Air-Co-Pred, AirV2X |
| 简化无人机模型（固定方向/高度） | V2U-COO, UAV3D, Air-Co-Pred |
| 缺乏遮挡感知标注 | CoPerception-UAVs, UAV3D, AeroCollab3D, AirV2X |
| 无跟踪ID | AGC-Drive |
| 仅2D标注 | CoPeD |

**关键差距**：没有一个AGC数据集同时具备遮挡感知标注、真实噪声仿真、多高度支持和跟踪ID。

### 本文切入角度

构建Griffin——首个同时支持遮挡感知3D标注、真实无人机动力学、多高度设置和通信干扰/定位误差仿真的AGC感知数据集，并提供统一的检测和跟踪基准框架。

## 方法详解

### 整体框架

Griffin由三部分组成：
1. **数据集**：CARLA-AirSim联合仿真 → 多传感器数据采集 → 遮挡感知标注
2. **基准框架**：四种融合范式（早期/中间BEV级/中间实例级/晚期）的标准化实现
3. **评估协议**：精度+通信效率+鲁棒性（延迟、丢包、定位误差）

### 关键设计

#### 1. 数据采集与场景多样性

**功能**：通过CARLA-AirSim联合仿真生成同步多智能体数据。

**传感器配置**：
- **地面车辆**：4个宽视场RGB相机（108.8°, 1920×1080）+ 80线LiDAR（10Hz，垂直FOV -25°到15°）
- **空中无人机**：5个向下相机（受SWaP约束，无LiDAR）

**场景多样性**：
- 4张CARLA地图（2个城市 + 2个郊区）
- 天气：晴天/雨天/雾天 × 正午/日落/夜晚 × 风速0-9m/s
- 高度：Griffin-Random（20-60m）、Griffin-25m/40m/55m（各±2m）
- 255个场景片段，每段~15秒，总计37.7K帧、339.3K图像、914.8K 3D标注

**无人机动力学真实性**：利用AirSim的物理引擎模拟，俯仰/横滚角分布在零附近而非尖峰——反映了真实无人机持续的微调和抗风姿态调整。

**设计动机**：
- CARLA提供丰富环境和交通流，AirSim提供真实无人机物理模型
- 无LiDAR的无人机配置贴近实际（如比亚迪-大疆方案，小型无人机载荷<1kg）
- 变化高度和天气条件测试方法的泛化能力

#### 2. 遮挡感知标注

**功能**：量化每个目标对每个智能体的可见率，过滤不可见目标。

**核心思路**：
1. 采集RGB和实例分割图像（完全对齐）
2. 在每个3D包围盒内采样点，投影到分割图像上
3. 比对投影像素的语义类别和实例ID与目标的一致性
4. 计算每个智能体的可见率
5. **协同感知GT**：保留对任一智能体可见的目标

**设计动机**：
- 许多数据集仅按距离过滤标注（黄色框），忽略了严重遮挡的目标（红色框）
- 不做遮挡过滤会引入不可见目标的标注噪声，降低模型训练质量
- 实验验证：不做遮挡过滤的训练使Early Fusion AP从0.607降至0.586

#### 3. 基准框架

**功能**：在统一backbone（BEVFormer + ResNet-50）上实现四种融合范式。

**四种融合策略**：

| 融合类型 | 代表方法 | 通信量(BPS) | 特点 |
|---------|---------|-----------|------|
| Early Fusion | 原始图像传输 | 3.11×10⁸ | 性能上界，带宽极高 |
| BEV级中间融合 | V2X-ViT, Where2comm | 3.3-8.0×10⁵ | 场景级BEV特征，压缩后传输 |
| 实例级中间融合 | UniV2X, CoopTrack | 0.56-1.17×10⁵ | 稀疏目标query，带宽更低 |
| Late Fusion | 检测结果传输 | 1.56×10³ | 极低带宽，性能有限 |

**评估协议**：
- **精度**：NuScenes AP和AMOTA
- **通信效率**：每秒传输字节数(BPS)
- **鲁棒性**：通信延迟（0-400ms）、丢包率（0-50%）、定位误差（平移0-2.5m, 旋转0-5°）

### 损失函数 / 训练策略

- AdamW优化器，学习率2×10⁻⁴，batch size 8
- 4×NVIDIA 3090 GPU分布式训练
- 输入图像从1920×1080降采样到960×540
- 目标合并为3类（汽车、行人、两轮车）
- 感知范围：以ego车辆为中心的102.4m×102.4m区域

## 实验关键数据

### 主实验

#### 各方法在不同高度数据集上的表现

| 方法 | Griffin-25m AP/AMOTA | Griffin-55m AP/AMOTA | 通信量(BPS) |
|------|---------------------|---------------------|------------|
| No Fusion | 0.375/0.365 | 0.335/0.359 | 0 |
| Early Fusion | **0.607/0.670** | 0.483/0.522 | 3.11×10⁸ |
| V2X-ViT | 0.465/0.508 | 0.350/0.379 | 8.00×10⁵ |
| Where2comm | 0.396/0.406 | 0.317/0.353 | 3.30×10⁵ |
| **CoopTrack** | 0.479/0.488 | 0.364/0.402 | **1.17×10⁵** |
| UniV2X | 0.419/0.456 | 0.323/0.349 | 5.58×10⁴ |
| Late Fusion | 0.378/0.377 | 0.306/0.332 | 1.56×10³ |

#### Griffin-Random（20-60m混合高度）

| 方法 | AP | vs No Fusion |
|------|-----|-------------|
| No Fusion | 0.459 | — |
| Early Fusion | 0.583 | +0.124 |
| V2X-ViT | 0.400 | **-0.059** |
| Where2comm | 0.406 | -0.053 |
| CoopTrack | 0.468 | +0.009 |
| UniV2X | 0.402 | -0.057 |

### 消融实验

#### 遮挡感知标注的影响（Griffin-25m）

| 模型 | 标注方式 | AP | AMOTA |
|------|---------|-----|-------|
| Early Fusion | 遮挡感知（baseline） | 0.607 | 0.670 |
| Early Fusion | 无过滤 | 0.586(↓) | 0.636(↓) |
| Vehicle Side | 遮挡感知 | 0.477 | 0.457 |
| Vehicle Side | 无过滤 | 0.412(↓) | 0.433(↓) |

#### 通信鲁棒性

| 延迟(ms) | Early Fusion AP降幅 | 中间融合表现 |
|---------|------------------|------------|
| 100 | ~10% | 仍优于No Fusion |
| 200 | ~20% | 检测勉强优于, 跟踪仍好 |
| 400 | >30% | 跟踪仍维持优势 |

#### 定位鲁棒性

| 平移误差std(m) | V2X-ViT | UniV2X |
|--------------|---------|--------|
| 0.5 | 正常 | 正常 |
| 1.5 | 低于No Fusion | **仍优于No Fusion** |
| 2.5 | 严重退化 | **仍保持优势** |

### 关键发现

1. **高度变化对协同感知影响巨大**：25m高度时协同增益最大，随高度增加性能下降。在混合高度（20-60m）下，多数中间融合方法甚至**不如单车baseline**
2. **实例级融合比BEV级更鲁棒**：CoopTrack是唯一在Griffin-Random上保持正增益的中间融合方法——因为实例级方法将几何变换和语义特征解耦，对视角不一致更鲁棒
3. **Where2comm和UniV2X在AGC场景中表现不佳**：无人机鸟瞰视角下目标稀疏，基于正样本检测的空间置信图/稀疏query训练不充分
4. **丢包比延迟的影响更小**：丢包只导致信息缺失（减少增益），不引入错误数据；而延迟导致空间不对齐
5. **UniV2X对定位误差最鲁棒**：选择性融合和实例级过滤能降权不可靠信号
6. **遮挡感知标注很重要**：不做过滤使协同和单车模型性能都下降
7. **跟踪比检测对延迟更鲁棒**：时序信息帮助缓解帧间对齐问题

## 亮点与洞察

1. **"高度变化使协同感知失效"是AGC特有的深刻发现**：V2V/V2I中不存在这个问题，但对AGC至关重要
2. **遮挡感知标注方法简洁有效**：利用仿真器的实例分割GT量化可见率，避免了手动标注的成本
3. **鲁棒性评估范围激进**（2.5m平移/5°旋转/400ms延迟/50%丢包）：远超标准评估范围，揭示了方法的真实失效边界
4. **CARLA-AirSim联合仿真框架**：巧妙利用两个仿真器各自的优势（CARLA的环境+AirSim的无人机物理）
5. **对BEV vs 实例级融合的深入对比**：在AGC场景中为融合策略选择提供了明确指导

## 局限与展望

1. **仿真数据与真实数据的域差距**：虽然已尽力逼近真实（无LiDAR无人机、噪声注入），但sim-to-real gap仍存在
2. **仅评估汽车类别**：行人和两轮车的评估结果未在主文展示
3. **固定backbone（ResNet-50 BEVFormer）**：更强的单车检测器可能改变融合方法的相对排名
4. **未考虑天气对不同方法的差异化影响**：虽然数据覆盖多种天气，但分析中未按天气分组
5. 应开发**高度自适应和尺度感知的融合机制**来解决核心挑战
6. 可探索更先进的Late Fusion策略：在极低带宽下可能获得更好的性价比

## 相关工作与启发

- **OPV2V** (Xu et al., ICRA 2022)：V2V协同感知开创性工作，Griffin填补了AGC空白
- **DAIR-V2X** (Yu et al., CVPR 2022)：真实V2I数据集，但相机高度固定20-25m
- **BEVFormer** (Li et al., 2022)：统一backbone用于所有baseline
- **V2X-ViT, Where2comm**：BEV级中间融合代表，本文揭示其在AGC场景的局限
- **CoopTrack** (Zhong et al., ICCV 2025)：实例级融合在高度变化场景中更鲁棒
- **启发**：AGC场景需要全新的融合设计理念——不能简单迁移V2V/V2I方法，高度变化带来的尺度和视角不一致是核心挑战

## 评分

- 新颖性: ⭐⭐⭐⭐ — AGC数据集不是全新概念，但遮挡感知标注和系统化鲁棒性评估有新意
- 实验充分度: ⭐⭐⭐⭐⭐ — 6种方法×4个高度×3种干扰类型，评估极为全面
- 写作质量: ⭐⭐⭐⭐⭐ — 数据集构建细节和实验分析都非常深入
- 价值: ⭐⭐⭐⭐⭐ — 填补AGC感知研究的关键数据空白，"高度鲁棒性"发现对后续研究有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](../../ICCV2025/3d_vision/event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2026\] VGA: Empowering Aerial-Ground Localization by Visual Geometry Alignment](../../CVPR2026/3d_vision/vga_empowering_aerial-ground_localization_by_visual_geometry_alignment.md)
- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)
- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](../../CVPR2025/3d_vision/aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)

</div>

<!-- RELATED:END -->
