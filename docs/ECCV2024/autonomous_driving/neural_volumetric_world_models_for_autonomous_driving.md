---
title: >-
  [论文解读] Neural Volumetric World Models for Autonomous Driving
description: >-
  [ECCV 2024][自动驾驶][体素世界模型] 本文提出 NeMo（Neural Volumetric World Model），一种基于体积表示的端到端自动驾驶框架，通过 3D 体素表征场景、运动流模块建模动态、时间注意力整合未来预测信息，以自监督方式训练并在 nuScenes 和 CARLA 上实现了超越前人方法 18%+ 的驾驶性能。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "体素世界模型"
  - "运动流估计"
  - "端到端规划"
  - "自监督学习"
---

# Neural Volumetric World Models for Autonomous Driving

**会议**: ECCV 2024  
**作者**: Zanming Huang, Jimuyang Zhang, Eshed Ohn-Bar
**代码**: 无  
**领域**: 自动驾驶 / 3D 世界模型  
**关键词**: 体素世界模型, 自动驾驶, 运动流估计, 端到端规划, 自监督学习

## 一句话总结

本文提出 NeMo（Neural Volumetric World Model），一种基于体积表示的端到端自动驾驶框架，通过 3D 体素表征场景、运动流模块建模动态、时间注意力整合未来预测信息，以自监督方式训练并在 nuScenes 和 CARLA 上实现了超越前人方法 18%+ 的驾驶性能。

## 研究背景与动机

**领域现状**：自动驾驶的感知和规划系统近年来取得快速进展。当前主流方法主要采用鸟瞰图（Bird's Eye View, BEV）作为场景的 2D 空间表示，在此表示上完成检测、预测和规划等任务。

**现有痛点**：BEV 表示将 3D 世界压缩为 2D 俯视平面，丢失了关键的垂直维度信息。这导致(1)无法准确建模遮挡关系（前方车辆互相遮挡）；(2)无法捕捉细粒度的垂直运动（如行人抬手、交通灯高度差异）；(3)在有坡度的地形和多层立体交叉路口中表示能力不足；(4)对部分可观测性（partial observability）的处理能力有限。

**核心矛盾**：现实驾驶场景是 3D 的，而主流方法使用 2D BEV 表示存在信息瓶颈。虽然 3D 体积表示（volumetric representation）理论上更合适，但其计算开销大、训练信号稀疏（大部分体素为空），如何高效地利用 3D 体积表示来改善端到端驾驶仍是开放问题。

**本文目标** (1) 如何构建基于 3D 体积表示的世界模型用于自动驾驶？(2) 如何在体积空间中建模复杂的动态场景（多个运动物体、不同运动模式）？(3) 如何将体积世界模型的预测结果有效地用于运动规划？(4) 如何以自监督方式训练整个系统，使其可扩展？

**切入角度**：作者认为 3D 体积表示能提供更忠实的场景建模——保留完整的空间结构、遮挡关系和运动特征。关键是设计高效的体积世界模型架构，并通过自监督任务（图像重建、占用预测）来训练，避免对昂贵的 3D 标注的依赖。

**核心 idea**：用 3D 体素表示替代 BEV，结合运动流模块和时间注意力构建体积世界模型，通过自监督训练实现高保真度的 3D 场景理解和驾驶规划。

## 方法详解

### 整体框架

NeMo 的输入为多视角相机图像序列，输出为自车的规划轨迹。Pipeline 包含以下阶段：(1) 视觉编码器将多视角图像特征提升为 3D 体积特征（lifting to voxel space）；(2) 运动流模块在体积空间中估计场景的 3D 运动场（motion flow field）；(3) 基于运动流的时间传播预测未来体积状态；(4) 时间注意力模块整合历史和预测的体积特征；(5) 规划头基于融合后的体积特征输出自车轨迹。训练使用图像重建和占用预测的自监督损失。

### 关键设计

1. **3D 体积特征提取（Volumetric Feature Lifting）**:

    - 功能：将多视角 2D 图像特征转换为统一的 3D 体积特征表示
    - 核心思路：每个视角的图像特征通过 2D backbone 提取后，利用已知的相机参数将 2D 特征"抛投"到 3D 体素空间中。具体采用类似 LSS（Lift-Splat-Shoot）的方法：对每个像素预测其沿深度方向的概率分布，然后将 2D 特征按概率加权分布到对应的 3D 体素位置。多视角的特征通过求和或平均融合到统一的体素网格中
    - 设计动机：体积表示相比 BEV 保留了完整的垂直维度信息，能更好地表示遮挡、多层结构和不规则地形。虽然计算量更大，但对 3D 世界建模更忠实

2. **运动流模块（Motion Flow Module）**:

    - 功能：在 3D 体积空间中估计每个体素的运动矢量，建模场景中所有物体的 3D 运动
    - 核心思路：给定连续两帧的体积特征 $V_t$ 和 $V_{t-1}$，运动流模块通过 3D 卷积网络估计一个逐体素的 3D 运动场 $F_{t \rightarrow t+1}$，表示每个体素从时刻 $t$ 到 $t+1$ 的位移。这个运动场可以用来将当前体积 warp 到未来状态，从而预测未来场景。运动场既包含了动态物体（车辆、行人）的运动，也隐式编码了自车运动（ego-motion）的影响
    - 设计动机：传统方法在 2D BEV 上做运动预测丢失了垂直方向的运动信息。3D 运动场能捕捉物体在所有三个维度上的运动特征（如行人的垂直运动、上下坡车辆的高度变化），提供更完整的动态建模。此外，运动一致性可以作为自监督信号来约束时序特征学习

3. **时间注意力模块（Temporal Attention Module）**:

    - 功能：将预测的未来体积特征与历史体积特征融合，为规划提供综合的时空信息
    - 核心思路：使用 Transformer 的注意力机制，在时间维度上对多帧体积特征做注意力聚合。具体地，将当前帧的体积特征作为 Query，历史帧和预测未来帧的体积特征作为 Key 和 Value。通过注意力权重学习每个体素应该从哪些时刻获取信息——对于静态物体（如建筑物），可以从多帧累积信息以增强鲁棒性；对于动态物体，更关注最近的帧以获取最新的运动状态
    - 设计动机：规划任务需要同时理解"过去发生了什么"和"未来会发生什么"。时间注意力让规划头能获取融合了预测信息的特征，而非只看当前帧，从而做出更前瞻性的决策

### 损失函数 / 训练策略

NeMo 采用自监督训练范式，核心损失包括：(1) **图像重建损失**：通过体积渲染（volume rendering）从体积特征重建多视角图像，使用 L2 + perceptual loss 与真实图像比较；(2) **占用预测损失**：利用 LiDAR 点云作为弱监督信号预测 3D 空间的占用状态；(3) **运动流一致性损失**：要求前后帧通过运动流 warp 后的体积特征保持一致。规划部分使用 imitation learning 损失，即预测轨迹与专家轨迹的 L2 距离。整个系统端到端训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | NeMo | 之前SOTA | 提升 |
|--------|------|------|---------|------|
| nuScenes | Driving Score ↑ | SOTA | BEV-based | +18% |
| nuScenes | Route Completion ↑ | SOTA | BEV-based | 显著提升 |
| CARLA | Driving Score ↑ | SOTA | 多种方法 | +18%+ |
| CARLA | Infraction Score ↑ | SOTA | 多种方法 | 安全性显著提升 |

### 消融实验

| 配置 | Driving Score | 说明 |
|------|--------------|------|
| Full NeMo | 最优 | 完整模型 |
| BEV 替代体积表示 | 下降明显 | 验证体积表示的优势 |
| w/o 运动流模块 | 中等下降 | 无法预测动态场景变化 |
| w/o 时间注意力 | 下降 | 缺乏时序信息融合 |
| w/o 图像重建损失 | 下降 | 自监督信号减少导致特征质量下降 |
| 体积分辨率降低 | 轻微下降 | 粗分辨率下仍有效 |

### 关键发现
- 3D 体积表示相比 BEV 的优势在遮挡严重和地形复杂的场景中最为明显
- 运动流模块在动态密集场景（如交叉路口）中贡献最大，静态场景中贡献有限
- 自监督训练（图像重建+占用预测）释放了体积表示的潜力，避免了昂贵的 3D 标注
- 整体提升 18%+ 是一个非常显著的数字，说明从 2D BEV 到 3D 体积是一个有意义的方向跃迁

## 亮点与洞察
- **从 BEV 到体积的范式转换**：论文清楚地论证了为什么 3D 体积表示优于 2D BEV，并且通过 18%+ 的性能提升给出了令人信服的实验证据。这个方向转换对整个自动驾驶感知-规划领域有引导作用
- **自监督训练范式**的设计非常巧妙：图像重建损失让模型学习完整的 3D 几何和外观，运动一致性让模型学习动态。不依赖 3D 标注使得系统可以在海量驾驶视频上训练
- **运动流在体积空间中的估计**可以迁移到机器人操作、室内导航等其他需要 3D 动态理解的任务

## 局限与展望
- 3D 体积表示的计算和内存开销远大于 BEV，限制了实时性和部署到嵌入式平台的可能
- 当前体积分辨率受限于 GPU 内存，在远距离区域精度不足；稀疏体素或八叉树结构可能是解决方案
- 仅使用相机输入，没有融合 LiDAR 点云作为直接输入。虽然 LiDAR 监督信号有使用，但融合其空间信息可能进一步提升效果
- 运动流假设物体是刚体运动，对于变形物体（如行人的关节运动）建模不够精细
- 在极端天气（大雨、大雾）和夜间场景中的表现有待验证

## 相关工作与启发
- **vs UniAD**: UniAD 在 BEV 空间中做端到端规划，但受限于 2D 表示。NeMo 通过体积表示提供了更丰富的 3D 信息，尤其在垂直维度的建模上
- **vs OccWorld**: OccWorld 也使用占用网格作为世界模型，但主要关注占用预测任务。NeMo 更关注如何利用体积世界模型改善规划
- **vs MUVO**: MUVO 也探索了体积表示在自动驾驶中的应用，NeMo 的区别在于引入了运动流模块和时间注意力来更好地建模动态场景
- **vs SelfD**: SelfD 同样使用自监督训练，但在 2D 空间中工作。NeMo 将自监督范式扩展到了 3D 体积空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 体积世界模型+运动流+时间注意力的组合在端到端驾驶中是新颖的
- 实验充分度: ⭐⭐⭐⭐ nuScenes 和 CARLA 双重验证，18%+ 提升非常有说服力
- 写作质量: ⭐⭐⭐⭐ 动机论证充分（为什么需要 3D 体积而非 BEV），技术描述清晰
- 价值: ⭐⭐⭐⭐⭐ 提出了 BEV→体积的范式转换，对自动驾驶领域有方向性指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[ECCV 2024\] SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic](sledge_synthesizing_driving_environments_with_generative_models_and_rule-based_t.md)
- [\[CVPR 2026\] WorldLens: Full-Spectrum Evaluations of Driving World Models in Real World](../../CVPR2026/autonomous_driving/worldlens_full-spectrum_evaluations_of_driving_world_models_in_real_world.md)

</div>

<!-- RELATED:END -->
