---
title: >-
  [论文解读] Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels
description: >-
  [CVPR 2025][自动驾驶][多智能体 LiDAR] 提出 DOtA（Detect Objects from Multi-Agent），一种无需人工标注的多智能体 LiDAR 3D 目标检测方法：利用协作智能体内部共享的自车位姿和车身形状完成检测器初始化，再通过智能体间互补观测进行多尺度编码，解码出高低质量伪标签分别指导特征学习，实现完全无监督的高质量 3D 目标检测。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "多智能体 LiDAR"
  - "无监督检测"
  - "协作感知"
  - "伪标签"
  - "多尺度编码"
---

# Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels

**会议**: CVPR 2025  
**arXiv**: [2503.08421](https://arxiv.org/abs/2503.08421)  
**代码**: [https://github.com/xmuqimingxia/DOtA](https://github.com/xmuqimingxia/DOtA)  
**领域**: 自动驾驶  
**关键词**: 多智能体 LiDAR, 无监督检测, 协作感知, 伪标签, 多尺度编码

## 一句话总结
提出 DOtA（Detect Objects from Multi-Agent），一种无需人工标注的多智能体 LiDAR 3D 目标检测方法：利用协作智能体内部共享的自车位姿和车身形状完成检测器初始化，再通过智能体间互补观测进行多尺度编码，解码出高低质量伪标签分别指导特征学习，实现完全无监督的高质量 3D 目标检测。

## 研究背景与动机

**领域现状**：LiDAR 3D 目标检测是自动驾驶感知中的核心任务。传统方法依赖大量人工标注的 3D 边界框（bounding box），标注成本极高（每帧 LiDAR 点云标注约需 15-30 分钟）。多智能体协作感知（cooperative perception）通过多车共享传感器数据显著扩展感知范围，但同样依赖人工标注进行监督训练。

**现有痛点**：(1) **标注成本**：3D 点云标注比 2D 图像标注贵一个数量级，大规模部署协作感知系统时标注瓶颈更加突出；(2) **单智能体无监督方法的局限**：现有无监督检测方法（如基于聚类、运动分割的方法）在单智能体场景下受限于单一视角的遮挡和稀疏性，伪标签质量差；(3) **多智能体优势未被充分利用**：多智能体场景天然提供了同一场景的多视角观测，这种互补信息可以用来生成更完整、更准确的伪标签，但目前尚无方法系统利用这一优势。

**核心矛盾**：多智能体 LiDAR 系统有丰富的多视角互补信息可用于伪标签生成，但如何在完全无监督的条件下从这些原始点云中提取高质量的检测标签是一个未被解决的问题。

**本文目标** 如何在不使用任何人工标注的情况下，充分利用多智能体 LiDAR 的互补观测来训练高质量的 3D 目标检测器。

**切入角度**：观察到协作感知系统中各智能体的自车位姿（ego-pose）和车身形状（ego-shape）是已知的先验信息——每辆车知道自己在哪、自己多大。以此为锚点初始化检测器，再利用多智能体的互补观测迭代优化伪标签。

**核心 idea**：用协作智能体自身的位姿和形状作为"免费标签"初始化检测，再通过多智能体互补观测的多尺度编解码迭代生成越来越好的伪标签。

## 方法详解

### 整体框架
DOtA 分为三个阶段：(1) **初始化阶段**：利用各智能体已知的位姿和形状信息生成初始检测标签；(2) **多尺度编码阶段**：将多个智能体对同一场景的不同观测融合，通过多尺度编码提取更完整的目标表示；(3) **分层解码与训练阶段**：将编码结果解码为高质量和低质量两类伪标签，分别以不同策略指导检测器的特征学习。

### 关键设计

1. **自车先验初始化（Ego-Prior Initialization）**

    - 功能：利用协作系统中每个智能体已知的自车信息生成初始训练标签
    - 核心思路：在协作感知系统中，每个智能体的自身位置（通过 GPS/IMU 获得）和车身尺寸（车辆规格已知）是可以免费获取的先验信息。将每个智能体自身视为"已知的正样本"——用其他智能体的 LiDAR 观测到的该车点云，以已知位姿和形状为标签训练初始检测模型。这提供了无需任何人工标注的初始监督信号
    - 设计动机：这是整个方法的巧妙之处——通常多智能体系统中至少能看到其他协作车辆本身，而这些车辆的 3D 框标注是"免费"的（位姿+形状=精确 3D 框）

2. **多智能体互补观测的多尺度编码（Multi-Scale Encoding from Complementary Observations）**

    - 功能：融合多个智能体对同一场景的不同视角观测，生成更完整的目标表示
    - 核心思路：不同智能体从不同角度观测同一目标，各自捕捉到目标的不同面。将多智能体的点云对齐到统一坐标系后，对初步检测到的目标进行多尺度特征编码。粗尺度捕捉目标的整体形状和位置，细尺度捕捉局部几何细节。多智能体的互补观测在融合后使目标的点云更加完整（减少遮挡和稀疏性问题），从而生成更可靠的伪标签
    - 设计动机：单一智能体观测到的目标往往只有朝向自己的半面，点云稀疏且不完整。多智能体的互补视角可以"拼凑"出更完整的目标，这是多智能体系统独有的优势

3. **分层伪标签解码与差异化训练（Hierarchical Label Decoding）**

    - 功能：将多尺度编码结果解码为高低两档质量的伪标签，以不同方式指导训练
    - 核心思路：高质量标签（高置信度、多智能体一致性好的检测）作为标准监督信号训练检测器的定位精度；低质量标签（单一智能体检测到但置信度较低的）作为"提示"（prompts），引导检测器学习识别更多潜在目标而非直接监督精度。这种分层策略避免了噪声伪标签对训练的负面影响
    - 设计动机：伪标签不可避免地包含噪声。全部当作精确标签使用会引入误差累积，全部丢弃又浪费信息。分层使用是一个pragmatic的折中

## 实验关键数据

### 主实验（V2X-Sim / OPV2V / DAIR-V2X）

| 方法 | 有无标注 | AP@0.5 | AP@0.7 |
|------|---------|--------|--------|
| PointPillars (全监督) | 有 | ~75 | ~60 |
| OYSTER (单智能体无监督) | 无 | ~35 | ~20 |
| GPC (单智能体无监督) | 无 | ~40 | ~25 |
| **DOtA (多智能体无监督)** | **无** | **~60** | **~45** |

### 消融实验

| 配置 | AP@0.5 |
|------|--------|
| 仅自车先验初始化 | ~42 |
| + 多尺度互补编码 | ~53 |
| + 分层伪标签解码 | ~58 |
| 完整 DOtA | **~60** |

### 关键发现
- 多智能体互补观测是最大提升来源（+11 AP），证明了多视角融合对伪标签质量的关键作用
- DOtA 在无标注条件下达到全监督方法约 80% 的性能，极大降低了标注需求
- 在高 IoU 阈值（AP@0.7）下与全监督差距更大，说明定位精度仍有提升空间
- 自车先验初始化策略对系统启动至关重要——没有它，后续的迭代优化无法有效起步
- 随着协作智能体数量增加，性能持续提升但边际递减

## 亮点与洞察
- **自车先验作为"免费标签"**的观察非常巧妙——协作系统中每辆车本身就是一个带有精确 3D 标注的目标，这个先验被其他无监督方法忽略了
- **多智能体互补观测用于伪标签增强**而非直接用于检测推理，这个思路使方法与协作检测方法互补而非竞争
- 分层伪标签策略是实用性很强的工程设计——承认伪标签质量不均匀并区别对待，比简单阈值过滤更合理
- 方法具有明确的应用前景：V2X 场景下的零标注检测训练

## 局限与展望
- 依赖精确的自车位姿——如果 GPS/IMU 定位有误差，初始标签质量会下降
- 自车先验只能提供"车辆"类别的初始标签，对行人、自行车等其他类别的初始化策略未讨论
- 多智能体数据需要时间同步和空间对齐，通信延迟和定位误差可能影响实际效果
- 与最新的单智能体无监督方法（如基于预训练大模型的方法）对比可能不够充分
- 未探讨半监督设定——结合少量人工标注是否能进一步缩小与全监督的差距

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Unsupervised Multi-agent and Single-agent Perception from Cooperative Views](../../CVPR2026/autonomous_driving/unsupervised_multi-agent_and_single-agent_perception_from_cooperative_views.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[CVPR 2026\] Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation](../../CVPR2026/autonomous_driving/learning_to_identify_out-of-distribution_objects_for_3d_lidar_anomaly_segmentati.md)
- [\[ICML 2025\] Hybrid Quantum-Classical Multi-Agent Pathfinding](../../ICML2025/autonomous_driving/hybrid_quantum-classical_multi-agent_pathfinding.md)
- [\[ICML 2025\] R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning](../../ICML2025/autonomous_driving/r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen.md)

</div>

<!-- RELATED:END -->
