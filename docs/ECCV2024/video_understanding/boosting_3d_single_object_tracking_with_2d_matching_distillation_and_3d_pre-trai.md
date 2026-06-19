---
title: >-
  [论文解读] Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training
description: >-
  [ECCV 2024][视频理解][3D单目标跟踪] 本文提出了一个统一的3D单目标跟踪（SOT）框架，通过3D生成式预训练和2D预训练基础跟踪器的匹配知识蒸馏，解决了点云数据稀缺和LiDAR扫描稀疏不完整的问题，在KITTI、Waymo和nuScenes上达到SOTA性能。 领域现状：3D单目标跟踪是自动驾驶和机器人领域的…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "3D单目标跟踪"
  - "知识蒸馏"
  - "点云预训练"
  - "LiDAR"
  - "匹配学习"
---

# Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 视频理解 / 3D视觉  
**关键词**: 3D单目标跟踪, 知识蒸馏, 点云预训练, LiDAR, 匹配学习

## 一句话总结

本文提出了一个统一的3D单目标跟踪（SOT）框架，通过3D生成式预训练和2D预训练基础跟踪器的匹配知识蒸馏，解决了点云数据稀缺和LiDAR扫描稀疏不完整的问题，在KITTI、Waymo和nuScenes上达到SOTA性能。

## 研究背景与动机

**领域现状**：3D单目标跟踪是自动驾驶和机器人领域的核心任务，旨在给定初始帧中目标的3D边界框后，在后续帧中持续定位该目标。目前主流方法基于点云进行模板-搜索区域匹配，主要有两种范式：无记忆的Siamese方法和基于上下文记忆的方法。

**现有痛点**：学习鲁棒的3D SOT跟踪器面临两大挑战：（1）特定类别的点云训练数据有限，不像2D视觉领域有海量标注数据；（2）LiDAR扫描本身具有稀疏性和不完整性，远处或被遮挡的物体只有极少的点，导致匹配困难。这使得3D跟踪器的特征表示能力和匹配能力都受到严重制约。

**核心矛盾**：2D视觉领域已经有非常强大的预训练基础模型（如2D跟踪器），具有丰富的匹配知识和特征表示能力，但由于模态差异（2D图像 vs 3D点云），这些知识无法直接迁移到3D跟踪任务中。同时，3D点云数据的稀缺使得从头训练一个好的3D匹配器非常困难。

**本文目标** （1）如何利用2D预训练模型的匹配知识来增强3D跟踪器的匹配能力；（2）如何通过3D预训练来弥补点云数据不足的问题。

**切入角度**：作者观察到2D跟踪器在模板-搜索匹配上已经非常成熟，如果能让3D跟踪器学习2D跟踪器的匹配模式，就能有效提升3D匹配质量。关键在于设计一个不需要微调2D模型、同时能高效传递匹配知识的桥梁。

**核心 idea**：通过目标感知投影（TAP）将点云投影到2D平面供预训练2D跟踪器使用，再用IoU引导的匹配蒸馏框架将2D匹配知识迁移到3D跟踪器中。

## 方法详解

### 整体框架

整个方法框架包含两个核心组件：（1）3D生成式预训练，通过在大量点云数据上进行自监督预训练来获得更好的3D特征表示；（2）2D匹配知识蒸馏，利用预训练好的2D基础跟踪器作为教师，指导3D跟踪器学习更好的模板-搜索区域匹配。框架被应用于两种主流3D SOT范式：无记忆的Siamese方法（SiamDisst）和基于上下文记忆的方法（MemDisst）。

输入为连续帧的点云及初始帧的3D边界框，输出为当前帧中目标的3D边界框。中间过程依次经历：点云特征提取（预训练backbone）→ 3D模板-搜索匹配 → 同步进行2D投影和2D匹配（教师网络）→ 匹配蒸馏对齐 → 目标定位输出。

### 关键设计

1. **目标感知投影模块（Target-Aware Projection, TAP）**:

    - 功能：将3D点云投影到2D平面，生成适合2D预训练跟踪器处理的伪图像
    - 核心思路：不同于简单的正交投影，TAP会根据目标的位置和朝向来调整投影方向，确保投影后的2D表示能最大程度保留目标的结构信息。投影过程是轻量级的，不需要对2D跟踪器进行任何微调
    - 设计动机：直接投影点云会丢失很多关于目标的结构信息，特别是在稀疏区域。TAP通过感知目标的3D位置来选择最优投影视角，使得2D跟踪器能在投影后的结果上产生有意义的匹配响应

2. **IoU引导的匹配蒸馏框架**:

    - 功能：将2D预训练跟踪器的模板-搜索匹配知识迁移到3D跟踪器
    - 核心思路：核心约束是3D跟踪器的模板-搜索匹配应与对应的2D模板-搜索匹配保持一致。具体来说，将3D的匹配响应图和2D的匹配响应图进行对齐，使用IoU指标来引导蒸馏权重——IoU越高的区域，蒸馏权重越大，因为这些区域的2D匹配更可靠。蒸馏损失约束3D匹配分布向2D匹配分布靠拢
    - 设计动机：并非所有2D匹配结果都同样可靠，特别是在点云投影后存在遮挡或失真的区域。IoU引导可以自适应地选择可靠的蒸馏区域，避免错误的2D匹配知识被传递给3D跟踪器

3. **3D生成式预训练**:

    - 功能：通过自监督方式在大规模点云数据上预训练3D backbone，增强特征表示能力
    - 核心思路：采用生成式预训练范式（类似于MAE），对点云进行掩码并重建，让网络学习到通用的3D空间结构知识。预训练后的backbone作为3D跟踪器的特征提取器
    - 设计动机：3D跟踪任务的标注数据稀缺，通过预训练可以利用大量无标注的点云数据来提升特征质量，缓解数据不足的瓶颈

### 损失函数 / 训练策略

总训练损失包含：（1）跟踪定位损失：标准的3D边界框回归损失；（2）匹配蒸馏损失：约束3D匹配与2D匹配的一致性，使用IoU加权的KL散度；（3）预训练阶段使用重建损失。训练分为两个阶段：先进行3D生成式预训练，再进行跟踪器的端到端训练（含匹配蒸馏）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(SiamDisst) | 本文(MemDisst) | 之前SOTA | 提升 |
|--------|------|-----------------|----------------|----------|------|
| KITTI (Car) | Success/Precision | 高 | 高 | - | SOTA |
| Waymo Open Dataset | Success/Precision | 高 | 高 | - | SOTA |
| nuScenes | Success/Precision | 高 | 高 | - | SOTA |

在三个主流自动驾驶数据集上，SiamDisst和MemDisst均达到了SOTA性能。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 无预训练 | 下降 | 3D预训练显著提升基础特征质量 |
| 无匹配蒸馏 | 下降 | 2D匹配知识迁移效果显著 |
| 无IoU引导 | 下降 | IoU引导的选择性蒸馏优于全局蒸馏 |
| 无TAP | 下降 | 目标感知投影优于简单投影 |

### 关键发现

- SiamDisst在RTX3090上运行速度超过90 FPS，MemDisst也能达到25 FPS以上，满足实时性需求
- 两种框架设计（Siamese和Memory-based）都能从所提出的预训练和蒸馏策略中获益，证明了方法的通用性
- 2D到3D的匹配知识蒸馏在稀疏点云场景下特别有效，能显著改善远距离和遮挡目标的跟踪

## 亮点与洞察

- **跨模态知识迁移**：巧妙地将2D预训练模型的匹配能力迁移到3D跟踪器，TAP模块设计精巧，不需要微调2D模型
- **IoU引导蒸馏**：不是简单地对齐所有匹配响应，而是根据匹配质量自适应调整蒸馏强度，避免错误知识传播
- **实用性强**：方法兼容两种主流3D SOT框架，同时保持实时推理速度
- **一致性架构**：3D跟踪器采用与2D跟踪器一致的模板-搜索匹配架构，天然便于知识迁移

## 局限与展望

- TAP投影过程仍可能丢失3D空间中的重要信息，特别是沿投影方向的深度信息
- 对2D跟踪器的选择可能影响蒸馏效果，不同2D跟踪器的匹配模式可能有差异
- 在极度稀疏的场景下（如超远距离目标只有几个点），TAP投影后的2D表示可能过于稀疏，影响蒸馏质量
- 可以考虑多视角投影融合来进一步提升信息保留度

## 相关工作与启发

- **3D SOT方法**：P2B、BAT、M2Track等建立了3D SOT的基本框架，本文在此基础上加入了跨模态知识迁移
- **知识蒸馏**：2D到3D的知识蒸馏在检测任务中已有探索，本文将其扩展到跟踪任务的匹配学习
- **点云预训练**：Point-MAE、Point-BERT等自监督方法为3D预训练提供了基础
- 对视频/3D跟踪领域的启发：利用成熟的2D模型来辅助训练3D模型是一个有效的策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨模态匹配蒸馏思路新颖，TAP和IoU引导设计精巧
- 实验充分度: ⭐⭐⭐⭐ 三个数据集验证，消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，动机推导自然
- 价值: ⭐⭐⭐⭐ 为3D跟踪提供了有效的跨模态学习范式，实时性好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers](onetrack_demystifying_the_conflict_between_detection_and_tracking_in_end-to-end_.md)
- [\[ECCV 2024\] On the Utility of 3D Hand Poses for Action Recognition](on_the_utility_of_3d_hand_poses_for_action_recognition.md)
- [\[ECCV 2024\] CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner](crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)
- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](../../ICCV2025/video_understanding/an_empirical_study_of_autoregressive_pre-training_from_videos.md)
- [\[ECCV 2024\] DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video](dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)

</div>

<!-- RELATED:END -->
