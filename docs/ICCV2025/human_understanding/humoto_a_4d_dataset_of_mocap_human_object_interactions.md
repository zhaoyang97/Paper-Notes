---
title: >-
  [论文解读] HUMOTO: A 4D Dataset of Mocap Human Object Interactions
description: >-
  [ICCV 2025][人体理解][人物交互数据集] 提出 HUMOTO，一个高保真 4D 人物交互数据集，包含 735 段序列（7875 秒，30fps），涵盖 63 个精确建模物体和 72 个可动部件，创新性地使用 LLM 驱动的场景脚本生成流程和多传感器捕获系统，在手部姿态精度和交互质量上显著超越现有数据集。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "人物交互数据集"
  - "动作捕捉"
  - "手部姿态"
  - "多物体交互"
  - "LLM脚本生成"
---

# HUMOTO: A 4D Dataset of Mocap Human Object Interactions

**会议**: ICCV 2025  
**arXiv**: [2504.10414](https://arxiv.org/abs/2504.10414)  
**代码**: [https://jiaxin-lu.github.io/humoto/](https://jiaxin-lu.github.io/humoto/)  
**领域**: 人体理解 / 人物交互  
**关键词**: 人物交互数据集, 动作捕捉, 手部姿态, 多物体交互, LLM脚本生成

## 一句话总结

提出 HUMOTO，一个高保真 4D 人物交互数据集，包含 735 段序列（7875 秒，30fps），涵盖 63 个精确建模物体和 72 个可动部件，创新性地使用 LLM 驱动的场景脚本生成流程和多传感器捕获系统，在手部姿态精度和交互质量上显著超越现有数据集。

## 研究背景与动机

4D 人物交互（HOI）数据对计算机视觉、机器人、图形学和生成式 AI 至关重要。然而，现有数据集存在明显不足：

**单物体局限**：大多数数据集（GRAB、BEHAVE、OMOMO 等）仅涉及单物体交互

**手部缺失**：许多数据集缺乏精细手部运动（如 BEHAVE、OMOMO 仅使用标准手模板）

**语义断裂**：现有序列往往是孤立的、无目的的动作，缺乏连贯的任务逻辑

**交互质量差**：手与物体经常出现严重穿透或完全脱离

获取高质量 4D HOI 数据非常昂贵，需要复杂的动捕设备和大量人工清理。HUMOTO 旨在填补多物体、全身+手部、有意义任务序列的数据空白。

## 方法详解

### 整体框架

数据采集流程：场景设计 → LLM 脚本生成 → 动捕+相机录制 → 数据处理 → 多阶段质量保证 → 文本标注

### 关键设计

1. **Scene-Driven LLM Scripting（场景驱动LLM脚本）**：借鉴电影制作流程，先将 63 个物体按逻辑分组为"房间"（如厨房、书房），再将物体分组提供给 LLM 生成连贯的交互脚本。LLM 层次化生成：先确定场景主题，再细化为具体动作序列（如"打开抽屉取出物品→在桌上整理→准备餐点"），确保每个动作有明确目的和自然衔接，避免孤立无意义的动作。

2. **多传感器捕获系统**：

    - **人体动捕**：Rokoko 智能动捕衣 + 配套手套，30fps 惯性传感器网络追踪全身骨骼和手指关节
    - **物体追踪**：双 Kinect RGB-D 相机最大化覆盖、最小化遮挡，使用 FoundationPose 算法获取 6DoF 物体姿态
    - **定制环境**：木制抬高舞台（减少金属对惯性传感器的磁干扰），双电脑 UDP 时间同步
    - **遮挡处理**：SAM2 配合人工标注提供物体掩码，动态重置机制应对快速运动导致的追踪丢失

3. **多阶段质量保证**：

    - **技术精修**：专业动画师修正捕获伪影（漂移、追踪误差），确保交互逻辑一致性
    - **独立验证**：另一组团队验证自然合理的交互，修复关节抖动、足滑等问题
    - 两个阶段迭代直到所有质量标准满足
    - **文本标注**：三层标注——简短标题、短脚本、详细长脚本

### 损失函数 / 训练策略

本文为数据集论文,无训练损失。但提出了一套 HOI 数据集质量评估指标体系：
- **人体运动**：Foot Sliding（足滑距离）、Jerk（加加速度平滑度）、MSNR（运动信噪比，以 Mixamo 为基线）、Coherence（运动连贯性）、Diversity（运动多样性）
- **物体运动**：Jerk（操作平滑度）
- **交互质量**：Penetration（穿透深度）、Contact Entropy（接触状态分布多样性）、State Consistency（状态一致性）

## 实验关键数据

### 主实验（数据集质量对比）

| 数据集 | 足滑(cm)↓ | Jerk(m/s³)↓ | MSNR(dB)→ | 物体Jerk↓ | 穿透(cm)↓ | 接触熵↑ |
|--------|----------|-------------|-----------|----------|----------|---------|
| BEHAVE | 4.556 | 4.08 | 5.51 | 10.40 | 0.0606 | 2.2915 |
| OMOMO | 2.130 | 15.10 | 12.37 | 27.40 | 0.0602 | 1.9468 |
| IMHD | 1.474 | 1.14 | 14.20 | 24.06 | 0.1172 | 2.4265 |
| ParaHome | 3.008 | 9.19 | 1.82 | 0.08 | 0.2167 | 1.0254 |
| **HUMOTO** | **0.958** | **1.87** | **9.42** | **1.13** | **0.0068** | **1.4587** |
| Mixamo (参考) | 3.184 | 8.14 | 10.88 | - | - | - |

### 消融实验（数据集统计对比）

| 数据集 | 时长(h) | 人数 | 物体数 | 手部 | 身体 | 最大场景物体数 | 采集方式 |
|--------|--------|------|--------|------|------|--------------|---------|
| GRAB | 3.8 | 10 | 51 | ✓ | ✓ | 1 | 站立 |
| BEHAVE | 4.2 | 8 | 20 | ✗ | ✓ | 1 | 便携 |
| OMOMO | 10.1 | 17 | 15 | ✗ | ✓ | 1 | 便携 |
| ParaHome | 8.1 | 38 | 22 | ✓ | ✓ | 22 | 房间 |
| **HUMOTO** | **2.2** | **1** | **63** | **✓** | **✓** | **15** | **场景** |

### 关键发现

1. **足滑最小**：HUMOTO 的 0.958cm 远优于所有其他数据集（次优 IMHD 为 1.474cm），归功于严格的动捕流程和专业清理
2. **穿透最低**：0.0068cm 的穿透深度比 BEHAVE（0.0606cm）和 OMOMO（0.0602cm）低一个数量级，包含精细手部姿态的情况下依然保持极佳精度
3. **物体操作自然**：物体 Jerk 仅 1.13，远低于 OMOMO（27.40）和 IMHD（24.06），表明物体操作平滑且真实
4. **MSNR 接近 Mixamo 基准**：9.42dB 接近专业动画数据 Mixamo 的 10.88dB
5. **感知评估**：82% 受试者给 HUMOTO 最高质量评分，96% 在整体质量上更偏好 HUMOTO 而非 BEHAVE

## 亮点与洞察

- LLM 场景脚本生成是数据采集流程的创新：将电影分镜思路引入动捕规划，确保序列的语义连贯性和任务完整性
- 多传感器 + 木制舞台 + 电磁场动捕衣的组合是实用的工程方案，有效解决了遮挡和磁干扰问题
- 穿透深度指标比其他数据集低一个数量级，对手部交互建模的精度提升非常关键
- 下游应用展示丰富：动作生成（MotionGPT 在 HUMOTO 提示上效果不佳说明挑战性）、机器人抓取（与 DexGraspNet 对比）、姿态估计（4D Humans/TRAM 均失败）

## 局限与展望

- **单一表演者**：仅 1 名演员，可能引入体型和运动风格偏差，限制了跨人泛化
- **总时长有限**：2.2 小时虽质量高但规模不算大，小于 OMOMO（10.1h）和 ParaHome（8.1h）
- **人工清理成本高**：每段序列都需专业动画师多轮清理验证，难以规模化
- **物体非实际扫描**：使用艺术家建模而非 3D 扫描，可能存在与真实物体几何的微妙差异
- 未提供力/力矩等接触力学信息，对需要力反馈的机器人学习有局限

## 相关工作与启发

- **GRAB**：全身 HOI 数据集先驱，但仅上半身、单物体、站立姿态
- **BEHAVE/OMOMO**：更复杂场景但缺手部姿态
- **ParaHome**：家居场景多物体交互，但手部贴标签干扰自然运动
- **FoundationPose**：用于物体 6DoF 追踪，效果好但需动态重置机制辅助
- **SAM2**：辅助生成物体分割掩码，提升追踪鲁棒性

## 评分

- **新颖性**: ⭐⭐⭐⭐ LLM脚本驱动和多传感器方案新颖，但核心贡献是数据集而非算法
- **实验充分度**: ⭐⭐⭐⭐ 定量指标全面，感知评估扎实，下游应用展示丰富
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，采集流程描述详尽，图表直观
- **价值**: ⭐⭐⭐⭐⭐ 填补多物体精细手部HOI数据空白，对动作生成、机器人和具身AI有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions](../../CVPR2026/human_understanding/interprior_scaling_generative_control_for_physics-based_human-object_interaction.md)
- [\[ICCV 2025\] MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation](mdd_a_dataset_for_text-and-music_conditioned_duet_dance_generation.md)
- [\[CVPR 2026\] HUMAPS-4D: A Multimodal Dataset for HUman Motion Analysis with Physiological and Semantic informations](../../CVPR2026/human_understanding/humaps-4d_a_multimodal_dataset_for_human_motion_analysis_with_physiological_and_.md)
- [\[ICCV 2025\] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras](udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
