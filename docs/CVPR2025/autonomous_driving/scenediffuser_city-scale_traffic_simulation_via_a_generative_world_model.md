---
title: >-
  [论文解读] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model
description: >-
  [CVPR 2025][自动驾驶][交通仿真] 提出 SceneDiffuser++，一个端到端的城市级交通仿真扩散模型，通过软裁剪（soft clipping）处理稀疏张量中的智能体出入场问题，实现 60 秒以上的行程级（trip-level）交通仿真，在 WOMD-XLMap 上达到 0.2423 综合 JS 散度。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "交通仿真"
  - "扩散模型"
  - "城市级"
  - "软裁剪"
  - "稀疏张量"
---

# SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model

**会议**: CVPR 2025  
**arXiv**: [2506.21976](https://arxiv.org/abs/2506.21976)  
**代码**: 无  
**领域**: 自动驾驶 / 交通仿真  
**关键词**: 交通仿真, 扩散模型, 城市级, 软裁剪, 稀疏张量

## 一句话总结

提出 SceneDiffuser++，一个端到端的城市级交通仿真扩散模型，通过软裁剪（soft clipping）处理稀疏张量中的智能体出入场问题，实现 60 秒以上的行程级（trip-level）交通仿真，在 WOMD-XLMap 上达到 0.2423 综合 JS 散度。

## 研究背景与动机

### 领域现状

**领域现状**：自动驾驶仿真需要逼真的交通场景生成。现有方法要么只能短时间仿真（<10 秒），要么不能处理智能体的出入场（车辆进入/离开场景）。城市级长时间仿真需要同时建模数百个智能体的行为、交通信号灯、遮挡关系和出入场动态。

**现有痛点**：稀疏张量是核心技术难题——不同智能体在不同时刻出现/消失，导致数据张量中有大量无效位置。传统做法用零填充或硬裁剪处理，但这在扩散模型推理中会导致生成质量下降。

**核心矛盾**：扩散模型在去噪过程中需要对所有位置同时操作，但稀疏张量的有效/无效位置需要差异化处理——有效位置需要精确去噪，无效位置需要保持为零。

**切入角度**：让模型同时预测特征值和有效性掩码，推理时用掩码软裁剪将无效位置平滑归零。

**核心 idea**：v-prediction 扩散 + 软裁剪处理稀疏智能体 + 多张量异构建模 = 端到端城市级交通仿真。

## 方法详解

### 关键设计

1. **软裁剪（Soft Clipping）**:

    - 功能：在扩散推理中优雅处理稀疏张量的有效/无效位置
    - 核心思路：模型同时预测特征值和有效性掩码 $M(x)$。推理时 $\hat{x}_t \leftarrow V(\hat{x}_t) \cdot M(\hat{x}_t)$——有效掩码接近 1 的位置保留特征值，接近 0 的位置被归零。与硬裁剪（二值化阈值）相比，软裁剪保持了可微性
    - 设计动机：硬裁剪在去噪早期的中间步骤中引入不连续，导致生成质量下降

2. **多张量异构场景建模**:

    - 功能：统一处理不同维度的场景元素
    - 核心思路：智能体（位置/朝向/速度 = 5 维）和交通信号灯（状态 = 4 维）作为不同张量处理，共享同一扩散模型但有各自的投影层
    - 设计动机：交通灯与智能体特征维度和语义完全不同，强行统一到同一维度会损失信息

### 损失函数 / 训练策略

v-prediction 扩散损失 $L = \mathbb{E}[\|(\tilde{v}_\theta - v_t) \cdot w\|_2^2]$，有效位置的权重更高。训练数据：WOMD-XLMap（1km 半径增广地图），600 步（60 秒）rollout，每 40 步重新规划。

## 实验关键数据

### 主实验

WOMD-XLMap 60s 仿真 JS 散度↓：

| 方法 | 综合分 | 交通灯违规 | 有效智能体 |
|------|--------|-----------|-----------|
| SceneDiffuser | ~0.29 | ~0.20 | ~0.35 |
| **SceneDiffuser++** | **0.2423** | **0.1625** | **0.3053** |

### 消融实验

| 裁剪策略 | 综合分 |
|---------|--------|
| 无裁剪 | 差 |
| 硬裁剪 | 中 |
| **软裁剪** | **最优** |

### 关键发现
- 软裁剪是稀疏张量扩散生成的关键——硬裁剪在去噪过程中引入伪影
- 40 步重规划是速度-质量最佳平衡点
- 超长仿真（3000步/5分钟）仍可行但性能有退化

## 亮点与洞察
- **首个端到端城市级行程仿真**——一个模型处理智能体行为/出入场/交通灯/遮挡
- **软裁剪的通用性**——可推广到任何稀疏张量上的扩散生成任务

## 局限与展望
- 仿真漂移在极长时间后仍存在
- 仅 WOMD 数据集验证
- 智能体可能在超长仿真中跑出地图

## 评分
- 新颖性: ⭐⭐⭐⭐ 软裁剪和多张量架构解决了实际问题
- 实验充分度: ⭐⭐⭐⭐ 多指标详细评估
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 推进了城市级交通仿真的实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)
- [\[CVPR 2026\] SimScale: Learning to Drive via Real-World Simulation at Scale](../../CVPR2026/autonomous_driving/simscale_learning_to_drive_via_real-world_simulation_at_scale.md)
- [\[CVPR 2025\] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)
- [\[CVPR 2025\] MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction](maskgwm_a_generalizable_driving_world_model_with_video_mask_reconstruction.md)

</div>

<!-- RELATED:END -->
