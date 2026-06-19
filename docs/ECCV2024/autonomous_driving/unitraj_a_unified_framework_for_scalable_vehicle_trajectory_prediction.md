---
title: >-
  [论文解读] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction
description: >-
  [ECCV 2024][自动驾驶][轨迹预测] UniTraj 构建了一个统一多数据集（nuScenes、Argoverse 2、WOMD）、多模型（AutoBot、MTR、Wayformer）和多评估策略的车辆轨迹预测框架，揭示模型跨数据集泛化能力显著下降，但通过扩大数据规模和多样性可大幅提升性能，合并训练在 nuScenes 排行榜达到第 1 名。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "轨迹预测"
  - "统一框架"
  - "跨数据集泛化"
  - "数据规模扩展"
  - "多数据集训练"
---

# UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction

**会议**: ECCV 2024  
**arXiv**: [2403.15098](https://arxiv.org/abs/2403.15098)  
**代码**: [https://github.com/vita-epfl/UniTraj](https://github.com/vita-epfl/UniTraj)  
**领域**: 自动驾驶  
**关键词**: 轨迹预测, 统一框架, 跨数据集泛化, 数据规模扩展, 多数据集训练

## 一句话总结
UniTraj 构建了一个统一多数据集（nuScenes、Argoverse 2、WOMD）、多模型（AutoBot、MTR、Wayformer）和多评估策略的车辆轨迹预测框架，揭示模型跨数据集泛化能力显著下降，但通过扩大数据规模和多样性可大幅提升性能，合并训练在 nuScenes 排行榜达到第 1 名。

## 研究背景与动机

## 研究背景与动机
车辆轨迹预测是自动驾驶安全和碰撞避免的核心任务。随着深度学习的发展，数据驱动的预测模型已能取得很高的精度，但严重依赖特定训练域。自动驾驶系统可能遇到不同地理位置、交通规则、道路布局等多样场景，这些域偏移会显著影响模型性能。

**三大数据集兼容性障碍**：
- **数据格式不同**：WOMD 用 TFRecord，Argoverse 2 用 Apache Parquet，无法直接混合使用
- **数据特征差异大**：时间长度（8-20秒）、地图精度（0.2-2m）、采样率、标注类型各异
- **评估指标不统一**：WOMD 用 mAP，Argoverse 2 用 brier-minFDE，无法直接比较

**两个核心研究问题**：
- **RQ1**：轨迹预测模型跨数据集/城市的泛化能力如何？性能下降多少？
- **RQ2**：增大训练数据规模和多样性能否提升预测性能？有多大提升空间？

**本文切入角度**：构建 UniTraj 统一框架，从数据格式、特征、模型到评估全面标准化，使多数据集联合训练和系统性泛化研究首次成为可能。

## 方法详解

### 整体框架

UniTraj 由三大组件构成：(1) **统一数据**——统一多数据集的格式和特征；(2) **统一模型**——适配多个 SOTA 预测模型到统一数据格式；(3) **统一评估**——提供通用和细粒度评估指标。框架构建了最大的公开车辆轨迹预测数据集，支持跨数据集训练、评估和分析。

### 关键设计

1. **统一数据格式 (Unified Data Format)**: 基于 ScenarioNet 将不同格式（TFRecord、Apache Parquet 等）转换为统一场景描述格式。ScenarioNet 原为交通仿真设计，本文将其重新用于轨迹预测任务，并扩展了对 Argoverse 2 的支持。这消除了为每个数据集编写独立预处理代码的需求。

2. **统一数据特征 (Unified Data Features)**: 系统性处理四类差异：

   | 特征 | Argoverse 2 | WOMD | nuScenes | UniTraj 统一方案 |
   |------|-------------|------|----------|-----------------|
   | 坐标系 | 场景中心 | 场景中心 | 场景中心 | **Agent-centric** |
   | 历史时长 | 5s | 1s | 2s | [0-8]s 可配置 |
   | 未来时长 | 6s | 8s | 6s | [1-8]s 可配置 |
   | 地图精度 | 0.2-2m | ~0.5m | ~1m | 0.5m（线性插值标准化） |
   | 地图范围 | ~200m | ~200m | ~500m | [0-500]m 可配置 |

   核心处理包括：
    - **坐标转换**: 将场景级原始数据转换为 agent-centric 向量化格式
    - **时间对齐**: 截断所有轨迹为统一 8 秒长度，灵活配置历史/未来分割
    - **智能体特征**: 统一为 2D 坐标 + 速度 + 航向，补充加速度和 agent 类型 one-hot 编码
    - **地图特征**: 通过线性插值标准化连续点间距为 0.5m，支持降采样，丰富车道方向和类型编码

3. **统一模型平台 (Unified Models)**: 集成三个具有代表性的 SOTA 模型，覆盖不同参数量级：

    - **AutoBot** (1.5M params): Transformer-based，等变特征学习 + 多头注意力
    - **MTR** (60.1M params): 2022 WOMD Challenge 冠军，全局意图先验 + 局部运动细化
    - **Wayformer** (16.5M params): 多轴编码器 + 潜在查询（本文重新实现）

   通过标准化输出格式，新模型可无缝对接 UniTraj 的评估和日志工具。

4. **统一评估策略 (Unified Evaluation)**: 包含两层评估：

    - **通用指标**: minADE/minFDE（最小平均/终点位移误差）、Miss Rate（minFDE>2m 的比例）、brier-minFDE（加入概率惩罚项 $(1-p)^2$）
    - **细粒度评估**:
        - *轨迹类型分层*: 按 WOMD 分类法将轨迹分为 8 类（stationary、straight、left-turn 等），专门评估稀有但安全关键的类型
        - *Kalman 难度分层*: 用 Kalman 滤波器 FDE 量化场景难度，专门评估模型在困难场景上的表现

### 损失函数 / 训练策略

- 各模型保持原始配置和超参数不变
- 地图范围统一为 100m 半径，空间分辨率 0.5m
- 历史轨迹 2 秒，预测未来 6 秒
- 多数据集训练时直接合并所有数据集的训练集
- 支持多进程处理和缓存机制实现高效数据处理

## 实验关键数据

### 跨数据集泛化实验 (RQ1)

| 训练数据 | #trajs | 评估 nuScenes↓ | 评估 Argoverse 2↓ | 评估 WOMD↓ |
|---------|--------|---------------|-------------------|-----------|
| nuScenes | 32k | **2.86** | 4.50 | 7.38 |
| Argoverse 2 | 180k | 3.72 | **2.08** | 4.68 |
| WOMD | 1800k | 3.10 | 3.63 | **2.13** |
| **All** | **2012k** | **2.27** | **1.99** | **2.13** |

（以 MTR 模型 brier-minFDE 为例，所有模型趋势一致）

### 跨城市泛化实验

| 训练城市 | 评估 Pittsburgh↓ | 评估 Boston↓ | 评估 Singapore↓ | 平均↓ |
|---------|-----------------|-------------|----------------|------|
| Pittsburgh | **2.4** | 2.7 | 3.5 | 2.8 |
| Boston | 4.1 | **2.2** | 3.4 | 3.2 |
| Singapore | 4.9 | 3.5 | **2.1** | 3.5 |

### nuScenes 排行榜

| 方法 | 排名 | minADE5↓ |
|------|------|---------|
| **MTR-UniTraj** | **1** | **0.96** |
| Goal-LBP | 2 | 1.02 |
| CASPNet++ | 3 | 1.16 |
| AutoBot-UniTraj | 11 | 1.26 |
| AutoBot | 19 | 1.37 |

### 关键发现

- **泛化性差**: 所有模型在跨数据集评估时性能显著下降，MTR 在 nuScenes 训练评估 WOMD 时 brier-minFDE 从 2.13 恶化到 7.38
- **WOMD 泛化最好**: 跨数据集泛化能力排序始终为 WOMD > Argoverse 2 > nuScenes，即使控制数据集大小相同也成立，说明多样性比规模更重要
- **数据规模提升显著**: 联合训练使 MTR 在 nuScenes 上 brier-minFDE 从 2.86 降至 2.27（提升 20.6%），登顶排行榜
- **模型容量影响收益**: MTR (60.1M params) 比 AutoBot (1.5M) 从更大数据中获益更多
- **多样性是关键**: WOMD 的轨迹类型分布最均匀（左/右转弯数量是其他数据集的 2 倍），解释了其更好的泛化能力
- **Singapore 泛化最差**: 左行交通城市的模型在右行城市上泛化差，强调地理多样性的重要性

## 亮点与洞察

- **系统性贡献**: 首个开源的车辆轨迹预测统一框架，解决了长期困扰社区的数据集兼容性问题
- **数据多样性 > 数据规模**: 控制数据量实验表明，WOMD 更好的泛化不仅因为数据更多，更因为场景更多样化
- **Kalman 难度的洞察**: 通过 Kalman 滤波器基线区分简单/困难样本，发现联合训练在中等难度样本上提升最大（因为合并后中等难度样本增多）
- **轨迹类型分析**: 联合训练在所有轨迹类型上都有提升，尤其是 right u-turn（从 8.13 降至 2.98），说明稀有类型的数据补充至关重要
- **实用工程价值**: 框架支持一键集成新数据集和新模型，多进程和缓存机制保证效率

## 局限与展望

- 当前仅支持车辆轨迹预测，未涵盖行人和骑行者
- nuPlan 因缺乏官方预测任务训练/验证集划分而仅用于跨城市实验
- 数据集间残留差异（如标注噪声）可能影响公平比较
- 联合训练时简单地合并数据集，未探索更精细的数据混合策略（如上采样稀有类型）
- 仅评估了三个模型，未涵盖最新的基于 LLM 或扩散模型的方法
- 未来可探索 domain adaptation 或 continual learning 方法来进一步提升泛化

## 相关工作与启发

- **ScenarioNet**: 提供统一场景描述格式，本文将其从仿真扩展到轨迹预测
- **Trajnet++**: 行人轨迹预测的统一基准，本文是车辆轨迹的首个统一框架
- **trajdata**: 统一人类轨迹数据集接口，启发了 UniTraj 的设计思路
- **启发**: 统一框架不仅是工程工具，更能催生"数据多样性 vs 规模"等新研究问题；22M 样本仍未到性能天花板，更大规模数据集的构建有巨大价值

## 评分

- 新颖性: ⭐⭐⭐ 框架构建为主，方法创新有限，但系统级贡献价值大
- 实验充分度: ⭐⭐⭐⭐⭐ 跨数据集、跨城市、数据规模、细粒度分析、控制变量实验极为充分
- 写作质量: ⭐⭐⭐⭐ 两个 RQ 主线清晰，实验分析逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 开源框架对社区有巨大推动价值，数据多样性的洞察对数据集构建有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RAG-TP: A General Framework for Vehicle Trajectory Prediction via Retrieval-Augmented Generation](../../CVPR2026/autonomous_driving/rag-tp_a_general_framework_for_vehicle_trajectory_prediction_via_retrieval-augme.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[ECCV 2024\] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)
- [\[NeurIPS 2025\] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning](../../NeurIPS2025/autonomous_driving/unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)

</div>

<!-- RELATED:END -->
