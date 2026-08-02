---
title: >-
  [论文解读] ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate
description: >-
  [CVPR 2025][自动驾驶][人体运动恢复] 构建了首个大规模攀岩运动多模态数据集 AscendMotion（412K帧，RGB+LiDAR+IMU），并提出 ClimbingCap 方法通过分离坐标解码、后处理优化和半监督训练，在世界坐标系中精确恢复攀岩者的3D运动。 1. 领域现状：人体运动恢复（HMR）研究主要关…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "人体运动恢复"
  - "攀岩"
  - "多模态"
  - "LiDAR"
  - "RGB"
  - "SMPL"
---

# ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate

**会议**: CVPR 2025  
**arXiv**: [2503.21268](https://arxiv.org/abs/2503.21268)  
**代码**: [http://www.lidarhumanmotion.net/climbingcap/](http://www.lidarhumanmotion.net/climbingcap/)  
**领域**: 自动驾驶(人体运动)  
**关键词**: 人体运动恢复, 攀岩, 多模态, LiDAR, RGB, SMPL

## 一句话总结

构建了首个大规模攀岩运动多模态数据集 AscendMotion（412K帧，RGB+LiDAR+IMU），并提出 ClimbingCap 方法通过分离坐标解码、后处理优化和半监督训练，在世界坐标系中精确恢复攀岩者的3D运动。

## 研究背景与动机

1. **领域现状**：人体运动恢复（HMR）研究主要关注跑步、行走等**地面运动**。现有方法如 WHAM、GVHMR 等在估计全局轨迹时假设人体主要沿水平方向运动，这一假设对地面活动成立但无法推广到其他场景。

2. **现有痛点**：攀岩是一种**离地运动**——攀岩者用手脚攀附岩壁向上移动，涉及极端的肢体伸展和全身发力。现有全局 HMR 方法在攀岩场景下存在两个核心问题：(1) 无法准确恢复攀岩者在世界坐标系中的位置（尤其是垂直方向），(2) 相机坐标和全局坐标之间的转换存在固有歧义。

3. **核心矛盾**：攀岩运动数据集严重不足。此前仅有 SPEED21（2D，46K帧）和 CIMI4D（3D，180K帧，普通攀岩者）两个公开数据集，规模小且动作不够挑战性。数据不足导致社区对攀岩运动缺乏深入理解。

4. **本文要解决什么？** (1) 构建大规模、高质量的攀岩运动数据集；(2) 设计能在全局坐标系中精确恢复攀岩运动的方法；(3) 有效利用容易获取的无标注攀岩数据。

5. **切入角度**：RGB 和 LiDAR 两种模态各有优势——RGB 适合估计相机坐标下的姿态，LiDAR 的 3D 点云天然提供世界坐标信息。将两者在不同坐标系下分别解码再联合优化，可以有效解决坐标系歧义问题。

6. **核心 idea 一句话**：用 RGB+LiDAR 多模态分别在相机坐标和全局坐标解码姿态，再通过场景感知后处理和半监督训练纠正攀岩运动的全局轨迹。

## 方法详解

### 整体框架

ClimbingCap 由三个阶段组成：(1) 分离坐标解码（SCD）——RGB 估计相机坐标下的 SMPL 参数，LiDAR 估计全局坐标下的平移参数；(2) 后处理优化——通过三个专用损失函数联合优化两个坐标系的结果；(3) 半监督训练——用教师-学生框架利用大量无标注攀岩数据。

### 关键设计

1. **分离坐标解码（Separate Coordinate Decoding）**:
    - 做什么：分别在相机坐标和全局坐标系中预测人体运动参数
    - 核心思路：点云通过外参矩阵 $\mathcal{P}_c = \Omega_{w2c} \cdot \mathcal{P}_w$ 从世界坐标转到相机坐标。RGB 图像通过 ViT 提取视觉特征，点云通过 PointNet++ 提取几何特征。Camera Coordinate Decoder 迭代更新 SMPL 姿态 $\theta$、体型 $\beta$ 和相机平移 $\Delta c$；Global Coordinate Decoder 迭代更新全局平移 $\Gamma^{trans}$
    - 设计动机：相机坐标和全局坐标的信息来源不同（RGB 擅长姿态，LiDAR 擅长定位），分开解码可各取所长

2. **后处理优化（三个损失函数）**:
    - 做什么：在世界坐标系中联合优化 SCD 阶段输出的姿态和位置
    - **Global Refit Loss ($\mathcal{L}_{GR}$)**：计算 SMPL 顶点与人体点云之间的加权 Chamfer 距离，对躯干和四肢施加不同距离阈值（{limb} \leq d_{torso}$，因为攀岩时四肢离岩壁更近）
    - **Scene Touch Loss ($\mathcal{L}_{ST}$)**：防止 SMPL 模型穿透岩壁场景网格，计算穿透深度 $\eta(v_i) = (v_i - q_j) \cdot n_j$，穿透时施加惩罚
    - **Velocity Smoothing Loss ($\mathcal{L}_{VLR}$)**：约束肢体运动速度方向的平滑性，纠正异常的肢体预测
    - 设计动机：攀岩者紧贴岩壁运动，人-场景交互约束是不可或缺的先验；LiDAR 提供了世界坐标 3D 信息使得后处理成为可能

3. **半监督训练**:
    - 做什么：利用无标注攀岩数据提升模型性能
    - 核心思路：SCD+后处理训练好的模型作为教师模型，克隆参数给学生模型。教师模型对无标注数据生成伪标签，学生模型在伪标签上进一步训练
    - 设计动机：标注攀岩数据成本高（需要 IMU MoCap + 人工校正），而无标注攀岩视频/点云容易获取。AscendMotion 包含 441 分钟无标注数据（vs 344 分钟有标注）

### 损失函数 / 训练策略

- SCD 阶段总损失：$\mathcal{L} = \mathcal{L}_{kp3d} + \mathcal{L}_{kp2d} + \mathcal{L}_\theta^{smpl} + \mathcal{L}_\beta^{smpl} + \mathcal{L}_{traj}$
- 后处理阶段用 Adam 优化器优化全局姿态
- 半监督阶段使用教师-学生框架

## 实验关键数据

### 主实验（AscendMotion 水平场景/垂直场景）

| 方法 | 模态 | MPJPE↓ | PA-MPJPE↓ | PCK@0.3↑ | WA-MPJPE↓ | W-MPJPE↓ | RTE↓ |
|------|------|--------|-----------|----------|-----------|----------|------|
| TRACE | RGB | 875.56/577.60 | 69.21/85.81 | 0.06/0.09 | 144.33/385.71 | 254.38/703.35 | 14.73/26.17 |
| GVHMR | RGB | 107.09/124.60 | 60.06/80.30 | 0.77/0.71 | 105.15/1002.11 | 202.45/1442.50 | 4.09/7.91 |
| WHAM | RGB | 110.92/143.17 | 76.09/73.36 | 0.76/0.62 | 229.42/1125.77 | 647.70/1499.85 | 5.16/9.04 |
| LiDARCapV2 | LiDAR | 244.60/234.52 | 192.17/156.39 | 0.53/0.50 | 282.12/1396.42 | 442.12/1518.29 | 16.42/10.85 |
| LEIR | L+R | 297.95/299.62 | 187.26/150.56 | 0.41/0.37 | 266.82/1313.09 | 282.31/1435.92 | 9.78/9.97 |
| **ClimbingCap** | **L+R** | **75.45/88.92** | **61.73/74.50** | **0.91/0.78** | **62.95/85.26** | **78.99/106.95** | **1.57/3.12** |

ClimbingCap 大幅领先所有基线，尤其在垂直场景中优势更为显著（GVHMR 的 W-MPJPE 高达 1442mm，ClimbingCap 仅 107mm）。零样本迁移到 CIMI4D 数据集同样最优（MPJPE 84.03mm）。

### 消融实验

| 配置 | MPJPE↓ | PA-MPJPE↓ | PCK@0.3↑ | WA-MPJPE↓ | W-MPJPE↓ | RTE↓ |
|------|--------|-----------|----------|-----------|----------|------|
| (1) 仅 RGB 输入 | 105.67 | 63.05 | 0.78 | 117.17 | 174.53 | 7.64 |
| (2) w/o $\mathcal{L}_{LWD}$ | 80.46 | 52.15 | 0.89 | 70.04 | 91.23 | 2.02 |
| (3) w/o $\mathcal{L}_{SDS}$ | 99.13 | 60.66 | 0.81 | 109.35 | 164.11 | 7.03 |
| (4) w/o $\mathcal{L}_{VLR}$ | 91.10 | 61.85 | 0.83 | 88.59 | 120.11 | 3.34 |
| (5) w/o 半监督 | 77.43 | 52.57 | 0.90 | 65.30 | 82.09 | 1.83 |
| **完整模型** | **75.45** | **50.51** | **0.91** | **62.95** | **78.99** | **1.57** |

### 关键发现

- **LiDAR 不可或缺**：去掉 LiDAR 后 MPJPE 从 75.45 升至 105.67（+40%），全局轨迹指标更是大幅恶化（W-MPJPE 翻倍以上）
- **速度方向平滑损失 $\mathcal{L}_{SDS}$ 最关键**：去掉后 MPJPE 从 75.45 升至 99.13，影响最大，说明运动一致性约束对攀岩至关重要
- **场景触碰损失贡献显著**：$\mathcal{L}_{VLR}$ 去掉后 MPJPE 升至 91.10，全局轨迹严重退化
- **半监督有帮助但贡献有限**：去掉后 MPJPE 仅上升 2mm（75.45→77.43），说明主要改进来自方法设计而非数据量

## 亮点与洞察

- **数据集构建的匠心**：AscendMotion 的标注流程（PTP 时钟同步 + multi-stage 全局优化 + 人工校正）设计精良。22 名专业攀岩教练的数据比 CIMI4D 的业余攀岩者动作更具挑战性——数据集质量胜过单纯追求规模
- **坐标系分离解码**：将相机坐标和全局坐标解码分离是一个巧妙设计。RGB 擅长姿态估计但缺乏深度信息，LiDAR 天然提供 3D 全局位置——让不同模态做各自擅长的事，在后处理阶段再统一
- **场景感知损失设计**：Scene Touch Loss 利用攀岩者紧贴岩壁的物理先验，将人-场景交互编码为优化约束，是领域知识的有效利用

## 局限性 / 可改进方向

- 数据采集硬件要求高（LiDAR + RGB + IMU + 高精度扫描仪），难以大规模推广
- 方法假设有精确的外参矩阵进行坐标系转换，野外场景标定可能不可靠
- 半监督训练的贡献有限（~2mm），可能需要更先进的伪标签策略
- 未考虑攀岩者手-脚抓握岩点的精细交互建模

## 相关工作与启发

- **vs GVHMR**: GVHMR 通过预测水平速度估计全局运动方向，在垂直攀岩场景完全失效（W-MPJPE 高达 1442mm），说明地面运动假设是硬伤
- **vs WHAM**: WHAM 在相机坐标指标尚可，但全局坐标指标极差，进一步证实仅用 RGB 无法可靠估计离地运动的全局位置
- **vs CIMI4D 数据集**: AscendMotion 在规模（412K vs 180K帧）、多样性（22人12面墙 vs 12人）、挑战性（专业教练 vs 业余者）三方面全面超越

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性解决攀岩运动恢复问题，数据集和方法都是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集交叉验证、零样本泛化测试、详细消融实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数据集构建描述详尽
- 价值: ⭐⭐⭐⭐ AscendMotion 数据集对社区有长期价值，方法设计有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation](lisu_a_dataset_and_method_for_lidar_surface_normal_estimation.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[CVPR 2025\] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method](towards_satellite_image_road_graph_extraction_a_global-scale_dataset_and_a_novel.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)

</div>

<!-- RELATED:END -->
