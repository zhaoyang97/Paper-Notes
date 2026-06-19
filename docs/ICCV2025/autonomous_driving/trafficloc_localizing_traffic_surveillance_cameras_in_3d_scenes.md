---
title: >-
  [论文解读] TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes
description: >-
  [ICCV 2025][自动驾驶][图像-点云配准] 提出 TrafficLoc，一种粗到细的图像-点云配准方法，通过几何引导注意力损失(GAL)、模态间-模态内对比学习(ICL)和稠密训练对齐(DTA)，实现交通监控相机在3D参考地图中的高精度定位，在自建 Carla Intersection 数据集上较 SOTA 提升达 86%。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "图像-点云配准"
  - "交通相机定位"
  - "跨模态特征融合"
  - "对比学习"
  - "6-DoF位姿估计"
---

# TrafficLoc: Localizing Traffic Surveillance Cameras in 3D Scenes

**会议**: ICCV 2025  
**arXiv**: [2412.10308](https://arxiv.org/abs/2412.10308)  
**代码**: [GitHub](https://tum-luk.github.io/projects/trafficloc/)  
**领域**: 自动驾驶  
**关键词**: 图像-点云配准, 交通相机定位, 跨模态特征融合, 对比学习, 6-DoF位姿估计

## 一句话总结
提出 TrafficLoc，一种粗到细的图像-点云配准方法，通过几何引导注意力损失(GAL)、模态间-模态内对比学习(ICL)和稠密训练对齐(DTA)，实现交通监控相机在3D参考地图中的高精度定位，在自建 Carla Intersection 数据集上较 SOTA 提升达 86%。

## 研究背景与动机
交通监控相机是协同感知中经济且易部署的路侧传感器，能提供广阔的全局交通视角。将其数据与车载传感器融合可增强态势感知，支持提前障碍物检测和车辆定位等应用。然而，交通相机定位面临三大挑战：

**大视角差异**：图像和3D参考点云在不同时间、不同视角下采集，传统配准方法所需的精确初始猜测难以获取

**跨模态匹配困难**：直接将点云投影到图像上会产生"bleeding problem"，模态间特征差异大

**内参变化**：交通相机通常使用变焦镜头，内参频繁变化

现有方法要么需要人工干预（如手动2D-3D特征匹配），要么依赖额外的全景图像或渲染图像，部署复杂度高。现有 I2P 配准方法（如 CoFiI2P、CFI2P）虽在车载相机上表现不错，但在交通路口大视角变化场景下性能急剧下降。

**核心 idea**：设计几何感知的跨模态特征融合模块，结合模态间-模态内对比学习和稠密训练对齐，实现一阶段训练的高精度交通相机定位。

## 方法详解

### 整体框架
TrafficLoc 采用粗到细的定位策略。给定一张交通相机图像和3D场景点云，首先分别提取2D图像 patch 特征和3D点组特征，通过几何引导特征融合（GFF）模块融合后进行粗匹配（patch-group级别），再进行精匹配（pixel-point级别），最后用 EPnP-RANSAC 估计6-DoF相机位姿。

### 关键设计
1. **双分支特征提取**:

    - 功能：分别提取图像和点云的多尺度特征
    - 核心思路：图像端使用 ResNet-18 提取多层特征，并利用 DUSt3R 的预训练 ViT 编码器增强空间关系，通过 $1 \times 1$ 卷积降维至 256 维。点云端使用 PointNet 提取逐点特征，通过 FPS 生成 $M$ 个超点并构建点组，再用 Point Transformer 增强局部几何特征
    - 设计动机：DUSt3R 具有强大的3D坐标回归能力，可有效编码图像的空间结构信息

2. **几何引导特征融合（GFF）+ Geometry-guided Attention Loss（GAL）**:

    - 功能：增强 Transformer 跨模态融合时的几何感知能力
    - 核心思路：在 Fusion Transformer 的最后一层交叉注意力上施加几何引导监督。对 I2P 注意力，根据相机射线 $OI_i$ 与点组中心 $P_j$ 之间的角度半径 $\text{Rad}(i,j)$ 构建指示函数：
    $\mathbb{1}_{I2P}(i,j) = \begin{cases} 1, & \text{if } \text{Rad}(i,j) < \theta_{low} \\ 0, & \text{if } \text{Rad}(i,j) > \theta_{up} \end{cases}$
      使用 BCE 损失监督原始交叉注意力图：$L_{I2P}(i,j) = \text{BCE}(\sigma(ATT_{I2P}(i,j)), \mathbb{1}_{I2P}(i,j))$。类似地，P2I 注意力使用点到相机射线的距离作为指示函数
    - 设计动机：直接使用 Transformer 做跨模态融合缺乏几何感知，在大视角变化下鲁棒性差。中间阈值区间不监督，允许网络灵活学习

3. **模态间-模态内对比学习（ICL）+ 稠密训练对齐（DTA）**:

    - 功能：增强粗匹配阶段的特征判别性和全局对齐能力
    - 核心思路：ICL 不仅拉近正样本对（匹配的 patch-group），还增大同模态内不同 patch/group 间的特征距离。损失函数：
    $L_{coarse}^S = \log[1 + \sum_j \exp^{\alpha_p(1 - s_p^j + m_p)} \sum_k \exp^{\alpha_n(s_n^k - m_n)}]$
      其中 $\alpha_p, \alpha_n$ 为自适应权重。DTA 通过 soft-argmax 对所有图像 patch 进行稠密的位置回归：$\hat{u}_x = \text{SoftArgmax}(S_x)$，使梯度传播到所有 patch
    - 设计动机：传统对比学习仅考虑跨模态对，忽略模态内特征差异；稀疏采样的 patch-group 对训练忽略了额外的全局特征

### 损失函数 / 训练策略
总损失为四部分的加权和：
$$L = \lambda_1 L_{Att} + \lambda_2 L_{det} + \lambda_3 L_{coarse} + \lambda_4 L_{fine}$$
- $L_{Att}$：几何引导注意力损失（BCE）
- $L_{det}$：视锥内检测损失（BCE）
- $L_{coarse} = L_{coarse}^S + L_{coarse}^D$：ICL + DTA 粗匹配损失
- $L_{fine} = L_{fine}^S + L_{fine}^D$：CE + L2 精匹配损失

## 实验关键数据

### 主实验

| 数据集 | 指标 | TrafficLoc | CoFiI2P (前SOTA) | 提升 |
|--------|------|-----------|-----------------|------|
| Carla Test_{T1-T7} | RRE(°) / RTE(m) | **0.66 / 0.51** | 4.24 / 2.82 | RRE↓85%, RTE↓82% |
| Carla Test_{T1-T7hard} | RRE(°) / RTE(m) | **2.64 / 1.13** | 7.87 / 5.34 | RRE↓66%, RTE↓78% |
| Carla Test_{T10} (unseen) | RRE(°) / RTE(m) | **2.53 / 2.69** | 17.78 / 7.43 | RRE↓86%, RTE↓64% |
| KITTI Odometry | RRE(°) / RTE(m) | **0.87 / 0.19** | 1.14 / 0.29 | RTE↓34% |
| NuScenes | RRE(°) / RTE(m) / RR(%) | **1.38 / 0.78 / 99.45** | 1.48 / 0.87 / 98.67 | - |

### 消融实验

| 配置 | RRE(°) | RTE(m) | 说明 |
|------|--------|--------|------|
| Baseline (NCL) | 1.53 | 0.82 | 仅用普通对比学习 |
| + ICL | 1.27 | 0.74 | RTE↓9.8% |
| + ICL + DTA | 1.01 | 0.62 | RRE↓20.5%, RTE↓16.2% |
| + ICL + DTA + CM | 0.84 | 0.62 | 加入粗匹配 |
| Full (+ FM + GAL) | **0.66** | **0.51** | GAL带来RTE↓17.7% |

### 关键发现
- GAL 使 P2I 注意力集中于点组投影区域，I2P 注意力沿相机射线分布，显著提升大视角变化场景性能
- ICL 有效增大模态内特征距离，使相似性矩阵分布更清晰
- DTA 消除稀疏监督导致的多峰问题，相似性图仅保留单个峰值
- 模型从 Carla 训练后直接迁移到真实 USTC 交叉口数据集，定性结果良好，证明 Sim2Real 泛化能力

## 亮点与洞察
- **新数据集**：Carla Intersection 提供 75 个交叉口、8 个世界，填补了交通相机定位数据集的空白
- **几何引导注意力**：将投影几何作为交叉注意力的监督信号，是一种通用的跨模态融合增强思路
- **ICL 的设计思想**：同时优化跨模态对齐和模态内判别性，比单纯的跨模态对比学习更有效
- 在未知内参场景下，可用 DUSt3R 预测内参进行初始化

## 局限与展望
- 当前方法假设点云已预处理完成（累积+下采样），实时性受限
- Carla 仿真数据与真实场景仍有域差距，虽然 Sim2Real 定性结果不错，但缺乏定量评估
- 推理速度（0.85s with GT K）可进一步优化
- 对于极端遮挡或低纹理场景的鲁棒性未充分评估

## 相关工作与启发
- 与 LoFTR 的粗到细策略相似，但扩展至跨模态（图像-点云）
- GAL 的几何指示函数设计可推广到其他需要几何感知的注意力机制中
- 可探索将 ICL 推广到其他跨模态匹配任务（如图像-文本、图像-音频）

## 评分
- 新颖性: ⭐⭐⭐⭐ GAL和ICL设计新颖，但整体框架属渐进式改进
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集验证+详细消融+可视化分析+Sim2Real测试
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ 交通相机定位是智慧交通的关键基础能力，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception](unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](../../AAAI2026/autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[ICCV 2025\] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation](long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)
- [\[CVPR 2025\] EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](../../CVPR2025/autonomous_driving/ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)
- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
