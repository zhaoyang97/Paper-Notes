---
title: >-
  [论文解读] LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos
description: >-
  [CVPR 2025][乒乓球预测] LATTE-MV 提出一套从单目乒乓球比赛视频中重建 3D 比赛数据的可扩展系统，并训练 Transformer 模型预判对手击球意图，结合共形预测实现不确定性感知的预判式控制，将仿真中机器人回球率从 49.9% 提升至 59.0%。 1. 领域现状：乒乓球机器人是人形机器人研究的经典试…
tags:
  - "CVPR 2025"
  - "乒乓球预测"
  - "单目视频3D重建"
  - "预判式控制"
  - "Transformer"
  - "共形预测"
---

# LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos

**会议**: CVPR 2025  
**arXiv**: [2503.20936](https://arxiv.org/abs/2503.20936)  
**代码**: [https://sastry-group.github.io/LATTE-MV/](https://sastry-group.github.io/LATTE-MV/) (项目页)  
**领域**: 其他  
**关键词**: 乒乓球预测, 单目视频3D重建, 预判式控制, Transformer, 共形预测

## 一句话总结
LATTE-MV 提出一套从单目乒乓球比赛视频中重建 3D 比赛数据的可扩展系统，并训练 Transformer 模型预判对手击球意图，结合共形预测实现不确定性感知的预判式控制，将仿真中机器人回球率从 49.9% 提升至 59.0%。

## 研究背景与动机

1. **领域现状**：乒乓球机器人是人形机器人研究的经典试验台，近年来深度学习驱动的系统已能实现与人类的合作/对抗级打球。但现有系统面对高速击球表现不佳，核心原因是缺乏对对手意图的预判能力。

2. **现有痛点**：之前利用预判的研究（如 IDDM 或 LSTM 预测发球落点）受限于数据集规模过小（均 <1000 个乒乓球回合），无法充分学习复杂的比赛动态。同时，公开可用的大规模竞技乒乓球 3D 数据集几乎不存在。

3. **核心矛盾**：预判能力需要大量专业比赛数据来学习，但收集专业比赛的 3D 数据传统上需要多目相机或 RGB-D 设备等专用录制设备，成本极高、难以规模化。

4. **本文目标** (1) 如何从廉价的单目视频中大规模重建乒乓球比赛的 3D 数据？(2) 如何利用大规模数据学习一个带不确定性估计的对手意图预判模型？

5. **切入角度**：作者观察到网上存在大量公开的乒乓球比赛录像，如果能自动从这些单目视频中提取 3D 信息，就能绕过专用设备的瓶颈。利用各种预训练模型（YOLO 分割、HMR 人体重建、TrackNetV3 球追踪）的组合，可以构建全自动的 3D 重建流水线。

6. **核心 idea**：从公开单目乒乓球视频大规模重建 3D 比赛数据（73,222 个回合），训练 Transformer 预判对手击球，用共形预测量化预测不确定性并指导机器人预布局。

## 方法详解

### 整体框架
系统分为两大模块：(1) **3D 重建流水线**——从 ~800 小时原始乒乓球录像中筛选出 ~50 小时实际比赛画面，通过实体追踪（球台、球拍、球员、球）和全局定位（相机标定、球员 SMPL 重建、球 3D 轨迹重建）提取 73,222 个回合的 3D 数据；(2) **预判式控制**——用 Transformer 学习比赛动态，通过集成 + 共形预测构建置信区间，指导仿真机器人在对手击球前预先布局。

### 关键设计

1. **单目视频 3D 重建系统**:

    - 功能：从单目乒乓球视频自动提取球员 3D 姿态和球的 3D 轨迹
    - 核心思路：分三步完成——(a) 视频裁剪：训练 CNN 分类器筛选出实际打球画面（从 800h 筛到 50h）；(b) 实体追踪：用 YOLOv8 分割球台/球拍表面，用自定义 UNet 检测球台 6 个关键点（4 角 + 2 网柱交点），用 HMR 2.0 重建球员 SMPL mesh，用 TrackNetV3 追踪球的 2D 位置；(c) 全局定位：利用 ITTF 标准球台尺寸和检测到的 10 个图像校准点估计相机内外参，将球员和球投影到世界坐标系。球的 3D 轨迹通过检测击球点和弹跳点，在两点之间拟合含 Stokes 阻力的抛物线模型 $x_k(t), y_k(t), z_k(t)$，通过最小化重投影误差优化阻力系数 $k$。
    - 设计动机：通过组合多个预训练视觉模型实现全自动化，无需专用硬件，可扩展至互联网上海量公开比赛录像。

2. **Transformer 预判模型**:

    - 功能：基于比赛历史序列预测对手未来的击球轨迹
    - 核心思路：将每帧重建数据（对手 SMPL 关节位置、己方根位置、球位置）tokenize 后输入 decoder-only Transformer（$d=256$, $L=4$ 层, 16 头），自回归建模 $p(t) = \prod p(t_i | t_{i-1}, ..., t_1)$。假设高斯分布，训练目标退化为 MSE 损失 $\mathcal{L} = \sum \|\hat{t} - t\|_2^2$。模型仅 3.2M 参数，推理 <10ms。
    - 设计动机：Transformer 能在大规模数据上有效捕获时序依赖和比赛动态模式，且推理快速满足实时性要求。

3. **共形预测（Conformal Prediction）不确定性量化**:

    - 功能：为预测构建有理论覆盖率保证的置信区间
    - 核心思路：训练 5 个 Transformer 的集成模型（分别在不重叠的数据子集上训练），对每个样本取集成均值 $\hat{f}(X)$ 和标准差 $\hat{\sigma}(X)$。在校准集上计算归一化残差 $R_i = |Y_i - \hat{f}(X_i)| / \hat{\sigma}(X_i)$，取分位数 $\hat{q}_\alpha$。预测时构造置信区间 $\mathcal{C}_\alpha(X) = [\hat{f}(X) \pm \hat{q}_\alpha \hat{\sigma}(X)]$，分别对 x/y/z 三轴构建后取笛卡尔积，理论保证 $\Pr(b_t \in \mathcal{C}_\alpha) \geq 1-3\alpha$。
    - 设计动机：仅靠集成标准差无法保证覆盖率，共形预测提供无分布假设的有限样本覆盖率保证，且能过滤不确定性过高的预测。

### 损失函数 / 训练策略
- Transformer 训练使用 MSE 损失（等价于高斯 NLL 中固定方差的情况），训练 200 epochs
- 数据集划分：5 个训练子集 + 2500 校准集 + 1000 测试集
- 球轨迹重建优化阻力系数 $k$ 时使用重投影误差（Eq. 5）

## 实验关键数据

### 主实验

| 预布局策略 | 回球率 | 返回精度 (m) | 姿态精度 (m / °) |
|-----------|--------|-------------|-----------------|
| Baseline (无预布局) | 49.9% | 0.497 | 0.25 / 13.3° |
| Anticipatory (本文) | **59.0%** | **0.463** | **0.19 / 9.86°** |
| Oracle (真实轨迹) | 64.5% | 0.453 | 0.15 / 6.26° |

在 KUKA 机器人仿真中，预判式控制将回球率提升了 9.1 个百分点（+18.2%），弥合了与 Oracle 之间约 62% 的差距。

### 共形预测覆盖率

| $1-\alpha$ | $\mathcal{C}_{\alpha,x}$ | $\mathcal{C}_{\alpha,y}$ | $\mathcal{C}_{\alpha,z}$ | 理论 $1-3\alpha$ | 实际 $\mathcal{C}_\alpha$ |
|-----------|------|------|------|----------|---------|
| 0.90 | 0.885 | 0.906 | 0.905 | 0.70 | 0.763 |
| 0.85 | 0.830 | 0.858 | 0.862 | 0.55 | 0.652 |
| 0.80 | 0.782 | 0.796 | 0.822 | 0.40 | 0.532 |

### 关键发现
- 置信区间在极端击球（y 值 > 0.75m）时，虽然多数情况过度覆盖，但在有偏向性的子集中，85% 以上偏向正确方向
- 数据集平均球速 11.25 m/s，击球间隔 0.56 秒，数据分布存在偏置：球员倾向于向右侧击球（y 分布双峰）
- 重建精度：球位置平均误差 8.9 cm，人体关节 28 cm（640×360 分辨率下）
- 数据集未能重建得分点的最后一个片段（球未被击中的时刻），引入了显著的分布偏差

## 亮点与洞察
- **用公开视频构建大规模 3D 数据集的思路极具启发性**：组合多个预训练模型（YOLO + HMR + TrackNet + 相机标定）实现全自动 pipeline，思路可推广到其他运动场景（篮球、网球等）
- **共形预测 + 集成模型的组合**提供了有理论保证的不确定性量化，比单纯的集成标准差更可靠，在安全关键的机器人应用中尤为重要
- **预判式控制的通用框架**（预测 → 置信区间 → 可达集检查 → 预布局目标选择）可迁移到其他需要快速反应的人机交互场景

## 局限与展望
- 未重建球拍姿态，无法获取旋转信息，而旋转在专业乒乓球中至关重要
- 重建精度有限（球 8.9cm、人体 28cm），低分辨率视频是主要瓶颈
- 数据集偏差明显：无法捕获得分球的最后一击（最关键的策略信息），且存在群体级击球方向偏好
- 仅在仿真中验证，未部署到真实硬件
- 控制器使用简单的阻挡策略（无挥拍），未优化回球策略本身

## 相关工作与启发
- **vs i-sim2real (Abeyruwan et al.)**: 实现了人机协作打乒乓球，但不涉及预判。本文补充了预判能力的维度
- **vs IDDM (意图驱动动力学模型)**: 使用潜变量模型建模人类意图，但数据集 <1000 个回合。本文用大规模数据 + Transformer 替代，数据量提升 73 倍
- 该工作的重建 pipeline 思路可以作为其他运动分析研究的参考模板

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次从大规模单目视频重建乒乓球 3D 数据集并用于预判控制
- 实验充分度: ⭐⭐⭐ 仿真验证充分但缺少真实硬件实验，重建精度评估有限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，但部分细节（如控制器设计）放在附录
- 价值: ⭐⭐⭐⭐ 数据集和 pipeline 对社区有较高价值，预判框架可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](../../ACL2025/others/tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)
- [\[CVPR 2026\] VideoWorld 2: Learning Transferable Knowledge from Real-world Videos](../../CVPR2026/others/videoworld_2_learning_transferable_knowledge_from_real-world_videos.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](../../ACL2025/others/mapmake_schema_guided_text_to_table_generation.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)

</div>

<!-- RELATED:END -->
