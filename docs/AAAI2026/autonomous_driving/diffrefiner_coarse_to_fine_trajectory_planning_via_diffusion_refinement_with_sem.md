---
title: >-
  [论文解读] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving
description: >-
  [AAAI2026][自动驾驶][扩散模型] 提出 DiffRefiner，通过"粗到精"两阶段框架——先用判别式 Proposal Decoder 生成粗轨迹，再用扩散模型迭代精炼——结合细粒度语义交互模块，在 NAVSIM v2 和 Bench2Drive 两个基准上均达到 SOTA。 端到端自动驾驶（E2E-AD）将原…
tags:
  - "AAAI2026"
  - "自动驾驶"
  - "扩散模型"
  - "trajectory planning"
  - "coarse-to-fine"
  - "semantic interaction"
---

# DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving

**会议**: AAAI2026  
**arXiv**: [2511.17150](https://arxiv.org/abs/2511.17150)  
**代码**: [nullmax-vision/DiffRefiner](https://github.com/nullmax-vision/DiffRefiner)  
**领域**: 自动驾驶  
**关键词**: end-to-end autonomous driving, diffusion model, trajectory planning, coarse-to-fine, semantic interaction

## 一句话总结

提出 DiffRefiner，通过"粗到精"两阶段框架——先用判别式 Proposal Decoder 生成粗轨迹，再用扩散模型迭代精炼——结合细粒度语义交互模块，在 NAVSIM v2 和 Bench2Drive 两个基准上均达到 SOTA。

## 背景与动机

端到端自动驾驶（E2E-AD）将原始传感器输入直接映射为轨迹规划，相比传统模块化流水线更高效。当前轨迹预测方法主要分三类：

1. **单次回归方法**：计算高效但无法处理驾驶行为的多模态性，容易在复杂路口产生均值化的次优预测
2. **基于锚点的打分方法**（如 VADv2、HydraMDP++）：将预测转为分类问题，但锚点集增大导致计算复杂度大幅上升，难以满足实时性要求
3. **扩散模型方法**（如 DiffusionDrive）：通过去噪迭代生成多样轨迹，天然捕捉多模态性，但初始化依赖无结构的高斯噪声或固定锚点，缺乏场景适应性，需要大量去噪步数

核心动机：**能否将判别式模型的快速、有结构化先验的优势与扩散模型的灵活生成能力结合？** 即先用判别式方法给出"粗"轨迹提案，再用扩散模型在此基础上做"精"细化。

## 核心问题

- 现有扩散方法从随机噪声或固定锚点出发去噪，初始分布偏离可行运动空间，导致需要多步去噪，增加延迟
- 轨迹预测与环境语义（可行驶区域、障碍物）之间缺乏细粒度交互，容易产生碰撞或偏离车道

## 方法详解

### 整体架构

DiffRefiner 由三个核心模块组成：

1. **BEV 感知模块**：将多视角相机输入编码为 BEV 特征，通过稀疏检测头和密集分割头分别完成目标检测与语义分割
2. **Proposal Decoder（粗阶段）**：基于轻量 Transformer，从聚类轨迹锚点出发预测偏移，生成粗轨迹提案
3. **Diffusion Refiner（精阶段）**：条件扩散模型，以粗轨迹作为初始化，通过迭代去噪生成高质量最终轨迹

### Proposal Decoder

- 使用离线聚类得到的 20 条轨迹锚点作为离散运动候选
- 每条锚点经过位置编码和 MLP 投射为查询向量
- 通过 cross-attention 与 Planning Token 交互，输出上下文增强的轨迹提案
- 这一阶段本质是判别式方法，为后续扩散精炼提供强初始化

### Diffusion Refiner

**训练阶段**：对粗轨迹执行前向扩散过程（加高斯噪声），在随机步 $t$ 得到噪声轨迹 $\tilde{Y}$，模型学习逆向去噪

**推理阶段**：以粗轨迹为起点（而非纯噪声），只需极少去噪步数即可获得高质量结果（实验表明 1 步即接近最优）

### Fine-Grained Semantic Interaction Module (FGSIM)

这是方法的核心创新，嵌入去噪 decoder 中分两阶段运作：

1. **道路感知精炼**：轨迹特征与 BEV 特征及可行驶区域分割交互，约束预测在物理可通行区域内
2. **交互感知精炼**：引入动态智能体特征，建模交通参与者的交互关系和碰撞规避

每个语义类别的交互通过三级机制实现：

- **全局 cross-attention**：建立轨迹特征与 BEV 语义区域的密集对应，编码全局场景上下文
- **局部 deformable attention**：在轨迹端点附近自适应关注关键区域语义，提取局部几何结构
- **门控融合**：通过可学习的门控网络动态平衡全局与局部表示，公式为 $Q_r = Q_r^{(c)} \cdot \text{Gate} + Q_r^{(d)} \cdot (1 - \text{Gate})$

### 训练损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proposal}} + \mathcal{L}_{\text{refinement}} + \mathcal{L}_{\text{perception}}$$

采用两阶段训练：第一阶段单独训练感知网络，第二阶段端到端联合优化感知与规划。使用 winner-takes-all 策略选择最接近真值的轨迹计算损失。

## 实验关键数据

### NAVSIM v2（开环评测）

| 方法 | Backbone | EPDMS |
|------|----------|-------|
| HydraMDP++ | V2-99 | 85.1 |
| DriveSuprim | V2-99 | 86.0 |
| **DiffRefiner** | **V2-99** | **87.4** |
| DiffusionDrive | ResNet34 | 84.0 |
| **DiffRefiner** | **ResNet34** | **86.2** |

在 ResNet34 backbone 下超越前作 DriveSuprim 3.7%，在 V2-99 下超越 1.6%。

### Bench2Drive（闭环评测）

| 方法 | DS | SR(%) |
|------|-----|-------|
| TF++ | 84.2 | 67.3 |
| HiPAD | 86.8 | 69.1 |
| **DiffRefiner** | **87.1** | **71.4** |

在无模型集成条件下，DS 提升 0.3，SR 提升 2.3，在多能力指标的大部分子项上均领先。

### 关键消融实验

- **两阶段 vs 单阶段**：加入 Refiner 后 EPDMS 从 85.0 提升到 86.2（+1.2）
- **生成式 vs 判别式 Refiner**：生成式 Refiner（86.2）优于判别式（78.3），验证扩散精炼更适合细粒度调整
- **FGSIM 各组件**：Planning Token → +Agent Token → +BEV Modulation → +可行驶区域 → +交通参与者，EPDMS 从 82.4 逐步提升至 85.0
- **门控融合 vs 加法融合**：门控融合（86.2）优于加法融合（85.9），自适应平衡避免了信息冲突
- **去噪步数**：仅需 1 步即可达到接近最优性能，端到端延迟仅 27ms

## 亮点

- **粗到精的混合范式**：判别式提供强先验 + 生成式做细粒度优化，两者优势互补，是一种优雅的设计
- **FGSIM 模块设计精巧**：全局-局部-门控三级语义交互，显式建模轨迹与环境的对齐关系
- **极少去噪步数**：高质量粗提案使得扩散仅需 1 步精炼，满足实时性要求（27ms）
- **双基准 SOTA**：同时在开环（NAVSIM v2）和闭环（Bench2Drive）基准上刷新纪录，验证了方法的鲁棒性

## 局限与展望

- 仅使用相机输入，未融合 LiDAR；在 NAVSIM 上 Camera+LiDAR 的 GaussianFusion 在部分子指标上仍有优势
- Bench2Drive 上与基于规则的 PDM-Lite（DS 97.0）差距仍然明显，说明学习方法在稳定性上还有提升空间
- 20 条聚类锚点的选择和数量对性能的影响未深入探讨
- 闭环评测中 Overtake（60.0）和 GiveWay（50.0）能力仍然较弱，复杂交互场景有待提升
- 未探索将该框架扩展到多智能体联合预测与规划

## 与相关工作的对比

| 维度 | DiffusionDrive | DriveSuprim | DiffRefiner |
|------|---------------|-------------|-------------|
| 范式 | 纯生成式 | 纯判别式 | 混合（判别式+生成式） |
| 初始化 | 锚点高斯混合 | 锚点分类+偏移 | 判别式粗提案 |
| 语义交互 | BEV 空间调制 | 隐式特征交互 | 显式细粒度 FGSIM |
| NAVSIM v2 (V2-99) | - | 86.0 | **87.4** |
| Bench2Drive DS | - | - | **87.1** |

核心差异在于 DiffRefiner 用判别式模块替代了无结构的初始化，并通过 FGSIM 显式引入语义约束。

## 启发与关联

- "粗到精"的两阶段思路具有通用性，可推广到其他生成式规划任务（如机器人运动规划、无人机路径规划）
- FGSIM 中全局-局部门控融合的设计可借鉴到其他需要多尺度语义对齐的任务中
- 高质量先验大幅减少扩散步数的发现，对加速扩散模型推理有启示意义
- 与 GoalFlow、DriveTransformer 等"粗到精"工作形成对比，本文特色在于精细化阶段用扩散而非确定性 Transformer

## 评分

- 新颖性: 7/10 — 粗到精 + 扩散的组合并非全新，但 FGSIM 的设计有独到之处
- 实验充分度: 9/10 — 双基准 SOTA + 详尽消融，各组件贡献清晰
- 写作质量: 8/10 — 结构清晰，图示规范，公式表述完整
- 价值: 8/10 — 方法实用、性能突出、延迟可控，工程与学术价值兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](../../CVPR2025/autonomous_driving/diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] ActiveAD: Planning-Oriented Active Learning for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/activead_planning-oriented_active_learning_for_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[CVPR 2026\] ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/resad_normalized_residual_trajectory_modeling_for_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
