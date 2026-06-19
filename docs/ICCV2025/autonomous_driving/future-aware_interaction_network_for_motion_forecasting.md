---
title: >-
  [论文解读] Future-Aware Interaction Network For Motion Forecasting
description: >-
  [ICCV 2025][自动驾驶][运动预测] 提出 FINet，将潜在未来轨迹提前建模并融入场景编码阶段进行联合优化，同时引入 Mamba 架构替代 Transformer 进行时空建模，实现了高效且准确的运动预测。 运动预测是自动驾驶的关键组件，需要根据历史轨迹和地图信息预测多条未来可能轨迹。现有方法主要有两类： MLP…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "运动预测"
  - "Mamba"
  - "状态空间模型"
  - "轨迹预测"
---

# Future-Aware Interaction Network For Motion Forecasting

**会议**: ICCV 2025  
**arXiv**: [2503.06565](https://arxiv.org/abs/2503.06565)  
**代码**: 无（论文提及将根据接收情况公开）  
**领域**: 自动驾驶  
**关键词**: 运动预测, Mamba, 状态空间模型, 自动驾驶, 轨迹预测

## 一句话总结

提出 FINet，将潜在未来轨迹提前建模并融入场景编码阶段进行联合优化，同时引入 Mamba 架构替代 Transformer 进行时空建模，实现了高效且准确的运动预测。

## 研究背景与动机

运动预测是自动驾驶的关键组件，需要根据历史轨迹和地图信息预测多条未来可能轨迹。现有方法主要有两类：

**MLP-based**: 直接从 agent 当前状态通过 MLP 生成未来轨迹

**Query-based**: 用可学习 query 从编码表示中聚合信息再解码轨迹

这两类方法的共同问题是：**未来轨迹在场景编码阶段是缺失的**，导致历史状态和未来状态的优化是分离的，可能产生不合理的预测（如错误预测左转）。此外，Transformer 的二次复杂度在多 agent 场景下效率低下。

本文的动机是：(1) 将未来轨迹引入场景编码，通过联合优化获得更全面的交通表示；(2) 用 Mamba（线性复杂度）替代 Transformer，提升效率。

## 方法详解

### 整体框架

FINet 包含三个主要组件：
- **Lightweight Scene Encoder (LSEnc)**: 将场景转换为 token 表示
- **Future-Aware Interaction Mamba (FIM)**: 建模未来轨迹并与场景元素联合编码
- **Temporal Enhanced Decoder (TEDec)**: 解码未来轨迹

### 关键设计

1. **Lightweight Scene Encoder (LSEnc)**:

    - 用 Mamba blocks 编码 agent 历史轨迹（线性复杂度），取最后时刻 token 代表整条轨迹
    - 用 mini-PointNet 编码车道地图（处理更多点效率更高）
    - 每个轨迹/车道段编码为一个 token，加上语义类别嵌入（车辆/行人/车道类型）
    - 公式：$\mathcal{ST}_i^A = \text{MambaBlocks}(\mathcal{T}_i^{hist})[0] + Cls_i^A$

2. **Future-Aware Interaction Mamba (FIM)**:

    - **未来轨迹建模**: 将未来轨迹表示为当前运动状态 + 驾驶意图 + 归纳偏置的组合：$\mathcal{T}^{fut} = \mathcal{T}_0^{hist} + \mathcal{T}^{bias} + \mathcal{T}^{DI}$
    - 驾驶意图用 K 个可学习 token 建模，归纳偏置仅加到第一条轨迹上并通过 Mamba 传播
    - **Adaptive Reorder Strategy (ARS)**: 解决 Mamba 无法直接处理无序空间数据的问题。通过预测参考点，按场景元素到参考点的距离排序，将无序数据转为有序序列
    - focal agent token 放在排序末尾确保对未来轨迹影响最大
    - 使用双向 Mamba blocks 进行空间交互建模
    - 第二阶段参考点由第一条未来轨迹 token 预测，并用辅助监督对齐到 GT 终点

3. **Temporal Enhanced Decoder (TEDec)**:

    - 将未来轨迹 token 通过插值扩展为时序格式：$\mathcal{IDT}^{fut} = \frac{t}{T^{fut}} \cdot \mathcal{ST}^{fut}$
    - 通过 Cross-Attention + Mamba (CAMBlock) 聚合场景信息并时序精炼
    - Cross-attention 聚合场景信息，Mamba 按时间顺序处理确保时序一致性
    - 最终用 MLP 输出轨迹和置信度分数

### 损失函数 / 训练策略

总损失包含五项：
$$\mathcal{L} = \mathcal{L}_{traj} + \mathcal{L}_{score} + \mathcal{L}_{traj}^{int} + \mathcal{L}_{score}^{int} + L_{align}$$

- $\mathcal{L}_{traj}$: Smooth L1 轨迹回归损失
- $\mathcal{L}_{score}$: 交叉熵分类损失
- 中间输出同样施加轨迹和分数损失
- $L_{align}$: 参考点对齐损失（Smooth L1）
- 采用 Winner-Take-All 策略，仅优化最佳预测

## 实验关键数据

### 主实验 (表格)

**Argoverse 2 测试集**：

| 方法 | b-minFDE6↓ | minADE6↓ | minFDE6↓ | MR6↓ | minADE1↓ | minFDE1↓ | MR1↓ |
|------|-----------|----------|----------|------|----------|----------|------|
| QCNet | 1.91 | 0.65 | 1.29 | 0.16 | 1.69 | 4.30 | 0.59 |
| ProphNet | 1.88 | 0.66 | 1.32 | 0.18 | 1.76 | 4.77 | 0.61 |
| **FINet** | **1.93** | **0.66** | **1.27** | **0.15** | **1.60** | **4.02** | **0.57** |

**Argoverse 1 验证集**（minADE6 从 0.66 降至 **0.59**，提升约 10%）

### 消融实验 (表格)

**解码器类型与归纳偏置的影响**：

| 方法 | b-minFDE6↓ | minADE6↓ | minFDE6↓ | minADE1↓ | minFDE1↓ |
|------|-----------|----------|----------|----------|----------|
| MLP-based | 2.09 | 0.74 | 1.45 | 1.74 | 4.34 |
| Query-based | 2.08 | 0.73 | 1.43 | 1.73 | 4.28 |
| Interaction (w/o bias) | 1.99 | 0.66 | 1.32 | 1.60 | 4.03 |
| Interaction (all bias) | 1.98 | 0.66 | 1.35 | 1.60 | 4.02 |
| Interaction (t=0 bias) | **1.93** | **0.65** | **1.27** | **1.57** | **3.94** |

**效率对比**（vs QCNet）：

| 指标 | QCNet | FINet | 提升 |
|------|-------|-------|------|
| FLOPs (G) | 28.0 | 1.47 | **95%↓** |
| 延迟 (ms) | 54.55 | 17.72 | **68%↓** |
| 模型大小 (M) | 7.7 | 3.7 | **52%↓** |
| GPU 内存 (G) | 2.92 | 0.55 | **81%↓** |

### 关键发现

- Interaction-based 方法显著优于 MLP-based 和 Query-based，证明未来轨迹参与场景编码的有效性
- 归纳偏置仅加在第一条轨迹上效果最好（Mamba 扫描机制可传播该信息）
- K=1 时性能提升大于 K=6，说明联合优化有助于生成更准确的分数和更多样化的轨迹
- FINet 在几乎所有效率指标上大幅领先纯 Transformer 方法

## 亮点与洞察

- 首次提出 Interaction-based 范式，将未来轨迹提前融入场景编码进行联合优化，从概率角度将 $P(\hat{\mathcal{T}}^{fut}|\mathcal{ST})$ 变为 $P(\hat{\mathcal{T}}^{fut}, \mathcal{ST})$
- ARS 策略巧妙解决了 Mamba 无法处理无序空间数据的问题
- 用 Mamba 进行时序精炼确保轨迹时间一致性是很自然的设计
- 效率指标令人印象深刻：FLOPs 降低 95%，实际延迟降低 68%

## 局限与展望

- Mamba 虽然理论 FLOPs 低，但因依赖顺序计算，GPU 加速不如 Transformer 充分
- ARS 的参考点预测依赖启发式设计，可能存在场景泛化问题
- 仅预测 focal agent 的轨迹，未涉及多 agent 联合预测
- 未来可考虑将场景流或占据栅格作为额外输入

## 相关工作与启发

- 与 QCNet 的对比说明联合优化历史/未来状态确实优于分离优化
- Mamba 在自动驾驶时空建模中首次应用，为该领域引入了新的高效backbone选择
- ARS 的思想可推广到其他需要处理无序集合的Mamba应用场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将未来轨迹纳入场景编码的 Interaction-based 范式新颖
- **实验充分度**: ⭐⭐⭐⭐ 在两个标准数据集上验证，消融实验充分，效率对比详细
- **写作质量**: ⭐⭐⭐⭐ 论文结构清晰，动机和方法阐述明确
- **价值**: ⭐⭐⭐⭐ 高效+高精度的运动预测方法具有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](../../CVPR2025/autonomous_driving/trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[ICCV 2025\] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[AAAI 2026\] Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Forecasting in Autonomous Driving](../../AAAI2026/autonomous_driving/differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)

</div>

<!-- RELATED:END -->
