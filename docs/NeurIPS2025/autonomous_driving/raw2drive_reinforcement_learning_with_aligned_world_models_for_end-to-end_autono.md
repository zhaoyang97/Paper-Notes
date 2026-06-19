---
title: >-
  [论文解读] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving
description: >-
  [NeurIPS 2025][自动驾驶][强化学习] 提出 RAW2Drive，首个从原始传感器输入到规划的基于模型的强化学习 (MBRL) 端到端自动驾驶框架。通过双流世界模型设计——先训练特权世界模型，再通过引导机制指导原始传感器世界模型学习——在 CARLA v2 和 Bench2Drive 上取得 SOTA，大幅超越 IL 方法。
tags:
  - "NeurIPS 2025"
  - "自动驾驶"
  - "强化学习"
  - "world model"
  - "end-to-end driving"
  - "CARLA"
  - "dual-stream"
---

# RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving

**会议**: NeurIPS 2025  
**arXiv**: [2505.16394](https://arxiv.org/abs/2505.16394)  
**代码**: 无  
**领域**: Autonomous Driving  
**关键词**: model-based reinforcement learning, world model, end-to-end driving, CARLA, dual-stream

## 一句话总结

提出 RAW2Drive，首个从原始传感器输入到规划的基于模型的强化学习 (MBRL) 端到端自动驾驶框架。通过双流世界模型设计——先训练特权世界模型，再通过引导机制指导原始传感器世界模型学习——在 CARLA v2 和 Bench2Drive 上取得 SOTA，大幅超越 IL 方法。

## 研究背景与动机

端到端自动驾驶 (E2E-AD) 的主流范式是模仿学习 (IL)，但 IL 面临**因果混淆**（模型将动作与错误原因关联）和**分布偏移**（遇到未见场景无法泛化）两大根本限制。

RL 是有前景的替代方案，但应用于 E2E-AD 仍有巨大困难：
- **无模型 RL 效率低**：MaRLn 需要约 5000 万步（57 天训练），性能仍远低于 IL
- **MBRL 的输入差距**：Think2Drive 证明 MBRL 可解决 CARLA v2，但依赖**特权信息**（GT bbox、HD Map）。原始传感器数据高维、冗余、有噪声，直接训练世界模型极其困难

截至本文，CARLA v2 上**没有任何基于 RL 的端到端方法**——这正是本文要攻克的方向。

## 方法详解

### 整体框架

Raw2Drive 是双流 MBRL 框架：
- **流 I (特权流)**：使用 BEV 语义掩码等特权信息训练世界模型和配套策略（与 DreamerV3 类似）
- **流 II (原始传感器流)**：使用多视角图像 + IMU 作为输入，在特权流的引导下训练世界模型和端到端策略
- 推理时仅使用原始传感器输入

### 关键设计

1. **特权流 (Privileged Stream)**：

    - 输入：时序 BEV 语义掩码（标准 MBRL 输入）
    - 世界模型：Encoder + RSSM + 三个 Head（Reward/Decoder/Continue），与 DreamerV3 架构相同
    - 策略：Actor-Critic 网络通过世界模型 rollout 训练
    - 目的：(I) 训练特权策略；(II) 为原始传感器流提供指导

2. **原始传感器流 (Raw Sensor Stream)**：

    - 编码器：使用 BEVFormer 将多视角图像编码为 BEV 特征
    - 世界模型：架构与特权流类似，但**仅使用 Decoder Head**——奖励和继续标志从特权流获取
    - 关键发现：直接训练 reward/continue head 会导致收敛失败，因为相邻帧极相似但奖励值可能剧烈波动，网络无法学到稳定模式
    - RSSM 参数从特权世界模型初始化

3. **引导机制 (Guidance Mechanism)**——双流框架的核心：
   
   **Rollout Guidance**：确保两个世界模型在 rollout 过程中预测一致
    - **时空对齐 (Spatial-Temporal Alignment)**：对 Encoder State 施加 MSE loss，确保每个时间步的空间一致性
    - **抽象状态对齐 (Abstract-State Alignment)**：
      - 确定性状态 $h$：L2 loss 保持预测一致（建模自车状态）
      - 随机状态 $s$：KL 散度约束分布一致（建模其他交通参与者行为）
    - **消除累积误差**：只从原始传感器流的分布采样一次随机状态，同时输入两个流，避免采样随机性导致的累积偏差
   
   **Head Guidance**：原始传感器策略训练时，reward 和 continue flag 直接从特权世界模型获取（而非自己的 head），确保监督信号准确稳定。

   总 Rollout Loss：
    $\mathcal{L}_{Rollout} = \beta_e \sum_t \sum_{i} \text{MSE}(e_t^i, \hat{e}_t^i) + \sum_t (\beta_s \text{KL}(s_t, \hat{s}_t) + \beta_h \text{MSE}(h_t, \hat{h}_t))$

### 损失函数 / 训练策略

两阶段训练：
- **Phase I**：训练特权世界模型和策略（可复用 Think2Drive 的结果，节省 ~24 GPU days）
- **Phase II**：在特权流引导下训练原始传感器世界模型和策略

总训练成本：64 H800 GPU days（复用 Think2Drive 时仅 40 GPU days），相比之下 UniAD 约 30 GPU days 但只能处理 3~4 种 corner case。

## 实验关键数据

### 主实验

Bench2Drive 闭环评测：

| 方法 | 范式 | DS ↑ | SR (%) ↑ | Efficiency ↑ | Comfort ↑ |
|------|-----|------|---------|-------------|-----------|
| UniAD-Base | IL | 45.81 | 16.36 | 129.21 | 43.58 |
| DriveAdapter* | IL | 64.22 | 33.08 | 70.22 | 16.01 |
| DriveTrans | IL | 63.46 | 35.01 | 100.64 | 20.78 |
| **Raw2Drive** | **RL** | **71.36** | **50.24** | 214.17 | 22.42 |
| Think2Drive | RL (特权) | 91.85 | 85.41 | 269.14 | 25.97 |

多能力评测：

| 方法 | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|------|---------|-----------|----------------|----------|-------------|------|
| UniAD | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| DriveTrans | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| **Raw2Drive** | **43.35** | **51.11** | **60.00** | **50.00** | **62.26** | **53.34** |

### 消融实验

| 配置 | DS ↑ | SR |说明 |
|------|------|-----|-----|
| 无 Decoder Head | 17.4 | 1.2/10 | 无有效监督 |
| 仅 Decoder Head | **83.5** | **7.5/10** | 最佳配置 |
| Decoder + Reward | 46.6 | 3.4/10 | Reward head 引入噪声 |
| Decoder + Reward + Continue | 34.5 | 2.2/10 | 更差 |
| 无空间对齐 | 9.24 | 0.8/10 | 如同"盲驾" |
| 无时间对齐 | 13.6 | 1.2/10 | 未来预测不一致 |
| 完整对齐 | **83.5** | **7.5/10** | 三组件缺一不可 |

### 关键发现

- **RL 大幅超越 IL**：在所有端到端方法中，Raw2Drive 的 DS 和 SR 均远超 IL 方法（DS: 71.36 vs DriveTrans 63.46）
- **Reward/Continue Head 有害**：直接在原始传感器流训练这些 head 反而损害性能（DS 从 83.5 降至 34.5）
- **三组件对齐缺一不可**：编码器/确定性/随机状态对齐任缺其一，模型只能处理简单直行
- **RSSM 和 Decoder 参数共享有益**：促进更高效的表示学习

## 亮点与洞察

- 核心贡献：证明了 MBRL 可以在仅使用原始传感器输入的条件下解决 CARLA v2 难题，填补了该领域空白
- 双流设计的哲学：特权信息是低维、结构化的"捷径"，用它先建立好的世界模型，然后指导高维传感器流学习决策相关的信息
- 消除采样随机性累积误差的技巧简单但关键——仅从原始传感器流采样一次，避免两流分叉
- 训练成本仅 64 H800 GPU days，在工程化方面具有实用性

## 局限与展望

- 与特权信息方法 Think2Drive (DS 91.85) 仍有较大差距，说明原始传感器世界模型的表达能力仍受限
- 仅使用 CARLA 仿真器评估，未在真实世界数据上验证
- BEVFormer 编码器较重，推理延迟可能不满足实时要求
- 特权流在推理时不使用，但训练时需要特权信息，这限制了真实数据的直接训练

## 相关工作与启发

- 与 Think2Drive 的关系：Think2Drive 是 MBRL + 特权输入；Raw2Drive 将其扩展到原始传感器输入，是对 Think2Drive 的自然延伸
- 与 DreamerV3 的关系：世界模型架构基于 DreamerV3，但引入双流设计和引导机制
- 启发：特权到端到端的迁移范式（先用简单输入学好，再迁移到复杂输入）或许可推广到机器人操作等领域

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个原始传感器 MBRL E2E-AD，双流引导机制设计精巧
- 实验充分度: ⭐⭐⭐⭐ CARLA v2 + Bench2Drive 两个基准，充分消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机解释充分
- 价值: ⭐⭐⭐⭐⭐ 证明 RL 在 E2E-AD 的可行性和优越性，对学术界和工业界都有重要参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[CVPR 2026\] EE-RL: Vision Language Guided Reinforcement Learning with Explorer and Expert model for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/ee-rl_vision_language_guided_reinforcement_learning_with_explorer_and_expert_mod.md)

</div>

<!-- RELATED:END -->
