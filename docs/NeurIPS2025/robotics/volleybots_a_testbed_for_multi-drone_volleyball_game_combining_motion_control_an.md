---
title: >-
  [论文解读] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play
description: >-
  [NeurIPS 2025][机器人][多无人机系统] 本文提出 VolleyBots，一个多无人机排球竞技测试平台，融合了合作-对抗博弈、回合制交互与敏捷 3D 机动控制，基于 Isaac Sim 构建了从单体训练到多体竞技的任务课程体系，并通过分层策略在 3v3 任务中取得 69.5% 胜率，同时展示了零样本 sim-to-real 部署能力。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "多无人机系统"
  - "机器人运动"
  - "多智能体强化学习"
  - "博弈论"
  - "仿真到现实"
---

# VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play

**会议**: NeurIPS 2025  
**arXiv**: [2502.01932](https://arxiv.org/abs/2502.01932)  
**代码**: [https://github.com/thu-uav/VolleyBots](https://github.com/thu-uav/VolleyBots)  
**领域**: 强化学习  
**关键词**: 多无人机系统, 机器人运动, 多智能体强化学习, 博弈论, 仿真到现实

## 一句话总结
本文提出 VolleyBots，一个多无人机排球竞技测试平台，融合了合作-对抗博弈、回合制交互与敏捷 3D 机动控制，基于 Isaac Sim 构建了从单体训练到多体竞技的任务课程体系，并通过分层策略在 3v3 任务中取得 69.5% 胜率，同时展示了零样本 sim-to-real 部署能力。

## 研究背景与动机
机器人运动竞技（robot sports）因其目标明确、规则显式、交互动态等特性，被视为展示具身智能的理想场景。现有工作分别在各自子领域取得进展：足球机器人（RoboCup）侧重团队合作与对抗；乒乓球机器人强调回合制交互；无人机追逃任务要求 3D 空间敏捷机动。然而，没有一个平台能同时涵盖**混合合作-对抗博弈**、**回合制交互**和**敏捷 3D 机动**三大特征。

排球运动天然具备这三大特征：队内需要紧密配合完成传球-扣球序列（合作），队间需要预判和利用对手策略（对抗），球的运动轨迹要求无人机在欠驱动四旋翼动力学下做出快速加速、急转弯和精细定位（3D 机动）。这些特征相互交织，形成了一个同时需要**低级运动控制**和**高级战略博弈**的复杂问题，且不存在可用的专家演示数据。

核心切入角度是：以排球运动为载体，借鉴人类学习排球的渐进过程，设计从简单到困难的任务课程体系，系统性地评测和推进多智能体 RL、博弈论算法在复杂具身任务上的能力。

## 方法详解

### 整体框架
VolleyBots 构建于 NVIDIA Isaac Sim 之上，利用 OmniDrones 仿真器实现 GPU 并行数据采集。平台由三部分组成：（1）环境——定义实体、观测、动作和奖励；（2）任务——包含 3 个单体任务、3 个多体合作任务和 3 个多体竞技任务；（3）算法——涵盖 RL、MARL 和博弈论基线。

### 关键设计
1. **仿真实体与物理交互**:

    - 采用 Iris 四旋翼模型，附加半径 0.2m 的虚拟球拍（恢复系数 0.8）
    - 球体半径 0.1m、质量 5g、恢复系数 0.8，支持真实的弹跳与碰撞
    - 场地遵循标准排球尺寸 $9\text{m} \times 18\text{m}$，网高 2.43m
    - 设计动机：高保真物理交互是 sim-to-real 迁移的基础

2. **渐进式任务课程设计**:

    - **单体任务**：Back and Forth（往返冲刺）、Hit the Ball（击球距离）、Solo Bump（连续颠球），评估基础 3D 机动能力
    - **合作任务**：Bump and Pass（双人对颠）、Set and Spike Easy/Hard（传球-扣球配合），引入队内回合制协调
    - **竞技任务**：1v1、3v3、6v6，融合合作、对抗、回合制和 3D 机动全部挑战
    - 设计动机：模仿人类从基本功到团队配合再到正式比赛的学习路径

3. **双层动作空间与分层奖励**:

    - 动作空间提供两种选择：CTBR（集体推力+角速率，高层抽象）和 PRT（单旋翼推力，细粒度控制）
    - 奖励函数由三部分组成：misbehave penalty（通用违规惩罚）、task reward（稀疏任务奖励）、shaping reward（辅助引导奖励）
    - 设计动机：PRT 提供更强操控能力但训练较难，CTBR 便于 sim-to-real；分层奖励兼顾探索和学习效率

4. **分层策略（Hierarchical Policy）**:

    - 低级层：用 PPO 训练一组基本技能（Hover, Serve, Pass, Set, Attack）
    - 高级层：基于规则的事件驱动策略，在每次击球时决定为每架无人机分配哪个低级技能
    - 攻击技能中高级策略以等概率选择左右方向
    - 设计动机：直接端到端方法在兼顾运动控制和战略博弈时表现不佳，分层解耦能有效缓解这一问题

### 损失函数 / 训练策略
- 单体和合作任务使用标准 PPO/MAPPO 的 clip surrogate objective
- 竞技任务结合 Self-Play、Fictitious Self-Play (FSP)、PSRO（Uniform/Nash meta-solver）等博弈论训练范式
- 所有算法在同一任务内使用统一超参数以评估跨任务鲁棒性

## 实验关键数据

### 主实验

| 任务 | 指标 | PPO (PRT) | TD3 (PRT) | SAC (PRT) | 说明 |
|------|------|-----------|-----------|-----------|------|
| Back and Forth | 到达点数 | **10.04±0.20** | 0.99±0.01 | 0.83±0.25 | PPO 远超 off-policy |
| Hit the Ball | 击球距离(m) | **11.40±0.06** | 5.29±1.28 | 3.87±2.34 | PPO 2x SOTA |
| Solo Bump | 颠球次数 | **10.83±1.24** | 3.68±1.43 | 1.36±0.60 | PPO 优势显著 |

| 竞技任务 | 指标 | SP | FSP | PSRO_Nash | 分层策略 |
|----------|------|------|------|-----------|----------|
| 3v3 | Exploitability↓ | **25.76** | 38.86 | 35.83 | — |
| 3v3 | Win Rate↑ | 0.59 | 0.52 | **0.61** | **0.695** |
| 3v3 | Elo↑ | 1077 | 906 | **1268** | — |

### 消融实验

| 配置 | Bump and Pass | Set&Spike (Easy) | Set&Spike (Hard) |
|------|---------------|-----------------|-----------------|
| MAPPO w/o shaping | 11.32±0.91 | 0.25 | 0.25 |
| MAPPO w. shaping | **13.71±0.58** | **0.99** | 0.75 |
| MADDPG w. shaping | 0.84±0.09 | 0.23 | 0.22 |
| QMIX w. shaping | 0.09±0.00 | 0.02 | 0.02 |

### 关键发现
- On-policy 方法（PPO/MAPPO）在所有任务上一致优于 off-policy 方法，且跨任务超参数鲁棒性更强
- PRT 动作空间最终性能略优于 CTBR，但 CTBR 学习速度更快
- 6v6 任务所有算法均未能收敛到有效策略，暴露当前方法的可扩展性瓶颈
- 分层策略在 3v3 任务中以 69.5% 胜率击败最强基线（SP 的纳什均衡策略）

## 亮点与洞察
- **首个将合作-对抗-回合制-3D 机动统一到一个平台的工作**，填补了机器人运动竞技领域的空白，为具身智能研究提供了富有挑战性的标准测试环境。
- **从 sim-to-real 的零样本部署验证了平台的实际意义**，训练的 Solo Bump 策略直接在真实四旋翼上成功执行颠球，说明仿真器能够较好地逼近真实物理。

## 局限与展望
- 6v6 赛制目前无算法能收敛，需要更先进的分层或通信机制来处理大规模协调
- 分层策略的高级层仍是手工规则，未来可用 RL/LLM 来自动学习战略决策
- 当前观测空间为 state-based，未来可引入视觉观测探索更真实的感知挑战

## 相关工作与启发
- **vs MQE (四足足球)**: MQE 同样支持合作-对抗，但 VolleyBots 独有回合制交互和 3D 飞行机动，对低级控制要求更高
- **vs Robot Table Tennis**: 乒乓球机器人有回合制但缺乏多智能体合作，VolleyBots 在多体协调维度更丰富
- **vs SMPLOlympics**: 人形奥运场景覆盖面广但缺乏 3D 飞行和团队配合的深度

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将排球运动迁移到多无人机平台，任务设计系统但核心贡献偏benchmark
- 实验充分度: ⭐⭐⭐⭐⭐ 9 个任务、多种 RL/MARL/博弈论算法、消融、sim-to-real 全面覆盖
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，任务描述详尽
- 价值: ⭐⭐⭐⭐ 作为标准化benchmark对多智能体具身智能研究有长期推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](../../AAAI2026/robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[CVPR 2026\] Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control](../../CVPR2026/robotics/iterative_closed-loop_motion_synthesis_for_scaling_the_capabilities_of_humanoid_.md)
- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)

</div>

<!-- RELATED:END -->
