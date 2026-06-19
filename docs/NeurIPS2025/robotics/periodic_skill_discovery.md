---
title: >-
  [论文解读] Periodic Skill Discovery
description: >-
  [NeurIPS 2025][机器人][无监督技能发现] 提出 Periodic Skill Discovery (PSD) 框架，通过将状态映射到圆形潜空间来自然编码周期性，实现无监督地发现具有不同周期的多样化运动技能。 核心矛盾 核心矛盾：领域现状：无监督技能发现是强化学习中的重要方向，旨在不依赖外部奖励的情况下学习多样…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "无监督技能发现"
  - "周期性行为"
  - "圆形潜空间"
  - "运动控制"
---

# Periodic Skill Discovery

**会议**: NeurIPS 2025  
**arXiv**: [2511.03187](https://arxiv.org/abs/2511.03187)  
**代码**: 有 (jonghaepark.github.io/psd)  
**领域**: 强化学习 / 技能发现  
**关键词**: 无监督技能发现, 周期性行为, 圆形潜空间, 运动控制, 机器人

## 一句话总结

提出 Periodic Skill Discovery (PSD) 框架，通过将状态映射到圆形潜空间来自然编码周期性，实现无监督地发现具有不同周期的多样化运动技能。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：无监督技能发现是强化学习中的重要方向，旨在不依赖外部奖励的情况下学习多样化行为。然而现有方法存在一个被忽视的问题：

**忽略技能的周期性本质**：多数方法专注于最大化状态与技能间的互信息或最大化潜空间中的行进距离

**运动任务的周期性需求**：许多机器人任务（特别是locomotion）本质上需要不同时间尺度的周期性行为（如步行、跑步、跳跃）

**现有方法的局限**：基于互信息的DIAYN等方法难以自然地发现不同周期的技能

PSD 的核心动机：利用圆形潜空间的拓扑结构来天然编码周期性，从而发现具有多样化周期的运动技能。

### 解决思路

**本文目标**：### 整体框架

PSD 框架包含三个核心组件：
1. **圆形潜空间编码器**：将状态映射到单位圆上
2. **时间距离感知训练**：学习编码器以捕捉时间距离信息
3. **周期性技能策略**：基于潜空间中的周期性表示生成行为

### 关键设计

**圆形潜空间**：
- 将状态 $s$ 映射到单位圆 $\mathcal{S}^1$ 上的一个角度 $\phi(s) \in [0, 2\pi)。


## 方法详解

### 整体框架

PSD 框架包含三个核心组件：
1. **圆形潜空间编码器**：将状态映射到单位圆上
2. **时间距离感知训练**：学习编码器以捕捉时间距离信息
3. **周期性技能策略**：基于潜空间中的周期性表示生成行为

### 关键设计

**圆形潜空间**：
- 将状态 $s$ 映射到单位圆 $\mathcal{S}^1$ 上的一个角度 $\phi(s) \in [0, 2\pi)$
- 圆形拓扑天然支持周期性：从 $0$ 到 $2\pi$ 形成一个完整周期
- 不同的角速度对应不同的技能周期

**时间距离编码**：
- 训练编码器使得圆上两点间的弧长距离近似对应状态间的时间距离
- 这确保了：时间上相邻的状态在圆上也相邻
- 一个完整的运动周期对应圆上的一个完整旋转

**技能参数化**：
- 每个技能由其在圆形潜空间中的角频率 $\omega$ 参数化
- $\omega$ 小 → 慢周期运动（如缓步行走）
- $\omega$ 大 → 快周期运动（如快速奔跑）
- 不同的 $\omega$ 值自然产生不同周期的技能

**像素观测支持**：
- 编码器可以直接处理像素级观测
- 无需手工设计状态特征

### 损失函数 / 训练策略

- 时间距离对比损失：确保编码器正确捕捉时间距离
- 互信息项：增强技能多样性
- 策略梯度方法更新技能策略
- 分阶段训练：先训练编码器，再训练技能策略

## 实验关键数据

### 主实验

在多个MuJoCo机器人环境上的技能发现结果：

| 环境 | 方法 | 技能多样性 | 周期性覆盖 | 下游任务性能 |
|------|------|----------|----------|-----------|
| Ant | DIAYN | 中等 | 差 | 基线 |
| Ant | CIC | 较好 | 差 | 略好 |
| Ant | **PSD** | **最好** | **最好** | **最好** |
| HalfCheetah | DIAYN | 中等 | 差 | 基线 |
| HalfCheetah | CIC | 较好 | 差 | 略好 |
| HalfCheetah | **PSD** | **最好** | **最好** | **最好** |

下游任务（跨栏）性能：

| 方法 | Ant 跨栏成功率 | HalfCheetah 跨栏成功率 |
|------|-------------|-------------------|
| DIAYN | 低 | 低 |
| CIC | 中等 | 中等 |
| PSD | **高** | **高** |
| PSD + DIAYN (组合) | **最高** | **最高** |

### 消融实验

| 消融实验 | 变体 | 影响 |
|---------|------|------|
| 潜空间拓扑 | 欧几里得 vs 圆形 | 圆形显著优于欧几里得 |
| 时间距离 | 有 vs 无 | 时间距离编码关键 |
| 像素 vs 状态 | 观测类型 | 像素观测下仍有效 |
| 周期范围 | 窄 vs 宽 | 宽范围发现更多样化技能 |

### 关键发现

1. 圆形潜空间天然适配运动任务的周期性需求
2. PSD发现的技能展现出明显的多样化周期，从缓慢爬行到快速奔跑
3. 周期性技能在跨栏等下游任务上显著优于非周期性基线
4. PSD与现有技能发现方法（如DIAYN）互补，组合使用效果更好

## 亮点与洞察

1. **几何直觉清晰**：圆形=周期性，这是一个简洁而有效的设计
2. **填补空白**：首次系统性地在技能发现中利用周期性结构
3. **组合互补性**：PSD不替代而是补充现有方法，扩展了技能库
4. **像素级适用**：不依赖手工特征，适用于更通用的场景

## 局限与展望

1. 圆形潜空间主要适用于单频率周期运动，多频率复合运动的建模有待探索
2. 目前主要在locomotion任务上验证，其他周期性任务（如操控、飞行）的适用性无
3. 训练过程需要分阶段进行，端到端训练可能更高效
4. 未探索连续周期而非离散周期集合的技能发现
5. 真实机器人部署的sim-to-real问题未涉及

## 相关工作与启发

- **DIAYN**：基于互信息的无监督技能发现
- **CIC**：对比内在控制
- **DADS**：基于动力学感知的技能发现
- **Spectral RL**：利用频谱分解理解状态空间结构
- 启发：将拓扑/几何结构引入潜空间设计是一个有前景的方向

## 评分

- 新颖性：⭐⭐⭐⭐ (圆形潜空间编码周期性是创新思路)
- 技术深度：⭐⭐⭐⭐ (理论动机清晰)
- 实验充分性：⭐⭐⭐⭐ (多环境+下游任务+消融)
- 实用价值：⭐⭐⭐⭐ (运动控制的直接应用)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[AAAI 2026\] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search](../../AAAI2026/robotics/human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)
- [\[CVPR 2025\] Let Humanoids Hike! Integrative Skill Development on Complex Trails](../../CVPR2025/robotics/let_humanoids_hike_integrative_skill_development_on_complex_trails.md)
- [\[CVPR 2026\] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](../../CVPR2026/robotics/atomicvla_unlocking_the_potential_of_atomic_skill_learning_in_robots.md)

</div>

<!-- RELATED:END -->
