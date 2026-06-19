---
title: >-
  [论文解读] Real-World Reinforcement Learning of Active Perception Behaviors
description: >-
  [NeurIPS 2025][机器人][主动感知] 提出非对称优势加权回归（AAWR），在训练时利用额外特权传感器来估计更准确的优势函数，从而高效学习真实世界中的主动感知策略，在8个涵盖不同部分可观测程度的操控任务上均超越所有基线方法。 领域现状：机器人的瞬时传感器观测往往无法揭示完成任务所需的全部状态信息…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "主动感知"
  - "非对称强化学习"
  - "特权信息"
  - "POMDP"
  - "真实机器人"
---

# Real-World Reinforcement Learning of Active Perception Behaviors

**会议**: NeurIPS 2025  
**arXiv**: [2512.01188](https://arxiv.org/abs/2512.01188)  
**代码**: [https://penn-pal-lab.github.io/aawr/](https://penn-pal-lab.github.io/aawr/)  
**领域**: 强化学习  
**关键词**: 主动感知, 非对称强化学习, 特权信息, POMDP, 真实机器人

## 一句话总结

提出非对称优势加权回归（AAWR），在训练时利用额外特权传感器来估计更准确的优势函数，从而高效学习真实世界中的主动感知策略，在8个涵盖不同部分可观测程度的操控任务上均超越所有基线方法。

## 研究背景与动机

**领域现状**：机器人的瞬时传感器观测往往无法揭示完成任务所需的全部状态信息。在这种部分可观测场景下，最优策略通常需要显式的信息收集行为——如扫描场景寻找目标物体、用手腕相机探查遮挡区域。这类行为被称为主动感知（active perception）或交互感知（interactive perception）。

**现有痛点**：当前主流的机器人学习技术难以产生有效的主动感知行为。模仿学习不适合因为获取最优主动感知演示非常困难（如强迫操控者只通过手腕相机视角进行遥操作）。标准RL在完全可观测设置下已经够低效了，部分可观测设置更是雪上加霜。Sim-to-real迁移也不适用，因为主动感知与传感器能力紧密相关，而深度、RGB、触觉等传感器都很难在仿真中精确建模。当前最先进的通用策略（如π₀）在大量遥操作数据上训练，但面对简单的搜索任务也束手无策。

**核心矛盾**：RL理论上能通过交互学习主动感知，但实际样本效率太低；特权信息方法在sim-to-real中成功应用，但sim-to-real不适用于主动感知任务；离线RL可以利用次优演示但在POMDP中值函数估计不准确。

**本文目标** 如何在真实机器人上高效学习主动感知策略，同时只需少量次优演示和易获取的粗略初始策略？

**切入角度**：在训练时额外使用特权传感器（如物体检测器、分割掩码），为critic和value网络提供更准确的监督信号。关键理论发现是：在POMDP中做AWR式策略改进时，使用包含环境状态的特权优势估计是mathematically correct的做法。

**核心 idea**：用训练时的特权传感器为RL的优势函数提供比部分观测更准确的估计，从而高效引导策略学习信息收集行为。

## 方法详解

### 整体框架

AAWR采用offline-to-online RL范式：首先在少量次优离线演示上预训练策略和特权值函数，然后在真实环境中进行在线微调。训练时，策略只接收部分观测（如手腕相机图像），而critic/value网络额外接收特权观测（如物体位置、分割掩码）。部署时仅使用部分观测策略，不需要特权传感器。

### 关键设计

1. **非对称优势加权回归（AAWR）目标函数**:

    - 功能：在POMDP中正确实现策略改进的加权行为克隆
    - 核心思路：将POMDP转化为等价MDP，其状态为 $(s, z)$（环境状态+智能体状态）。在此MDP上推导AWR的KL约束策略改进目标，得到AAWR损失：$\mathcal{L}_{AAWR}(\pi) = \mathbb{E}_{(s,z) \sim d_\mu} \mathbb{E}_{a \sim \mu}[\exp(A^\mu(s,z,a)/\beta) \log \pi(a|z)]$。其中优势函数 $A^\mu(s,z,a) = Q^\mu(s,z,a) - V^\mu(s,z)$ 同时依赖环境状态 $s$ 和智能体状态 $z$，而策略 $\pi(a|z)$ 仅依赖 $z$。Theorem 1证明这一目标等价于POMDP中最大化期望策略改进的拉格朗日松弛
    - 设计动机：对称版本SAWR（去掉 $s$，只用 $z$ 估计优势）无法正确恢复最优解，因为仅使用智能体状态的优势估计器不足以估计等价MDP中的优势。此外，非特权值函数不是对应Bellman方程的不动点，而特权值函数是

2. **基于IQL的特权值函数训练**:

    - 功能：高效训练特权critic和value网络
    - 核心思路：使用Implicit Q-Learning（IQL）的期望分位数回归训练 $Q_\phi^\mu(s,z,a)$ 和 $V_\theta^\mu(s,z)$。IQL以其在离线RL和offline-to-online微调中的稳定性著称。Critic接收 $(o_t, s_t)$ 或增强观测 $(o_t, o_t^p)$，策略只接收 $o_t$
    - 设计动机：IQL避免了标准Q-learning中对策略外动作取max的需要，适合小样本offline-to-online设置。使用特权信息使得值函数在POMDP中也能准确估计

3. **Offline-to-Online训练流程**:

    - 功能：从少量次优演示启动到在线自主改进
    - 核心思路：分两阶段——离线阶段在 $\mathcal{D}_{off}$ 上用IQL目标更新 $Q,V$ 并用AAWR目标更新 $\pi$。在线阶段执行策略收集轨迹存入 $\mathcal{D}_{on}$，从两个buffer等量采样组成batch继续更新。关键是在线阶段策略仍使用特权值函数指导
    - 设计动机：次优演示提供初始覆盖，避免从头探索；在线微调允许策略通过试错发现演示中不包含的主动感知行为。如Distillation方法因特权专家不了解相机视野限制而陷入局部最优，AAWR则能通过在线探索突破

### 损失函数 / 训练策略

训练使用IQL的期望分位数回归损失训练值函数，AAWR加权交叉熵损失训练策略。离线/在线数据按1:1比例混合采样。

## 实验关键数据

### 主实验

仿真任务性能（10个种子平均）：

| 任务 | AAWR | AWR | BC |
|------|------|-----|----|
| Camouflage Pick | ~95% | ~45% | ~25% |
| Fully Obs. Pick | ~95% | ~30% | ~50% |
| AP Koch (最终成功率) | **100%** | ~70% | ~40% |

真实机器人Koch交互感知任务：

| 方法 | 抓取率(%) | 拾取率(%) |
|------|----------|----------|
| On. AAWR (ours) | **94** | **89** |
| Off. AAWR | 88 | 71 |
| On. AWR | 71 | 55 |
| Off. AWR | 65 | 62 |
| BC | 47 | 41 |

### 消融实验

π₀通用策略接力任务（真实Franka机器人）：

| 方法 | Bookshelf-P搜索 | 完成率 | Shelf-Cabinet搜索 | 完成率 |
|------|----------------|--------|-------------------|--------|
| AAWR | **92.4** | **44.4** | **78.2** | **40.0** |
| AWR | 79.6 | 0.0 | 52.3 | 10.0 |
| BC | 29.9 | 20.0 | 3.8 | 0.0 |

### 关键发现

- AAWR在所有8个任务上均优于无特权对应方法AWR，包括完全可观测任务（Fully Obs. Pick），说明特权critic不仅帮助处理遮挡，还帮助从像素中更好地提取信息
- AP Koch任务中，Distillation方法达到80%后停滞——特权专家直接奔向物体而不顾相机视野限制，导致蒸馏出来的策略学会了次优的"冲向中心"策略。AAWR作为策略迭代算法能通过在线探索发现扫描行为
- VIB（变分信息瓶颈）在部署时崩溃，因为特权信息不再可用
- 在π₀接力任务中，AAWR能学会在遮挡严重的场景中系统地搜索可能的藏物点

## 亮点与洞察

- **理论推导优雅**：从POMDP→等价MDP→约束策略改进→AAWR目标的推导路径清晰自然，Theorem 1给出了AAWR的理论依据，同时证明了SAWR的不足。这不是简单的"把特权信息喂给critic"的工程trick，而是有严格理论支撑的做法
- **真实世界部署能力突出**：在3种真实机器人上验证，只需100-150个次优演示和少量在线交互（最少1200步），不需要仿真环境。特权传感器可以是未标定的RGB相机上的简单物体检测器
- **与通用VLA策略的组合方式新颖**：AAWR训练搜索策略→找到目标后切换到π₀执行抓取，这种分工方式巧妙地解决了通用策略无法处理主动感知的问题，可推广到其他VLA

## 局限与展望

- 需要设计任务特定的奖励函数，这在某些任务中可能不容易
- 特权传感器的选择需要领域知识——什么信息算"特权"、用什么传感器获取需要人工设计
- 在线微调阶段仍需要特权传感器在场，这限制了某些场景的应用
- 当前智能体状态表示使用滑动窗口，未探索更复杂的历史编码（如Transformer），可能限制了在需要长期记忆的任务上的表现
- 搜索策略→执行策略的切换机制相对简单，更复杂的衔接方式可能带来进一步提升

## 相关工作与启发

- **vs 信息论主动感知（如next-best-view）**: 这类方法优化不确定性减少或信息增益，但不考虑任务约束——可能发现架子背面很"有信息量"但那里不可能有目标物体。AAWR通过任务奖励学习以任务为中心的搜索行为
- **vs Sim-to-real特权学习（如RMA, DAgger-style）**: 这些方法在模拟中利用数十亿特权状态转移进行训练，AAWR只用数百个真实世界轨迹，且不需要精确仿真主动感知所依赖的传感器
- **vs Distillation方法**: Distillation先训练特权专家再蒸馏到部分观测策略，但特权专家不了解感知限制会产生次优示范。AAWR直接在部分观测下优化策略，通过特权值函数间接利用特权信息，避免了这一问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论推导新颖地将特权信息与AWR在POMDP中结合，但asymmetric RL本身不是新概念
- 实验充分度: ⭐⭐⭐⭐⭐ 8个任务、3种机器人、仿真+真实、多种基线对比、与通用VLA的组合实验，非常全面
- 写作质量: ⭐⭐⭐⭐ 理论部分清晰严谨，实验描述充分，但paper较长
- 价值: ⭐⭐⭐⭐ 为真实世界主动感知学习提供了实用且理论扎实的方案，与VLA的组合方式有很好的应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] APPLE: Toward General Active Perception via Reinforcement Learning](../../ICLR2026/robotics/apple_toward_general_active_perception_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning](to_distill_or_decide_understanding_the_algorithmic_trade-off_in_partially_observ.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2025/robotics/sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)

</div>

<!-- RELATED:END -->
