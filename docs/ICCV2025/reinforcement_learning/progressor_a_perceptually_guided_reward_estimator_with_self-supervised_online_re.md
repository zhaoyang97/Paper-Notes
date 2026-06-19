---
title: >-
  [论文解读] Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement
description: >-
  [ICCV 2025][强化学习][视觉奖励学习] 提出Progressor框架，从无标注视频中自监督学习任务无关的奖励函数，通过预测任务进度分布提供稠密奖励信号，并在在线RL训练中通过对抗性push-back策略应对分布偏移问题。 强化学习的实际应用面临奖励函数设计难题：手动设计的稠密奖励费时且可能导致非预期行为…
tags:
  - "ICCV 2025"
  - "强化学习"
  - "视觉奖励学习"
  - "自监督"
  - "目标条件RL"
  - "对抗在线精炼"
  - "任务进度估计"
---

# Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement

**会议**: ICCV 2025  
**arXiv**: [2411.17764](https://arxiv.org/abs/2411.17764)  
**代码**: [https://ripl.github.io/progressor](https://ripl.github.io/progressor)  
**领域**: 强化学习 / 机器人学习  
**关键词**: 视觉奖励学习, 自监督, 目标条件RL, 对抗在线精炼, 任务进度估计

## 一句话总结

提出Progressor框架，从无标注视频中自监督学习任务无关的奖励函数，通过预测任务进度分布提供稠密奖励信号，并在在线RL训练中通过对抗性push-back策略应对分布偏移问题。

## 研究背景与动机

强化学习的实际应用面临奖励函数设计难题：手动设计的稠密奖励费时且可能导致非预期行为，稀疏奖励则样本效率低下。从无标注视频中学习奖励函数是一个有前途的替代方案，但现有方法存在以下不足：

**时间对比学习（TCN）**对帧率敏感，假设奖励对称

**生成式方法（VIPER、Diffusion Reward）**需要昂贵的生成过程来估计奖励

**Rank2Reward**需要为每个任务训练不同模型，且通过额外分类器处理分布偏移

核心挑战在于：预训练的奖励模型仅在专家轨迹上训练，当策略在线探索时产生的非专家观测与训练分布不同，导致奖励估计失真。

## 方法详解

### 整体框架

Progressor分为两阶段：（1）在专家视频上自监督预训练奖励模型，学习预测任务进度分布；（2）在线RL训练中，通过对抗性push-back策略持续精炼奖励模型，同时训练策略网络。

### 关键设计

1. **自监督进度估计**: 给定帧三元组 $(o_i, o_j, o_g)$（初始、当前、目标观测），通过归一化相对位置定义进度标签：$\delta(o_i, o_j, o_g) = |j-i|/|g-i| \in [0,1]$。将进度建模为高斯分布 $\mathcal{N}(\mu_{\tau_k}, \sigma_{\tau_k}^2)$，其中均值为进度标签，标准差 $\sigma = \max(1/(g-i), \epsilon)$。编码器 $E_\theta$ 使用共享视觉骨干网络预测进度分布参数 $(\mu, \sigma^2)$，通过KL散度优化：$\mathcal{L}_{expert} = D_{KL}(p_{target} \| E_\theta)$。

2. **奖励函数设计**: 将预测的进度分布转化为奖励信号：$r_\theta(o_i, o_j, o_g) = \mu - \alpha \mathcal{H}(\mathcal{N}(\mu, \sigma^2))$。第一项 $\mu$ 表示当前进度，第二项惩罚高不确定性的预测（$\alpha=0.4$）。由于任务进度单调递增，该进度估计天然适合作为稠密奖励。

3. **对抗性Push-Back在线精炼**: 在线RL中，策略早期产生的随机动作会导致分布外观测。为此，对来自在线rollout的帧三元组，将当前进度预测乘以衰减因子 $\beta=0.9$，构造push-back目标分布 $p_{push-back} = sg(\mathcal{N}(\beta\mu_{\tau'}, 1/(g-i)^2))$，通过 $\mathcal{L}_{push-back} = D_{KL}(p_{push-back} \| E_\theta)$ 更新模型。结合专家数据上的 $\mathcal{L}_{expert}$ 进行混合优化，防止模型被非专家数据过度偏置。

### 损失函数 / 训练策略

- **预训练**：在专家视频上通过KL散度训练进度估计模型
- **在线精炼**：交替使用专家数据损失和push-back损失
- **策略优化**：使用DrQ-v2算法，以Progressor奖励替代环境奖励
- **真实机器人实验**：使用ResNet34作为骨干网络，在EPIC-KITCHENS（约129万帧）上预训练，batch size 128，Adam优化器，学习率 $2 \times 10^{-4}$，训练30000次迭代

## 实验关键数据

### 主实验 — Meta-World仿真

在6个Meta-World操作任务上（door-open, drawer-open, hammer, peg-insert-side, pick-place, reach），训练1.5M步，对比TCN、GAIL、Rank2Reward：

| 方法 | door-open | drawer-open | hammer | peg-insert | pick-place | reach |
|------|-----------|-------------|--------|------------|------------|-------|
| TCN | 失败 | 失败 | 失败 | 失败 | 失败 | 失败 |
| GAIL | 低 | 中 | 低 | 失败 | 低 | 中 |
| Rank2Reward | 低 | 中 | 中 | 失败 | 中 | 高 |
| **Progressor** | **高** | **高** | **高** | **高** | **高** | **高** |

Progressor在几乎所有任务上显著优于基线，尤其在困难任务（door-open、peg-insert-side）上其他方法完全失败。在drawer-open和hammer任务中，仅需10%训练步数即可超越基线。

### 消融实验 — Push-Back的影响

| 任务 | 有Push-Back | 无Push-Back |
|------|-----------|------------|
| door-open | 高成功率 | 低成功率 |
| hammer | 高成功率 | 中等成功率 |
| drawer-open | 高 | 高（接近） |
| reach | 高（后期略降） | 高（稳定） |

Push-back在困难任务上贡献显著。在reach任务中push-back可能过度惩罚已接近专家的行为，导致后期性能轻微下降。

### 真实机器人实验

在4个UR5桌面操作任务上（20个成功+20个失败demo），使用RWR-ACT进行离线RL：

| 方法 | Drawer-Open | Drawer-Close | Push-Block | Pick-Place-Cup |
|------|-------------|-------------|------------|----------------|
| Vanilla ACT | 低 | 中 | 极低 | 极低 |
| R3M-RWR-ACT | 中 | 高 | 低 | 低 |
| VIP-RWR-ACT | 中 | 高 | 低 | 低 |
| **Progressor-RWR-ACT** | **高** | **高** | **高** | **高** |

Progressor在所有任务上一致优于所有基线，尤其在困难任务（Push-Block、Pick-Place-Cup）上优势明显。

### 关键发现

- 仅在专家数据上训练而不进行在线更新的TCN在所有任务上失败，证明在线训练对应对分布偏移的重要性
- Progressor在成功和失败轨迹间提供更显著的奖励区分度（在步骤125后差异尤为明显）
- 在EPIC-KITCHENS人类视频上预训练的模型可零样本迁移到机器人任务，生成合理的奖励预测

## 亮点与洞察

- **任务无关的统一奖励模型**：单一模型即可处理多种任务，无需像Rank2Reward那样为每个任务单独训练
- **进度分布而非点估计**：建模为高斯分布，能估计不确定性，在奖励中显式惩罚高方差预测
- **Push-Back策略优雅简洁**：仅通过一个衰减因子 $\beta$ 即可实现在线奖励精炼，无需额外的分类器网络
- **从人类视频到机器人的零样本迁移**：在EPIC-KITCHENS上预训练后直接用于机器人任务

## 局限与展望

- 假设任务进度线性单调递增，不适用于循环观测的任务（如DeepMind Control Suite中的部分环境）
- 进度估计为单峰预测，无法处理多路径任务
- 衰减因子 $\beta$ 固定为0.9，动态调整可能进一步提升性能
- 在reach任务中观察到push-back导致后期性能下降的问题需要解决

## 相关工作与启发

- 与Rank2Reward的关键区别：同样利用时序排序信息，但Progressor使用分布估计而非排序分类，且为任务无关的统一模型
- 与VIP/R3M的对比：这些方法使用对比学习获取视觉表征，但在区分成功/失败轨迹方面不如进度估计
- Push-back思想源自域对抗训练，可启发其他需要在线适应的视觉奖励学习方法

## 评分

- **新颖性**: ⭐⭐⭐⭐ 进度分布估计和对抗push-back策略设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 仿真+真实机器人，多基线对比和消融
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，方法动机充分
- **综合价值**: ⭐⭐⭐⭐ 为视觉奖励学习提供了实用且有效的框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[ICML 2025\] Reward-free World Models for Online Imitation Learning](../../ICML2025/reinforcement_learning/reward-free_world_models_for_online_imitation_learning.md)
- [\[ICCV 2025\] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)
- [\[ICML 2025\] Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning](../../ICML2025/reinforcement_learning/log-sum-exponential_estimator_for_off-policy_evaluation_and_learning.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](../../ICML2025/reinforcement_learning/online_pre-training_for_offline-to-online_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
