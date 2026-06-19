---
title: >-
  [论文解读] A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge
description: >-
  [AAAI 2026][强化学习][UAV swarm] 提出 reMARL 框架，利用图像处理领域知识（active contour model）设计多智能体强化学习奖励函数，实现无人机集群的协作避碰，相比传统元启发式方法反应时间缩短 98.75%、能耗降低 85.37%。 1. 领域现状： 无人机集群避碰需要安全性和能效…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "UAV swarm"
  - "collision avoidance"
  - "MARL"
  - "domain knowledge"
  - "active contour model"
---

# A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge

**会议**: AAAI 2026  
**arXiv**: [2507.10913](https://arxiv.org/abs/2507.10913)  
**代码**: 无  
**领域**: Agent / Multi-Agent Reinforcement Learning  
**关键词**: UAV swarm, collision avoidance, MARL, domain knowledge, active contour model

## 一句话总结

提出 reMARL 框架，利用图像处理领域知识（active contour model）设计多智能体强化学习奖励函数，实现无人机集群的协作避碰，相比传统元启发式方法反应时间缩短 98.75%、能耗降低 85.37%。

## 研究背景与动机

1. **领域现状**: 无人机集群避碰需要安全性和能效性。传统方法包括 Velocity Obstacle (VO)、人工势场 (APF)、元启发式优化等，近年来 MARL 方法如 COMA、VDN、QMIX 等被广泛探索。
2. **现有痛点**: VO 方法频繁改变速度导致低能效；APF 存在局部最优陷阱；元启发式方法反应时间过长不适合实时应用。MARL 方法中，观察共享方案在集群规模增大时性能下降，信用分配方案（VDN/QMIX 的 IGM 假设）导致无界发散和不可预测行为。
3. **核心矛盾**: 如何在不依赖复杂信用分配或观察共享机制的前提下，实现大规模无人机集群的高效协作避碰。
4. **本文要解决什么？** 设计一种可扩展到大规模集群的 MARL 框架，通过域知识驱动的奖励函数消除对复杂网络结构的依赖。
5. **切入角度**: 借鉴图像处理中的 active contour model，将环境建模为二维势场，设计使无人机沿等高线飞行的奖励函数。
6. **核心idea一句话**: 用图像处理的等高线提取思想设计 MARL 奖励函数，使协作行为从个体奖励最大化中自然涌现。

## 方法详解

### 整体框架

reMARL 将环境映射为二维势场 $\Phi(q)$，障碍物和集群虚拟中心作为势场峰值。通过 active contour model 的代价函数设计个体奖励，每个 UAV 独立使用 DDPG 学习策略，协作行为通过共享势场自然涌现。

### 关键设计

1. **势场构建** — 叠加障碍物排斥场 $\Phi_o(q)$ 和集群排斥场 $\Phi_s(q)$。障碍物场强度与距离平方成反比，安全距离 $d_{safe}$ 内场强恒定取最大值。集群作为整体建模为单一排斥场，避免 UAV 位于各自单独排斥场的峰值。

2. **域知识驱动奖励** — 奖励函数 $r = -f(S, \Phi(q)) + r_{form} \cdot r_{collide}$，其中：
    - **Contour 部分**: 基于 active contour model 代价函数 $f(S) = \int \frac{1}{2}|S''|^2 - \frac{1}{2}|\nabla\Phi|^2 d\rho$，最小化该函数使轨迹逼近势场等高线
    - **Swarming 部分**: $r_{form}$ 通过速度余弦相似度保持编队，$r_{collide}$ 作为安全硬约束

3. **PSO 等高线调整** — 用粒子群优化调整各 UAV 位置使相邻等高线距离满足最小安全间距 $\bar{d}_{U2U}$，代价函数 $f_{pso} = f_{thres} + f_{shift}$ 同时保证安全距离和最小位移。

### 损失函数 / 训练策略

- 每个 UAV 独立使用 DDPG（策略网络 + 价值网络），动作为速度方向变化量 $[-\pi/4, \pi/4]$
- 策略网络：FC(256, ReLU) → Output(1, tanh)
- 价值网络：观察和动作分别 FC(256) → 拼接 → FC(256) → Output(1)
- TD loss 更新价值网络，策略梯度更新策略网络

## 实验关键数据

### 主实验

在 2U1O 到 10U1O、3U2O 到 7U2O 等多种场景下对比 COMA、VDN、QMIX、MAPPO、IQL：

| 指标 | reMARL | 元启发式方法 | 提升 |
|------|--------|-------------|------|
| 反应时间 (s) | 0.006 ± 0.03 | 0.48 ± 0.05 | **98.75%** |
| 能耗（曲率积分） | 19.72 ± 31.8 | 134.88 ± 8.3 | **85.37%** |
| 最小 UAV-障碍物距离 | 24.78 | 38.70 | 35.96% 更接近 |
| 最小 UAV-UAV 距离 | 29.45 | 40.31 | 26.94% 更接近 |

### 消融实验

| 方法 | 集群规模 ≤ 3 | 集群规模 > 3 |
|------|-------------|-------------|
| SOTA MARL（无 Contour 奖励） | 优于 reMARL | 性能急剧下降 |
| reMARL（完整奖励） | 略逊 | **显著优于所有 baseline** |

学习曲线显示：集群规模 ≤ 3 时 UAV 无需沿等高线即可安全飞行；规模增大后，遵循等高线成为最优策略。

### 关键发现

- reMARL 的优势完全来自 Contour 奖励项，DDPG 本身无协作机制
- UAV 在训练后能发现等高线不可行时的替代路径（如在障碍物间穿行），体现了超越域知识先验的适应能力
- 观察共享仅用于构建奖励而非网络结构，避免了维度灾难

## 亮点与洞察

- **跨学科创新**: 将图像处理的 active contour model 迁移到 MARL 奖励设计，idea 新颖且有效
- **可扩展性强**: 消除了信用分配和观察共享对网络结构的依赖，支持大规模集群（10+ UAV）
- **超越先验知识**: 训练后的策略不仅能沿等高线飞行，还能在等高线不可行时自适应寻找更优路径

## 局限性 / 可改进方向

- 仅在二维空间验证，高度变化的三维避碰未考虑
- PSO 等高线调整仅在训练时使用，部署时如何处理动态变化未明确
- 假设 UAV 速度恒定，实际飞行中速度变化不可避免
- 可引入更复杂的障碍物运动模型（当前障碍物随机运动）

## 相关工作与启发

- 域知识驱动奖励的思路可推广到其他 MARL 任务场景
- 等高线思想在势场方法中有悠久历史，本文将其与 RL 结合是有意义的扩展

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 跨学科奖励设计思路独特
- 实验充分度: ⭐⭐⭐⭐ 多场景对比充分但都是仿真环境
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分公式排版不佳
- 价值: ⭐⭐⭐⭐ 对 UAV 集群避碰有参考价值，但实际部署验证缺失

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)
- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[AAAI 2026\] ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing](charteditor_a_reinforcement_learning_framework_for_robust_chart_editing.md)
- [\[AAAI 2026\] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](../../ICLR2026/reinforcement_learning/rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)

</div>

<!-- RELATED:END -->
