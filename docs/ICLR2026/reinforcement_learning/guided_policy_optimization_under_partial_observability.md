---
title: >-
  [论文解读] Guided Policy Optimization under Partial Observability
description: >-
  [ICLR 2026][强化学习][POMDP] 针对"用特权信息训练老师再蒸馏给学生"时常出现的**模仿差距**问题，提出 GPO 框架：让 guider（用特权信息）和 learner（只看部分观测）**同时协同训练**，并通过"回溯"约束把 guider 始终拉回到 learner 能模仿的范围内，从理论上保证学生的监督学习等价于直接 RL，从而既榨干特权信息又不留下学不会的"过于优秀的老师"。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "POMDP"
  - "特权信息"
  - "师生学习"
  - "模仿差距"
  - "策略镜像下降"
  - "GPO"
---

# Guided Policy Optimization under Partial Observability

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SYLarqWqVH](https://openreview.net/forum?id=SYLarqWqVH)  
**代码**: 待确认  
**领域**: 强化学习 / 部分可观测 RL  
**关键词**: POMDP, 特权信息, 师生学习, 模仿差距, 策略镜像下降, GPO  

## 一句话总结
针对"用特权信息训练老师再蒸馏给学生"时常出现的**模仿差距**问题，提出 GPO 框架：让 guider（用特权信息）和 learner（只看部分观测）**同时协同训练**，并通过"回溯"约束把 guider 始终拉回到 learner 能模仿的范围内，从理论上保证学生的监督学习等价于直接 RL，从而既榨干特权信息又不留下学不会的"过于优秀的老师"。

## 研究背景与动机
- **领域现状**：在机器人等场景，真实部署时传感器是部分可观测、带噪声的（POMDP），但训练时（如仿真器）往往能拿到完整状态等**特权信息**。常见做法是用特权信息训练一个老师，再通过模仿学习 / 师生学习（TSL）/ 策略蒸馏把知识传给学生。
- **现有痛点**：当老师拿着特权信息时，它的最优策略可能是学生**根本无法模仿**的——这就是"过于优秀的老师"（impossibly good teacher）或**模仿差距**（imitation gap）。论文用 TigerDoor 例子点透：老师直接开正确的门，但学生必须先"听"才能定位老虎；老师从不"听"，学生跟着学只能在两扇门间瞎猜，期望收益只有 0.5，永远学不到先听后开的最优解。
- **核心矛盾**：现有补救要么在老师不可模仿时**退化成纯 RL**（浪费了昂贵的特权老师），要么通过 reward shaping 间接监督（监督信号弱、还需额外学习），且**没有任何方法能从理论上保证老师的监督一定有益**。
- **本文目标**：训练一个"**possibly good**"的老师——其策略始终待在学生的可模仿区域内，既能利用特权信息高效学习，又能被学生稳稳跟上。
- **核心 idea**：**协同训练 + 回溯约束**。受 Guided Policy Search (GPS) 启发，引入一个中间 agent（guider）用 RL + 特权信息快速学习，learner 通过监督学习模仿 guider，再用 learner 反过来约束 guider，使监督学习在理论上等价于对 learner 直接做 RL。

## 方法详解

### 整体框架
GPO 协同训练两个实体：**guider** $\mu(a|s)$（可访问特权信息/全局状态 $s$）和 **learner** $\pi(a|o)$（只看部分观测 $o$），通过四步迭代循环对齐两者，直到收敛。与传统 TSL 的关键区别是：老师不再独立预训练，而是与学生一起练，并且**用学生反向约束老师**。

```mermaid
flowchart LR
    A[数据收集<br/>用 guider μ 采轨迹] --> B[guider 训练<br/>RL 目标 V_μ 更新 μ→μ̂]
    B --> C[learner 训练<br/>最小化 D 模仿 μ̂ → π]
    C --> D[guider 回溯<br/>μ ← π 拉回可模仿区]
    D --> A
```

### 关键设计

**1. 协同训练 + 回溯，把监督学习"变成" RL：** GPO 的核心是四步循环——guider 用特权信息执行并采集轨迹、用 PPO 等信赖域 RL 更新 guider、learner 通过最小化 KL 散度 $D(\pi,\hat\mu)=\mathbb{E}[D_{KL}(\mu(\cdot|s),\pi(\cdot|o))]$ 模仿 guider、最后回溯令 $\mu^{(k+1)}(\cdot|s)=\pi^{(k+1)}(\cdot|o)$。论文证明（Proposition 1）：若 guider 用策略镜像下降更新，则 learner 的更新恰好等价于一个**带约束的策略镜像下降** $\pi^{(k+1)}=\arg\min_{\pi\in\Pi}\{-\eta_k\langle\nabla V(\pi^{(k)}),\pi\rangle+D_{\pi^{(k)}}(\pi,\pi^{(k)})\}$。这意味着 learner 即便从不直接与环境交互、只做监督学习，其策略更新也继承了 TRPO/PPO 的策略改进性质，从而获得"等价于直接 RL"的最优性保证。其好处在于**把高方差的 RL 梯度交给拿特权信息的 guider**，让部分可观测的 learner 只做低方差的监督学习，显著降低复杂度——例如训练抗噪鲁棒性时，guider 用干净输入、learner 用带噪输入做监督即可。

**2. GPO-penalty：自适应系数平衡"超前"与"回拉"：** 一个关键洞察是 guider **不必严格回溯到 learner**，只要待在可模仿区内即可；让 guider 略微超前反而能采集更好的轨迹。为此引入系数 $\alpha$ 调制 guider 的回溯损失 $L(\mu)=L_1(\mu)+\alpha L_3(\mu)$，其中 $\alpha$ 按回溯距离 $L_3(\mu)$ 相对阈值 $d$ 自适应：$\alpha=k\alpha\ (\text{若}\ L_3>kd)$，$\alpha=\alpha/k\ (\text{若}\ L_3<d/k)$，与 PPO-penalty 的 KL 惩罚调节如出一辙。同时由于 Proposition 1 表明 GPO+PPO 等价于直接对 learner 跑 PPO，论文给 learner 额外加了一个 PPO 目标 $L_4(\pi)$，合并为 $L(\pi)=\alpha L_4(\pi)+L_2(\pi)$：当 learner 完全跟上 guider 时 $\alpha\to0$、纯靠监督即可达最优；跟不上时 RL 项补位。Proposition 2 进一步说明当 $d_{targ}$ 较小时行为策略与 learner 策略足够接近，可以安全复用 guider 采的样本训练 learner。

**3. GPO-clip：双重裁剪 + 回溯掩码，把 guider 钉在可模仿边界：** 理想的 guider 应停在 learner 可模仿区的边界——太远 learner 跟不上，太近又失去探索与提供更优轨迹的价值。GPO-clip 借鉴 PPO-clip，用**双重裁剪函数**替换内层 ratio：$\rho^{\mu,\pi}_{clip}=\text{clip}(\text{clip}(\frac{\mu(a|s)}{\pi(a|o)},1-\delta,1+\delta)\cdot\frac{\pi(a|o)}{\beta(a|s)},1-\epsilon,1+\epsilon)$，当 guider 已偏离 learner 的 $\delta$ 区时停止其继续远离的更新。但由于多次更新中 $\pi$ 与 $\mu$ 的差距会累积、单靠双裁剪拉不回来，论文再加一个**回溯掩码** $m(s,a)=\mathbb{I}(\frac{\mu(a|s)}{\pi(a|o)}\notin(1-\delta,1+\delta))$，只在 guider 漂出 $\delta$ 区时才施加回溯惩罚，从而取代 penalty 版的自适应 $\alpha$。此外，因 guider 与 learner 解的是同一任务、策略结构相似，二者**共享同一策略网络**：guider 输入 $o_g=[s,o,1]$、learner 输入 $o_l=[\vec{0},o,0]$，用末位指示标志位区分两种角色，再配合 stop-gradient 写成统一损失 $L_{\text{GPO-clip}}(\theta)$。

## 实验关键数据

### 主实验
在 Brax 连续控制（去掉关节速度信息 + 加高斯噪声构造 POMDP，noise scale $\sigma\in\{0,0.1,0.2,0.3\}$）上，性能层级为 **GPO-clip > GPO-penalty > PPO-asym > GPO-naive > 其他 baseline**。依赖预训练特权老师的方法（DAgger / ADVISOR / ELF 等）仅在 Halfcheetah、Swimmer 上表现尚可，且随噪声增大**性能迅速崩塌**——因为老师对学生"太强"时几乎提供不了有用监督甚至有害。

| 方法 | 训练 guider | 行为策略 | 训练 learner | 价值函数 | 是否需预训练老师 |
|------|------|------|------|------|------|
| PPO | - | $\pi(a|o_l)$ | PPO | $V(o_l)$ | 否 |
| PPO-asym | - | $\pi(a|o_l)$ | PPO | $V(o_g)$ | 否 |
| PPO+BC | PPO | $\mu(a|o_g)$ | BC | $V(o_g)$ | 否 |
| A2D | PPO | $\pi(a|o_l)$ | BC | $V(o_l)$ | 否 |
| ADVISOR-co | PPO | $\pi(a|o_l)$ | BC+PPO | $V(o_l)$ | 否 |
| **GPO-naive** | PPO | $\mu(a|o_g)$ | BC | $V(o_g)$ | 否 |
| **GPO-penalty** | PPO | $\mu(a|o_g)$ | BC+PPO | $V(o_g)$ | 否 |
| **GPO-clip** | PPO | $\mu(a|o_g)$ | BC+PPO | $V(o_g)$ | 否 |

在 POPGym 的 15 个记忆型任务（Autoencode / Battleship / CountRecall / CartPole / RepeatPrevious 各 Easy/Medium/Hard）上结论一致：GPO-clip ≳ GPO-penalty > PPO-asym > PPO。在 TigerDoor / TigerDoor-alt 两个 didactic 任务上，所有 GPO 变体都收敛到最优，而 PPO+BC 仍停在次优——其中 **GPO-naive 纯靠监督学习就达到最优**，直接验证了 Proposition 1 的最优性保证。

### 消融实验
- **去掉监督（GPO-ablation = GPO-penalty 无监督）**：在 Humanoid 上 GPO-ablation 仍优于 PPO-asym，说明"用 guider 采的数据"本身就提升学习效率（更好的行为策略）。
- **去掉 RL（GPO-clip 纯监督）**：在记忆密集型任务（AutoencodeEasy）上 GPO-clip 优于 GPO-ablation / PPO+BC / PPO-asym，说明此类任务里**监督比 RL 更有价值**。
- **回溯的必要性**：PPO+BC 与 GPO-naive 仅差在是否约束 guider；PPO+BC 在噪声任务上崩、在记忆任务上才与 PPO-asym 持平，凸显把 guider 约束到可模仿区的重要性。
- **KL 阈值 $d$ / clip 参数 $\delta$**：单一参数下个别任务（BattleshipMedium、CountRecallHard）未夺冠，调参后可改善。

### 关键发现
- **预训练老师 + TSL 为何常失败**：Fig.5 显示 ADVISOR、PPO+BC 在全可观测 Ant（老师训练环境）表现好，但切到部分可观测 Ant 时老师被判为不可模仿，算法直接**退化成 PPO**。
- GPO 的优势来自两点：learner 的有效 RL 训练（更好的行为数据）+ guider 的有效监督（被约束在可模仿区却仍学得快）。

## 亮点与洞察
- **理论桥接**：用策略镜像下降把"learner 的监督学习"严格证明等价于"learner 的带约束 RL"，这是大多数 TSL 方法缺失的最优性保证。
- **视角转换**："possibly good teacher" 比 "impossibly good teacher" 更可学——主动把老师拉到学生能够得着的水平，而非被动在老师不可模仿时弃用它。
- **工程友好**：guider/learner 共享网络 + 输入标志位，几乎不增加额外参数；penalty/clip 两个版本对应 PPO 的两种风格，易于落地。

## 局限与展望
- 仍需在训练时访问特权信息/全局状态，对没有仿真器或无法构造特权输入的场景不适用。
- 单一超参（KL 阈值 / clip $\delta$）在所有任务上未必最优，个别记忆任务需逐任务调参。
- 记忆模型实际无法存下全部关键信息，POPGym 复杂任务上 guider 的"理论零模仿差距"假设会打折扣。
- 与基于"从部分观测重构特权信息"的方法（要求 MDP decodable）正交，未深入探讨二者结合。

## 相关工作与启发
- **Guided Policy Search (GPS)**：GPO 的直接灵感来源——引入中间 agent 指导策略学习，但 GPS 是 model-based 轨迹优化，GPO 将其思想迁移到 POMDP 下的 model-free RL。
- **师生学习 / 策略蒸馏（ADVISOR、TGRL、ELF、A2D、DAgger）**：现有方法通过动态权重退化为 RL 或 reward shaping 间接监督，GPO 用协同训练 + 回溯约束从根上避免"不可模仿老师"。
- **策略镜像下降（TRPO / PPO）**：作为理论分析的统一框架，使 GPO 的最优性论证得以成立。
- **启发**：当存在"训练时强、部署时弱"的信息不对称时，与其训练一个最强的老师再硬蒸馏，不如让老师和学生协同演化、把老师约束在学生的能力边界上，这一思路可迁移到 sim-to-real、多模态蒸馏等更广场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— "possibly good teacher" + 协同训练 + 回溯约束的组合提出明确，并给出监督学习等价于约束 RL 的理论保证，区别于以往"退化成 RL"或"reward shaping"的补救思路。
- **实验充分度**: ⭐⭐⭐⭐ —— 覆盖 didactic（TigerDoor）、Brax 连续控制（多噪声）、POPGym 记忆任务三类场景，对比 13 个 baseline，并有针对 RL/监督/回溯各组件的消融。
- **写作质量**: ⭐⭐⭐⭐ —— 用 TigerDoor 把模仿差距讲得直观，理论与实现（penalty/clip 两版）衔接清晰；公式较密集，需一定 RL 背景。
- **价值**: ⭐⭐⭐⭐ —— 为"如何用好特权信息"这一 sim-to-real / POMDP 核心问题提供了有理论支撑且即插即用的框架，实用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] R2PS: Worst-Case Robust Real-Time Pursuit Strategies under Partial Observability](r2ps_worst-case_robust_real-time_pursuit_strategies_under_partial_observability.md)
- [\[ICLR 2026\] Multi-Agent Guided Policy Optimization](multi-agent_guided_policy_optimization.md)
- [\[ICLR 2026\] Boosting Multi-Domain Reasoning of LLMs via Curvature-Guided Policy Optimization](boosting_multi-domain_reasoning_of_llms_via_curvature-guided_policy_optimization.md)
- [\[ICLR 2026\] Single-stream Policy Optimization](single-stream_policy_optimization.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](../../ACL2026/reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
