---
title: >-
  [论文解读] Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning
description: >-
  [ICLR 2026][强化学习][Actor-Critic] NSAC 把 A2C 中黑箱的 actor 换成「加性规则集成」(additive rule ensembles)，用神经网络 critic 估值、用一组 IF-THEN 规则直接做决策，并通过策略梯度 + 正交梯度提升在线学习规则，做到与 DQN/PPO/A2C 等黑箱方法相当的性能同时具备内生可解释性。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "Actor-Critic"
  - "规则集成"
  - "可解释性"
  - "A2C"
  - "正交梯度提升"
  - "神经符号"
---

# Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0XIsA0PxJM](https://openreview.net/forum?id=0XIsA0PxJM)  
**代码**: 待确认  
**领域**: 可解释强化学习 / 神经符号  
**关键词**: Actor-Critic, 规则集成, 可解释性, A2C, 正交梯度提升, 神经符号  

## 一句话总结
NSAC 把 A2C 中黑箱的 actor 换成「加性规则集成」(additive rule ensembles)，用神经网络 critic 估值、用一组 IF-THEN 规则直接做决策，并通过策略梯度 + 正交梯度提升在线学习规则，做到与 DQN/PPO/A2C 等黑箱方法相当的性能同时具备内生可解释性。

## 研究背景与动机
- **领域现状**：基于神经网络的 actor-critic（A2C、PPO）凭借强大的函数逼近能力在高维状态-动作空间中表现出色，但 actor 本质是黑箱，无法揭示"为什么选这个动作"。在医疗、金融、法律等需要透明与合规的场景下，这一缺陷成为落地障碍。
- **现有痛点**：符号化 RL 试图提供透明决策，但普遍存在三类问题——(i) **依赖预定义知识**（专家规则、预训练教师），换个领域就失效；(ii) **后验解释失真**：从黑箱教师蒸馏出决策树/规则，存在 size-fidelity 权衡，且后验解释器（如 SHAP）只标相关而非因果，扰动一下模型就翻盘；(iii) **表达过于复杂**：符号回归/程序合成产生含三角函数、对数、程序语法的表达式，人类无法 simulate。
- **核心矛盾**：神经网络的**可扩展性/适应性** 与 符号系统的**透明性/可追溯性** 难以兼得——纯符号难处理复杂性，纯神经无法解释。
- **本文目标**：构造一个 actor-critic 框架，actor 直接由人类可读的规则构成、从环境交互中**直接学习**（不靠预定义知识、不靠后验蒸馏），同时性能不输黑箱 RL。
- **核心 idea**：**[神经 critic + 符号 actor]** 让神经网络负责值估计这种"计算密集但不需解释"的部分，让加性规则集成承担"需要解释"的决策部分；**[加性而非层级]** 用扁平的规则集成代替决策树以回避树的可扩展性瓶颈；**[正交梯度提升在线挖规则]** 用 OGB 自动发现规则条件，仅需基础特征比较。

## 方法详解

### 整体框架
NSAC 保留 A2C 的 actor-critic 反馈环：critic 仍是神经网络 $V_\phi(s)$，用 TD 误差最小化更新；actor 则被替换为「每个动作一组加性规则集成」。每个动作 $a$ 对应一个集成 $f_a(s)=\sum_{j=1}^{k} w_{a,j}q_{a,j}(s)$，直接预测该动作的优势值，再经 softmax 得到策略 $\pi(a|s)=\frac{\exp(f_a(s))}{\sum_{a'}\exp(f_{a'}(s))}$。每一步用 critic 算出的优势 $A_t$ 作为学习信号，对被选中/未被选中的动作分别施加策略梯度，并配合「规则替换 + 正交梯度提升 + 全修正权重重算」迭代刷新规则集，使符号策略既反映真实决策又保持紧凑。

```mermaid
flowchart LR
    S[状态 s] --> C[神经 Critic V_phi]
    S --> A[符号 Actor: 每动作一组规则集成 f_a]
    A -->|softmax| P[策略 pi a|s]
    P --> ACT[执行动作 a]
    ACT --> ENV[环境]
    ENV -->|r, s'| TD[TD 误差 delta = r + gamma V_s' - V_s]
    C --> TD
    TD -->|更新| C
    TD -->|优势 A_t| RU[规则替换 + OGB 挖新规则 + 全修正权重]
    RU --> A
```

### 关键设计

**1. 加性规则集成做 actor：用 IF-THEN 规则替换神经策略头，回避决策树的可扩展瓶颈。** 不同于决策树的层级结构（树一大就难读），NSAC 为每个动作维护一个**扁平的加性规则集成**：每条规则是若干布尔命题的合取 $q_i(x)=\prod_{j=1}^{c_i}p_{i,j}(x)$（每个 $p_{i,j}$ 是阈值比较 $\mathbb{I}[s\cdot x^{(j)}\le s\cdot x_l^{(j)}]$），整体输出是规则按权重 $w_i$ 的线性组合。由于每条规则平等贡献、可独立读作"IF 条件 THEN 权重"，模型天然满足 Murdoch et al. 提出的 simulatability（可模拟）、modularity（模块化）、low complexity（低复杂度）三条可解释判据，这正是论文给出的可解释性形式化保证。

**2. 在 A2C 框架内对规则做策略梯度更新：把 $(w,q)$ 当作可微参数，按动作是否被选中分两支。** actor 的损失是带正则的策略梯度目标 $\nabla L_\lambda(w,q)=-\mathbb{E}[\nabla_{w,q}\log\pi(a_t|s_t,w,q)A_t]+\lambda\nabla_{w,q}\|w\|_2^2$。把 $\theta=(w;q)$ 合并后推导出二阶梯度，并区分两种情形——当集成 $f_a$ 对应当前实际执行动作（$a=a_t$）时 $\nabla L = \mathbb{E}[-\nabla A(s_t,a_t)(1-\pi(a_t|s))]$；当不对应（$a\ne a_t$）时 $\nabla L = \mathbb{E}[\nabla A(s_t,a_t)\pi(a_t|s)]$。这种"选中的动作往优势方向推、其余动作往反方向压"的分支让规则集成的更新与真实采样动作对齐。critic 端则照常用 $L_V(\phi)=\mathbb{E}[(R_t+\gamma V(s_{t+1})-V_\phi(s_t))^2]$ 最小化 TD 误差。论文同时给出 NSAC 收敛到局部最优的理论证明。

**3. 正交梯度提升 (OGB) 在线挖规则 + 全修正权重重算：保证规则一般化且权重最优。** 每次迭代不是简单堆规则，而是用 OGB 目标 $\text{obj}_{\text{ogb}}(q)=|g_\perp^\top q|/(\|q_\perp\|+\epsilon)$ 在候选维度里找新规则，其中 $g_\perp=g-BB^\top g$ 是把梯度投影到已有规则正交补空间——这逼迫新规则提供"新方向"的信息而非冗余，从而选出更一般、风险-复杂度权衡更好的规则。配套的"规则替换"会删掉权重最小的规则、再用全修正（fully-corrective）优化重算所有权重（解 $k\times k$ Gram 矩阵 $Q^\top Q$ 上的凸问题）。代价是每条规则 $O(d^2nk)$，比标准梯度提升的 $O(d^2n)$ 多 $k$ 倍，但在可解释场景常用的小规则集下可控。

## 实验关键数据

### 主实验（平均回报 ± 标准差，10 个随机种子）
在 5 个经典 Gym 环境 + Sinergym HVAC 楼宇控制基准上，对比黑箱 RL（Q-table/DQN/A2C/PPO/SDSAC/SACBBF/Rainbow）与符号方法（SYMPOL/πaffine-D/D-SDT）。

| 环境 | DQN | A2C | PPO | Rainbow | SYMPOL(符号最佳) | **NSAC** |
|---|---|---|---|---|---|---|
| MountainCar-v0 | -135.07 | -157.51 | -150.40 | -137.76 | -200 | **-132.25** |
| Acrobot-v1 | -112.68 | -98.93 | -82.63 | -89.75 | -80.02 | -87.71 |
| CartPole-v1 | 161.00 | 453.51 | 498.73 | 498.53 | 500 | 499.14 |
| Blackjack-v1 | -0.06 | -0.07 | -0.06 | -0.06 | -0.06 | -0.06 |
| Postman | 31.91 | 24.23 | 34.35 | 34.67 | 25.34 | 27.14 |
| HVAC-1Zone | -1445367 | -1334562 | -1865276 | -1465732 | -1478783 | **-1251321** |
| HVAC-5Zone | -1876253 | -1984843 | -1676288 | -1602142 | -1586352 | **-1463601** |

- NSAC 在 MountainCar、两个 HVAC 任务上**取得全场最佳**，其余任务与最强黑箱基线持平。
- 符号基线（SYMPOL/πaffine-D/D-SDT）在简单任务可与神经方法掰手腕，但**一上规模就崩**（如 πaffine-D 在 Acrobot 跌到 -425、CartPole 仅 109），而 NSAC 全环境稳定，且在最难的 HVAC 上明显领先所有符号方法。

### 消融实验（CartPole-v1，规则数 × warm start）

| 配置 | 现象 |
|---|---|
| 每动作规则数 5/10/**12**/20/30/40/50 | **12 条规则 + warm start** 回报最高（≈500）；<10 条覆盖不足，过多则过拟合噪声 |
| Warm start vs 无 | 规则数较少时 warm start 更关键，能更快收敛、避开差的局部最优 |

### 关键发现
- 规则数与性能非单调：存在"复杂度-表达力"甜点（12 条/动作），印证可解释模型不是越大越好。
- 学到的规则可直接读出（如 "IF vel<4 and pos>8 → w=0.32"），支持对策略做定性诊断（RQ3）。

## 亮点与洞察
- **分工哲学清晰**：把"需要计算"的值估计交给神经网络、"需要解释"的决策交给符号规则，避开了"全符号难处理复杂性 / 全神经无法解释"的两难，是神经符号在 RL 里一个干净的落点。
- **直接从环境学规则**：不蒸馏黑箱教师、不要预定义知识，因此规则反映的是**真实决策过程**而非后验近似——直击 SHAP 等后验解释"只标相关、扰动即翻盘"的软肋。
- **可解释性给了形式化判据**：把规则策略对应到 simulatability/modularity/low-complexity 三条标准，而不只是"看起来能读"，并配收敛性证明。
- **HVAC 真实基准**：在高维、强耦合、带舒适约束的多区楼宇控制上拿下最佳，说明加性规则集成不止能玩具任务。

## 局限与展望
- **离散动作 + softmax**：框架基于每动作一个规则集成 + softmax，连续动作空间如何扩展未充分展开。
- **计算开销**：OGB 比标准梯度提升贵 $k$ 倍（$O(d^2nk)$），全修正权重 $O(k^2n)$，仅在小规则集下可控，规则数一大成本上升。
- **规则条件形式受限**：命题是阈值比较的合取，对需要非线性/关系型条件的任务表达力有限。
- **可解释性的人因验证缺失**：论文用形式化判据论证可读性，但没有人类被试实验证明"人确实更信任/更会用"。
- **环境规模**：仍偏经典控制 + 单一真实基准，缺大规模/视觉输入域的检验。

## 相关工作与启发
- **树策略 RL**（Native tree / 蒸馏树 / 可微树 D-SDT）：可解释但有 size-fidelity 与可扩展性问题；NSAC 用"加性扁平规则"换"层级树"。
- **符号策略发现**（神经引导 DSP、遗传编程、程序合成 πaffine）：摆脱了预定义知识，但产出复杂数学/程序表达；NSAC 坚持"基础阈值比较 + IF-THEN"以保人类可读。
- **加性规则集成 / OGB**（Friedman-Popescu, Yang et al. 2024）：本文把监督学习里的规则集成首次系统嵌进 actor-critic 的策略梯度环。
- **启发**：神经符号不必"端到端可微一刀切"，按"是否需要解释"对模块做职责切分，可能是落地可信 RL 更务实的路线。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把加性规则集成 + OGB 嵌入 A2C 策略梯度、做内生可解释 actor 的组合是新的，且配收敛证明与可解释判据；但各组件（A2C、规则集成、OGB）均为已有，属巧妙集成而非全新机制。
- **实验充分度**: ⭐⭐⭐ — 覆盖 5 个经典环境 + 真实 HVAC、对比黑箱与符号两类基线、10 种子、含规则数/warm start 消融；但环境偏小、连续动作与视觉域缺位、无人因可解释性验证。
- **写作质量**: ⭐⭐⭐⭐ — 动机与"后验解释失真"的批判到位，公式推导完整，图示对比直观。
- **价值**: ⭐⭐⭐⭐ — 为需要透明决策的高风险 RL 场景提供了"性能不掉 + 规则可读 + 直接从环境学"的实用方案，HVAC 上的领先有说服力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Finite-Time Analysis of Actor-Critic Methods with Deep Neural Network Approximation](finite-time_analysis_of_actor-critic_methods_with_deep_neural_network_approximat.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns](chunking_the_critic_a_transformer-based_soft_actor-critic_with_n-step_returns.md)
- [\[ICLR 2026\] Flowing Through States: Neural ODE Regularization for Reinforcement Learning](flowing_through_states_neural_ode_regularization_for_reinforcement_learning.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)

</div>

<!-- RELATED:END -->
