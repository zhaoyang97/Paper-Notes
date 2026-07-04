---
title: >-
  [论文解读] Adaptive Cooperative Transmission Design for URLLC via Deep RL
description: >-
  [NeurIPS 2025][强化学习][URLLC] 提出 DRL-CoLA 算法，用双 Agent DQN 分别在源节点和中继节点上自适应配置 5G NR 传输参数（numerology、mini-slot、MCS），在两跳中继系统中仅用本地 CSI 即可达到接近全局 CSI 最优的 URLLC 可靠性。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "URLLC"
  - "cooperative transmission"
  - "5G NR"
  - "dual-agent DQN"
---

# Adaptive Cooperative Transmission Design for URLLC via Deep RL

**会议**: NeurIPS 2025  
**arXiv**: [2511.02216](https://arxiv.org/abs/2511.02216)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: URLLC, cooperative transmission, deep reinforcement learning, 5G NR, dual-agent DQN  

## 一句话总结
提出 DRL-CoLA 算法，用双 Agent DQN 分别在源节点和中继节点上自适应配置 5G NR 传输参数（numerology、mini-slot、MCS），在两跳中继系统中仅用本地 CSI 即可达到接近全局 CSI 最优的 URLLC 可靠性。

## 研究背景与动机

1. **领域现状**：下一代无线通信需支持远程手术、自动驾驶等任务关键应用，要求误包率低至 $10^{-5}$~$10^{-7}$、端到端延迟毫秒级。协作中继通信（两跳传输）是提升可靠性的重要手段。
2. **现有痛点**：
    - 现有两跳传输方案多为一次性传输(one-shot)，任一跳解码失败即丢包，且假设双跳全局 CSI 已知——获取全局 CSI 的开销在 URLLC 时延预算下不可接受；
    - ARQ 重传协议可提升可靠性但增加延迟；5G NR 的 AMC、可伸缩 numerology、mini-slot 等特性此前只被单独优化（只优化 AMC 或只优化 numerology），未联合利用；
    - 没有现有工作考虑 ARQ 重传对两跳中继系统在时延约束下可靠性的影响。
3. **核心矛盾**：两跳传输中，总端到端延迟 $\mathcal{T}$ 是随机变量（取决于信道衰落和重传次数），其分布难以解析建模，传统优化方法无法处理 $\mathcal{T} \le T_{\text{th}}$ 这一约束。
4. **本文目标**
    - 在每次（重）传输尝试中优化两跳各自的 numerology $\mu$、mini-slot 大小 $N_{\text{sym}}$、MCS $I_{\text{MCS}}$
    - 最大化端到端成功递送概率，同时满足严格延迟约束
    - 仅依赖本地 CSI 和 ARQ 反馈，无需全局 CSI
5. **切入角度**：将两跳自适应传输建模为 MDP，源节点和中继节点作为两个独立 Agent，各自学习时延感知的传输策略。
6. **核心 idea**：双 Agent DQN 分布式学习逐跳传输参数配置策略，用 DOR（delay outage rate）作为跨跳协调信号，在无全局 CSI 下实现 URLLC。

## 方法详解

### 整体框架

系统为 S → R → D 两跳半双工中继：
- **输入**：每个 Agent 观测本地 SNR $\gamma_i$、下一跳平均 SNR $\bar{\gamma}_{i+1}$、包大小 $H$、剩余时延预算 $\tau_n$
- **输出**：传输参数三元组 $(μ, N_{\text{sym}}, I_{\text{MCS}})$
- **流程**：S 先传，R 解码成功后转发；任一跳解码失败触发 ARQ 重传；重复直到 D 收到包或时延预算耗尽

### 关键设计

1. **MDP 建模**：
    - 做什么：将两跳传输的逐次参数选择建模为 MDP
    - 核心思路：状态 $s_n^{(i)} = (\gamma_i, \bar{\gamma}_{i+1}, H, \tau_n)$ 是 4 维的；动作空间 $\mathcal{A} = \{(\mu, N_{\text{sym}}, I_{\text{MCS}})\}$ 共 $5 \times 4 \times 15 = 300$ 个离散动作；状态转移由解码错误率 $\varepsilon_i$ 和剩余预算决定，有 Success 和 Failure 两个吸收终止状态
    - 设计动机：MDP 天然适合序贯决策问题，且通过 RL 可避免对 $\mathcal{T}$ 分布的解析建模

2. **DOR 奖励设计**：
    - 做什么：用 delay outage rate 作为跨跳协调信号
    - 核心思路：S 的奖励不仅考虑本跳成功，还需估计下一跳在剩余预算 $\tau_{n+1}$ 内成功的概率。DOR 定义为 $\mathcal{P}_{\text{DOR}}(\bar{\gamma}_i, \tau) = 1 - \exp(-\frac{1}{\bar{\gamma}_i}(2^{H/(W\tau)} - 1))$。成功时奖励 $1 - \mathcal{P}_{\text{DOR}}$，失败 $-1$，重传 $-0.1$
    - 设计动机：S 无法直接观测第二跳结果，DOR 利用下一跳平均 SNR 和剩余预算间接估计成功概率，实现分布式协调

3. **双 Agent DQN 架构**：
    - 做什么：S 和 R 各自维护一个 DQN 网络，独立学习
    - 核心思路：每个 Agent 用 $\epsilon$-greedy 探索，experience replay 训练，target network 稳定。解码错误概率用有限块长(finite blocklength)公式计算：$\varepsilon_i = Q(\ln 2 \sqrt{m_i/V_i} (\log_2(1+\gamma_i) - H/m_i))$
    - 设计动机：DQN 适合离散动作空间（300 个动作），且双 Agent 解耦设计避免了全局 CSI 需求，只需 ARQ 反馈即可协调

### 损失函数 / 训练策略

- 标准 DQN MSE 损失：$\mathcal{L}_i(\theta_i) = \mathbb{E}[(y_n^{(i)} - Q_i(s_n^{(i)}, a_n^{(i)}; \theta_i))^2]$
- Target value: $y_n^{(i)} = \mathcal{R}_{n+1}^{(i)} + \gamma \max_{a'} Q_i(s_{n+1}^{(i)}, a'; \theta_i^-)$
- Target network 每 $E'$ episode 同步一次
- 重传次数由策略隐式优化（小负奖励鼓励减少不必要重传）

## 实验关键数据

### 主实验——端到端可靠性

| 配置 | 丢包率 | 对比 |
|------|--------|------|
| DRL-CoLA (本地 CSI) | 接近最优 | 与全局 CSI one-shot 几乎重合 |
| One-shot (全局 CSI) | 理论下界 | 需要完美全局 CSI |

中继位置 $d_1 + d_2 = 1000$m 时，丢包率曲线呈 V 形，$d_1 = d_2$ 对称放置时最低。$d_1 > d_2$ 时比 $d_1 < d_2$ 略好，因为第二跳延迟预算更紧张，R 更接近 D 有利于在有限时间内完成传输。

### 消融实验——RL 算法选择

| RL 算法 | 收敛速度 | 累积奖励 | 说明 |
|---------|---------|---------|------|
| DQN | 最快 | 最高 | 离散动作空间的最佳选择 |
| A2C | 较慢 | 较低 | Policy-gradient 在此场景效率不如 value-based |
| PPO | 中等 | 中等 | 同上 |

### 关键发现

- **无全局 CSI 也能接近最优**：DRL-CoLA 仅用本地 CSI + ARQ 反馈就达到与全局 CSI one-shot 方案几乎相同的丢包率，证明分布式学习方案的有效性。
- **DQN 优于 A2C/PPO**：在 300 个离散动作的场景下，value-based DQN 的收敛速度和最终性能都优于 policy-gradient 方法。
- **DOR 奖励的协调作用**：DOR 让 S 在选择传输参数时"考虑"第二跳的成功概率，避免 S 耗尽太多延迟预算导致 R 无法完成传输。

## 亮点与洞察

- **DOR 作为跨 Agent 协调信号**：利用 delay outage rate 巧妙解决了两个独立 Agent 之间的隐式协调问题——S 不需要知道 R 的具体决策，只需通过 DOR 估计留给 R 的时间是否够用。这一思路可推广到多跳网络。
- **5G NR 特性联合优化**：首次在两跳中继系统中联合优化 numerology + mini-slot + MCS 三个维度，之前工作只单独优化其中一个。
- **有限块长失效概率建模**：在 URLLC 短包场景下用 Polyanskiy 有限块长公式替代 Shannon 容量假设，更切合实际。
- **实用的分布式架构**：双 Agent 各自维护独立 DQN，部署简单，无需集中式训练或通信开销。

## 局限与展望

- **仅 Rayleigh 衰落**：信道假设简单（单径 Rayleigh），未考虑多径、莱斯衰落或频率选择性衰落。
- **完美 ARQ 假设**：假设 ARQ 请求总能成功接收，实际中 ARQ 本身也可能出错。
- **双跳限制**：仅考虑两跳，多跳场景的可扩展性未验证。
- **静态信道模型**：假设信道在整个时延预算内不变（准静态），在高速移动场景下不成立。
- **改进方向**：
    - 扩展到多跳多中继场景，用多 Agent RL 框架
    - 加入信道估计误差和不完美 ARQ
    - 考虑多用户多载波的资源分配联合优化

## 相关工作与启发

- **vs Saatchi et al. (2023)**：他们在单跳点对点场景联合优化 numerology/mini-slot/MCS，本文扩展到两跳中继并引入 ARQ 重传。
- **vs 传统 one-shot 方案**：one-shot 需全局 CSI 且无重传机会，本文方案更实用但性能仍接近。
- **vs 多 Agent RL (MARL)**：本文的双 Agent 是独立学习+隐式协调（通过 DOR），未用 CTDE 等 MARL 框架，在两 Agent 场景下已足够有效。

## 评分

- 新颖性: ⭐⭐⭐⭐ DRL 用于通信参数优化已有先例，DOR 奖励设计是亮点
- 实验充分度: ⭐⭐⭐⭐ 与 one-shot 和不同 RL 算法的对比充分，但缺少与其他 MARL 方案的对比
- 写作质量: ⭐⭐⭐⭐⭐ 问题建模清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 对 5G URLLC 中继系统有实际应用价值，学术新颖性一般

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [\[AAAI 2026\] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](../../AAAI2026/reinforcement_learning/chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)

</div>

<!-- RELATED:END -->
