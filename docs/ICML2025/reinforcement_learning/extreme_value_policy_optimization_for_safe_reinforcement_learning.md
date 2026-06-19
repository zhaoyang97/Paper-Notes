---
title: >-
  [论文解读] Extreme Value Policy Optimization for Safe Reinforcement Learning
description: >-
  [ICML2025][强化学习][安全RL] 提出 EVO 算法，将极值理论 (EVT) 引入约束强化学习，用广义 Pareto 分布 (GPD) 建模代价尾部的极端样本，并设计极端分位数约束与极端优先回放机制，在训练中实现零约束违反的同时保持竞争性策略性能。 约束强化学习 (CRL) 的目标是在满足预设约束条件下最大化累积…
tags:
  - "ICML2025"
  - "强化学习"
  - "安全RL"
  - "极值理论"
  - "约束满足"
  - "广义Pareto分布"
  - "经验回放优先级"
---

# Extreme Value Policy Optimization for Safe Reinforcement Learning

**会议**: ICML2025  
**arXiv**: [2601.12008](https://arxiv.org/abs/2601.12008)  
**代码**: [ShiqingGao/EVO](https://github.com/ShiqingGao/EVO)  
**领域**: 安全强化学习 / 约束强化学习  
**关键词**: 安全RL, 极值理论, 约束满足, 广义Pareto分布, 经验回放优先级

## 一句话总结

提出 EVO 算法，将极值理论 (EVT) 引入约束强化学习，用广义 Pareto 分布 (GPD) 建模代价尾部的极端样本，并设计极端分位数约束与极端优先回放机制，在训练中实现零约束违反的同时保持竞争性策略性能。

## 研究背景与动机

约束强化学习 (CRL) 的目标是在满足预设约束条件下最大化累积奖励。现有方法主要分为两类：

**基于期望的约束方法**（CPO、PID Lagrangian 等）：以累积代价的期望值作为约束，仅保证"平均"满足约束，忽略了代价分布的变异性，尤其是尾部的极端事件（黑天鹅事件），导致频繁的约束违反。

**基于概率的约束方法**（WCSAC、QCPO 等）：使用 CVaR 或分位数约束，但 WCSAC 用高斯近似代价分布，无法准确捕捉尾部衰减行为；QCPO 忽略了极端样本在训练中的关键影响。

核心问题：极端样本（低概率、高影响）在安全关键场景中至关重要，但它们天然稀缺、方差高，难以准确建模和有效利用。

## 方法详解

### 1. 极端分位数约束 (Extreme Quantile Constraint)

将累积代价 $C = \sum_{t=0}^{\infty} \gamma^t c$ 的分布分为主体 (body) 和尾部 (tail)：

- **安全边界** $q_\mu$：由累积代价期望确定，划分主体与尾部
- **风险边界** $q_{\mu+\nu}$：整合极端值的约束边界

利用 EVT 的 Pickands 定理，超过阈值 $q_\mu$ 的条件超额分布渐近服从 GPD：

$$q_{\mu+\nu} \simeq q_\mu + q^H_{\frac{\nu}{1-\mu}}$$

其中 $q^H_{\frac{\nu}{1-\mu}}$ 是 GPD 下的分位数。优化目标为：

$$\arg\max_{\pi \in \Pi} J_R(\pi) \quad \text{s.t.} \quad q_\mu + q^H_{\frac{\nu}{1-\mu}} \leq d$$

在信赖域内的代理优化目标：

$$\pi_{k+1} = \arg\max_{\pi} \mathbb{E}_{s \sim d^{\pi_k}, a \sim \pi}[A_R^{\pi_k}(s,a)]$$

$$\text{s.t.} \quad J_C(\pi_k) + \frac{1}{1-\gamma}\mathbb{E}[A_C^{\pi_k}(s,a)] + q^H_{\frac{\nu}{1-\mu}} \leq d, \quad D(\pi \| \pi_k) \leq \delta$$

### 2. GPD 参数估计

超过安全边界 $q_\mu$ 的样本（峰值集 $Y_\mu$）用于 MLE 拟合 GPD 的形状参数 $\xi$ 和尺度参数 $\sigma$：

$$\log \mathcal{L}(\xi, \sigma) = -N_\mu \log \sigma - (1 + \frac{1}{\xi})\sum_{i=1}^{N_\mu} \log(1 + \frac{\xi}{\sigma} Y_i)$$

风险边界计算：

$$q_\mu + q^H_{\frac{\nu}{1-\mu}} = q_\mu + \frac{\sigma}{\xi}\left((1 - \frac{\nu n}{N_\mu})^{-\xi} - 1\right)$$

### 3. 极端优先回放 (Extreme Prioritization)

分别对奖励和代价的极端样本建模 GPD，构建极端集合：

- 极端代价集 $Z_C: \{C > q_\mu + q^H_{\frac{\nu}{1-\mu}}\}$
- 极端奖励集 $Z_R: \{A_R > q^r_\mu + q^{H,r}_{\frac{\nu}{1-\mu}}\}$

优先级得分由 GPD 分位数水平决定：

$$p = \omega_r + \omega_c, \quad P(s_i) = \frac{p(s_i)}{\sum_{k=1}^N p(s_k)}$$

分位数水平越高（概率越低的极端样本）→ 回放优先级越高。

### 4. 离策略重要性重采样

为缓解极端样本稀缺导致的高方差，利用旧策略 $\pi_0$ 的存储样本进行重要性采样校正：

$$A'_R = \frac{\pi(a|s)}{\pi_0(a|s)} A_R, \quad C' = \frac{\pi(a|s)}{\pi_0(a|s)} C$$

扩充极端样本规模，提高 GPD 拟合稳定性。

## 实验关键数据

### 实验设置

- **环境**: Safety Gymnasium（导航避障）+ Safety MuJoCo（运动控制）
- **训练步数**: $10^7$ 步，最大轨迹长度 1000
- **随机种子**: 每种方法 6 个种子
- **基线**: CPO、WCSAC、Saute、Simmer
- **约束阈值**: 25

### 主要结果

| 维度 | EVO 表现 |
|------|---------|
| 约束满足 | 快速收敛到可行域后全程零约束违反 |
| 策略性能 | 与 CPO 可比，优于 Saute 和 WCSAC |
| 方差 | 低于分位数回归方法 (理论证明 + 实验验证) |
| 违反概率 | 低于基于期望方法一个 $\nu_0$ 的边际 |

### 消融实验

| 消融组件 | 影响 |
|---------|------|
| 移除 EVT 约束 → 常数分位数约束 | 策略性能下降，靠牺牲回报满足约束 |
| 移除极端优先回放 | 性能退化，未充分利用极端样本的学习信号 |
| 移除离策略重采样 | GPD 方差增大，尾部建模不准确 |

### 其他发现

- GPD 在多种分布形状下均优于高斯拟合（KS 检验值更低）
- 样本量仅 10-20 个 EVO 即可有效工作
- 不同代价阈值 (0/25/35) 下 EVO 均能自适应

## 亮点与洞察

1. **理论贡献扎实**: 提供了约束违反上界 (Theorem 4.1)、违反概率下界 (Theorem 4.2) 和方差下界 (Theorem 4.3) 三大理论保证
2. **EVT 与 CRL 结合新颖**: 首次将极值理论系统性地引入约束 RL，用 GPD 替代高斯近似建模尾部
3. **零违反保证**: 通过零违反开发范围 $\nu_0$，理论上保证更新后策略期望严格满足约束
4. **样本效率**: 即使极端样本稀缺（10-20个），EVO 仍然有效
5. **双重极端利用**: 同时建模奖励和代价两侧的极端值，兼顾性能与安全

## 局限与展望

1. **GPD 适用性**: 当极端值与正常值差异较小时，GPD 拟合质量下降（需要阈值 $t$ 足够大），论文建议用非线性变换放大差异但未实验验证
2. **IID 假设**: EVT 要求样本独立同分布，但 RL 中相邻时间步的样本天然有相关性
3. **环境复杂度**: 仅在 Safety Gymnasium 和 MuJoCo 上验证，未测试更复杂的现实场景（如真实机器人、自动驾驶）
4. **多约束扩展**: 当前仅处理单一约束，多约束场景下如何分别建模各约束的 GPD 有待讨论
5. **计算开销**: GPD 拟合 + 重要性重采样 + 优先回放增加了额外计算成本，论文未报告时间开销

## 相关工作与启发

- **CPO** (Achiam et al., 2017): 经典的基于期望约束的信赖域方法，EVO 在其基础上引入 EVT 尾部约束
- **WCSAC** (Yang et al., 2021): 用高斯近似 + CVaR 的概率约束，EVO 用 GPD 替代高斯实现更精确的尾部建模
- **QCPO** (Jung et al., 2022): 分位数约束方法，EVO 在理论上证明了比分位数回归更低的方差
- **EVAC** (NS et al., 2023): 将 EVT 用于降低极端回报的方差，但未处理约束满足问题

## 评分

- 新颖性: ⭐⭐⭐⭐ (EVT + CRL 的结合具有新意，GPD 建模尾部直觉清晰)
- 实验充分度: ⭐⭐⭐⭐ (消融完整、敏感性分析、GPD 拟合验证、样本量实验均覆盖)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论推导完整，图表直观)
- 价值: ⭐⭐⭐⭐ (安全RL中尾部风险是核心问题，方法实用且有理论保证)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning](scaling_value_iteration_networks_to_5000_layers_for_extreme_long-term_planning.md)
- [\[ICML 2025\] Wasserstein Policy Optimization](wasserstein_policy_optimization.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[ICML 2026\] CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning](../../ICML2026/reinforcement_learning/cspo_constraint-sensitive_policy_optimization_for_safe_reinforcement_learning.md)
- [\[ICML 2026\] Safe In-Context Reinforcement Learning](../../ICML2026/reinforcement_learning/safe_in-context_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
