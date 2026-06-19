---
title: >-
  [论文解读] Risk-Sensitive Exponential Actor Critic
description: >-
  [AAAI2026][强化学习][risk-sensitive RL] 针对 entropic risk measure 下 policy gradient 的高方差和数值不稳定问题，推导了完整的 on/off-policy 风险敏感策略梯度定理，并提出 rsEAC 算法，通过 log-domain critic 参数化和梯度归一化裁剪机制实现稳定的风险敏感连续控制。
tags:
  - "AAAI2026"
  - "强化学习"
  - "risk-sensitive RL"
  - "entropic risk measure"
  - "policy gradient"
  - "actor-critic"
  - "numerical stability"
---

# Risk-Sensitive Exponential Actor Critic

**会议**: AAAI2026  
**arXiv**: [2602.07202](https://arxiv.org/abs/2602.07202)  
**领域**: 强化学习  
**关键词**: risk-sensitive RL, entropic risk measure, policy gradient, actor-critic, numerical stability

## 一句话总结
针对 entropic risk measure 下 policy gradient 的高方差和数值不稳定问题，推导了完整的 on/off-policy 风险敏感策略梯度定理，并提出 rsEAC 算法，通过 log-domain critic 参数化和梯度归一化裁剪机制实现稳定的风险敏感连续控制。

## 研究背景与动机
标准 RL 优化期望回报，但在自动驾驶、机器人、金融等场景中需要风险感知的决策。Entropic risk measure 是常用的风险度量：

$$J^\beta(\pi_\theta) = \frac{1}{\beta} \log \mathbb{E}_{p_\pi(\tau)} \left[ e^{\beta \sum_t r_t} \right]$$

- $\beta > 0$：risk-seeking（寻险）
- $\beta < 0$：risk-averse（避险）
- $\beta \to 0$：退化为标准 risk-neutral 目标

**现有方法的核心困难**：
- likelihood ratio trick 估计梯度需要完整轨迹，且被 exponentiated return 缩放，导致高方差和数值不稳定
- 指数值函数 $Z^\beta(s,a) = e^{\beta Q(s,a)}$ 在函数逼近中极易溢出/下溢
- 现有 model-free 方法（如 R-AC）仅能处理简单任务和 tabular 设定

## 方法详解

### 风险敏感策略梯度定理
**Theorem 1（随机策略）**：

$$\nabla_\theta J^\beta = \frac{1}{\beta} \int_S \rho_\pi^*(s) \int_A \nabla_\theta \pi_\theta(a|s) \cdot e^{\beta(Q^\beta(s,a) - V^\beta(s))} \, da \, ds$$

其中 $\rho_\pi^*$ 为 exponential twisted dynamics 下的状态分布。关键区别：用指数化优势替代 Q 值，引入数值风险。

**Theorem 2（确定性策略）**：

$$\nabla_\theta J^\beta = \int_S \rho_\mu^*(s) \nabla_\theta \mu_\theta(s) \nabla_a Q^\beta_{\mu_\theta}(s,a) \big|_{a=\mu_\theta(s)} \, ds$$

避免了指数项和动作积分，更适合实际使用。并证明 off-policy 近似下的 deterministic policy improvement（Theorem 3）。

### Log-domain Critic 参数化
直接学习 $Z_\psi(s,a) = e^{\beta Q(s,a)}$ 会导致数值溢出/下溢。关键改进：

- 参数化为 $Z_\psi(s,a) = e^{Q_\psi(s,a)}$，其中 $Q_\psi$ 为神经网络
- $\frac{1}{\beta} Q_\psi(s,a)$ 近似 soft-value function
- 梯度在 log-domain 计算，天然更稳定

### 梯度稳定化机制
Exponential TD loss 的梯度形如 $e^x(e^x - e^y)$，需要稳定化：

1. **Helper function**：$f(x,y) = (1-e^{y-x})$ 或 $(e^{x-y}-1)$，值域 $[-1,1]$，保证稳定
2. **Batch normalization**：对指数前导项减去 batch 内的均值 $z$ 进行归一化
3. **Gradient clipping**：限制指数参数范围，防止梯度爆炸/消失

### rsEAC 算法
基于 TD3 框架构建：
- **Twin critics**：两个 $Q_\psi$ 网络，$\beta > 0$ 时取 min，$\beta < 0$ 时取 max（控制过估计方向）
- **Actor**：确定性策略 $\mu_\theta$，按 off-policy deterministic gradient 更新
- **Exploration**：添加高斯噪声

## 实验关键数据

### GridWorld（Tabular）
验证 $\beta$ 的风险调制作用：
- $\beta = -1$：避险策略绕过悬崖区域走远路
- $\beta = 1$：寻险策略紧贴悬崖走最短路
- $|\beta|$ 较大时指数值函数数值爆炸（验证了稳定化的必要性）

### Inverted Pendulum
- rsEAC 在 $\beta=1$ 和 $\beta=-1$ 下均能学到高回报策略
- R-AC 因数值不稳定导致策略质量差且缺乏风险敏感性

### MuJoCo 风险变体（Swimmer / HalfCheetah / Ant）
加入随机噪声风险区域（$\mathcal{N}(0,10^2)$ 或 $\mathcal{N}(0,7^2)$）：

| 方法 | 风险特征 | 平均回报表现 |
|------|---------|------------|
| rsEAC | 低风险区域访问率 | 与 MVPI 相当 |
| R-AC | 数值不稳定 | 所有任务最差 |
| MVPI | 低风险访问率 | Ant 较差 |
| MG (PPO) | 中等避险 | 中等 |

- rsEAC 在所有任务上均优于 R-AC，验证稳定化机制的关键作用
- 在高维 Ant 任务上优于 MVPI

### 稳定性对比（CartPole）
- $Z_\psi$ 直接学习：值函数估计溢出/下溢，4 种 $\beta$ 仅 1 种学到最优策略
- $Q_\psi$（本文）+ 梯度归一化裁剪：所有 $\beta$ 设定均学到最优策略

## 亮点
- **完整理论框架**：首次为 entropic risk measure 推导 on/off-policy × stochastic/deterministic 的四种策略梯度定理
- **实用稳定化方案**：log-domain 参数化 + batch normalization + clipping，解决了指数值函数学习的长期痛点
- **风险可调**：单参数 $\beta$ 控制 risk-seeking/averse 程度
- **首个可处理复杂连续任务的 model-free 风险敏感 actor-critic**

## 局限性
- 指数函数的固有不稳定性无法完全消除，极端 $\beta$ 下仍可能失效
- 风险参数 $\beta$ 需要针对不同任务调优，缺乏自适应机制
- 仅在 MuJoCo 连续控制任务上验证，未测试更多安全关键场景

## 评分
- 新颖性: ⭐⭐⭐⭐ — 策略梯度定理推导严谨，稳定化方案有明确的工程贡献
- 实验充分度: ⭐⭐⭐ — tabular + 连续控制覆盖合理，但任务多样性偏窄
- 写作质量: ⭐⭐⭐⭐ — 理论呈现清晰，数值问题可视化直观
- 价值: ⭐⭐⭐⭐ — 为风险敏感 RL 的实际应用铺平道路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](../../ICLR2026/reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](../../ICML2025/reinforcement_learning/enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](../../NeurIPS2025/reinforcement_learning/global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[ICML 2026\] Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access](../../ICML2026/reinforcement_learning/informed_asymmetric_actor-critic_leveraging_privileged_signals_beyond_full-state.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)

</div>

<!-- RELATED:END -->
