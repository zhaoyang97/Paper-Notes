---
title: >-
  [论文解读] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning
description: >-
  [AAAI2026][强化学习][离策略强化学习] 提出 Behaviour Policy Optimization (BPO)，通过优化一个专用行为策略来采集离策略数据，使得回报估计的方差可证明低于在策略采集，从而提升 REINFORCE 和 PPO 的样本效率与稳定性。 - 策略梯度方法（如 REINFORCE、PPO）…
tags:
  - "AAAI2026"
  - "强化学习"
  - "离策略强化学习"
  - "variance reduction"
  - "重要性采样"
  - "behaviour policy"
  - "策略梯度"
---

# Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning

**会议**: AAAI2026  
**arXiv**: [2511.10843](https://arxiv.org/abs/2511.10843)  
**代码**: [sacktock/BPO](https://github.com/sacktock/BPO)  
**领域**: 强化学习  
**关键词**: 离策略强化学习, variance reduction, 重要性采样, behaviour policy, 策略梯度

## 一句话总结

提出 Behaviour Policy Optimization (BPO)，通过优化一个专用行为策略来采集离策略数据，使得回报估计的方差可证明低于在策略采集，从而提升 REINFORCE 和 PPO 的样本效率与稳定性。

## 背景与动机

- 策略梯度方法（如 REINFORCE、PPO）依赖回报估计进行策略更新，高方差的回报估计会导致梯度震荡、学习不稳定以及样本效率低下
- 传统 off-policy 方法（如 IMPALA、ACER）主要解决多个并行 worker 与异步策略更新之间的数据不匹配问题，通过截断重要性权重来控制方差
- 离策略评估（off-policy evaluation）领域的最新成果（ODI, Liu et al. 2024）揭示了一个反直觉的结论：**精心设计的行为策略采集的数据，其回报估计方差可证明低于在策略采集**
- 然而 ODI 仅考虑无折扣有限视野的策略评估场景，直接用于在线 RL 的策略改进面临非平稳目标策略、不适合用完整 Monte Carlo 回报等挑战

## 核心问题

如何将离策略评估中"设计最优行为策略以降低回报估计方差"的思想，扩展到在线强化学习场景中，使策略评估和策略改进同时受益于低方差估计？

## 方法详解

### 核心思想

不再按照当前目标策略 $\pi$ 在策略上采集数据，而是维护一个专门优化过的行为策略 $\mu$，使其采集的离策略数据经重要性加权后得到的回报估计方差更低。

### 方差最优行为策略

基于 ODI 的理论结果，一步最优行为策略 $\hat{\mu}$ 的形式为：

$$\hat{\mu}(a|s) \propto \pi(a|s) \sqrt{\hat{q}_\pi(s,a)}$$

其中 $\hat{q}_\pi(s,a)$ 包含了 $q_\pi^2$、下一状态值函数方差以及后续 PDIS 方差等项，直觉上该策略在 $|q_\pi|$ 较大的动作上分配更多概率，从而降低重要性权重的方差。

### 截断重要性加权 TD(λ) 回报

论文提出了新的回报估计器 $G_t^{\text{TIS},\lambda}$，结合 TD(λ) 和截断重要性权重：

$$G_t^{\text{TIS},\lambda} = v_\pi(S_t) + \sum_{k=t}^{\infty} (\gamma\lambda)^{k-t} \left(\prod_{i=t}^{k-1} c_i\right) \delta_k$$

其中 $c_t = \min(\bar{c}, \pi(A_t|S_t)/\mu(A_t|S_t))$ 为截断 trace 系数，$\delta_k$ 为 TD 误差。该估计器可通过递推高效计算（类似 eligibility trace），并具有以下理论保证：

- **无偏性（Theorem 1）**：当 $\bar{c}, \bar{\rho} = \infty$ 时，对任意 $\mu \in \Lambda$，$\mathbb{E}_\mu[G_t^{\text{TIS},\lambda}|S_t=s] = v_\pi(s)$
- **方差缩减（Theorem 2）**：当 $\lambda=1, \bar{c}, \bar{\rho}=\infty$ 时，使用 $\hat{\mu}$ 采集数据的方差严格不高于使用 $\pi$ 在策略采集

### 学习 $\hat{q}_\pi$

$\hat{q}_\pi$ 可表示为以修改后的奖励 $\hat{r}_\pi(s,a) = 2r(s,a)q_\pi(s,a) - r^2(s,a)$ 和折扣因子 $\gamma^2$ 定义的另一个 Q 函数。因此只需额外训练一个 Q 网络即可。

### 算法实现

BPO 在基础策略梯度算法之上增加三个辅助模块：

1. **两个 Q 网络**：通过 Fitted Q-Evaluation (FQE) 分别估计 $q_\pi$ 和 $\hat{q}_\pi$，使用 symlog 目标变换稳定训练
2. **行为策略优化**：离散动作空间用交叉熵损失匹配目标分布；连续动作空间用对数概率距离损失
3. **离策略数据采集**：用优化过的 $\mu$ 采集 rollout，用截断 IS TD(λ) 回报计算优势估计

### 与 PPO 的集成

- 将 PPO 的 clipped surrogate 目标中的比率改为 $r_t(\theta) = \pi_\theta(A_t|S_t) / \mu(A_t|S_t)$
- 优势估计改为 $\hat{A}_t = \hat{G}_t^{\text{TIS},\lambda} - V_\omega(S_t)$
- 值函数用截断 IS TD(λ) 回报作为回归目标
- 策略和值函数的更新方式与标准 PPO 一致，仅数据来源和回报估计方式发生变化

## 实验关键数据

### REINFORCE + ShortCorridor

- BPO 收敛更快且在训练后期更稳定，虽然在这个简单环境上优势较小
- 最佳截断参数：$\bar{c}=1.0, \bar{\rho}=1.5$

### PPO + MuJoCo

在 Ant-v5、HalfCheetah-v5、Hopper-v5、Walker2d-v5 上的结果（10 次独立运行）：

| 环境 | PPO | BPO (最佳配置) | 提升 |
|------|-----|----------------|------|
| Ant-v5 (gSDE) | 1106±111 | **1690±125** | +53% |
| HalfCheetah-v5 (gSDE) | 3425±468 | 3742±408 | +9% |
| Hopper-v5 | 3527±670 | **4749±419** | +35% |
| Walker2d-v5 | 2126±492 | **2770±296** | +30% |
| Walker2d-v5 (default) | 2091±408 | **3044±332** | +46% |

- 几乎所有 BPO 配置都优于 baseline PPO，大多数改进具有统计显著性
- BPO 在训练初期收敛更快，后期更稳定
- 轨迹级截断（traj $\bar{c}=1.0$）在 Walker2d 上表现最优

## 亮点

1. **理论扎实**：不仅给出了方差缩减的直觉，还提供了无偏性和方差缩减的严格证明，将 off-policy evaluation 的理论优雅地扩展到在线 RL
2. **反直觉洞察**：证明了在策略采集（on-policy）并非方差最优，精心设计的 off-policy 采集可以更好
3. **实用的估计器设计**：截断 IS TD(λ) 回报兼顾了理论上的无偏性保证和实际中的方差控制需求
4. **即插即用**：BPO 作为附加模块可以叠加到现有策略梯度算法上，不修改原算法的核心更新逻辑
5. **连续动作支持**：为连续动作空间设计了替代损失函数并给出理论证明（Theorem 4）

## 局限与展望

1. **计算开销增大**：需要额外训练两个 Q 网络和一个行为策略网络，增加了计算和超参数调优复杂度
2. **超参数敏感**：截断参数 $\bar{c}, \bar{\rho}$ 的选择对性能影响较大，不同环境的最优配置不同
3. **实验规模有限**：仅在 MuJoCo 连续控制任务上测试 PPO，未涉及 Atari 等离散高维任务或更复杂的环境
4. **Q 网络估计误差传播**：FQE 的过估计或近似误差会影响行为策略质量，虽然用了 symlog 目标缓解但未完全解决
5. **未扩展到 actor-critic 家族**：论文仅实验了 REINFORCE 和 PPO，尚未验证对 A2C、SAC、TD3 等方法的效果

## 与相关工作的对比

| 方法 | 核心思路 | 与 BPO 的区别 |
|------|----------|--------------|
| IMPALA (V-trace) | 截断 IS 权重修正多 worker 异步数据 | BPO 关注单 worker 的最优行为策略设计，正交方向 |
| ACER | 截断 IS + experience replay | 未主动优化行为策略，仅被动截断权重 |
| GAE | 多步回报的偏差-方差权衡 | BPO 从采样分布层面降低方差，与 GAE 互补 |
| ODI (Liu et al.) | 设计最优行为策略用于离策略评估 | BPO 将其扩展到在线 RL，引入 TD(λ) 估计器并处理非平稳性 |
| Retrace(λ) | 截断 IS 权重至 1 的 TD(λ) | BPO 的截断通常更温和，且行为策略天然使 IS 权重良性 |
| ROS | 重加权行为策略覆盖欠代表状态 | 假设完整轨迹和已知策略，不用 IS 修正 |

## 启发与关联

- 该方法的核心洞察——**采集数据的策略可以与训练的目标策略不同且更优**——对 RLHF/RLAIF 等 LLM 对齐场景也有潜在价值，可以设计更高效的探索策略
- 与 curriculum learning 的思想相呼应：不是均匀采样训练数据，而是自适应地选择对学习最有帮助的数据分布
- BPO 的框架理论上可以扩展到 model-based RL，利用学到的环境模型设计全局最优行为策略

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 off-policy evaluation 的方差最优采样理论首次应用于在线 RL，思路新颖
- 实验充分度: ⭐⭐⭐ — MuJoCo 结果较好但环境覆盖不全面，缺少离散动作高维任务
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰完整，从直觉到证明层层递进
- 价值: ⭐⭐⭐⭐ — 提供了一个通用的方差缩减框架，对策略梯度方法有实际改进价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)
- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[ICML 2025\] Wasserstein Policy Optimization](../../ICML2025/reinforcement_learning/wasserstein_policy_optimization.md)
- [\[ICML 2025\] Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning](../../ICML2025/reinforcement_learning/log-sum-exponential_estimator_for_off-policy_evaluation_and_learning.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)

</div>

<!-- RELATED:END -->
