---
title: >-
  [论文解读] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic
description: >-
  [NeurIPS 2025][强化学习][约束MDP] 提出Primal-Dual Natural Actor-Critic（PDNAC）算法，首次在一般参数化策略下的平均奖励约束MDP中实现 $\tilde{\mathcal{O}}(1/\sqrt{T})$ 的全局收敛率和约束违反率，匹配理论下界。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "约束MDP"
  - "平均奖励"
  - "原始-对偶"
  - "自然策略梯度"
  - "全局收敛"
---

# Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic

**会议**: NeurIPS 2025  
**arXiv**: [2505.15138](https://arxiv.org/abs/2505.15138)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 约束MDP, 平均奖励, 原始-对偶, 自然策略梯度, 全局收敛

## 一句话总结

提出Primal-Dual Natural Actor-Critic（PDNAC）算法，首次在一般参数化策略下的平均奖励约束MDP中实现 $\tilde{\mathcal{O}}(1/\sqrt{T})$ 的全局收敛率和约束违反率，匹配理论下界。

## 研究背景与动机

无限时域平均奖励设置对建模现实长期目标至关重要（如交通网络中的配送时间约束、通信网络中的资源预算限制）。约束马尔可夫决策过程（CMDP）通过引入代价函数来处理这些约束，要求在最大化平均奖励的同时确保平均代价不超过阈值。

**核心差距**：

| 设置 | 已有最优率 | 理论下界 |
|------|----------|---------|
| 表格型CMDP | $\tilde{\mathcal{O}}(1/\sqrt{T})$ ✓ | $\Omega(1/\sqrt{T})$ |
| 线性MDP CMDP | $\tilde{\mathcal{O}}(1/\sqrt{T})$ ✓ | $\Omega(1/\sqrt{T})$ |
| **一般参数化CMDP** | $\tilde{\mathcal{O}}(1/T^{1/5})$ ✗ | $\Omega(1/\sqrt{T})$ |

一般参数化（通过有限维参数索引策略，$d \ll |\mathcal{S}||\mathcal{A}|$）是处理大/无限状态空间的关键方法，但此前最优率仅为 $\tilde{\mathcal{O}}(1/T^{1/5})$，与下界差距巨大。

**根本挑战**：CMDP在一般参数化下缺乏强凸性，直接应用原始-对偶方法时，对偶问题的收敛无法自动转化为原始问题的收敛。对偶学习率 $\beta$ 的选择面临本质矛盾：$\beta$ 过低则约束违反收敛慢，$\beta$ 过高则原始更新方差大。

## 方法详解

### 整体框架

PDNAC算法通过求解鞍点优化来处理CMDP：
$$\max_{\theta \in \Theta} \min_{\lambda \geq 0} \mathcal{L}(\theta, \lambda) = J_r(\theta) + \lambda J_c(\theta)$$

算法采用嵌套循环结构：
- **外循环**（$K$ 轮）：更新原始参数 $\theta_k$ 和对偶参数 $\lambda_k$
- **内循环**（$H$ 步）：运行自然策略梯度（NPG）和最优Critic参数估计子程序

**关键参数设计**：为达到 $\mathcal{O}(1/\sqrt{T})$ 收敛，$H$ 应近似为常数而 $K$ 应为 $\tilde{\Theta}(T)$，使算法实质上接近单时间尺度。这与无约束MDP中 $K = H = \Theta(\sqrt{T})$ 的典型设置形成鲜明对比。

### 关键设计

#### Critic估计（基于MLMC）

Critic子程序需估计两个量：平均奖励/代价 $J_g(\theta_k)$ 和值函数 $V_g^{\pi_{\theta_k}}$。

**平均奖励估计**：将 $J_g(\theta_k)$ 表示为优化问题的解：
$$\min_{\eta \in \mathbb{R}} R_g(\theta_k, \eta) = \frac{1}{2}\sum_{s,a} \nu_g^{\pi_{\theta_k}}(s,a)\{\eta - g(s,a)\}^2$$

**值函数估计**：使用线性Critic函数 $\hat{V}_g(\zeta, s) = \langle \phi_g(s), \zeta \rangle$ 逼近 $V_g^{\pi_{\theta_k}}$。

**MLMC估计器**：关键创新在于使用Multi-Level Monte Carlo估计梯度。对于每个内循环步 $(k,h)$：
1. 采样几何分布 $Q_h^k \sim \text{Geom}(1/2)$
2. 按轨迹长度 $l_{kh} = 2^{Q_h^k}$ 收集样本
3. 构建MLMC梯度估计：$\mathbf{v}_g = \mathbf{v}_{g,kh}^0 + 2^{Q_h^k}(\mathbf{v}_{g,kh}^{Q_h^k} - \mathbf{v}_{g,kh}^{Q_h^k - 1})$

**MLMC的优势**：
- 达到与平均 $T_{\max}$ 个样本相同的偏差，但只需 $\tilde{\mathcal{O}}(\log T_{\max})$ 个样本
- 几何分布采样不需要混合时间知识，消除了先前工作的混合时间假设
- 无需保存长度为 $H$ 的轨迹，内存复杂度降低 $H$ 倍

#### NPG估计量

在获得Critic估计 $\xi_g^k = [\eta_g^k, (\zeta_g^k)^\top]^\top$ 后，通过 $H$ 步内循环估计自然策略梯度：
$$\omega_{g,h+1}^k = \omega_{g,h}^k - \gamma_\omega \hat{\nabla}_\omega f_g(\theta_k, \omega_{g,h}^k, \xi_g^k)$$

其中梯度估计使用TD误差作为优势函数估计：
$$\hat{A}_g = g(s,a) - \eta_g^k + \zeta_g^k(\phi_g(s') - \phi_g(s))$$

最终组合为 $\omega_k = \omega_r^k + \lambda_k \omega_c^k$。

### 损失函数 / 训练策略

**原始-对偶更新**：
$$\theta_{k+1} = \theta_k + \alpha \omega_k, \quad \lambda_{k+1} = \mathcal{P}_{[0, 2/\delta]}[\lambda_k - \beta \eta_c^k]$$

其中 $\alpha = T^{-1/2}$（原始学习率），$\beta = T^{-1/2}$（对偶学习率），$\delta$ 为Slater条件参数。

**收敛分析关键**（Lemma 4.6）：全局收敛率分解为：
$$\frac{1}{K}\sum_k (\mathcal{L}(\pi^*, \lambda_k) - \mathcal{L}(\theta_k, \lambda_k)) \leq \sqrt{\epsilon_{\text{bias}}} + \text{NPG偏差项} + \text{方差项} + \frac{1}{\alpha K}\text{KL散度}$$

需要精确控制NPG偏差和方差，这通过Theorem 4.7和4.8中的Critic/Actor收敛界实现。

## 实验关键数据

### 主实验（理论结果对比）

| 算法 | 全局收敛率 | 约束违反 | 混合时间未知 | Model-free | 设置 |
|------|----------|---------|------------|-----------|------|
| Chen et al. (Alg.1) | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | ✗ | ✗ | 表格型 |
| UC-CURL | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | 0 | ✓ | ✗ | 表格型 |
| Ghosh (Alg.3) | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | ✗ | — | 线性MDP |
| Bai et al. (SOTA) | $\tilde{\mathcal{O}}(1/T^{1/5})$ | $\tilde{\mathcal{O}}(1/T^{1/5})$ | ✗ | ✓ | 一般参数化 |
| **PDNAC (本文)** | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | ✗ | ✓ | **一般参数化** |
| **PDNAC (τ未知)** | $\tilde{\mathcal{O}}(1/T^{0.5-\epsilon})$ | $\tilde{\mathcal{O}}(1/T^{0.5-\epsilon})$ | ✓ | ✓ | **一般参数化** |
| 下界 | $\Omega(1/\sqrt{T})$ | — | — | — | — |

### 消融实验（两个定理的关系）

**Theorem 4.9**（已知混合时间）：
- $H = \tilde{\Theta}(\tau_{\text{mix}}^2)$, $K = T/H$
- 收敛率：$\mathcal{O}(\sqrt{\epsilon_{\text{bias}}} + \sqrt{\epsilon_{\text{app}}} + 1/\sqrt{T})$

**Theorem 4.10**（未知混合时间）：
- $H = T^\epsilon$, $K = T^{1-\epsilon}$
- 条件：$T \geq \tilde{\Theta}(\tau_{\text{mix}}^{2/\epsilon})$
- 收敛率：$\mathcal{O}(\sqrt{\epsilon_{\text{bias}}} + \sqrt{\epsilon_{\text{app}}} + 1/T^{0.5-\epsilon})$
- $\epsilon$ 越小越接近最优，但需更长时域

**NPG和Critic误差分析**（Theorem 4.7 & 4.8）：

| 误差项 | 上界组成 |
|--------|---------|
| Critic偏差 $\|\mathbb{E}[\xi_g^k] - \xi_g^*\|^2$ | $1/T^2 + \tau_{\text{mix}}^2/T_{\max}$ |
| Critic方差 $\mathbb{E}[\|\xi_g^k - \xi_g^*\|^2]$ | $1/T^2 + \tau_{\text{mix}}/H + \tau_{\text{mix}}/T_{\max}$ |
| NPG偏差 | Critic偏差 + $\epsilon_{\text{app}} + \tau_{\text{mix}}^2/T^2$ |
| NPG方差 | Critic方差 + $\epsilon_{\text{app}}$ |

### 关键发现

1. 一般参数化CMDP中首次匹配理论下界 $\Omega(1/\sqrt{T})$
2. 相比先前SOTA（$\tilde{\mathcal{O}}(1/T^{1/5})$）有巨大提升
3. $H$ 近似常数 + $K \approx T$ 的参数选择是关键——使算法接近单时间尺度
4. MLMC估计器同时解决了样本效率和混合时间未知两个问题
5. $\epsilon_{\text{bias}}$ 和 $\epsilon_{\text{app}}$ 决定了不可消除的近似残差

## 亮点与洞察

- **缩小约束与无约束的差距**：先前文献中CMDP率远差于无约束MDP（$1/T^{1/5}$ vs $1/T^{1/4}$），本文证明两者可以匹配
- **单时间尺度化**：通过将 $H$ 设为近似常数，将嵌套循环算法转化为近似单时间尺度，配合MLMC保持低偏差
- **MLMC的多重收益**：（a）低样本复杂度；（b）消除混合时间假设；（c）减少内存
- **对偶学习率的精细平衡**：$\alpha = \beta = T^{-1/2}$ 的对称选择配合参数调整解决了约束-优化的固有矛盾

## 局限与展望

1. 未知混合时间时需要 $T \geq \tilde{\Theta}(\tau_{\text{mix}}^{2/\epsilon})$，对混合慢的问题可能要求极长时域
2. $\epsilon_{\text{bias}}$ 和 $\epsilon_{\text{app}}$ 是不可消除的近似残差，但对丰富神经网络参数化可忽略不计
3. 假设遍历性（Ergodicity），排除了非遍历MDP
4. Critic使用线性函数逼近，未扩展到神经网络
5. 未提供实验验证，纯理论工作
6. 将结果扩展到多约束或时变约束是开放方向

## 相关工作与启发

- **Bai et al. (2024)**：先前一般参数化CMDP的SOTA，$\tilde{\mathcal{O}}(1/T^{1/5})$ 收敛率
- **Ganesh et al. (2024)**：加速平均奖励无约束MDP的Actor-Critic方法
- **Suttle et al. (2023) & Patel et al. (2024)**：在无约束平均奖励中使用MLMC消除混合时间假设
- **Wei et al. (2022)**（Triple-QA）：表格型设置下零约束违反但 $\tilde{\mathcal{O}}(1/T^{1/6})$ 收敛
- 启发：MLMC技术 + 原始-对偶框架的结合可能推广到更复杂的约束RL场景

## 评分

- **创新性**: ★★★★★ — 首次在一般参数化CMDP中匹配理论下界，突破性成果
- **实验充分性**: ★★☆☆☆ — 纯理论工作，无实验验证
- **实用价值**: ★★★☆☆ — 理论贡献卓越，但实际部署需进一步工程化
- **写作质量**: ★★★★☆ — 理论严谨，符号统一，但内容密度极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](../../ICML2025/reinforcement_learning/enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](../../ICLR2026/reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)

</div>

<!-- RELATED:END -->
