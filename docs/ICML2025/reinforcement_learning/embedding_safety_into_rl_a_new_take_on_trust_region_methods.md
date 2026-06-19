---
title: >-
  [论文解读] Embedding Safety into RL: A New Take on Trust Region Methods
description: >-
  [ICML 2025][强化学习][安全强化学习] 提出 C-TRPO 算法，通过修改策略空间的几何结构（在 KL 散度中嵌入约束感知的障碍项），使信赖域天然只包含安全策略，从而在训练全程保障约束满足，同时保持与 SOTA 相当的回报性能。 安全强化学习的核心框架是约束马尔可夫决策过程 (CMDP)：，要求在最大化累积奖励的…
tags:
  - "ICML 2025"
  - "强化学习"
  - "安全强化学习"
  - "信赖域方法"
  - "约束MDP"
  - "自然策略梯度"
  - "障碍函数"
---

# Embedding Safety into RL: A New Take on Trust Region Methods

**会议**: ICML 2025  
**arXiv**: [2411.02957](https://arxiv.org/abs/2411.02957)  
**代码**: [github.com/milosen/ctrpo](https://github.com/milosen/ctrpo)  
**领域**: 强化学习  
**关键词**: 安全强化学习, 信赖域方法, 约束MDP, 自然策略梯度, 障碍函数

## 一句话总结

提出 C-TRPO 算法，通过修改策略空间的几何结构（在 KL 散度中嵌入约束感知的障碍项），使信赖域天然只包含安全策略，从而在训练全程保障约束满足，同时保持与 SOTA 相当的回报性能。

## 研究背景与动机

安全强化学习的核心框架是**约束马尔可夫决策过程 (CMDP)**，要求在最大化累积奖励的同时满足代价约束 $V_{c_i}^\pi(\mu) \le b_i$。现有方法存在三大痛点：

**拉格朗日方法**（PPO-Lag、TRPO-Lag）：将约束转化为惩罚项，但对偶变量的更新不稳定，容易在约束边界振荡，导致训练过程中频繁违反约束。

**惩罚/障碍方法**（IPO、P3O）：引入固定权重的惩罚项简化优化，但会改变原始目标函数，产生**优化偏差**——最优解不再是原始 CMDP 的最优解。

**CPO 类信赖域方法**：在信赖域和安全策略集的交集中优化，理论上安全但实际依赖嘈杂的代价优势估计，导致在约束边界附近高频振荡和超调。

本文的核心观察：TRPO 使用的状态平均 KL 散度 $D_K$ 可以看作由负条件熵 $\Phi_K$ 诱导的 Bregman 散度。如果我们修改这个镜像函数 $\Phi$ 使其在约束边界处趋于无穷，那么任意有限半径的信赖域都不会碰到约束边界——从几何上根本消除了违反约束的可能性。

## 方法详解

### 整体框架

C-TRPO 的核心思路是**将约束信息嵌入策略空间的度量结构**，而非将约束作为额外的优化约束来处理。具体分三步：

1. **构造安全镜像函数 $\Phi_C$**：在标准条件熵基础上叠加关于代价约束的凸障碍项
2. **推导约束 KL 散度 $D_C$**：由安全镜像函数诱导的 Bregman 散度
3. **设计可行的近似算法**：用代理散度 $\bar{D}_C$ 替代精确散度，结合共轭梯度和回溯线搜索

整个算法流程：每一步先判断当前策略是否安全——若安全则执行约束信赖域更新（用 $\bar{D}_C$ 作为散度），若不安全则执行恢复步骤（仅最小化代价）。

### 关键设计

#### 1. 安全镜像函数

标准 TRPO 使用的镜像函数是负条件熵：

$$\Phi_K(d_\pi) = \sum_{s,a} d_\pi(s,a) \log \pi(a|s)$$

C-TRPO 将其扩展为：

$$\Phi_C(d) = \Phi_K(d) + \sum_{i=1}^{m} \beta_i \phi(b_i - c_i^\top d)$$

其中 $\phi: \mathbb{R}_{>0} \to \mathbb{R}$ 是满足 $\phi'(x) \to +\infty$ 当 $x \searrow 0$ 的凸函数。这意味着当占据度量 $d$ 接近约束边界 $b_i - c_i^\top d = 0$ 时，镜像函数的梯度趋于无穷。作者实验中采用 $\phi(x) = x\log(x)$（熵形式），效果优于对数障碍 $\phi(x) = -\log(x)$。

#### 2. 约束 KL 散度

由 $\Phi_C$ 诱导的 Bregman 散度为：

$$D_C(d_1 \| d_2) = D_K(d_1 \| d_2) + \sum_{i=1}^{m} \beta_i D_{\phi_i}(d_1 \| d_2)$$

其中 $D_{\phi_i}$ 是关于第 $i$ 个约束的 Bregman 散度项。**关键性质**：$D_C$ 仅在安全占据度量集 $\mathscr{D}_{\text{safe}}$ 内部有定义，当 $d_2$ 趋向约束边界时散度趋于无穷。因此对于任意有限 $\delta$，信赖域 $\{d : D_C(d_k \| d) \le \delta\}$ 必然包含在安全集内。

#### 3. 代理散度（实际可计算的版本）

精确的 $D_C$ 依赖于新策略的代价回报 $V_c(\pi)$，在更新前不可用。C-TRPO 用代理散度替代：

$$\bar{D}_C(\pi \| \pi_k) = \bar{D}_{KL}(\pi \| \pi_k) + \beta \bar{D}_\phi(\pi \| \pi_k)$$

其中 $\bar{D}_\phi$ 用**代价策略优势** $\mathbb{A}_c^{\pi_k}(\pi)$ 近似代价回报差 $V_c(\pi) - V_c(\pi_k)$。定义约束裕度 $\delta_b = b - V_c^{\pi_k}$，代理散度项为：

$$\bar{D}_\phi(\pi \| \pi_k) = \phi(\delta_b - \mathbb{A}_c^{\pi_k}(\pi)) - \phi(\delta_b) + \phi'(\delta_b) \mathbb{A}_c^{\pi_k}(\pi)$$

这个代理在策略参数 $\theta$ 处与精确散度的二阶展开一致，理论上足够准确。

#### 4. 与 CPO 的关系与区别

- **$\beta \to 0$ 时**：C-TRPO 退化为 CPO（约束进入信赖域约束中的原始线性约束形式）
- **$\beta > 0$ 时**：C-TRPO 比 CPO 更保守——更新方向被偏转为更平行于约束面而非直接穿越
- **$\beta \to +\infty$ 时**：在代价增大方向上的更新被完全抑制

与 CPO 相比，C-TRPO 的内层优化更简单：仅需近似一个二次约束，而 CPO 需要同时处理一个二次约束（信赖域）和一个线性约束（安全性）。

#### 5. 滞回恢复机制

当策略因估计误差离开安全集时，C-TRPO 执行恢复步骤。为避免在约束边界反复振荡，引入**滞回条件**：恢复的目标不是刚好回到 $V_c \le b$，而是回到更严格的 $V_c \le b_H$（其中 $b_H = 0.8b$），为后续更新留出安全裕度。

### 损失函数 / 训练策略

C-TRPO 的参数更新遵循与 TRPO 相同的框架：

$$\theta_{k+1} = \theta_k + \alpha^i \sqrt{\frac{2\delta}{g_k^\top H_k^{-1} g_k}} \cdot H_k^{-1} g_k$$

其中 $g_k = \nabla_\theta \mathbb{A}_r^{\pi_k}$ 是奖励优势梯度，$H_k = \nabla_\theta^2 \bar{D}_C(\pi_\theta \| \pi_{\theta_k})$ 是约束散度的 Hessian，用共轭梯度法近似 $H^{-1}g$。$\alpha^i$ 通过回溯线搜索确定，确保 $\bar{D}_C \le \delta$。

Hessian 的具体形式为：

$$\bar{H}_C(\theta_k) = G_K(\theta_k) + \beta \phi''(b - V_c(\theta)) \nabla_\theta V_c(\theta) \nabla_\theta V_c(\theta)^\top$$

即标准 Fisher 信息矩阵加上一个代价梯度的**秩一修正**，计算开销与 CPO 相当，相比 TRPO 仅多出代价值函数的估计。

**C-NPG（约束自然策略梯度）**是 C-TRPO 的连续时间极限：$\partial_t \theta_t = G_C(\theta_t)^+ \nabla V_r(\theta_t)$。理论分析表明 C-NPG 保证安全集不变性（Theorem 4.4）和全局收敛到最优安全策略（Theorem 4.5）。

## 实验关键数据

### 主实验

在 Safety Gymnasium 基准上与 9 种安全 RL 算法对比，4 个导航任务 + 4 个运动任务，每个任务 5 个种子运行 10M 步。

| 算法 | 最终代价（归一化，0=阈值） | 奖励（归一化，PPO=1） | 代价遗憾（归一化，CPO=1） | 收敛安全? |
|------|------|------|------|------|
| **C-TRPO** | **< 0（安全）** | **~0.85** | **~0.6** | **✓** |
| CPO | > 0（不安全） | ~0.9 | 1.0 | ✗ |
| PCPO | < 0（安全） | ~0.7 | ~0.5 | ✓ |
| PPO-Lag | < 0（安全） | ~0.8 | ~0.8 | ✓ |
| TRPO-Lag | < 0（安全） | ~0.8 | ~1.2 | ✓ |
| FOCOPS | > 0（不安全） | ~0.9 | ~0.9 | ✗ |
| CUP | > 0（不安全） | ~0.85 | ~0.7 | ✗ |
| P3O | < 0（安全） | ~0.6 | ~0.3 | ✓ |
| IPO | < 0（安全） | ~0.65 | ~0.7 | ✓ |

### 消融实验

| 配置 | 最终代价 | 奖励 | 说明 |
|------|------|------|------|
| C-TRPO（完整，β=1, b_H=0.8b） | 安全 | 高 | 默认配置 |
| 去掉 $\bar{D}_\phi$（仅 KL） | 不安全 | 高 | 退化为近似 CPO，约束违反增加 |
| 去掉滞回（b_H=b） | 边界振荡 | 高 | 恢复后立即又违反 |
| β=0.01 | 轻微不安全 | 高 | 接近 CPO 行为 |
| β=10 | 非常安全 | 降低 | 过度保守，噪声放大 |
| $\phi(x)=-\log(x)$（对数障碍） | 安全 | 略低 | 熵形式效果更好 |
| $\phi(x)=x\log(x)$（熵） | 安全 | 更高 | 默认选择 |

### 关键发现

1. **C-TRPO 在安全性和回报之间取得最佳折中**：在收敛时安全的算法中回报最高，在高回报算法中安全性最好
2. **代价遗憾显著低于 CPO**：C-TRPO 约为 CPO 的 60%，训练全程的约束违反大幅减少
3. **拉格朗日方法虽然最终可能安全，但训练过程中振荡严重**：TRPO-Lag 的代价遗憾是最高的
4. **P3O 代价遗憾最低但以牺牲回报为代价**：其最终代价远低于阈值，说明过于保守
5. **β=1 是鲁棒的默认选择**：在所有 8 个环境中无需调参，β≤1 几乎不影响回报

## 亮点与洞察

1. **几何视角的创新**：不是在优化层面添加约束，而是从策略空间的信息几何出发重新设计度量结构。让信赖域本身就是安全的，而非事后检查或投影
2. **理论与实践的统一**：C-NPG 的连续时间分析提供了安全集不变性和全局收敛的理论保证，C-TRPO 是其可计算的离散近似，二阶一致性建立了两者的桥梁
3. **与 CPO 的优雅联系**：通过 $\beta$ 参数平滑插值于 CPO（$\beta=0$）和完全保守更新（$\beta \to \infty$）之间，提供了一个连续的安全-性能控制旋钮
4. **实现极简**：相比 TRPO 仅多一个秩一矩阵修正项，相比 CPO 反而简化了内层优化

## 局限与展望

1. **散度估计的有限样本特性未分析**：代理散度依赖代价优势和值函数估计的准确性，在样本不足时可能导致约束违反
2. **仅处理平均代价约束**：CMDP 框架限制约束为轨迹平均代价，无法直接建模逐状态或逐轨迹的安全约束
3. **离散动作空间的扩展**：实验聚焦于连续控制，对离散动作空间的效果未充分验证
4. **超参数 $\beta$ 的自适应调节**：虽然 $\beta=1$ 在实验中表现鲁棒，但理论上最优的 $\beta$ 应随训练进展和约束裕度动态调整
5. **与模型基方法的结合**：可与 ActSafe 等方法结合，利用模型信息进一步减少估计噪声

## 相关工作与启发

- **CPO (Achiam et al., 2017)**：C-TRPO 的直接前身，将约束作为额外线性约束加入信赖域优化，本文证明这是 $\beta \to 0$ 的特例
- **PCPO (Yang et al., 2020)**：通过投影保障安全，但投影步骤限制了回报最大化
- **P3O (Zhang et al., 2022)**：惩罚方法中约束满足最好的，但存在优化偏差
- **控制障碍函数 (Ames et al., 2017)**：C-TRPO 的障碍项设计直接受控制论中 CBF 方法启发
- **信息几何与策略优化 (Neu et al., 2017; Müller & Montúfar, 2023)**：将策略优化理解为占据度量空间上的镜像下降，是本文理论框架的基础

## 评分

- 新颖性: ⭐⭐⭐⭐ - 从信息几何角度重新设计安全信赖域，思路优雅，但核心技术是已有工具的组合
- 实验充分度: ⭐⭐⭐⭐ - 8 个任务、9 个基线、5 种子、丰富消融，代价遗憾指标有新意，但缺少高维复杂任务
- 写作质量: ⭐⭐⭐⭐⭐ - 对 TRPO/CPO/C-TRPO 的几何直觉阐述清晰，理论与实验配合得当
- 价值: ⭐⭐⭐⭐ - 为安全 RL 提供了简洁实用的改进方向，实现开销极低，值得在更多场景中验证

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)
- [\[NeurIPS 2025\] Boundary-to-Region Supervision for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/boundary_to_region_supervision_for_offline_safe_rl.md)
- [\[ICML 2025\] Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics](safety_certificate_against_latent_variables_with_partially_unidentifiable_dynami.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)
- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)

</div>

<!-- RELATED:END -->
