---
title: >-
  [论文解读] Time-Aware World Model for Adaptive Prediction and Control
description: >-
  [ICML 2025][world model] 提出时间感知世界模型 TAWM，通过将时间步长 $\Delta t$ 作为显式输入条件并在训练中混合多种 $\Delta t$ 采样，使模型能以单步预测适应任意时间分辨率的推理，且不增加训练样本量。 领域现状：基于模型的强化学习（MBRL）通过学习环境动态模型（世界模型）来提…
tags:
  - "ICML 2025"
  - "world model"
  - "model-based RL"
  - "time-step conditioning"
  - "Nyquist-Shannon theorem"
  - "multi-scale dynamics"
---

# Time-Aware World Model for Adaptive Prediction and Control

---

**会议**: ICML 2025  
**arXiv**: [2506.08441](https://arxiv.org/abs/2506.08441)  
**代码**: [https://github.com/anh-nn01/Time-Aware-World-Model](https://github.com/anh-nn01/Time-Aware-World-Model)  
**领域**: 其他  
**关键词**: world model, model-based RL, time-step conditioning, Nyquist-Shannon theorem, multi-scale dynamics

## 一句话总结

提出时间感知世界模型 TAWM，通过将时间步长 $\Delta t$ 作为显式输入条件并在训练中混合多种 $\Delta t$ 采样，使模型能以单步预测适应任意时间分辨率的推理，且不增加训练样本量。

---

## 研究背景与动机

**领域现状**：基于模型的强化学习（MBRL）通过学习环境动态模型（世界模型）来提升样本效率，代表方法包括 Dreamer 系列和 TD-MPC2。这些方法在潜空间中建模状态转移 $D:(s_t, a_t) \to s_{t+1}$，然后利用学到的动态模型进行规划或策略优化。

**现有痛点**：现有世界模型训练时使用固定的时间步长 $\Delta t$（如 2.5ms），导致三个问题。（1）**时间分辨率过拟合**：模型训练在某个固定 $\Delta t$ 上，部署到不同观测率的真实世界时性能急剧下降（如从 400Hz 降到 50Hz），因为累积误差随滚动预测步数增加而放大。（2）**不准确的动态学习**：不以 $\Delta t$ 为条件的模型可能无法捕获系统的真实底层动态，只能拟合特定采样率下的表面行为。（3）**低效的动态学习**：真实系统通常包含多时间尺度的子系统，仅在高频率采样会对慢变子系统产生大量冗余数据，浪费计算资源。

**核心矛盾**：观测率（$\Delta t$ 的倒数）与动态学习效率之间的矛盾——根据 Nyquist-Shannon 采样定理，信号重建的最优采样率取决于其最高频率，但多子系统有不同频率，不存在单一最优采样率。

**本文目标** 如何高效训练一个世界模型，使其能在不增加样本复杂度的前提下，学习跨不同时间步长 $\Delta t$ 的底层任务动态？

**切入角度**：将 $\Delta t$ 视为世界模型的显式条件输入，而非固定常数。基于 Nyquist-Shannon 定理的洞察——不同子系统有不同最优采样率——在训练时混合多种 $\Delta t$ 值，使每个子系统都有机会在接近其最优采样率的数据上被学习。

**核心 idea**：将时间步长作为世界模型的显式输入条件，配合混合时间步长训练策略，用单步预测替代多步滚动预测，实现跨时间分辨率的鲁棒控制。

## 方法详解

### 整体框架

TAWM 基于 TD-MPC2 架构进行修改。编码器 $h$ 将观测 $o_t$ 映射到潜空间 $z_t$（不以 $\Delta t$ 为条件）。动态模型、奖励模型、价值模型和策略先验都以 $\Delta t$ 为额外输入。训练时每个 episode 从对数均匀分布 $\text{Log-Uniform}(\Delta t_{min}, \Delta t_{max})$ 中采样一个 $\Delta t$，以该时间步长与环境交互收集数据并更新模型。模型仅需单步即可预测任意 $\Delta t$ 后的状态，避免了滚动预测的累积误差。

### 关键设计

1. **时间感知潜动态模型**:

    - 功能：在潜空间中以 $\Delta t$ 为条件预测状态转移
    - 核心思路：将基线的直接映射 $z_{t+1} = D(z_t, a_t)$ 改为 Euler 积分形式 $\hat{z}_{t+\Delta t} = z_t + d(z_t, a_t, \Delta t) \cdot \tau(\Delta t)$，其中 $d(\cdot)$ 是 MLP 学习的潜空间导数函数，$\tau(\Delta t) = \max(0, \log_{10}(\Delta t) + 5)$ 是对数归一化函数。这自动保证 $\Delta t = 0$ 时状态不变：$z_{t+0} = z_t$
    - 设计动机：$\Delta t$ 跨越多个数量级（$10^{-3}$ 到 $5 \times 10^{-2}$），直接用线性 $\Delta t$ 会导致数值不稳定。对数变换 $\tau$ 将 $\Delta t$ 归一化到窄范围，使 MLP 更容易学习。对于动态复杂的任务还可选用 RK4 积分替代 Euler

2. **混合时间步长训练策略**:

    - 功能：使世界模型在固定训练预算内学习多时间尺度的底层动态
    - 核心思路：每个 episode 开始时从 $\text{Log-Uniform}(\Delta t_{min}, \Delta t_{max})$ 采样 $\Delta t$，该 episode 内以此 $\Delta t$ 采样数据并训练。对数均匀分布在时间尺度的每个数量级上分配相等的概率质量，确保高频和低频子系统都有充分的训练数据。所有训练数据存入统一的 replay buffer $\mathcal{B}$，每步从中采样 batch 更新模型
    - 设计动机：根据 Nyquist-Shannon 定理，多子系统 $f_i$ 各有最高频率 $f_{max}^i$，通过随机变化采样率覆盖各频率的 Nyquist 率附近，避免对高频子系统欠采样和对低频子系统过采样

3. **时间感知奖励与价值模型**:

    - 功能：使奖励预测和价值估计同样适应时间步长变化
    - 核心思路：奖励模型 $\hat{r}_t = R(z_t, a_t, \Delta t)$ 和价值模型 $\hat{q}_t = Q(z_t, a_t, \Delta t)$ 都以 $\Delta t$ 为输入。这是因为相同的状态-动作对在不同 $\Delta t$ 下可能产生不同的即时奖励和长期回报。策略先验 $\hat{a}_t = p(z_t, \Delta t)$ 也以 $\Delta t$ 为条件
    - 设计动机：如果只有动态模型感知时间而奖励/价值模型不感知，规划器无法正确评估不同时间分辨率下的动作质量

### 理论分析

**Lemma 4.1**：当环境动态可被 $\Delta\bar{t}$ 完全捕获时（即更小的 $\Delta t$ 不提供额外信息），最优动态函数 $d^*$ 在不同 $\Delta t$ 之间满足简单的插值关系。

**Lemma 4.2**：减少在 $\Delta\bar{t}$ 处的建模误差会降低所有 $\Delta t < \Delta\bar{t}$ 处的误差上界——大时间尺度上的改进可以"转移"到小时间尺度。

这两个引理从理论上解释了混合 $\Delta t$ 训练不增加样本复杂度的原因。

## 实验关键数据

### 主实验——Meta-World 控制任务

| 任务 | 基线 ($\Delta t=2.5$ms) 成功率 | TAWM-Euler 成功率 | TAWM-RK4 成功率 |
|------|------|------|------|
| Assembly @2.5ms | ~80% | ~85% | **~90%** |
| Assembly @20ms | ~10% | ~70% | **~80%** |
| Basketball @2.5ms | ~90% | **100%** | **100%** |
| Basketball @20ms | ~0% | ~60% | **~70%** |
| Faucet Open @50ms | ~0% | ~80% | **~90%** |

*基线在非默认 $\Delta t$ 下性能急剧下降，TAWM 保持鲁棒*

### 与 MTS3 对比

| 方法 | Faucet Open @2.5ms | @10ms | @30ms | @50ms |
|------|-------------------|-------|-------|-------|
| MTS3 (H=2) | ~80% | ~40% | ~10% | ~0% |
| MTS3 (H=5) | ~80% | ~50% | ~20% | ~5% |
| TAWM-RK4 | **~95%** | **~95%** | **~90%** | **~90%** |

*MTS3 随 $\Delta t$ 增大性能快速下降（累积误差），TAWM 保持 ~90%*

### 关键发现

- TAWM 在默认 $\Delta t=2.5$ms 上也不弱于甚至优于专门在该 $\Delta t$ 训练的基线，且在大 $\Delta t$ 上显著领先
- 仅在低观测率（$\Delta t \geq 10$ms）训练的基线完全无法收敛（所有任务 0% 成功率），而 TAWM 的混合训练始终优于任何固定 $\Delta t$ 训练
- Euler 积分在大多数 Meta-World 任务上已足够好，RK4 在动态复杂的 PDE 控制任务上更有优势
- 学习曲线显示 TAWM 的收敛速度与基线相当甚至更快，证实不增加样本需求
- 对数均匀采样在高频动态任务上更好，均匀采样在低频动态任务上额外有利

## 亮点与洞察

- 核心 idea 极其简洁：只需将 $\Delta t$ 作为额外输入 + 混合 $\Delta t$ 训练，就能获得显著的跨时间分辨率泛化能力
- Nyquist-Shannon 定理为方法提供了优雅的理论动机——将信号处理中的经典洞察引入世界模型训练
- 对数归一化函数 $\tau(\Delta t)$ 的设计虽然简单，但对训练稳定性至关重要
- 架构无关性：TAWM 可即插即用地应用于任何现有世界模型架构

## 局限与展望

- $\Delta t_{max}$ 的选择仍需根据任务经验设定，缺少自动确定最高频率 $f_{max}$ 的方法
- 当前动态模型是确定性的，在大 $\Delta t$ 下真实转移具有更大的随机性，概率化扩展值得探索
- 理论分析依赖较强的假设（"模型有足够表达能力"、"插值关系可学习"），严格性有限
- 仅在 Meta-World 和 1D PDE 控制上验证，未涉及视觉输入的复杂控制任务
- 每 episode 固定一个 $\Delta t$ 可能不如 episode 内动态调整 $\Delta t$ 更优

## 相关工作与启发

- TD-MPC2 (Hansen et al., 2024) 是直接的基线框架，TAWM 仅修改了其条件输入维度
- MTS3 (Shaj Kumar et al., 2023) 关注多时间尺度但仍在固定 $\Delta t$ 上训练且使用多步滚动，本文通过单步条件预测规避了累积误差
- Neural ODE 类方法也将连续时间引入神经网络，但计算代价更高且不直接适用于 MBRL

## 评分

⭐⭐⭐⭐ 方法简洁有效（仅加 $\Delta t$ 输入 + 混合 $\Delta t$ 训练），实验在 9 个 Meta-World 任务和 3 个 PDE 控制任务上充分展示了跨时间分辨率泛化的实用价值。Nyquist-Shannon 定理提供了优雅的理论动机，Lemma 4.1-4.2 给出了样本效率的理论解释。与 MTS3 的对比清晰展示了单步条件预测 vs 多步滚动预测的优势。但理论分析的严格性有限（依赖较强假设），且未在视觉输入场景验证。对 sim-to-real 迁移具有潜在重要性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Prediction-Powered Adaptive Shrinkage Estimation](prediction-powered_adaptive_shrinkage_estimation.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](../../NeurIPS2025/others/deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2025\] General Agents Contain World Models](general_agents_contain_world_models.md)

</div>

<!-- RELATED:END -->
