---
title: >-
  [论文解读] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization
description: >-
  [ICML 2026][优化/理论][动量方法] 本文从理论上证明：在最优点随时间漂移的非平稳强凸随机优化中，动量 SGD 因"惯性滞后"系统性劣于普通 SGD，性能恶化的代价是 $(1 - \beta)^{-2}$ 量级的放大因子；并通过信息论下界论证这种代价不是分析的产物，而是任何方法不可避免的根本障碍。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "动量方法"
  - "非平稳优化"
  - "跟踪误差"
  - "分布漂移"
  - "信息论下界"
---

# On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization

**会议**: ICML 2026  
**arXiv**: [2601.12238](https://arxiv.org/abs/2601.12238)  
**代码**: 待确认  
**领域**: 优化理论  
**关键词**: 动量方法, 非平稳优化, 跟踪误差, 分布漂移, 信息论下界

## 一句话总结
本文从理论上证明：在最优点随时间漂移的非平稳强凸随机优化中，动量 SGD 因"惯性滞后"系统性劣于普通 SGD，性能恶化的代价是 $(1 - \beta)^{-2}$ 量级的放大因子；并通过信息论下界论证这种代价不是分析的产物，而是任何方法不可避免的根本障碍。

## 研究背景与动机

**领域现状**：动量方法（Heavy-Ball、Nesterov）在静态凸优化中已被证明能加速收敛、降低梯度噪声；在深度学习里几乎是默认配置。但在线学习、联邦学习、强化学习等非平稳环境中，最优点 $\theta_t^*$ 随分布漂移持续移动，过去的梯度变得过时（stale）。

**现有痛点**：经验上动量在动态环境中常出现不稳定和更差的跟踪性能，但缺乏严格理论解释——既有动态遗憾分析（Zhang 2015；Hardt 2016）只给笼统的路径长度界，没有把"动量参数 $\beta$ 与性能恶化的关系"显式刻画出来；也没有信息论下界说明这是动量内在的代价还是分析松弛的结果。

**核心矛盾**：动量同时往相反方向推——（1）在静态噪声场景下平均历史梯度降方差；（2）在分布漂移下平均"过时梯度"产生惯性滞后，让算法系统性落后于移动的目标。

**本文目标**：定量刻画非平稳强凸光滑优化中动量 SGD 相对普通 SGD 的性能差异，给出何时动量帮助 / 何时伤害的清晰界限。

**切入角度**：把 SGDM 看作"参数 + 动量缓冲"组成的 2D 动态系统，用 Lyapunov 函数分析稳定性，使 $(1 - \beta)^{-1}$ 到 $(1 - \beta)^{-2}$ 的放大因子显式出来；再用变差预算下的 Assouad 风格构造证明这些因子是信息论必然。

**核心 idea**：跟踪误差可分解为"初始化遗忘 + 噪声底线 + 漂移诱导滞后"三项；动量对每一项都有 $(1 - \beta)^{-k}$ 量级的放大，且与紧匹配的下界一致。

## 方法详解

### 整体框架
考虑时变强凸光滑问题 $G_t(\theta) = \mathbb{E}_{X_t \sim \Pi_t}[g(\theta, X_t)]$，最优点 $\theta_t^*$ 随时间漂移。目标是追踪 $\theta_t^*$，而非收敛到单点。

SGD：$\theta_{t+1} = \theta_t - \gamma_t \nabla g(\theta_t, X_{t+1})$。

广义 SGDM：$\psi_t = \theta_t + \beta_1 (\theta_t - \theta_{t-1})$，$\theta_{t+1} = \psi_t - \gamma_t \nabla g(\psi_t, X_{t+1}) + \beta_2 (\psi_t - \psi_{t-1})$；Heavy-Ball 取 $\beta_1 = 0, \beta_2 = \beta$，Nesterov 取 $\beta_1 = \beta, \beta_2 = 0$。

### 关键设计

**1. 2D Lyapunov + 跟踪误差三项分解（上界）：把动量的放大因子显式逼出来**

要刻画动量何时帮忙何时伤害，先得给出 SGD 和 SGDM 的显式跟踪误差上界。对 SGD，

$$\mathbb{E}\|\theta_t - \theta_t^*\|^2 \lesssim (1 - \gamma\mu/2)^t \|\theta_0 - \theta_0^*\|^2 + \frac{\Delta^2}{\gamma^2 \mu^2} + \frac{\sigma^2 \gamma}{\mu}$$

三项分别是初始化遗忘、漂移滞后、噪声底线。对 SGDM，参数 $\theta_t$ 和动量缓冲两个递推相互耦合，如果拆成 1D 递推会丢掉耦合信息；本文把它们统一成 2D Lyapunov 函数联合追踪，于是三项各乘上 $(1-\beta)^{-2}$ 的放大因子——这正是 $(1-\beta)^{-2}$ 因子能被显式写出来的关键技巧。结论很直白：动量在静态噪声场景下靠平均历史梯度降方差，但在分布漂移下平均的是"过时梯度"，惯性滞后让算法系统性落后于移动的目标。

**2. 时间分辨高概率界 + 加权历史漂移：不需要均匀漂移上界**

均匀漂移上界 $\Delta$ 假设漂移处处一样大，但实际漂移往往是间歇性、局部性的（季节性、突变）。这一步用鞅差分的可选停时论证替代 MGF 递推，给出任意 $t$ 时刻、概率 $1-\delta$ 下的界

$$\|\theta_t - \theta_t^*\|^2 \lesssim (1 - \gamma\mu/2)^t \|\theta_0 - \theta_0^*\|^2 + \frac{\mathfrak{D}_t}{\gamma\mu} + O(d\sigma^2\gamma/\mu)$$

其中 $\mathfrak{D}_t = \sum_{\ell=0}^{t-1}(1-\gamma\mu/2)^{t-\ell-1}\|\Delta_\ell\|^2$ 是加权历史漂移而非固定上界。它能自适应捕捉漂移的局部性——近期漂移权重大、久远漂移权重衰减，从而直接启发重启和窗口化策略。

**3. 信息论下界 + 惯性窗口：证明动量的恶化是宿命而非分析松弛**

上界说明动量会变差，但这是动量内在代价还是分析不够紧？本文在变差预算 $\mathrm{GVar}_{p,q}(g)\leq\mathbb{V}_T$ 约束下构造最坏漂移序列，给出 SGDM 的动态遗憾下界 $\mathfrak{M}_T(\Pi_\beta,\mathbb{V}_T)\gtrsim\max\{(1-\beta)^{-2/(\alpha q+2)}\cdot\mathbb{V}_T^{2q/(\alpha q+2)}T^{\alpha q/(\alpha q+2)},\ldots\}$，显式包含 $(1-\beta)^{-1}$ 到 $(1-\beta)^{-2}$ 的因子，与上界紧匹配。下界经"分块漂移"构造进一步证明：任何 SGDM 在分布变化后都必须花 $\Omega(\kappa/(1-\beta))$ 步的"惯性窗口"做瞬态调整。紧匹配的上下界一起说明"惯性滞后"是动量在非平稳下的根本宿命，不是次优分析的产物。

### 训练策略
- 恒定步长：$\gamma^* = \arg\min_\gamma \left[ \frac{192 (2 + \beta)^2}{\mu^2 \gamma^2} \Delta^2 + \frac{96}{\mu (1 - \beta)} \sigma^2 \gamma \right]$。
- 时期衰减 + 动量重启：按对数时间增加步长，时期边界处把动量缓冲重置为 0，打破过时梯度的累积。

## 实验关键数据

### 主实验：强凸二次目标 + 随机游走漂移

| 设置 | SGD ($\gamma = 0.01$) | HB | NAG | 结论 |
|------|-------------|-----|-----|------|
| $\beta = 0.50, \sigma^2 = 0.1$ | 1.036 | 0.342 | 0.349 | 适度动量帮助 |
| $\beta = 0.50, \sigma^2 = 0.8$ | 1.305 | 0.961 | 1.019 | 高噪声下动量有利 |
| $\beta = 0.90, \sigma^2 = 0.1$ | 1.029 | 0.497 | 0.453 | 轻漂移 + 低噪声，动量仍助 |
| $\beta = 0.90, \sigma^2 = 0.8$ | 1.466 | **3.899** | **3.721** | 轻漂移 + 高噪声，动量恶化 |
| $\beta = 0.99, \sigma^2 = 0.8$ | 1.403 | **38.802** | **21.038** | 强动量 + 高噪声，动量崩盘 |

5000 步后的跟踪误差。$\beta$ 从 0.50 → 0.99 时 HB / NAG 急剧恶化，SGD 相对稳健。

### 消融实验：条件数 + 漂移幅度的相互作用

| 数据集 | 条件数 $\kappa$ | SGD | HB ($\beta = 0.9$) | NAG ($\beta = 0.9$) | HB/SGD |
|--------|---------------|-----|-------------------|-------------------|--------|
| 线性回归 | 10 | 0.31 | 2.47 | 1.73 | 7.97× |
| 线性回归 | 1000 | 1.28 | 12.30 | 9.19 | 9.61× |
| 逻辑回归 | 10 | 0.42 | 3.56 | 2.18 | 8.48× |
| 教师-学生 MLP | — | 0.58 | 5.23 | 3.27 | 9.02× |

### 关键发现
- 条件数 $\kappa$ 越大，动量伤害越明显——病态问题需要更小步长 $\gamma \lesssim (1 - \beta)^2 / L$ 维稳，进一步减慢收敛。
- 漂移幅度 $\delta_{\text{rw}}$ 增加 → HB / NAG 与 SGD 差距迅速拉大。
- 高噪声 $\sigma^2 = 0.8$ + 中等漂移 $\beta = 0.9$ 是动量最脆弱区域，惯性滞后 × 噪声放大叠加。

## 亮点与洞察
- **2D Lyapunov 动态系统视角**：把 SGDM 的两个耦合递推（参数 + 动量）统一分析，是 $(1 - \beta)^{-2}$ 因子显式化的关键，可借鉴给其他带辅助变量的优化算法分析。
- **$(1 - \beta)^{-2}$ 的根本性**：通过紧匹配的上下界证明这是信息论必然，不是分析松弛。
- **时间分辨边界**：用加权历史 $\mathfrak{D}_t$ 替代均匀漂移上界 $\Delta$，能自适应间歇性漂移，直接启发"梯度-动量对齐度" $S_t = 1 - \frac{\langle \nabla g, v \rangle}{\|\nabla g\| \|v\|}$ 作为变化检测信号。
- **漂移-噪声权衡可视化**：清晰呈现动量同时放大初始化敏感度、噪声底线和漂移滞后，使权衡空间狭窄。

## 局限与展望
- 强凸假设限制；非凸场景（PŁ 条件等）可类推但作者未给结果。
- 稳定性条件 $\gamma \leq \mu (1 - \beta)^2 / (4 L^2)$ 偏保守；定性结论稳健，定量预测需更精细分析。
- 假设最优点 $\theta_t^*$ 可测；对随机 / 对抗漂移缺分析。
- 改进方向：扩展到非凸；研究自适应 $\beta(t)$ 调度；与二阶信息结合保留方差降低优势。

## 相关工作与启发
- **vs Loizou & Richtárik 2020**：他们在慢适应平稳设定下证明动量不减少 MSE；本文扩展到完全非平稳，证明动量系统性伤害。
- **vs Allen-Zhu & Hazan 2016**：他们证明确定性凸下加速最优；本文表明随机 + 非平稳组合下加速优势消失甚至反向。
- **vs Zhang 2015 / Hardt 2016（动态遗憾）**：本文用变差预算给出更精细的下界，首次定量刻画动量的信息论代价。
- **启发**：对所有"基于历史平均"的方法（如 SWA、EMA shadow weights），需要重新审视它们在非平稳场景下的表现。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  首个严格定量证明动量在分布漂移下系统性劣势的论文；2D Lyapunov + 信息论下界都具创新性。
- 实验充分度: ⭐⭐⭐⭐  从强凸二次到 MLP 四级递进，消融充分；缺少深度学习真实场景（如非平稳 RL / 联邦）的实证。
- 写作质量: ⭐⭐⭐⭐⭐  定理陈述精确，$(1 - \beta)^{-2}$ 主线贯穿全文，图表直观。
- 价值: ⭐⭐⭐⭐⭐  解决长期实践疑惑（为什么动量在非平稳下失效），为算法设计提供理论指导（重启、步长调度、动量衰减的必要性）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](../../ICLR2026/optimization/provable_and_practical_in-context_policy_optimization_for_self-improvement.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](../../NeurIPS2025/optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)

</div>

<!-- RELATED:END -->
