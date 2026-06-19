---
title: >-
  [论文解读] Infrequent Exploration in Linear Bandits
description: >-
  [NeurIPS 2025][linear bandits] 提出 INFEX 框架，按给定调度表在探索步执行基线算法（如 LinUCB/LinTS）、其余时刻贪心选臂，证明只要探索次数超过 $\omega(\log T)$ 即可达到与全时刻探索相同的多项对数 regret，同时大幅降低计算开销（80%-99% 时间步为贪心）。
tags:
  - "NeurIPS 2025"
  - "linear bandits"
  - "infrequent exploration"
  - "greedy policy"
  - "regret bound"
  - "exploration-exploitation"
---

# Infrequent Exploration in Linear Bandits

**会议**: NeurIPS 2025  
**arXiv**: [2510.26000](https://arxiv.org/abs/2510.26000)  
**代码**: 无  
**领域**: 其他  
**关键词**: linear bandits, infrequent exploration, greedy policy, regret bound, exploration-exploitation

## 一句话总结
提出 INFEX 框架，按给定调度表在探索步执行基线算法（如 LinUCB/LinTS）、其余时刻贪心选臂，证明只要探索次数超过 $\omega(\log T)$ 即可达到与全时刻探索相同的多项对数 regret，同时大幅降低计算开销（80%-99% 时间步为贪心）。

## 研究背景与动机

**领域现状**：线性 bandit 问题中，UCB 和 Thompson Sampling 每步都进行探索，能获得对数或次线性 regret 的最优理论保证。另一极端，纯贪心策略依赖强分布假设（如上下文充分多样性）才能成功，在固定臂集设置下通常导致线性 regret。

**现有痛点**：持续探索在医疗、安全关键系统中可能有害或不道德；纯贪心在缺乏上下文多样性时必然失败。两极之间缺乏系统研究——非频繁探索是否足以保证近优 regret？

**核心矛盾**：文献中缺少刻画"探索频率如何影响 regret"的严格分析框架。$\epsilon$-greedy 虽有偶尔探索但 regret 次优（$O(T^{2/3})$）；Explore-Then-Commit (ETC) 分阶段但同样次优。

**核心问题**：(a) 要达到对数 regret，是否必须每步探索？(b) 探索频率的阈值下界是多少？(c) 非频繁探索能否同时带来计算优势？

**核心 idea**：设计通用框架 INFEX——给定任意基线探索算法和探索调度表，在调度时刻探索、其余时刻贪心，理论上保证渐近 regret 不受探索频率影响（只要超过 $\omega(\log T)$）。

## 方法详解

### 整体框架
INFEX 接受两个输入：(1) 基线探索算法 $\mathsf{Alg}$（如 LinUCB 或 LinTS）；(2) 探索调度表 $\mathcal{T}_e \subset \mathbb{N}$（决定哪些时刻探索）。每步 $t$：若 $t \in \mathcal{T}_e$ 则按 $\mathsf{Alg}$ 选臂并观察收益；否则用 ridge 估计 $\hat{\theta}_{t-1}$ 贪心选择 $X_t = \arg\max_{x} x^\top \hat{\theta}_{t-1}$。

### 关键设计

1. **模块化探索框架**：

    - 功能：将探索与利用显式分离，允许任意基线算法作为探索组件
    - 核心思路：在时刻 $t \in \mathcal{T}_e$ 调用 $\mathsf{Alg}$，其余时刻贪心。探索频率函数 $f(t) = |\mathcal{T}_e \cap \{1,\ldots,t\}|$ 刻画截至 $t$ 的探索次数
    - 设计动机：通用性强（可插入 LinUCB、LinTS 或任何满足对数 regret 的算法），且将计算密集的探索操作限制在稀疏时刻，减少整体计算量

2. **Regret 分解与分析**：

    - 功能：建立非频繁探索下的 regret 上界
    - 核心思路：总 regret 分解为三项——(a) 基线算法在 $f(T)$ 步上的 regret $\mathcal{R}_{\mathsf{Alg}}(f(T))$；(b) 与 $T$ 无关的常数项 $G_{\text{const}}$（取决于探索调度的疏密）；(c) 贪心步的 regret $G(T) = O(\frac{(\log T + d\log\log T)^2}{\Delta})$
    - 关键引理：贪心步的 regret 与最优臂选择次数 $N_{\text{opt}}(t)$ 相关——每次选到最优臂都降低 $\hat{\theta}$ 对 $x^*$ 的估计误差，从而控制后续贪心步的 regret

3. **探索阈值的必要性证明（Theorem 3）**：

    - 功能：证明 $\omega(\log T)$ 探索次数是必要的
    - 核心思路：构造反例——若 $f(t) \neq \omega(\log t)$，则存在问题实例使 INFEX 的期望 regret 至少为 $\Omega(T^{1-\epsilon})$
    - 设计动机：给出了不可松弛的理论下界，确立了探索频率的相变阈值

### 理论结果

**Theorem 1 (INFEX Regret)**：若基线 $\mathsf{Alg}$ 有对数 regret 且 $f(t) = \omega(\log t)$，则

$$\mathcal{R}_{\text{INFEX}}(T) \leq \mathcal{R}_{\mathsf{Alg}}(f(T)) + G_{\text{const}}(\tau_{\mathsf{Alg}}, f) + O\left(\frac{(\log T + d\log\log T)^2}{\Delta}\right)$$

其中只有最后一项依赖 $T$，且与 LinUCB 的 instance-dependent 界一致。

**Theorem 2 (LinTS 新界)**：首次给出 LinTS 的 instance-dependent 对数 regret 上界。

## 实验关键数据

### 主实验 —— Regret 对比 ($d=10$, $T=10000$)

| 方法 | K=10 Regret | K=100 Regret | K=1000 Regret |
|------|-------------|--------------|---------------|
| LinUCB（每步探索） | 基线 | 基线 | 基线 |
| INFEX(LinUCB, m=5) | ≈基线或更优 | ≈基线或更优 | ≈基线或更优 |
| INFEX(LinUCB, m=20) | ≈基线 | ≈基线 | 略高于基线 |
| INFEX(LinUCB, m=100) | 略高于基线 | 略高于基线 | 高于基线 |
| LinTS（每步探索） | 基线 | 基线 | 基线 |
| INFEX(LinTS, m=5/20/100) | **均优于基线** | **均优于基线** | **均优于基线** |
| 纯贪心 | 线性 regret | 线性 regret | 线性 regret |
| $\epsilon$-greedy | 次优 | 次优 | 次优 |

### 计算效率对比

| 方法 | K=10 | K=100 | K=1000 |
|------|------|-------|--------|
| LinUCB | 1× | 1× | 1× |
| INFEX(LinUCB, m=5) | ~5× 加速 | ~5× 加速 | ~3× 加速 |
| INFEX(LinUCB, m=100) | ~20× 加速 | ~15× 加速 | ~10× 加速 |
| LinTS | 1× | 1× | 1× |
| INFEX(LinTS, m=5) | ~3× 加速 | ~3× 加速 | ~2× 加速 |

### 关键发现
- **INFEX(LinTS) 的 regret 反而优于纯 LinTS**：偶尔贪心避免了 TS 的过度探索，这是一个意外但有意义的发现
- 固定周期 $m=5$ 到 $m=100$（80%-99% 贪心）均能保持接近最优 regret
- OLSBandit 因需要 $\Omega(d^2 \log T)$ 的 forced sampling 而效率极低
- $\epsilon$-greedy 有 $\Omega(T^{2/3})$ regret 下界，无法达到对数级

## 亮点与洞察
- **填补理论空白**：在"每步探索"和"纯贪心"两极之间建立了完整的理论框架，证明了 $\omega(\log T)$ 是渐近最优 regret 的精确阈值
- **模块化设计**：INFEX 是一个 meta-algorithm，可以包装任何线性 bandit 算法，灵活且易于集成到现有系统
- **实践价值**：在安全关键场景（医疗、金融）中，将探索限制在 1%-20% 的时刻内同时保持理论保证，极具应用价值
- **LinTS 新 regret 界**：Theorem 2 首次给出 LinTS 的 instance-dependent 界，独立于本文的 INFEX 框架也有价值

## 局限与展望
- 分析要求最小 gap $\Delta > 0$ 且最优臂唯一，不直接推广到线性上下文 bandit（臂集随时间变化）
- $\omega(\log T)$ 阈值针对确定性调度表；自适应调度可能更优但分析困难
- 仅验证了 instance-dependent regret，minimax regret（$O(\sqrt{T})$）情况下的最优探索策略是开放问题
- 缺少真实应用实验（如推荐系统、临床试验），仅有合成数据验证

## 相关工作与启发
- **vs LinUCB/LinTS**：INFEX 在大多数时刻替换为贪心，计算量大幅减少，regret 保持不变甚至更优
- **vs $\epsilon$-greedy**：$\epsilon$-greedy 有 $\Omega(T^{2/3})$ 不可逾越的 regret 下界；INFEX 能达到对数级
- **vs ETC (Explore-Then-Commit)**：ETC 前期纯探索后期纯利用，INFEX 全程交替，避免了 ETC 的次优 regret
- **vs 纯贪心**：纯贪心需强分布假设，INFEX 不需要上下文多样性假设

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 填补了线性 bandit 探索频率理论的重要空白
- 实验充分度: ⭐⭐⭐ 合成实验充分但缺少真实应用验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论展开严谨，定理陈述清晰，讨论透彻
- 价值: ⭐⭐⭐⭐ 理论贡献显著，对安全关键场景有直接应用意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](../../ICML2025/learning_theory/heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)
- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](../../ICML2026/learning_theory/a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[ICML 2026\] Bandit Social Learning with Exploration Episodes](../../ICML2026/learning_theory/bandit_social_learning_with_exploration_episodes.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)

</div>

<!-- RELATED:END -->
