---
title: >-
  [论文解读] Convergence of an actor-critic gradient flow for entropy regularised MDPs in general spaces
description: >-
  [ICLR2026][强化学习][actor-critic] 本文为连续状态/动作空间、熵正则化的无穷时域 MDP，证明了一类"critic 用 TD、actor 用策略镜像下降"的耦合 actor-critic 梯度流的**稳定性与全局收敛性**，核心结论是：只要让 critic 在一个**指数级更快的时间尺度**上更新，整个流就不会有限时间爆炸，并以指数速率收敛到最优正则化价值函数。
tags:
  - "ICLR2026"
  - "强化学习"
  - "actor-critic"
  - "熵正则 MDP"
  - "策略镜像下降"
  - "Fisher–Rao 梯度流"
  - "两时间尺度"
  - "收敛性证明"
---

# Convergence of an actor-critic gradient flow for entropy regularised MDPs in general spaces

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=KUlPxDQF3T](https://openreview.net/forum?id=KUlPxDQF3T)  
**代码**: 无（纯理论）  
**领域**: 强化学习理论  
**关键词**: actor-critic、熵正则 MDP、策略镜像下降、Fisher–Rao 梯度流、两时间尺度、收敛性证明

## 一句话总结
本文为连续状态/动作空间、熵正则化的无穷时域 MDP，证明了一类"critic 用 TD、actor 用策略镜像下降"的耦合 actor-critic 梯度流的**稳定性与全局收敛性**，核心结论是：只要让 critic 在一个**指数级更快的时间尺度**上更新，整个流就不会有限时间爆炸，并以指数速率收敛到最优正则化价值函数。

## 研究背景与动机

**领域现状**：actor-critic（AC）是 RL 的主力框架——critic 做策略评估（估计 advantage / Q 函数），actor 做策略改进（用 advantage 更新策略）。近十年里，策略改进端越来越多采用 TRPO / PPO 这类信赖域方法，而它们的"软化版"——**策略镜像下降（Policy Mirror Descent, PMD）**——因为把 TRPO 的硬约束换成 KL 惩罚、变成一个便于分析的一阶方法，在理论上尤其受欢迎：对直接参数化的熵正则问题已知它能以次线性甚至线性速率收敛。

**现有痛点**：已有的 AC 收敛性分析几乎都依赖一个隐含前提——**动作空间是有限的（finite cardinality）**。在有限动作空间里，对任意有全支撑的参考测度 $\mu$，KL 散度天然被 $\log|A|$ 封顶，于是稳定性几乎"免费"得到。但连续动作空间（机器人控制、连续决策）里，相对熵正则项 $\mathrm{KL}(\pi(\cdot|s)\,\|\,\mu)$ **没有上界，可以取到 $+\infty$**，价值函数也因此缺乏先验界。

**核心矛盾**：在一般动作空间里，AC 流面临两个互相纠缠的风险。其一，KL 项可能在有限时间内爆炸（blow-up），使动力系统出现奇点；其二，镜像下降的收敛速率本身依赖一个常数项 $\int_S \mathrm{KL}(\pi^*(\cdot|s)\,\|\,\pi_0(\cdot|s))\,\mathrm d\pi^*_\rho$，而在无正则化时几乎不可能挑到一个让它有限的初始策略 $\pi_0$——因为无正则化的最优策略支撑在一堆 Dirac 分布上，要求 $\pi_0$ 提前"猜中"这些支撑并不现实。

**本文目标**：在 $Q^\pi_\tau$-可实现（线性函数逼近）假设下回答一个明确问题——*连续动作空间、熵正则 MDP 的 actor-critic 梯度流，是否稳定、是否收敛、收敛多快？*

**切入角度**：作者把离散的"TD + 镜像下降"理想化为**连续时间的耦合动力系统**，并引入**两时间尺度（two-timescale）**机制——让 critic 跑得比 actor 快很多。熵正则在这里不是负担而是钥匙：正则化保证最优策略 $\pi^*$ 有全支撑，于是只要让 $\pi_0$ 也有全支撑，上面那个 KL 常数项就有限了。

**核心 idea**：用测度空间上的凸分析 + 欧氏空间凸分析 + 两时间尺度分析三套工具拼出一个 Lyapunov 稳定性框架，证明"时间尺度分离"既是稳定性、也是收敛性的关键——critic 必须以指数级更快的速度更新，才能把连续动作空间里无界 KL 带来的"驱动力"压住。

## 方法详解

### 整体框架

本文研究的不是一个新算法，而是对一个**已被广泛使用的 AC 方案**做连续时间极限下的严格分析。考虑无穷时域 MDP $(S, A, P, c, \gamma)$，其中状态空间 $S$、动作空间 $A$ 都是 Polish 空间（完备可分度量空间，可连续），$c$ 是有界代价函数，$\gamma\in(0,1)$。给定参考测度 $\mu\in\mathcal P(A)$ 和正则强度 $\tau>0$，正则化价值函数定义为

$$V^\pi_\tau(s) = \mathbb E^\pi_s\!\left[\sum_{n=0}^\infty \gamma^n\big(c(s_n,a_n) + \tau\,\mathrm{KL}(\pi(\cdot|s_n)\,\|\,\mu)\big)\right].$$

目标是最小化 $V^\pi_\tau$。整套分析的输入是初始 critic 参数 $\theta_0$ 与初始策略 $\pi_0\in\Pi_\mu$，输出是一条轨迹 $(\theta_t,\pi_t)_{t\ge0}$，要证明它（i）不爆炸、（ii）收敛到 $V^*_\tau$。

具体地，critic 用线性逼近 $Q(s,a;\theta)=\langle\theta,\phi(s,a)\rangle$，通过最小化均方贝尔曼误差（MSBE）的 semi-gradient 来更新；actor 用 PMD 更新。把这两端的离散更新取连续时间极限，得到一对耦合的演化方程：

$$\frac{\mathrm d\theta_t}{\mathrm dt} = -\eta_t\, g(\theta_t,\pi_t),\qquad \partial_t\pi_t(\mathrm da|s) = -A(s,a;\theta_t)\,\pi_t(\mathrm da|s),$$

其中 $g$ 是 MSBE 的 semi-gradient，$A(\cdot;\theta)$ 是逼近软 advantage，$\eta_t\ge1$ 是**时间尺度分离函数**——它把 critic 流（第一式）相对于 actor 流（第二式）加速。第二式被称为**近似 Fisher–Rao 梯度流**。全部分析就围绕这对方程的稳定性（第 5 节）与收敛性（第 6 节）展开。

### 关键设计

**1. 连续时间两时间尺度耦合流：把离散 AC 理想化为可分析的动力系统**

直接分析离散的"每步先解一个 MSBE 最小化、再做一次镜像下降"的更新非常困难。作者沿用 Konda–Tsitsiklis、Zhang 等人的思路，把它放到连续时间极限里：critic 不再求解到底（避免 (6) 里的 $\arg\min$），而是做 semi-gradient 下降 $\theta_{n+1}=\theta_n - h_n\,g(\theta_n,\pi_n)$，并令时间尺度分离比 $\eta_n := h_n/\lambda_n > 1$（$h_n$ 是 critic 步长、$\lambda_n$ 是 actor 步长）。取极限后就得到上面的耦合方程。这样做的好处是把"两端交替更新"的离散组合爆炸，化简为一个连续动力系统——critic 的 semi-gradient 流 + actor 的近似 Fisher–Rao 流，从而能动用微分不等式、Lyapunov 函数、Grönwall 引理等连续工具。在一般动作空间里，因为 KL 可能无界，作者还指出实践中可能需要在每次 actor 更新前对 critic 做多次（甚至递增次数的）更新——这正是连续时间里 $\eta_t$ 随时间增长的离散对应物。

**2. 镜像下降取极限 → 测度空间上的 Fisher–Rao 梯度流**

actor 端的核心是把 PMD 的更新

$$\frac{\mathrm d\pi_{n+1}}{\mathrm d\pi_n}(a,s) = \frac{\exp\!\big(-\lambda A^{\pi_n}_\tau(s,a)\big)}{\int_A \exp(-\lambda A^{\pi_n}_\tau(s,a))\,\pi_n(\mathrm da|s)}$$

在时间变量上插值、并令步长 $\lambda\to0$，从而退化出连续时间的 **Fisher–Rao 梯度流**

$$\partial_t \ln\frac{\mathrm d\pi_t}{\mathrm d\mu}(s,a) = -A^{\pi_t}_\tau(s,a),$$

这里用到了一个关键事实 $\int_A A^\pi_\tau(s,a)\pi(\mathrm da|s)=0$（软 advantage 在策略下均值为零）。其深层意义在于：软 advantage 恰好是价值函数关于策略 $\pi$ 的**泛函导数**，因此这条流可以被理解为价值函数在概率核空间 $\mathcal P(A|S)$ 上的梯度流。这一视角把"策略改进"从参数空间搬到了**测度空间**，使得可以用测度空间上的凸分析来刻画收敛——这也是处理无界 KL 的必要工具，因为在一般动作空间里无法再靠 $\log|A|$ 这种有限基数界来兜底。

**3. forcing–damping 型 Lyapunov 稳定性框架：证明 KL 不会有限时间爆炸**

这是全文最硬的部分。记 $K_t := \sup_{s\in S}\mathrm{KL}(\pi_t(\cdot|s)\,\|\,\mu)$，$\Gamma := \lambda_\beta(1-\gamma)(1-\sqrt\gamma)$。Lemma 5.1 把耦合流刻画成一个**受迫—阻尼系统**：

$$\frac{1}{2\eta_t}\frac{\mathrm d}{\mathrm dt}|\theta_t|^2 \le -\frac{\Gamma}{2}|\theta_t|^2 + \frac{\tau^2\gamma^2 K_t^2}{\Gamma} + \frac{|c|_{B_b}^2}{\Gamma}.$$

其中**阻尼项** $-\frac{\Gamma}{2}|\theta_t|^2$ 来自损失 $\theta\mapsto L(\theta,\pi_t;\mathrm d\pi^t_\beta)$ 的 $(1-\gamma)\lambda_\beta$-强凸性（依赖 Assumption 4.3：特征二阶矩矩阵最小特征值 $\lambda_\beta>0$，且这个假设与策略无关，从而绕开了特征值连续性的麻烦），**受迫项**则是策略更新带来的 $K_t$。在有限动作空间里 $K_t$ 被常数封顶，稳定性立得；但连续空间不行，必须在 $\mathcal P(A|S)$ 上对 Fisher–Rao 流做分析。结合两者，Theorem 5.1 在 $\eta_0 > \tau/\Gamma$ 的条件下导出关于 $K_t$ 的 Grönwall 型积分不等式 $K_t^2 \le a_1 + a_2\int_0^t e^{-\tau(t-r)}K_r^2\,\mathrm dr$，进而 Corollary 5.1 得到 $\mathrm{KL}(\pi_t(\cdot|s)\,\|\,\mu)^2 \le a_1 e^{a_2 t}$、Corollary 5.2 得到 $|\theta_t|\le r_1 e^{r_2 t}$——即 KL 和 critic 参数都**不会在有限时间内爆炸**（虽然界本身随时间指数增长，但有限时间内有限，这就排除了奇点）。门槛条件 $\eta_0>\tau/\Gamma$ 第一次把"时间尺度分离"和稳定性直接挂钩。

**4. 三段式收敛证明链：把指数收敛压回连续动作空间**

稳定性保证流不爆炸后，第 6 节用三步证明收敛。第一步 Lemma 6.1 给出 $Q^{\pi_t}_\tau$ 沿流的时间导数，并指出在 critic 精确（$\partial_t\pi_t=-A^{\pi_t}_\tau$）时 $\frac{\mathrm d}{\mathrm dt}Q^{\pi_t}_\tau\le0$，即 $\{Q^{\pi_t}_\tau\}$ 沿流耗散。第二步 Theorem 6.1 证明 AC 流**保留了 $\tau$ 正则化诱导的指数收敛**，只多出一个"critic 没解到底"的误差项：

$$\min_{r\in[0,t]} V^{\pi_r}_\tau(\rho) - V^{\pi^*}_\tau(\rho) \le \frac{\tau}{2(1-\gamma)(1-e^{-\frac\tau2 t})}\Big(e^{-\frac\tau2 t}\!\!\int_S \mathrm{KL}(\pi^*\|\pi_0)\,\mathrm d\pi^*_\rho + \frac{1}{2\tau}\!\int_0^t e^{-\frac\tau2(t-r)}|\theta_r-\theta^{\pi_r}|^2\mathrm dr\Big).$$

第三步 Theorem 6.2 证明这个误差项 $|\theta_r-\theta^{\pi_r}|^2$ 又会指数衰减，只剩一个依赖真值函数变化率 $\frac{\mathrm d}{\mathrm dr}\theta^{\pi_r}$ 和时间尺度 $1/\eta_r$ 的积分。把三步串起来，主定理 **Theorem 6.3** 给出最终结论：取 $\eta_t = \eta_0 e^{k_1 t}$（critic 以指数级加速），则对**所有** $\gamma\in(0,1)$ 和 $t>0$，

$$\min_{r\in[0,t]} V^{\pi_r}_\tau(\rho) - V^{\pi^*}_\tau(\rho) \le \frac{\tau e^{-\frac\tau2 t}}{2(1-\gamma)(1-e^{-\frac\tau2 t})}\Big(\int_S \mathrm{KL}(\pi^*\|\pi_0)\,\mathrm d\pi^*_\rho + \frac{k_2}{2\tau}\Big),$$

即整条 actor-critic 流以指数速率收敛到最优正则化价值函数。这条链的关键就是"让 $\eta_t$ 指数增长"——它把第二步里的 critic 误差积分压到可控，使得连续动作空间也能享受到原本只在精确 advantage / 有限动作空间下才有的指数收敛。

### 损失函数 / 训练策略

critic 端的目标是均方贝尔曼误差 $\mathrm{MSBE}(\theta,\pi)=\frac12\int(Q(s,a;\theta)-T^\pi Q(s,a;\theta))^2\,\mathrm d\pi_\beta$，其 semi-gradient 为 $g(\theta,\pi)=\int(Q-T^\pi Q)\phi\,\mathrm d\pi_\beta$。在 $Q^\pi_\tau$-可实现（Assumption 4.1）下 MSBE$=0$ 当且仅当 $Q(\cdot;\theta)=Q^\pi_\tau$。配套的强凸损失 $L(\theta,\pi;\zeta)=\frac12\int(\langle\theta,\phi\rangle-Q^\pi_\tau)^2\,\mathrm d\zeta$ 提供了稳定性证明里的阻尼；Lemma 4.1 把 semi-gradient $g$ 与 $\nabla_\theta L$ 的几何联系起来（$-\langle g,\theta-\theta^\pi\rangle\le-(1-\sqrt\gamma)(1-\gamma)\langle\nabla_\theta L,\theta-\theta^\pi\rangle$），是连接 critic 动力学与凸性的关键引理。actor 端没有显式 loss，而是直接沿 Fisher–Rao 流演化。

## 实验关键数据

本文是**纯理论工作，没有数值实验**，核心"结果"是定理给出的稳定性与收敛性保证。下表汇总主要结论及其条件：

| 结论 | 形式 | 关键条件 |
|------|------|----------|
| KL 稳定性 (Cor. 5.1) | $\mathrm{KL}(\pi_t\|\mu)^2 \le a_1 e^{a_2 t}$（有限时间不爆炸） | Assump. 4.2/4.3，$\eta_0>\tau/\Gamma$ |
| critic 参数稳定性 (Cor. 5.2) | $|\theta_t|\le r_1 e^{r_2 t}$ | 额外 $\frac{\mathrm d}{\mathrm dt}\eta_t\le\alpha\eta_t$ |
| 收敛（含 critic 误差）(Thm 6.1) | 指数收敛 + critic 误差积分项 | Assump. 4.1/4.2 |
| 误差衰减 (Thm 6.2) | $\int_0^t e^{-\frac\tau2(t-r)}|\theta_r-\theta^{\pi_r}|^2 \le b_1 e^{-\frac\tau2 t}+b_2\!\int\frac{1}{\eta_r}|\tfrac{\mathrm d}{\mathrm dr}\theta^{\pi_r}|^2$ | $\eta_0>1/\Gamma,\ 0<\tau<1$ |
| **主定理：指数收敛 (Thm 6.3)** | $\min_r V^{\pi_r}_\tau-V^{\pi^*}_\tau \le \frac{\tau e^{-\tau t/2}}{2(1-\gamma)(1-e^{-\tau t/2})}(\cdots)$ | $\eta_t=\eta_0 e^{k_1 t}$，**对所有** $\gamma\in(0,1)$ |

关键依赖关系（"消融"视角）：

| 移除 / 削弱的条件 | 后果 |
|------|------|
| 去掉熵正则（$\tau\to0$，一般动作空间） | KL 常数项 $\int\mathrm{KL}(\pi^*\|\pi_0)$ 几乎必为 $+\infty$（$\pi^*$ 支撑在 Dirac 上），收敛分析失效 |
| 去掉时间尺度分离（$\eta_t$ 不够大 / 不增长） | 受迫项 $K_t$ 压不住，稳定性与收敛均无法保证 |
| 动作空间从一般空间退回有限基数 | KL $\le\log|A|$ 自动封顶，稳定性"免费"得到（即旧工作的情形） |

### 关键发现
- **时间尺度分离是命门**：稳定性需要 $\eta_0>\tau/\Gamma$，收敛主定理需要 $\eta_t$ 随时间**指数增长**（$\eta_t=\eta_0 e^{k_1 t}$）。critic 跑得越快，actor 看到的 advantage 估计越准，无界 KL 带来的受迫力才被压制。
- **熵正则不是绊脚石而是钥匙**：它保证最优策略 $\pi^*$ 全支撑，从而只要 $\pi_0$ 全支撑，那个原本可能发散的 KL 常数项就有限——这是把分析从有限动作空间推广到连续空间的关键转折。
- 若 MDP 因折扣因子 $\gamma$ 足够小而"自带正则化"（有效时域短），则 KL 沿流**一致有界**，且可放宽对 $\eta_t$ 的要求（Cor. E.1–E.3）。

## 亮点与洞察
- **泛函导数视角统一了 actor 端**：把软 advantage 认成价值函数对策略的泛函导数，于是 PMD 的连续极限就是测度空间 $\mathcal P(A|S)$ 上的 Fisher–Rao 梯度流——这一抽象让"策略改进"可以用测度空间凸分析来处理，是突破有限动作空间限制的思想支点。
- **forcing–damping 分解很有迁移价值**：把耦合 AC 流写成"强凸损失提供阻尼、策略更新提供受迫"的形式（Lemma 5.1），再用 Grönwall 控制受迫项，这套"先证不爆炸、再证收敛"的两段式结构，对其他两时间尺度耦合系统（GAN、双层优化、min-max 流）都有借鉴意义。
- **把"critic 加速"量化成 $\eta_t$ 的增长率**：以往两时间尺度只是定性地说"一个快一个慢"，这里精确到 $\eta_t=\eta_0 e^{k_1 t}$ 才能换来对所有 $\gamma$ 的指数收敛，给实践中"critic 多更新几次"提供了理论刻度。

## 局限与展望
- **只分析连续时间理想化**：真实算法是离散更新，离散时间的严格分析（带步长、带误差累积）留作未来工作，作者明确承认连续时间结论只是离散版的"洞见"。
- **critic 限定为线性逼近**：实践里 critic 多用非线性神经网络；作者提到把基函数推到 $N\to\infty$ 的 Hilbert 空间逼近（去掉可实现性假设）是正在进行的工作，但分析会复杂很多。
- **假设所有积分精确求值**：尤其是 semi-gradient (7)，实际需从样本估计，会引入额外 Monte-Carlo 误差，本文未纳入分析。
- $Q^\pi_\tau$-可实现假设（Assumption 4.1）回避了函数逼近误差，虽是该领域惯例，但也意味着结论对"逼近不充分"的现实情形是乐观的。

## 相关工作与启发
- **vs Kerimkulov et al. (2025a) / Lan (2023)**：他们在 advantage **完全可得**（精确镜像下降）下证明熵正则诱导指数收敛；本文把 advantage 换成 critic 的 TD 估计，多出 critic 误差项，再靠时间尺度分离把它压回指数收敛——把"理想 actor"推进到"真实耦合 actor-critic"。
- **vs Cayci et al. (2024a;b)**：同样研究熵正则下 natural AC 的收敛，但其稳定性/收敛**本质依赖动作空间有限基数**（KL $\le\log|A|$）；本文的核心贡献正是去掉这一限制，处理无界 KL 的一般动作空间。
- **vs Zhang et al. (2021)**：同样分析连续时间两时间尺度 AC，但针对无正则化 MDP、critic 用过参数化神经网络，且**未证明收敛到最优策略**、还需 restarting 机制保稳定；本文通过引入熵正则与测度空间分析，既保证稳定又证到最优正则化价值。
- **vs Borkar & Konda (1997) / Konda & Tsitsiklis (1999)**：奠基性的两时间尺度 AC 只给出无正则化、连续时间极限下的**渐近**收敛；本文给出带**速率**的指数收敛。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把熵正则 AC 梯度流的稳定性+收敛性推广到一般（连续）动作空间，攻克无界 KL 难题
- 实验充分度: ⭐⭐⭐ 纯理论无数值实验，但定理体系完整、条件清晰（理论论文按理论标准看）
- 写作质量: ⭐⭐⭐⭐ 难点（无界 KL、时间尺度分离）剖析透彻，证明路线"稳定性→收敛性"层次分明
- 价值: ⭐⭐⭐⭐ 为连续控制场景的 actor-critic 提供了理论背书，forcing-damping 框架可迁移到其他耦合优化系统

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](../../NeurIPS2025/reinforcement_learning/global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[ICLR 2026\] Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning](neuralsymbolic_approaches_for_interpretable_actor-critic_reinforcement_learning.md)
- [\[ICLR 2026\] Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns](chunking_the_critic_a_transformer-based_soft_actor-critic_with_n-step_returns.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)

</div>

<!-- RELATED:END -->
