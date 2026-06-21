---
title: >-
  [论文解读] A Faster Parameter-Free Regret Matching Algorithm
description: >-
  [ICLR2026][学习理论][遗憾匹配] 本文提出无参数的遗憾匹配变体 **MI-SPRM+**，通过一个叫"自适应遗憾域（ARD）"的技巧单调抬高累积遗憾 1-范数的下界，在两人零和博弈中**既保留无需调参的性质、又达到 $O(1/T)$ 的理论收敛率**——这是已知第一个同时做到这两点的 RM 类算法。
tags:
  - "ICLR2026"
  - "学习理论"
  - "博弈求解"
  - "在线学习"
  - "遗憾匹配"
  - "纳什均衡"
  - "收敛速率"
  - "无参数算法"
  - "自适应步长"
---

# A Faster Parameter-Free Regret Matching Algorithm

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=JLllvi7dsg](https://openreview.net/forum?id=JLllvi7dsg)  
**代码**: 待确认  
**领域**: 学习理论 / 博弈求解 / 在线学习  
**关键词**: 遗憾匹配, 纳什均衡, 收敛速率, 无参数算法, 自适应步长

## 一句话总结
本文提出无参数的遗憾匹配变体 **MI-SPRM+**，通过一个叫"自适应遗憾域（ARD）"的技巧单调抬高累积遗憾 1-范数的下界，在两人零和博弈中**既保留无需调参的性质、又达到 $O(1/T)$ 的理论收敛率**——这是已知第一个同时做到这两点的 RM 类算法。

## 研究背景与动机
**领域现状**：在两人零和正规形博弈（NFG）里求纳什均衡（NE），目前最流行的是基于遗憾最小化框架的算法。其中 Regret Matching（RM）及其变体（RM+、PRM+、DRM 等）因为在扑克等大规模博弈上的优异实战表现，被超人级博弈 AI 广泛采用，地位高于基于 OMD/FTRL 的传统遗憾最小化算法。

**现有痛点**：尴尬的是，虽然许多 OMD/FTRL 算法能拿到 $O(1/T)$ 的理论收敛率，但绝大多数 RM 变体只能证明出 $O(1/\sqrt{T})$。Farina 等人 2023 年提出的"光滑 RM+ 变体"（以 SPRM+ 为代表）首次把 RM+ 的理论收敛率提升到 $O(1/T)$，做法是让累积遗憾的 1-范数下界恒大于一个正常数。**但 SPRM+ 为此丢掉了 RM+/PRM+ 最迷人的"无参数（parameter-free）"性质**——即不需要调任何超参数。

**核心矛盾**：无参数性质在实战中极其重要，因为遗憾最小化算法对步长 $\eta$ 异常敏感。本文实验里，即便是 SPRM+ 原论文用的那个 $3\times3$ 博弈，在 $10^5$ 步后，SPRM+（$\eta=0.1$）拿到的对偶间隙（duality gap，到 NE 的距离，越小越好）就比 $\eta=0.01$ 小 10 倍、比 $\eta=1$ 小 5 倍。这意味着用 SPRM+ 必须花大量精力调参，否则收敛慢得多。于是出现一个核心矛盾：**$O(1/T)$ 收敛率与无参数性质，此前在 RM 框架下不可兼得**。

**本文目标**：设计一个 RM+ 变体，同时满足（1）无参数；（2）$O(1/T)$ 理论收敛率；（3）经验收敛也确实快（不像某些理论 $O(1/T)$ 算法实测很慢）。

**切入角度**：作者抓住 SPRM+ 收敛证明里的一个关键依赖关系——SPRM+ 能拿到 $O(1/T)$ 的步长区间是 $0<\eta<R\cdot C_0$，其中 $R$ 是累积遗憾 1-范数的下界、$C_0$ 是只与博弈相关的常数。换句话说，对**任意**给定的 $\eta>0$，只要让 $R>\eta/C_0$，$O(1/T)$ 就成立。

**核心 idea**：既然收敛与否取决于 $R$ 和 $\eta$ 的比值，那就**别去调 $\eta$，转而自适应地把 $R$ 抬上去**，让它单调增长直到超过 $\eta/C_0$——这样无需任何人工调参就能进入 $O(1/T)$ 区间。这个抬高 $R$ 的机制就是自适应遗憾域（ARD）。

## 方法详解

### 整体框架
先交代记号与背景。两人零和 NFG 中每个玩家 $i$ 在单纯形 $\mathcal{X}^i$ 上选策略 $x_i$，用对偶间隙 $\mathrm{dg}(x)=\max_{x'\in\mathcal{X}}\langle \ell^x, x-x'\rangle$ 衡量到 NE 的距离，$\mathrm{dg}(x)=0$ 当且仅当 $x$ 是 NE。遗憾最小化框架的目标是让社会遗憾 $\sum_{t=1}^{T}\langle \ell^t, x^t-x\rangle$ 次线性增长，则时间平均策略 $\bar{x}^T=\sum_t x^t/T$ 收敛到近似 NE。

RM+ 在非负象限上更新累积遗憾：$\theta^{t+1}_i=[\theta^t_i+\eta F^t_i(x^t)]_+$，策略由归一化得到 $x^t_i=\theta^t_i/\|\theta^t_i\|_1$，其中 $F^t_i=\langle\ell^t_i,x^t_i\rangle\mathbf{1}-\ell^t_i$ 是瞬时遗憾。PRM+ 在此基础上引入"用上一步反馈作当前预测"来加速。SPRM+ 则把更新空间从非负象限 $\mathbb{R}^{|A_i|}_{\geq0}$ 平移到 $\mathbb{R}^{|A_i|}_{\geq R}$（每个分量下界为常数 $R>0$），借此保证瞬时遗憾的稳定性 $\|F^{t-1}(\theta^{t-1})-F^t(\theta^t)\|_2^2\leq O(\|\theta^{t-1}-\theta^t\|_2^2)$，从而把 PRM+ 的 $O(1/\sqrt{T})$ 拉到 $O(1/T)$，代价是 $R$ 与 $\eta$ 需要满足 $0<\eta<R\cdot C_0$，即要调参。

本文的 MI-SPRM+（Algorithm 1）整体只在 SPRM+ 上做一处关键改动：**把固定的下界常数 $R$ 换成逐迭代单调增长的 $R_t$**，决策空间随之变成 $\mathbb{R}^{|A_i|}_{\geq R_t}$。每一步流程为：用上一步反馈 $F^{t-1}_i(\theta^{t-1})$ 在 $\mathbb{R}^{|A_i|}_{\geq R_t}$ 上做一次预测式更新得到 $\theta^t_i$ 并归一化出策略 $x^t_i$；再用当前反馈 $F^t_i(\theta^t)$ 更新 $\hat\theta^{t+1}_i$；最后按一个判据决定 $R_{t+1}$ 是 $R_t+1$ 还是保持 $R_t$。整个算法每步只调用一次损失梯度（single-call），单步复杂度 $O(1)$，且不含任何需要人工设定的超参数。最终输出用 $R_t$ 加权的平均策略 $\bar{x}^T=\frac{\sum_{t=1}^T R_t x^t}{\sum_{t=1}^T R_t}$。

### 关键设计

**1. 关键观察：$O(1/T)$ 的步长区间由累积遗憾 1-范数下界撑着**

这是整篇文章的支点。作者从 SPRM+ 的收敛定理（Theorem B.1）里提炼出：SPRM+ 在 $0<\eta<R\cdot C_0$ 时达到 $O(1/T)$，其中博弈相关常数 $C_0=1/\sqrt{8D(2L^2+4DL^2+4DP^2)}$（$D=\max_i|A_i|$，$L,P$ 为损失梯度的 Lipschitz 与范数界）。把这个条件反过来读：**对任意固定 $\eta>0$，只要 $R>\eta/C_0$，收敛就成立**。也就是说，决定收敛快慢的不是 $\eta$ 的绝对值，而是 $R$ 是否足够大。这个观察把"调步长"的问题转化成了"抬下界"的问题——而下界恰好是算法可以自己控制的量。正是这一步把无参数和 $O(1/T)$ 这对看似冲突的目标接到了一起。

**2. 自适应遗憾域 ARD：单调抬高 $R_t$ 而非压低步长**

ARD 是实现核心 idea 的具体机制：它通过每步调整决策空间的下界 $R_t$（即把更新约束在 $\mathbb{R}^{|A_i|}_{\geq R_t}$ 内），让累积遗憾 1-范数的下界单调递增，最终越过阈值 $1/C_0$（因为可固定取 $\eta$ 为某常数，关键是 $R_t$ 增长到超过 $\eta/C_0$）。与已有的无参数 OMD 算法 DS-OptMD 形成鲜明对比：DS-OptMD 走的是"自适应**压低**步长 $\eta$"的路线，但它的削减太保守，导致实测收敛慢；ARD 反其道而行，走"**激进地抬高** $R_t$"的路线，既保证理论 $O(1/T)$，又维持快的经验收敛。这正是为什么 MI-SPRM+ 不像 DS-OptMD 那样"理论快、实测慢"。

**3. $R_t$ 的增长判据：只在"稳定性条件被违反"时才 +1**

$R_t$ 不是无脑递增，而是带一个触发条件。具体地：
$$
R_{t+1}=
\begin{cases}
R_t+1 & \text{if } \|F^t(\theta^t)-F^{t-1}(\theta^{t-1})\|_2^2-\dfrac{B_\psi(\hat\theta^t,\theta^{t-1})+B_\psi(\hat\theta^t,\theta^t)}{2}>0,\\[4pt]
R_t & \text{else,}
\end{cases}
$$
初始 $R_1=1$，$B_\psi$ 是二次正则子 $\psi(\cdot)=\|\cdot\|_2^2/2$ 对应的 Bregman 散度。判据左边那一项正是 SPRM+ 用来保证 $O(1/T)$ 的稳定性不等式 $\|F^t-F^{t-1}\|_2^2\leq O(\|\theta^t-\theta^{t-1}\|_2^2)$ 的余量：一旦瞬时遗憾的波动超过了当前下界所能压住的程度（差值为正），就说明 $R_t$ 还不够大、稳定性条件被违反，于是把 $R_t$ 加 1 扩大下界；否则保持不变。这样 $R_t$ 只在"需要"的时候增长，避免过度放大决策空间拖慢实际收敛。

**4. 以 $R_t$ 加权的平均策略输出**

因为决策空间下界 $R_t$ 在变，简单的时间平均 $\sum_t x^t/T$ 不再对应正确的遗憾界。MI-SPRM+ 改用 $R_t$ 加权平均 $\bar{x}^T=\frac{\sum_{t=1}^T R_t x^t}{\sum_{t=1}^T R_t}$ 作为最终策略——后期 $R_t$ 大（此时已进入稳定的 $O(1/T)$ 区间）的迭代被赋予更大权重，前期 $R_t$ 小、尚未稳定的迭代权重小。Theorem 4.1 证明在两人零和 NFG 中所有玩家都用 MI-SPRM+ 时，这个加权平均以 $O(1/T)$ 速率收敛到近似 NE。

### 损失函数 / 训练策略
本文是博弈求解的在线学习算法，不涉及训练损失。收敛性的证明思路是：先用 $R_t$ 单调增长且最终超过 $1/C_0$ 这一性质（由 $\mathbb{R}^{|A_i|}_{\geq R_t}$ 的定义保证 $R_t$ 是累积遗憾 1-范数的下界），把每步纳入 SPRM+ 那套稳定性论证；再结合 $R_t$ 加权得到社会遗憾上界，最终化为 $O(1/T)$ 的对偶间隙界（详见原文附录 A）。对多人一般和 NFG，作者只能给出 $O(1)$ 的社会遗憾界 $\big(\sum_t R_t\langle\ell^t,x^t-x\rangle\big)/\big(\sum_t R_t\big)\leq O(1)$（因为求 NE 属 PPAD 类，无多项式时间收敛保证），与 SPRM+ 原论文一致；对扩展形博弈（EFG），算法设计可直接延伸，但 $O(1/T)$ 理论保证不再成立，不过与 CFR 框架结合后经验上仍呈现 $O(1/T)$ 甚至更快的收敛。

## 实验关键数据

### 主实验
评测在两类博弈上进行：SPRM+ 原论文的 $3\times3$ 零和 NFG（收益矩阵 $[[3,0,-3],[0,3,-4],[0,0,1]]$），以及随机生成的、维度 $\{10,30,50,100\}$ 的两人零和 NFG（每维 60 个独立实例，收益矩阵分别取标准差 1/10/100 的高斯分布）。对比对象：RM+、PRM+、SPRM+、OGDA、OMWU、DS-OptMD。指标为对偶间隙（越小越好）。其中 MI-SPRM+、RM+、PRM+、DS-OptMD 是无参数算法；对非无参数的 SPRM+、OGDA 则从 $\{0.01,0.1,1\}$ 里挑步长。

| 算法（$3\times3$ NFG，$10^5$ 步） | 是否无参数 | 理论收敛率 | 对偶间隙 |
|------|------|------|------|
| **MI-SPRM+（本文）** | 是 | $O(1/T)$ | **2.6e-5** |
| SPRM+（$\eta=0.1$） | 否 | $O(1/T)$ | 4.0e-5 |
| SPRM+（$\eta=1$） | 否 | $O(1/T)$ | 2.1e-4 |
| SPRM+（$\eta=0.01$） | 否 | $O(1/T)$ | 3.9e-4 |

MI-SPRM+ 在**完全不调参**的情况下，对偶间隙比调到最优步长的 SPRM+（$\eta=0.1$）还小，更是把没调好的 SPRM+（$\eta=0.01$/$\eta=1$）甩开约一个数量级。在随机 NFG 上，MI-SPRM+ 在所有维度都拿到 $O(1/T)$ 的经验收敛率，并优于全部对比算法。

### 消融 / 对比分析

| 配置 | 现象 | 说明 |
|------|------|------|
| SPRM+ 换不同 $\eta$ | $10^5$ 步对偶间隙相差最多约 10 倍 | 印证 RM+ 类算法对步长高度敏感、必须调参 |
| DS-OptMD（无参数 OMD） | 经验收敛显著慢于 $O(1/T)$ | "压低步长"路线太保守，理论快但实测慢 |
| OGDA / OMWU（传统遗憾最小化） | 比 RM+ 类更敏感，标准差 1/10 时仅个别 $\eta$ 收敛 | 传统 OMD/FTRL 调参更难 |
| **MI-SPRM+（ARD 抬高 $R_t$）** | 无参数、理论与经验均 $O(1/T)$ 且最快 | "抬高下界"路线兼顾理论与实战 |

### 关键发现
- MI-SPRM+ 的核心增益来自 ARD：把 SPRM+ 的固定下界 $R$ 换成自适应增长的 $R_t$，是无参数化的唯一来源。
- 与同为无参数、同样理论 $O(1/T)$ 的 DS-OptMD 对比尤为说明问题——后者实测慢，验证了"激进抬高 $R$"优于"保守压低 $\eta$"。
- 与 CFR 结合后可处理扩展形博弈，经验收敛达 $O(1/T)$ 或更快，单步复杂度仍 $O(1)$（相比之下 Clairvoyant CFR 单步 $O(\log T)$）。

## 亮点与洞察
- **把"调参"问题转译成"控量"问题**：核心洞察是 SPRM+ 的收敛条件 $0<\eta<R\cdot C_0$ 里，$R$ 是算法能自己掌控的量。既然 $\eta/R$ 的比值才决定收敛，那就不动 $\eta$、转而抬 $R$——一个非常干净的视角转换，值得迁移到其他"收敛依赖某可控下界/上界"的在线学习算法。
- **"抬下界"对比"压步长"的方向之争**：DS-OptMD 压步长理论快实测慢，本文抬下界两头都快。这提醒：当多个量共同决定收敛时，挑那个"可以激进调整且不伤经验收敛"的量去自适应，往往比保守缩另一个量更划算。
- **触发式增长 + 加权平均的配合**：$R_t$ 只在稳定性余量为负时才 +1，配合 $R_t$ 加权输出，让"该稳的时候稳、该重视的迭代重视"，是无参数算法里值得借鉴的工程化处理。

## 局限与展望
- **仅证明两人零和 NFG 的 $O(1/T)$**：多人一般和 NFG 只能给 $O(1)$ 社会遗憾界（受 PPAD 复杂度所限，这是该类问题的固有壁垒，非本文缺陷）；EFG 上 $O(1/T)$ 理论保证不再成立，只有结合 CFR 后的经验结果。
- **未达到 stepsize-invariance（强无参数）**：RM+/PRM+ 具有"输出与 $\eta$ 取值完全无关"的更强性质，本文坦言据其所知尚无算法能同时做到 $O(1/T)$ 和 stepsize-invariance，MI-SPRM+ 只聚焦较弱的无参数性质。
- **$R_t$ 整数 +1 的增长粒度较粗**：判据触发即 $R_t\leftarrow R_t+1$，步幅固定。能否用更细的自适应步幅在保持理论保证下进一步提速，是一个自然的改进方向。

## 相关工作与启发
- **vs SPRM+（Farina et al., 2023）**：SPRM+ 用固定下界 $R$ 把 RM+ 提到 $O(1/T)$，但需调 $\eta$ 满足 $\eta<R\cdot C_0$；本文把 $R$ 变成单调增长的 $R_t$，去掉了调参需求，且经验上还更快。二者每步都 single-call、$O(1)$ 复杂度。
- **vs DS-OptMD（Hsieh et al., 2021）**：同为无参数且理论 $O(1/T)$，但 DS-OptMD 走 OMD 路线、靠自适应削减步长，太保守导致实测远慢于 $O(1/T)$；本文走 RM+ 路线、靠抬高遗憾域下界，理论与经验兼快。
- **vs RM+/PRM+（Tammelin 2014；Farina et al., 2021）**：它们是（强）无参数算法但只有 $O(1/\sqrt{T})$；本文在保住无参数的同时把收敛率提到 $O(1/T)$。
- **vs doubling trick（Auer et al., 1995）**：经典无参数化手段，但即便底层算法是 $O(1/T)$，doubling trick 也只能得到 $O(\log^2 T/T)$；ARD 直接抬下界，避免了这层 $\log^2 T$ 损失。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 已知首个同时达到无参数与 $O(1/T)$ 的 RM 类算法，"抬下界代替调步长"的视角清晰且非平凡。
- 实验充分度: ⭐⭐⭐⭐ 覆盖固定 + 多维随机零和 NFG 并与 6 个代表算法对比，但多人/EFG 仅经验性验证。
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导紧扣 SPRM+ 收敛条件，ARD 的来龙去脉解释清楚。
- 价值: ⭐⭐⭐⭐ 对依赖 RM+ 的大规模博弈 AI 很实用——省去敏感的步长调参，理论保证也更强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bandit Learning in Matching Markets Robust to Adversarial Corruptions](bandit_learning_in_matching_markets_robust_to_adversarial_corruptions.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Dimension-Free Decision Calibration for Nonlinear Loss Functions](dimension-free_decision_calibration_for_nonlinear_loss_functions.md)
- [\[ICLR 2026\] Convergence Dynamics of Over-Parameterized Score Matching for a Single Gaussian](convergence_dynamics_of_over-parameterized_score_matching_for_a_single_gaussian.md)
- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)

</div>

<!-- RELATED:END -->
