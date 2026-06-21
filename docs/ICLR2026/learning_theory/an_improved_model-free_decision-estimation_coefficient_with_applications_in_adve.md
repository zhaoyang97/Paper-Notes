---
title: >-
  [论文解读] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs
description: >-
  [ICLR 2026][学习理论][决策估计系数(DEC)] 本文提出 Dig-DEC——一种"无需乐观主义、纯靠信息增益驱动探索"的无模型决策估计系数，它恒不大于已有的乐观 DEC，并因此能首次处理"随机转移 + 对抗奖励"的混合 MDP 在 bandit 反馈下的无模型学习，同时把在线函数估计的遗憾率从 $T^{3/4}/T^{5/6}/T^{2/3}$ 一路收紧到 $T^{2/3}/T^{7/9}/\sqrt{T}$。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "强化学习理论"
  - "决策估计系数(DEC)"
  - "无模型学习"
  - "对抗 MDP"
  - "信息增益"
  - "遗憾界"
---

# An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lbLAgGF8OO](https://openreview.net/forum?id=lbLAgGF8OO)  
**代码**: 待确认  
**领域**: 学习理论 / 强化学习理论  
**关键词**: 决策估计系数(DEC)、无模型学习、对抗 MDP、信息增益、遗憾界

## 一句话总结
本文提出 Dig-DEC——一种"无需乐观主义、纯靠信息增益驱动探索"的无模型决策估计系数，它恒不大于已有的乐观 DEC，并因此能首次处理"随机转移 + 对抗奖励"的混合 MDP 在 bandit 反馈下的无模型学习，同时把在线函数估计的遗憾率从 $T^{3/4}/T^{5/6}/T^{2/3}$ 一路收紧到 $T^{2/3}/T^{7/9}/\sqrt{T}$。

## 研究背景与动机

**领域现状**：在线决策（包括各类 RL）的复杂度，近年被统一到**决策估计系数（Decision-Estimation Coefficient, DEC）**这一框架下刻画。[FKQR21, FGH23] 提出 DEC 与配套的 **Estimation-to-Decision（E2D）** 算法原则：在"结构化观测下的决策（DMSO）"问题里，学习者每轮选策略 $\pi_t$、观测 $o_t\sim M_t(\cdot|\pi_t)$，DEC 同时给出了遗憾的上下界。

**现有痛点**：[FGH23] 的上下界之间留有一个 $\log|\mathcal{M}|$ 的缝隙（$\mathcal{M}$ 是模型类）。这个缝隙正是**模型估计的代价**——下界只反映"决策/探索"的难度，上界却额外背上了"在整个模型类上做估计"的开销。可大量实用 RL 算法是**无模型、基于值函数**的，只估计值函数 $\mathcal{F}$、不估计模型，按理代价应只含 $\log|\mathcal{F}|$。为此 [FGQ+23] 提出**乐观 DEC（optimistic DEC）**，把上界从 $|\mathcal{M}|$ 降到只依赖值函数类 $|\mathcal{F}|$。

**核心矛盾**：但乐观 DEC 用的是**乐观主义（optimism）原则**驱动探索（在目标里塞进 $V_\phi(\pi_\phi)$ 这个乐观值），而 DEC 框架本来的精神是**纯靠信息增益**探索。乐观主义一是"可能不本质、某些情形下次优"，二是**只会处理随机环境**——一旦环境对抗（每轮可换奖励），乐观更新需要**显式构造奖励估计器**，在 bandit 反馈下根本写不出来。[LWZ25] 用 DEC 框架攻下了混合 MDP（转移随机、奖励对抗），却只给了"模型类算法（估计误差大）"和"需要全信息奖励反馈的无模型算法"，**无模型 + bandit 反馈这一格留作公开问题**。

**本文目标**：(1) 设计一个无模型 DEC，去掉乐观主义、回归纯信息增益，且复杂度恒不劣于乐观 DEC；(2) 用它一举解决混合 MDP 的无模型 bandit 学习；(3) 顺手改进在线函数估计的速率。

**切入角度**：作者观察到——乐观主义在目标里扮演的角色，其实可以**用两个信息增益项替换**：一个 KL **正则项**把后验拉住、替代乐观值；一个 KL **信息增益项**捕捉乐观 DEC 忽略的分布差异。

**核心 idea**：把乐观值 $V_\phi(\pi_\phi)$ 换成"双重信息增益（dual information gain）"——由此得到的复杂度记作 **Dig-DEC**，它恒 $\le$ 乐观 DEC，特例下可任意更小，且因为不再需要奖励估计器而天然适配对抗奖励。

## 方法详解

### 整体框架

舞台是 **DMSO（Decision Making with Structured Observations）**：模型空间 $\mathcal{M}$、策略空间 $\Pi$、观测空间 $\mathcal{O}$、值函数 $V_M:\Pi\to[0,1]$。每轮环境先（不透露地）选 $M_t$，学习者选 $\pi_t$、观测 $o_t$，对参照策略 $\pi^\star$ 的遗憾是 $\mathrm{Reg}(\pi^\star)=\sum_t\big(V_{M_t}(\pi^\star)-V_{M_t}(\pi_t)\big)$。

作者沿用 [LWZ25] 的 **信息集（infoset）+ $\Phi$-受限环境** 表述统一所有情形：把 $\mathcal{M}\times\Pi$ 切成互不相交、各自只含单一策略的子集 $\phi$（每个 $\phi$ 对应唯一最优策略 $\pi_\phi$）。随机 MDP、混合 MDP、纯对抗都是这个 $\Phi$-受限环境的特例——区别只在对手在 $\phi^\star$ 内能否逐轮换模型（随机：选定 $M^\star$ 后不变；混合：转移 $P^\star$ 固定、奖励 $R_t$ 任意；纯对抗：逐轮任意）。

整套方法是一个**统一的极小极大 + 后验更新**循环（Algorithm 1）。给定信息集分布 $\rho_t\in\Delta(\Phi)$，求解带**一般散度 $D$** 的鞍点目标
$$\mathrm{AIR}^{\Phi,D}_\eta(p,\nu;\rho)=\mathbb{E}_{\pi\sim p}\mathbb{E}_{(M,\pi^\star)\sim\nu}\Big[V_M(\pi^\star)-V_M(\pi)-\tfrac{1}{\eta}D_\pi(\nu\|\rho)\Big],$$
$$p_t,\nu_t=\arg\min_{p\in\Delta(\Pi)}\max_{\nu\in\Delta(\Psi)}\mathrm{AIR}^{\Phi,D}_\eta(p,\nu;\rho_t),$$
执行 $\pi_t\sim p_t$、观测 $o_t$，再 `PosteriorUpdate` 得 $\rho_{t+1}$。遗憾被拆成两块（Theorem 6）：
$$\mathbb{E}[\mathrm{Reg}(\pi_{\phi^\star})]\le\underbrace{\mathbb{E}\Big[\textstyle\sum_t\min_p\max_\nu \mathrm{AIR}^{\Phi,D}_\eta\Big]}_{\le\,T\cdot\text{dig-dec}}+\underbrace{\mathbb{E}[\mathrm{Est}]/\eta}_{\text{估计误差}}.$$
前一块由复杂度 Dig-DEC 控制（决策难度），后一块 $\mathrm{Est}$ 由 `PosteriorUpdate` 控制（估计难度）。整篇论文的两条主线就是：**把第一块换成 Dig-DEC（去乐观）**，以及**把第二块的 $\mathrm{Est}$ 估得更小（更优估计器）**。

这是一篇理论论文，核心不在 pipeline 而在"定义—界—证明思路"，故不强加框架图。

### 关键设计

**1. Dig-DEC：用"双重信息增益"替换乐观值**

针对乐观 DEC 必须靠乐观主义、无法处理对抗的痛点，作者在一般散度框架里实例化出一个特定的 $D$：
$$D_\pi(\nu\|\rho)=\mathbb{E}_{M\sim\nu}\mathbb{E}_{o\sim M(\cdot|\pi)}\Big[\mathrm{KL}\big(\nu_\phi(\cdot|\pi,o),\rho\big)+\mathbb{E}_{\phi\sim\rho}\big[\bar D^\pi(\phi\|M)\big]\Big].$$
代入后第一块复杂度就是 **Dig-DEC（dual information gain DEC）**：
$$\text{dig-dec}^{\Phi,D}_\eta=\max_{\rho}\min_{p}\max_{\nu}\mathbb{E}_{\pi\sim p}\mathbb{E}_{(M,\pi^\star)\sim\nu}\Big[V_M(\pi^\star)-V_M(\pi)-\tfrac1\eta\mathbb{E}_o[\mathrm{KL}(\nu_\phi(\cdot|\pi,o),\rho)]-\tfrac1\eta\mathbb{E}_{\phi\sim\rho}[\bar D^\pi(\phi\|M)]\Big].$$
名字里的"双重"指目标里**两个都是信息增益度量**的项（KL 项 + $\bar D$ 项）。关键是把这个多出来的 KL 项再拆成两半：$\mathrm{KL}(\nu_\phi,\rho)+\mathbb{E}_o[\mathrm{KL}(\nu_\phi(\cdot|\pi,o),\nu_\phi)]$。第一半是**正则项**，让 $\nu$ 的边缘不偏离 $\rho$ 太远——正是这一项**替代了乐观值 $V_\phi(\pi_\phi)$**，从而摆脱乐观主义；仅靠它就能在随机情形复现乐观 DEC 的界。第二半是**信息增益项**，捕捉到乐观 DEC 看不见的东西：因为 bilinear 散度、平方 Bellman 误差这类常用 $\bar D$ 都是**均值型**、忽略分布差异，而这个 KL 增益恰能抓住分布层面的差别，使 Dig-DEC 能**严格优于**乐观 DEC（见设计 4 的玩具例子）。

**2. 基于 Bregman 散度的统一分析：让框架吃下任意散度 $D$**

[XZ23]、[LWZ25] 的后验更新固定为 $\rho_{t+1}(\phi)=\nu_t(\phi|\pi_t,o_t)$，其分析依赖一个"构造性极小极大定理"，**只适用于严格凸散度（如 KL）**，换别的散度就笨重难推。作者改用**一阶最优性 + Bregman 散度**的论证：利用 $\nu_t$ 是 $p_t$ 的最优响应，得到
$$\mathbb{E}_{\pi\sim p_t}\Big[V_{M_t}(\pi_{\phi^\star})-V_{M_t}(\pi)-\tfrac1\eta D_\pi(\delta_{M_t,\pi_{\phi^\star}}\|\rho_t)\Big]\le\max_\nu \mathrm{AIR}^{\Phi,D}_\eta(p_t,\nu;\rho_t)-\tfrac1\eta\mathbb{E}_{\pi\sim p_t}\big[\mathrm{Breg}_{D_\pi(\cdot\|\rho_t)}(\delta_{M_t,\pi_{\phi^\star}},\nu_t)\big],$$
其中 $\mathrm{Breg}_F(x,y)=F(x)-F(y)-\langle\nabla F(y),x-y\rangle\ge0$。这套写法**与镜像下降（mirror descent）的标准分析自然衔接**，因而能处理一般（非 KL）散度，算法也更灵活——例如复现 [LWZ25] 全信息混合 MDP 结果时，能选到使 $\mathrm{Est}$ 连 $\log|\Phi|$ 都不带的 $D$，而原文要用更复杂的两层算法才行。

**3. 两类估计误差 $\bar D$ 与更优的 `PosteriorUpdate`：把 $\mathrm{Est}$ 估得更小**

第二块 $\mathrm{Est}$ 的速率取决于 $\bar D$ 选哪种。作者给出两套，对应两类问题，并都改进了 [FGQ+23] 的估计器：

- **平均估计误差 $\bar D_{av}$**（标准可实现性假设，$\bar D$ 实例化为平均 Bellman 误差）：要逼近 $\sum_h(\mathbb{E}_{\pi_k,M^\star}[\ell_h(\phi;o_h)])^2$。[FGQ+23] 用**有偏**估计 $\big(\tfrac1\tau\sum_i\ell_h\big)^2$；作者把一个 epoch 的 $\tau$ 条样本**对半劈开**构造**无偏**估计
$$L_k(\phi)=\sum_h\Big(\tfrac2\tau\textstyle\sum_{i=1}^{\tau/2}\ell_h(\phi;o_h^i)\Big)\Big(\tfrac2\tau\sum_{i=\tau/2+1}^{\tau}\ell_h(\phi;o_h^i)\Big),$$
两半互相独立、乘积期望即平方真值且无偏，集中更尖锐，把 $\mathrm{Est}$ 从 $\sqrt T$ 降到 $\mathbb{E}[\mathrm{Est}]\lesssim N\log|\Phi|\,T^{1/3}$（Theorem 7）。

- **平方估计误差 $\bar D_{sq}$**（额外要 **Bellman 完备性**，$\bar D$ 实例化为平方 Bellman 误差）：此时**不需要分批（batching）**，用一个**两时间尺度、顶层带偏置损失的两层学习**结构（沿 [FGQ+23]、[AZ22] 并精炼之），把 $\mathrm{Est}$ 压到**常数级** $\mathbb{E}[\mathrm{Est}]\lesssim\log^2|\Phi|$（Theorem 11），且分析更简单。这层结构与"带比较者依赖二阶界的模型选择算法"相关，但有独特之处，作者认为可独立成趣。

**4. 不劣性与严格改进：Dig-DEC 的理论位置**

为说服读者"换掉乐观一定不亏、还常常赚"，作者给了两个夹逼。其一，对任意非负 $\bar D$ 都有 $\text{dig-dec}^{\Phi,D}_\eta\le\mathrm{dec}^\Phi_\eta$（[LWZ25] 的无模型 DEC）；更关键的是 **Theorem 13**：随机情形下 $\text{dig-dec}^{\Phi,D}_\eta\le\text{o-dec}^{\Phi,D}_\eta+\eta$，即**凡乐观 E2D 用某个 $D$ 能办到的，本算法用同一 $D$ 都能办到**（因为 DEC 通常是 $(\eta d)^\alpha$ 阶、$\alpha\le1$，那个 $+\eta$ 不改变阶）。其二，**Theorem 14** 给出一个 **3-臂 bandit** 反例：对任意 $T\ge1,\eta\le1$，[FGQ+23] 的算法 $\max_a\mathbb{E}[\mathrm{Reg}(a)]\ge\Omega(\sqrt T)$，而本文算法 $\le1$——证明了"那个 KL 信息增益项捕捉分布差异"带来的**改进可以任意大**。

## 实验关键数据

本文为纯理论工作，"关键数据"即各设定下的**遗憾界**。核心改进汇总（与 [FGQ+23] 同设定对比）：

### 在线函数估计速率改进

| 估计类型 | 子情形 | [FGQ+23] | 本文 | 备注 |
|---------|--------|----------|------|------|
| 平均估计误差 | on-policy | $T^{3/4}$ | $T^{2/3}$ | 无偏分半估计器 |
| 平均估计误差 | off-policy | $T^{5/6}$ | $T^{7/9}$ | 同上 |
| 平方误差（Bellman 完备） | — | $T^{2/3}$ | $\sqrt T$ | 两时间尺度、$\mathrm{Est}$ 降为常数 |

其中 Bellman 完备 MDP 下达到 $\sqrt T$，是 **DEC 类方法首次追平乐观主义方法**（[JLM21, XFB+23]）的表现。

### 具体 MDP 类的遗憾界（部分）

| 环境 | 设定 | $\bar D$ | 遗憾 $\mathbb{E}[\mathrm{Reg}]$ |
|------|------|------|------|
| 随机 / bilinear on-policy | Bellman 完备 | $D_{sq}$ | $H\sqrt{dT\log|\Phi|}$ |
| 随机 / coverable | Bellman 完备 | $D_{sq}$ | $H\sqrt{dT\log|\Phi|}$ |
| 随机 / bilinear on-policy | — | $D_{av}$ | $H\sqrt{d\log|\Phi|}\,T^{2/3}$ |
| 混合（对抗奖励）/ bilinear on-policy | — | $D_{av}$ | $d(H^5\log|\Phi|)^{1/4}T^{5/6}$ |
| 混合 / bilinear on-policy | Bellman 完备 | $D_{sq}$ | $d(H^5\log^2|\Phi|)^{1/4}T^{3/4}$ |
| 混合 / coverable | Bellman 完备 | $D_{sq}$ | $d(H^5\log^2|\Phi|)^{1/4}T^{3/4}$ |

### 关键发现
- **混合 MDP（随机转移 + 对抗奖励、bandit 反馈、线性奖励）的无模型亚线性遗憾是首次给出**，直接解决 [LWZ25] 的公开问题；其关键正是去掉乐观——无需显式奖励估计器，对抗奖励才得以处理。
- 所有遗憾界都**只依赖值函数类 $\log|\Phi|$、不依赖模型类 $|\mathcal{M}|$**，兑现了"无模型"（这里"无模型"指界与 $|\mathcal{M}|$ 无关，并非禁止访问 $\mathcal{M}$）。
- 框架还**推广并简化了 [XZ23]、[LWZ25] 的 AIR 分析**——一套 Bregman/镜像下降论证统一覆盖随机、混合、纯对抗，并支持非 KL 散度。

## 亮点与洞察
- **"用两个信息增益项替换乐观值"是核心洞见**：把乐观 DEC 里那个略显 ad-hoc 的乐观值 $V_\phi(\pi_\phi)$ 拆解为"KL 正则（顶替乐观）+ KL 信息增益（捕捉分布差异）"，既统一了直觉又拿到严格改进，思想很干净。
- **去乐观 = 解锁对抗**：一个看似纯理论的简化（不要乐观），恰好绕开了"对抗奖励下无法构造奖励估计器"的工程死结——理论简化反过来扩大了适用范围，是很漂亮的因果。
- **无偏分半估计器**：把一个 epoch 的样本对半劈、两半相乘得平方项的无偏估计，是一个朴素却有效、可迁移到其它"估计平方期望"场景的 trick。
- **统一框架的可迁移性**：用 Bregman/镜像下降替换"构造性极小极大定理"，让 DEC 类分析摆脱"只能用严格凸散度"的限制，为后续引入更多散度（更灵活的算法）铺了路。

## 局限与展望
- **Assumption 3 / 4 是主要约束**：混合情形只覆盖**已知特征的线性奖励**，且 $\Phi$ 的划分要满足"给定 $\phi$ 奖励到值唯一映射"。对于**奖励特征未知的混合低秩 MDP**，按本文方式划分会让 $\log|\Phi|$ 随可能特征映射数**多项式增长**，而 [LMWZ24] 只需对数增长——这一情形本框架尚未拿下，作者明确留作 future work。
- **未追平的速率仍在**：平均估计误差下 off-policy 仅到 $T^{7/9}$、混合情形多停在 $T^{3/4}\sim T^{5/6}$，离 $\sqrt T$ 还有距离；是否能在更弱假设下进一步收紧未知。
- **纯理论、无数值验证**：所有结论为遗憾界，未给实验佐证常数/实际表现。
- **计算复杂度未讨论**：每轮要解一个 $\Delta(\Pi)\times\Delta(\Psi)$ 上的鞍点 + 后验更新，计算可行性与可扩展性未展开。

## 相关工作与启发
- **vs 乐观 DEC / 乐观 E2D [FGQ+23]**：他们用乐观主义驱动探索、只能处理随机环境；本文用双重信息增益替换乐观，恒 $\le$ 其复杂度（Theorem 13），并能处理对抗奖励——这是本文最直接的对标与超越对象。
- **vs [LWZ25]（AIR / 混合 MDP 的 DEC）**：他们首次用 DEC 攻混合 MDP，但只有"模型类算法（估计误差大）"或"需全信息奖励反馈的无模型算法"；本文补上"无模型 + bandit 反馈"这一格，并把其 AIR 框架推广简化。
- **vs 模型类 DEC / E2D [FKQR21, FGH23]**：他们的界带 $\log|\mathcal{M}|$（模型估计代价）；本文走无模型路线，界只含 $\log|\Phi|$。
- **vs 纯对抗 DEC [FRSS22, XZ23]**：纯对抗下决策复杂度常常爆炸、在 MDP 中近乎 vacuous，代价变为 $\log|\Pi|$；混合设定是更可控的折中，本文正建立其无模型 bandit 结果。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Dig-DEC 用双重信息增益替换乐观、并据此解决长期公开的混合 MDP 无模型 bandit 问题，概念与结果都新。
- 实验充分度: ⭐⭐⭐⭐ 理论论文，遗憾界覆盖随机/混合多类设定并给出严格改进与反例；无（也不需）数值实验，但部分速率未追平。
- 写作质量: ⭐⭐⭐⭐ 框架—复杂度—估计—应用层层递进，定义与界清晰；但符号繁多、对读者前置知识要求高。
- 价值: ⭐⭐⭐⭐⭐ 统一并推进了 DEC 理论，去乐观的洞见与无偏估计器都具独立可迁移价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)
- [\[ICLR 2026\] Dimension-Free Decision Calibration for Nonlinear Loss Functions](dimension-free_decision_calibration_for_nonlinear_loss_functions.md)
- [\[ICLR 2026\] Improved High-Dimensional Estimation with Langevin Dynamics and Stochastic Weight Averaging](improved_high-dimensional_estimation_with_langevin_dynamics_and_stochastic_weigh.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Online Decision-Focused Learning](online_decision-focused_learning.md)

</div>

<!-- RELATED:END -->
