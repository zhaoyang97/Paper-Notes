---
title: >-
  [论文解读] Online Decision-Focused Learning
description: >-
  [ICLR 2026][学习理论][静态/动态 regret] 本文首次把决策聚焦学习（decision-focused learning, DFL）从 i.i.d. 批量设定推广到目标函数随时间变化的在线设定，通过"正则化内层问题换可微性 + 近似 oracle 与扰动换可处理非凸"两招，提出 DF-FTPL 和 DF-OGD 两个算法，分别给出可证明的静态与动态 regret 上界。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "在线学习"
  - "决策聚焦学习"
  - "静态/动态 regret"
  - "双层优化"
  - "扰动 leader"
---

# Online Decision-Focused Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FJhtHBphCt](https://openreview.net/forum?id=FJhtHBphCt)  
**代码**: 待确认  
**领域**: 学习理论 / 在线学习 / 决策聚焦学习  
**关键词**: 决策聚焦学习, 在线学习, 静态/动态 regret, 双层优化, 扰动 leader

## 一句话总结
本文首次把决策聚焦学习（decision-focused learning, DFL）从 i.i.d. 批量设定推广到目标函数随时间变化的在线设定，通过"正则化内层问题换可微性 + 近似 oracle 与扰动换可处理非凸"两招，提出 DF-FTPL 和 DF-OGD 两个算法，分别给出可证明的静态与动态 regret 上界。

## 研究背景与动机
**领域现状**：很多现实决策问题在不确定下做决定，主流是 predict-then-optimize 框架——先用历史数据训一个预测模型，再把预测喂进一个优化问题来指导决策。决策聚焦学习（DFL，又叫 smart predict-then-optimize）更进一步：不再单纯优化预测精度，而是让模型直接最小化下游决策带来的损失，使模型对预测误差更鲁棒。

**现有痛点**：DFL 的全部理论分析都停留在**批量（batch）设定**——假设数据是预先收集好的 i.i.d. 样本、目标函数固定不变。可现实里环境是动态的、数据分布会漂移，这个假设根本不成立。

**核心矛盾**：要把 DFL 搬到在线设定，会把两类困难叠加起来。一方面，DFL 的目标函数 $f_t(\theta)=\langle \bar g_t(X_t), w_t^\star(\theta)\rangle$ 因为内层是"在多面体上最小化线性函数"，最优解 $w_t^\star(\theta)$ 只在有限个顶点之间跳变，导致**梯度要么是 0 要么无定义**，标准一阶方法用不了；另一方面，由于双层结构，损失**一般非凸**。在线学习本身又要求目标可随时间任意变化。已有的在线双层优化工作都依赖光滑性假设，与 DFL 的结构不兼容。

**本文目标**：为在线决策聚焦学习建立第一套可证明的理论保证，并设计能真正跑起来的算法。

**切入角度**：作者把问题形式化为一个序贯双层优化——每轮内层是"在多面体上解线性规划做决策"，外层是"优化所产生的决策成本"。既然梯度不存在，就先把内层问题正则化成可微的代理问题；既然非凸，就借用近年非凸在线学习的"近似 oracle + 扰动"思路。

**核心 idea**：用"正则化让内层决策可微 + 近似 oracle 配扰动绕过非凸"两件事组合，把 DFL 改造成可在线更新、且 regret 可证明亚线性收敛的方法。

## 方法详解

### 整体框架
考虑 $T$ 轮序贯决策。每轮 $t$，自然先选定协变量 $X_t$ 的分布和一个成本函数 $\bar g_t$；决策者**先观测到 $X_t$**（但看不到真实成本 $\bar g_t(X_t)$），用参数化模型 $g(\theta_t, X_t)$ 预测成本，并据此在凸多面体 $W=\mathrm{Conv}(v_1,\dots,v_K)$ 上做出动作

$$w_t^\star(\theta_t)\in\arg\min_{w\in W}\langle g(\theta_t,X_t), w\rangle .$$

轮末，真实成本 $\bar g_t(X_t)$ 被揭示，决策者据此把参数更新为 $\theta_{t+1}$。决策聚焦的关键在于：更新参数时直接最小化下游决策成本 $\langle \bar g_t(X_t), w_t^\star(\theta)\rangle$（而非预测误差），这正是一个**双层优化**——内层做决策、外层优化决策成本。

由于内层最优解只在顶点间跳变，外层目标 $f_t$ 不可微也非凸。本文的整体思路是：① 给内层目标加正则项 $R$，得到落在多面体严格内部、可微的代理解 $\tilde w_t(\theta)$，从而代理目标 $\tilde f_t$ 可求梯度；② 在非凸的代理目标上，借助"近似 offline oracle + 随机扰动"做在线更新，导出两个算法——DF-FTPL（对累积损失扰动，得静态 regret）和 DF-OGD（对梯度评估点扰动，得动态 regret）。整套方法是纯理论框架（无神经网络 pipeline），核心在两类正则化选择、oracle 假设与两条 regret 证明。

衡量算法好坏的两个 regret 定义为：

$$R^s_T=\sum_{t\in[T]} f_t(\theta_t)-\inf_{\theta\in\Theta}\sum_{t\in[T]}\mathbb E[f_t(\theta)],\qquad R^d_T=\sum_{t\in[T]} F_t(\theta_t)-\sum_{t\in[T]}\inf_{\theta\in\Theta} F_t(\theta),$$

其中 $F_t(\theta)=\mathbb E[f_t(\theta)\mid \mathcal H_{t-1}]$。静态 regret 与"按平均损失最优的固定策略"比；动态 regret 与"每轮各自最优的 oracle 序列"比，且取条件期望以防比较者过拟合到 $X_t$ 的具体实现、变成不切实际的强基准。

### 关键设计

**1. 正则化内层问题换取可微性：把"跳变的顶点解"磨成可微代理**

外层目标不可微的根源是 $w_t^\star(\theta)$ 在多面体顶点集 $\{v_1,\dots,v_K\}$ 之间跳变。作者沿用 Wilder et al. (2019) 的思路，给内层目标加一个正则项 $R$，得到代理决策

$$\tilde w_t(\theta)\in\arg\min_{w\in W}\big\{\langle g(\theta,X_t),w\rangle+\alpha_t R(w)\big\},$$

并改为最小化代理目标 $\tilde f_t(\theta)=\langle\bar g_t(X_t),\tilde w_t(\theta)\rangle$，其梯度形如 $\nabla\tilde f_t(\theta)=\nabla\tilde w_t(\theta)^\top\bar g_t(X_t)$。对一般多面体 $W=\{w: A^\top w\preceq b\}$，作者选**对数障碍** $R(w)=-\sum_{i=1}^n\ln(b_i-A_i^\top w)$，它把 $\tilde w_t$ 逼在 $W$ 严格内部，于是由隐函数定理可得显式 Jacobian（Lemma 4）；当 $W$ 恰为单纯形时，可改用**负熵** $R_0(w)=\sum_i w_i\ln w_i$，此时 $\tilde w_t$ 退化为 softmax，同样可微且 Jacobian 闭式。这里有一个本质 trade-off：$\alpha_t$ 越大，$\tilde w_t$ 越光滑（越好优化），但也越偏离真正想逼近的 $w_t^\star$；后续正是靠精调 $\alpha_t$ 在两者间取平衡。

**2. 近似 oracle + 扰动绕过非凸：不求全局最优，只求"够好"的局部解**

即使 $R$ 强凸、模型 $g$ 很简单，代理目标 $\tilde f_t$ 仍只 Lipschitz 而**一般非凸**，标准在线凸优化算法失效。作者沿用非凸在线学习的做法，假设可访问一个 $\xi$-近似 offline oracle $O_\xi$：对类 $\mathcal C$ 中任意 $f:\Theta\to\mathbb R$，输出 $\vartheta=O_\xi(f)$ 满足 $f(\vartheta)\le\inf_\theta f(\theta)+\xi$。这刻画了"非凸下拿不到全局最优、只能要局部最优"的现实——$\xi$ 衡量解的质量，在友好的损失景观里会很小；当 $\mathcal C$ 是可微 Lipschitz 函数时，SGD 就能当这个 oracle（深网下 SGD 也能收敛到局部最优）。

作者特意指出：另一条不靠 oracle 的路是退而求 local regret $\sum_t\|\nabla F_t(\theta_t)\|$（鼓励收敛到驻点），但在 DFL 里梯度本就为 0 或无定义，local regret **没有意义**，所以必须走 oracle + 扰动这条路。"扰动"是关键配料：非凸时往累积代价里注入随机噪声，能起到正则化整体目标、稳定更新的作用，下面两个算法分别在不同位置施加扰动。

**3. DF-FTPL：对累积损失做扰动，拿静态 regret**

第一个算法 DF-FTPL（Algorithm 1）脱胎于 Follow-the-Leader：每轮本应选"最小化迄今所有目标之和"的参数，但非凸下直接 FTL 不稳，于是采用 Follow-the-Perturbed-Leader——每轮抽一个分量服从指数分布 $\sigma_{j,t}\sim\mathrm{Exp}(\eta)$ 的噪声向量 $\sigma_t$，用 oracle 求

$$\theta_{t+1}=O_\xi\Big(\sum_{i=1}^t \tilde f_i-\langle\sigma_t,\cdot\rangle\Big).$$

注意 oracle 作用在**正则化后的可微 Lipschitz 损失** $\tilde f_i$ 上，所以实践中能直接用梯度法近似求解。定理 1 在 H1、H2 下给出静态 regret 界：取常数 $\alpha_t=\alpha$ 时

$$T^{-1}\mathbb E[R^s_T]=\tilde O\!\Big(\eta m^2 D\tfrac1{\alpha^2}+\tfrac{mD}{\eta T}+\xi+\alpha n\Big),$$

再取 $\eta\propto m^{1/4}T^{-3/4}n^{-1/2}$、$\alpha\propto m^{3/4}n^{1/2}T^{-1/4}$ 得 $T^{-1}\mathbb E[R^s_T]=\tilde O(m^{3/4}n^{3/2}T^{-1/4}+\xi)$。即只要 $\xi=O(T^{-1/4})$，平均 regret 以 $T^{-1/4}$ 衰减；它对参数维度 $m$ 是多项式依赖，但对决策空间维度 $d$ 仅通过 $\ln\ln(d)$ 项依赖——决策空间高维时格外有竞争力。相比 Suggala & Netrapalli (2020) 的单层 $O(T^{-1/2})$，本文慢一档，代价正来自双层、不可微带来的额外 $\alpha$ trade-off。

**4. DF-OGD：对梯度评估点做扰动，拿动态 regret**

DF-FTPL 的技术拿不到动态 regret，而动态 regret 在高度非平稳、最优决策逐轮剧变的环境里才是真正想要的指标。作者据此提出 DF-OGD（Algorithm 2），基于 Online Gradient Descent。每轮先用 oracle 求最近一轮代理损失的近优解 $\vartheta_t=O_\xi(\tilde f_t)$，再抽 $\delta_t\sim\mathrm{Unif}[0,1]$、令 $u_t=\vartheta_t+\delta_t(\theta_t-\vartheta_t)$ 是 $[\theta_t,\vartheta_t]$ 上的随机点，在该点评估梯度 $\tilde\nabla_t(u_t)=\nabla\tilde w_t(u_t)^\top\bar g_t(X_t)$，最后做投影梯度步 $\theta_{t+1}=\Pi_\Theta(\theta_t-\eta_t\tilde\nabla_t(u_t))$。与 DF-FTPL 的区别在于：FTPL 扰动的是**整条累积目标**，DF-OGD 扰动的是**梯度评估的位置**，且用逐轮变化的 $(\alpha_t,\eta_t)$ 序列以适应环境变化。定理 2 给出动态 regret 界：

$$T^{-1}\mathbb E[R^d_T]=\tilde O\!\Big(\mathbb E\big[\tfrac{1+P_T}{T\eta_T}+\tfrac1T\textstyle\sum_t(\tfrac{\eta_t}{\alpha_t^2}+n\alpha_t)\big]+\xi\Big),$$

其中 $P_T=\sum_{t=1}^{T-1}\|\vartheta_{t+1}-\vartheta_t\|$ 是用近优解漂移度量的环境变化量（path length）。取 $\alpha_t\propto n^{-1/2}t^{-1/4}(1+P_t)^{1/4}$、$\eta_t\propto n^{-1/2}t^{-3/4}(1+P_t)^{3/4}$ 得 $T^{-1}\mathbb E[R^d_T]=\tilde O(\mathbb E[\sqrt n(1+P_T)^{1/4}T^{-1/4}]+\xi)$。这个界**完全不依赖 $\Theta$ 的维度**、对 $d$ 仅 $\ln\ln(d)$，在高维下尤为可贵；且本文损失既非 Lipschitz（甚至可能因内层线性而不连续）也不要求平稳，唯一约束是下面的 H2 边际假设。

> **关键假设 H2（边际/margin 条件）**：存在 $C_0>0$、$\beta\in(0,1]$，使得几乎处处对任意 $t,\theta,\varepsilon\in[0,1]$，$\mathbb P\big(\inf_{j\ne I_t(\theta)}\{u_j(\theta,X_t)-u_{I_t(\theta)}(\theta,X_t)\}\ge\varepsilon\mid\mathcal H_{t-1}\big)\ge 1-C_0\varepsilon^\beta$，其中 $u_i(\theta,x)=\langle g(\theta,x),v_i\rangle$，$I_t(\theta)$ 是最优顶点下标。它刻画"最优顶点与其它顶点的目标间隔有多大"，即最优决策有多容易识别（$\varepsilon=0$ 表示无法区分）。这是控制真实最优决策 $w_t^\star$ 与正则化近似 $\tilde w_t$ 之间期望距离的核心，借自分类与 bandit 文献。

## 实验关键数据

### 主实验（背包/knapsack 决策）
实验沿用 Mandi et al. (2024) 的背包例子：每轮从 $K$ 个物品中按预测成本挑一个，成本由协变量 $X$ 生成。合成数据用高度非线性、非平稳的生成过程 $c_t(X_t)=A\sin^4((2X_t\theta_t^\star)^{-1})+\varepsilon_t$（特征相关、参数 $\theta_t^\star$ 逐轮随机漂移），决策者只用线性预测器 $g:(\theta,X)\mapsto X\theta$（故意 misspecified）。由于决策空间是单纯形，两算法都用负熵正则化。对比两个基准：PF-OGD（预测聚焦，在线最小化 MSE 后贪心决策）和 online SPO（Smart Predict-then-Optimize，用可微凸上界损失）。$T=5000$、10 次重复。

| 指标 | DF-FTPL / DF-OGD（本文） | PF-OGD | online SPO | 结论 |
|------|------|------|------|------|
| 平均累积决策成本 $t^{-1}\sum_s \bar g_{s,v_s}(X_s)$ | 更低（更优） | 较高 | 较高 | 本文两法在静态与动态环境下都更省成本 |
| 平均预测误差 MSE | 反而更高 | 低 | 中 | 符合 DFL 取向：只在乎决策成本、不在乎统计精度 |

### 关键发现
- **DFL 的"反直觉"特征被验证**：本文算法决策成本更低但 MSE 反而更高——这正是决策聚焦的本质，模型不追求预测准、只追求决策好；当模型明显 misspecified 时，这种取向才显出价值。
- 两点结论：(i) 模型明显 misspecified 时 DFL 优于 PFL；(ii) 本文算法也胜过备受推崇的 online SPO。附录还研究了模型 misspecification 程度如何影响 DFL 与 PFL 的相对表现。

## 亮点与洞察
- **"正则化换可微 + oracle/扰动换非凸"的解耦很干净**：把 DFL 在线化的两大障碍（不可微、非凸）分别用两套独立工具对付，再由两个算法各自组合，理论叙事清晰、可复用到其它"内层是线性规划"的双层在线问题。
- **维度依赖的非对称性**：两条界对决策空间维度 $d$ 都只有 $\ln\ln(d)$ 的极弱依赖，DF-OGD 甚至完全不依赖参数维度 $m$——这意味着方法天然适合高维决策空间，是很实用的卖点。
- **动态 regret 在如此恶劣条件下仍亚线性**：损失既非 Lipschitz、可能不连续、也不要求平稳，只靠一个边际假设 H2 就拿到 $(1+P_T)^{1/4}T^{-1/4}$，作者把这点视为核心贡献。
- 用 path length $P_T$（近优解的累积漂移）刻画环境变化、并据此自适应调 $(\alpha_t,\eta_t)$，是把"环境多变就多放松正则、多迈步"这一直觉落到可证明界上的巧手。

## 局限与展望
- **收敛率慢一档**：$T^{-1/4}$ 比单层光滑设定的 $T^{-1/2}$ 慢，根源是双层 + 不可微逼出的额外 $\alpha$ trade-off；作者猜想用"离散化参数空间 + expert aggregation"或许能提速，但会带来对 $m$ 指数级的依赖。
- **依赖近似 oracle 与边际假设**：理论建立在能拿到 $\xi$-近似 oracle（实践用 SGD 近似）和 H2 边际条件之上，二者在复杂模型/退化几何下未必成立或难以验证。
- **只在单纯形/背包的合成实验上验证**：决策空间是单纯形、模型是线性预测器、数据是合成的，缺少真实大规模 OR 任务的检验。
- 作者列出的方向：用 Moreau-Yosida / 近端算子等替代平滑技术处理一般凸集 $W$；研究 i.i.d. 等较温和环境下能否拿到更强保证；做更有野心的实验。

## 相关工作与启发
- **vs 批量 DFL（KKT 微分 / 随机扰动平滑 / SPO 对偶代理 / pairwise ranking）**：它们都假设固定 i.i.d. 批量数据、目标不变；本文首次把 DFL 推到目标与分布随时间任意变化的在线设定，并给出第一组可证明保证。
- **vs 在线双层优化（Shen 2023 / Tarzanagh 2024 / Lin 2023）**：这些工作依赖光滑性假设，与 DFL "梯度为 0 或无定义、非凸" 的结构不兼容，无法直接套用；本文正是为绕开这种不兼容才设计专门的正则化 + oracle 方案。
- **vs Suggala & Netrapalli (2020) 非凸在线学习**：本文 DF-FTPL 借鉴其 FTPL + 近似 oracle 思路，但对方是单层、可达 $O(T^{-1/2})$，本文是双层不可微、率降为 $T^{-1/4}$ 但对决策维度依赖更弱。
- **vs 动态 regret 文献（Zhang 2018 $O(T^{-1/2}\sqrt{T(P_T+1)})$、Huang & Wang 2025 $O((1+P_T^\infty)^{1/3}T^{-1/3})$）**：它们都建立在单层、凸/Lipschitz、甚至"准平稳"等更友好条件上；本文在非凸、非 Lipschitz、可能不连续、非平稳的更苛刻条件下取得收敛，正是其难度与价值所在。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次形式化在线决策聚焦学习并给出第一组静态/动态 regret 保证。
- 实验充分度: ⭐⭐⭐ 仅单纯形背包合成实验，规模与真实性有限，但理论为主、足以佐证。
- 写作质量: ⭐⭐⭐⭐ 双层难点拆解清晰、与已有界的对比到位。
- 价值: ⭐⭐⭐⭐ 为非平稳环境下的 predict-then-optimize 提供了可证明的算法基石。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Online Decision Making with Generative Action Sets](online_decision_making_with_generative_action_sets.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Oracle-Efficient Hybrid Online Learning with Constrained Adversaries](oracle-efficient_hybrid_learning_with_constrained_adversaries.md)
- [\[ICLR 2026\] Conformalized Decision Risk Assessment](conformalized_decision_risk_assessment.md)
- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)

</div>

<!-- RELATED:END -->
