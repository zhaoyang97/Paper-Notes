---
title: >-
  [论文解读] DR-Submodular Maximization with Stochastic Biased Gradients: Classical and Quantum Gradient Algorithms
description: >-
  [ICLR2026][优化/理论][DR-submodular 最大化] 本文系统研究"随机有偏梯度"下的连续 DR-submodular 最大化：把分析用的 Lyapunov 框架从精确梯度推广到带偏差+噪声的梯度，借此为一类新约束（带最大元的凸集）证明 $1/e$ 近似、突破一般凸约束 $1/4$ 的 hardness，并给出经典（$O(\epsilon^{-3})$ 迭代）和量子（$O(\epsilon^{-1})$ 迭代）两套零阶算法，从理论和数值上展示量子加速。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "DR-submodular 最大化"
  - "有偏随机梯度"
  - "Lyapunov 框架"
  - "零阶优化"
  - "量子加速"
---

# DR-Submodular Maximization with Stochastic Biased Gradients: Classical and Quantum Gradient Algorithms

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=0CZAimzcVr](https://openreview.net/forum?id=0CZAimzcVr)  
**代码**: 待确认  
**领域**: optimization  
**关键词**: DR-submodular 最大化、有偏随机梯度、Lyapunov 框架、零阶优化、量子加速  

## 一句话总结
本文系统研究"随机有偏梯度"下的连续 DR-submodular 最大化：把分析用的 Lyapunov 框架从精确梯度推广到带偏差+噪声的梯度，借此为一类新约束（带最大元的凸集）证明 $1/e$ 近似、突破一般凸约束 $1/4$ 的 hardness，并给出经典（$O(\epsilon^{-3})$ 迭代）和量子（$O(\epsilon^{-1})$ 迭代）两套零阶算法，从理论和数值上展示量子加速。

## 研究背景与动机
**领域现状**：连续 DR-submodular（递减收益子模）最大化是机器学习、社交网络影响力最大化、非凸/非凹二次规划等问题的统一抽象。函数 $F$ 定义在 $[0,1]^d$ 上，DR-submodular 意味着 Hessian 处处非正（$\partial^2 F/\partial x_i\partial x_j\le 0$），最大化是 NP-hard。一条成熟的技术路线是把离散问题连续化后用一阶方法（Frank-Wolfe 变体、投影梯度上升）求近似解：单调情形可达 $(1-e^{-1})$，非单调一般凸约束可达 $1/4$（且 $1/4$ 已被证明是紧的 hardness），下闭凸约束可达 $0.385$ 乃至 $0.401$。

**现有痛点**：上述近似算法几乎都假设有一个**精确梯度 oracle**。但在影响力最大化、无线网络资源分配等真实场景里，梯度往往只能靠采样估计，天然带噪声、而且常常**有偏**（隐私扰动、采样限制、只有取值 oracle 等都会引入偏差）。一个被广泛使用的绕路办法是 Hazan 等人的平滑技巧 $\hat F(x)=\mathbb E_{u\sim B}[F(x+ru)]$：它给 $\nabla\hat F$ 提供无偏估计，于是可以套用"无偏梯度"那一套算法——但代价是优化目标被偷换成了 $\hat F$ 而不是原函数 $F$，$\|\nabla\hat F-\nabla F\|\ne 0$，需要额外去界定 $\hat F$ 与 $F$ 的差距。

**核心矛盾**：现有理论只覆盖"精确梯度"或"无偏随机梯度"两种干净的设定，而真实梯度是"有偏 + 有噪"的。直接针对有偏梯度做分析、不绕道平滑函数，才更贴近实际，但此前在 DR-submodular 领域一直是空白。

**本文目标**：(1) 建立一个能直接吃"有偏随机梯度"的统一分析框架；(2) 在此框架下覆盖一般凸约束、下闭凸约束，并探索能否找到突破 $1/4$ hardness 的新约束类；(3) 把框架落地成实际可用的零阶算法（只用函数取值），并探究量子手段能否加速。

**核心 idea**：把 Du(2022) 用于精确梯度的 **Lyapunov 框架**推广，引入对偏差 $b(x)$ 和噪声 $n(x)$ 的有界假设，让"势函数单调不减"这一充分条件在有偏设定下退化为"近似不减"，误差被显式量化进近似比和迭代复杂度——从而像搭积木一样为不同约束、不同 oracle 派生出带理论保证的算法。

## 方法详解

### 整体框架
本文不是提出单个算法，而是提出一台"算法生成器"：给定约束类型和梯度 oracle，套用同一套 Lyapunov/势函数推导，就能产出一个带近似比和迭代复杂度保证的算法。整条管线分三层。

**第一层——梯度模型**。把随机有偏梯度写成 $g(x(t))=\nabla F(x(t))+b(x(t))+n(x(t),\xi)$，其中 $b$ 是偏差、$n$ 是零均值噪声（$\mathbb E_\xi[n]=0$ 但方差非零）。为保证收敛，对偏差和噪声各加一个相对有界假设：$(m,\eta_b)$-有界偏差 $\|b(x)\|_2\le m\|\nabla F(x)\|_2+\eta_b(t)$，$(M,\eta_n)$-有界噪声 $\mathbb E[\|n(x)\|_2]\le M\|\nabla F(x)\|_2+\eta_n(t)$。两者合起来给出"估计梯度与真梯度的总方差" $A(x(t),t)\triangleq (m+M)\|\nabla F(x)\|_2+\eta_b(t)+\eta_n(t)$，这是后面所有误差项的载体。

**第二层——Lyapunov/势函数框架**。把迭代算法看成连续动力系统 $\dot x(t)=f(x(t),g(x(t)))$ 的数值离散，目标是最大化期望近似比 $\mathbb E[F(x(T))]/F(x^*)$。直接优化这个分式问题不可解，于是构造 Lyapunov 函数 $\mathcal E(x(t))=a(t)\mathbb E[F(x(t))]-b(t)F(x^*)$，把"分式优化"转成"差值优化"。只要保证 $\mathcal E$ 不减，就有 $\mathbb E[F(x(T))]\ge \frac{b(T)-b(0)}{a(T)}F(x^*)$，即近似比 $=\frac{b(T)-b(0)}{a(T)}$。离散化后对应势函数 $P(x(t_j))=a(t_j)\mathbb E[F(x(t_j))]-b(t_j)F(x^*)$。

**第三层——为具体约束实例化**。给定约束，用 DR-submodular 的不等式（如带最大元凸集的 Lemma 1）写出最优解上界的显式系数 $p(x(t)),q(x(t)),v(x(t))$，代回变分/序列优化问题 (10)/(13)，解出参数函数 $a(t),b(t)$，从而读出近似比和迭代步数 $N$。把这台机器分别接上"精确梯度 / 量子零阶估计 / 经典零阶估计"三种 oracle，就得到三套算法。

### 关键设计

**1. 把 Lyapunov 框架推广到有偏随机梯度：用"近似不减"吸收偏差与噪声**

痛点是原始 Lyapunov 框架（Du, 2022）只在精确梯度下成立——那时势函数可以严格不减。本文的做法是：在最优解上界（Assumption 3/4）里**显式保留**偏差噪声项。离散势函数的最优解上界写成 $F(x^*)\le p(x(t_j))F(x(t_j))+q(x(t_j))\mathbb E_{t_j}[\langle g(x(t_j)),v(x(t_j))\rangle]-q(x(t_j))\mathbb E_{t_j}[\|b+n\|_2\|v\|_2]$，最后那一项就是偏差噪声对上界的"侵蚀"。于是势函数从"严格不减"放松为"近似不减"，每步累积的误差正比于总方差 $A(x(t_j),t_j)$。重构后的序列优化问题 (13) 同时优化两个目标：第一项 $\frac{b(t_N)-b(t_0)}{a(t_N)}$ 决定近似比，第二项 $-\frac{1}{a(t_N)}\sum_j[2(b(t_{j+1})-b(t_j))q\,\mathbb E[A]\|v\|_2+\frac{L_1}{2}\mathbb E[a(t_{j+1})\|x(t_{j+1})-x(t_j)\|_2^2]]$ 决定迭代复杂度。这样设计的好处是：当 oracle 强化为精确梯度（$A\to 0$），第二项归零，"近似不减"自动退化为"严格不减"，复杂度回到 $O(\epsilon^{-1})$——框架对精确/有偏两种设定**平滑兼容**，并把"想压低方差影响就得增加迭代步数"这一 trade-off 量化出来。

**2. 引入"带最大元的凸集"约束，证明 $1/e$ 近似、突破 $1/4$ hardness**

一般凸约束下非单调 DR-submodular 的近似比 $1/4$ 已被证明是紧的（Mualem & Feldman, 2023），想做得更好必须给约束加结构。本文观察到一类现实约束：**带最大元的凸集**——存在 $x_{\text{large}}\in C$ 使得 $\forall x\in C, x\le x_{\text{large}}$。它自然出现在资源分配里，比如无线网络给每个用户分配资源、同时满足每人资源下界 $C=\{x\in[0,1]^d\mid Ax\ge b\}$（$A,b$ 非负）。利用 Lemma 1 的上界 $(1-\theta(t))F(x^*)\le F(x^*\vee x(t))\le F(x(t))+\langle\nabla F(x(t)),x(t)\vee x^*-x(t)\rangle$（其中 $\theta(t)=\|x(t)\|_\infty$），可解出系数 $p(x(t))=q(x(t))=\frac{1}{1-\theta(t)}$、$v(x(t))=v_{\max}(t)-x(t)$，$v_{\max}(t)=\arg\max_{v\in C}\langle g(x(t)),v\rangle$。代入连续时间方程得到 $\dot x(t)=\frac{\dot a(t)}{a(t)}v(x(t))$，取 $T=1$ 时显式解 $a(t)=e^t,\ b(t)=\|1-x(0)\|_\infty t$，最终近似比 $\frac{b(T)-b(0)}{a(T)}=\|1-x(0)\|_\infty\frac{a(0)}{a(T)}\ln\frac{a(T)}{a(0)}\le \frac{\|1-x(0)\|_\infty}{e}$，即 $1/e$ 量级。离散算法是一个 Frank-Wolfe 式凸组合更新 $x(t_{j+1})=(1-\tfrac1N e^{-1/N})x(t_j)+\tfrac1N e^{-1/N}v(t_j)$，精确梯度下 $N=O(1/\epsilon)$ 步即可。这条结果之所以"超过 hardness"，是因为 $1/4$ 的下界只对**一般**凸约束成立，最大元这个额外结构让上界分析能用上 $x^*\vee x(t)\in C$ 这一可行性，绕过了一般情形的障碍。

**3. 量子零阶梯度估计：把迭代复杂度压到 $O(\epsilon^{-1})$，追平经典一阶**

当只能查函数取值时，经典零阶估计的方差较大，导致迭代复杂度高。本文借用改进版量子 Jordan 算法（van Apeldoorn et al., 2023）+ 量子态层析来估梯度，并针对"有限量子比特只能得到 $\beta$-近似取值 oracle $F_\beta$"这一现实刻画其方差。Theorem 2 给出：要让 $\mathbb E[\|g(x)-\nabla F(x)\|_\infty]\le\sigma$，只需取值精度 $\beta\le\frac{\sigma^2}{288\pi L_1 d(8\lceil\ln(36L_0 d/\sigma)\rceil+1)}$，且取值 oracle 的查询复杂度仅 $O(8\lceil\ln(36L_0 d/\sigma)\rceil)$——对维度 $d$ 是对数级。借助 DR-submodular 的连续性常数 $L_0=\|\nabla F(0)\|_2+\|\nabla F(1)\|_2$，可把量子估计的总方差 $\mathbb E[A(x(t_j),t_j)]$ 控到 $\sigma$ 级。代回关键设计 1 的框架，Theorem 3 得到：量子零阶算法在保持 Theorem 1 各约束下相同近似比的同时，迭代复杂度只要 $O(\epsilon^{-1})$，**与经典一阶方法持平**，这就是"量子加速"的来源——量子估梯度的方差小到不拖累迭代步数。

**4. 经典零阶梯度估计 + 方差缩减：$O(\epsilon^{-3})$，匹配无偏随机梯度**

作为对照，本文也给出纯经典的零阶算法。用两点平滑估计 $\nabla\hat F(x)=\frac{d}{2r}\mathbb E_{u\sim S}[(F(x+ru)-F(x-ru))u]$，它对平滑函数 $\hat F$ 无偏、但对原函数 $F$ 有偏。Lemma 2 把这份偏差和噪声界住：偏差 $\|b(x)\|_2\le \frac{dL_1 r}{2}$（随采样半径 $r$ 缩小而减小），噪声 $\mathbb E[\|n(x)\|_2]\le\sqrt{16\sqrt{2\pi}d L_0^2}$。直接这么用方差仍大，于是叠加动量型方差缩减 $d_{t_j}=(1-\rho_{t_j})d_{t_j-1}+\rho_t g(x(t_j))$。Theorem 4 证明在 $\|x(t_{j+1})-x(t_j)\|_2\le\frac{\sqrt D}{N}$ 时，$\mathbb E[\|\nabla F(x(t_j))-d_{t_j}\|_2^2]\le\frac{Q(t_j)}{(j+9)^{2/3}}$，即方差随迭代以 $j^{-2/3}$ 衰减。代回框架后 Theorem 5 给出：经典零阶算法达到 Theorem 1 相同近似比所需迭代复杂度为 $O(\epsilon^{-3})$，与"无偏随机梯度"那一档持平。把它和设计 3 对照，就清楚量子方法把 $\epsilon^{-3}$ 直接拉到 $\epsilon^{-1}$，两个量级的差距正是量子加速的量化体现。

## 实验关键数据

### 主实验
数值实验在三个标准测试问题（DR-submodular 二次规划、DR-submodular 二次函数有限和、regular coverage 函数）× 三种约束（一般凸 / 下闭凸 / 带最大元凸集）上比较三种算法。受限于量子算法的经典模拟成本极高，维度只取 $d=3$。

| 约束 / 问题 | 量子零阶 vs 经典零阶 | 量子零阶 vs 经典一阶 |
|--------------|----------------------|----------------------|
| 一般凸约束 / 二次规划及有限和 | 收敛更快 | 解质量相当 |
| 下闭凸约束 / 二次规划 | 收敛更快、解质量更优 | 解质量略逊一点 |
| 带最大元凸集 / 二次规划及有限和 | 迭代复杂度与解质量均更优 | 收敛速度接近 |
| 带最大元凸集 / regular coverage | 收敛率仍更快 | 解质量有所下降 |

结论是：量子零阶算法在收敛速度上稳定优于经典零阶，并在多数设定下逼近经典一阶——与 $O(\epsilon^{-1})$ vs $O(\epsilon^{-3})$ 的理论一致。

### 量子时钟时间粗估

| 维度 | 误差 $\epsilon_{\text{ALG}}$ | 电路深度 | 量子比特 | 时钟时间 |
|------|------|----------|----------|----------|
| $d=10$ | $0.001$ | $\approx 543$ | $\approx 230$ | $\approx 54\mu s$ |
| $d=100$ | $0.1$ | $\approx 512$ | $\approx 1600$ | $\approx 51\mu s$ |
| $d=100$ | $0.001$ | $\approx 1002$ | $\approx 2600$ | $\approx 100\mu s$ |

假设单/双比特门耗时约 $50/100$ns，同深度门可并行，$d=100$、$\epsilon_{\text{ALG}}=0.001$ 时相位估计约 $100\mu s$；若值 oracle 的多份量子态可在多台量子计算机上并行制备，总时钟时间约几百 $\mu s$，作者认为随硬件发展还会进一步降低。

### 关键发现
- 量子加速不是空谈：理论上迭代复杂度 $O(\epsilon^{-1})$ 对 $O(\epsilon^{-3})$ 差两个量级，数值上量子零阶在所有设定下都比经典零阶收敛更快。
- "带最大元凸集"这一新约束既有理论收益（$1/e$ 突破 $1/4$）也在数值上表现最好（量子零阶逼近经典一阶）。
- regular coverage 函数是相对薄弱的场景：量子零阶解质量会下降，说明加速主要在二次型问题上稳定兑现。

## 亮点与洞察
- **把"有偏"当一等公民而非麻烦**：以往要么假设精确梯度、要么用平滑函数把目标偷换成 $\hat F$。本文直接在最优解上界里保留偏差噪声项，让框架对"精确↔有偏"平滑兼容（$A\to0$ 自动退回精确情形），这种"误差显式入账"的思路可迁移到其他需要分析有偏 oracle 的优化问题。
- **用约束结构换近似比**：$1/4$ 是一般凸约束的紧 hardness，想更好只能加结构。"最大元"是个很轻量却现实（资源分配下界约束）的结构，却足以把上界分析里 $x^*\vee x(t)\in C$ 用起来、做到 $1/e$。这提示我们：遇到 hardness 墙时，先问"我的真实约束是不是比'一般凸'更有结构"。
- **量子估梯度的价值在于方差**：量子加速来自 Jordan 算法估梯度时方差足够小，从而不拖累迭代步数——把"量子 oracle"插进经典分析框架的接口设计得很干净（只通过 $\mathbb E[A]\le\sigma$ 这一条进入），是值得复用的范式。

## 局限与展望
- **量子实验维度极小**：因经典模拟量子算法成本高，实验只做到 $d=3$，时钟时间是粗估而非真机测量，量子加速的实际可行性仍依赖未来硬件。
- **regular coverage 上解质量下降**：量子零阶的优势主要在二次型问题，覆盖类函数上有退化，说明加速并非对所有 DR-submodular 实例都稳健。
- **最大元约束的紧性未知**：表 1 中"带最大元凸集"的 hardness 仍是 Unknown，$1/e$ 是否最优、有没有更高近似比的算法是个 open question。
- **有界偏差/噪声假设**：分析依赖偏差噪声相对有界（Assumption 1/2），无界情形会导致不收敛，实际 oracle 是否满足需逐案验证。

## 相关工作与启发
- **vs Du (2022) 的 Lyapunov 框架**: Du 在精确梯度下统一了 DR-submodular 近似算法的设计与分析；本文把它推广到有偏随机梯度，新增对偏差噪声的刻画，并在框架里多生出"带最大元凸集"这条约束线。
- **vs Hazan et al. (2016) 平滑技巧**: 平滑法把目标换成 $\hat F$ 来获得无偏梯度，需额外界定 $\hat F$ 与 $F$ 的差；本文直接对原函数 $F$ 做有偏分析，避免目标偷换，更直接也更贴近实际。
- **vs Mokhtari et al. (2020) / Pedramfar et al. (2023) 的随机零阶方法**: 它们在无偏随机梯度下用动量方差缩减做到 $O(\epsilon^{-3})$；本文的经典零阶算法在更难的有偏设定下保持同样的 $O(\epsilon^{-3})$，并额外给出量子版把它降到 $O(\epsilon^{-1})$。
- **vs Augustino et al. (2025) 量子凸优化**: 后者用量子梯度方法加速凸优化；本文把量子 Jordan 估梯度迁移到非凸的 DR-submodular 最大化，并针对有限比特的 $\beta$-近似取值 oracle 重新刻画方差。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究有偏梯度下的 DR-submodular，新约束突破紧 hardness，并引入量子加速。
- 实验充分度: ⭐⭐⭐ 理论为主，数值实验受量子模拟成本限制只到 $d=3$，更多是验证性的。
- 写作质量: ⭐⭐⭐⭐ 框架层次清晰、定理-算法-复杂度对应工整，但理论密度高、对读者门槛不低。
- 价值: ⭐⭐⭐⭐ 给有偏 oracle 下的子模优化提供了统一分析工具，量子复杂度结果有前瞻性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Efficient Submodular Maximization for Sums of Concave over Modular Functions](efficient_submodular_maximization_for_sums_of_concave_over_modular_functions.md)
- [\[ICML 2026\] Differentially Private Submodular Maximization with a Knapsack Constraint](../../ICML2026/optimization/differentially_private_submodular_maximization_with_a_knapsack_constraint.md)
- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](../../ICML2026/optimization/a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)
- [\[ICML 2026\] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](../../ICML2026/optimization/budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)

</div>

<!-- RELATED:END -->
