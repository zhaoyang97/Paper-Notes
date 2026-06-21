---
title: >-
  [论文解读] Globally Aware Optimization with Resurgence
description: >-
  [ICLR 2026][优化/理论][非凸优化] 本文把数学物理里的 **resurgence（再现/复活）理论**搬进神经网络优化：先算参数空间配分函数 $Z(g)=\int e^{-L(\theta)/g}\,d\theta$ 的发散渐近级数，再用 Borel 变换把级数的奇点一一对应到损失函数所有临界点的取值，从而为局部梯度优化器提供"目标损失值"这一**全局信息**，包成可即插的学习率调度器 SURGE。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "非凸优化"
  - "resurgence theory"
  - "Borel 变换"
  - "配分函数"
  - "学习率自适应"
  - "全局信息"
---

# Globally Aware Optimization with Resurgence

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=dhnyoea2Qj](https://openreview.net/forum?id=dhnyoea2Qj)  
**代码**: 待确认  
**领域**: optimization  
**关键词**: 非凸优化, resurgence theory, Borel 变换, 配分函数, 学习率自适应, 全局信息  

## 一句话总结
本文把数学物理里的 **resurgence（再现/复活）理论**搬进神经网络优化：先算参数空间配分函数 $Z(g)=\int e^{-L(\theta)/g}\,d\theta$ 的发散渐近级数，再用 Borel 变换把级数的奇点一一对应到损失函数所有临界点的取值，从而为局部梯度优化器提供"目标损失值"这一**全局信息**，包成可即插的学习率调度器 SURGE。

## 研究背景与动机
- **领域现状**：现代非凸优化几乎被梯度法垄断（SGD、Adam、AdamW、Muon）。高维参数空间下找全局最优是 NP-hard（穷举需 $O(2^d)$），梯度法本质是对这个难题的"逐步局部解"。
- **现有痛点**：梯度法是**天生近视的**——只看局部曲率，对全局结构（临界点在哪、取值多少）一无所知。由此带来对初始化敏感、易收敛到次优解、需要大量调参等顽疾。Polyak step 一类方法虽用"已知最优损失值"调步长，但训练神经网络时这个最优值根本拿不到。
- **核心矛盾**：优化的核心对象是**临界点的损失值**，而这恰恰是全局量；局部计算（梯度）天然给不出全局量。如何只靠局部可算的东西反推出全局结构？
- **本文目标**：找到一个"可计算"的渠道，把全局临界值从局部信息里提取出来，并以最小开销喂给任意梯度优化器。
- **核心 idea**：**【发散级数反而藏着全局信息】** 物理里微扰展开常给出阶乘发散（$\sum n!g^n$）的级数，但它的 Borel 变换 $\sum \zeta^n=\frac{1}{1-\zeta}$ 收敛，其复平面奇点精确编码了原函数的非微扰信息。本文把这套机制对应到：配分函数渐近级数的 Borel 奇点 = 损失函数临界点的取值。

## 方法详解

### 整体框架
SURGE（Singularity Unified Resurgent Gradient Enhancement）分两阶段：**分析阶段**（初始化时一次性完成）算出全局目标值集合，**优化阶段**用这些目标值动态缩放任意优化器的学习率。

```mermaid
flowchart LR
    A["初始化参数 θ_i<br/>算损失 L_0"] --> B["估配分函数<br/>Z(g)=∫e^{-L/g}dθ"]
    B --> C["拟合渐近级数<br/>Z(g)=Σ a_j g^j"]
    C --> D["Borel 变换<br/>b_n=a_n/Γ(n+1)"]
    D --> E["检测正实轴奇点<br/>→ 临界损失值集 T"]
    E --> F["优化阶段：每步取<br/>目标 ζ_t 缩放学习率"]
```

### 关键设计

**1. 配分函数作为局部到全局的桥梁：把优化景观写成统计力学问题。** 对参数 $\theta\in\mathbb{R}^d$ 与损失 $L(\theta)$，定义带"温度"耦合 $g>0$ 的配分函数 $Z(g)=\int_{\mathbb{R}^d} e^{-L(\theta)/g}\,d\theta$。当 $g\to 0^+$ 时它有渐近展开 $Z(g)\sim\sum_{n=0}^\infty a_n g^n$，系数 $a_n$ 编码了景观在临界点附近的几何。对交叉熵 $Z(g)=\int\prod_i p_\theta(y_i|x_i)^{1/g}d\theta$、对 MSE 则是高斯似然 $\mathcal N(y;f_\theta(x),\sigma^2=g)$ 上的玻尔兹曼分布——这把"全局景观信息"装进了一个虽然发散、但系数可算的级数里。

**2. Borel 奇点一一对应临界损失值：本文的理论支点。** 给定渐近级数，其 Borel 变换 $B[Z](\zeta)=\sum_n \frac{a_n}{\Gamma(n+1)}\zeta^n$ 把阶乘发散级数变成收敛函数，而它在正实轴上的奇点对应非微扰效应。文章证明（Critical Point Correspondence）：若 $\theta^\*$ 满足 $\nabla L(\theta^\*)=0$，则 $L(\theta^\*)$ 正好是 $B[Z](\zeta)$ 的一个奇点，并给出水平集表达 $B[Z(g)](t)=\int_{t=L(x)}\frac{d\sigma(x)}{|\nabla L(x)|}$。由于水平集约束 $t=L(x)$，**奇点位置 $t_i$ 就是临界点处的损失值本身**——这就把"在指数大参数空间搜临界点"等价转化为"在 $O(1)$ 大小的 Borel 正实轴区间搜奇点"。

**3. 可落地的数值流水线：变分下界估 $Z$ + 最小二乘拟级数 + 奇点检测。** 高维下蒙特卡洛估 $Z(g)$ 受维数灾难拖累，作者改用采样器 $q_\psi(\theta|g)$ 配合凹下界 $-\log\int e^{E}dq\ge -c-e^{-c}\int e^{E}dq+1$，训练一个辅助网络最大化 $J(\psi,c,g)=-c-\mathbb E_{q_\psi}[\exp(-E_\psi-c)]+1$，最优时 $c^\*(g)=\log Z(g)$，给出稳健估计。随后用加权最小二乘 $\min\sum_s w_s(Z(g_s)-\sum_j a_j g_s^j)^2$（权 $w_s=1/(g_s+\epsilon)$ 强调小耦合区）拟合系数，再算 $b_n=a_n/\Gamma(n+1)$，用比值判据 $R=\lim b_n/b_{n+1}$ 或阈值 $|\sum b_n\zeta_k^n|>\tau$ 定位奇点。整体复杂度 $O(N^2 Bp)$，对网络规模线性。

**4. SURGE 学习率包装器：把全局目标转成步长缩放。** 临界目标集 $T=\{\zeta\in S(B[Z]):\zeta\in\mathbb R^+,\zeta<L_0\}$。第 $t$ 步选当前损失之下最近的目标 $\zeta_t=\max\{\zeta\in T:\zeta<L_{\text{current}}\}$，再更新 $\theta^{(t+1)}=\theta^{(t)}-\eta\cdot\alpha(k)\cdot\nabla L$，其中 $\alpha(k)=1+\lambda\cdot\min\!\big(\frac{L(\theta^{(t)})-\zeta_t}{L(\theta^{(t)})},1\big)$。语义很直观：困在局部极小、离目标很远时第二项趋于 $\lambda$，学习率被放大 $1+\lambda$ 倍以大步逃逸；逼近目标 $L\approx\zeta_t$ 时退化回原优化器。当 Borel 分析失败时算法**优雅降级**为普通自适应优化。

## 实验关键数据
> 注：论文主结果以损失曲线图（图 1–7）呈现，未给出大表格化的数值；以下为定性结论。

### 主实验设置

| 任务 | 网络 | 数据集 | 基线优化器 |
|---|---|---|---|
| 一维函数拟合 | FC (12,10,8) | $f(x)=\sin 2x+0.5\cos 5x+0.3\sin 10x+0.1x^2$ | SGD/Adam/... |
| 分类 | MLP | MNIST | SGD, Adam, AdamW, Muon |
| 文本生成 | 小型 Transformer (~10k 参数) | Shakespeare | Adam 等 |

### 关键发现
- 摘要中报告跨任务在最终目标损失上有 **15–30%** 的一致提升。
- SURGE 包装版（虚线）相比裸优化器**加速初期收敛**并能**快速逃离局部极小**。
- 副作用：学习率被"暴力"放大会带来训练**不稳定**；若原优化过程本身泛化差，SURGE 会**加速过拟合**。
- 消融（附录 E 图 5）用随机目标替换 SURGE 算出的目标做对照，验证 Borel 目标确实"有意义"而非任意标量都能加速。

## 亮点与洞察
- **跨学科嫁接漂亮**：把 resurgence/Borel-Écalle 这套量子场论里的工具，用"配分函数渐近级数奇点↔临界损失值"一句话精准对应到优化，理论动机清晰且有定理支撑。
- **全局信息以 $O(1)$ 代价获取**：搜索空间被压到正实轴 $(0,L_0)$ 区间，且只在初始化做一次；优化阶段只是一个学习率乘子，**优化器无关**、即插即用。
- **诚实的"概念验证"定位**：作者反复强调这只是对全局目标的"粗糙用法"，把更精巧的用法留作公开邀请，姿态克制。

## 局限与展望
- **实验规模小**：仅 MNIST/小 Transformer(~10k 参数)/一维拟合，缺乏现代大模型与强基线的硬碰硬对比，15–30% 提升的普适性存疑。
- **稳定性与过拟合风险**：学习率暴力缩放导致训练震荡，且会放大原本就差的泛化，缺少自动稳压机制。
- **配分函数估计是软肋**：高维下 Borel 系数的可靠提取依赖辅助网络与级数拟合的数值稳定性，"分析失败即降级"意味着在难景观上可能根本拿不到有效目标。
- **目标用法单一**：当前只把临界值用于线性缩放学习率，未利用临界点的更多结构（如鞍点/极小的区分），潜力远未释放。

## 相关工作与启发
- **自适应优化**：Adam/AdamW/Muon 用启发式动量与学习率调度，但对全局几何盲视；Polyak step、D-Adaptation 用损失值调步长却需已知最优值——SURGE 正是补上"可算的最优/临界值"这一环。
- **采样视角的非凸优化**：Langevin dynamics、Gibbs 采样、SGLD 同样用 $e^{-L/g}$ 的玻尔兹曼形式探索景观，本文复用该配分函数但目的从"采样"转为"提取奇点"。
- **数学物理工具进入 ML**：Borel-Écalle resurgence、trans-series、Lefschetz thimble 等过去用于场论的工具，本文示范了它们分析高维优化景观的可能性，是一条值得继续挖的交叉方向。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 用 resurgence 理论把"发散级数奇点"对应到"临界损失值"，视角极其新颖，几乎没有先例。
- **实验充分度**: ⭐⭐ 仅玩具到小规模任务，无大模型/强基线对照，结果以曲线为主、缺硬数据表。
- **写作质量**: ⭐⭐⭐⭐ 数学直觉铺垫充分（从 $\sum n!g^n$ 例子讲起），定理与算法衔接清楚，定位诚实。
- **价值**: ⭐⭐⭐⭐ 作为概念验证打开了"用全局临界值指导局部优化"的新思路与一整套可延伸的数学工具箱，启发性强于即用性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Symmetry-Aware Bayesian Optimization via Max Kernels](symmetry-aware_bayesian_optimization_via_max_kernels.md)
- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](../../ICML2026/optimization/cost-aware_stopping_for_bayesian_optimization.md)
- [\[ICLR 2026\] Deep-ICE: The First Globally Optimal Algorithm for Minimizing 0–1 Loss in Two-Layer ReLU and Maxout Networks](deep-ice_the_first_globally_optimal_algorithm_for_empirical_risk_minimization_of.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)

</div>

<!-- RELATED:END -->
