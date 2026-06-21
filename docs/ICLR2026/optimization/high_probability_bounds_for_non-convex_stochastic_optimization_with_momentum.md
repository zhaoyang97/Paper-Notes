---
title: >-
  [论文解读] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum
description: >-
  [ICLR 2026][优化/理论][SGDM] 本文为带动量的随机梯度下降（SGDM）在非凸设定下补齐了**高概率**的收敛界与泛化界：通过把噪声放宽到 sub-Weibull 重尾、并逐级叠加 PL 与 Bernstein 结构假设，得到一条从 $\tilde O(1/\sqrt T)$、$\tilde O(1/T)$ 到维度无关 $\tilde O(1/n^2)$ 的完整界层级，其中 SGDM 的泛化界属业界首次。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "SGDM"
  - "高概率界"
  - "非凸优化"
  - "泛化界"
  - "重尾噪声"
---

# High Probability Bounds for Non-Convex Stochastic Optimization with Momentum

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KirKWFPYJA](https://openreview.net/forum?id=KirKWFPYJA)  
**代码**: 无（未公开）  
**领域**: 非凸随机优化 / 学习理论  
**关键词**: SGDM、高概率界、非凸优化、泛化界、重尾噪声

## 一句话总结
本文为带动量的随机梯度下降（SGDM）在非凸设定下补齐了**高概率**的收敛界与泛化界：通过把噪声放宽到 sub-Weibull 重尾、并逐级叠加 PL 与 Bernstein 结构假设，得到一条从 $\tilde O(1/\sqrt T)$、$\tilde O(1/T)$ 到维度无关 $\tilde O(1/n^2)$ 的完整界层级，其中 SGDM 的泛化界属业界首次。

## 研究背景与动机

**领域现状**：带动量的随机梯度下降（SGDM，即 Polyak 重球法）几乎是深度学习里默认的优化器，更新规则在 SGD 基础上加一项动量
$$x_{t+1} = x_t - \eta_t \nabla f(x_t; z_{j_t}) + \gamma(x_t - x_{t-1}),$$
靠历史方向的惯性平滑随机梯度。但理论侧严重滞后于实践：分析 SGDM「学得好不好」需要从两个互补角度看——**收敛界**（对经验风险 $F_S$ 优化得多好）和**泛化界**（学到的模型在未见数据上表现如何）。

**现有痛点**：对 SGDM 的现有分析几乎全是**期望意义**（in-expectation）的界。期望界有两个硬伤：一是它无法排除「极差结果」的可能；二是大规模训练往往只跑一次（single run），单次运行的表现更该用**高概率界**来刻画。高概率收敛界目前只有两篇：Li & Orabona (2020) 在 sub-Gaussian 噪声下给出平方梯度范数的 $\tilde O(1/\sqrt T)$，Cutkosky & Mehta (2021) 在 $\theta$ 阶矩条件下给出 $\tilde O(T^{-(\theta-1)/(3\theta-2)})$（且用的是带裁剪、归一化的动量变体，不是纯 Polyak 动量）。两者速率都偏慢，而且**都没有给任何泛化界**。

**核心矛盾**：泛化侧的困难更根本。证 SGD 泛化界的主流工具是**算法稳定性**（uniform stability），但 Ramezani-Kebrya et al. (2024) 构造了反例：即便损失是凸的，多轮 SGDM 的均匀稳定性间隙也可能**发散**。这意味着「稳定性 → 泛化」这条标准路径对 SGDM 基本走不通——快收敛与稳定性之间存在 trade-off，SGDM 偏向快收敛那一端。

**本文目标**：在非凸设定下，同时给出 SGDM 的高概率收敛界与高概率泛化界，并尽量覆盖比 sub-Gaussian 更现实的重尾噪声。

**切入角度**：作者绕开稳定性，改用**一致收敛**（uniform convergence）这条证明路线；同时用「逐级加码结构假设」的方式，把不同强度的假设映射到不同档位的速率，让读者看清「假设强一分、速率快一分」的全景。

**核心 idea**：在 sub-Weibull 重尾噪声下，沿「一般非凸 → PL → PL + Bernstein」三级结构，证出一条从 $\tilde O(1/\sqrt T)$ 收敛、$\tilde O(\sqrt{d/n})$ 泛化，一路改进到维度无关、低噪声下 $\tilde O(1/n^2)$ 泛化的界层级。

## 方法详解

### 整体框架

这是一篇纯理论论文，没有需要画 pipeline 的算法模块，核心是「问题设定 + 假设体系 + 一组主定理 + 证明技术」。整体逻辑是一条**结构假设递进、速率随之变快**的主线：把目标函数的结构从「一般非凸」逐步收紧到「满足 PL 条件」，再到「PL + 梯度满足 Bernstein 条件」，每收紧一级，就换出一档更快的收敛/泛化速率；同时全程把梯度噪声放宽到 sub-Weibull 重尾，并用一个尾参数 $\theta$ 统一刻画「噪声越重、速率越慢」的退化规律。

**问题设定**：参数空间 $\mathcal X \subseteq \mathbb R^d$，损失 $f(x;z)\ge 0$ 可能非凸。目标是最小化总体风险 $F(x)=\mathbb E_{z\sim P}[f(x;z)]$，但 $P$ 未知，实际只能优化由 $n$ 个 i.i.d. 样本构成的经验风险 $F_S(x)=\frac1n\sum_{i=1}^n f(x;z_i)$。算法就是 Algorithm 1 的 SGDM：每步均匀采一个样本 $j_t$，更新动量 $m_t=\gamma m_{t-1}+\eta_t\nabla f(x_t;z_{j_t})$，再令 $x_{t+1}=x_t-m_t$，等价于上面那条带 $\gamma(x_t-x_{t-1})$ 的展开式。注意这里 batch size 取 1，是动量相对 SGD 最难体现优势的「最坏情形」。

**关心的量**：一般非凸情形下无法保证找到全局极小，于是用**平均平方梯度范数**衡量——优化看 $\frac1T\sum_{t=1}^T\|\nabla F_S(x_t)\|^2$，泛化看 $\frac1T\sum_{t=1}^T\|\nabla F(x_t)\|^2$；进入 PL 情形后，可以直接谈**函数值误差**——优化看 $F_S(x_{T+1})-F_S(x(S))$，泛化看超额风险 $F(x_{T+1})-F(x^*)$，且这些界是对**最后一个迭代点**（last iterate）而非平均迭代点给出的，更贴近实际部署。

下面四个关键设计分别对应：噪声模型怎么放宽（设计 1）、三级结构如何换出三档速率（设计 2）、泛化界为什么改走一致收敛（设计 3）、以及如何借 Bernstein 把维度 $d$ 抹掉（设计 4）。

### 关键设计

**1. sub-Weibull 噪声模型：用一个尾参数统一刻画从轻尾到重尾**

现有 SGDM 高概率分析（Li & Orabona, 2020）只覆盖 sub-Gaussian 噪声，但越来越多证据表明随机梯度噪声常常比高斯更重尾。本文把噪声假设（Assumption 2.9）放宽为 sub-Weibull：要求梯度噪声 $e_t=\nabla f(x_t;z_{j_t})-\nabla F_S(x_t)$ 满足
$$\mathbb E_{j_t}\!\left[\exp\!\Big(\big(\|e_t\|/K\big)^{1/\theta}\Big)\right]\le 2,\qquad \theta\ge \tfrac12 .$$
这里 $\theta$ 就是尾参数：$\theta=1/2$ 退化为 sub-Gaussian，$\theta=1$ 对应 sub-exponential，$\theta>1$ 则是指数型重尾，$\theta$ 越大尾巴越重。这一设计的价值在于，它让所有主定理的速率都显式地写成 $\theta$ 的函数（典型形式是收敛界里出现 $\log^{2\theta}(1/\delta)$ 之类的因子），从而**定量地**回答了「噪声从轻尾走向重尾时，收敛和泛化会慢多少」这个一直含糊的问题——结论是 $\theta$ 越大速率越差，和直觉一致。

**2. 三级结构假设 → 三档收敛与泛化速率**

这是全文的骨架：通过对目标函数施加越来越强的曲率结构，换取越来越快的速率。

*第一级（一般非凸）*：只要 $L$-光滑（Assumption 2.1）+ sub-Weibull 噪声。Theorem 3.1 给出收敛界
$$\frac1T\sum_{t=1}^T\|\nabla F_S(x_t)\|^2=\tilde O\!\Big(\tfrac{1}{\sqrt T}\Big),$$
Theorem 3.3 在取迭代数 $T\asymp n/d$ 时给出泛化界 $\frac1T\sum_t\|\nabla F(x_t)\|^2=\tilde O\big(\sqrt{d/n}\big)$。这里 $\tilde O(1/\sqrt T)$ 已经触及 smooth 非凸一阶 oracle 的下界（Arjevani et al., 2019），相比 Li & Orabona (2020) 同假设下的结果，本文把 $\log(T/\delta)$ 因子精炼成 $\log(1/\delta)$，并且是**维度无关**的（对手的 delayed AdaGrad 变体则带 $d$）。

*第二级（PL 条件）*：再假设 $F_S$（及泛化时 $F$）满足 Polyak–Łojasiewicz 不等式
$$f(x)-f^*\le \tfrac{1}{2\mu}\|\nabla f(x)\|^2,$$
即梯度范数能反控函数值次优间隙——这是替代强凸性的最弱曲率条件之一，两层网络、矩阵补全、相位恢复、过参数化浅网络等都局部满足。配上递减步长 $\eta_t=\frac{1}{\mu(S)(t+t_0)}$，Theorem 3.5 把收敛率从 $\tilde O(1/\sqrt T)$ **加速到 $\tilde O(1/T)$**（且是 last-iterate、函数值误差），Theorem 3.7 把泛化（超额风险）从 $\tilde O(\sqrt{d/n})$ 改进到
$$F(x_{T+1})-F(x^*)=\tilde O\!\Big(\tfrac{d+\log(1/\delta)}{n}\Big),$$
对 $n$ 的依赖由 $1/\sqrt n$ 提升到 $1/n$。据作者所知，SGDM 在 PL 下的 $\tilde O(1/T)$ 高概率率此前未有。

*第三级（PL + Bernstein）*：见设计 4。三级叠起来就构成了从「慢但通用」到「快但需结构」的完整频谱，这种**层级化**呈现本身就是论文的组织亮点。

**3. 改走一致收敛而非算法稳定性来证泛化**

如动机所述，SGDM 的均匀稳定性会发散，靠稳定性证泛化此路不通。本文的泛化界改用**一致收敛**路线（Xu & Zeevi, 2020; Lei & Tang, 2021）：直接证「假设类里所有假设的经验风险一致收敛到其总体风险」，再把算法迭代点代进去。这条路绕开了稳定性对 SGDM 失效的难题，代价是一致收敛在一般非凸下天然带一个 $\sqrt d$ 的维度因子（Feldman, 2016），这正是 Theorem 3.3 里 $\sqrt{d/n}$ 中 $d$ 的来源。换句话说，作者用「能算出来的维度依赖」换掉了「根本算不出来的稳定性」——这是本文能首次给出 SGDM 泛化界的关键技术抉择。

**4. Bernstein 条件做局部化：抹掉维度 $d$，低噪声下冲到 $\tilde O(1/n^2)$**

一致收敛带来的 $\sqrt d$ 依赖在高维神经网络里是致命的。为去掉它，本文再加一条温和的 Bernstein 矩条件（Assumption 2.3）：在最优点 $x^*$ 处，梯度范数的高阶矩被二阶矩控制，
$$\mathbb E_z\big[\|\nabla f(x^*;z)\|^k\big]\le \tfrac12 k!\,\mathbb E_z\big[\|\nabla f(x^*;z)\|^2\big]B_*^{k-2},\quad k\ge 2 .$$
Bernstein 条件本质上等价于 sub-exponential，比「梯度一致有界」弱，是学习理论里的标准假设。它的作用是做**局部化**（localization）分析：把泛化误差锚定到 $x^*$ 附近的方差结构上，从而摆脱对整个假设类做一致覆盖时产生的维度项。在 $T\asymp n^2$、并满足一个温和的样本量下界后，Theorem 3.9 把超额风险改进到
$$F(x_{T+1})-F(x^*)=\tilde O\!\Big(\tfrac{F(x^*)}{n}+\tfrac{1}{n^2}\Big),$$
其中 $F(x^*)$ 是最优总体风险、通常极小。在**低噪声**机制（$F(x^*)=O(1/n)$，如插值/过参数化常见）下，这退化为 $\tilde O(1/n^2)$——学习理论里罕见的、对样本量二次依赖的快率；更关键的是这个界**不含维度 $d$**，因此能直接套用到高维大模型。这是据作者所知 SGDM 的首个高概率 $\tilde O(1/n^2)$ 泛化保证。

### 证明思路与关键引理

收敛侧的高概率界靠把动量更新展开后，将随机梯度噪声 $e_t$ 组织成一个**鞅差序列**，再对带 sub-Weibull 尾的向量值鞅施加集中不等式（Fan & Giraudo, 2019; Li, 2021 一类弱指数尾鞅集中）——尾参数 $\theta$ 正是在这一步进入最终速率，表现为 $\log^{2\theta}(1/\delta)$ 等因子。动量项的处理依赖把 $\eta_t=ct^{-1/2}$（或 PL 下的 $\frac{1}{\mu(S)(t+t_0)}$）这类步长与常数 $C_\gamma=1+\frac{2}{\ln^2\gamma}-\frac{3}{\ln\gamma}$（只依赖 $\gamma$）配合，控制惯性累积。泛化侧则如设计 3 走一致收敛、设计 4 在 PL+Bernstein 下做局部化。值得留意的是 Assumption 2.5 把有界梯度假设放宽成 $\eta_t\|\nabla F_S(x_t)\|\le G$——因为步长在衰减，这允许 $\|\nabla F_S(x_t)\|$ 随 $t$ 增长，仅在 $\theta>1/2$ 的重尾档才需要。

## 实验关键数据

本文是理论论文，数值实验只用来**佐证速率随尾参数 $\theta$ 的退化趋势**，不追求 SOTA。实验在 6 个 LIBSVM 数据集（Heart、Fourclass、German、Australian、Diabetes、Phishing）上做二分类，8:2 划分训练/测试，用测试集经验风险 $F_{S'}$ 作总体风险 $F$ 的代理；每步对每个坐标独立抽 sub-Weibull 噪声注入梯度，归一化为零均值单位方差，取 $\theta\in\{1/2,1,5\}$，重复 100 次取平均。

### 主结果：尾越重、泛化越差

| 设定 | 度量 | 观察到的趋势 | 与理论对应 |
|------|------|------|------|
| Huber 损失（图 1，6 数据集） | $\frac1T\sum_t\|\nabla F_{S'}(x_t)\|^2$ vs 训练轮数 | $\theta$ 越大曲线越高，$\theta=5$ 明显最差 | 印证 Theorem 3.3（$\theta>1$ 重尾退化） |
| 平方损失（图 2，6 数据集） | 同上 | 增大 $\theta$ 系统性变差 | 与 Theorem 3.3 一致 |

超参固定为 $\gamma=0.9$、$\eta_t=0.1\,t^{-1/2}$、Huber 阈值 $\tau=0.1$。

### 关键发现
- 两类损失、6 个数据集上，**平均平方梯度范数随 $\theta$ 单调变差**，且 $\theta=5$（重尾 $\theta>1$ 区）显著劣于 $\theta=1/2,1$，与理论预测的「尾参数进入速率」完全吻合。
- 实验取 batch size 1，正是动量相对 SGD 最难占优的最坏情形；这也呼应正文论断——在该机制下 SGDM 与 SGD 的高概率速率同阶（Kidambi et al., 2018 也观察到此时动量未必加速）。
- 论文未做不同优化器横向对比，因为目标只是验证「尾参数 → 速率」的定性关系，而非比谁更快。

## 亮点与洞察
- **「假设递进 ↔ 速率分档」的层级叙事**：把一般非凸、PL、PL+Bernstein 三级假设与 $\tilde O(1/\sqrt T)$/$\tilde O(1/T)$、$\tilde O(\sqrt{d/n})$/$\tilde O(1/n)$/$\tilde O(1/n^2)$ 几档速率一一对应，读者能一眼看清「每多一个结构假设买到什么」，是理论论文很值得学的组织方式。
- **「绕开稳定性」这一招很关键**：既然 SGDM 的均匀稳定性会发散，硬走稳定性注定失败；改用一致收敛虽引入 $\sqrt d$，却换来了 SGDM 史上首批泛化界——「换工具而非硬刚」的思路可迁移到其他稳定性失效的算法（如 Nesterov 加速）。
- **Bernstein 局部化抹掉维度**：$\tilde O(1/n^2)$ 且维度无关的泛化界对高维模型尤其友好，说明「最优点处的方差结构 + 低噪声」是拿到快率的现实抓手，这套局部化技巧可复用到其他过参数化场景。
- **sub-Weibull 把噪声从假设里「参数化」出来**：用单个 $\theta$ 贯穿所有定理，让「重尾代价」从模糊直觉变成可写进速率的显式因子，是处理重尾随机优化的干净范式。

## 局限与展望
- **batch size 固定为 1**：作者明确把大 batch 留作未来工作。正是在 batch=1 这一最坏机制下，本文只能证到「SGDM 与 SGD 同阶」，无法体现动量的理论加速；动量的好处可能恰恰依赖大 batch 带来的噪声缩减，这一块尚未刻画。
- **未真正分离 SGD 与 SGDM**：论文坦承在非凸随机设定下，SGD 与 SGDM 的清晰理论分离仍是 open problem——本文是「补齐 SGDM 不输 SGD」，而非「证明动量更优」。
- **PL/Bernstein 等结构假设的现实性**：快率（$\tilde O(1/T)$、$\tilde O(1/n^2)$）都依赖 PL 乃至低噪声 $F(x^*)=O(1/n)$ 这类较强条件，且 PL 参数 $\mu(S)$ 与 $\mu$ 需分别成立；这些条件在真实深网上多为局部、近似满足，界的实际紧度有待检验。
- **实验偏「验证趋势」**：只在小规模 LIBSVM + 人工注入噪声上验证 $\theta$ 的定性影响，没有在真实深度模型上量化界的紧度，也没有横向优化器对比。

## 相关工作与启发
- **vs Li & Orabona (2020)**：同样分析 Polyak 动量，但他们只覆盖 sub-Gaussian 噪声、给收敛率 $\tilde O(1/\sqrt T)$（带 $\log(T/\delta)$）、且无泛化界。本文把噪声推广到 sub-Weibull 重尾，把对数因子精炼到 $\log(1/\delta)$，并首次补上泛化界。
- **vs Cutkosky & Mehta (2021)**：他们用带梯度裁剪、归一化的动量变体，在 $\theta$ 阶矩下给 $\tilde O(T^{-(\theta-1)/(3\theta-2)})$（$\theta=2$ 时 $\tilde O(T^{-1/4})$）。本文分析的是**无裁剪的纯 Polyak 动量**，由 Jensen 不等式可推出同样的 $\tilde O(T^{-1/4})$ 速率，且额外给出泛化界。
- **vs Ramezani-Kebrya et al. (2024) / Chen et al. (2018)**：他们指出 SGDM 的均匀稳定性会发散、稳定性路线对 SGDM 失效。本文以此为出发点改走一致收敛，从而绕开稳定性证出泛化界。
- **vs Li & Liu (2022)（SGD 无动量）**：在相同假设下，本文 SGDM 的高概率收敛/泛化界与该工作 SGD 的结果同阶（差常数与对数因子），从而**闭合了 SGDM 相对 SGD 的理论缺口**——证明广泛使用的动量法在非凸/PL/Bernstein 假设下也享有与 SGD 同阶的高概率保证。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次给出 SGDM 的高概率泛化界，并在低噪声下证到罕见的维度无关 $\tilde O(1/n^2)$ 快率。
- 实验充分度: ⭐⭐⭐ 实验只为佐证 $\theta$ 趋势，规模小、无横向对比，但理论论文这样够用。
- 写作质量: ⭐⭐⭐⭐⭐ 三级假设—三档速率的层级叙事清晰，Remark 把与前作的对比讲得很透。
- 价值: ⭐⭐⭐⭐ 闭合了 SGDM 高概率分析的长期空白，对随机优化与学习理论方向有扎实参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] High-Probability Bounds for the Last Iterate of Clipped SGD](high-probability_bounds_for_the_last_iterate_of_clipped_sgd.md)
- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)
- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)

</div>

<!-- RELATED:END -->
