---
title: >-
  [论文解读] High-Probability Bounds for the Last Iterate of Clipped SGD
description: >-
  [ICLR 2026][优化/理论][Clipped-SGD] 本文首次证明了 Clipped-SGD 在凸光滑目标、重尾噪声（仅有限 $\alpha$ 阶矩，$\alpha\in(1,2]$）下**末次迭代**（last iterate）的高概率收敛速率，并给出一套把"高概率界"转换成"期望界"的通用技术。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "Clipped-SGD"
  - "重尾噪声"
  - "末次迭代"
  - "高概率收敛界"
  - "随机优化"
---

# High-Probability Bounds for the Last Iterate of Clipped SGD

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4sGEvpwyxN](https://openreview.net/forum?id=4sGEvpwyxN)  
**代码**: 未公开（纯理论 + 数值仿真）  
**领域**: optimization  
**关键词**: Clipped-SGD, 重尾噪声, 末次迭代, 高概率收敛界, 随机优化  

## 一句话总结
本文首次证明了 Clipped-SGD 在凸光滑目标、重尾噪声（仅有限 $\alpha$ 阶矩，$\alpha\in(1,2]$）下**末次迭代**（last iterate）的高概率收敛速率，并给出一套把"高概率界"转换成"期望界"的通用技术。

## 研究背景与动机
- **领域现状**：梯度裁剪（gradient clipping）已是训练 LLM 的标配组件，理论上它能对抗重尾噪声、提供高概率收敛保证。在凸问题上，Clipped-SGD 的平均迭代（average iterate）收敛理论已相当完善，可达最优速率 $\tilde{O}(1/K^{(\alpha-1)/\alpha})$。
- **现有痛点**：实践中真正部署的是**最后一个迭代点** $x_K$（没人会去存所有迭代再做平均），但凸光滑 + 重尾噪声下，Clipped-SGD 末次迭代的收敛性几乎是空白——已有的末次迭代结果都额外要求强凸、PL 条件、噪声对称性等更强假设。
- **核心矛盾**：平均迭代有漂亮理论但不是实际所用；末次迭代是实际所用但缺乏在重尾噪声下的高概率保证。两者之间隔着一道长期未填的理论缺口。
- **本文目标**：在仅 $\alpha$ 阶矩有界（$\alpha\in(1,2]$，涵盖方差无界的重尾情形）这一弱假设下，给出 Clipped-SGD 末次迭代的高概率收敛速率，且对置信参数 $\delta$ 只有 polylog 依赖、参数选择不需预知总步数 $K$。
- **核心 idea**：**用势函数（potential function）做高概率归纳分析** + **把裁剪阈值按 $1/\sqrt{d_k}$ 精细缩放** + **horizon-free 的失败预算 $\delta_k\sim 1/k^2$**，三者合力首次拿下末次迭代高概率界；再补一招"a.s. 有界更新 → 期望界"的转换技术。

## 方法详解

### 整体框架
研究对象是带时变步长 $\gamma_k$ 和裁剪阈值 $\lambda_k$ 的 Clipped-SGD：
$$x_{k+1}=x_k-\gamma_k\cdot\mathrm{clip}(\nabla f_{\xi_k}(x_k),\lambda_k),\quad \mathrm{clip}(x,\lambda):=\min\!\Big\{1,\tfrac{\lambda}{\|x\|}\Big\}x.$$
分析以一个 Lyapunov 势函数 $\Phi_k$ 为骨架，对其单步下降做高概率控制：把势函数变化拆成几个鞅型求和项，用 Freedman/Bernstein 不等式逐项卡住，再做并集界（union bound），归纳证明 $f(x_k)-f^*$ 以高概率按 $O(1/d_k)$ 衰减。整套分析由三个技术创新支撑，最后再补一个高概率→期望的转换器。

```mermaid
flowchart TD
    A[Clipped-SGD: 时变 γk, λk] --> B[势函数 Φk = dk·(f-f*) + L‖x-x*‖²]
    B --> C[高概率下降引理: 拆出鞅型求和]
    C --> D[Freedman/Bernstein + 时变失败预算 δt~1/t²]
    D --> E[归纳证明 f-f* = O(1/dk) 高概率成立]
    E --> F[λk ~ 1/√dk 精细缩放, 平衡裁剪偏差/方差]
    F --> G[Theorem 1: 末次迭代高概率速率]
    G --> H[4.2 节: a.s. 有界更新 → 期望界转换]
```

### 关键设计

**1. 势函数驱动的高概率归纳证明：把"平均迭代分析"升级到"末次迭代+高概率"。** 作者采用势函数 $\Phi_k=d_k(f(x_k)-f^*)+L\|x_k-x^*\|^2$，其中 $d_{k+1}=d_k+2\gamma_k L,\ d_0=0$。这个势函数源自 Bansal & Gupta 对梯度下降的分析，Taylor & Bach 用它推过末次迭代的期望界。本文的不同在于目标是**高概率**界，于是下降引理改写成
$$\Phi_K\le\Phi_0-\sum_{k}2\gamma_kL\langle x_k-x^*,\theta_k\rangle-\sum_k d_k\gamma_k\langle\nabla f(x_k),\theta_k\rangle+\sum_k(d_{k+1}+1)L\gamma_k^2\|\theta_k\|^2,$$
其中 $\theta_k:=g_k-\nabla f(x_k)$ 是裁剪后梯度 $g_k=\mathrm{clip}(\nabla f_{\xi_k}(x_k),\lambda_k)$ 的"误差"。右侧三个鞅型求和正是高概率分析的关键：用 Freedman/Bernstein 型集中不等式控制，配合**时变失败预算** $\delta_t\sim 1/t^2$ 再做并集界。势函数里 $f-f^*$ 被 $d_k$（随步长累加而增长）加权，正是这一加权把对**末次迭代**的控制变得可行。

**2. 裁剪阈值按 $1/\sqrt{d_k}$ 精细缩放：在重尾下更紧地平衡裁剪的偏差与方差。** 已有平均迭代分析通常取 $\lambda_k\propto 1/\gamma_k$，本文则用
$$\lambda_k=\frac{R_0}{80\gamma_k\ln^{1/2}\!\big(6(k+1)^2/\delta\big)}\cdot\min\!\Big\{\tfrac{1}{\sqrt{d_k}},1\Big\},$$
多出来的 $\min\{1/\sqrt{d_k},1\}$ 因子是点睛之笔。动机在于：既然归纳已证 $f(x_k)-f^*$ 高概率按 $O(1/d_k)$ 衰减，由光滑性 $\|\nabla f(x_k)\|\le\sqrt{2L(f(x_k)-f^*)}=O(1/\sqrt{d_k})$，即真实梯度本身在变小。让裁剪阈值同步收缩，就能把裁剪引入的**偏差项**（裁掉真信号）和**方差项**（重尾尾部）更精细地对冲，从而得到更紧的高概率界——这是末次迭代速率能成立的核心技术杠杆。

**3. Horizon-agnostic 调度 + $\delta_k\sim 1/k^2$ 失败预算：参数选择不需预知 $K$。** 步长 $\gamma_k$ 与阈值 $\lambda_k$ 都是 any-time 的，不依赖总步数 $K$，因此适用于流式/无限期训练这类无法用重启（restart）方案的场景。由于 horizon 未知，无法像有限步分析那样在每步取失败概率 $\delta/K$，作者改在第 $k$ 步用 $\delta/k^2$，借助 $\sum_k 1/k^2\le\pi^2/6$ 保证总失败概率仍受 $\delta$ 控制。代价是最终界里多出 $\sim\ln^2(6(K+1)^2/\delta)$ 这样的 polylog 因子，并在指数上引入参数 $\beta\ge(2+\alpha)/(3\alpha)$，取最优 $\beta=(2+\alpha)/(3\alpha)$ 时得到 Corollary 1 的末次迭代速率 $\tilde O\big(1/K^{(2\alpha-2)/(3\alpha)}\big)$。需要说明 $\gamma_k,\lambda_k$ 依赖 $\delta$，这是重尾下做归纳论证（需证迭代点高概率有界）所必需的，并非人为稳定动力学。

**4. 从高概率到期望的通用转换：用 a.s. 有界更新补上期望界。** 因为 $\gamma_k,\lambda_k$ 依赖 $\delta$，无法靠积分尾界直接得到期望保证。作者另辟蹊径：Clipped-SGD 的更新被裁剪 a.s. 有界，于是 $\|x_K-x^*\|\le R_0+\sum_k\gamma_k\lambda_k\le KR_0$，从而**以概率 1** 有 $f(x_K)-f^*\le \tfrac{L}{2}\|x_K-x^*\|^2\le \tfrac{L R_0^2 K^2}{2}$。把"高概率成立的好界 (6)"与"概率 1 成立的粗界 (7)"加权组合，取 $\delta=1/K^3$ 即得期望界（Corollary 2），在 $\alpha=2$ 时复现 Taylor & Bach 的结果（差 log 因子）。这一招对任意**更新 a.s. 有界**的方法都成立（normalized SGD、SignSGD、Muon 等），是超出 Clipped-SGD 本身的通用工具。

## 实验关键数据
论文为纯理论工作，实验仅用数值仿真验证"末次迭代优于平均迭代"，跨 1000 次重复报告 0.95 分位数与均值±标准差。

### 主实验设置与结论

| 实验 | 目标函数 | 噪声模型 | 维度 | 观察 |
|------|----------|----------|------|------|
| 损坏梯度 #1 | $f(x)=\ln(1+e^{\langle x,a\rangle})+\tfrac{\lambda}{2}\|x\|^2$ | Pareto，$\alpha\!\approx\!2$（>2.001 阶矩无穷） | $d=100$ | 末次迭代优于平均 |
| 损坏梯度 #2 | $f(x)=\tfrac12\|x\|^2$ | 同上 | $d=100$ | 末次迭代优于平均 |
| 统计学习 | logistic 损失期望 $\mathbb{E}[\ln(1+e^{-Y\langle x,Z\rangle})]$ | Student-t（2.001 自由度，高阶矩无穷） | $d=10$ | 末次迭代优于平均 |

### 收敛速率对比（核心理论结果）

| 来源 | 收敛类型 | 迭代 | 噪声假设 | 速率 |
|------|----------|------|----------|------|
| Liu & Zhou (2024) | 期望 | 末次 | As.4, $\alpha\in(1,2]$ | $O\big(\tfrac{LR_0^2}{K^{2(\alpha-1)/\alpha}}+\tfrac{R_0\sigma}{K^{(\alpha-1)/\alpha}}\big)$ |
| Nguyen et al. (2023) | 高概率 | 平均 | As.4, $\alpha\in(1,2]$ | $\tilde O\big(\tfrac{LR_0^2}{K}+\tfrac{R_0\sigma}{K^{(\alpha-1)/\alpha}}\big)$ |
| **本文** | **高概率** | **末次** | As.4, $\alpha\in(1,2]$ | $\tilde O\big(\tfrac{LR_0^2}{K}+\tfrac{D}{K^{2(\alpha-1)/3\alpha}}\big)$ |

其中 $D:=\max\{R_0\sigma,\ L^{(\alpha-1)/(3\alpha-1)}R_0^{(4\alpha-2)/(3\alpha-1)}\sigma^{2\alpha/(3\alpha-1)},\ L^{1/3}R_0^{4/3}\sigma^{2/3}\}$。

### 关键发现
- 末次迭代在所有三个仿真中都优于平均迭代，且评估对两者用的是**同一套（为末次迭代优化的）调度**，对平均迭代其实是有利对照而非刻意压制。
- 当 $\alpha\to 1$ 时结果只收敛到有限邻域，与 Lipschitz 凸情形的下界一致——这是重尾极限下的本质限制而非证明松弛。
- 期望界（Corollary 2，取 $\delta=1/K^3$）在 $\alpha=2$ 时复现 Taylor & Bach (2019) 的 $O(1/K^{1/3})$ 末次迭代速率（差 log 因子），验证转换技术不会"虚假地"靠 $\delta$ 依赖参数稳定动力学。
- 噪声构造刻意使所有 >2.001 阶矩发散（Pareto / Student-t），精确对应假设 4 在 $\alpha=2$ 临界处的最难情形，说明优势在真正重尾下也成立。

## 亮点与洞察
- **填补长期空白**：首次给出凸光滑 + 重尾噪声下 Clipped-SGD 末次迭代的高概率界，把"理论分析的对象"和"实践真正用的迭代"对齐。
- **$\lambda_k\sim 1/\sqrt{d_k}$ 的洞见很漂亮**：把"梯度随收敛而变小"这一动态事实反馈进裁剪阈值设计，是偏差-方差权衡的精细化，思路可迁移到其他裁剪/归一化方法。
- **高概率→期望转换器是真正的通用件**：只要更新 a.s. 有界即可套用，对 SignSGD、normalized SGD with momentum、Muon 等一大类方法都成立，价值超出本文具体算法。
- **horizon-free**：参数选择不需预知 $K$，天然适配流式/无限期训练，工程上更友好。
- **单次运行可信**：高概率界直接约束单次训练失败的概率，比只刻画"平均行为"的期望界对实践更有指导意义。

## 局限与展望
- **速率仍非最优**：$\alpha=2$ 时末次迭代高概率速率与最优期望速率 $1/\sqrt K$ 间存在多项式 gap（指数 $2(\alpha-1)/3\alpha$ 而非 $(\alpha-1)/\alpha$），是否能闭合仍是开问题。
- **参数依赖 $\delta$**：高概率结果中 $\gamma_k,\lambda_k$ 依赖失败概率 $\delta$，这是重尾归纳论证所需；要在 $\delta$-无关参数下拿到同样末次迭代界仍待解决。
- **期望界牺牲了 horizon-free**：Corollary 2 取 $\delta=1/K^3$ 后参数变成 horizon-dependent；在 horizon-agnostic 参数下达到同样期望速率是明确列出的 open question。
- **仅凸 + 光滑**：未覆盖非凸、广义光滑等更贴近深度学习的设置。
- **实验规模有限**：仅在 $d\le100$ 的合成凸目标上验证，未在真实神经网络训练上检验末次迭代优势是否依然显著。

## 相关工作与启发
- **平均迭代重尾理论**：Gorbunov et al. (2020)、Sadiev et al. (2023)、Nguyen et al. (2023)、Parletta et al. (2024/2025) 等证明裁剪让平均迭代达 $\tilde O(1/K^{(\alpha-1)/\alpha})$；本文把战线推进到末次迭代。
- **末次迭代理论**：Shamir & Zhang (2013)、Jain et al. (2021)、Liu & Zhou (2024) 在更强噪声假设下做末次迭代；Sadiev et al. (2023) 在重尾下需强凸/PL，本文去掉了这些额外结构假设。
- **更弱噪声谱系**：Eldowa & Paudice (2024)、Madden et al. (2024) 用 sub-Weibull 尾放宽 sub-Gaussian，但仍蕴含所有阶矩存在；本文的 $\alpha\in(1,2]$ 模型允许方差无界，是更激进的重尾设定。
- **势函数技术**：Bansal & Gupta (2017) 的 GD 势函数、Taylor & Bach (2019) 的末次迭代期望分析，是本文势函数的来源，本文将其高概率化。
- **启发**：把"收敛进度反馈进超参（此处是裁剪阈值）"的思路，以及"a.s. 有界更新 ⇒ 高概率界换期望界"的转换范式，都是可复用到其他随机优化方法的通用工具；后者已被作者点名可套到 normalized SGD with momentum、SignSGD、Muon 等带界更新的算法上。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次证明凸光滑 + 重尾下 Clipped-SGD 末次迭代高概率界，填补长期空白，$\lambda_k\sim1/\sqrt{d_k}$ 与高概率→期望转换都是新工具。
- 实验充分度: ⭐⭐⭐ 纯理论工作，仅数值仿真验证末次>平均，规模小但与定位匹配。
- 写作质量: ⭐⭐⭐⭐ 贡献清晰、证明草图三条技术创新讲得明白、与 SOTA 对比表完整。
- 价值: ⭐⭐⭐⭐ 对齐"理论分析对象"与"实践所用迭代"，转换技术可迁移至 SignSGD/Muon 等一大类方法。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)
- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)
- [\[NeurIPS 2025\] From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications](../../NeurIPS2025/optimization/from_average-iterate_to_last-iterate_convergence_in_games_a_reduction_and_its_ap.md)
- [\[ICLR 2026\] Birch SGD: A Tree Graph Framework for Local and Asynchronous SGD Methods](birch_sgd_a_tree_graph_framework_for_local_and_asynchronous_sgd_methods.md)

</div>

<!-- RELATED:END -->
