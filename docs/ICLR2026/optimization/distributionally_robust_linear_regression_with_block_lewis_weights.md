---
title: >-
  [论文解读] Distributionally Robust Linear Regression with Block Lewis Weights
description: >-
  [ICLR 2026][优化/理论][群体分布鲁棒优化] 本文为"经验群体分布鲁棒最小二乘"（即 min-max 群体回归）设计了一个迭代复杂度仅为 $\tilde{O}(\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3})$ 的二阶算法，关键是用 **block Lewis weights** 构造一个最优逼近椭球几何，配合加速近端方法，使迭代次数几乎不依赖群体数 $m$。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "群体分布鲁棒优化"
  - "最小最大回归"
  - "Block Lewis Weights"
  - "加速近端方法"
  - "准自和谐"
  - "ℓ∞/ℓp 回归"
---

# Distributionally Robust Linear Regression with Block Lewis Weights

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0r1KU3dlps](https://openreview.net/forum?id=0r1KU3dlps)  
**代码**: 论文附带 Jupyter notebook（合成实验）  
**领域**: optimization  
**关键词**: 群体分布鲁棒优化, 最小最大回归, Block Lewis Weights, 加速近端方法, 准自和谐, ℓ∞/ℓp 回归  

## 一句话总结
本文为"经验群体分布鲁棒最小二乘"（即 min-max 群体回归）设计了一个迭代复杂度仅为 $\tilde{O}(\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3})$ 的二阶算法，关键是用 **block Lewis weights** 构造一个最优逼近椭球几何，配合加速近端方法，使迭代次数几乎不依赖群体数 $m$。

## 研究背景与动机

**领域现状**：在房价预测、跨行业工资估计、贷款额度预测等需要可解释模型的场景里，人们常对 $m$ 个群体（数据源）拟合同一个线性模型 $x\in\mathbb{R}^d$。最直接的"功利主义"目标是最小化各组平方误差的平均（objective 1），但由于群体异质性，这种平均化会偏向主流数据源，对少数群体给出明显更差的预测。为此，近年大量工作转向"平等主义"的群体 DRO 目标：

$$\min_{x\in\mathbb{R}^d}\ \max_{i\in[m]}\ \tfrac{1}{n_i}\|A_{S_i}x-b_{S_i}\|_2^2 .$$

**现有痛点**：这个目标虽然凸，但各类标准方法都不理想。① 一阶方法（次梯度、Abernethy 等）的收敛率**依赖几何条件数**——当堆叠矩阵 $A$ 病态时收敛极慢；② 目标是若干范数的逐点最大值，**不光滑**，朴素次梯度只能拿到 $1/\varepsilon^2$ 的迭代复杂度；③ 把它当 min-max 博弈用 no-regret 方法做，光滑在线凸优化仍是 $1/\varepsilon^2$，并不更好；④ 用内点法（IPM）改写成 epigraph 形式，迭代复杂度是 $O(\sqrt{m})$，群体数大时超线性，而且把每个组复制 $k$ 份后值不变、复杂度却暴涨到 $\sqrt{mk}$。

**核心矛盾**：理想算法既要**无几何依赖**（不被条件数拖累），又要让迭代次数**几乎独立于 $m$**（最好是 $\min\{\mathrm{rank}(A),m\}$ 的低次幂），同时每步只解一个结构化线性系统。已有方法没有一个能同时满足。

**本文目标**：给出一个二阶算法，用 $\tilde{O}(\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3})$ 次"形如 $A^\top BA$（$B$ 块对角）的线性系统求解"逼近 $(1+\varepsilon)$ 最优解，并在 $n_i=1$ 退化为 $\ell_\infty$ 回归时不劣于已知最优结果。

**核心 idea**：**几何即一切**——把 min-max 目标看成 $\mathbb{R}^d$ 上的一个范数，用 **block Lewis weights** 算出最优逼近的椭球范数 $\|\cdot\|_M$，再在这个"对的几何"里跑加速近端（ball-oracle）方法，让 $\|x_0-x^\star\|_M$ 被 $\sqrt{d}$ 界住，从而把 $m$ 依赖换成 $\min\{\mathrm{rank}(A),m\}$。

## 方法详解

### 整体框架
算法把困难的 min-max 问题拆成"在某个椭球几何 $\|\cdot\|_M$ 下反复求解近端子问题"。三个环节缺一不可：(1) **快速解子问题**——在 $\|\cdot\|_M$ 球内把不光滑/高阶的目标变成 Hessian 稳定、可用阻尼牛顿求解的形式；(2) **加速迭代子问题**——用 Monteiro–Svaiter 型加速框架把子问题解串成最终解，得到 $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$ 的外层速率；(3) **选对几何 $M$**——用 block Lewis weights 找到使距离 $\|x_0-x^\star\|_M\lesssim\sqrt{d}$ 的椭球，这一步才把 $m$ 依赖压成 $\min\{\mathrm{rank}(A),m\}^{1/3}$。

```mermaid
flowchart LR
    A[min-max 群体回归<br/>不光滑/几何依赖] --> B[选几何 M<br/>block Lewis weights]
    B --> C[近端子问题 O&#40;q&#41;<br/>在 M-球内]
    C --> D{p=∞ 还是 2≤p<∞}
    D -->|p=∞| E[log-sum-exp 平滑<br/>+ 准自和谐 + 阻尼牛顿]
    D -->|有限 p| F[ℓp 近端正则<br/>+ 强凸 + 镜像下降]
    E --> G[Monteiro–Svaiter 加速<br/>串联子问题解]
    F --> G
    G --> H[(1+ε) 最优解<br/>Õ&#40;min&#123;rank,m&#125;^1/3 ε^-2/3&#41;]
```

### 关键设计

**1. 用 block Lewis weights 选"对的椭球几何"，把 $m$ 换成 $\mathrm{rank}(A)$。** 这是全文的几何核心。外层加速速率正比于 $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$，所以问题归结为：选一个椭球范数 $\|\cdot\|_M$，让它以尽量小的失真 $\triangle\ge 1$ 夹住真正的损失范数 $\|y\|_{G_p}=(\sum_i\|y_{S_i}\|_2^p)^{1/p}$，即 $\|x-b\|_M\le \big(\sum_i\|A_{S_i}x-b_{S_i}\|_2^p\big)^{1/p}\le \triangle\,\|x-b\|_M$。若取最朴素的 $M=A^\top A$，靠 $\ell_2$ 与 $\ell_p$ 的范数等价只能得到失真 $m^{1/2-1/p}$，于是速率退化成 $m^{1/3}$。改进的关键在于：损失本身是 $\mathbb{R}^d$ 上的一个**范数**，其单位球是对称凸体，由 **John 定理**存在内接最大体积椭球使失真 $\triangle\le\sqrt{d}$，从而 $\|x_0-x^\star\|_M\lesssim\sqrt{d}$。把它代回即得 $\min\{\mathrm{rank}(A),m\}^{1/3}$（再按 $m\lessgtr d$ 取小者）。但 John 椭球只是存在性，论文调用 **Manoj & Ovsiankin (2025) 的 block Lewis weights 算法**真正算出这个对角加权 $W$，保证 $\|W^{1/2-1/p}(Ax-cb)\|_2$ 与 $G_p$ 损失在 $(2(\mathrm{rank}(A)+1))^{1/2-1/p}$ 因子内等价——这把"存在好几何"落地成"可计算的好几何"。

**2. $p=\infty$ 鲁棒情形：log-sum-exp 平滑 + 准自和谐，使不可微目标变成 Hessian 稳定。** 由于 $f$ 是若干欧氏范数的逐点最大值，处处不可微，无法直接谈 $\nabla^2 f$ 的稳定性。作者引入双参数光滑代理

$$\tilde f_{\beta,\delta}(x)=\beta\log\!\Big(\sum_{i=1}^m\exp\tfrac{\sqrt{\delta^2+\|A_{S_i}x-b_{S_i}\|_2^2}-\delta}{\beta}\Big),$$

即"带温度 $\beta$ 的 softmax"复合"内函数 $\sqrt{\delta^2+\|\cdot\|_2^2}-\delta$"。可证 $|\tilde f_{\beta,\delta}-f|\le\beta\log m+\delta$，取 $\beta=\varepsilon/4\log m,\ \delta=\varepsilon/4$ 即足够。难点是要让代理满足 **准自和谐**（quasi-self-concordance，$|\frac{d^3}{dt^3}f|\le\nu\|d\|\frac{d^2}{dt^2}f$），这是 Carmon 等人的二阶子问题求解器对 Hessian 稳定的充分条件。已有结果只知道"softmax 复合线性函数"是准自和谐；本文证明了一个更一般、可能独立有用的**复合定理**（Lemma C.3）：softmax 复合任意一族准自和谐的内函数仍准自和谐。再证内函数 $\sqrt{\delta^2+\|A_{S_i}x\|_2^2}-\delta$ 各自 $O(1/\delta)$-准自和谐，组合后 $\tilde f_{\beta,\delta}$ 在 $\max_i\|A_{S_i}x\|_2$ 范数下 $O(1/\beta+1/\delta)$-准自和谐，于是固定半径 $r_q=\Theta(1/\varepsilon)$ 的近端子问题就能用阻尼牛顿快速求解。

**3. $2\le p<\infty$ 插值情形：ℓp 近端正则 + 范数强凸，拿到加速所需的近似驻点。** 对插值目标 $\min_x\frac{1}{m}\sum_i(\frac{1}{n_i}\|A_{S_i}x-b_{S_i}\|_2^2)^{p/2}$（$p=2$ 是平均最小二乘，$p\to\infty$ 回到鲁棒目标），子问题不再硬约束半径，而是在 $\|\cdot\|_M$ 下正则化移动量：$\arg\min_x f(x)+\tilde p^p\|x-q\|_M^p$。这推广了 Jambulapati 等人的 $\ell_p$ 回归近端问题，但本文更进一步——加速框架要求子问题返回**近似驻点**而非仅小函数值。为此作者证明了关于 $\|y\|_2^p$ 的**强凸引理**（Lemma D.3）：$\|v+\triangle\|_2^p\ge\|v\|_2^p+p\|v\|_2^{p-2}\langle v,\triangle\rangle+4^{-2p}\|\triangle\|_2^p$，由它得到 $\|x-q\|_M^p$ 的强凸，从而把"函数值近优"转成"参数空间近优"；再配合该目标的局部梯度 Lipschitz（源于 Hessian 稳定），即得近似驻点。最终函数值的近优则靠"相对光滑+相对强凸于一个简单参考函数"，用一个对标准镜像下降做轻微改造的求解器（处理非精确子问题求解）完成。

**4. Monteiro–Svaiter 型加速串联子问题，达到 $\varepsilon^{-2/3}$ 与高精度。** 朴素地反复调用近端 oracle $O(q)$ 只能给 $\|x_0-x^\star\|_M/\varepsilon$。本文采用 Monteiro–Svaiter (2013) 起源、经 Carmon 等refine 的加速方案，把外层速率压到 $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$，正好对应 Theorem 1 的形状。对有限 $p$，再用强凸引理证明若干次迭代后 $\|x_t-x^\star\|_M\le 0.5\|x_0-x^\star\|_M$，反复减半即得**高精度**（多对数 $1/\varepsilon$）解，对应 Theorem 2。值得注意的是有限 $p$ 用的是 Carmon 2022 那版**无需解隐式方程**求查询点的加速，因此在 $\ell_p$ 回归特例上还顺带改进了 Jambulapati 等人的结果。

## 实验关键数据

实验为合成的"群体异质回归"构造（默认 $d=10$、$100$ 组、其中 $5$ 组对抗，堆叠 Gram 矩阵条件数约 $10^5$），多数组几何对齐、少数组在某方向曲率极大且最优点远离群体中心，刻意制造"平均损失最优 $\ne$ 最差组损失最优"。鲁棒最优值用 CVXPY 以 epigraph 凸规划精确求得，所有曲线汇报与该最优值的差距。

### 主实验（迭代复杂度，定性结论）

| 方法 | 在对抗实例上的迭代行为 |
|------|------------------------|
| 次梯度法 | 短暂下降后迅速停滞，远离最优 |
| 平滑一阶（GD / Heavy-Ball / Nesterov） | 即便最优调参也停滞在最优值上方 |
| Ball-oracle（欧氏几何，朴素球） | 随外层迭代稳定下降最差组损失 |
| Ball-oracle（Lewis-weight 几何，本文） | 随外层迭代稳定下降，且优于朴素球 |
| 内点法 IPM | 收敛最快、最终损失最好（因 CVXPY 本就用同款算法算最优值） |

### 消融实验（几何 / 加速的作用，定性 + 理论对照）

| 变体 | 失真 $\triangle$ | 迭代复杂度（$p=\infty$） | 说明 |
|------|------------------|--------------------------|------|
| 朴素几何 $M=A^\top A$ | $m^{1/2-1/p}$ | $m^{1/3}\varepsilon^{-2/3}$ | 仅靠 $\ell_2$–$\ell_p$ 范数等价 |
| **block Lewis 几何（本文）** | $\le\sqrt{d}$（John 椭球） | $\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3}$ | 真正算出最优逼近椭球 |
| 朴素近端迭代（无加速） | — | $\|x_0-x^\star\|_M/\varepsilon$ | 线性 $1/\varepsilon$ |
| **+ Monteiro–Svaiter 加速** | — | $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$ | 三分之二次幂，有限 $p$ 还能高精度 |

消融对照表明：把几何从 $A^\top A$ 换成 block Lewis 几何，把 $m$ 依赖换成 $\min\{\mathrm{rank}(A),m\}$；叠加加速框架再把 $1/\varepsilon$ 压到 $\varepsilon^{-2/3}$，两者正交且各自必要。

### 关键发现
- **一阶方法被几何拖死**：因多数组共享几何，梯度平均隐式强调它们的曲率，导致对抗组所在的"尖锐方向"进展极慢——正是理论"无几何依赖"动机的实证。
- **Lewis 几何 > 朴素球**：用 block Lewis weights 定义信赖域的 ball-oracle 比欧氏球收敛更快，验证了"选对几何"的价值。
- **IPM 在小 $m$ 下仍强**：最终精度最好，但其优势随 $m$ 增大会被 $O(\sqrt{m})$ 迭代复杂度侵蚀——这正是本文 $m^{1/3}$ 理论的适用区间（$\varepsilon\ge m^{-1/4}$）。
- **运行时观察**：更贵的一阶方法在 wall-clock 上初期看似有竞争力，但因牛顿步本身昂贵而后期被反超；论文未深度优化代码，仅作对照。

## 亮点与洞察
- **把"鲁棒回归"翻译成"几何选择"问题**：核心洞见是 min-max 损失是一个范数，其单位球的最优逼近椭球（John 椭球 / Lewis weights）直接决定加速速率，从而把迭代复杂度从 $\sqrt{m}$（IPM）或 $1/\varepsilon^2$（一阶）降到 $\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3}$。
- **复制不变性是好的诊断信号**：作者用"复制每组 $k$ 份目标值不变但 IPM 复杂度涨到 $\sqrt{mk}$"论证迭代复杂度应独立于 $m$，这个反例很有说服力。
- **可能独立有用的理论工具**：softmax 复合准自和谐函数仍准自和谐（Lemma C.3）、以及 $\|y\|_2^p$ 的强凸引理（Lemma D.3），都超出本问题范围，可服务于更广的二阶优化分析。
- **统一框架覆盖 $\ell_\infty$ 与 $\ell_p$**：$n_i=1$ 时精确退化为 Jambulapati 等人的 $\ell_\infty/\ell_p$ 回归保证，且对 $p$ 是无条件多项式（对方需轻微数值稳定性假设）。

## 局限与展望
- **非高精度（针对 Theorem 1）**：鲁棒目标 $p=\infty$ 的 $\varepsilon$ 依赖是 $\varepsilon^{-2/3}$ 而非 $\mathrm{polylog}(1/\varepsilon)$；能否在 Theorem 2 同等条件下做出高精度求解器是公开问题。
- **改进区间有限**：相对 log-barrier IPM 的改进只在 $\varepsilon\ge m^{-1/4}$ 的中等精度区间成立，且要 $m\gg\mathrm{rank}(A)$ 才明显；高精度下 IPM 仍占优。
- **实验仅合成**：评测停留在 $d=10$、$100$ 组的合成构造，未在真实公平/多源数据集上验证，代码也未做时间复杂度的精细优化。
- **更宏大目标未达**：作者视本文为"一般凸二次规划取得 $\min\{\mathrm{rank},m\}^{1/3}$ 复杂度"的第一步；推广到通用 QP、考虑 $\mathrm{rank}(A)$ 之外的复杂度度量、刻画 $p$ 变化时鲁棒-效用的统计权衡，都留作未来工作。

## 相关工作与启发
- **Lewis weights / John 椭球谱系**：直接站在 Lee & Sidford (2019)、Jambulapati 等 (2022) 的"几何理解给出维度更小的迭代复杂度"思路上，把 $\ell_p$/线性规划的 Lewis weights 推广为处理群体块结构的 block Lewis weights（Manoj & Ovsiankin 2025）。
- **群体 DRO / 算法公平**：对应 Sagawa、Duchi、Mohri 等的群体分布鲁棒目标，是"egalitarian"公平的代表；插值目标 $p$ 则连通多目标优化的 scalarization 与 Pareto 前沿思想。
- **加速近端框架**：复用 Monteiro–Svaiter (2013) 与 Carmon 等 (2020/2022) 的 ball-oracle 加速，并指出有限 $p$ 用"无隐式方程"版本可顺带改进 $\ell_p$ 回归。
- **启发**：当一个鲁棒/min-max 目标可写成范数时，"先花预算算最优逼近椭球、再在该几何里跑二阶法"是把 $m$ 依赖压成 $\mathrm{rank}$ 依赖的可复制套路；softmax 与准自和谐的复合性质也提示了平滑 max 型目标的统一处理方式。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首次把 block Lewis weights 几何引入群体 DRO 回归，得到迭代复杂度几乎独立于群体数 $m$ 的二阶算法，并附带两条独立有用的引理。
- **实验充分度**: ⭐⭐⭐ 合成构造刻意且解释清晰，定性结论与理论吻合，但缺真实数据与规模化验证，代码未做性能优化。
- **写作质量**: ⭐⭐⭐⭐ 动机层层递进（逐条否定一阶/博弈/IPM），技术综述清楚，三大环节（解子问题/加速/选几何）组织得当。
- **价值**: ⭐⭐⭐⭐ 为公平/多源线性回归提供了无几何依赖且 $m$-无关的高效算法，并为"通用凸二次规划取得 rank 级复杂度"指明第一步。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Distributionally Robust Optimization via Generative Ambiguity Modeling](distributionally_robust_optimization_via_generative_ambiguity_modeling.md)
- [\[ICLR 2026\] Byzantine-Robust Federated Learning with Learnable Aggregation Weights](byzantine-robust_federated_learning_with_learnable_aggregation_weights.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/optimization/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ICLR 2026\] MuonBP: Faster Muon via Block-Periodic Orthogonalization](muonbp_faster_muon_via_block-periodic_orthogonalization.md)

</div>

<!-- RELATED:END -->
