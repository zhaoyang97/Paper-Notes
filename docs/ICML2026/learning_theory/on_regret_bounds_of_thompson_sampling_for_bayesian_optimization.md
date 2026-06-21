---
title: >-
  [论文解读] On Regret Bounds of Thompson Sampling for Bayesian Optimization
description: >-
  [ICML 2026][学习理论][高斯过程] 这篇论文系统补齐了高斯过程 Thompson 采样（GP-TS）在贝叶斯设定下的遗憾分析：先构造了一个反例证明 GP-TS 对失败概率 $\delta$ 只能做到多项式依赖（拿不到 $\log(1/\delta)$），再给出累积遗憾二阶矩上界把 $\delta$ 依赖收紧 $1/\sqrt{\delta}$ 倍、首个期望宽容遗憾的多对数上界、以及在更宽松 Matérn 条件下的 $\tilde O(\sqrt T)$ 高概率遗憾上界，让 GP-TS 在理论保证上基本追平了被研究得最透的 GP-UCB。
tags:
  - "ICML 2026"
  - "学习理论"
  - "贝叶斯优化"
  - "多臂老虎机"
  - "高斯过程"
  - "Thompson 采样"
  - "累积遗憾"
  - "宽容遗憾"
  - "信息增益"
---

# On Regret Bounds of Thompson Sampling for Bayesian Optimization

**会议**: ICML 2026  
**arXiv**: [2603.09276](https://arxiv.org/abs/2603.09276)  
**代码**: 无（纯理论）  
**领域**: 学习理论 / 贝叶斯优化 / 多臂老虎机  
**关键词**: 高斯过程, Thompson 采样, 累积遗憾, 宽容遗憾, 信息增益

## 一句话总结
这篇论文系统补齐了高斯过程 Thompson 采样（GP-TS）在贝叶斯设定下的遗憾分析：先构造了一个反例证明 GP-TS 对失败概率 $\delta$ 只能做到多项式依赖（拿不到 $\log(1/\delta)$），再给出累积遗憾二阶矩上界把 $\delta$ 依赖收紧 $1/\sqrt{\delta}$ 倍、首个期望宽容遗憾的多对数上界、以及在更宽松 Matérn 条件下的 $\tilde O(\sqrt T)$ 高概率遗憾上界，让 GP-TS 在理论保证上基本追平了被研究得最透的 GP-UCB。

## 研究背景与动机
**领域现状**：贝叶斯优化（BO）是黑盒、昂贵函数优化的主流框架，核心是用高斯过程（GP）当代理模型、按采集函数序贯选点。理论上人们关心**累积遗憾** $R_T=\sum_{t=1}^T f(\boldsymbol x^*)-f(\boldsymbol x_t)$，只要它 $o(T)$ 次线性，就保证算法收敛到最优。两条最主流的算法是 GP-UCB（确定性置信上界）和 GP-TS（每轮从后验里采一条样本路径、取其最大值点）。

**现有痛点**：GP-UCB 被分析得非常透——既有高概率上界、又有期望上界，还有针对宽容遗憾（lenient regret）和更紧的累积遗憾上界。而 GP-TS 虽然实践效果常常更好（因为它不依赖 GP-UCB 那个需要精调、且实践中没保证的置信宽度参数），但理论分析明显落后：它的高概率遗憾上界只能由「期望上界 + Markov 不等式」直接推出，导致对 $\delta$ 的依赖是 $1/\delta$（GP-UCB 是 $\log(1/\delta)$）；而宽容遗憾上界、更紧的累积遗憾上界对 GP-TS 都是空白，Iwazaki (2025b) 甚至把它明确列为公开问题。

**核心矛盾**：GP-TS 的随机性（每轮采一条后验样本）让 $\boldsymbol x_t$ 和 $\boldsymbol x^*$ 在给定历史后同分布且条件独立，这个性质是 GP-TS 期望遗憾分析的基石；但它也让「高概率」分析变难——直接走 Markov 就只能换来 $1/\delta$ 这种糟糕依赖。问题是：GP-TS 对 $\delta$ 的差依赖到底是**证明技术不够好**，还是**算法本身的硬限制**？

**本文目标**：把 GP-UCB 和 GP-TS 之间的分析鸿沟补上，具体拆成四个子问题——(i) GP-TS 能否做到 $\log(1/\delta)$ 依赖？(ii) 能否收紧 $1/\delta$ 依赖？(iii) 能否给出宽容遗憾上界？(iv) 能否把对时间步 $T$ 的依赖收紧到 $\tilde O(\sqrt T)$？

**核心 idea**：对 (i) 用一个两臂反例给出**否定回答**（这是硬限制，不是技术问题）；对 (ii)(iii)(iv) 则发展新的证明工具（二阶矩分解、椭圆势计数式的宽容遗憾证明、放宽 Matérn 条件的精细分析）逐一给出**肯定回答**。

## 方法详解

### 整体框架
本文不是提出新算法，而是围绕**同一个 GP-TS 算法**（算法 1：每轮更新 GP 后验 → 采一条样本路径 $g_t\sim p(f\mid\mathcal D_{t-1})$ → 取 $\boldsymbol x_t=\arg\max_{\boldsymbol x} g_t(\boldsymbol x)$ → 观测带噪 $y_t$）给出四组遗憾结果。所有分析都建立在贝叶斯设定下：目标函数 $f\sim\mathcal{GP}(0,k)$ 是 GP 的一条样本路径（区别于 frequentist 设定假设 $f$ 在某 RKHS 内），观测被高斯噪声污染 $y_i=f(\boldsymbol x_i)+\epsilon_i$。复杂度由**最大信息增益** $\gamma_T:=\max_A I(\boldsymbol y_A;\boldsymbol f_A)$ 刻画，对常用核它都是 $T$ 的次线性量（SE 核 $O((\log T)^{d+1})$、Matérn-$\nu$ 核 $O(T^{d/(2\nu+d)}\cdots)$）。

四个结果之间有清晰的逻辑递进：**反例（Thm 3.1）** 划出 GP-TS 在 $\delta$ 上能力的天花板；**二阶矩（Thm 3.2）** 在这个天花板下把 $\delta$ 依赖尽量收紧；**宽容遗憾（Thm 3.3）** 是一个独立贡献，同时又是把 $T$ 依赖收紧的关键引理；**改进的 $T$ 依赖（Lemma 3.4 + Thm 3.5）** 借用宽容遗憾结果 + 放宽 Matérn 条件的精细分析，得到 $\tilde O(\sqrt T)$ 高概率上界。

### 关键设计

**1. 两臂反例：证明 GP-TS 对 $\delta$ 的多项式依赖是硬限制**

针对「GP-TS 对 $\delta$ 的差依赖是不是只是证明松」这个疑问，作者直接构造反例否掉。取最简单的两臂问题 $\mathcal X=\{\boldsymbol x^{(1)},\boldsymbol x^{(2)}\}$，两臂的函数值服从相关高斯先验 $\big(f(\boldsymbol x^{(1)}),f(\boldsymbol x^{(2)})\big)\sim\mathcal N\big(\boldsymbol 0,\big[\begin{smallmatrix}1&1/2\\1/2&1\end{smallmatrix}\big]\big)$，噪声 $\epsilon_t\sim\mathcal N(0,1)$。Theorem 3.1 证明：当 $T>e$ 时

$$\Pr\big(R_T\geq T/2\big)\geq \frac{c_1}{T^{c_2}},$$

其中 $c_1,c_2>0$ 是常数。这意味着至少有 $c_1/T^{c_2}$ 的概率，GP-TS 会一路选错臂、吃下线性级遗憾 $T/2$。把 $\delta=c_1/T^{c_2}$ 代回去，就能推出 GP-TS 一般情况下**不可能**有 $O(\log(1/\delta))$ 的累积遗憾上界——这是算法层面的纯理论限制，而非分析技术不到位。值得注意的是作者算出 $c_2=17$ 很大，所以这个反例只说明渐近上界的形式天花板，并**不**意味着 GP-TS 在实践中真的不稳定（实践里它往往表现优异）。这一结果反过来给后面「能把 $1/\delta$ 收紧到什么程度」定了方向。

**2. 累积遗憾的二阶矩上界：把 $\delta$ 依赖收紧 $1/\sqrt\delta$ 倍**

既然拿不到 $\log(1/\delta)$，那就尽量收紧多项式依赖。已有最好的高概率上界是 $O(\sqrt{T\gamma_T\log T}/\delta)$，对 $\delta$ 是 $1/\delta$。本文不直接做高概率分析，而是去界**累积遗憾的二阶矩**，Theorem 3.2 给出

$$\mathbb E[R_T^2]=O(T\gamma_T\log T),$$

再由 Markov 不等式立刻得到：对任意 $\delta\in(0,1)$，以至少 $1-\delta$ 概率 $R_T=O\big(\sqrt{T\gamma_T\log T/\delta}\big)$，把 $\delta$ 依赖从 $1/\delta$ 收紧成 $1/\sqrt\delta$。证明的巧处在于沿用 Russo & Van Roy 的分解，利用 GP-TS 的核心性质——给定历史 $\mathcal D_{t-1}$ 时 $\boldsymbol x^*$ 与 $\boldsymbol x_t$ 同分布且条件独立——把 $\mathbb E[R_T^2]$ 拆成 $T\sum_t \mathbb E\big[(f(\boldsymbol x^*)-U_t(\boldsymbol x^*))_+^2+\beta_t\sigma_{t-1}^2(\boldsymbol x_t)\big]$ 两块：后验方差之和可由 $\gamma_T$ 控住，而 $(f(\boldsymbol x^*)-U_t(\boldsymbol x^*))_+$ 的**平方**期望是已有分析没碰过的量，本文用新引理（Lemma E.5/E.6）补上界，这是从一阶矩升到二阶矩的关键一步。二阶矩本身还顺带说明了 GP-TS 累积遗憾的集中性质。

**3. 期望宽容遗憾的多对数上界：首个针对任意算法的此类界**

宽容遗憾 $LR_T=\sum_{t\in\mathcal T}f(\boldsymbol x^*)-f(\boldsymbol x_t)$ 只对「次优程度超过容差 $\Delta$」的轮次 $\mathcal T=\{t\mid f(\boldsymbol x^*)-f(\boldsymbol x_t)\geq\Delta\}$ 计遗憾，把「足够好」的点不算账。Theorem 3.3 证明 GP-TS 的期望宽容遗憾是 $T$ 的**多对数**量 $\mathbb E[LR_T]=O\big(\sqrt{\beta_T T_{\max}\tilde\gamma_{T_{\max}}}\big)$，且 $\mathbb E[|\mathcal T|]\leq T_{\max}$，其中 $T_{\max}$ 随核不同而异（SE 核 $O(\beta_T\log^{d+1}T/\Delta^2)$，Matérn 核 $O((\beta_T\log^{\cdots}T/\Delta^2)^{1+d/(2\nu)})$）。这是首个针对任意算法的**期望**宽容遗憾上界（此前 Cai 等、Iwazaki 只有 GP-UCB 的高概率版）。证明思路借鉴椭圆势计数引理：先用 $|\mathcal T|=\sum_{t\in\mathcal T}\min\{1,\frac{f(\boldsymbol x^*)-f(\boldsymbol x_t)}{\Delta}\}\leq\sum_{t\in\mathcal T}\frac{(f(\boldsymbol x^*)-f(\boldsymbol x_t))^2}{\Delta^2}$ 把计数转成平方遗憾之和，再用 Lemma E.7 把 $\mathbb E[\sum_{t\in\mathcal T}\sigma_{t-1}^2(\boldsymbol x^*)]$ 和 $\mathbb E[\sum_{t\in\mathcal T}\sigma_{t-1}^2(\boldsymbol x_t)]$ 都用 $\tilde\gamma_{\mathbb E[|\mathcal T|]}$ 控住——难点在于条件于 $t\in\mathcal T$ 后 $\boldsymbol x_t$ 与 $\boldsymbol x^*$ 不再同分布，这个引理是绕过它的核心。这套证明还能顺手推出 GP-UCB 的期望宽容遗憾界。

**4. 放宽 Matérn 条件的精细分析：拿到 $\tilde O(\sqrt T)$ 高概率遗憾**

最后把对 $T$ 的依赖收紧。Lemma 3.4 是一个通用结果：在 Lemma 2.4 的三个全局极大值点条件 + 事件 $E=\{\forall t,\,f(\boldsymbol x^*)-f(\boldsymbol x_t)\leq 2\beta_t^{1/2}\sigma_{t-1}(\boldsymbol x_t)+1/t^2\}$ 成立时，$R_T$ 对 SE 核是 $O(\sqrt T\log T)$、对 Matérn 核（$\nu>2$）是 $\tilde O(\sqrt T)$。关键突破是把 Matérn 核的条件从 Iwazaki (2025b) 的 $2\nu+d\leq\nu^2$ **放宽到只需 $\nu>2$**（这正是 Iwazaki 留下的公开问题），且没有引入对方猜想需要的更强正则条件。技术上靠两点：把容差 $\Delta$ 按 $T$ 动态调度，使每一层 $\mathcal T_i$ 的遗憾都 $\tilde O(\sqrt T)$；并对 Matérn 核反复套用「宽容遗憾上界 + $\mathcal B(\rho;\boldsymbol x^*)$ 上信息增益正比于 $\rho$」的论证，把 $\boldsymbol x_t$ 逐步收缩到 $\boldsymbol x^*$ 附近。由于 Lemma 3.4 的前提对 GP-UCB 天然成立，它可直接用于 GP-UCB；而用到 GP-TS 上还需 Theorem 3.5 做额外处理——因为 GP-TS 只有不条件于 Lemma 2.4 条件的期望遗憾界，需把累积遗憾按 $\tilde{\mathcal T}_i$ 分层、对每层证 $\mathbb E[\cdots]=\tilde O(\sqrt T)$ 再用联合界拼起来，最终以至少 $1-\delta-\delta_{\rm GP}$ 概率得到 $R_T=\tilde O(\sqrt T)$。

### 损失函数 / 训练策略
不涉及——本文是纯理论分析，没有训练目标或超参；唯一「可调量」是分析中用于分层的容差序列 $\{\Delta_i\}$ 和宽容遗憾的容差 $\Delta$。

## 实验关键数据

本文为纯理论论文，无数值实验。下面以表格汇总四组结果相对已有工作的改进。

### 主结果：GP-TS 遗憾上下界 vs 已有结果

| 结果 | 本文 | 已有最好（GP-TS） | 改进点 |
|------|------|------------------|--------|
| 累积遗憾对 $\delta$（下界）| Thm 3.1：$\Omega(1/\delta^{c})$ 概率吃线性遗憾 | 无 | 首次证明 $\log(1/\delta)$ 不可达，是硬限制 |
| 累积遗憾对 $\delta$（高概率上界）| Thm 3.2：$O(\sqrt{T\gamma_T\log T/\delta})$ | $O(\sqrt{T\gamma_T}/\delta)$ | $\delta$ 依赖收紧 $1/\sqrt\delta$ 倍 |
| 期望宽容遗憾 | Thm 3.3：$T$ 的多对数 | 无（仅 GP-UCB 高概率版）| 首个任意算法的期望宽容遗憾界 |
| 累积遗憾对 $T$（高概率上界）| Thm 3.5：$\tilde O(\sqrt T)$（SE / Matérn $\nu>2$）| $O(\sqrt{\gamma_T T\log T}/\delta)$ | $T$ 依赖收紧到 $\tilde O(\sqrt T)$ |

### 与 GP-UCB 的定位对比

| 维度 | GP-UCB | GP-TS（本文之后） | 备注 |
|------|--------|------------------|------|
| $\delta$ 依赖 | $\log(1/\delta)$ | 多项式 $1/\sqrt\delta$（Thm 3.2）| Thm 3.1 证明 GP-TS 无法做到对数 |
| 宽容遗憾 | 高概率界（已有）| 期望界（Thm 3.3）| 本文证法可反推 GP-UCB 期望界 |
| $T$ 依赖 | $\tilde O(\sqrt T)$ | $\tilde O(\sqrt T)$（Thm 3.5）| 已追平 |
| Matérn 条件 | $\nu>2$ | $\nu>2$（放宽自 $2\nu+d\leq\nu^2$）| Lemma 3.4 通用于两者 |
| 实践依赖参数 | 需精调置信宽度 $\beta_t$ | 无需 | GP-TS 的实践优势 |

### 关键发现
- **$\delta$ 依赖的差距是本质的**：Thm 3.1 的 $c_2=17$ 极大，说明这个下界是「形式天花板」而非实践不稳定的证据——GP-TS 实践常优于 GP-UCB，恰恰因为它不需要 GP-UCB 那个缺乏保证的置信宽度参数。
- **二阶矩是收紧 $\delta$ 的钥匙**：直接做高概率分析会卡在 $1/\delta$；改去界 $\mathbb E[R_T^2]$ 再走 Markov，自然得到 $1/\sqrt\delta$，同时附赠集中性质。
- **宽容遗憾既是结果又是工具**：Thm 3.3 本身是新结果，又是 Lemma 3.4 / Thm 3.5 把 $T$ 依赖收紧的关键中间件。
- **Matérn 条件 $\nu>2$ 仍是遗留缺口**：常用的 $\nu=1/2$ 或 $3/2$ 都不满足，作者把这是否可进一步放宽列为公开问题。

## 亮点与洞察
- **用一个两臂反例钉死了一类上界形式**：很多人默认「GP-TS 对 $\delta$ 的差依赖只是证明松」，本文反例直接证明它是算法硬限制，这种「先划天花板、再在天花板下做到最好」的叙事非常干净。
- **「升一阶矩」的技巧可迁移**：把高概率分析转化为二阶矩 + Markov，绕开随机算法直接做高概率界的困难，对其他随机化 BO 算法（同样靠 $\boldsymbol x^*\perp\!\!\!\perp\boldsymbol x_t\mid\mathcal D_{t-1}$ 的方法）都可能适用。
- **宽容遗憾的期望分析填了 BO 文献空白**：椭圆势计数式的证法把「计数 $|\mathcal T|$」转成「平方遗憾之和」，再用信息增益收口，思路优雅且可反哺 GP-UCB。
- **放宽 Matérn 条件兼有独立价值**：Lemma 3.4 通用于 GP-UCB 和 GP-TS，等于顺手解决了 Iwazaki (2025b) 列出的两个公开问题（「光滑性条件」和「推广到其他算法」）。

## 局限与展望
- **$\delta$ 上的最优性未定**：Thm 3.1 的下界并不能证明 Thm 3.2 的 $1/\sqrt\delta$ 就是紧的，二者之间仍有缺口；连续域 $\mathcal X$ 下能否得到同样的二阶矩结果也未知。
- **Matérn 条件 $\nu>2$ 限制实用性**：该条件来自 Assumption 2.2 与 Lemma 2.4，但排除了最常用的 $\nu=1/2,3/2$；能否避免或进一步放宽是开放问题。
- **方差膨胀版 GP-TS 未覆盖**：作者猜想带方差膨胀（Chowdhury & Gopalan, 2017）的 GP-TS 可拿到对 $\delta$ 更优的 $O(\sqrt{T\gamma_T\log(T/\delta)})$ 界，但因 Azuma–Hoeffding 带来的 $\tilde O(\sqrt T)$ 依赖，宽容遗憾与 $T$ 改进分析不易直接移植，留作未来工作。
- **忽略后验采样近似误差**：GP 后验样本路径无法解析采样，实践需 RFF 等近似（Wilson 等），本文分析直接忽略这部分误差。

## 相关工作与启发
- **vs GP-UCB（Srinivas 2010 / Iwazaki 2025b）**：GP-UCB 是被分析最透的 BO 算法，有 $\log(1/\delta)$ 高概率界、宽容遗憾高概率界、紧 $T$ 依赖。本文把 GP-TS 在 $T$ 依赖和宽容遗憾上追平、在 $\delta$ 依赖上尽量收紧（但证明 $\log(1/\delta)$ 不可达），且 Lemma 3.4 的精细分析对两者通用。
- **vs Russo & Van Roy (2014) / Takeno et al. (2024)**：他们给出 GP-TS 的期望累积遗憾 $\tilde O(\sqrt{T\gamma_T})$ 及由 Markov 直推的 $O(\sqrt{T\gamma_T}/\delta)$ 高概率界；本文沿用其分解框架，但升到二阶矩、补上宽容遗憾、并收紧 $T$ 依赖。
- **vs Cai et al. (2021) / Iwazaki (2025b) 的宽容遗憾**：他们做的是 GP-UCB 的**高概率**宽容遗憾界；本文用不同证法给出 GP-TS 的**期望**宽容遗憾界，并指出可反推 GP-UCB 期望界。
- **vs Scarlett (2018) 下界**：Scarlett 给的是算法无关的 $\Omega(\sqrt T)$ 条件期望遗憾下界（一维）；本文 Thm 3.1 是针对**概率** $\delta$ 的、专门作用于 GP-TS 的下界，刻画的是不同维度的限制，严格的高概率遗憾下界与最优性证明仍是公开问题。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 反例 + 二阶矩 + 期望宽容遗憾 + 放宽 Matérn 条件，四组结果都填了实打实的空白
- 实验充分度: ⭐⭐⭐⭐ 纯理论无实验，但定理覆盖完整、与已有结果对比清晰（理论文按此标准已足）
- 写作质量: ⭐⭐⭐⭐⭐ 「先划天花板再做到最优」的叙事线清晰，证明草图到位
- 价值: ⭐⭐⭐⭐⭐ 把 GP-TS 的理论保证基本追平 GP-UCB，并解决 Iwazaki (2025b) 的多个公开问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning Credal Ensembles via Distributionally Robust Optimization](learning_credal_ensembles_via_distributionally_robust_optimization.md)
- [\[ICLR 2026\] Alternating Diffusion for Proximal Sampling with Zeroth Order Queries](../../ICLR2026/learning_theory/alternating_diffusion_for_proximal_sampling_with_zeroth_order_queries.md)
- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](../../NeurIPS2025/learning_theory/sample-adaptivity_tradeoff_in_on-demand_sampling.md)
- [\[ICLR 2026\] Better Bounds for the Distributed Experts Problem](../../ICLR2026/learning_theory/better_bounds_for_the_distributed_experts_problem.md)

</div>

<!-- RELATED:END -->
