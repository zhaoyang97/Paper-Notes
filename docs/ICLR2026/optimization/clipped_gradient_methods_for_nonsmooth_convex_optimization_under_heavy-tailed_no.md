---
title: >-
  [论文解读] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis
description: >-
  [ICLR 2026][优化/理论][重尾噪声] 这篇论文对重尾噪声下的 Clipped SGD 给出一套更精细的收敛分析：通过更聪明地使用 Freedman 不等式、给出更紧的裁剪误差界，把已知最优的高概率收敛率再提速一个 $\mathrm{poly}(1/d_{\mathrm{eff}})$ 因子（$d_{\mathrm{eff}}$ 是作者定义的"广义有效维数"），并在期望收敛上得到能**突破已知下界**、且与作者新证下界**完全匹配**（即最优）的新速率。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "重尾噪声"
  - "梯度裁剪"
  - "高概率收敛"
  - "非光滑凸优化"
  - "有效维数"
---

# Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2dbLeXDe39](https://openreview.net/forum?id=2dbLeXDe39)  
**代码**: 无（纯理论论文）  
**领域**: optimization  
**关键词**: 重尾噪声, 梯度裁剪, 高概率收敛, 非光滑凸优化, 有效维数

## 一句话总结
这篇论文对重尾噪声下的 Clipped SGD 给出一套更精细的收敛分析：通过更聪明地使用 Freedman 不等式、给出更紧的裁剪误差界，把已知最优的高概率收敛率再提速一个 $\mathrm{poly}(1/d_{\mathrm{eff}})$ 因子（$d_{\mathrm{eff}}$ 是作者定义的"广义有效维数"），并在期望收敛上得到能**突破已知下界**、且与作者新证下界**完全匹配**（即最优）的新速率。

## 研究背景与动机
**领域现状**：随机一阶优化里，SGD 在"梯度噪声二阶矩有限（有限方差）"的经典假设下被研究得非常透彻。但近年大量经验观察发现，现代机器学习任务里的梯度噪声往往是**重尾**的——二阶矩可能根本不存在，更现实的假设是存在某个 $p\in(1,2]$ 使得 $p$ 阶矩有界，即 $\mathbb{E}\|g-\nabla f\|^p\le \sigma_l^p$。在这种重尾噪声下普通 SGD 会失效，而**梯度裁剪**（clipping）是被反复验证既实用又有理论保证的解药，对应方法即 Clipped SGD。

**现有痛点**：对非光滑（强）凸问题，此前最好的高概率收敛率来自 Liu & Zhou (2023)：凸情形 $O\!\big(\sigma_l\ln(1/\delta)\,T^{\frac1p-1}\big)$、强凸情形 $O\!\big(\sigma_l^2\ln^2(1/\delta)\,T^{\frac2p-2}\big)$。长期以来人们认为这两个率已经"最优"，因为只要把 $\mathrm{poly}(\ln(1/\delta))$ 视作常数，它们就匹配已有的期望下界。但 Das et al. (2024) 在 $p=2$ 的特例下做出了一个更快的率，引入了**有效维数** $d_{\mathrm{eff}}\in[1,d]$，暗示前述"最优"结论在含 $\mathrm{poly}(\ln(1/\delta))$ 的项上其实并不成立——可能存在对所有 $p\in(1,2]$ 都通用的改进空间。

**核心矛盾**：问题出在分析手法的两处"松"。其一，证明 Term I 的集中时，文献一直用 $\mathbb{E}\|d_t^u\|^2$ 来界条件方差，但真正需要控制的只是某个预测向量方向上的方差，用整个二阶矩去界明显偏大。其二，裁剪误差（尤其是偏差项 $\|d_t^b\|$）一直沿用 $O(\sigma_l^p\tau^{1-p})$ 这个界，作者发现它并不紧。而 Das et al. (2024) 虽然隐含用到了第一点，却靠一套复杂的"迭代精化"（iterative refinement）走通，代价是终界里多出 $\ln(\ln T/\delta)$ 因子、还附带 $T\ge\Omega(\ln\ln d)$ 的限制。

**本文目标**：(1) 证明对所有 $p\in(1,2]$ 都存在通用改进，给出更快的高概率率；(2) 把同样的洞见延伸到期望收敛，看能否突破已知下界；(3) 补上重尾设定下缺失的下界，判断新速率是否最优。

**切入角度 / 核心 idea**：作者把"噪声有多重尾"刻画得更细——在标准的 $\mathbb{E}\|d\|^p\le\sigma_l^p$ 之外，额外要求任意单位方向上的投影噪声满足 $\mathbb{E}|\langle e,d\rangle|^p\le\sigma_s^p$（$\sigma_s\le\sigma_l$）。由此定义**广义有效维数** $d_{\mathrm{eff}}\triangleq\sigma_l^2/\sigma_s^2$，并用它去吃掉那些原本被"用最坏方向估计"而浪费掉的量。一句话：**用细粒度噪声假设 + 更紧的两处不等式，把分析里被高估的项压回真实大小。**

## 方法详解

### 整体框架
论文研究的算法本身就是经典的 Clipped (Proximal) SGD（Algorithm 1），并不改动算法结构，全部贡献在**分析**。复合优化问题写作 $\inf_{x\in\mathcal X}F(x)\triangleq f(x)+r(x)$，其中 $f$ 是 $G$-Lipschitz 凸函数、$r$ 是 $\mu$-强凸（$\mu\ge0$，$\mu=0$ 即一般凸）。每步先采一个无偏噪声梯度 $g_t=g(x_t,\xi_t)$，做裁剪 $g_t^c=\mathrm{clip}_{\tau_t}(g_t)$，再走一步近端更新：

$$g_t^c=\min\!\Big\{1,\frac{\tau_t}{\|g_t\|}\Big\}g_t,\qquad x_{t+1}=\arg\min_{x\in\mathcal X}\;r(x)+\langle g_t^c,x\rangle+\frac{\|x-x_t\|^2}{2\eta_t}.$$

整篇论文的逻辑是：先把收敛性归约成对一条几乎处处成立的"残差不等式"里三个随机项（记作 I、II、III）的高概率控制（见下方公式的结构），然后用两件新工具——**更好地用 Freedman 不等式**和**更紧的裁剪误差界（Lemma 1）**——把 I 和 III 压小，从而得到比 Liu & Zhou (2023) 更快的率；同样两件工具还能顺手改进期望收敛，并配上信息论方法证出的匹配下界。这是一篇理论论文，无 pipeline 框架图可画，核心在下面的"关键设计"。

裁剪阈值取 $\tau_t=\max\{2G,\ \tau_\star T^{1/p}\}$，其中关键量
$$\tau_\star=\sigma_l/\varphi_\star^{1/p},\qquad \varphi_\star\triangleq\max\Big\{\sqrt{d_{\mathrm{eff}}}\,\ln\tfrac3\delta,\ d_{\mathrm{eff}}\,\mathbb 1[p<2]\Big\}.$$

### 关键设计

**1. 细粒度重尾噪声假设与广义有效维数 $d_{\mathrm{eff}}$：把"噪声各方向同样重尾"这个隐含浪费显式化**

文献里的重尾假设只有一个标量 $\sigma_l$：$\mathbb{E}\|d(x,\xi)\|^p\le\sigma_l^p$（$d=g-\nabla f$ 是梯度估计误差）。作者额外引入投影方向上的矩界 $\mathbb{E}|\langle e,d(x,\xi)\rangle|^p\le\sigma_s^p$（对所有单位向量 $e$），并证明这并非额外假设——由 Cauchy-Schwarz，只要 $\mathbb{E}\|d\|^p\le\sigma_l^p$ 成立，就一定存在 $0\le\sigma_s\le\sigma_l$ 使该投影界成立；反过来还有 $\sigma_l\le\sqrt{\pi d/2}\,\sigma_s$。于是定义

$$d_{\mathrm{eff}}\triangleq\sigma_l^2/\sigma_s^2\in\{0\}\cup[1,\pi d/2]=O(d).$$

当 $p=2$ 时，把 $\sigma_l^2=\sup_x\mathrm{Tr}(\Sigma(x))$、$\sigma_s^2=\sup_x\|\Sigma(x)\|$（$\Sigma$ 为噪声协方差），$d_{\mathrm{eff}}=\mathrm{Tr}(\Sigma)/\|\Sigma\|$ 恰好退回 Das et al. (2024) 的有效维数。这个量的意义在于：当噪声在各方向上"各向同性"时 $\sigma_s\approx\sigma_l$、$d_{\mathrm{eff}}\approx1$，没有便宜可占；但当噪声集中在少数方向（如低秩协方差）时 $\sigma_s\ll\sigma_l$、$d_{\mathrm{eff}}$ 可大到 $\Omega(d)$，后面所有 $1/\mathrm{poly}(d_{\mathrm{eff}})$ 的加速都来自此处。它是把"噪声其实没那么各向重尾"这件事变现的载体。

**2. 更好地使用 Freedman 不等式：用协方差算子范数而非二阶矩来界条件方差**

残差里的 Term I 形如 $\big(\max_{t}\sum_{s\le t}\langle d_s^u,y_s\rangle\big)^2$，其中 $d_t^u=g_t^c-\mathbb{E}_{t-1}[g_t^c]$ 是裁剪误差的无偏部分，$y_t$ 是某个 $\|y_t\|\le1$ 的可预测向量。令 $X_t=\langle d_t^u,y_t\rangle$ 是一个鞅差序列，Freedman 不等式给出（示意）
$$\sqrt{\mathrm{I}}\le O\!\Big(\max_t|X_t|\ln\tfrac1\delta+\sqrt{\textstyle\sum_t\mathbb{E}_{t-1}[X_t^2]\,\ln\tfrac1\delta}\Big).$$
文献一直用 $\mathbb{E}_{t-1}[X_t^2]\le\mathbb{E}_{t-1}\|d_t^u\|^2\le O(\sigma_l^p\tau^{2-p})$ 来界条件方差。作者的关键观察是：因为 $\|y_t\|\le1$，
$$\mathbb{E}_{t-1}[X_t^2]=y_t^\top\,\mathbb{E}_{t-1}\!\big[d_t^u(d_t^u)^\top\big]\,y_t\le\big\|\mathbb{E}_{t-1}[d_t^u(d_t^u)^\top]\big\|,$$
而协方差矩阵的**算子范数**至多等于 $\mathbb{E}\|d_t^u\|^2$（即其迹），却可能远小于它。配合下面 Lemma 1 给出的算子范数界 $O(\sigma_s^p\tau^{2-p}+\sigma_l^p G^2\tau^{-p})$（注意主导项里出现的是更小的 $\sigma_s$ 而非 $\sigma_l$），Term I 的高概率界被实实在在压紧。这一招其实在 Das et al. (2024) 的 $p=2$ 证明里被隐含用过，但他们陷入"迭代精化"的复杂论证，代价是终界多 $\ln(\ln T/\delta)$、还要 $T\ge\Omega(\ln\ln d)$；作者证明这套复杂化完全不必要，保持简单即可，且对所有 $p\in(1,2]$ 通用。

**3. 更精细的裁剪误差界（Lemma 1）：把偏差项与方差项的"松界"逐个收紧**

把裁剪误差拆成无偏部分 $d_t^u$ 与偏差部分 $d_t^b=\mathbb{E}_{t-1}[g_t^c]-\nabla f(x_t)$，Lemma 1 在 $\tau\ge2G$ 下给出
$$\big\|\mathbb{E}_{t-1}[d_t^u(d_t^u)^\top]\big\|\le O(\sigma_s^p\tau^{2-p}+\sigma_l^p G^2\tau^{-p}),\qquad \|d_t^b\|\le O(\sigma_s\sigma_l^{p-1}\tau^{1-p}+\sigma_l^p G\tau^{-p}),$$
并把二阶矩界 $\mathbb{E}_{t-1}\|d_t^u\|^2\le O(\sigma_l^p\tau^{2-p})$ 中原本需要的 $\tau\ge2G$ 条件去掉、常数也略改善。与文献常用界（记作 (6)）对照：偏差项从 $O(\sigma_l^p\tau^{1-p})$ 收紧为 $O(\sigma_s\sigma_l^{p-1}\tau^{1-p}+\cdots)$——因为 $\sigma_s\le\sigma_l$ 且 $\tau\ge2G$，新界严格不大；协方差算子范数界则是重尾设定下全新的结果。Term III（即 $\big(\sum_t\|d_t^b\|\big)^2$）正是靠收紧后的偏差界变小。作者还在附录 Theorem 9 给出不限于裁剪梯度法、甚至不要求 $\tau\ge2G$ 的更一般裁剪误差界，称其可能独立有用。

**4. 从两条洞见延伸出的下游结果：anytime 变体、突破期望下界、以及匹配的新下界**

同样两件工具被推到几个方向。其一，一般凸情形需预知 $T$ 才能去掉 $\mathrm{poly}(\ln T)$ 因子，作者提出 **Stabilized Clipped SGD**（引入 Fang et al. 2022 的 stabilization 技巧），以几乎相同的率做到 anytime（无需 $T$、不付 $\mathrm{poly}(\ln T)$ 代价）。其二，用更紧的裁剪误差界改进**期望收敛**，对一般凸得到 $O(\sigma_l d_{\mathrm{eff}}^{-(2-p)/(2p)}T^{\frac1p-1})$、强凸得到 $O(\sigma_l^2 d_{\mathrm{eff}}^{-(2-p)/p}T^{\frac2p-2})$，只要 $p<2$ 就**严格快于**已知期望下界 $\Omega(\sigma_l T^{\frac1p-1})$、$\Omega(\sigma_l^2 T^{\frac2p-2})$ 一个 $\mathrm{poly}(1/d_{\mathrm{eff}})$ 因子。其三，作者用信息论方法对**细粒度 oracle 类** $\mathcal G^p_{\sigma_s,\sigma_l}\subseteq\mathcal G^p_{\sigma_l}$ 证出新下界——正因为旧下界是对更大的类 $\mathcal G^p_{\sigma_l}$ 证的，对子类可能偏松，这解释了"为什么上界能突破旧下界"；而新的期望下界与上界**完全匹配**，从而证明了细粒度设定下期望收敛的最优性。

### 损失函数 / 训练策略
不适用：本文不引入新的训练目标或损失，算法即标准 Clipped (Proximal) SGD，全部新意在收敛性分析与阈值/步长的设定（$\eta_t$、$\tau_t$ 见上）。

## 实验关键数据
本文为纯理论工作，无数值实验。下面两张表概括其核心定量贡献：收敛率的提速幅度，以及裁剪误差界的收紧。

### 主结果：收敛率对比（主导项，$T\to\infty,\ \delta\to0$）

| 设定 | 之前最优 | 本文 | 改进 |
|------|----------|------|------|
| 高概率·凸 | $O(\sigma_l\ln\frac1\delta\,T^{\frac1p-1})$ (Liu & Zhou 23) | $O(\sigma_l\,d_{\mathrm{eff}}^{-\frac1{2p}}\ln^{1-\frac1p}\!\frac1\delta\,T^{\frac1p-1})$ | $\mathrm{poly}(1/d_{\mathrm{eff}},1/\ln\frac1\delta)$ |
| 高概率·强凸 | $O(\sigma_l^2\ln^2\!\frac1\delta\,T^{\frac2p-2})$ (Liu & Zhou 23) | $O(\sigma_l^2\,d_{\mathrm{eff}}^{-\frac1p}\ln^{2-\frac2p}\!\frac1\delta\,T^{\frac2p-2})$ | $\mathrm{poly}(1/d_{\mathrm{eff}},1/\ln\frac1\delta)$ |
| 期望·凸 | 下界 $\Omega(\sigma_l\,T^{\frac1p-1})$ | $O(\sigma_l\,d_{\mathrm{eff}}^{-\frac{2-p}{2p}}\,T^{\frac1p-1})$ | 突破下界 $\Theta(1/d_{\mathrm{eff}}^{\frac{2-p}{2p}})$，且与新下界匹配（最优） |
| 期望·强凸 | 下界 $\Omega(\sigma_l^2\,T^{\frac2p-2})$ | $O(\sigma_l^2\,d_{\mathrm{eff}}^{-\frac{2-p}{p}}\,T^{\frac2p-2})$ | 突破下界 $\Theta(1/d_{\mathrm{eff}}^{\frac{2-p}{p}})$，且与新下界匹配（最优） |

当 $d_{\mathrm{eff}}=\Omega(d)$（论文附录给出可达的例子）时，加速因子可达 $\mathrm{poly}(1/d)$ 量级。$p=2$ 的特例下，高概率凸结果不仅严格优于 Das et al. (2024)（甩掉多余项、$\delta$ 依赖从 $\ln(\ln T/\delta)$ 降到 $\ln(1/\delta)$），还对任意 $T\in\mathbb N$ 成立、且在 $\sigma_l\to0$ 时平滑退回确定性非光滑凸优化的标准率 $O(GD/\sqrt T)$。

### 分析层面：裁剪误差界对比（$\tau\ge2G$）

| 量 | 文献常用界 (6) | 本文 Lemma 1 |
|----|----------------|--------------|
| $\|d_t^u\|$ | $O(\tau)$ | $O(\tau)$（不变） |
| $\mathbb{E}_{t-1}\|d_t^u\|^2$ | $O(\sigma_l^p\tau^{2-p})$（需 $\tau\ge2G$） | $O(\sigma_l^p\tau^{2-p})$（去掉 $\tau\ge2G$，常数更优） |
| $\|\mathbb{E}_{t-1}[d_t^u(d_t^u)^\top]\|$ | —（无） | $O(\sigma_s^p\tau^{2-p}+\sigma_l^p G^2\tau^{-p})$（新，主导项含更小的 $\sigma_s$） |
| $\|d_t^b\|$ | $O(\sigma_l^p\tau^{1-p})$ | $O(\sigma_s\sigma_l^{p-1}\tau^{1-p}+\sigma_l^p G\tau^{-p})$（严格更紧） |

### 关键发现
- 提速的两个来源各司其职：协方差算子范数界（配 Freedman）压紧方差型的 Term I，偏差界 $\|d_t^b\|$ 收紧压小 Term III；二者都把 $\sigma_l$ 换成了更小的 $\sigma_s$，这正是 $d_{\mathrm{eff}}=\sigma_l^2/\sigma_s^2$ 因子的物理来源。
- 高概率上界仍残留一个高阶项（凸为 $O(\varphi GD/T)$、强凸为 $O(\varphi^2 G^2/(\mu T^2))$），在临界时间 $T_\star=\Theta(d_{\mathrm{eff}}\ln^2(1/\delta)+d_{\mathrm{eff}}^2\mathbb 1[p<2])$ 之后可忽略；能否对所有 $T$ 去掉它仍是公开问题。
- 期望上下界完全匹配 ⇒ 期望收敛已最优；但高概率上下界在含 $\mathrm{poly}(\ln(1/\delta))$ 的项上仍有 gap，是明确的未来方向。

## 亮点与洞察
- **"用对方向"比"用更强工具"更值钱**：Term I 的提速不是换了更强的集中不等式，而是发现真正要控的是 $y_t$ 方向上的方差，于是用协方差算子范数（$\le$ 迹）替掉二阶矩——一个免费却被长期忽视的紧化。
- **把复杂证明做简单**：Das et al. (2024) 靠"迭代精化"才拿到 $p=2$ 的改进，副作用是多余对数因子和 $T\ge\Omega(\ln\ln d)$ 限制；本文证明这套复杂化非必要，简单直接还能推广到全 $p\in(1,2]$，是"少即是多"的范例。
- **下界与上界对偶推进**：作者点破"旧下界是对更大 oracle 类 $\mathcal G^p_{\sigma_l}$ 证的，对子类 $\mathcal G^p_{\sigma_s,\sigma_l}$ 偏松"，既解释了上界为何能突破旧下界，又顺势在细粒度类上证出匹配下界——这种"先看清下界证在哪个类上"的视角可迁移到其他被认为"已最优"的设定。
- **可迁移性**：细粒度噪声假设 + 两条裁剪误差洞见并不局限于非光滑凸，作者明示它们是通用结果，光滑/非凸等设定值得照搬。

## 局限与展望
- 作者承认的局限：高概率率仍带一个 $d_{\mathrm{eff}}$ 相关的高阶残留项，是否对任意 $T$ 可去未知；高概率上下界在 $\mathrm{poly}(\ln(1/\delta))$ 项上尚有 gap。
- 假设层面：加速完全寄生于 $d_{\mathrm{eff}}=\sigma_l^2/\sigma_s^2$ 较大（噪声各向异性）这一前提；若噪声近各向同性 $\sigma_s\approx\sigma_l$、$d_{\mathrm{eff}}\approx1$，则相比 Liu & Zhou (2023) 几乎无改进。$d_{\mathrm{eff}}$ 在实际任务里是否常大并不显然，论文只给了构造性可达例子。
- 纯理论、无实验：阈值依赖未知量 $\sigma_s,\sigma_l,d_{\mathrm{eff}}$，实际如何估计/自适应未触及，落地需配自适应裁剪。
- 改进思路：把两条洞见落到光滑/非凸重尾设定；探索 $d_{\mathrm{eff}}$ 未知时的 parameter-free 阈值；闭合高概率 gap。

## 相关工作与启发
- **vs Liu & Zhou (2023)**：他们给出此前最优的高概率率并被认为"已最优"；本文在相同算法上靠更紧分析提速 $\mathrm{poly}(1/d_{\mathrm{eff}})$，说明"匹配旧下界"不等于真最优——关键在旧下界证在了更粗的 oracle 类上。
- **vs Das et al. (2024)**：他们首次在 $p=2$ 引入有效维数并改进凸率，但用复杂的迭代精化、带 $\ln(\ln T/\delta)$ 与 $T\ge\Omega(\ln\ln d)$ 限制；本文以更简单的论证覆盖全 $p\in(1,2]$、去掉这些副作用，并把有效维数推广为 $d_{\mathrm{eff}}=\sigma_l^2/\sigma_s^2$。
- **vs Gorbunov et al. (2024a/b)、Parletta et al. (2024)**：同属重尾下裁剪法的高概率分析谱系；Gorbunov et al. (2024b) 是唯一另一项处理复合（带 $r$）重尾优化的工作，但其 Prox-Clipped-SGD-Shift 与本文的 Proximal Clipped SGD 不同。
- **启发**：协方差算子范数替二阶矩、细粒度噪声假设、"先辨清下界证在哪个 oracle 类"这三点，构成一套可复用的"收紧重尾分析"的方法论，值得搬到光滑/非凸或在线优化等被认为已饱和的设定重新审视。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 推翻"已最优"共识，证出突破旧下界且匹配新下界的最优期望率
- 实验充分度: ⭐⭐⭐⭐ 纯理论，证明完整（含上界、下界、anytime 变体），但无数值验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机—洞见—结果的逻辑链清晰，proof sketch 把两条改进讲得很透
- 价值: ⭐⭐⭐⭐⭐ 方法论（算子范数 + 细粒度噪声 + 下界类）通用，可迁移到更多重尾设定

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](decentralized_nonconvex_optimization_under_heavy-tailed_noise_normalization_and_.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](../../ICML2026/optimization/can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICLR 2026\] High-Probability Bounds for the Last Iterate of Clipped SGD](high-probability_bounds_for_the_last_iterate_of_clipped_sgd.md)

</div>

<!-- RELATED:END -->
