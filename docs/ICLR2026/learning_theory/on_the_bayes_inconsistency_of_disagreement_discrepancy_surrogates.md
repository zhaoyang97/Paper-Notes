---
title: >-
  [论文解读] On the Bayes Inconsistency of Disagreement Discrepancy Surrogates
description: >-
  [ICLR 2026][学习理论][差异度差异] 本文证明了现有用于"差异度差异（disagreement discrepancy）"的代理损失在多分类（$K>2$）下**不是 Bayes 一致**的——优化代理并不一定优化真实目标，并据此设计了一个新的不一致损失 $-\log(1-\sigma(s)_y)$，配合交叉熵得到**首个可证 Bayes 一致**的代理，在误差界估计和有害偏移检测两类下游任务上都更可靠。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "分布偏移泛化"
  - "差异度差异"
  - "代理损失"
  - "Bayes 一致性"
  - "分布偏移"
  - "误差界"
---

# On the Bayes Inconsistency of Disagreement Discrepancy Surrogates

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=VwCyRQJ51H](https://openreview.net/forum?id=VwCyRQJ51H)  
**代码**: 复用 Rosenfeld & Garg (2023) 的实验代码，论文未给出独立仓库  
**领域**: 学习理论 / 分布偏移泛化  
**关键词**: 差异度差异、代理损失、Bayes 一致性、分布偏移、误差界

## 一句话总结
本文证明了现有用于"差异度差异（disagreement discrepancy）"的代理损失在多分类（$K>2$）下**不是 Bayes 一致**的——优化代理并不一定优化真实目标，并据此设计了一个新的不一致损失 $-\log(1-\sigma(s)_y)$，配合交叉熵得到**首个可证 Bayes 一致**的代理，在误差界估计和有害偏移检测两类下游任务上都更可靠。

## 研究背景与动机

**领域现状**：在分布偏移（distribution shift）下评估和提升模型可靠性，一条新兴路线是"差异度差异"。它度量同一个**评判模型（critic）** $f$ 与**参考模型（reference）** $h$ 的分歧（disagreement）在从源分布 $S$ 到目标分布 $T$ 时变化了多少。把这个量对 critic 取最大化，被用于：给无标签目标数据上的模型误差**定上界**（Rosenfeld & Garg, 2023）、构造**有害偏移的统计检验**（Ginsberg et al., 2023, Detectron）、以及训练**更鲁棒的集成**（Pagliardini et al., 2023, D-BAT）。

**现有痛点**：真实的差异度差异目标里含**零一损失** $\ell_{zo}(y,y')=\mathbb{1}_{y\neq y'}$，不可微，没法直接梯度优化。所以所有现有方法都换成可微的**代理损失**（surrogate）——agreement 端统一用交叉熵，disagreement 端各家不同（RG23 用 $\ell^{RG}_{dis}$、Ginsberg 用 $\ell^{GLK}_{dis}$）。但有一个被整条线忽略的问题：**优化代理目标，真的等于优化真实差异度差异吗？** 这不是空想——Mishra & Liu (2025) 已经报告了训练中的不稳定，暗示当前代理可能不胜任。

**核心矛盾**：经典的代理一致性理论（Zhang 2004；Bartlett et al. 2006）都是为**单个风险**（一个分布上的分类风险）建立的。而差异度差异是**两个不同分布上、用不同损失的风险之差** $\alpha R[\ell_{zo},h_2,Af](T)-R[\ell_{zo},h_1,Af](S)$。两个风险在优化上是**耦合**的，不能把现成理论分别套到两项上，所以"代理是否一致"在这个设定里一直没有答案。

**本文目标**：(1) 把一致性框架从"单风险分类"扩展到"风险之差"这个目标；(2) 判定现有代理是否 Bayes 一致；(3) 若不一致，设计出一致的代理。

**核心 idea**：用 pseudo-loss 把"风险之差"**解耦成两个不相交的风险之和**，从而能逐点（pointwise）分析最优 critic；用这套机器一边证下界（说明现有代理不一致）、一边证上界（说明新代理一致），并据"与零一损失逐点对齐"这个判据反推出新的 disagreement 损失。

## 方法详解

### 整体框架

论文要回答"代理优化是否忠实于真实目标"，整体走三步。**第一步**把一般化的差异度差异（Definition 1）写成

$$d_\alpha[h,f](S,T) = \alpha R[\ell_{zo},h_2,Af](T) - R[\ell_{zo},h_1,Af](S)$$

这里 $Af(x)=\arg\max_i f(x)_i$（带最小下标 tie-break），$\alpha>0$ 平衡两项；RG23 取 $h_1=h_2$、$\alpha=1$，Ginsberg 取 $h_1$ 为真标签函数、$h_2$ 为待测模型、$\alpha\approx 1/N$。代理目标则是 Definition 2：$\hat d_\alpha=R[\ell_{agr},h_1,f](S)+\alpha R[\ell_{dis},h_2,f](T)$，设计成**最小化**（与真实目标的最大化相反）。

**第二步**是关键的数学手术：把这个"差"重写成两个**逐点不相交**的风险之和（Reformulation），让经典单风险理论可以逐段套用。**第三步**用重写后的形式做两件相反的事——对现有代理证一个**下界**（optimality gap 的下界），证明它们在 $K>2$ 下不一致；对新提出的代理证一个**上界**，证明它一致。这是一篇纯理论+实证验证的论文，方法核心是公式与证明结构，没有可画的 pipeline 流程，因此不配框架图。

### 关键设计

**1. 用 pseudo-loss 把"风险之差"解耦成两个不相交风险之和**

痛点在于：差异度差异是两个分布、两种损失的风险**相减**，经典一致性理论（建立在单一风险上）无法直接搬过来，而且两项在优化上互相纠缠。本文引入源/目标密度 $p_S,p_T$，按 $p_S(x)$ 与 $p_T(x)$ 谁大把输入空间切成两半，构造两个损失泛函：

$$L_1[\ell_1,\ell_2](x,y,z)=\mathbb{1}_{p_S(x)\ge p_T(x)}\Big(\ell_1(x,y_1,z)+\tfrac{p_T(x)}{p_S(x)}\ell_2(x,y_2,z)\Big)$$

$$L_2[\ell_1,\ell_2](x,y,z)=\mathbb{1}_{p_S(x)<p_T(x)}\Big(\tfrac{p_S(x)}{p_T(x)}\ell_1(x,y_1,z)+\ell_2(x,y_2,z)\Big)$$

于是 $d_\alpha[h,f](S,T)=R[\ell_1,h,Af](S)+R[\ell_2,h,Af](T)$，其中 pseudo-loss 取 $\ell_1=L_1[-\ell_{zo},\alpha\ell_{zo}]$、$\ell_2=L_2[-\ell_{zo},\alpha\ell_{zo}]$（代理同理用 $\ell_{agr},\alpha\ell_{dis}$ 替换）。这个改写有一个**至关重要**的性质：对任意输入 $x$，$\ell_1(x,\cdot)$ 和 $\ell_2(x,\cdot)$ **恰有一个非零**。密度比当权重把两个分布"摊"到同一个积分里，indicator 又保证两项不重叠，于是可以对每个 $x$ **逐点优化** $f$，把原本耦合的两个风险彻底拆开——这是后续所有证明的地基。

**2. 用 optimality-gap 下界证明现有代理不 Bayes 一致**

有了解耦，作者把 Zhang (2004) 的上界框架反过来，先对单风险发展出一个 optimality gap 的**下界**（Appendix A），再搬到差异度差异上（Theorem 4）。直觉是：存在一片输入区域，**代理的最优 critic 与真实目标的最优 critic 给出的预测不一致**。具体地，对 $K>2$，在受限子空间

$$\mathcal{X}'=\Big\{x: h_1(x)=h_2(x),\ p_T(x)>0,\ \lambda+\delta\le \tfrac{p_S(x)}{\alpha p_T(x)}\le 1-\delta\Big\}$$

上，存在一个在 0 处连续的凸函数 $\zeta$，满足

$$\sup_{f'}d_\alpha[h,f'](S,T)-d_\alpha[h,f](S,T)\ \ge\ \zeta\big(\hat d_\alpha[\cdot](S|_{\mathcal{X}'},T|_{\mathcal{X}'})-\inf_{f'}\hat d_\alpha[\cdot]\big)$$

且 $\zeta(0)=\frac{\delta}{1-\delta}\mathbb{1}_{S(\mathcal{X}')>0}+\alpha\delta\,\mathbb{1}_{T(\mathcal{X}')>0}$。关键就在 **$\zeta(0)$ 可以严格大于 0**：只要源或目标在 $\mathcal{X}'$ 上有正测度，哪怕代理被优化到完美（gap 为 0），真实目标仍残留一个不消失的缝隙。这直接违反 Definition 3 的一致性条件，于是 Corollary 5：RG23 和 Ginsberg 的代理在 $K>2$ 时**都不是 Bayes 一致**的——即使无限数据、无限模型容量，优化它们也未必优化真实目标。这把"为什么会训练不稳定"从经验观察提升为理论必然。

**3. 与零一损失对称对齐的新 disagreement 损失，配交叉熵得首个一致代理**

不一致的根源是代理最优解与真实最优解错位。作者据此反推出一个新的 disagreement 损失（公式 9）：

$$\ell^{Ours}_{dis}(x,y,s)=-\log\big(1-\sigma(s)_y\big)$$

设计动机是它与交叉熵 agreement 损失 $-\log\sigma(s)_y$ **严格对称**：交叉熵把 $\sigma(s)_y$ 推向 1（鼓励同意 $y$），而这个损失把 $\sigma(s)_y$ 推向 0（鼓励不同意 $y$）。更关键的是，它**只要求 $\sigma(s)_y\to 0$、不规定其余概率怎么分配**——这恰好与真实 disagreement 损失 $-\mathbb{1}_{y\ne A(s)}$ 的逐点最优行为对齐（真实损失也只关心"别等于 $y$"）。正是相比之下，$\ell^{RG}_{dis},\ell^{GLK}_{dis}$ 会对其余类的 logit 施加额外结构，导致最优解偏离。沿用 Zhang (2004) 的上界框架并借助前面的解耦，作者证明（Theorem 6）存在在 0 处连续、$\xi(0)=0$ 的凹函数 $\xi$，使真实 optimality gap 被代理 gap 经 $\xi$ 上界控制；由 $\xi(0)=0$ 立得 Corollary 7：该代理对**所有 $K\ge 2$ 都 Bayes 一致**，是这一任务上第一个有此保证的代理。Remark 8 还指出二分类 $K=2$ 时三种代理彼此等价，所以不一致问题只在多分类才暴露。

### 损失函数 / 训练策略
最终代理目标是 agreement 端交叉熵 $\ell_{agr}=\ell_{ce}$ + disagreement 端 $\ell^{Ours}_{dis}=-\log(1-\sigma(s)_y)$ 的组合，对 critic $f$ 最小化即近似最大化真实差异度差异。实验里 critic 是在冻结的待测模型 $h$ 上接一层可调线性层得到（仅微调该层 logit 变换）。

## 实验关键数据

围绕两个下游任务验证：协变量偏移下的误差界、有害偏移检测。

### 主实验：误差界估计（自然 + 对抗目标数据）

复现 RG23 的设定，跨 11 个视觉偏移基准（WILDs / BREEDs / DomainNet 等）、5 种训练法（ERM / FixMatch / BN-adapt / DANN / CDAN），共 130 个"偏移×模型"组合。由于真实最大差异度差异不可算，用"任一代理达到的最大值"作参照。

| 设定 | 指标 | 本文 (Ours) | 对比 | 结论 |
|--------|------|------|----------|------|
| 自然偏移 130 组合 | 取得最大差异度的比例 | ≈80% | RG23 余下 | 单边 Wilcoxon $p=1.8\times10^{-11}$ |
| 误差界校准 | observed vs desired 违约率 $\delta$ | 更贴近 $y=x$ | RG23 偏离更大 | 小 $\delta$ 下两者违约率都偏高 |
| 对抗目标 $f=0\%$ | 取得 rank-1 比例 | 87.5% | RG23 / GLK23 | $p=3.3\times10^{-4}$ |
| 对抗目标 $f=50\%$ | 取得 rank-1 比例 | 100% | RG23 / GLK23 | $p<6.0\times10^{-10}$ |

低估真实差异度差异会得到"看似更紧、实则可能失效"的误差界（真实误差超过界），所以"恢复出更大的差异度"= 更可信的界。本文代理在自然和对抗设定下都更接近真实最大值。

### 有害偏移检测（Detectron / UCI-HD）

采用 Ginsberg et al. (2023) 的 Detectron 框架，在 UCI Heart Disease 上保留原始 5 类（而非前人用的二分类，因为 $K=2$ 时本文与基线等价），用 XGBoost critic，目标样本量 $N\in\{10,20,50\}$，每组重复 500 次估 ROC。

| $N$ | GLK23 AUC-ROC | 本文 AUC-ROC |
|------|------|------|
| 10 | $0.821^{+0.025}_{-0.025}$ | $0.908^{+0.019}_{-0.017}$ |
| 20 | $0.913^{+0.016}_{-0.019}$ | $0.984^{+0.005}_{-0.006}$ |
| 50 | $0.995^{+0.003}_{-0.002}$ | $1.000^{+0.000}_{-0.000}$ |

三个样本量下 95% 置信区间均不重叠，说明改进在统计上显著——**理论上的一致性确实转化为更高的检验统计功效**。

### 关键发现
- 不一致只在 $K>2$ 暴露：二分类时三种代理等价，这解释了为何过去在二分类基准上没人发现问题。
- 对抗目标数据是更严苛的压力测试：攻击比例越高，本文代理领先越明显（rank-1 比例从 87.5% 升到 100%），说明现有代理的失效在困难分布下被放大。
- 误差界校准在小 $\delta$ 下两种方法都偏高，提示即便代理一致，下游应用（误差界）自身的假设也可能不成立。

## 亮点与洞察
- **把"风险之差"解耦成两个不相交风险**是全文的杠杆点：用密度比加权 + indicator 切分，使任意 $x$ 处恰有一项非零，从而能逐点优化、把单风险一致性理论搬过来。这个技巧对任何"目标=两分布风险之差"的问题（域适配、不变性正则）都可能复用。
- **同一套 optimality-gap 机器，下界证伪、上界证真**：下界用来证明现有代理不一致（$\zeta(0)>0$），上界用来证明新代理一致（$\xi(0)=0$），两端对称，逻辑非常干净。
- **新损失的设计判据是"与零一损失逐点最优对齐"**：$-\log(1-\sigma(s)_y)$ 只压低目标类概率、不约束其余类，正好镜像 $-\mathbb{1}_{y\ne A(s)}$ 的行为，这种"对齐真实损失的最优解"的反推思路很值得借鉴。

## 局限与展望
- 只做了 **Bayes 一致性**（在全体可测函数类上优化），更强的 **H-consistency**（受限假设类）对深网仍是开放问题——作者明确说当前没有针对现代深网的成功 H-一致性分析。
- 一致性是**渐近**保证（无限数据、无限容量）；完整的有限样本保证还需要 estimation error 的界，本文只解决了 calibration error 一侧。
- 即使代理一致，下游应用（如误差界）的**前提假设也可能不成立**，对抗实验中误差界仍会被攻破，说明实践部署仍需谨慎。
- 实验里 critic 仅是冻结模型上加一层线性层，是否在更强 critic 类下结论依旧、文中未展开。

## 相关工作与启发
- **vs Rosenfeld & Garg (2023)**: 他们提出用 critic 最大化差异度差异来给目标误差定界，并用 $\ell^{RG}_{dis}$ 做代理；本文证明该代理在 $K>2$ 时不一致，会低估真实差异度从而给出失效的"假紧"界，新代理修复了这一点。
- **vs Ginsberg et al. (2023) Detectron**: 他们用 disagreement 交叉熵做有害偏移检验；本文证明其代理同样不一致，并在 UCI-HD 多分类上展示新代理带来更高的检验功效。
- **vs Mishra & Liu (2025)**: 他们经验性地报告了训练不稳定并引入 discounted disagreement 缓解；本文给出了不稳定背后的理论根因（代理不一致）。
- **vs Pagliardini et al. (2023) D-BAT**: 用差异度差异目标当多样性正则训练集成；本文为这类训练方法提供了一条"换成一致代理"的可靠化路径。
- **vs $H\Delta H$-divergence (Ben-David et al. 2010)**: 差异度差异是其单 critic 的可操作化版本，本文进一步补上了代理一致性的理论缺口。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把代理一致性理论扩展到"风险之差"目标，并据此设计出第一个可证一致的代理。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 11 个偏移基准 + 对抗压力测试 + 检测任务，统计检验完整；但 critic 类较受限。
- 写作质量: ⭐⭐⭐⭐⭐ 理论叙述清晰，下界/上界对称结构和动机交代得很到位。
- 价值: ⭐⭐⭐⭐⭐ 修复了一整条研究线（误差界/偏移检测/鲁棒训练）共享的理论缺陷，地基级贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PAC-Bayes Bounds for Cumulative Loss in Continual Learning](pac-bayes_bounds_for_cumulative_loss_in_continual_learning.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICML 2026\] Realizable Bayes-Consistency for General Metric Losses](../../ICML2026/learning_theory/realizable_bayes-consistency_for_general_metric_losses.md)
- [\[ICLR 2026\] Why Ask One When You Can Ask k? Learning-to-Defer to the Top-k Experts](why_ask_one_when_you_can_ask_k_learning-to-defer_to_the_top-k_experts.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)

</div>

<!-- RELATED:END -->
