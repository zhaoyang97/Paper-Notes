---
title: >-
  [论文解读] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem
description: >-
  [ICLR 2026][学习理论][Tree-Sliced Wasserstein] 本文指出 Tree-Sliced Wasserstein（TSW）相比 Sliced Wasserstein（SW）的真正优势在于它的采样同时编码了"位置"信息，而现有 TSW 变体的采样（高斯中心放在数据均值）并没有用好这一点；作者借助经典的 Fermat–Weber 问题，用**几何中位数**作为树系统交点的采样中心，提出 FW-TSW / FW-TSW\*，在几乎不增加计算开销的前提下提升了梯度流、主题建模与扩散模型训练的效果。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "最优传输"
  - "Tree-Sliced Wasserstein"
  - "Fermat–Weber 问题"
  - "几何中位数"
  - "采样策略"
---

# Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=kDqG03v05B](https://openreview.net/forum?id=kDqG03v05B)  
**代码**: https://github.com/thanhquangtran/FW-TSW  
**领域**: 学习理论 / 最优传输  
**关键词**: 最优传输, Tree-Sliced Wasserstein, Fermat–Weber 问题, 几何中位数, 采样策略

## 一句话总结
本文指出 Tree-Sliced Wasserstein（TSW）相比 Sliced Wasserstein（SW）的真正优势在于它的采样同时编码了"位置"信息，而现有 TSW 变体的采样（高斯中心放在数据均值）并没有用好这一点；作者借助经典的 Fermat–Weber 问题，用**几何中位数**作为树系统交点的采样中心，提出 FW-TSW / FW-TSW\*，在几乎不增加计算开销的前提下提升了梯度流、主题建模与扩散模型训练的效果。

## 研究背景与动机

**领域现状**：最优传输（OT）能尊重数据几何地比较两个概率测度，但精确求解 OT 的复杂度随支撑点数超立方增长。Sliced Wasserstein（SW）把高维测度投影到一维直线、利用一维 OT 的闭式解再平均，把复杂度压到近线性，成为可扩展的主流替代。近年来 Tree-Sliced Wasserstein（TSW）进一步用"树系统"（tree system，即 $\mathbb{R}^d$ 中一组共点的 $k$ 条直线 $T=(x,\theta_1,\dots,\theta_k)$）取代单条投影线，配合一个 splitting map 把质量分摊到各条线上，从而在保持低成本的同时刻画更复杂的拓扑结构。

**现有痛点**：SW 与 TSW 之间一个被忽视的本质差异是**采样策略**。SW 只采样方向 $\theta\in S^{d-1}$，而 TSW 的树系统除了方向，还要采样一个**交点（位置）** $x\in\mathbb{R}^d$——也就是说 TSW 天然携带"方向 + 位置"两类信息。但现有 TSW 变体（TSW-SL、Db-TSW）对这个交点的处理很随意：因为 $\mathbb{R}^d$ 非紧、没有像球面 $S^{d-1}$ 上那样的标准均匀分布，它们简单地把交点从"以数据均值为中心"的高斯里采样。

**核心矛盾**：均值对离群点敏感，且这种启发式采样并没有真正"对齐"源分布和目标分布的几何中心，等于浪费了 TSW 相对 SW 最值钱的位置自由度。同时，常用的 splitting map（公式 $\alpha(y,T)=\mathrm{softmax}(\{\xi\cdot d(y,T)_i\})$）本身就是**位置相关**的——交点选得好不好，直接影响质量分摊是否合理。

**本文目标**：为 TSW 设计一个**有原则、数据相关**的交点采样分布，让树系统的根落在"离源/目标数据都近"的位置上，同时把方向采样也做得更有信息量。

**切入角度**：OT 的目的是把源分布对齐到目标分布，那么交点 $x$ 理应**最小化它到数据点的平均距离**——这正是位置理论里经典的 Fermat–Weber 问题，其解称为**几何中位数**。几何中位数比均值对离群点稳健，且可用 Weiszfeld 算法高效逼近。

**核心 idea**：用几何中位数（而非数据均值）作为树系统交点采样分布的中心，把 Fermat–Weber 原则注入 TSW 的树构造过程，得到 FW-TSW；再把方向采样偏向"源点到目标点"的方向，得到增强版 FW-TSW\*。

## 方法详解

### 整体框架
FW-TSW 的整条逻辑可以这样鸟瞰：给定源测度 $\mu$ 与目标测度 $\nu$，TSW 类方法的核心都是"采样若干棵树系统 $T_1,\dots,T_L$，在每棵树上算闭式 OT 距离 $W_1$，再平均"。本文**不改变**这个 Monte Carlo 估计骨架（即式 $\widehat{\mathrm{TSW}}=\frac1L\sum_l W_1(R^\alpha_{T_l}f_\mu, R^\alpha_{T_l}f_\nu)$），而只是**重新设计采样树系统的分布 $\sigma_T$**。

具体地，一棵树系统由"一个交点 $x\in\mathbb{R}^d$ + $k$ 个方向 $\theta_i\in S^{d-1}$"组成，采样分布被建模为这 $k+1$ 个分量的乘积。本文动的就是其中两个分量：

- **交点分量**：从源、目标各采 $m$ 个点合在一起，用 Weiszfeld 算法求几何中位数 $x^\*$，再以 $x^\*$ 为中心的高斯采样交点 → 得到 $\sigma_{\mathrm{FW},\mu,\nu}$ 与 FW-TSW。
- **方向分量（可选增强）**：把方向从均匀采样改成偏向"源点 $x_i$ 减目标点 $y_j$"的方向 → 得到 $\sigma^\*_{\mathrm{FW},\mu,\nu}$ 与 FW-TSW\*。

把这两件改造插回上面的平均骨架，再配合 splitting map 把质量分摊到树的各条线、用树上 OT 闭式解求 $W_1$，就得到最终的距离 FW-TSW($\mu,\nu$) $=\int_{\mathbb{T}} W_1(\mu_T,\nu_T)\,d\sigma_{\mathrm{FW},\mu,\nu}(T)$。因为这是一个纯采样策略 + 解析性质的工作（没有多阶段神经网络 pipeline），下面直接讲三个关键设计。

### 关键设计

**1. 几何中位数采样交点：用 Fermat–Weber 解替代数据均值**

这一点直击"交点采样无原则"的痛点。给定从源/目标合并采出的点集 $\{x_1,\dots,x_m,y_1,\dots,y_m\}$，Fermat–Weber 问题求的是到所有点距离之和最小的位置：

$$x^\* = \arg\min_{x\in\mathbb{R}^d}\ \frac1n\sum_{i=1}^n \lVert x - x_i\rVert_2,$$

其解 $x^\*$ 即几何中位数。然后交点从以它为中心的高斯采样 $x\sim\mathcal{N}(x^\*, cI_d)$，整棵树系统的分布写成 $\mathcal{N}(x^\*,cI_d)\otimes U(S^{d-1})^{\otimes k}$。常数 $c$ 控制交点围绕 $x^\*$ 的集中度，论文发现数据归一化后取 $c=1$ 即稳定好用。

为什么有效：均值会被离群点拉偏，而几何中位数是稳健的"中心"，能让树系统的根落在数据真正密集的地方，从而让位置相关的 splitting map 分摊质量更合理。把源、目标点合并求 $x^\*$ 还顺带保证了对称性 $\sigma_{\mathrm{FW},\mu,\nu}=\sigma_{\mathrm{FW},\nu,\mu}$。求解上用 **Weiszfeld 算法**迭代逼近：

$$x^{(t+1)} = \left(\sum_i \frac{x_i}{\lVert x^{(t)}-x_i\rVert_2}\right)\Big/\left(\sum_i \frac{1}{\lVert x^{(t)}-x_i\rVert_2}\right),$$

即每步用"到当前估计的距离倒数"作权重做加权平均，迭代到 $\lVert x^{(t+1)}-x^{(t)}\rVert_2\le\varepsilon$ 即停。固定迭代步数即可，开销只是额外的 $O(Tnd)$，相对整体可忽略。

**2. 数据相关的方向采样：让投影方向对齐源—目标位移（FW-TSW\*）**

光把交点选好还不够——均匀采样的方向同样"informative 与否不分"。本文为方向也设计了数据相关的增强：随机取一个源点 $x_i$ 和目标点 $y_j$，构造方向

$$\theta = \frac{\psi + \zeta\cdot s\cdot(x_i - y_j)}{\lVert \psi + \zeta\cdot s\cdot(x_i - y_j)\rVert_2}\in S^{d-1},$$

其中 $\psi\sim U(S^{d-1})$ 是一个均匀随机方向打底，$s\sim U(\{\pm1\})$ 是随机符号，$\zeta>0$ 控制方向被偏向位移向量 $(x_i-y_j)$ 的强度。这样采出的方向倾向于沿着"源点到目标点"的搬运方向，因而更能区分两个分布的差异。把它记为 $\sigma_{\mathrm{dir},\mu,\nu}$，配合几何中位数交点得到增强分布 $\sigma^\*_{\mathrm{FW},\mu,\nu}=\mathcal{N}(x^\*,I_d)\otimes(\sigma_{\mathrm{dir},\mu,\nu})^{\otimes k}$，对应 FW-TSW\*。随机符号 $s$ 的引入保证了方向分布对称 $\sigma_{\mathrm{dir},\mu,\nu}=\sigma_{\mathrm{dir},\nu,\mu}$，从而整体分布对称。

**3. 可证的度量性质与可控的上界**

这一点是把前两个设计"扶正"为一个良定义的差异度。作者证明 FW-TSW 与 FW-TSW\* 都是 $\mathcal{P}(\mathbb{R}^d)$ 上的**半度量**（semi-metric）：满足非负性、对称性、不可分者同一性，并满足一个**拟三角不等式**

$$\mathrm{FW\text{-}TSW}(\mu_1,\mu_2)\le \mathrm{FW\text{-}TSW}_{\mu_1,\mu_2}(\mu_1,\mu_3)+\mathrm{FW\text{-}TSW}_{\mu_1,\mu_2}(\mu_2,\mu_3),$$

其中中间项把采样分布固定在由 $\mu_1,\mu_2$ 导出的 $\sigma_{\mathrm{FW},\mu_1,\mu_2}$ 上。它们还在欧氏变换群 $E(d)$ 下**不变**（与经典 2-Wasserstein、SW 一致）。更关键的是，正因为交点被几何中位数"锚定"，本文能给出一个先前 TSW 变体难以得到的**上界**（Theorem 4.5）：当采样中心取在源/目标联合 Fermat–Weber 解 $v^\*$ 处时，

$$\int_{\mathbb{T}} W_1(\mu_T,\nu_T)\,d\bar\sigma_{\mathrm{FW},\mu,\nu}(T)\le k\,W_2(\mu,\nu) + k(k-1)\cdot\frac{2\pi^{d/2}}{\Gamma\!\big(\tfrac{d+1}{2}\big)\Gamma\!\big(\tfrac12\big)}f(v^\*),$$

其中 $f(v)=\int\lVert x-v\rVert_2\,d\mu+\int\lVert x-v\rVert_2\,d\nu$。先前变体里位置信息不受控、bound 难求；几何中位数让位置可控，从而把这个上界做了出来。⚠️ 公式细节以原文为准。

### 损失函数 / 训练策略
FW-TSW 本身是一个**距离度量**，不引入额外训练损失；它作为可微的传输代价被插入到三类下游目标里：梯度流中作为被最小化的 $D(\mu_t,\nu)$；主题建模中替换 VAE 目标里的 KL 项（$\lambda\,\mathrm{FW\text{-}TSW}(q_\phi(\theta),p(\theta))$）；扩散模型中嵌入 DDGAN 的 AGME 损失。计算复杂度为 $O(Lkn\log n + Lkdn + Tnd)$，其中 $Tnd$ 是 Weiszfeld 求几何中位数的额外项，FW-TSW\* 再加一个 $O(Lkd)$ 生成方向的项，二者相对整体都可忽略。

## 实验关键数据

### 主实验

**梯度流（25 Gaussians，平均 Wasserstein 距离，5 次平均，越小越好）**：

| 方法 | step 1000 | step 2000 | step 2500 |
|------|-----------|-----------|-----------|
| SW | 2.42e-03 | 1.69e-03 | 1.01e-03 |
| TSW-SL | 1.37e-06 | 9.13e-07 | 8.76e-07 |
| Db-TSW | 1.55e-06 | 9.50e-07 | 8.55e-07 |
| Db-TSW⊥ | 1.79e-06 | 1.14e-06 | 1.03e-06 |
| **FW-TSW (ours)** | 1.51e-06 | 9.18e-07 | 8.40e-07 |
| **FW-TSW\* (ours)** | 1.50e-06 | **9.04e-07** | **8.29e-07** |

从 step 2000 起 FW-TSW / FW-TSW\* 取得最优，step 2500 时 FW-TSW\* 整体最佳。

**扩散模型（DDGAN，CIFAR-10 无条件生成，FID 越小越好）**：

| 模型 | FID ↓ | Time/Epoch(s) ↓ |
|------|-------|------------------|
| DDGAN | 3.64 | 72 |
| RPSW-DD | 2.82 | 76 |
| TSW-SL-DD | 2.83 | 80 |
| Db-TSW-DD | 2.60 | 84 |
| Db-TSW-DD⊥ | 2.53 | 85 |
| **FW-TSW-DD (ours)** | 2.336 ± 0.003 | 85 |
| **FW-TSW\*-DD (ours)** | **2.315 ± 0.002** | 87 |

相比当前最强 OT 基线 Db-TSW-DD⊥，FW-TSW / FW-TSW\* 分别把 FID 降低 0.194 / 0.215，而单 epoch 训练时间基本持平（85→87s）。

### 消融 / 分析（主题建模，topic coherence CV，越高越好）

| 方法 | DBLP | M10 | BBC |
|------|------|-----|-----|
| SW-TM | 0.432 | 0.484 | 0.760 |
| TSW-SL-TM | 0.453 | 0.456 | 0.796 |
| Db-TSW-TM | 0.441 | 0.458 | 0.787 |
| **FW-TSW-TM (ours)** | 0.505 | 0.498 | 0.792 |
| **FW\*-TSW-TM (ours)** | **0.511** | **0.502** | **0.801** |

FW-TSW 系在三个数据集上整体优于 SW 与 TSW 系基线，FW-TSW\* 几乎全面最佳。

### 关键发现
- **方向增强（FW-TSW\*）几乎总是再补一刀**：在三类任务里，FW-TSW\* 普遍略好于 FW-TSW，说明数据相关的方向采样和几何中位数交点是**互补**的两块增益。
- **几乎零额外开销**：FID 大幅下降的同时训练时间只从 85s 升到 87s，几何中位数（Weiszfeld）和方向采样的成本相对树上 OT 可忽略。
- **位置信息是关键**：相对只换交点中心（均值→几何中位数）这一个改动就能稳定超过 Db-TSW 系，印证了"TSW 的优势在位置、而旧方法没用好位置"这一核心论点。

## 亮点与洞察
- **把一个被忽视的差异点（采样）讲成核心贡献**：作者敏锐地指出 SW 与 TSW 的本质区别不只是"线 vs 树"，而是"只有方向 vs 方向+位置"，并据此找到改进抓手——这种"重新审视已有框架的隐含假设"的研究品味很值得学习。
- **经典数学工具的现代复用**：Fermat–Weber 问题 / 几何中位数 / Weiszfeld 算法都是上世纪的经典结果，被干净地嫁接到 TSW 采样上，既稳健又自带高效求解器。
- **稳健中心带来可证上界**：几何中位数不仅经验上更好，还在理论上把"位置不受控导致 bound 难求"的老问题解决了，是"经验增益与理论性质同源"的范例。
- **可迁移性**：凡是"需要在 $\mathbb{R}^d$ 上选一个数据相关采样中心"的 sliced/tree-sliced 方法，都可以考虑用几何中位数替代均值这一招。

## 局限与展望
- **缺少显式传输映射**：作者承认这是所有 TSW 变体共有的局限——FW-TSW 给出的是距离值而非传输计划/映射，无法直接用于需要 push-forward 的场景，未来可发展能产出传输计划的 tree-sliced 框架。
- **半度量而非完整度量**：只满足拟三角不等式，并非严格三角不等式，理论上仍弱于真正的 Wasserstein 度量。
- **超参依赖**：交点集中度 $c$、方向偏置强度 $\zeta$ 仍需调（虽然论文称归一化后 $c=1$ 稳定），在更复杂数据上是否依然鲁棒需要更多验证。
- **几何中位数的近似误差**：Weiszfeld 只跑固定步数得到近似 $x^\*$，近似质量对最终距离的影响在论文中未深入分析。

## 相关工作与启发
- **vs SW（Sliced Wasserstein）**：SW 只采样方向、用均匀球面分布，无法区分有信息/无信息方向；FW-TSW 在树系统上同时利用方向与位置，且两者都做了数据相关采样，表达力更强。
- **vs TSW-SL / Db-TSW（前代 Tree-Sliced）**：它们已用树系统引入位置，但交点采样中心放在数据**均值**、方向仍是均匀采样，没充分挖掘位置自由度；FW-TSW 把中心换成更稳健的**几何中位数**、方向偏向源—目标位移，并因此得到可控上界。
- **vs 可训练 slicing 分布（如 max-SW 系）**：那类方法靠迭代优化选最优方向，计算昂贵且可能不稳定；FW-TSW 用闭式的几何中位数 + 解析的方向构造，避免了内层优化，稳定且廉价。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"采样策略"这一被忽视的差异重新审视，并用经典 Fermat–Weber 问题给出有原则的解法，视角清新但属于已有框架上的改进。
- 实验充分度: ⭐⭐⭐⭐ 覆盖梯度流、主题建模、扩散模型三类任务，均有稳定增益且开销可忽略；不过多为中小规模基准。
- 写作质量: ⭐⭐⭐⭐ 动机链条清晰，从"SW vs TSW 差异"自然推到几何中位数，理论与实验衔接顺畅。
- 价值: ⭐⭐⭐⭐ 提供了即插即用、几乎零成本的 TSW 采样改进，并附带可证性质，对 sliced-OT 社区实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Slicing Wasserstein over Wasserstein via Functional Optimal Transport](slicing_wasserstein_over_wasserstein_via_functional_optimal_transport.md)
- [\[ICLR 2026\] Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis](best-of-n_through_the_smoothing_lens_kl_divergence_and_regret_analysis.md)
- [\[ICLR 2026\] On Coreset for LASSO Regression Problem with Sensitivity Sampling](on_coreset_for_lasso_regression_problem_with_sensitivity_sampling.md)
- [\[ICLR 2026\] Revisiting Active Sequential Prediction-Powered Mean Estimation](revisiting_active_sequential_prediction-powered_mean_estimation.md)
- [\[ICLR 2026\] Better Bounds for the Distributed Experts Problem](better_bounds_for_the_distributed_experts_problem.md)

</div>

<!-- RELATED:END -->
