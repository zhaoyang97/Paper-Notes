---
title: >-
  [论文解读] SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration
description: >-
  [ICLR 2026][优化/理论][自适应预条件] 本文为 AdaGrad 类自适应预条件 SGD 建立了一个**统一收敛分析框架**：通过把预条件算子约束在一个满足特定假设的算子子空间内，用**一份证明**同时复现 AdaGrad-Norm、AdaGrad、ASGO/One-sided Shampoo 的 SOTA 收敛率，并首次给出 DASGO 的收敛保证；进一步证明对角预条件（AdaGrad、DASGO）可以叠加 Nesterov 动量被**可证加速**，从而首次从理论上解释了 Adam「对角预条件 + 动量」双重机制为何高效。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "自适应预条件"
  - "AdaGrad"
  - "Nesterov动量"
  - "矩阵光滑性"
  - "统一收敛分析"
---

# SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=XhXMzPJJ7J](https://openreview.net/forum?id=XhXMzPJJ7J)  
**领域**: 优化理论 / 自适应梯度方法  
**关键词**: 自适应预条件, AdaGrad, Nesterov动量, 矩阵光滑性, 统一收敛分析

## 一句话总结
本文为 AdaGrad 类自适应预条件 SGD 建立了一个**统一收敛分析框架**：通过把预条件算子约束在一个满足特定假设的算子子空间内，用**一份证明**同时复现 AdaGrad-Norm、AdaGrad、ASGO/One-sided Shampoo 的 SOTA 收敛率，并首次给出 DASGO 的收敛保证；进一步证明对角预条件（AdaGrad、DASGO）可以叠加 Nesterov 动量被**可证加速**，从而首次从理论上解释了 Adam「对角预条件 + 动量」双重机制为何高效。

## 研究背景与动机

**领域现状**：深度学习训练几乎被 Adam / AdamW 垄断，它们的源头是带 AdaGrad-Norm 步长的梯度下降。一大类后续算法（AdaGrad、RMSProp、Shampoo、One-sided Shampoo、ASGO、Muon、Scion、DASGO）都属于「带预条件的梯度方法」，更新规则统一形如 $x_{k+1}=\arg\min_{x}\langle g_k,x\rangle+\tfrac12\|x-x_k\|_{H_k^{-1}}^2$，区别只在预条件算子 $H_k$ 的结构（标量 / 对角 / 矩阵）。

**现有痛点**：每提出一个新的预条件方法，就要单独写一套收敛证明，尽管这些更新规则和证明结构高度雷同。Gupta et al. (2017) 的统一框架部分回答了这个问题（把 $H_k$ 定义为某子空间上的优化解），但仍需为不同算法分别证明，只覆盖非光滑情形，也无法解释一般预条件算子的好处。

**核心矛盾**：自适应优化的理论严重落后于实践——一方面缺一个能同时覆盖标量 / 对角 / 矩阵预条件的统一分析；另一方面，Adam 的两大支柱「对角预条件」和「动量」在理论上从未被证明可以**同时**生效（已有的 AdaGrad 加速结果都只针对标量步长）。

**本文目标**：(Q1) 能否设计一份覆盖 AdaGrad / Shampoo / ASGO 等大多数自适应预条件方法的统一收敛分析？(Q2) 能否设计一个**同时**受益于对角 AdaGrad 预条件和动量的方法？

**切入角度**：作者沿用 Gupta et al. (2017) 把 $H_k$ 限制在算子子空间 $\mathcal{H}\subset\mathcal{S}$ 内的思路，但给该子空间补上两条结构性假设（投影保序 + 对算子函数封闭），从而能**显式解出**预条件算子并接入更强的分析工具；再把光滑性和噪声都用「加权 / 矩阵范数」度量，统一标量、对角、矩阵三种情形。

**核心 idea**：用一个「算子子空间 $\mathcal{H}$ + 势函数 $\phi$」的抽象，把所有 AdaGrad 类预条件器收编为同一套更新与同一份证明；在此之上引入「光滑/噪声算子与预条件器对易」这一条对对角情形自动成立的假设，叠加 Nesterov 动量得到加速率。

## 方法详解

### 整体框架
本文不提出新优化器，而是提出一个**分析框架**：所有 AdaGrad 类方法被统一写成 Algorithm 1（带预条件的自适应 SGD），其唯一可变部分是预条件算子所属的子空间 $\mathcal{H}$。每一步采样随机梯度 $g_k=\nabla f(x_k;\xi_k)$，按统一公式解出 $H_k$，再做一步镜像下降式更新 $x_{k+1}=\arg\min_x\langle g_k,x\rangle+\tfrac12\|x-x_k\|_{H_k^{-1}}^2$，输出迭代平均 $\bar x_K=\tfrac{1}{K+1}\sum x_k$。

框架的精髓在于：**选不同的 $\mathcal{H}$ 就得到不同的已知算法**（见下表），而收敛证明只写一遍。

| 算法 | 空间 $\mathcal{X}$ | 预条件子空间 $\mathcal{H}$ | 等价范数 $R(\cdot)$ |
|------|------|------|------|
| AdaGrad-Norm | $\mathbb{R}^d$ | $\{g\mapsto\beta g:\beta\in\mathbb{R}\}$（标量） | $\tfrac{1}{\sqrt d}\|\cdot\|$ |
| AdaGrad | $\mathbb{R}^d$ | $\{g\mapsto b\odot g:b\in\mathbb{R}^d\}$（对角） | $\|\cdot\|_\infty$ |
| ASGO/One-sided Shampoo | $\mathbb{R}^{m\times n}$ | $\{G\mapsto BG:B\in\mathcal{S}^m\}$（矩阵） | $\tfrac{1}{\sqrt n}\sigma_{\max}(\cdot)$ |
| DASGO | $\mathbb{R}^{m\times n}$ | $\{G\mapsto\mathrm{diag}(b)G:b\in\mathbb{R}^m\}$（行对角） | $\tfrac{1}{\sqrt n}\|\cdot\|_{2\to\infty}$ |

在非加速框架（Algorithm 1）之上，本文再叠一层 Nesterov 动量构成加速框架（Algorithm 2），二者共用同一套预条件器与同一类 FTL-BTL 分析工具。

### 关键设计

**1. 统一预条件框架：把"造一个新优化器"变成"选一个算子子空间"**

针对「每个算法都要重写一份证明」的痛点，作者把预条件器 $H_k$ 限制在算子子空间 $\mathcal{H}\subset\mathcal{S}$ 内，并对 $\mathcal{H}$ 施加两条假设（Assumption 1）：(A1.1) 到 $\mathcal{H}$ 的投影**保序**，即正定算子投影后仍正定；(A1.2) $\mathcal{H}$ 对任意算子函数**封闭**，$\psi(H)\in\mathcal{H}$。在此基础上，$H_k$ 被定义为下式的解：

$$H_k=\arg\min_{H\in\mathcal{H}\cap\mathcal{S}_{++}}\langle H,S_k\rangle+\langle I,\phi(H)\rangle,\quad S_k=\textstyle\sum_{i=0}^{k}g_i\langle g_i,\cdot\rangle$$

其中 $S_k$ 是累积梯度外积（AdaGrad 的核心统计量）。关键在于选取势函数 $\phi(h)=\delta h+\eta^2/h$，配合 Assumption 1 就能把这个抽象优化问题**显式解出**（Lemma 2）：

$$H_k=\eta\big(\delta I+\mathrm{proj}_{\mathcal{H}}(S_k)\big)^{-1/2},\qquad H_{k+1}\preceq H_k$$

这正是 AdaGrad 类「累积梯度平方开根再取逆」的算子版本，且预条件器单调不增。与 Gupta et al. (2017) 相比，这里第一次能写出闭式解，使得后续能借助 FTL-BTL 引理（Lemma 1：$\sum_i\|g_i\|_{H_i}^2\le\langle H_k,S_k\rangle+\langle I,\phi(H_k)\rangle$）把全部算法纳入同一分析骨架。

**2. 矩阵 Hölder 光滑 + 一般化噪声下的统一收敛定理**

要让一份证明覆盖标量/对角/矩阵预条件，光滑性和噪声不能再用标准欧氏范数度量。作者用 Assumption 2 把目标函数设为关于范数 $\|\cdot\|_L$ 的 $(\|L\|_{\mathrm{tr}}^{(1-\nu)/2},\nu)$-**矩阵 Hölder 光滑**（$\nu\in[0,1]$，$L\in\mathcal{H}$）；用 Assumption 3 把噪声方差约束在 $\mathbb{E}\|n(x;\xi)\|_{\Sigma^{-1}}^2\le\|\Sigma\|_{\mathrm{tr}}$（比 An/Xie 等用的排序型假设更弱、更一般）。这两条假设都能翻译成对一个非欧范数 $R(x)=\|\mathrm{proj}_{\mathcal{H}}(X)\|_{\mathrm{op}}^{1/2}$ 的光滑/方差界，而 $R(\cdot)$ 恰好对不同 $\mathcal{H}$ 退化成谱范数、$\|\cdot\|_\infty$、$\|\cdot\|_{2\to\infty}$ 等。

主结果 Theorem 1：取 $\eta=R$（$R$ 几乎必然界住 $\max_k R(x_k-x^*)$），Algorithm 1 输出满足

$$\mathbb{E}[f(\bar x_K)-f(x^*)]\le\frac{3\|L\|_{\mathrm{tr}}R^{1+\nu}}{(K+1)^{(1+\nu)/2}}+\frac{3\|\Sigma\|_{\mathrm{tr}}R}{\sqrt{K+1}}+\frac{3\sqrt{\delta}\,R\,\dim(\mathcal{X})}{K+1}$$

三项分别对应「光滑/确定性项」「随机噪声项」「正则项 $\delta$」。在 $\nu=1$ 光滑情形它复现 AdaGrad（各向异性光滑，Liu et al. 2024b）与 ASGO/One-sided Shampoo 的 SOTA 率；在 $\nu=0$ 非光滑情形比 An et al. (2025) 更一般；并首次覆盖 $0<\nu<1$ 的 Hölder 中间情形。更重要的是，它给出了 **DASGO 的首个收敛保证**，并揭示连接：ASGO/Shampoo↔Muon、DASGO↔Scion（把 $S_k$ 中的梯度累积关掉、只留 $g_k\langle g_k,\cdot\rangle$，就从 DASGO 退化到带 $\|\cdot\|_{2\to\infty}$ 的 Scion）。

**3. Nesterov 动量加速：让对角预条件首次"动量 + 预条件"双收益**

为回答 Q2，作者在统一框架上叠 Nesterov 动量（Algorithm 2），采用 Kovalev-Borodich (2024) 的「时变函数」解读：定义 $f_k(x)=\alpha_k^{-2}f(\alpha_k x+(1-\alpha_k)\bar x_k)$，对它跑预条件 SGD，再用 $\bar x_{k+1}=\alpha_k x_{k+1}+(1-\alpha_k)\bar x_k$ 回插，$\alpha_k=2/(k+2)$。加速分析的关键是 Assumption 4：光滑/噪声算子 $L,\Sigma$ 与预条件子空间 $\mathcal{H}$ **对易**（$LH=HL$）。这条假设对**对角算子自动成立**，因此对 AdaGrad、DASGO 无需任何额外代价。

在对易假设下，$H_k^2$ 本身又是某优化问题的解（Lemma 8），从而能再次套用 FTL-BTL 得到 $\sum_i\|g_i\|_{BH_i^2}^2$ 的对数界（Lemma 9）。最终 Theorem 2：取 $\eta=2R$，

$$\mathbb{E}[f(\bar x_{K+1})-f(x^*)]\le\frac{C_K\|L\|_{\mathrm{tr}}R^{1+\nu}}{(K+2)^{(1+3\nu)/2}}+\frac{C_K\|\Sigma\|_{\mathrm{tr}}R}{\sqrt{K+2}}+\frac{4\sqrt{\delta}\,R\,\dim(\mathcal{X})}{(K+2)^2}$$

其中 $C_K=O(1+\ln K+\ln\tfrac{\|L\|_{\mathrm{tr}}R^\nu}{\sqrt\delta}+\ln\tfrac{\|\Sigma\|_{\mathrm{tr}}}{\sqrt\delta})$ 只含对数因子。对比非加速的 Theorem 1，**光滑项的指数从 $(1+\nu)/2$ 提升到 $(1+3\nu)/2$**（$\nu=1$ 时从 $1$ 升到 $2$，即从 $O(1/K)$ 加速到 $\tilde O(1/K^2)$），噪声项保持 $O(1/\sqrt K)$ 不变（随机下界使然）。这是**首次**证明 AdaGrad/DASGO 能同时受益于对角预条件与 Nesterov 动量，为 Adam 的实际高效提供了理论解释。

### 损失函数 / 训练策略
纯理论分析，无训练损失。算法超参为势函数参数 $\delta,\eta>0$；加速版取 $\eta=2R$、$\alpha_k=2/(k+2)$。Theorem 1/2 要求 $\max_k R(x_k-x^*)\le R$ 几乎必然成立，随机情形可用每步加一个到半径 $R$ 球的投影步骤来保证（Appendix D）。

## 实验关键数据

> ⚠️ 本文是**纯理论论文，没有数值实验**。下面给出的是论文用来对比的**收敛率理论结果**（针对 DASGO/AdaGrad 在各向异性光滑 + 各向异性噪声下，$\delta\ll1$ 简化后的 $\tilde O$ 速率）。

### 主结果：加速 vs 非加速（DASGO / AdaGrad）

| 设置 | 光滑（确定性）项 | 噪声项 | 来源 |
|------|------|------|------|
| 非加速（本文 Thm 1，式 29） | $\dfrac{\|l\|_1\|X^*\|_{2\to\infty}^{1+\nu}}{K^{(1+\nu)/2}}$ | $\dfrac{\|\sigma\|_1\|X^*\|_{2\to\infty}}{\sqrt{K+1}}$ | 复现 AdaGrad（$\nu{=}1,n{=}1$）+ 首给 DASGO |
| 加速（本文 Thm 2，式 30） | $\dfrac{\|l\|_1\|X^*\|_{2\to\infty}^{1+\nu}}{K^{(1+3\nu)/2}}$ | $\dfrac{\|\sigma\|_1\|X^*\|_{2\to\infty}}{\sqrt{K+1}}$ | 光滑项指数 $\uparrow$ |

加速把光滑项从 $K^{(1+\nu)/2}$ 改善到 $K^{(1+3\nu)/2}$（$\nu=1$ 即 $K\to K^2$），噪声项不变。

### 与已有标量 AdaGrad 加速结果对比

| 方法 | 光滑项常数 | 噪声项常数 | 何时本文更优 |
|------|------|------|------|
| 本文对角加速（式 30） | $\|l\|_1$、$\|X^*\|_{2\to\infty}$ | $\|\sigma\|_1$ | — |
| 标量 AdaGrad 加速（Kavis/Rodomanov，式 31） | $\|l\|_\infty$、$\|X^*\|$ | $\sqrt m\,\|\sigma\|_\infty$ | 当 $\|l\|_1\sim\|l\|_\infty$、$\|\sigma\|_1\sim\|\sigma\|_\infty$ 且 $\|X^*\|\gg\|X^*\|_{2\to\infty}$ |

### 关键发现
- **对角 / 矩阵预条件的好处来自"稀疏光滑 + 稠密解"**：当光滑常数向量 $l$、噪声 $\sigma$ 稀疏而最优解 $X^*$ 稠密时，$\|l\|_1\sim\|l\|_\infty$ 但 $\|X^*\|\gg\|X^*\|_{2\to\infty}$，对角预条件显著优于标量步长，与 Liu et al. (2024b) 对 AdaGrad 的结论一致。
- **动量只加速"光滑项"、不改善"噪声项"**：噪声项卡在 $O(1/\sqrt K)$ 是随机一阶方法的统计下界，动量无能为力；这解释了为什么实践中动量在低噪声/大 batch 时收益更明显。
- **一份证明的统一力**：通过切换 $\mathcal{H}$，同一个 Theorem 1 复现了 AdaGrad-Norm / AdaGrad / ASGO / One-sided Shampoo 四个算法各自原本需要单独证明的 SOTA 率。

## 亮点与洞察
- **"选子空间 = 选算法"的抽象非常优雅**：把预条件器约束在算子子空间 $\mathcal{H}$ 并加两条结构假设，就把「标量 / 对角 / 矩阵」三类方法统一进一个闭式解 $H_k=\eta(\delta I+\mathrm{proj}_{\mathcal{H}}(S_k))^{-1/2}$，证明只写一遍。这种「用约束子空间参数化整族算法」的思路可迁移到其他优化器家族分析。
- **"对易假设对对角自动成立"是加速能落地的关键**：Nesterov 加速需要 $L,\Sigma$ 与 $\mathcal{H}$ 对易，而对角算子天然对易——于是 AdaGrad/DASGO 不付任何额外假设代价就拿到加速率，这正好对上 Adam「对角 + 动量」的工程现实。
- **把多个新优化器接进同一张网**：顺手证明了 DASGO↔Scion、ASGO/Shampoo↔Muon 的对应关系（关掉梯度累积即互相转换），把 Muon/Scion 这类非欧优化器纳入同一理论视角。
- **Hölder 光滑 $\nu\in[0,1]$ 全覆盖**：同一定理无缝衔接光滑（$\nu=1$）、非光滑（$\nu=0$）与中间情形，且能自适应到未知的光滑等级（universality）。

## 局限与展望
- **无任何数值实验**：全文是收敛率分析，没有在真实深度学习任务上验证加速版 DASGO/AdaGrad 是否真比 Adam 快，理论与实践之间仍隔一层。
- **凸性假设**：分析建立在凸（含 Hölder 光滑）目标上，而深度学习是非凸的；作者在 Appendix C 讨论了凸假设的合理性，但这仍是与实践的主要 gap。
- **加速依赖对易假设**：Theorem 2 对**非对角**（如矩阵预条件 ASGO/Shampoo）的加速并不自动成立，需要 $L,\Sigma$ 与 $\mathcal{H}$ 对易这一额外条件；一般矩阵预条件能否加速仍开放。
- **需要已知 $R$**：步长取 $\eta\propto R$ 依赖对 $\max_k R(x_k-x^*)$ 的先验界，虽可用投影步骤保证，但实际中 $R$ 未知；与 parameter-free AdaGrad 路线的结合是自然的下一步。
- **展望**：作者建议在 Pethick et al. (2025) 指出的、适合非欧 $\|\cdot\|_{2\to\infty}$ 范数的实际场景中试用 DASGO——它迭代廉价（无矩阵求逆）、自带自适应预条件、理论性质又优于 Scion。

## 相关工作与启发
- **vs Gupta et al. (2017)（统一框架的前身）**：二者都把 $H_k$ 定义为子空间上的优化解，但 Gupta 仍需分算法证明、只覆盖非光滑、无法解释一般预条件的好处；本文靠 Assumption 1 + 势函数 $\phi(h)=\delta h+\eta^2/h$ 拿到**闭式预条件器**和**单一证明**，并覆盖 Hölder 光滑全谱。
- **vs Liu et al. (2024b) / An et al. (2025) / Xie et al. (2025)（各向异性 / 矩阵光滑分析）**：它们对 AdaGrad、ASGO/One-sided Shampoo 分别给出 SOTA 率，但用更强的噪声排序假设、且不覆盖 Hölder 情形；本文用更弱的方差界把它们**统一复现**，并补上 DASGO 的首个保证。
- **vs Trifonov et al. (2025)（唯一一个尝试"预条件 + 动量"的工作）**：他们对预条件器动态做了不现实假设，且只在光滑强凸、非随机设置下分析；本文在随机、矩阵/各向异性 Hölder 光滑下证明了 AdaGrad/DASGO 的加速，是**首个**真正意义上的「对角预条件 + 动量」双收益结果。
- **vs Kavis et al. (2019) / Rodomanov et al. (2024)（标量 AdaGrad 加速）**：已有加速结果都只针对标量步长；本文式 (30) 在「稀疏光滑 + 稠密最优解」下严格优于它们的标量结果式 (31)，把加速推广到对角预条件。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用算子子空间统一整族 AdaGrad 方法，并首次证明对角预条件 + Nesterov 动量的双收益，直击「为什么 Adam 有效」。
- 实验充分度: ⭐⭐ 纯理论论文，零数值实验，加速版是否真在实践中更快未验证。
- 写作质量: ⭐⭐⭐⭐ 假设—引理—定理层层递进，连接 Muon/Scion 的讨论清晰，但符号密度高、对读者数学背景要求高。
- 价值: ⭐⭐⭐⭐⭐ 给 AdaGrad/Shampoo/ASGO/DASGO/Scion/Muon 提供了统一分析视角和 Adam 高效性的理论解释，是优化理论的重要一砖。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](../../ICML2026/optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICLR 2026\] DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD](dnt_a_deeply_normalized_transformer_that_can_be_trained_by_momentum_sgd.md)
- [\[ICML 2026\] Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration](../../ICML2026/optimization/selecting_samples_on_graphs_a_unified_dataset_pruning_framework_for_lossless_tra.md)

</div>

<!-- RELATED:END -->
