---
title: >-
  [论文解读] Contrastive Predictive Coding Done Right for Mutual Information Estimation
description: >-
  [ICLR2026][自监督学习][对比学习] 这篇论文从理论上戳穿了"InfoNCE 是互信息估计器"这个流传已久的误解——它其实是另一种散度（K-way JSD）的变分下界，永远逼不到 KL 散度；作者用一个加"锚类"的简单改动（InfoNCE-anchor）让 critic 直接学到无歧义的密度比，得到低偏差、低方差的即插即用 MI 估计器，并用 proper scoring rule 把 NCE / InfoNCE / f-散度一族对比目标统一进同一框架。
tags:
  - "ICLR2026"
  - "自监督学习"
  - "对比学习"
  - "互信息估计"
  - "InfoNCE"
  - "密度比估计"
  - "proper scoring rule"
---

# Contrastive Predictive Coding Done Right for Mutual Information Estimation

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=JodkBXWgbA](https://openreview.net/forum?id=JodkBXWgbA)  
**代码**: 待确认  
**领域**: 自监督 / 表示学习  
**关键词**: 对比学习, 互信息估计, InfoNCE, 密度比估计, proper scoring rule  

## 一句话总结
这篇论文从理论上戳穿了"InfoNCE 是互信息估计器"这个流传已久的误解——它其实是另一种散度（K-way JSD）的变分下界，永远逼不到 KL 散度；作者用一个加"锚类"的简单改动（InfoNCE-anchor）让 critic 直接学到无歧义的密度比，得到低偏差、低方差的即插即用 MI 估计器，并用 proper scoring rule 把 NCE / InfoNCE / f-散度一族对比目标统一进同一框架。

## 研究背景与动机
**领域现状**：InfoNCE（van den Oord 等，2018）最初是为对比表示学习提出的，但原文附录顺手把它解释成互信息（MI）的一个变分下界，于是社区就把它当成了 MI 估计的默认工具，被 Poole、Song & Ermon、Gowri 等一大批后续工作沿用。

**现有痛点**：大家其实早就知道 InfoNCE 给出的 MI 下界很松——它被普遍描述为"低方差但高偏差"的估计器：当真实 MI 较大时，受 $\log K$（$K$ 是负样本数 / batch 内对比数）的天花板压制，估出来的值系统性偏低。但"为什么有效（低方差）"和"为什么受限（高偏差）"这两件事一直没被讲清楚，社区只是经验性地接受了它的缺陷。

**核心矛盾**：问题的根本不在"$K$ 不够大"，而在于 InfoNCE 优化的目标压根就不是 KL 散度的变分表示。作者证明：即便 $\log K \ge D(q_1\|q_0)$，InfoNCE 的目标值对任何有限 $K$ 都严格小于 KL 散度，所以再增大 $K$ 也填不平这个差。更要命的是，InfoNCE 学到的 critic 只能把密度比 $\frac{p(x,y)}{p(x)p(y)}$ 估计到"差一个任意函数 $C(y)$"的程度，无法被代入即插即用（plug-in）估计器。

**本文目标**：(1) 厘清 InfoNCE 到底在度量什么散度、和真正的 MI 差多远；(2) 给出一个最小改动，让 critic 能**无歧义地**学到密度比本身，从而支持低偏差的 plug-in MI 估计；(3) 把这个改动放进一个更一般的理论框架里，统一已有的一众对比目标。

**切入角度**：作者把密度比估计重述成一个**张量化（tensorized）的分类问题**，并往里塞进一个"锚类"作为固定参照分布。一旦有了这个不动的参照，密度比的乘性歧义就被钉死了。

**核心 idea**：给 InfoNCE 的分类设定加一个全部来自噪声分布 $q_0$ 的"锚类（class 0）"——它充当固定参照，消除任意缩放，使 critic 直接、一致地估到 $\frac{q_1(x)}{q_0(x)}$，由此得到的 plug-in 估计器既保留 InfoNCE 的低方差，又大幅降低其偏差。

## 方法详解

### 整体框架
论文的逻辑链是"先解构、再重建、最后统一"三步。**第一步**先把现有变分界 MI 估计器分成三类，定位 InfoNCE 的病灶；**第二步**给出 InfoNCE 目标的精确刻画（它是 K-way JSD 的紧下界而非 KL），说明它为什么不能做 plug-in；**第三步**提出 InfoNCE-anchor 修复它，并用 proper scoring rule 把整族对比目标收编进一个统一框架。

先看作者给出的**估计器三分类法**（论文 Table 1），这是理解全文的地图：

- **Type 1（训练、评估同用一个变分下界）**：DV、NWJ、InfoNCE。简单自然，但 McAllester & Stratos（2020）证明任何 distribution-free 的高概率 MI 下界都被 $\log N$ 卡住，存在样本量天花板。
- **Type 2（训练用一个下界、评估代入另一个下界）**：MINE、JS、SMILE。训练更稳，但优化目标和评估目标之间引入了失配，$\log N$ 的批评依旧适用。
- **Type 3（训练去学密度比、评估用 plug-in）**：PCC/D-RFC、f-DIME、JSD-LB，以及本文的 InfoNCE-anchor。直接学密度比 $\frac{p(x,y)}{p(x)p(y)}$，再把它代进 MI 的定义算值，把"密度比学习"和"具体下界"解耦，从而绕开变分下界的固有限制、有望得到更低偏差。

InfoNCE 之所以被困在 Type 1，正是因为它学不到无歧义的密度比，没法升级到 Type 3；InfoNCE-anchor 的全部价值就是把它"扶"进 Type 3。

### 关键设计

**1. 精确刻画 InfoNCE：它是 K-way JSD 的紧下界，不是 KL 的变分表示**

要修一个东西，先得说清它坏在哪。作者把 InfoNCE 抽象成对比两个分布 $q_1$（正样本分布，特化时取 $p(x|y)$）和 $q_0$（噪声分布，取 $p(x)$）：critic $r_\theta$ 给正样本 $x_1\sim q_1$ 打高分、给 $K-1$ 个负样本打低分，目标是
$$D_{\text{InfoNCE}}(\theta)=\mathbb{E}\Big[\log\frac{r_\theta(x_1)}{\tfrac{1}{K}\sum_{z=1}^{K}r_\theta(x_z)}\Big].$$
经典结论（Proposition 1）只给出宽松的 $D_{\text{InfoNCE}}(\theta)\le\min\{\log K,\,D(q_1\|q_0)\}$。本文的 Theorem 2 给出**更紧**的上界：$D_{\text{InfoNCE}}(\theta)\le D_{K\text{-JS}}(q_1,q_0)$，其中 $D_{K\text{-JS}}$ 是 Jensen–Shannon 散度的 $K$ 路推广（K-way JSD），且**当且仅当** $r_\theta(x)\propto \frac{q_1(x)}{q_0(x)}$ 时取等。

这一刀切中两个要害。其一，既然只在 $r_\theta\propto q_1/q_0$（差一个乘性常数）时才取等，InfoNCE 学到的 critic 天生带乘性歧义，不能直接代进 plug-in 估计器。其二，作者用具体数字戳破"$K$ 够大就行"的幻觉：设 $D(q_1\|q_0)=2$ bits，则 $K=4$ 时 $D_{\text{InfoNCE}}\le 1.19$，即使 $K=64$ 也只到 $\le 1.93$，对任何有限 $K$ 都严格小于 KL 的 $2$。所以 InfoNCE **根本不是 KL 散度的变分表示**——这与 DV、NWJ 形成鲜明对比，后两者在 critic 等于真实对数密度比时下界会取到紧。

**2. InfoNCE-anchor：加一个锚类，把密度比的乘性歧义钉死**

病根是"密度比只学到差一个常数 / 函数"，那就想办法把这个自由度去掉。作者把密度比估计重写成一个 $X^K$ 上的 $(K+1)$ 类分类问题（张量化），关键是多出一个 **class 0（锚类）**：
$$\text{class }0:\ q_0(x_1)q_0(x_2)\cdots q_0(x_K),\qquad \text{class }z\ (z\in[K]):\ q_1(x_z)\prod_{i\ne z}q_0(x_i).$$
即 class $z$ 表示"第 $z$ 个样本来自 $q_1$、其余都来自 $q_0$"，而 class 0 表示"全都来自 $q_0$"。类先验取 $p(z{=}0)=\frac{\nu}{K+\nu}$、$p(z)=\frac{1}{K+\nu}\,(z\in[K])$，其中 $\nu\ge 0$ 控制锚类权重。锚类的意义在于：它是一个**固定不动的参照分布**，按贝叶斯公式算后验时，分母里多出的常数项 $\nu$ 把缩放自由度锁死了。后验为
$$p(z|x_{1:K})=\begin{cases}\dfrac{\nu}{\nu+\sum_i \frac{q_1(x_i)}{q_0(x_i)}}, & z=0\\[2mm]\dfrac{q_1(x_z)/q_0(x_z)}{\nu+\sum_i \frac{q_1(x_i)}{q_0(x_i)}}, & z\in[K].\end{cases}$$
照此把分类器参数化为 $p_\theta(z|x_{1:K})$（用 $r_\theta(x_i)$ 替换 $q_1(x_i)/q_0(x_i)$），再做极大似然，就得到 **InfoNCE-anchor 目标**：
$$L_{K;\nu}(\theta)=-\tfrac{K}{K+\nu}\mathbb{E}\Big[\log\tfrac{r_\theta(x_1)}{\nu+\sum_i r_\theta(x_i)}\Big]-\tfrac{\nu}{K+\nu}\mathbb{E}\Big[\log\tfrac{\nu}{\nu+\sum_i r_\theta(x_i)}\Big].$$
对比原 InfoNCE，它只是多了第二项（锚类的对数似然），实现上额外开销可忽略。**Fisher 一致性（Theorem 3）**给出了这个设计的保证：当 $\nu>0$ 时，全局最优 $r_{\theta^*}(x)=\frac{q_1(x)}{q_0(x)}$（在 $q_0$ 几乎处处意义下，**精确、无常数**）；而 $\nu=0$ 且 $K\ge2$ 时退化回 InfoNCE，只能学到 $C\cdot\frac{q_1(x)}{q_0(x)}$。一个 $\nu$ 的开关，就是"能不能做 plug-in"的分水岭。它也统一了若干旧目标：$K{=}2,\nu{=}0$ 对应 ranking NCE，$K{=}1,\nu{=}1$ 对应标准 NCE / JS 变分下界。

**3. proper scoring rule 推广：把一整族对比目标收编进同一框架**

上面的交叉熵（log score）只是估类后验的一种损失。作者指出，一旦把密度比估计重述成"类概率估计（CPE）"，**任何严格 proper 的 scoring rule** 都能产生一个一致的密度比估计目标。对一个严格凸、可微的生成函数 $\Psi$，可定义其诱导的 scoring rule $\lambda^\Psi$，并把总体目标写成 $L^\Psi_{K;\nu}(\eta_\theta)=\mathbb{E}_{p(x_{1:K},z)}[\lambda_z(\eta_\theta(x_{1:K}))]$。**Theorem 6** 用 Bregman 散度刻画了它与最优解的差距：对 $\nu>0$，
$$L^\Psi_{K;\nu}(\eta_\theta)-L^\Psi_{K;\nu}(\eta^*)=\tfrac{\nu}{K+\nu}\,\mathbb{E}_{q_0}\!\Big[B_\Psi\big(\tfrac{r^*(x_{1:K})}{\nu},\tfrac{r_\theta(x_{1:K})}{\nu}\big)\Big],$$
当且仅当 $r_\theta(x)=\frac{q_1(x)}{q_0(x)}$ 时取等。log score 是其中的典范实例、恰好给出 InfoNCE-anchor；而 $K{=}1,\nu{=}1$ 时框架退化为 $f$-散度的标准变分下界，从而把 f-DIME、f-MICL 等也纳进来。这一步的价值是"统一视角"：NCE、InfoNCE、$f$-散度变体被证明只是同一密度比估计框架在不同 scoring rule / 不同 $(K,\nu)$ 下的特例。作者还顺带给出 $K\to\infty$ 的渐近性质（Theorem 10）：令 $\nu/K\to\beta$，InfoNCE-anchor 在 DV 界（$\beta{=}0$，最紧）与 NWJ 界（$\beta{\to}\infty$，最松）之间单调插值。

### 损失函数 / 训练策略
实践中 critic 参数化为 $r_\theta(x,y)=e^{c_\theta(x,y)}$（$c_\theta$ 是神经网络），或表示学习里常用的指数/直接余弦形式 $r_\theta(x,y)=e^{\frac{1}{\tau}\frac{f_\theta(x)^\top g_\theta(y)}{\|f_\theta(x)\|\|g_\theta(y)\|}}$，$\tau$ 为温度。给定 batch 大小 $B$，取 $K=B-1$、默认 $\nu=1$，损失就是把上面两项写成 batch 内求和的形式，第二项（锚类）相对 InfoNCE 几乎不增成本。全程不调超参，默认设定即稳定优于各 baseline。

## 实验关键数据

### 主实验
作者在三块上验证：合成/真实 MI 估计基准、蛋白互作预测、自监督表示学习（其中第三块是有意呈现的**负结果**）。

蛋白互作预测（用预训练 pLM 的 PMI 做阈值判定是否互作，AUROC，20 次平均）：

| 目标 | Kinase (AUROC) | Ligand (AUROC) |
|------|------|------|
| LMI | 0.74 ±0.08 | 0.87 ±0.04 |
| χ² (DRF) | 0.76 ±0.07 | 0.92 ±0.03 |
| JS（=anchor 的 $K{=}1,\nu{=}1$ 特例） | 0.77 ±0.08 | 0.95 ±0.02 |
| **InfoNCE-anchor** | **0.80 ±0.06** | **0.97 ±0.01** |
| Spherical | 0.73 ±0.07 | 0.87 ±0.05 |

InfoNCE-anchor 两个任务都最优，JS 次优，再次印证"大 $K$ 对准确密度比估计的价值"。标准 InfoNCE 因带未知 $C(y)$ 在此场景直接不适用。MI 估计基准（Gaussian / MNIST / IMDB-BERT 文本，MI 从 2 到 10 bits）上，InfoNCE-anchor（log score）一致呈现低偏差、低方差，是几张图里唯一稳定贴近真值的估计器。

### 消融实验
自监督表示学习（ResNet-18 在 CIFAR-100 上预训练，$B{=}256$，线性探针）——这里是关键的负结果：

| 目标 | Top-1 (%) | Top-5 (%) | 说明 |
|------|------|------|------|
| InfoNCE | **65.98** | **89.69** | 标准对比目标，表示最强 |
| InfoNCE-anchor | 65.74 | 89.24 | 加锚类后 MI 估计变准，但下游**没涨** |
| χ² | 65.59 | 88.4 | 与 InfoNCE 相当 |
| JS（小 $K$） | 61.69 | 87.33 | 明显偏弱，凸显大 $K$ 的重要性 |
| Spherical | 4.33 | 17.91 | 几乎崩溃，优化几何不利 |

### 关键发现
- **MI 估准 ≠ 表示更好**：锚类让密度比可识别、MI 估计 SOTA，但 SSL 线性探针几乎不变（65.74 vs 65.98）。说明 InfoNCE 里那个未知乘性因子 $C(y)$ 对表示学习"要么近似常数、要么无关紧要"。
- **表征结构分析佐证**：InfoNCE 与 anchor 学到的特征 CKA ≈0.88（backbone 与 projector 都是），全局几何几乎一致；Uniformity 略升（0.350 vs 0.357）但 Alignment 略降（-3.77 vs -3.81），互相抵消。
- **scoring rule 的差异来自优化、不来自一致性**：所有严格 proper scoring rule 在总体/非参极限下都 Fisher 一致，但 Spherical 在 SSL 直接崩到 4.33%，训练目标几乎立刻平台化——差异源自优化地形（optimization landscape）而非统计不一致；log score 是唯一可靠工作的实例。
- **结论**：对比学习的收益来自"学到结构化密度比 / 对 PMI 做因式分解 + 大 $K$ + log score 的良好优化性质"，而**非**精确估计 MI。

## 亮点与洞察
- **把"误解"证伪到数字层面**：不止定性说 InfoNCE 是松界，而是给出 Theorem 2 的紧上界并算出"$D{=}2$ 时即便 $K{=}64$ 也只到 1.93"，把"再大 $K$ 也补不平"这件事钉死，澄清了流传多年的"表示学习=最大化 MI"的误读。
- **一个 $\nu>0$ 的开关解决可识别性**：锚类作为固定参照消除乘性歧义，思路极简却恰好对应 Fisher 一致性的分水岭——是"少即是多"的漂亮设计，且实现几乎零额外成本。
- **统一视角可迁移**：用 proper scoring rule + 张量化分类把 NCE / InfoNCE / f-散度收进一个框架，这套"密度比估计 = 类概率估计"的语言，可复用到任意需要无歧义密度比的下游（如本文的蛋白互作判定、统计检验、因果/独立性度量）。
- **诚实的负结果**：作者主动报告锚类在 SSL 上不涨点，并用 CKA / Uniformity-Alignment 解释为何"估准 MI"与"表示质量"解耦——这种自我证伪比一味报喜更有信息量。

## 局限与展望
- **SSL 上无增益**：核心改动在表示学习的主战场上没带来下游收益，价值主要落在"需要准确 MI / 密度比数值"的场景（估计、检验、科学计算），而非刷 SSL benchmark。
- **scoring rule 的优化地形未刻画**：为什么 log score 行、Spherical 崩，作者只观察到现象（目标立刻平台化），承认"完整刻画成功与失败实例的地形性质超出本文范围"，这是留给后续的硬骨头。
- **结论的适用边界**：负结果基于 ResNet-18 / CIFAR-100 / 线性探针这一设定，是否在更大模型、更大 batch、更难任务上仍然"MI 估准与表示质量解耦"，需要更广的验证（作者附录用了不同 backbone，但仍有限）。
- **改进思路**：把"良好优化性质"显式化——既然差异来自优化地形，或可设计在大 batch、低噪下地形更友好的 scoring rule，让"估得准"与"表示好"重新对齐。

## 相关工作与启发
- **vs InfoNCE（van den Oord 等，2018）**：本文证明 InfoNCE 是 K-way JSD 的紧下界、非 KL 变分表示，critic 只能学到差一个 $C(y)$ 的密度比；InfoNCE-anchor 通过加锚类去掉这个歧义，是 InfoNCE 的"正确做法"。
- **vs α-InfoNCE（Poole 等，2019）/ MLInfoNCE（Song & Ermon，2020）**：作者指出 α-InfoNCE 此前被声称是 α-skew KL 的紧下界的证明有漏洞，实际会超过该散度；MLInfoNCE 虽满足 $\le D(q_1\|q_0)$，但无法被理解为来自一个正当分类设定的损失——而 InfoNCE-anchor 天然源自一个合法的 MLE 分类问题。
- **vs Type 3 plug-in 方法（PCC/D-RFC、f-DIME、JSD-LB）**：它们都直接学密度比再 plug-in，本文用 proper scoring rule 把这些（以及 $K{=}1,\nu{=}1$ 对应的 f-散度变体）统一为同一框架的特例，并指出 GAN-DIME / HD-DIME 实质等同于 Tsai 等的估计器。
- **vs Wang & Isola（2020）的 Uniformity-Alignment 视角**：本文借其指标分析锚类对特征几何的影响，并把"对比学习受益于结构化密度比而非精确 MI"这一论点落到了表征结构的可测量证据上。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把广泛误用的 InfoNCE-as-MI 从理论上证伪，并给出最小且自洽的修复 + 统一框架
- 实验充分度: ⭐⭐⭐⭐ 三类场景验证充分且含诚实负结果，但 SSL 设定规模偏小、scoring rule 优化地形未深挖
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰、taxonomy 与定理层层递进，论点与反例配合到位
- 价值: ⭐⭐⭐⭐ 对"MI 估计"社区是重要澄清与新 SOTA 工具，对 SSL 实践的直接增益有限但概念校正价值大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bidirectional Predictive Coding](bidirectional_predictive_coding.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](maximizing_incremental_information_entropy_for_contrastive_learning.md)
- [\[CVPR 2026\] Robust Spiking Neural Networks by Temporal Mutual Information](../../CVPR2026/self_supervised/robust_spiking_neural_networks_by_temporal_mutual_information.md)
- [\[ICLR 2026\] Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning](self-predictive_representations_for_combinatorial_generalization_in_behavioral_c.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)

</div>

<!-- RELATED:END -->
