---
title: >-
  [论文解读] PAC-Bayes Bounds for Cumulative Loss in Continual Learning
description: >-
  [ICLR 2026][学习理论][累积损失] 这篇论文把现有的在线学习与 time-uniform 离线学习的 PAC-Bayes 上界推广到持续学习场景，给出了**第一个对任意任务分布、任意学习算法都成立的累积损失（学习可塑性）上界**，并在视觉持续学习任务上验证出非空（non-vacuous）的风险证书。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "持续学习"
  - "PAC-Bayes"
  - "累积损失"
  - "学习可塑性"
  - "风险证书"
---

# PAC-Bayes Bounds for Cumulative Loss in Continual Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=hWw269fPov](https://openreview.net/forum?id=hWw269fPov)  
**领域**: 学习理论 / 持续学习 / PAC-Bayes  
**关键词**: 持续学习, PAC-Bayes, 累积损失, 学习可塑性, 风险证书

## 一句话总结
这篇论文把现有的在线学习与 time-uniform 离线学习的 PAC-Bayes 上界推广到持续学习场景，给出了**第一个对任意任务分布、任意学习算法都成立的累积损失（学习可塑性）上界**，并在视觉持续学习任务上验证出非空（non-vacuous）的风险证书。

## 研究背景与动机
**领域现状**：持续学习要在「记住旧任务」和「学好新任务」之间权衡，对应记忆稳定性（memory stability）与学习可塑性（learning plasticity）这对矛盾。已有大量经验算法（EWC 正则、回放、变分推断等）来缓解灾难性遗忘，社区也越来越想给这些算法配上**可靠的风险证书**（risk certificate）。

**现有痛点**：理论侧的进展远落后于经验侧。已有的理论结果要么只刻画**遗忘**（forgetting / memory stability）的行为，要么局限于很窄的设定——比如连续线性回归、NTK（极宽网络）regime，并且往往绑定某个具体优化算法（SGD、OGD）。对**学习可塑性**这一侧、尤其是普适的上界，几乎是空白。

**核心矛盾**：度量持续学习有两类量。**平均损失**（average loss，对应平均准确率）是用最终的后验 $Q_T$ 回头去测所有任务，每来一个新任务要从头重算、还要存下每个任务的数据；**累积损失**（cumulative loss, CuL）则是顺着学习过程把每个任务当下的测试误差累加起来，对应前向迁移（forward transfer）、intransigence、学习可塑性。已有 PAC-Bayes 工作（meta-learning、Friedman & Meir 2025 的遗忘/平均误差界）都不覆盖 CuL。

**本文目标**：给持续学习的累积损失推出**算法无关、架构无关、任务结构无关**的高概率上界，并进一步分析这些界在不同任务相似度/任务顺序下能给出多紧的保证。

**切入角度**：作者注意到在线学习的 PAC-Bayes 框架（Haddouche & Guedj 2022/2023、Chugg et al. 2023 的 time-uniform 界）天然带有「后验随数据序列演化」的结构，只要把每个「任务」从单个样本扩成 $m$ 个 i.i.d. 样本，就能把这套机器搬到持续学习上。

**核心 idea**：用 **online predictive sequence + 随机核（stochastic kernel）** 把持续学习过程形式化，再把 Markov 不等式 / change-of-measure / Hoeffding 这套 PAC-Bayes 三件套搭配「坏事件分析」和「拉普拉斯方法」，得到既对一般假设类成立、又能在 Gibbs 后验下退化成 oracle 界的累积损失上界。

## 方法详解

### 整体框架
论文是纯理论工作，主线是「先把持续学习写成一个能套 PAC-Bayes 的概率框架 → 在该框架上推一般上界 → 再针对 Gibbs 后验推更可解释的 oracle 界 → 用 oracle 界对比不同任务场景」。

形式化的设定如下：数据空间 $Z = X \times Y$，固定每任务样本量 $m$，任务序列长度 $T$ 任意。第 $t$ 个任务对应未知分布 $D_t$，给定一份 i.i.d. 样本 $S_t \sim D_t^m$。学习器维护一串先验 $(P_t)$ 和一串后验 $(Q_t)$：先验从 data-free 的 $P_1=P$ 开始，关键设定是把**上一个任务的后验当作下一个任务的先验**（$P_t = Q_{t-1}$），于是 KL 项变成「相邻任务后验的变化量」。整个过程沿一个适配的 filtration $(F_t)$ 演化，用随机核（把数据集 $S$ 映到 $H$ 上的概率测度）来允许数据依赖的先验。要上界的目标量是累积损失

$$\mathrm{CuL}((Q_t)_{t=1}^{T}) = \sum_{t=1}^{T} \mathbb{E}_{h_t \sim Q_{t,S_{1:t}}}\big[\mathbb{E}_{z_t \sim D_t}[\ell(h_t, z_t)\mid F_{t-1}]\big],$$

即每个任务在「学完它当下」的期望测试误差之和。框架同时覆盖离线持续学习（每任务样本成批给）和在线持续学习（样本逐个到达、不可复用）。

### 关键设计

**1. 持续学习的 PAC-Bayes 形式化：用 online predictive sequence 把过程写成可套界的概率结构**

要给 CuL 推界，先得有一套能描述「后验随任务序列演化」的语言。作者把 Haddouche & Guedj 的在线框架做了关键改造：原框架里每个时间步只有一个样本，这里把每步换成一个 $m$ 样本的任务。核心对象是 **online predictive sequence**——一串随机核 $(P_t)$，要求 $P_t(S,\cdot)$ 对 $F_{t-1}$ 可测（只能用到此前任务的信息）且对 $P_{t-1}$ 绝对连续。这保证了「先验只看过去、后验可用当前数据」的因果结构，足以描述各种持续学习算法。基于此作者严格区分了 CuL、平均损失 AL 和 meta-learning 损失三者：AL 用 $Q_T$ 回测全部任务、需正比于任务数的额外内存；MetaL 测未来未见任务；唯有 CuL 顺序可加、来新任务只需追加一项，正是刻画可塑性的对象。这个形式化是后面一切界的地基，也是把在线 PAC-Bayes 机器接到持续学习上的接口。

**2. 累积损失的一般上界：从不收敛的朴素界到随任务数收敛的 $\sqrt{T}$ 法则**

先把经典 Catoni 界推广到 $m>1$ 且 sub-Gaussian 损失，得到 Corollary 3.1：

$$\tfrac{1}{T}\mathrm{CuL} \le \tfrac{1}{T}\sum_{t=1}^{T}\hat{L}(Q_t, S_t) + \tfrac{1}{\lambda T}\sum_{t=1}^{T}\mathrm{KL}(Q_t\|P_t) + \tfrac{\lambda K^2}{m} + \tfrac{\log(1/\delta)}{\lambda T}.$$

它形式干净，但右端有个 $\lambda K^2/m$ 项不随任务数 $T$ 衰减——在持续学习里我们更想要「即便每任务样本固定、随任务变多也收敛」的界。为此作者额外假设损失有界 $\ell\in[0,K]$、且每任务样本量不远小于任务数（$m \gg \sqrt{T}$），通过对任务/后验依赖关系的精细分析加上**坏事件分析**（bad-event analysis，PAC-Bayes 里不常用的技巧），得到 Theorem 3.2。取 $\lambda = T\sqrt{T}/K$、$\delta_2 = e^{-T\sqrt{T}}$ 后化简为可读的式（3）：

$$\tfrac{1}{T}\mathrm{CuL} \le \tfrac{1}{T}\sum_{t=1}^{T}\hat{L}(Q_t,S_t) + \tfrac{K}{T\sqrt{T}}\sum_{t=1}^{T}\mathrm{KL}(Q_t\|Q_{t-1}) + \tfrac{K}{4\sqrt{T}}\sqrt{2m} + \tfrac{K(1+\log(1/\delta))}{T\sqrt{T}}.$$

只要 $m > \sqrt{T}/2$，右端就随 $m,T\to\infty$ 收敛到左端。这条不仅给出风险证书，还顺带产出一条实用经验法则：**每任务样本数应超过总任务数的平方根**。论文举例说，若每天重训模型、持续一年，只需几十个样本就能给累积损失提供有效的高概率证书。

**3. Gibbs 后验下的 oracle 界：消掉 KL 项，把累积损失界成「过去任务训出的预测器」的损失**

一般界里的 KL 复杂度项对大模型不好估，作者转向 **Gibbs 后验** $\hat{Q}_t^\lambda(h) \propto e^{-\lambda\hat{L}(h,S_t)}\hat{Q}_{t-1}^\lambda(h)$（及其期望版本）。Gibbs 后验的好处是能把式（1）右端的 KL 项整体消掉，于是可对累积损失推 oracle 上界。在 $H$ 为紧致有界子集、总期望损失有严格全局极小且二阶可微等假设下，Theorem 4.1 给出

$$\lim_{m,T\to\infty}\tfrac{1}{T}\mathrm{CuL}((Q_t^\lambda)) \le \lim_{T\to\infty}\tfrac{1}{T}\sum_{t=2}^{T} L(h^*_{1:t-1}, D_t),$$

其中 $h^*_{1:t-1}$ 是前 $t-1$ 个任务总损失的最优解。直觉上：当下任务的累积损失，被「用之前所有任务训出的最优预测器」在当下任务上的损失所上界。换不同的极小假设还能得到 Corollary 4.2 的变体。这个证明主要靠**拉普拉斯方法**（Laplace's method）——在 PAC-Bayes 里也很少见。oracle 界的价值在于它把抽象的累积损失翻译成「前序最优预测器迁移得好不好」，从而能对任务相似度做定量比较。

**4. 任务相似度与顺序的场景分析：用 oracle 界解释「任务怎么排」对累积误差的影响**

有了 oracle 界，作者在 Lipschitz 损失（任务分布的相似度通过 Wasserstein 距离反映到损失上）和「过参数化可同时拟合两任务」等假设下，对四类典型场景给出显式界：①所有任务同分布时 $\frac{1}{T}\mathrm{CuL}\le L_1^* + O(1/T)$；②两分布交替出现时，若维度 $d\ge d_1+d_2$ 可达最优 $\frac{L_1^*+L_2^*}{2}$，否则多出一项正比于任务距离 $G_H d(D_1,D_2)$；③前半段一个分布、后半段另一个分布（单次切换）；④任务逐渐漂移（相邻距离 $\le r$、整体半径 $\le \phi$）时多出 $rG_H$ 项。结论是**任务顺序在容量受限时会显著影响累积误差**（单次切换优于反复交替），而**充分过参数化的模型能基本抹平任务顺序的影响**。这一节把前面抽象的界落到可对比的具体情形，给出了关于前向迁移、模型复杂度、任务相似度三者关系的洞见。

## 实验关键数据

### 主实验（视觉持续学习，验证一般界）
$T=120$ 个任务、CNN 模型、5 个随机种子，报告平均累积误差 $\frac{1}{T}\mathrm{CuL}$ 与式（3）上界（误差百分比，越低越好）：

| 数据集 | 方法 | CuL | 上界(eq.3) | Error@t=120 | 上界@t=120 |
|--------|------|-----|-----------|-------------|-----------|
| Perm.-MNIST | EWC | 1.0 | 10.6 | 1.0 | 5.1 |
| Perm.-MNIST | VI | 15.5 | 17.9 | 4.7 | 6.8 |
| Split-MNIST | EWC | 0.9 | 4.2 | 0.9 | 2.5 |
| Split-MNIST | VI | 17.6 | 19.4 | 5.1 | 7.5 |
| Split-CIFAR10 | EWC | 34.4 | 47.9 | 34.7 | 39.7 |
| Split-CIFAR10 | VI | 49.8 | 52.2 | 49.5 | 52.5 |

Split-ImageNet（随机模型约 98% 累积误差）：

| 方法 | CuL | 上界(eq.3) | Forgetting |
|------|-----|-----------|-----------|
| EWC | 33.5 | 40.1 | 5.7 |
| SGD | 34.8 | 41.4 | 7.8 |
| Replay | 55.7 | 62.5 | 2.8 |

### 关键发现
- **界对 VI 很紧、对 EWC 偏松**：因为 VI 直接以式（3）右端为优化目标，而 EWC 的训练过程和界没有直接挂钩。
- **任务越多界越紧**：式（3）多项随 $T$ 衰减，且 KL 项随过程下降——上一任务的后验当下一任务先验时，会越来越成为「信息充分的好先验」。除 Split-CIFAR10 的 VI 外，所有经验上界都是非空的，靠后的任务尤其能给出有用的风险证书。
- **早期任务界偏松**：可能源于前几个任务训练过程的随机性；这反而符合「持续重训复杂模型、迁移偏向新任务泛化」的直觉。
- **oracle 界验证任务顺序效应**（线性回归 + SGLD 近似 Gibbs 采样）：单次任务切换的累积损失明显优于反复交替，与理论一致；过参数化 regime 下累积误差近乎常数、上界几乎精确；该设定下界比基于 NTK 的 SGD 泛化界紧好几个数量级。

## 亮点与洞察
- **第一个学习可塑性的普适上界**：以往持续学习理论多聚焦遗忘/平均误差且绑定特定设定，这篇是累积损失方向上首个算法/架构/任务无关的高概率上界，补上了可塑性侧的空白。
- **「上一任务后验当下一任务先验」是点睛之笔**：让 KL 项变成相邻后验的变化量，既符合持续学习「无法回看旧数据」的约束，又解释了为何随任务推进界会自动变紧。
- **$m>\sqrt{T}$ 法则可直接落地**：从纯理论推导里掉出一条「每任务样本数应超过任务数平方根」的可操作经验法则，对实际持续重训系统有指导意义。
- **方法工具新颖**：在 PAC-Bayes 中较少用的坏事件分析、拉普拉斯方法被组合进来，为后续推导收敛型/ oracle 型界提供了可复用的技术模板。

## 局限与展望
- 假设损失有界或 sub-Gaussian；重尾损失需另行扩展（可借鉴 Haddouche & Guedj 2023）。
- 严格全局极小的假设可放宽到有限个全局极小；oracle 界目前是对期望 Gibbs 后验取的，经验 Gibbs 后验需改写极小假设。
- 离线持续学习里累积误差往往不如平均误差相关（常假设每任务样本无限、可从头学），且在线场景若任务边界模糊，$m$ 和 $T$ 需要估计才能用界。
- 一般界含 KL 复杂度项，对大模型难以缩放（oracle 界无此问题）；可考虑换其他散度度量或模型压缩界来缓解。

## 相关工作与启发
- **vs Friedman & Meir (2025)**：他们给的是持续学习**平均误差/遗忘**的 PAC-Bayes 上界，本文转向**累积损失**这条刻画可塑性的线，互补。
- **vs NTK/SGD 理论界（Bennani & Sugiyama 2020 等）**：那些界绑定 NTK regime 与具体优化算法，本文的界算法/架构无关，且实验中比 NTK-based 界紧好几个数量级。
- **vs meta-learning PAC-Bayes（Pentina & Lampert 2015、Amit & Meir 2018、Balcan et al. 2019）**：meta-learning 可回看旧任务数据、任务来自单一任务生成分布；follow-the-leader 类方法需访问历史数据，难直接用于持续学习。本文界同时覆盖离线与在线持续学习，不需回看旧数据。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个累积损失/学习可塑性的普适 PAC-Bayes 上界。
- 实验充分度: ⭐⭐⭐⭐ 视觉任务验证非空界 + 线性回归验证 oracle 界，覆盖到位但规模偏小。
- 写作质量: ⭐⭐⭐⭐ 框架与定理层层递进，符号严谨；细节多放附录。
- 价值: ⭐⭐⭐⭐⭐ 给持续学习算法提供可计算的风险证书，并产出可落地的 $\sqrt{T}$ 经验法则。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] On the Bayes Inconsistency of Disagreement Discrepancy Surrogates](on_the_bayes_inconsistency_of_disagreement_discrepancy_surrogates.md)
- [\[ICLR 2026\] Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity](barriers_for_learning_in_an_evolving_world_mathematical_understanding_of_loss_of.md)
- [\[ICLR 2026\] Multi-Synaptic Cooperation: A Bio-Inspired Framework for Robust and Scalable Continual Learning](multi-synaptic_cooperation_a_bio-inspired_framework_for_robust_and_scalable_cont.md)

</div>

<!-- RELATED:END -->
