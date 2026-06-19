---
title: >-
  [论文解读] Persuasive Privacy
description: >-
  [ICML 2026][AI安全][差分隐私] 本文用 Sender–Receiver 两方 Stackelberg 博弈 + Bayesian Persuasion 思想，把"隐私"重新表述为 Receiver 在最坏 data-prior 下的相对评分规则损失，给出统一定义 $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP，同时把 pure DP 和 probabilistic DP 收编为特例，并首次为**确定性算法**（如无噪经验均值）给出非平凡的形式化隐私保证。
tags:
  - "ICML 2026"
  - "AI安全"
  - "差分隐私"
  - "贝叶斯说服"
  - "评分规则"
  - "Stackelberg 博弈"
  - "确定性机制"
---

# Persuasive Privacy

**会议**: ICML 2026  
**arXiv**: [2601.22945](https://arxiv.org/abs/2601.22945)  
**代码**: 无  
**领域**: AI 安全 / 隐私理论  
**关键词**: 差分隐私, 贝叶斯说服, 评分规则, Stackelberg 博弈, 确定性机制

## 一句话总结
本文用 Sender–Receiver 两方 Stackelberg 博弈 + Bayesian Persuasion 思想，把"隐私"重新表述为 Receiver 在最坏 data-prior 下的相对评分规则损失，给出统一定义 $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP，同时把 pure DP 和 probabilistic DP 收编为特例，并首次为**确定性算法**（如无噪经验均值）给出非平凡的形式化隐私保证。

## 研究背景与动机

**领域现状**：过去二十年里 differential privacy (DP) 及其变体（Rényi DP、$f$-divergence privacy、pufferfish privacy、QIF…）已成为数据隐私事实标准。它们都基于"相邻数据集上输出分布几乎不可区分"的代数判据，提供干净的 worst-case 保证和良好的 composition / post-processing 性质。

**现有痛点**：DP 在工程落地中存在三类长期被诟病的问题——参数 $\varepsilon$ 难以向监管和公众解释；语义解释（如 Kasiviswanathan–Smith、Wasserman–Zhou）都是事后补丁，难以直接对应到"我害怕泄露什么"的现实诉求；以及实际部署中常常出现 $\varepsilon$ 过大乃至无意义的隐私预算。更关键地，DP 及其几乎所有变体**无法**对确定性机制给出非平凡保证（任何对相邻数据集输出不同的确定性函数都被判为非隐私），但美国十年人口普查等场景大量依赖确定性的"invariant statistics"。

**核心矛盾**：DP 把"隐私"建模成相邻输出分布的不可区分性，这种纯算法层定义无法吸纳 Sender 对"具体怕泄露什么"的偏好；同时 worst-case "全宇宙先验" 的隐式假设让确定性映射自动出局。要同时拥有 (i) 可面向特定语义裁剪、(ii) 能解释、(iii) 能覆盖确定性算法的框架，必须把博弈中的 Sender、Receiver、效用函数都显式建模。

**本文目标**：构造一个 game-theoretic 元框架，使其能 (a) 按需生成新的隐私定义并保留严格证明、(b) 反过来评估已有 DP 系列保证、(c) 对确定性机制也能给出非平凡保证。

**切入角度**：把数据发布看作 Bayesian Persuasion 的变体——Sender 持有真值 $x$ 并提交 Markov kernel $M$，Receiver 持先验 $Q$、观察 $T\sim M(x,\cdot)$ 后做贝叶斯决策；Sender 的"隐私函数" $\rho(d,x)$ 取负就是 Receiver 的损失（信息不对称 + 共享效用 + Sender robust）。结合 Grünwald–Dawid 的经典结果，Receiver 最优决策诱导的损失自动构成 **proper scoring rule**，这把"隐私评估"和"概率预测"放进了同一数学语言。

**核心 idea**：定义"相对隐私评分" $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ 作为发布前后 Receiver 对真值预测能力的提升，取关于 $x$、Receiver 先验 $Q\in\mathcal{Q}_x$、随机性 $T$ 三重 worst case 的 $\kappa$-尾概率条件作为统一的 PP 定义。

## 方法详解

### 整体框架
本文要解决的是"隐私缺一个能解释、能裁剪、还能覆盖确定性算法的统一定义"，做法是把数据发布重写成一个 Sender–Receiver 的 Stackelberg 博弈。Sender 持真值 $x$ 并预先公开承诺一个机制 $M$（Markov kernel）及其所属隐私类，Receiver 持先验 $Q$、观察输出 $T\sim M(x,\cdot)$ 后做贝叶斯决策；借 Grünwald–Dawid 的经典等价，Receiver 最优决策诱导的损失自动是一个 proper scoring rule，于是"隐私"就被表述成"发布前后 Receiver 对真值预测能力的相对提升"在最坏情形下的上界。整个框架因此先分三步把定义立起来——定下博弈语义与透明性假设、把 Sender 偏好转成评分规则、用相对评分写出统一的 PP 不等式；定义立住后再反过来验证自洽性：把 pure / probabilistic DP 收编为特例，并厘清"接收方后处理（receiver post-processing）"与"发送方后处理（sender post-processing）"两种含义的区别。

### 关键设计

**1. 博弈语义与评分规则化：把"Sender 偏好"翻译成 proper scoring rule**

要让隐私可以面向"具体怕泄露什么"裁剪，第一步是把 Sender 的偏好显式建模进博弈。Sender 持真值 $x\in\mathsf{X}$ 并预先公开承诺机制 $M:(\mathsf{X},\mathcal{T})\to[0,1]$ 和它所属的隐私类 $\mathfrak{C}$（Assumption 1 透明性：$M$ 与 $\mathfrak{C}$ 不依赖数据）；Receiver 持先验 $Q\in\mathcal{P}$、观察 $T\sim M(x,\cdot)$ 后做 Bayes-rational 决策（Assumption 2）。Sender 用一个"隐私函数" $\rho:(\mathcal{D},\mathsf{X})\to\mathbb{R}$ 表示自己对 Receiver 不同决策的偏好。关键一步是 Proposition 1：若令 Receiver 的损失函数 $\ell=\rho$（Assumption 3），Sender 拿到的就是最 robust 的 worst-case data-averaged 损失对手模型。此时 Receiver 最优决策为 $d^P\in\arg\inf_d \mathbb{E}_{X\sim P}[\rho(d,X)]$，由它诱导的 $S(P,x)=\rho(d^P,x)$ 恰好是一个 negatively-oriented proper scoring rule（Proposition 2，本质是 Grünwald–Dawid 2004 的结论），称为 privacy score。这一步之所以有效，是因为它把"隐私评估"和"概率预测"放进了同一套数学语言——任何 proper SR 都对应某个贝叶斯决策问题，反之亦然。

**2. 隐私 = 相对评分 + 最坏先验类：让对手强度变成可调旋钮（Definition 3）**

针对 DP 隐式假设"对手知一切"导致确定性算法被一棒打死的痛点，本文把隐私定义成相对评分而非绝对评分。用 $\Delta_S(Q,T,x)=S(Q,x)-S(Q_T,x)$ 度量"发布 $T$ 后 Receiver 对 $x$ 预测能力的提升"，机制 $M$ 称为 $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-Persuasive Private 当且仅当

$$\inf_{S\in\mathcal{S}}\inf_{x\in\mathsf{X}}\inf_{Q\in\mathcal{Q}_x}\mathbb{P}_x[\Delta_S(Q,T,x)\le\kappa]\ge 1-\delta$$

四元组分别对应要保护的语义集合 $\mathcal{S}$、考虑的对手先验类 $\mathcal{Q}_x$、最大允许隐私损失 $\kappa$、容错概率 $\delta$。用相对值 $\Delta_S$ 而非绝对 $S(Q_T,x)$ 是关键：若用绝对值，当 Receiver 先验已是 $Q=\delta_x$（完全知情）时所有机制都"等隐私"，定义会退化；改用差分后机制只需保证"不让 Receiver 信念向真值显著靠拢"。worst case 跨三重——$x$ 跨全数据空间（Assumption 5，保证 $\mathfrak{C}$ 不依赖数据）、$Q$ 跨 Sender 限定的对手强度类 $\mathcal{Q}_x$、$T$ 跨机制随机性（Assumption 4 的 $\kappa$-$\delta$ 尾概率条件）。这样既保留了 DP 风格的代数 robustness，又通过显式的 $\mathcal{Q}_x$ 把"Sender 愿意假设对手有多强"做成可调旋钮。

**3. DP 是 PP 的特例：用博弈语义解释 DP 究竟在防什么（Proposition 6）**

框架的自洽性体现在它能反过来当作"DP 的语义解释器"。取 $L$ 为离散负对数评分、取对手类 $\mathcal{H}=\{Q\in\mathcal{P}_2:\exists(x,x')\in\mathfrak{N},Q(\{x,x'\})=1\}$（两点 alternative-hypothesis 先验，对应 DP 的相邻对），则 $M$ 是 $(\varepsilon,\delta)$-PDP 当且仅当 $M$ 是 $(L,\mathcal{H},\varepsilon,\delta)$-PP，$\delta=0$ 即 pure $\varepsilon$-DP。证明揭示极小值在 Receiver 对真值的先验概率 $Q(\{x\})\to 0$ 处达到，由此给出 DP 的新解读——"$(\varepsilon,\delta)$-DP 防护的是 Receiver 几乎不相信真值时仍能从输出获取信息"，这也解释了 DP 实操中 $\varepsilon$ 难调难解释的根源。把 Assumption 4 的尾概率换成期望，还能类似地恢复 Rényi DP 和更广的 $f$-divergence privacy（Appendix C）。相比以往把 DP 语义当作事后构造（如 Kasiviswanathan–Smith 的假设性检验视角），PP 把 DP 直接放进 first-principles 推导，让每条假设都可对照现实场景测试和松弛。

**4. Receiver vs Sender post-processing 的拆分：澄清 PDP "缺 post-processing" 不是缺陷（Def 5–6, Prop 4–5）**

PDP 长期被批评"不满足 post-processing inequality"，本文指出这条批评混淆了两件事。传统的 $M\in\mathfrak{C}\Rightarrow MK\in\mathfrak{C}$ 实际上把 Receiver 拿到输出后任意后处理（receiver post-processing：$M\otimes K$）和 Sender 在发布前对输出再加变换（sender post-processing：$MK$，只发布边缘）混为一谈。所有 PP 保证都满足前者（Proposition 4，这才是真正的对手鲁棒性），但确实存在 PP 保证不满足后者（Proposition 5）；补救也很简单：Sender 同时发布 $M$ 的原始输出和 $K$ 的变换结果（发 $M\otimes K$ 而非 $MK$ 的边缘）即退化为 receiver 情形。这条拆分把 PDP"差一点"的定性重新校准——它丢的只是"用复杂机制证简单机制隐私"的代数工具，而非对手鲁棒性，同时也为 conjugate prior family 下的 composition rule（$\kappa_1+\kappa_2$、$\delta_1+\delta_2$，Proposition 3）打好基础。

### 损失函数 / 训练策略
框架本身是**定义性的而非训练性的**：没有要学的参数；"训练"对应 Sender 选机制 $M$ 来满足 PP 不等式。理论分析借助 proper scoring rule（Dawid–Sebastiani score、负对数评分、interval score 等）做替换以恢复不同 DP 变体；composition rule 需要 $\mathcal{Q}_x$ 对所考虑机制 conjugate（Definition 4），高斯先验族下自动成立。

## 实验关键数据

本文是纯理论工作，不涉及大规模数值实验，但给出两个 illustrative case 验证框架"能覆盖 DP 覆盖不到的场景"。

### 主结果对照表（PP 框架与既有隐私定义的关系）

| 既有定义 | 对应 PP 实例化 | 评分规则 $S$ | 对手先验类 $\mathcal{Q}_x$ | 备注 |
|---------|---------------|--------------|----------------------------|------|
| Pure $\varepsilon$-DP | $(L,\mathcal{H},\varepsilon,0)$-PP | 负对数评分 $L$ | 相邻两点先验类 $\mathcal{H}$ | Prop 6 的 $\delta=0$ 退化 |
| $(\varepsilon,\delta)$-PDP | $(L,\mathcal{H},\varepsilon,\delta)$-PP | 负对数评分 $L$ | $\mathcal{H}$ | Proposition 6 |
| Rényi DP | 期望版 PP（Appendix C） | 负对数评分 | $\mathcal{H}$ | Assumption 4 改期望 |
| $f$-divergence privacy | 期望版 PP + $f$ 评分 | $f$-散度对应评分 | $\mathcal{H}$ | Appendix C |
| **无噪经验均值** | $(\mathcal{I},\mathcal{G}_x^r,\kappa,\delta)$-PP | 边缘 Dawid–Sebastiani 评分 | 高斯先验 + 先验质量/条件数约束 | DP 给不出，PP 可证 |

### 关键定理与确定性机制示例

| 性质 / 结果 | 内容 | 意义 |
|------------|------|------|
| Composition (Prop 3) | 若 $\mathcal{Q}_x$ 对 $M_1$ conjugate，则 $M_1\otimes M_2$ 是 $(\mathcal{S},\mathcal{Q}_x,\kappa_1+\kappa_2,\delta_1+\delta_2)$-PP | 保留 DP 风格线性累加 |
| Receiver post-processing (Prop 4) | 所有 PP 保证下，对手任意后处理不增隐私损失 | 真正的对手鲁棒性 |
| Sender post-processing (Prop 5) | 存在 PP 不满足；但发布 $M\otimes K$ 即可恢复 | 解释 PDP "缺 post-processing" 不是缺陷 |
| DP 等价性 (Prop 6) | $(\varepsilon,\delta)$-PDP $\Leftrightarrow$ $(L,\mathcal{H},\varepsilon,\delta)$-PP | DP 的博弈式语义首证 |
| 无噪均值私隐 (§5.1) | $n\ge 2$ 时确定性发布 $\bar{x}=\frac{1}{n}\sum x_i$ 在高斯先验类 $\mathcal{G}_x^r$ 下满足边缘 DSS 的 PP | DP 全无能为力的场景 |

### 关键发现
- **DP 的隐式假设揭穿**：Prop 6 的证明把 DP 的极值点定位到 $Q(\{x\})\to 0$，等于说 DP 在防护"Receiver 几乎不信真值"这个语义反直觉的极端对手，这解释了 DP 实操中 $\varepsilon$ 难调和难解释的根本来源。
- **确定性机制并非天然 unprivate**：只要把对手强度 $\mathcal{Q}_x$（如高斯先验 + 条件数 + 先验偏差上界）和评分规则（边缘 DSS）按场景定制，无噪经验均值在 $n$ 足够大时可证明 PP，这与统计披露控制（SDC）领域的长期直觉首次对上。
- **Bayesian Persuasion 启示**：framework 本质上是 sender-commit、receiver-best-respond 的 Stackelberg，但加入信息不对称（Sender 知 $x$）+ 共享 utility（$\ell=\rho$）+ Sender robust（worst-case）三处与经典 BP 的关键差异。

## 亮点与洞察
- **First-principles 替代事后语义**：以往 DP 语义解释（hypothesis testing、Bayesian risk bound 等）都是 DP 定义出炉后倒推；PP 反过来从博弈和评分规则推出 DP，这种 first-principles 路径让"为什么是 $\exp(\varepsilon)$ 倍"这种问题有了原生答案。
- **可裁剪的对手类 $\mathcal{Q}_x$**：把"我假设对手有多强"从隐式（DP 默认对手知一切相邻信息）变为显式参数，这条工程上极有价值——监管和审计可以围绕 $\mathcal{Q}_x$ 而非难解释的 $\varepsilon$ 对话。
- **proper scoring rule 作为隐私语言**：Grünwald–Dawid 的等价（任何 proper SR 都对应某个 Bayes 决策问题）保证了 $S$ 与 $(\rho,\mathcal{D})$ 双向可换，意味着任何统计预测评估工具（如 CRPS、log-score、Brier score）都可被直接拿来定义新的隐私语义。
- **post-processing 拆分的工程启示**：分清 sender vs receiver post-processing 后，PDP 之类"差一点"的定义不再被一票否决，反而提供了一条 trade-off 曲线——放弃 sender post-processing 换取更紧的隐私参数。

## 局限与展望
- **理论框架，工程化空白**：论文未给出"如何在工业管道里选 $\mathcal{Q}_x$" 的最佳实践，对手先验类的可校准性、可审计性还需后续工作填补。
- **conjugacy 条件较强**：composition rule 依赖 $\mathcal{Q}_x$ 对所考虑机制 conjugate；离开高斯-高斯这类共轭族（如混合先验、heavy-tail 先验、神经网络后验近似）后 composition 还能怎么写仍未解决。
- **确定性机制只给两例**：§5 只覆盖经验均值和（Appendix 中的）线性约束；对真正复杂的确定性发布（如人口普查的全套 invariant statistics）的 PP 保证仍需逐场景构造。
- **未与 DP 实证库对接**：没有把现实 $(\varepsilon,\delta)$ 部署（OpenDP、Google DP、苹果差分隐私 telemetry）翻译成 PP 形式，回答"它们对应哪个 $\mathcal{Q}_x$"。后续若能给出转换表，将极大推广本框架。

## 相关工作与启发
- **vs Differential Privacy (Dwork et al. 2006)**：DP 是 $(L,\mathcal{H},\varepsilon,\delta)$-PP 的特例，PP 提供 first-principles 语义解释和对确定性机制的扩展；PP 牺牲了 DP 的 sender post-processing 闭包，但通过发布 $M\otimes K$ 可补救。
- **vs Pufferfish (Kifer & Machanavajjhala 2014)**：两者都允许用户自定义"要保护什么 secret"和"对手先验类"；PP 额外把 Sender 当作博弈第一玩家，且强制要求保护语义来自 proper SR，使理论结构更对称。
- **vs Bayesian Persuasion (Kamenica & Gentzkow 2011)**：经典 BP 假设 Sender 不知真值且与 Receiver 信息对称；PP 将之改造为 Sender 知真值、Sender 与 Receiver 效用相关、Sender robust 三点关键修改，把 BP 工具搬入隐私语境。
- **vs Quantitative Information Flow / $f$-divergence privacy**：PP 通过期望版变体即可吸纳这两族（Appendix C–D），统一了"信息流"视角和"评分规则"视角。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Sender-Receiver Stackelberg + proper SR 把 DP 从代数定义改写为博弈论一阶推导，为确定性机制开辟首例非平凡隐私论证。
- 实验充分度: ⭐⭐⭐⭐ 纯理论文章，证明覆盖 DP/PDP/Rényi DP/$f$-divergence privacy + 经验均值案例，理论分量充足；但缺真实部署 case study 翻译。
- 写作质量: ⭐⭐⭐⭐⭐ 概念栈（隐私函数→评分规则→PP 定义→DP 特例→确定性机制）层层递进，假设逐条编号并讨论必要性，可读性远超同类 DP 理论论文。
- 价值: ⭐⭐⭐⭐⭐ 给隐私定义提供可重构、可裁剪、可审计的"元框架"，对 SDC、官方统计、人口普查等场景具有立刻可用的理论工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)
- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICML 2026\] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)

</div>

<!-- RELATED:END -->
