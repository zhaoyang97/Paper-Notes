---
title: >-
  [论文解读] Performative Learning Theory
description: >-
  [ICML 2026][学习理论][Wasserstein 距离] 本文把"预测会改变它想预测的结果"这一 performative prediction 现象首次嵌入统计学习理论，在样本、总体、以及二者都被预测扰动的三种情形下证明了泛化误差/泛化间隙/超额风险的上界，揭示出"改变世界"与"从世界学习"之间的根本权衡，以及最坏情况下"总体自我否定、样本自我实现"形成的经验回声室。
tags:
  - "ICML 2026"
  - "学习理论"
  - "泛化界"
  - "Performative Prediction"
  - "Wasserstein 距离"
  - "自我实现/自我否定预测"
  - "分布鲁棒优化"
---

# Performative Learning Theory

**会议**: ICML 2026  
**arXiv**: [2602.04402](https://arxiv.org/abs/2602.04402)  
**代码**: https://github.com/rodemann/plt-jobseekers (案例研究复现)  
**领域**: 学习理论 / 泛化界 / Performative Prediction  
**关键词**: performative prediction, 泛化界, Wasserstein 距离, 自我实现/自我否定预测, 分布鲁棒优化

## 一句话总结
本文把"预测会改变它想预测的结果"这一 performative prediction 现象首次嵌入统计学习理论，在样本、总体、以及二者都被预测扰动的三种情形下证明了泛化误差/泛化间隙/超额风险的上界，揭示出"改变世界"与"从世界学习"之间的根本权衡，以及最坏情况下"总体自我否定、样本自我实现"形成的经验回声室。

## 研究背景与动机

**领域现状**：机器学习系统早已从"分析世界"变成"塑造世界"——导航 App 预测拥堵会让司机改道、使拥堵消失；就业中心按"长期失业风险"分配培训名额，被预测高风险者反而更快找到工作。Perdomo 等（2020）把这种反馈环形式化为 **performative prediction（PP）**。但已有 PP 工作（Perdomo, Brown, Miller 等）几乎都把"实际结果"定义为**整个总体**，研究的是重复风险最小化的稳定点与最优点，**回避了"能否从一个有限样本泛化到总体"这一问题**。

**现有痛点**：现实中我们通常只有总体的一个**有限样本**（如只在旧金山灰度上线、只在巴伐利亚试点），而 performative 效应既可能发生在样本内、也可能发生在样本外的总体上。经典学习理论假设训练/测试分布固定、互不影响，一旦预测会反作用于数据分布，"从训练集泛化到测试集"的标准结论就不再适用——而这恰恰是没人系统回答过的问题。

**核心矛盾**：performative 世界里，**学习目标本身会随预测漂移**。你越用模型去干预数据（帮更多人找到工作、让更多司机改道），样本就越偏离原总体，反而越难从中可靠地推断总体性质。这是一个"干预 vs 推断"的内在张力。

**本文目标**：把 PP 嵌入统计学习理论，定义清楚"在 performative 世界里泛化到底意味着什么"，并在三种扰动情形（仅样本、仅总体、二者皆有）下给出可计算的泛化保证。

**核心 idea**：在不假设转移映射 $\mathrm{Tr}$ 任何具体函数形式（只要它 Wasserstein 敏感）的前提下，用覆盖数 + Wasserstein 距离把 performative 漂移"算进"泛化界；并把最坏情况刻画为 Wasserstein 空间里的 **min-max（总体自我否定）** 与 **min-min（样本自我实现）** 风险泛函。

## 方法详解

### 整体框架

论文先在概念层把"performative 泛化"拆成四种场景并提出四个研究问题（RQ1–RQ4，见 Table 1），分别对应"在样本上重训 / 在总体上重训 / 二者都重训"等组合；再在技术层用一个**通用假设**（转移映射 $\mathrm{Tr}$ 的 Wasserstein 敏感性）把 performative 漂移嵌入学习理论，逐个证明超额风险、performative 超额风险、泛化间隙、累积 performative 超额风险的上界；最后揭示两个结构性洞察——"改变 vs 学习"的权衡，以及"经验回声室"，并给出一个反直觉但实用的推论：在被扰动的样本上重训反而能收紧界。证明的核心技术是用经验过程理论，对 Wasserstein 空间里的 inf-sup / inf-inf 风险泛函做对偶刻画（分别对应分布鲁棒优化 DRO 与分布有利优化 DFO）。这是一篇纯理论文，下面按"概念框架 → 嵌入与界 → 结构洞察 → 反直觉推论"四个贡献点展开。

### 关键设计

**1. 把 performative prediction 嵌入学习理论：四类泛化场景与重复经验风险最小化**

针对"已有 PP 只谈总体、不谈从样本泛化"的空白，作者区分了**样本 performativity**（模型只改变它训练用的样本/子总体，如导航只对灰度用户可见）、**总体 performativity**（经典 PP 设定）、以及**全 performativity**（样本和总体都反应）。相应地把经典 PP 的重复风险最小化（RRM, $\theta_{t+1}=G(d_t)$）扩展到**重复经验风险最小化**（RERM, $\widehat\theta_{t+1}=G(\widehat d_t)$，其中 $\widehat d_t=\mathrm{Tr}(\widehat d_{t-1},\widehat\theta_t)$）。Table 1 用"重训对象 × performative 效应位置"的二维表把 ERM、在线学习、经典 PP 以及四个开放 RQ 统一编排，明确了本文要新答的格子。采用 Brown 等（2022）的**有状态（stateful）**扩展 $d_t=\mathrm{Tr}(d_{t-1},\theta_t)$，使无状态情形 $d_t=\mathrm{Tr}_s(\theta_t)$ 成为特例。

**2. 用 Wasserstein 敏感性刻画未知漂移并给出泛化界**

为了在**不假设 $\mathrm{Tr}$ 具体形式**的情况下还能给界，作者只要求 Perdomo/Brown 框架里的一小撮条件：损失 $\gamma$-强凸（Cond. 3.1）、转移映射**联合 Wasserstein 敏感**（Cond. 3.2，$W_p(\mathrm{Tr}(d,\theta),\mathrm{Tr}(d',\theta'))\le\varepsilon W_p(d,d')+\varepsilon\|\theta-\theta'\|_2$）、损失对 $z$ Lipschitz 且对 $\theta$ 连续可微（Cond. 3.3）。证明策略是把三段 Wasserstein 距离 $W_p(\widehat d_0,d_0)$（Lemma 3.4，样本-总体收敛）、$W_p(\widehat d_0,\widehat d_T)$（Lemma 3.5，样本内 performative 漂移）、$W_p(d_0,d_T)$（Lemma 3.9，总体 performative 漂移）分别界住，再经 Kantorovich–Rubinstein 引理转成期望差。由于漂移可能把评估点推到 $d_0$ 支撑外，作者用**覆盖数熵积分** $\mathfrak C$（而非 Rademacher 复杂度）度量假设类丰富度。最终得到样本 performativity 下的超额风险界（Theorem 3.7）和全 performativity 下的 performative 超额风险界（Theorem 3.10）。一个关键可观测量是**performative 响应率** $m/n$（样本中因预测而改变的单位数 $m$ 占样本量 $n$ 的比例），界随 $m$ 增大而增大。

**3. self-negating 与 self-fulfilling：Wasserstein 空间里的 min-max / min-min 与经验回声室**

为了在更强一点的正则性下给出**更紧**的泛化间隙界（Theorems 3.13/3.15），作者揭示了 performative 世界泛化难的两个方向：总体可能**自我否定**（self-negating）你的预测——最坏情况是 $\sup_{d}\mathscr R(d,\widehat\theta_T)$，即在 Wasserstein 球 $\mathcal A=\{d:W_p(d_0,d)\le b\}$ 上取最坏分布，正对应**分布鲁棒优化（DRO）**的 inf-sup 泛函；而样本可能**自我实现**（self-fulfilling）——RERM 等价于在样本侧解 $\arg\inf_\theta\inf_{d\in\mathcal A'}\mathscr R(d,\theta)$，对应**分布有利优化（DFO）**的 inf-inf 泛函。两者叠加：样本欺骗性地确认你的预测、总体却反着来，模型被拉向"在样本上好、在总体上差"，形成**经验回声室**（empirical echo chamber）。在导航例子里就是：旧金山司机（样本）完全信任 App 并照预测改道，湾区司机（总体）却反其道而行。

**4. change-vs-learn 权衡与反直觉推论：在扰动样本上重训反而能收紧界**

界的形式直接读出两条洞察。其一是**改变 vs 学习的权衡**：performative 项由 $\varepsilon(1+L_a)$ 主导——当 $\varepsilon(1+L_a)>1$ 时随样本重训次数 $T$ 指数增长，$=1$ 时线性增长；同时界整体随 $m$ 增大。直觉上，就业中心若想帮更多人（增大 $m$），代价就是模型更难泛化到未见过的新客户。其二是一个**反直觉推论**（Corollary 3.11）：天真地反复重训会让 $\widehat\theta_t$ 越来越差（界随 $T$ 涨），**但**这些被扰动的样本 $\widehat d_1,\dots,\widehat d_T$ 能帮你更高效地**估计 $\mathrm{Tr}$**，从而收紧 Theorem 3.10 的界（Lemma 3.9 保守地只用反应最剧烈那一轮的 $m$，而我们其实观测到每轮 $m_t$）。实用结论：若必须在 performative 下重训，就**用初始拟合 $\widehat\theta_0$ 做样本外预测、再用观测到的各轮样本漂移估计它将引发的总体漂移**，即可得到最紧的保证。

## 实验关键数据

本文是理论工作，"实验"是用真实数据**示意**界的形态，而非应用。下表先汇总四个研究问题与对应主结果：

| 研究问题 | 场景 | 被界对象 | 主要定理 |
|------|------|------|------|
| RQ1 | 仅样本 performativity（只在样本上重训） | 经典超额风险 | Theorem 3.7, Cor. 3.8 |
| RQ2 | 全 performativity（样本+总体都反应） | performative 超额风险 / 泛化间隙 | Theorems 3.10, 3.13, 3.15 |
| RQ3 | 先在样本重训、再在总体重训 | 累积 performative 超额风险 | Theorem 3.16 |
| RQ4 | RERM vs 假想 RRM 的推断差 | inferential gap（统计性质） | 部分由 Li 等(2025) CLT 回答 |

### 案例研究：德国就业局求职者数据

数据来自德国联邦就业局 1975–2017 年劳动力市场行政记录（原始 >6000 万行，2% 抽样），任务是二元预测"求职者是否会长期失业"，用 L2 正则的逻辑回归（满足强凸 + Lipschitz + 可微三条件）模拟"按风险分配培训名额"的 performative 效应。

| 设定 | 样本量 n | 观测漂移 m / 响应率 | 界（95% 置信） |
|------|------|------|------|
| 样本 performativity（RQ1, Cor. 3.8） | 60147 | m=1816，$m/n\approx0.030$ | 泛化间隙 $\le 0.01+0.29\approx0.30$ nats |
| 全 performativity（RQ2, Thm. 3.13，半模拟） | 41585 | $m=\xi n,\ \xi\in\{0.01,\dots,0.5\}$ | 随分配比例 $\xi$ 单调增长（Figure 1.3） |

### 关键发现
- **响应率直接控制界**：样本 performativity 下界 $\approx0.30$ nats，其中采样项因 $n$ 大而很小（$\approx0.01$），主导项 $\approx0.29$ 完全由可观测的响应率 $m/n=1816/60147$ 决定——把抽象界变成了就业中心能算出来的"训练误差最多被超出 0.3 个 nats"的可操作保证。
- **干预越多越难学**：半模拟里把高风险者按比例 $\xi$ 分配培训，$\xi$ 越大（帮越多人、改越多单位），performative 泛化间隙界越大，直观印证 change-vs-learn 权衡（Figure 1.3 蓝线随 $\xi$ 上升）。
- **界的可分解性**：总界拆成自适应复杂度项、performative 项、采样项三部分，便于实践者看清"误差来自抽样有限还是来自干预过猛"。

## 亮点与洞察
- **把现象问题转成可计算的界**：performative prediction 此前多谈稳定/最优点，本文第一次把"从样本泛化到总体"做成有限样本泛化界，且关键量 $m/n$ 是institution 真能观测到的，可操作性强。
- **min-max ↔ DRO、min-min ↔ DFO 的对偶**：把"总体自我否定/样本自我实现"分别对应到分布鲁棒优化和分布有利优化两类成熟工具，是个漂亮的桥接，未来可借这两侧的进展继续收紧界。
- **反直觉的"重训有用"**：通常认为 performative 下重训只会越训越糟，本文却指出被扰动的样本能用来估计未知漂移 $\mathrm{Tr}$，从而收紧界——这个"坏样本仍有信息价值"的视角可迁移到任何需要估计未知分布漂移的在线/持续学习场景。

## 局限与展望
- **强凸假设较苛刻**：Cond. 3.1（损失强凸）与更紧界所需的 $\mathcal F$ 含 Lipschitz 函数（Cond. 3.12）都偏强；不过作者强调这些是关于"我们能控制的损失与模型类"的可验证假设，换来对"我们无法控制的未知 $\mathrm{Tr}$"零形式假设。
- **界偏松**：为换取对 $\mathrm{Tr}$ 形式的不可知性，界比假设具体漂移形式时更松（虽仍有洞察）。
- **RQ4 仅渐近回答**：RERM 与假想 RRM 的推断差目前只有 $n\to\infty$ 的 CLT 结果，有限样本与有状态情形待解。
- **案例仅为示意**：真实应用需要历史预测记录（通常不可得），论文明确说 Section 4 只是 illustration 而非落地应用。

## 相关工作与启发
- **vs Perdomo 等（2020）/ Brown 等（2022）经典 PP**：它们研究总体层面的稳定点/最优点，假设可访问整个总体；本文新增"从有限样本泛化到总体"的有限样本视角，并用 RERM 把 ERM 纳入 performative 框架。
- **vs Kirev 等（2025）**：后者对二分类、线性 performative 漂移给出 RQ1(a) 的部分答案；本文更一般——覆盖所有 Lipschitz 连续转移映射，且支持对 $Y$、$X$ 任意子集的漂移。
- **vs 分布鲁棒 performative 优化/预测（Jia 等 2025、Xue & Sun 2024）**：它们用 Wasserstein 模糊集 a priori 地鲁棒化优化/预测；本文研究的是**泛化**，且模糊集是**从样本的 performative 反应估计**出来的，而非先验指定。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 performative prediction 嵌入统计学习理论并给出从样本到总体的泛化界。
- 实验充分度: ⭐⭐⭐⭐ 理论为主，用真实就业数据示意界形态到位，但承认只是 illustration。
- 写作质量: ⭐⭐⭐⭐ 概念框架（Table 1）与两个跑例把抽象界讲得相当清楚。
- 价值: ⭐⭐⭐⭐⭐ 为"预测会改变数据"的系统给出第一套泛化分析工具，对高风险部署有现实意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] AI4SLT: Empirical Processes in Lean 4 for Formal Statistical Learning Theory](ai4slt_empirical_processes_in_lean_4_for_formal_statistical_learning_theory.md)
- [\[ICLR 2026\] A Statistical Theory of Overfitting for Imbalanced Classification](../../ICLR2026/learning_theory/a_statistical_theory_of_overfitting_for_imbalanced_classification.md)
- [\[ICML 2026\] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[ICML 2026\] Bandit Social Learning with Exploration Episodes](bandit_social_learning_with_exploration_episodes.md)

</div>

<!-- RELATED:END -->
