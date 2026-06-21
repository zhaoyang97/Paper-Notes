---
title: >-
  [论文解读] Stable Coresets: Unleashing the Power of Uniform Sampling
description: >-
  [ICLR 2026][学习理论][coreset] 本文提出介于 weak coreset 与 strong coreset 之间的"stable coreset"新概念，证明了**仅靠均匀采样**（一个大小为 $O(\epsilon^{-2}\log d)$ 的均匀样本）就能为 $\ell_1$ 度量下的 1-median 问题构造出 stable coreset，从而把"廉价、与数据无关、可流式/分布式"的均匀采样从启发式提升为有严格保证、且能传递到所有可嵌入 $\ell_1$ 的子度量（Kendall-tau、Jaccard 等）的工具。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "Coreset 理论"
  - "coreset"
  - "均匀采样"
  - "1-median"
  - "$\\ell_1$ 度量"
  - "VC 维"
---

# Stable Coresets: Unleashing the Power of Uniform Sampling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=sOpAa8iR0A](https://openreview.net/forum?id=sOpAa8iR0A)  
**代码**: https://github.com/amircarmel-lab/StableCoresets  
**领域**: 学习理论 / Coreset 理论  
**关键词**: coreset、均匀采样、1-median、$\ell_1$ 度量、VC 维

## 一句话总结
本文提出介于 weak coreset 与 strong coreset 之间的"stable coreset"新概念，证明了**仅靠均匀采样**（一个大小为 $O(\epsilon^{-2}\log d)$ 的均匀样本）就能为 $\ell_1$ 度量下的 1-median 问题构造出 stable coreset，从而把"廉价、与数据无关、可流式/分布式"的均匀采样从启发式提升为有严格保证、且能传递到所有可嵌入 $\ell_1$ 的子度量（Kendall-tau、Jaccard 等）的工具。

## 研究背景与动机

**领域现状**：大数据聚类普遍采用 "sketch-and-solve" 范式——先把数据压成一个与原数据规模无关的小摘要（coreset），再在小摘要上跑昂贵的优化/学习算法。文献里 coreset 主要有两类。**Weak coreset** 只保证"在 coreset 上求得的（近）最优解，对原数据也近似最优"，不保证目标值本身；它和 sketch-and-solve 范式对齐，关键是**有时能用均匀采样构造**，因而极快、易于流式与分布式实现。**Strong coreset** 更强：它要求对度量空间中**每一个**可能的中心 $c$，coreset 上的代价都近似等于原数据上的代价（即 $\mathrm{cost}(c,Q)\in(1\pm\epsilon)\,\mathrm{cost}(c,P)$），因此适用面更广，还有一条极有用的性质——**子度量传递**：若度量空间 $X$ 有 strong coreset，则任何等距嵌入进 $X$ 的子度量（如嵌入 $\ell_1$ 的 Kendall-tau）也自动有 coreset。

**现有痛点**：strong coreset 的强保证有沉重代价——它的构造算法**必然要读完整个数据集**。直觉上的反例是：数据点都密集成一团，只有一个极远的离群点；这个离群点几乎不影响最优中心，但对**目标值**的贡献可能很大，所以 strong coreset 必须把它找出来并包含进去——而要找到这种离群点就不可能做到亚线性时间，更不可能用均匀采样（均匀采样几乎抽不到稀有的小簇/离群点）。反过来，均匀采样虽然廉价、与数据无关，却只能给出 weak coreset，而 weak coreset **不具备子度量传递性**，每换一种度量都要重新分析。

**核心矛盾**：广适用性（strong 的子度量传递）与高效构造（uniform sampling 的亚线性、data-oblivious）二者不可兼得——strong 要读全量数据，weak 又不能传递到子度量。

**本文目标**：找到一个"甜点"概念，既保留 strong coreset 最有用的子度量传递能力，又能用均匀采样这种最廉价的方式构造出来；并先把它在最干净的 $k=1$（1-median）情形下讲透——因为 $k=1$ 时均匀采样无需任何额外数据假设就能成功，且它是一般 $k$-median 的重要积木。

**切入角度**：作者注意到，strong coreset 之所以贵，是因为它要**保住每个中心的绝对代价**；但很多下游应用（求最优解、做近似算法、加公平性约束）其实**只需要保住不同中心之间代价的相对大小顺序**，并不需要绝对值。这一观察正是均匀采样能奏效的关键——离群点改变绝对代价，却不改变中心间的排序。

**核心 idea**：定义 **stable coreset**——只保证"中心之间代价的相对顺序"在 coreset 与原数据上近似一致（而非保住绝对代价），它严格夹在 weak 与 strong 之间，既继承了子度量传递性，又恰好弱到能让均匀采样成立。

## 方法详解

### 整体框架

本文是一篇**理论论文**，没有可训练的 pipeline，核心是"定义新概念 + 两段式证明"。整体逻辑是：先把三种 coreset 用统一语言定义清楚并理清它们的强弱层级（strong ⊂ stable ⊂ weak），再分两步证明主定理 Theorem 1.4（均匀采样 → $\ell_1$ 的 stable coreset）：

- **第一步（任意度量的通用框架，Section 3）**：抽象出一个充分条件 "relative cost-difference approximation"（RCDA），证明只要子集 $Q$ 满足 RCDA（外加一个均匀样本天然满足的简单条件），$Q$ 就是 stable coreset。这一步与具体度量无关。
- **第二步（$\ell_1$ 的具体实例化，Section 4）**：用 PAC 学习里的 $\epsilon$-approximation 技术，证明 $\ell_1$ 下"轴对齐阈值函数族"的 VC 维只有 $O(\log d)$，于是大小 $O(\epsilon^{-2}\log d)$ 的均匀样本就是 $\epsilon$-approximation，进而（关键技术引理）是 $O(\epsilon)$-RCDA，套回第一步框架即得主定理。

随后（Section 5）借助"嵌入即传递"把结论免费推广到 Kendall-tau、Jaccard、Hamming、$\ell_2$ 等度量，并给出 $k$-median 近似算法与"在 C-dispersed 输入上均匀采样反而得到 strong coreset"的加强版。

### 关键设计

**1. Stable coreset 的定义：只保相对顺序，不保绝对代价**

针对的痛点是 strong coreset 必须读全量数据、抓离群点，而 weak coreset 又不能传递到子度量。作者把"保住绝对代价"放宽为"保住中心间的相对代价顺序"。正式地，子集 $Q\subseteq P$ 是 $P$ 的 stable $(\epsilon,\eta)$-coreset，当且仅当

$$\forall c_1,c_2\in X,\quad \mathrm{cost}(c_1,Q)\le(1+\epsilon)\,\mathrm{cost}(c_2,Q)\;\Rightarrow\;\mathrm{cost}(c_1,P)\le(1+\eta)\,\mathrm{cost}(c_2,P).$$

对照三种定义可以看清它的"夹心"位置：weak coreset（Def 1.1）只比较一个中心 $c$ 与 $Q$ 上最优值 $\mathrm{opt}(Q)$；strong coreset（Def 1.2）要求 $\mathrm{cost}(c,Q)\in(1\pm\epsilon)\,\mathrm{cost}(c,P)$ 对**每个** $c$ 成立（保绝对值）；而 stable coreset 像 strong 一样对**所有中心对** $c_1,c_2$ 施加几何约束（因此能传递到子度量），却像 weak 一样只用"比较式"结构（因此只需保排序）。回到离群点例子：strong coreset 必须包含那个远点以保住它对代价的巨大贡献，**stable coreset 则不必**——因为远点同等地抬高所有中心的代价，不改变它们之间的顺序。正是这一"无须抓离群点"的松弛，让均匀采样得以奏效，揭示了 stable coreset 与均匀采样之间的天然契合。

作者同时建立严格层级（Prop 2.1）：每个 strong $\epsilon$-coreset（$\epsilon\in(0,\tfrac15)$）都是 stable $(\epsilon,4\epsilon)$-coreset，每个 stable $(\epsilon,\eta)$-coreset 都是 weak $(\epsilon,\eta)$-coreset，反向一般不成立；并证明 stable 的保证可经等距嵌入传递到任何子度量（Prop 2.3）。

**2. RCDA 充分条件：把"是不是 stable coreset"归约为一个可验证的逼近条件**

直接证明"某个均匀样本满足 stable coreset 定义里对所有中心对的约束"很难。作者引入一个对任意度量都适用的充分条件作为桥梁——**relative cost-difference approximation（$\epsilon$-RCDA）**。记归一化代价 $\overline{\mathrm{cost}}(x,P):=\tfrac1{|P|}\mathrm{cost}(x,P)$，$\mu$ 为 $P$ 的最优 median，则称 $Q$ 是 $P$ 的 $\epsilon$-RCDA，若

$$\forall x\in X,\quad \big|\,(\overline{\mathrm{cost}}(x,P)-\overline{\mathrm{cost}}(\mu,P))-(\overline{\mathrm{cost}}(x,Q)-\overline{\mathrm{cost}}(\mu,Q))\,\big|\le \epsilon\cdot\overline{\mathrm{cost}}(x,P).$$

直观上它要求：任意中心 $x$ 相对参考点 $\mu$ 的"代价差"，在 $P$ 上和在 $Q$ 上几乎相同——只要这个差保住了，中心间的排序自然就保住了。框架的主结论（Theorem 3.1）是：若 $Q$ 是 $P$ 的 $\epsilon$-RCDA，且满足 $\overline{\mathrm{cost}}(\mu,Q)\le c\cdot\overline{\mathrm{cost}}(\mu,P)$（$c\ge1$），则 $Q$ 是 $P$ 的 stable $(\epsilon/c,\,4\epsilon)$-coreset。这一步的价值在于把"对所有中心对的二元约束"换成了"对每个中心的一元逼近 + 一个均匀样本几乎必然满足的均值条件"，后者可用经典工具攻克；而且参考点 $\mu$ 不必真是 median，换成任意固定点也只损失常数因子，使框架更灵活。

**3. $\ell_1$ 实例化：用 $\epsilon$-approximation + $O(\log d)$ 的 VC 维把 RCDA 兑现**

要在 $\ell_1$ 下证明均匀样本满足 RCDA，作者借用 PAC 学习中的 **$\epsilon$-approximation**（与计算几何的 range-counting 紧密相关）。定义轴对齐阈值函数族 $T=\{\tau_{i,r}:\tau_{i,r}(x)=\mathbf 1\{x[i]\le r\}\}$，并称 $Q$ 是 $P$ 的 $\epsilon$-approximation，若对每个坐标 $i$、每个阈值 $r$，经验分布函数都被保住：$|\mathrm{edf}_Q(i,r)-\mathrm{edf}_P(i,r)|\le\epsilon$。关键的两块拼图是：

- **VC 维只有对数级（Prop 4.2）**：$\lfloor\log d\rfloor\le \mathrm{VCdim}(T)\le 2\log d$。正是这个对数界让样本量只含 $\log d$ 而非 $d$。
- **从 $\epsilon$-approximation 到 RCDA 的技术引理（Lemma 4.6）**：在 $\ell_1^d$ 中，若 $Q$ 是 $P$ 的 $\epsilon$-approximation，则 $Q$ 是 $P$ 的 $20\epsilon$-RCDA。直觉是 $\ell_1$ 代价可沿每个坐标拆成"按阈值积分经验分布函数"的形式，于是逐坐标保住 edf 就等价于保住整条相对代价结构。

把它们串起来：由 Li–Long–Srinivasan 的紧界（Theorem 4.4）与 $\mathrm{VCdim}(T)=O(\log d)$ 得 Corollary 4.5——大小 $O(\epsilon^{-2}\log\tfrac{d}{\delta})$ 的均匀样本以概率 $1-\delta$ 是 $\epsilon$-approximation；再经 Lemma 4.6 成为 $O(\epsilon)$-RCDA；最后对 $\overline{\mathrm{cost}}(\mu,Q)$ 用 Markov 不等式（其期望恰等于 $\overline{\mathrm{cost}}(\mu,P)$，故 $\Pr[\overline{\mathrm{cost}}(\mu,Q)\ge6\,\overline{\mathrm{cost}}(\mu,P)]\le\tfrac16$）加 union bound，套回 Theorem 3.1，即得主定理 1.4：**大小 $O(\epsilon^{-2}\log d)$ 的均匀样本以概率 $\ge\tfrac45$ 是 $\ell_1^d$ 下 1-median 的 stable $(\epsilon/6,4\epsilon)$-coreset。**

**4. 嵌入传递与加强：把 1-median/$\ell_1$ 的结论免费推广**

stable coreset 继承了 strong coreset 的子度量传递性，因此主定理一旦在 $\ell_1$ 成立，所有能（近）等距嵌入 $\ell_1$ 的度量就自动获得"基于均匀采样"的 coreset——Hamming、Kendall-tau、Spearman-footrule、树度量等等。对低失真嵌入（distortion $1+\zeta$），Prop 5.1 给出参数调整规则，于是 $\ell_2$（经 Dvoretzky 定理）得到大小 $O(\epsilon^{-2}\log(d/\epsilon))$ 的 stable coreset，Jaccard（经 min-wise 嵌入）也首次有了基于均匀采样的 coreset（Cor 5.3）。在此基础上还能推出两类加强结果：(i) 借 Ackermann et al. 的框架把 1-median 的结论提升为**一般 $k$-median 的近似算法**，覆盖此前未知的 Hamming/Kendall-tau/Jaccard；(ii) 当输入是 **C-dispersed**（直径 $\le C$ 倍平均距离，$\max_{x,y}\|x-y\|_1\le C\cdot\tfrac1{|P|^2}\sum_{x,y}\|x-y\|_1$）时，均匀采样得到的不只是 stable 而是 **strong coreset**（Theorem 5.5），这在 median 问题 NP-hard 的离散度量上尤其有用——可直接把廉价的常数近似/启发式算法跑在 strong coreset 上换取大幅加速。

## 实验关键数据

实验在真实数据集上验证 stable coreset 的实际效果，统一指标为相对误差 $\widehat E=\dfrac{\mathrm{cost}(\hat c_Q,P)-\mathrm{cost}(\hat c_P,P)}{\mathrm{cost}(\hat c_P,P)}$（百分比），即"在 coreset 上算得的中心 $\hat c_Q$ 放回原数据后，比原数据最优解差多少"。数据集见下：

| 数据集 | 规模 $n$ | 维度 $d$ | 度量 / 用途 |
|--------|---------|---------|------------|
| Yellow Taxi NYC (YT) | 2.8M | 11 | $\ell_1$ 下 1-median |
| Twitter | 1.3M | 3 | $\ell_1$ 下 1-median |
| Single-Cell Gene Expr. (SCGE) | 7,865 | 33,586 | $\ell_1$ / 维度依赖测试 |
| My Anime List (MAL) | 234K | 50 | Kendall-tau 排名聚合 |

### 主实验：均匀采样 vs 重要性采样

| 对比维度 | 均匀采样（本文） | 重要性采样（Jiang et al. 2024） |
|---------|----------------|-------------------------------|
| 误差 $\widehat E$ | 三个数据集上**与重要性采样相当** | 基线 |
| 构造时间（YT / Twitter / SCGE） | 抽 500 点 ≈ **0.0001 秒** | ≈ 82 / 114 / 512 秒 |
| 是否需读全量数据 | 否（data-oblivious，常数时间/样本） | 是（线性时间，逐点算 sensitivity） |

核心结论：均匀采样达到与昂贵的重要性采样**相当的逼近质量**，但构造时间快了**五到六个数量级**，且完全不需要预先检查数据。

### 应用实验：Kendall-tau 与公平性约束

| 实验 | 设置 | 关键发现 |
|------|------|---------|
| Exp 2：排名聚合启发式 | MAL 234K 用户 / 50 标题，5 种方法（MC1/2/3、Borda、SFO） | 较小的 coreset 即可逼近原数据结果；因启发式不直接优化 Kendall-tau，coreset 上偶尔出现**负相对误差**（比原数据还好） |
| Exp 3：公平性约束 | 8,700 排名 / 16 标题，跑 fairness-constrained ILP | 即使约束在采样时**未被考虑**，小 coreset 仍能很好支持约束优化，验证了 stable coreset 对受约束问题的适用性 |
| Exp 4：维度依赖 | SCGE，随机取 $d$ 维，coreset 大小 150/300/500 | 相对误差**随维度基本平稳**，轻微上升可归因于采样方差，支持"界可改进到与维度无关"的猜想 |

### 关键发现
- 均匀采样的最大优势不是精度更高，而是**精度相当、成本几乎为零且与数据无关**——这正是 stable coreset 概念松弛"只保排序"换来的红利。
- stable coreset 天然支持采样时未知的下游约束（公平性），因为它保的是中心间的相对顺序，约束只是限制了可行中心的集合而不破坏这一顺序。
- 维度实验经验性地支持"$\log d$ 依赖可去掉"的猜想，暗示理论界仍有收紧空间。

## 亮点与洞察
- **"只保相对顺序"是点睛之笔**：把 strong coreset"保绝对代价"放宽为 stable coreset"保排序"，恰好绕开了均匀采样最致命的弱点（抓不到离群点/小簇），又保留了最值钱的子度量传递性。一个看似微小的定义松弛，换来了"廉价 + 广适用"的双赢。
- **RCDA 把难证的二元约束降为可证的一元逼近**：用一个相对代价差的逼近条件作中介，把"所有中心对"的 stable 性归约到经典的 $\epsilon$-approximation / VC 维工具，证明结构干净、可复用到未来其他度量。
- **嵌入即免费推广**：一旦在 $\ell_1$ 站住，Kendall-tau、Jaccard、Hamming、$\ell_2$ 全部顺势拿到首个均匀采样 coreset——这种"证一个、得一片"的杠杆正是 strong/stable 概念相对 weak 的根本价值。
- **C-dispersed 下退化为 strong**：揭示了"输入足够均匀分散时，均匀采样其实给的是 strong coreset"，把概念谱系两端打通，并给 NP-hard 离散 median 提供了实用加速路径。

## 局限与展望
- **维度依赖未消除**：主定理样本量含 $\log d$，作者明确将"是否能做到与维度无关"列为公开猜想，仅有经验证据支持。
- **仅限 $k=1$**：核心结果只覆盖 1-median；推广到一般 $k$-median 需借助额外框架且依赖结构假设，作者坦言 $k>1$ 时均匀采样无法可靠捕捉小簇。
- **依赖嵌入入 $\ell_1$**：所有推广都建立在"目标度量能（近）等距嵌入 $\ell_1$"之上，对无法低失真嵌入 $\ell_1$ 的度量（如某些情形的 Jaccard 需高维、breakpoint 距离需限制条件）适用性受限。
- **常数因子较松**：参数从 $\epsilon$-approximation（$\to 20\epsilon$-RCDA $\to$ stable $(\epsilon/6,4\epsilon)$）一路放大，理论常数偏大，离实践最优可能有差距。

## 相关工作与启发
- **vs Strong coreset（Chen 2009；Braverman et al. 2022 等"环采样"）**：它们保住每个中心的绝对代价、可传递子度量，但构造必须读全量数据（环采样的整体分布非均匀，时间非亚线性）；本文以"只保排序"换来真正的均匀采样与亚线性构造，代价是不保绝对值。
- **vs Weak coreset（Huang et al. 2023a 在 $\ell_2$；Danos 2021 在 $\ell_1$）**：它们同样能用均匀采样廉价构造，但**不传递子度量**，每换一个度量都要重新分析，且 Danos 的 weak $(0,\epsilon)$-coreset 需在 coreset 上精确求解才有用；stable coreset 在保留均匀采样高效性的同时补上了子度量传递这一关键能力。
- **vs 重要性采样 coreset（Jiang et al. 2024）**：用迭代 sensitivity 采样得到维度无关的强 coreset，但需读全量数据、构造耗时数百秒；本文实验证明均匀采样在相近精度下快五六个数量级。
- **启发**：把"strong vs weak"这种二元对立细化出"中间态 stable"，并用一个可逼近的充分条件（RCDA）作桥，是一种可迁移的理论设计范式——在其他需要"保排序而非保绝对值"的摘要/草图问题上或许同样能松弛出更易构造的中间概念。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 提出 weak/strong 之间的全新 stable coreset 概念，首次为均匀采样在 $\ell_1$ 及一大类嵌入度量上给出严格的 coreset 保证。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 个真实数据集、4 类实验（精度/时间/约束/维度），但作为理论论文实验主要起验证作用、规模有限。
- 写作质量: ⭐⭐⭐⭐⭐ 概念层级、框架与实例化两段式证明、嵌入推广脉络清晰，技术综述与定位到位。
- 价值: ⭐⭐⭐⭐⭐ 把"启发式"的均匀采样升格为有理论支撑且广泛适用的工具，对大规模聚类摘要有直接实践与理论意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On Coreset for LASSO Regression Problem with Sensitivity Sampling](on_coreset_for_lasso_regression_problem_with_sensitivity_sampling.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] Random Spiking Neural Networks are Stable and Spectrally Simple](random_spiking_neural_networks_are_stable_and_spectrally_simple.md)
- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)
- [\[ICLR 2026\] Discounted Online Convex Optimization: Uniform Regret Across a Continuous Interval](discounted_online_convex_optimization_uniform_regret_across_a_continuous_interva.md)

</div>

<!-- RELATED:END -->
