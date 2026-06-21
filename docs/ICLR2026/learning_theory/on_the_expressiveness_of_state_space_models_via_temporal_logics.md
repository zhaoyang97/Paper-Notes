---
title: >-
  [论文解读] On the Expressiveness of State Space Models via Temporal Logics
description: >-
  [ICLR 2026][学习理论][线性时序逻辑] 本文用有限迹上的纯过去线性时序逻辑（PLTLf）及其计数/模算子扩展，给不同门控机制（对角门控 S6、时不变 S4、混合门控）和不同算术精度（定宽 vs. 对数精度）的状态空间模型（SSM）刻画出一套表达能力下界层级，并证明了若干硬性不可表达结论（如定宽对角 SSM 无法识别 $(aa)^*$），同时把 SSM 与 Transformer 的已知逻辑刻画对齐起来。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "表达能力"
  - "状态空间模型"
  - "线性时序逻辑"
  - "表达能力下界"
  - "算术精度"
  - "不可表达性"
---

# On the Expressiveness of State Space Models via Temporal Logics

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Vg511oJScS](https://openreview.net/forum?id=Vg511oJScS)  
**代码**: 无  
**领域**: 学习理论 / 表达能力 / 状态空间模型  
**关键词**: 状态空间模型, 线性时序逻辑, 表达能力下界, 算术精度, 不可表达性

## 一句话总结
本文用有限迹上的纯过去线性时序逻辑（PLTLf）及其计数/模算子扩展，给不同门控机制（对角门控 S6、时不变 S4、混合门控）和不同算术精度（定宽 vs. 对数精度）的状态空间模型（SSM）刻画出一套表达能力下界层级，并证明了若干硬性不可表达结论（如定宽对角 SSM 无法识别 $(aa)^*$），同时把 SSM 与 Transformer 的已知逻辑刻画对齐起来。

## 研究背景与动机
**领域现状**：SSM（如 S4、Mamba/S6、RetNet、Griffin）作为 Transformer 的线性时间替代架构正快速兴起。与之并行，理论界一直在用「逻辑 / 电路复杂度」这套语言去刻画神经序列模型「原则上能识别哪些语言」——这类分析独立于训练数据和优化过程，回答的是「这个架构本质上能不能表示某种模式」。Transformer 这边已经有相当成熟的结果（UHAT 对应一阶逻辑 FO[<]、AHAT 对应带计数的时序逻辑等，见 Strobl 等的综述），但 SSM 这边的逻辑刻画几乎是空白。

**现有痛点**：已有 SSM 理论结果零散且只给出粗粒度的上下界。Merrill 等（2024）用电路复杂度给出上界：对角门控、时不变、定宽/对数精度的 SSM 都落在 $\mathrm{TC}^0$ 内；Sarrof 等（2024）给出对角 SSM 的下界——能识别所有无星（star-free）语言，并在特定输出函数下证明该下界是紧的。但这些工作：(1) 没有刻画时不变和混合 SSM；(2) 没有系统分析算术精度（定宽 vs. 对数精度）对表达能力的影响；(3) 没有把 SSM 放进 Transformer 的逻辑刻画图谱里做直接对照。

**核心矛盾**：SSM 的表达能力到底由什么决定？作者识别出两个关键维度——**门控机制类型**（门矩阵能否依赖输入、是否被限制为对角）和**算术精度**（每步用固定 $b$ 比特，还是随输入长度 $n$ 增长到 $O(\log n)$ 比特）。这两个维度如何共同决定一个 SSM 能识别的语言类，此前没有统一答案。

**本文目标**：把 SSM 的各个变体精确映射到 PLTLf 的不同片段上，从而 (1) 给出比已有工作更细的下界层级；(2) 厘清精度对计数能力的影响；(3) 与 Transformer 的逻辑刻画一一对齐。

**切入角度**：作者沿用刻画 Transformer 时用过的同一套逻辑工具——有限迹上的纯过去线性时序逻辑 PLTLf。之所以用「纯过去」而非完整 LTLf，是因为 SSM 的递归结构在每一步只能看到输入的前缀，天然只支持「向后看」的算子。这让 SSM 与 Transformer 的对比变得「同尺度可比」。

**核心 idea**：把每个 PLTLf 公式按句法树自底向上分解，让每个子公式对应 SSM 隐状态的一维、每层 SSM 计算一层子公式，从而把「逻辑公式 → SSM 构造」做成构造性证明；再据此分门控、分精度推出整套下界，并用单调性引理给出配套的不可表达性。

## 方法详解

### 整体框架
本文是一篇纯理论论文，没有实验，「方法」即一套形式化的表达能力分析。整体可以理解为一台「把时序逻辑公式翻译成 SSM」的构造机器加上一组配套的不可能性证明：先统一形式化 SSM（层 = 元组 $(h_0, \text{gate}, \text{inc}, \phi)$，递归 $h_t = \text{gate}(x_t)\cdot h_{t-1} + \text{inc}(x_t)$，输出 $z_t = \phi(h_t, x_t)$），区分四类门控（对角、时不变、混合、任意）和两类精度（定宽、对数精度）；再用 PLTLf 及其扩展（计数算子 $\overleftarrow{\#}$、模谓词 MOD）作为「标尺」，对每个 (门控, 精度) 组合给出它能识别的语言类下界；最后用一条单调性引理证出定宽对角 SSM 的硬上界（无法识别 $(aa)^*$），并把全部结论嵌进 Transformer 的 UHAT/AHAT 图谱。

核心证明骨架是「逐子公式逐层构造」：要在位置 $i$ 上评估 PLTLf 公式 $\varphi$，把 $\varphi$ 的子公式按嵌套深度 $\mathrm{nd}(\varphi)$ 排序（独立子公式可并行进同一层），每个子公式 $\psi$ 用一层 SSM 的一维隐状态来标记「$\sigma, i \models \psi$ 是否成立」，整个公式所需层数 = 句法树深度。不同门控/精度能否实现某个时序算子，决定了它对应到 PLTLf 的哪个片段。

### 关键设计

**1. 统一的 SSM 形式化与四类门控划分：把架构差异收敛成「门函数」一个变量**

要做精细的表达能力刻画，第一步是把五花八门的 SSM 架构抽象成可分析的对象。作者沿用 Merrill 等（2024）的广义设定：一层 SSM 是元组 $(h_0, \text{gate}, \text{inc}, \phi)$，把输入序列 $x_1\cdots x_k$ 经线性递归 $h_t = \text{gate}(x_t)\cdot h_{t-1} + \text{inc}(x_t)$ 变成隐状态再经 $\phi$（带残差，由 ReLU FNN 实现）输出。整个 $L$ 层 SSM 加上嵌入 emb 和输出 out，计算一个 $\Sigma^* \to \mathbb{R}$ 的函数，以 $S(\sigma)=1$ 判定接受。关键在于把架构差异全部归到门函数上：**时不变** SSM 要求存在常矩阵 $A$ 使 $\text{gate}(x)=A$（对应 S4）；**对角门控** SSM 允许 $\text{gate}(x)$ 是非负对角矩阵但可依赖 $x$（对应 S6/Mamba）；**混合** SSM 每层要么对角要么时不变（实践中暂未使用，但理论上其上界可直接套用到前两类）；**任意** 门控不加任何限制。这套划分让后续每条定理都能精确锚定到「哪一类门控 + 哪种精度」。

**2. 三层表达力标尺：PLTLf → 计数扩展 $\overleftarrow{\#}$ → 模谓词 MOD**

作者把「序列能力」拆成三个递进概念层，并各配一个逻辑片段做标尺。最底层是**模式匹配**——纯过去 LTLf（PLTLf，算子为 yesterday $Y$、previously $P$、since $S$），它刻画「按相对顺序检测事件而不计数」的能力，等价于一阶逻辑 FO[<] / 无星正则语言。中间层加**向后计数算子** $\overleftarrow{\#}$：$\overleftarrow{\#}\,\varphi$ 数的是当前位置 $i$ 及之前满足 $\varphi$ 的位置数，新增的计数子公式形如 $\sum_i a_i \overleftarrow{\#}\,\varphi_i \sim c$（$\sim\in\{<,\le,=,\ge,>\}$）；之所以只用 $\overleftarrow{\#}$ 而不用向前的 $\overrightarrow{\#}$，正是因为 SSM 每步只能看前缀。最上层加**模谓词** MOD：$\sigma, i \models \mathrm{MOD}^m_r \iff i \equiv r \pmod m$，它让模型能区分「周期性位置」（如奇偶），把表达力抬到 AC$^0$ 内全体正则语言（$\equiv$ FO[<, MOD]），这正好对应 Transformer 加位置编码的作用。三层标尺一搭起来，每个 SSM 变体落在哪一层就一目了然。

**3. 对角 SSM 的构造性下界：since 算子为何「必须」依赖输入的对角门**

Theorem 1 证明定宽对角 SSM 能识别所有 PLTLf 可定义语言（无星语言），且是构造性的。证明思路是：每个子公式占隐状态一维，inc 函数加上逐位置 FNN 负责非时序子公式（合取、否定、比较），每层之后隐状态是一个布尔向量标记各子公式是否成立。关键洞察在于不同时序算子对门控的要求不同：$Y$（昨天）只需引用前一位置、$P$（曾经）只需累积，二者用对角且时不变的门就能实现，无需与当前输入做合取；而 **since 算子 $S$ 本质递归**——$\varphi\,S\,\psi \equiv \psi \lor (\varphi \land Y(\varphi\,S\,\psi))$，求值要把前一状态与「当前输入对 $\varphi$ 的评估」做合取，这个「依赖当前输入」恰恰逼出了**对角门必须依赖输入** $x$ 的结构。换句话说，对角 SSM 之所以比时不变 SSM 强在能算 since，就是因为它的门能随当前符号变化。Theorem 4 进一步说明：配上对数精度后，再加一层用 inc 累计各 $\varphi_j$ 出现次数、用 FNN 检查线性组合是否满足比较 $\sim c$，对角 SSM 就能识别 PLTLf[$\overleftarrow{\#}$]，从而识别非正则甚至非上下文无关语言，例如 $\{a^n b^n c^n\}$ 可写成
$$\varphi = H\big((a \to \neg P(b\lor c)) \land (b \to \neg P\,c)\big) \land (\overleftarrow{\#}a - \overleftarrow{\#}b = 0) \land (\overleftarrow{\#}c - \overleftarrow{\#}b = 0).$$

**4. 单调性引理与时不变 SSM 的循环置换门：两类不可比的能力边界**

下界之外，作者用一条单调性引理给出对角 SSM 的硬上界。Lemma 2：定宽对角 SSM 在重复读同一符号 $\sigma$ 时，输出必然稳定——因为门对角意味着每一维独立演化 $(h_t)_i = \text{gate}(\text{emb}(\sigma))_i\cdot(h_{t-1})_i + \text{inc}(\text{emb}(\sigma))_i$，非负门使该序列单调，而定宽精度下每维只能取有限个值，故存在 $N$ 使 $n\ge N$ 时输出恒定。由此 Theorem 3 立得：定宽对角 SSM **无法识别** $(aa)^*$，因为它会对所有足够长的 $a^n$ 一律接受或一律拒绝，而 $(aa)^*$ 要区分奇偶长度。这是一条独立于训练数据/优化器的硬架构瓶颈。反观时不变 SSM，它做不了依赖当前输入的 since（Conjecture 1 猜想它连 $L(a\,S\,b)$ 都识别不了），却能做对角 SSM 做不到的事：Lemma 6 用一个 $m$ 阶**循环置换矩阵** $P$ 当门（$B=0$），让 $h_t = P^t h_0$ 在 $m$ 个基向量间循环，从而在隐状态里维护一个模 $m$ 计数器，单层即可计算任意模谓词，于是识别 $L(H\,a \land \mathrm{MOD}^2_0)=(aa)^*$。这就揭示对角与时不变 SSM **在表达力上不可比**：前者会 since 不会模周期，后者反之。把两者层混合（Mixed），Corollary 8 得到既能 since 又能模谓词，识别 PLTLf[MOD]，即 AC$^0$ 内全体正则语言；任意门控则按 Merrill 等结果识别全体正则语言。

### 一个例子：用模谓词识别奇数位置
取 $\mathrm{MOD}^2_1$（判断位置是否为奇数）为例走一遍 Lemma 6 的构造：$m=2$，置换矩阵 $P=\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$，初始 $h_0=(1,0)^T$。状态序列在奇数位变成 $e_1=(0,1)^T$、偶数位变成 $e_0=(1,0)^T$，如此交替。隐状态向量的「亮起的那一维」始终编码当前位置模 2 的余数——这就是时不变 SSM 仅靠门矩阵（不靠 inc 加性项）就能维护周期计数、进而识别 $(aa)^*$ 的全部机制；而同样的事对角 SSM 做不到，因为它的单调性会抹平奇偶差异。

## 实验关键数据
本文是纯理论工作，无训练实验，核心「数据」是表达能力层级表与各架构和 Transformer 的对齐关系。下面两张表概括主要结论。

### 主结果：SSM 各变体的表达力下界

| SSM 变体 | 定宽精度（下界） | 对数精度（下界） | 出处 |
|----------|------------------|------------------|------|
| 对角门控 | PLTLf $\equiv$ FO[<]（无星语言） | PLTLf[$\overleftarrow{\#}$]（含 $a^nb^nc^n$ 等非正则语言） | Thm. 1 / Thm. 4 |
| 时不变 | UN-PLTLf[MOD] | UN-PLTLf[MOD, $\overleftarrow{\#}$] | Thm. 7 |
| 对角且时不变 | UN-PLTLf | UN-PLTLf[$\overleftarrow{\#}$] | Cor. 5 |
| 混合 | PLTLf[MOD] $\equiv$ AC$^0\cap$REG | PLTLf[$\overleftarrow{\#}$, MOD] | Cor. 8 |
| 任意门控 | 全体正则语言 REG | — | Merrill 等（2024） |

### 不可表达性与精度门槛

| 结论 | 内容 | 说明 |
|------|------|------|
| Lemma 2（单调性） | 定宽对角 SSM 重复读同一符号，输出最终稳定 | 每维非负门单调 + 定宽取值有限 |
| Theorem 3 | 定宽对角 SSM 无法识别 $(aa)^*$ | $(aa)^*$ 非单调，矛盾于 Lemma 2 |
| Conjecture 1 | 猜想定宽时不变 SSM 无法识别 $L(a\,S\,b)$ | since 需依赖当前输入的门，时不变做不到 |
| 全局计数需对数精度 | $a^nb^nc^n$ 等计数语言严格需要 $O(\log n)$ 比特 | 定宽精度只能识别正则语言（Zubic 等 2025） |

### 关键发现
- **门控决定「质」，精度决定「计数」**：是否对角/时不变决定能不能算 since 或模谓词（质的差别），定宽 vs. 对数精度则决定能不能做全局计数（识别 $a^nb^nc^n$ 这类非正则语言）。两个维度正交。
- **对角与时不变 SSM 不可比**：对角会 since（star-free 全包）但抓不住周期奇偶；时不变会模谓词（能识别 $(aa)^*$）但疑似不会 since。混合层正好取二者之并。
- **与 Transformer 精确对齐**：定宽对角 SSM $\equiv$ 无位置编码的 UHAT（都是 FO[<]）；加时不变层 $\approx$ 给 Transformer 加位置编码（都获得模谓词，到 FO[<, MOD]）；对数精度下 SSM 严格包含软注意力 SAT 的 C-RASP（因 SSM 有 yesterday 这类强局部算子），但严格弱于 AHAT+PE——因为 SSM 受因果限制只能向后计数 $\overleftarrow{\#}$，而全局注意力能向前计数 $\overrightarrow{\#}$。
- **下界与上界之间留有缝隙**：本文给出的定宽 SSM 下界都落在 AC$^0$ 内，而 Merrill 等的上界是 $\mathrm{TC}^0$（AC$^0 \subsetneq \mathrm{TC}^0$）。作者推测上界可收紧到 AC$^0$（因为没发现任何能算「模常数计数 / parity」的机制），若成立则 SSM 与 UHAT 一样被卡在 AC$^0$。

## 亮点与洞察
- **「逐子公式逐层」的构造范式很优雅**：把逻辑公式的句法树深度直接换算成 SSM 层数、每个子公式占一维隐状态，这套构造既给出下界又天然解释了「为什么 since 需要输入依赖门」——机制层面看得见摸得着，而非黑箱论断。
- **单调性引理是「四两拨千斤」的不可能性武器**：仅靠「对角门非负 + 定宽取值有限 ⟹ 重复同一符号必稳定」就一锤定音地证明定宽对角 SSM 连 $(aa)^*$ 都识别不了，而且这是训练无关的硬瓶颈——optimizer 再强、数据再多也学不会。这类「impossibility」结论对实践选型很有指导价值。
- **SSM↔Transformer 的桥很有迁移价值**：把「门控类型 ↔ 注意力类型」「精度 ↔ 硬/软注意力」「时不变层 ↔ 位置编码」一一对上，等于给两套架构建立了可换算的能力词典。以后看到「某 Transformer 变体能做 X」，可以反查「对应 SSM 该用什么门控/精度」。
- **循环置换门当模计数器**是个可复用的小技巧：用置换矩阵的周期性 $P^m=I$ 在隐状态里免费维护一个模 $m$ 计数器，提示我们在设计需要周期感知的递归模块时，时不变门并非一无是处。

## 局限与展望
- **多为下界 + 一条关键结论仍是猜想**：核心结果是下界（「至少能识别…」），与上界（$\mathrm{TC}^0$）之间存在 AC$^0$ 的缝隙未闭合；且「时不变 SSM 不能识别 $a\,S\,b$」目前只是 Conjecture 1，作者坦言因时不变 SSM 各维会相互作用、加上定宽算术不结合/有饱和效应，形式化证明困难。
- **理论模型与工程实现有距离**：分析假设非负对角门、饱和定宽算术、ReLU FNN 等理想化条件，混合 SSM 实践中尚无人使用；真实 Mamba/S4 的浮点细节、归一化、训练动态都不在刻画范围内，「能表达」不等于「能学到」。
- **只刻画了「向后计数」**：受 SSM 因果结构所限，本文只覆盖 $\overleftarrow{\#}$，没有也无法覆盖向前计数 $\overrightarrow{\#}$，因此与 AHAT+PE 的差距是结构性的、不是分析疏漏。
- **可改进方向**：作者建议把上界收紧到 AC$^0$（验证 SSM 是否真的算不了 parity），以及按公式嵌套深度建立更细粒度的复杂度层级（类似 Transformer 已有的 nesting-depth 层级）。

## 相关工作与启发
- **vs Merrill 等（2024）**：他们用电路复杂度给上界（对角/时不变/各精度 SSM $\subseteq \mathrm{TC}^0$，任意门控 SSM $\supseteq$ NC$^1$-完全语言）。本文给的是逻辑下界，二者互补；本文进一步推测对定宽受限 SSM 上界可收紧到 AC$^0$，缩小与上界的缝隙。
- **vs Sarrof 等（2024）**：他们证对角 SSM 能识别全体无星语言并在特定输出函数下紧。本文 Theorem 1 把这一结果作为更大层级中的特例「复原」，并沿三条新轴扩展——刻画时不变/混合 SSM（接上模谓词）、分析精度影响（对数精度对非正则计数语言严格必要）、整合进 Transformer 图谱。
- **vs Zubic 等（2025）**：他们证定宽 SSM 只能识别正则语言。本文与之一致并细化：在正则范围内进一步按门控分出 PLTLf / UN-PLTLf[MOD] / PLTLf[MOD] 等层级。
- **vs Transformer 逻辑刻画（Yang 等 2024a、Barcelo 等 2024、Yang & Chiang 2024）**：UHAT$\equiv$FO[<]、UHAT+PE$\equiv$FO[<, MOD]、AHAT$\equiv$PLTLf[$\overleftarrow{\#},\overrightarrow{\#}$, MOD]、SAT$\equiv$C-RASP。本文把 SSM 各变体精确插进这张图：定宽对角 SSM 对齐 UHAT、加时不变层对齐 UHAT+PE、对数精度 SSM 严格包含 SAT 但严格弱于 AHAT+PE。这是首个把 SSM 表达力与形式逻辑直接连起来的工作。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次用时序逻辑系统刻画 SSM 表达力，并按门控×精度建出完整下界层级、与 Transformer 精确对齐。
- 实验充分度: ⭐⭐⭐⭐ 纯理论无实验，但定理/引理/反例链条完整自洽；唯一关键结论（时不变不能识别 since）仍是猜想。
- 写作质量: ⭐⭐⭐⭐⭐ 三层标尺 + 逐子公式逐层构造的叙述清晰，图 1–3 把层级与对齐关系交代得很直观。
- 价值: ⭐⭐⭐⭐⭐ 给出训练无关的硬架构瓶颈，对 SSM 选型与「能不能学」的判断有实质指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Quotient-Space Diffusion Models](quotient-space_diffusion_models.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [\[ICLR 2026\] Splat Regression Models](splat_regression_models.md)
- [\[ICLR 2026\] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks](from_neural_networks_to_logical_theories_the_correspondence_between_fibring_moda.md)

</div>

<!-- RELATED:END -->
