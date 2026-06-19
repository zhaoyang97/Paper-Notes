---
title: >-
  [论文解读] Formalizing and Falsifying Causal Pathways of Rare Events
description: >-
  [ICML 2026][因果推理][因果通路] 本文把罕见事件的"口头因果解释"形式化为 **causal pathway**——一个由二值化事件构成的子图，并定义 **pathway explanation score** 来量化"根因 + 中介通路"对目标事件的解释力，得到一套可证伪的因果解释评价框架。
tags:
  - "ICML 2026"
  - "因果推理"
  - "因果通路"
  - "罕见事件"
  - "解释分数"
  - "因果抽象"
  - "可证伪性"
---

# Formalizing and Falsifying Causal Pathways of Rare Events

**会议**: ICML 2026  
**arXiv**: [2605.31254](https://arxiv.org/abs/2605.31254)  
**代码**: 无  
**领域**: 因果推断 / 可解释 AI / 罕见事件根因分析  
**关键词**: 因果通路, 罕见事件, 解释分数, 因果抽象, 可证伪性  

## 一句话总结
本文把罕见事件的"口头因果解释"形式化为 **causal pathway**——一个由二值化事件构成的子图，并定义 **pathway explanation score** 来量化"根因 + 中介通路"对目标事件的解释力，得到一套可证伪的因果解释评价框架。

## 研究背景与动机

**领域现状**：异常 / 罕见事件（自然灾害、股市崩盘、技术系统故障、基因表达异常等）的因果分析目前主要走 **root cause analysis (RCA)** 路线——在结构因果模型 (SCM) 里找一小撮"根因节点"，让目标事件的反事实概率显著上升。

**现有痛点**：RCA 只回答"谁动了"，不回答"怎么动到目标的"。一条真实解释往往包含 (i) 多个互相作用的中介机制，(ii) 控制传播的上下文变量。仅给出根因列表 → 对人类不可解释；对 AI 系统也无法被实验证伪。

**核心矛盾**：现有的两类近邻工作各自残缺——
1. **极值统计因果模型** (Engelke 2025, Klüppelberg 2026)：依赖渐近 / 重尾参数假设，无法处理"非极端但统计上罕见"的事件（如非常接近 0 的取值、严重不平衡的二元事件）。
2. **路径中介分析** (path-specific effects)：分解的是平均因果效应，回答"路径 A 占多大份额"，但不回答"对于这一次具体观测，哪部分图能给出好的解释"。
3. **因果抽象** (Rubenstein 2017, Beckers 2020)：作用在整个模型层级，缺少 **event-level** 的局部抽象概念。

**本文目标**：建立一个 **event-level** 的因果解释形式系统，要满足：(a) 不依赖渐近极值假设；(b) 适用任意取值空间（连续 / 离散 / 文本嵌入）；(c) 解释本身可被数据或一致性测试证伪；(d) 能从更细粒度 SCM 通过抽象自动得到。

**切入角度**：把"解释"重新定义为一个**子图 + 二值化事件集合**，而不是一个根因列表。具体而言，给目标事件 $B_t=1$ 和一组根因 $\mathbf{B}_R$，看 $do(\mathbf{B}_R=\mathbf{1})$ 之后整条通路上其他事件 $\mathbf{B}_{K\setminus R}=\mathbf{1}$ 同时发生的对数似然有多接近目标的对数稀有度。这把"中介事件也要看起来合理"显式写进了打分函数里。

**核心 idea**：用 $\mathcal{E}^K_{R\to t} := 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(B_t=1)}$ 这一**对数似然比**作为可证伪的解释质量度量；并通过**特征单调性 (feature monotonicity)** 把任意空间的变量统一二值化，使整套理论从二值 SCM 平滑推广到连续 / 离散 / 文本变量。

## 方法详解

### 整体框架
本文要解决的是：怎么把"因为 A 所以 B 所以 C 所以目标事件"这种口头因果链，变成一个能被数据或概率信念证伪的分数。做法是把整条解释重新组织成一个二值事件的子图（pathway），先在二值 SCM 上定义一个 $[0,1]$ 的解释分数，再用"特征单调性 + 二值化"把这套理论从二值变量推广到实值 / 离散 / token 等任意空间，最后用"事件级因果抽象"量化从细粒度模型粗粒化时损失了多少。输入是 (SCM、观测样本、目标事件)，输出是 (pathway 子图 $\mathcal{P}$、根因集合 $R$、解释分数 $\mathcal{E}^K_{R\to t}$、抽象精度 $r$)。

### 关键设计

**1. Cluster / Pathway explanation score：让一条因果链可以被打分证伪**

最朴素的根因分析只会输出一组根因，并不审查"中介事件本身是否合理"，于是"听起来对"的解释无法被推翻。本文先沿用 Oesterle 2025 的 cluster 分数 $\mathcal{E}_{R\to K} = 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(\mathbf{B}=\mathbf{1})}$，它衡量"对根因集合 $R$ 做 $do(\mathbf{B}_R=\mathbf{1})$ 后，整簇事件被拉到多大可能"，越接近 1 越说明根因撑得起整簇。但 cluster 分数只看根因有没有抬高簇似然，不强求中介"看起来正常"。

pathway 分数把分母换成目标事件自己的对数稀有度 $\log P(B_t=1)$：$\mathcal{E}^K_{R\to t} = 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(B_t=1)}$。当目标比整簇更稀有时，这个分数比 cluster 分数更严格——任何"中介事件本身就很奇怪"的环节都会被分子里的 log-likelihood 项惩罚。两者满足仿射关系 $1-\mathcal{E}^K_{R\to t} = (1-\mathcal{E}_{R\to K}) \cdot \frac{\log P(\mathbf{B}=\mathbf{1})}{\log P(B_t=1)}$，因此 pathway 上每个节点的贡献仍然可加（公式 11），可以贪心地逐个挑根因。Lemma 3.7 进一步给出每条边的 log-likelihood gap $\Delta_i := [\log P(B_{\mathrm{Pa}(i)}=\mathbf{1}) - \log P(B_i=1)]_+$ 来控制分数上界，gap 越大说明该机制本身越稀有、越拖累整条链。这套定义正是把"口头因果链里每条边都隐含'这步不算太奇怪'的承诺"显式写进打分函数的关键。

**2. Feature monotonicity + 二值化：把理论从二值变量搬到任意空间**

真实系统的变量是实值 / 类别 / token / 嵌入，并不天然是二值事件。本文给每个 $X_j$ 配一个特征函数 $\tau_j:\mathcal{X}_j\to\mathbb{R}$，把变量映射成二值事件 $B_j := \{\tau_j(X_j) \geq \tau_j(x_j)\}$（即"$X_j$ 的特征至少和观测一样大"），并要求机制 $P(X_j\mid \mathbf{X}_{\mathrm{Pa}(j)})$ 对 $(\tau_j,\tau_{\mathrm{Pa}(j)})$ 单调：父变量特征越大，子变量特征的分布随机地越大。

在这个条件下，Lemma 4.2 给出关键的尾概率保证："从 $P(X_j\mid \mathbf{x}_{\mathrm{Pa}(j)})$ 采样的 $x_j$，其条件似然 $\leq \alpha$ 的概率不超过 $\alpha$"。Theorem 4.3 把它推广到整个 DAG：对任意 $\mathbf{x}_R$，从 $do(\mathbf{x}_R)$ 生成其余变量后，负对数似然 $L\geq c$ 的概率不超过 $\sum_{i=0}^{n-|R|-1}\frac{c^i}{i!}e^{-c}$（一个自由度修正后的 Poisson 尾）——这恰好是解释分数偏离 1 时可以报告的 **p 值**。选用"$X\geq x$ 解释 $Y\geq y$"而不是"$X=x$ 解释 $Y=y$"，是因为前者鲁棒于具体取值、更符合人类语言、也更可证伪；而即便真实分布不严格满足特征单调，这个 p 值仍能当作"允许多大偏离"的诊断阈值。特别地，取 $\tau_X(x):=-|x|$ 就能表达"$x$ 是正常值"这类**非极端但相关**的事件（Example 4.8），覆盖了渐近极值理论根本触及不到的场景。

**3. Pathway abstraction + natural micro-realization：量化粗粒化损失**

要从一个细粒度 SCM $(\mathcal{G}, P_\mathbf{X})$ 自动得到粗粒度的 pathway 解释 $(\mathcal{C}, \mathcal{P}, P_\mathbf{B})$，必须知道"二值化抽象丢了多少信息"。本文定义抽象精度 $r := 1 - \max_{S, \mathbf{b}_S} \frac{D_{KL}[P_\mathbf{X}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))\,\|\,P_\mathbf{B}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))]}{-\log P_\mathbf{X}(B_t=1)}$，把"抽象后干预分布与真实干预分布的 KL 距离"按目标稀有度归一化。难点在于 $do(\mathbf{B}_j=b_j)$ 在原模型里本是病态的——多个 $X_j$ 对应同一个 $B_j$，无法直接干预。**natural micro-realization** 把它定义清楚：$do(\mathbf{B}_S=\mathbf{b}_S)$ 解释为"先从 $\prod_{i\in S}P_\mathbf{X}(X_i\mid B_i=b_i)$ 独立采样底层变量，再在原模型上做 $do$"，从而得到一致的概率算子。

关键好处是，解释分数本身也能改写成同款 KL 形式 $\mathcal{E}^K_{R\to t} = 1 - \frac{D_{KL}(\delta_\mathbf{1}\|P_\mathbf{B}(\mathbf{B}\mid do(\mathbf{B}_R=\mathbf{1})))}{-\log P_\mathbf{B}(B_t=1)}$，与精度 $r$ 落在同一量纲。于是"要不要把某个上下文变量纳入 pathway"这种设计选择，就变成可计算的精度 vs 解释分数取舍。Example 4.8 里，保留上下文节点 $B_1$（"$|X|\leq x$"）的三元 pathway 在精度和解释分数上都明显优于二元简化 $B_2\to B_3$——一旦把上下文剔除，混淆路径的负向效应就被吃进解释里，让条件概率失真。

### 一个完整示例
以三元链 (Example 3.6) 看分数怎么逐节点补齐：设 $P(b_1^1)=10^{-3}$、$P(b_3^1\mid b_2^1)=10^{-3}$、$P(b_4^1\mid b_3^1)=10^{-2}$，目标是 $B_4=1$。先只取根因 $R=\{1,3\}$，此时 $B_4$ 那条边的机制概率 $10^{-2}$ 并没被纳入解释，算得 $\mathcal{E}^K_{R\to t}=3/4$——还差一截，因为"$B_3$ 触发 $B_4$"这步本身也稀有、没人解释。把 $B_4$ 也加进来后分数升到 1，对应"沿链每个机制都被审视、都合理"。这直观说明：只有那些"机制本身就稀有"的节点被显式纳入，pathway 才算补齐。

### 损失函数 / 训练策略
本文是纯理论框架，不涉及学习。评估时所有概率假设要么从观测样本估计（数据一致性测试），要么从 LLM / 专家以问答方式估计（内部一致性测试）。从 cluster 选根因集合 $R$ 的实操方式是贪心算法：$R\gets R\cup\{\arg\max_i \mathcal{E}^K_{\{i\}\cup R\to t}\}$，由公式 (11) 的可加性保证最优性，复杂度 $O(|K|\cdot|R|)$。

## 实验关键数据

本文以**理论 + 概念性算例**为主，没有标准基准比较。三类典型算例展示了框架行为：

### 主算例：解释分数随事件稀有度的形状

| 算例 | 设置 | 解释分数 | 含义 |
|------|------|---------|------|
| Gaussian 因果对 (Ex 4.6) | $Y=\rho X+N$, $\rho=0.5$, $x\geq 3$, $y\approx \rho x$ | $\geq 0.8$ | "$X\geq x$ 解释 $Y\geq y$ 至少 80%"，对应人类语言中可接受的因果陈述 |
| 三元链 (Ex 3.6) | $P(b_1^1)=10^{-3}, P(b_3^1\mid b_2^1)=10^{-3}, P(b_4^1\mid b_3^1)=10^{-2}$ | $R=\{1,3\}: 3/4$；$R=\{1,3,4\}: 1$ | 加入"机制本身就稀有"的节点才能补齐 pathway |
| 上下文混淆 (Ex 4.7) | $B_1$ 半概率，$P(B_2=1\mid B_1=1)=\delta$，$B_3=B_1\wedge B_2$ | 三元 pathway: $\to 1$；二元 $B_2\to B_3$ 精度 $\to 1/2$ ($\delta\to 0$) | 忽略上下文 $B_1$ 让 do-后验偏离 50% |

### 真实 LLM 演示：流浪汉成因 pathway

作者让 LLM 给一个虚构案例（35 岁精神分裂男性 $A$ → 被解雇 $B$ → 被驱逐 $C$ → 家人断联 $D$ → 长期流浪 $E$）生成因果链，再用同一 LLM 单独估计各机制条件概率：

| 边 | 条件概率 |
|----|---------|
| $P(B\mid A)$ | 0.55 |
| $P(C\mid B)$ | 0.80 |
| $P(D\mid C)$ | **0.05** |
| $P(E\mid D)$ | 0.20 |
| $P(E)$ 先验 | 0.0005 |

取 $R=\{A\}$，pathway explanation score $\mathcal{E}^K_{R\to t} = 1 - \frac{\log(0.55\cdot 0.8\cdot 0.05\cdot 0.2)}{\log 0.0005} \approx 0.29$，**明确指出弱链在 $C\to D$**——"被驱逐"本身并不能解释"家人断联"。建议改法：加 $A\to D$ 直边，或重写 $C$。这正是框架的**证伪能力**示例。

### 关键发现
- **链上"哪个机制最稀有"主导分数**：log-likelihood gap (Lemma 3.7) 越大，pathway 分数上界越紧。这给出诊断 LLM 因果叙事的工具：分数低且某条边的条件概率显著小 ⇒ 该边可疑。
- **"非稀有上下文"必须显式建模**：Example 4.7 / 4.8 表明，控制传播但本身不罕见的事件（如"$|X|$ 正常"）若被剔除，二元抽象精度可掉到 0.5。这与传统 RCA 只看"出格的根因"形成鲜明对比。
- **必要性是隐式的**：尽管解释分数没显式包含"反事实必要性"（rung 3），但因为目标事件稀有，干预后高似然 $\Rightarrow$ 高 probability of necessity（Appendix B），从而无需引入反事实公理。

## 亮点与洞察
- **把"口头因果链"翻译成可证伪打分**：人类 / LLM 自然语言中说"因为 A 所以 B 所以 C 所以 D"时，每条边都隐含了"中介事件的发生不算太奇怪"的承诺；本文把这种隐含承诺显式写成 log-likelihood 项进入分数，让"听起来对"的解释能被数据或同一 agent 的概率信念证伪。
- **特征单调性 + Poisson 尾 p 值**：用一个简洁的可微条件把整套理论从二值 SCM 推到任意空间，同时给出 p 值校正多检验。这是把统计极值理论思想换成"对数尺度"重做，避开了渐近假设，能处理 token / 嵌入这类离散对象。
- **explanation score 与 abstraction accuracy 同一量纲**：两者都被归一化到 $-\log P(B_t=1)$，使"我要不要把上下文 $B_1$ 加进 pathway"这种设计选择变成数值取舍。这种 KL 视角与 mediation 分析里的 path-specific effects 形成清晰互补——前者评估"对这一个观测的整体解释力"，后者拆分"对平均效应的占比"。
- **可迁移到 LLM 自检**：第 5 节的 homelessness 例子是一个最小可行 demo，演示了如何"用同一 LLM 既生成因果链、又估计边权重、再用框架打分"——这条 pipeline 可直接接到任何生成式 AI 的事实性 / 因果性自检流程。

## 局限与展望
- **作者承认**：框架只回答"解释是否与数据 / 信念一致"，不回答"潜在因果图是否真实"；高分不能证明解释正确。
- **特征单调性是较强假设**：实际系统（含 sin / 多模态 / 阈值响应）不一定满足，作者建议用 p 值容忍偏离，但缺少在真实数据集上的鲁棒性研究。
- **二值化破坏 Markov 性**：即使 $X_1\to X_2\to X_3$ 满足 $X_1\perp X_3\mid X_2$，二值化后可能不再独立，因此抽象 $P_\mathbf{B}$ 与真实 $P_\mathbf{X}$ 的 KL 距离需要主动控制；高维下 $r$ 的估计代价没有讨论。
- **未给出从数据自动选 pathway 的算法**：本文给的是"评估给定 pathway"的工具，发现 pathway 仍需人类 / LLM 提议；下一步可以把贪心选 $R$ 扩展到"贪心选 pathway 子图 + 上下文节点"。
- **LLM 演示规模小**：仅一个手工示例，没有大规模评估 LLM 因果叙事的统计指标，期待后续在 medical NLI / 法律因果论证等数据集上做系统实验。

## 相关工作与启发
- **vs RCA (Budhathoki 2022, Lin 2018/2024, Li 2022, Gnecco 2021)**：RCA 只输出根因集合 + 反事实贡献，本文额外要求中介通路本身可解释、被打分，并把"贡献分配"从 Shapley-value 模式（依赖 context 选择）换成对数似然加性分解（context-free，公式 6）。
- **vs 极值统计因果模型 (Engelke 2025, Klüppelberg 2026)**：他们做渐近 / 重尾，本文做"非渐近 / 任意稀有"，特别覆盖"二值变量极度不平衡 / 接近零值"等非 tail 情形（呼应 Ebtekar 2025）。
- **vs Path-specific Effect / Mediation (Robins, Singal 2024)**：mediation 拆**平均**效应到路径，本文打分**单次观测**的事件解释力——目标不同，互补使用。
- **vs Causal abstraction (Rubenstein 2017, Beckers 2020)**：传统抽象作用在整模型，本文做 **event-level local abstraction**，accuracy 按目标稀有度归一化，更适合"为一次罕见事件构造小图解释"。
- **启发**：(i) 把"$\log$-likelihood gap" 当作 LLM 推理链的边权诊断指标；(ii) 把 natural micro-realization 用作"在 LLM 自然语言事件上做 do-演算"的语义锚定方式，可能与 counterfactual prompting 结合做可证伪的反事实推理。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 RCA + 因果抽象 + 极值统计 + LLM 自检统一到一个 log-likelihood 打分框架，是少见的"概念缝合且自洽"的工作。
- 实验充分度: ⭐⭐ 仅有概念性算例与单个 LLM demo，无真实数据集 / 基准对比，理论贡献为主。
- 写作质量: ⭐⭐⭐⭐ 定义 / 引理 / 例子节奏紧凑，附录完整；部分符号 $\mathcal{E}^K_{R\to t}$ vs $\mathcal{E}_{R\to t}$ 上下标繁，初读吃力。
- 价值: ⭐⭐⭐⭐ 对 GenAI 因果性自检 / 解释可信度评估方向给了一个清晰可落地的形式工具，预期会被 causal NLP / LLM safety 社区采用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2026\] Causal Modeling of Selection in Evolution](causal_modeling_of_selection_in_evolution.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)

</div>

<!-- RELATED:END -->
