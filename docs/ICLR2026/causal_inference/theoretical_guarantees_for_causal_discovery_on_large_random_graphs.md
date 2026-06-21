---
title: >-
  [论文解读] Theoretical Guarantees for Causal Discovery on Large Random Graphs
description: >-
  [ICLR 2026][因果推理][因果发现] 这篇论文给"用随机单变量干预做因果定向"这件事，第一次推出了**有限维度的偏差集中界**（不是渐近一致性、也不是最坏情况界）：在稀疏 Erdős–Rényi 与广义 Barabási–Albert 随机图上，定向错误的假阴率（FNR）会随维度 $d$ 增大而**越来越集中、甚至消失**，从而证明"高维 + 度数重尾异质"这两个常被当作障碍的性质，反而能内在地正则化因果发现。
tags:
  - "ICLR 2026"
  - "因果推理"
  - "因果发现"
  - "假阴率(FNR)"
  - "偏差集中不等式"
  - "随机图模型"
  - "干预实验设计"
---

# Theoretical Guarantees for Causal Discovery on Large Random Graphs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=V7pT2ZRoTB](https://openreview.net/forum?id=V7pT2ZRoTB)  
**代码**: 随论文附补充材料（DiffIntersort，无独立公开 repo 链接）  
**领域**: 因果推断 / 因果发现理论  
**关键词**: 因果发现, 假阴率(FNR), 偏差集中不等式, 随机图模型, 干预实验设计

## 一句话总结
这篇论文给"用随机单变量干预做因果定向"这件事，第一次推出了**有限维度的偏差集中界**（不是渐近一致性、也不是最坏情况界）：在稀疏 Erdős–Rényi 与广义 Barabási–Albert 随机图上，定向错误的假阴率（FNR）会随维度 $d$ 增大而**越来越集中、甚至消失**，从而证明"高维 + 度数重尾异质"这两个常被当作障碍的性质，反而能内在地正则化因果发现。

## 研究背景与动机

**领域现状**：因果发现的目标是从数据里恢复变量间的有向无环图（DAG）。在系统生物学（基因调控网络重建）、神经科学（脑功能连接）等场景里，这类问题动辄涉及成百上千个变量，是天然的高维问题，而且通常借助单变量干预（敲除/扰动某个基因）来获得定向信息。

**现有痛点**：已有的理论分析几乎都是**最坏情况（worst-case）**视角——关心"恢复整张 DAG 最少需要多少次实验"（如 $O(\log d)$ 组多变量干预、$d-1$ 次原子干预），并且依赖一系列理想化假设：完整 faithfulness、因果充分性（无隐变量）、对抗式的最优干预设计。这些界过于保守，既不反映真实网络的结构特征（稀疏性、度数异质），也**完全不刻画随机实例之间的性能波动**——它只告诉你"最差能多差"，不告诉你"典型情况下误差围绕均值抖动多大"。

**核心矛盾**：实践者真正想知道的不是最坏界，而是"我在中等规模系统上测到的误差波动，放大到几千个变量时会怎么变？"。这需要的是**围绕期望的偏差（deviation）界**，而不是期望本身的上界，更不是最坏情况。但此前没有任何工作给出因果定向误差的有限维度偏差不等式。

**本文目标**：在尽量弱的假设下（允许隐变量混淆、随机而非最优的干预选择），刻画**假阴率 FNR**（被错误定向的真实边占总边数的比例）如何围绕其期望集中，并把这个分析覆盖到真实网络常见的随机图族（ER 与 BA）。

**切入角度**：作者借用 Chevalley et al. (2025c) 提出的 InterSort 评分框架与 $\epsilon$-干预 faithfulness 假设，把误差写成"随机干预向量 / 随机图"的函数；关键观察是——**翻转单个干预（或单条边）只会影响一小撮"邻域内"的边**，于是误差函数对每个输入坐标的敏感度（Lipschitz 常数）被图的局部结构卡住。一旦敏感度有界，McDiarmid / 有界差分不等式就能直接给出集中。

**核心 idea**：把"图的结构性质（度数分布、稀疏度）"通过 Lipschitz 敏感度翻译成"误差的概率集中速率"，从而得到**维度自适应、对 faithfulness 鲁棒**的偏差界——并发现重尾度数结构反而让误差更集中。

## 方法详解

### 整体框架

这是一篇纯理论论文，"方法"就是一条**从图结构到概率集中的证明流水线**，分四步走：

1. **定义误差度量**：用拓扑序偏差 $D_{top}(G,\pi)=\sum_{\pi(i)>\pi(j)} A^G_{ij}$ 数出一个候选因果序 $\pi$ 下被错误定向的边数；InterSort 评分的最优序 $\pi_{opt}$ 是其最大化者。记 $f(I)=D_{top}(G,\pi_{opt}(I))$ 为误定向计数，$g(I)=f(I)/|E|$ 为归一化的假阴率 FNR。
2. **建立 Lipschitz 敏感度**：证明翻转任一干预指示 $I_k$ 或边指示 $E_{ij}$，只会改变与该节点/边相邻的一小撮边（其祖先锥与后代锥，受限情形下只剩直接父子），于是逐坐标差分常数被节点度数卡住。
3. **套用集中不等式**：有界差分 → McDiarmid（密集 ER、BA）；稀疏 ER 因均值不消失需用 Warnke 的"典型有界差分"配合度数上界 $\deg_{\max}=O(\log d)$。
4. **分图族算速率**：在三种随机图（固定图、ER 密集/稀疏、广义 BA）下分别代入，得到 $g$ 围绕均值的典型偏差宽度，并读出"何时消失、何时常数、何时增长"。

输入是"图分布 + 干预概率 $p_{int}$"，输出是"FNR 的均值阶 + 偏差集中速率"。三者（度量定义 ↔ Lipschitz 敏感度 ↔ 各图族速率）一脉相承：敏感度越小（度数越受控），集中越紧。

### 关键设计

**1. $\epsilon$-干预 faithfulness + FNR 度量：把"可定向性"建立在弱假设上**

经典 faithfulness 要求图中每一处 d-分离都对应联合分布里的条件独立，这在有隐变量混淆时极易破坏。本文沿用 $\epsilon$-干预 faithfulness（假设 1）：对任意 $i\in I,\,j\in V$，
$$D\!\left(P^{C,(\emptyset)}_{X_j},\,P^{C,do(X_i:=\tilde N_i)}_{X_j}\right) > \epsilon \iff \text{存在有向路径 } i\rightsquigarrow j.$$
它只要求"干预节点 $i$ 对下游 $j$ 造成至少 $\epsilon$ 的边际分布漂移"，**不涉及条件独立结构**，因此在边缘化下保持、与隐变量混淆兼容。在此基础上定义评分
$$S(\pi,\epsilon,D,I,\cdots)=\sum_{i\in I,\,j\in V,\,\pi(i)<\pi(j)}\big[(D_{ij}-\epsilon)_+ + c\cdot d\cdot \mathbb{1}\{D_{ij}>\epsilon\}\big],$$
其最大化者 $\pi_{opt}$ 给出可达到的定向性能。作者强调：偏差界只依赖图结构、$p_{int}$ 与可达关系，**不依赖 InterSort 评分的具体形式**——任何用单节点干预、利用同一 $\epsilon$-faithfulness 的算法都受同样的界约束。相比前人只分析误定向计数 $f$，本文换用归一化的 FNR $g=f/|E|$，让结论可跨不同规模的图直接比较。

**2. 基于祖先/后代锥的 Lipschitz 敏感度：把图结构卡进有界差分常数**

集中不等式的前提是"改一个输入坐标，函数只动一点点"。论文用一组引理把这个"一点点"算清楚。引理 3：翻转 $I_k$ 只影响"以 $k$ 为证据"的边，其变化量
$$c_k \le |\mathrm{Anc}(k)| + |\mathrm{Desc}(k)|,$$
即落在 $k$ 的祖先锥与后代锥内的边；受限（parents-only）情形下（引理 4）进一步收紧到 $c_k\le \deg_{in}(k)+\deg_{out}(k)$，只剩直接相邻边。边变量同理（引理 5），翻转 $E_{ij}$ 的影响集 $A_{ij}=\{(k,j)\in E: i\notin \mathrm{Pa}(k)\}$，常数 $c_{ij}\le \deg_{in}(j)$。这里有个**必须钉死的前提**（假设 2 + 备注 6）：$\pi_{opt}$ 要靠确定性 tie-break 唯一化，否则评分出现并列最大值时 $\pi_{opt}$ 会跳变，使 $c_k$ 大到 $|E|$，集中彻底失效。正是这组"度数即敏感度"的翻译，搭起了从图论性质到概率集中的桥。

**3. 三类随机图上的集中速率：高维与重尾如何"反而"正则化误差**

把上面的 Lipschitz 常数代入集中不等式，按图族分别求速率，得到本文最反直觉的结论。**固定图 + 随机干预**：McDiarmid 直接给出
$$P(|g(I)-\mathbb{E}[g]|\ge t)\le 2\exp\!\Big(-\tfrac{2t^2|E|^2}{\sum_k c_k^2}\Big).$$
**Erdős–Rényi**：密集（$p_e=\Theta(1)$）时 $\mathbb{E}[g]=\Theta(1/d)$，偏差宽度 $O(d^{-1/2})$——均值与波动同时消失；稀疏（$p_e=c/d$）时均值**不消失**（$O(1)$，引理 22），但靠 Warnke 典型有界差分（排除 $\deg_{\max}=O(\log d)$ 的罕见高度数事件）仍得到 $g$ 集中于 $O\!\big(\tfrac{\log d}{\sqrt d}\big)$，只比密集情形差一个对数因子。**广义 Barabási–Albert**：度数指数 $\gamma=2+\kappa/m$，$\beta=1/(\gamma-1)$，引理 10 给出 $\deg(i)\le m+Cd^{\beta}$ w.h.p.，于是
$$P(|g(I)-\mathbb{E}[g\,|\,E]|\ge t)\le 2\exp\!\Big(-\tfrac{2t^2}{d^{\,2\beta-1}}\Big),$$
偏差宽度 $O(d^{\beta-1/2})$：当 $\gamma>3$（即 $\beta<1/2$）时它随 $d\to\infty$ **消失**，证明重尾、hub 主导的异质拓扑会通过抑制误差方差来正则化因果定向；而 $\kappa=1$（$\gamma=7/3<3$）落入重尾失控区，偏差反而以 $O(d^{1/4})$ 增长。

### 损失函数 / 训练策略
本文不训练模型，无损失函数。理论结论汇总见下表（$f$ 为误定向计数，$g=f/|E|$ 为 FNR，$\beta=\tfrac{1}{\gamma-1}$，$\gamma=2+\kappa/m$）：

| 图模型 | $\mathbb{E}[f]$ | $\mathbb{E}[g]$ | $f$ 偏差宽度 | $g$ 偏差宽度 |
|--------|-----------------|-----------------|--------------|--------------|
| ER 密集 ($p_e=\Theta(1)$) | $O\!\big(\tfrac{(1-p_{int})^2}{p_{int}}d\big)$ | $O\!\big(\tfrac{2(1-p_{int})^2}{p_{int}d}\big)$ | $O(d^2)$ | $O(d^{-1/2})$ |
| ER 稀疏 ($p_e=c/d$) | $O\!\big(\tfrac{(1-p_{int})^2}{p_{int}}d\big)$ | $O\!\big(\tfrac{2(1-p_{int})^2}{c\,p_{int}}\big)$ | $O(\sqrt d\log d)$ | $O(d^{-1/2}\log d)$ |
| 广义 BA ($\beta\in(0,1)$) | $(1-p_{int})^2 m d$ | $(1-p_{int})^2$ | $O(d^{\beta+1/2})$ | $O(d^{\beta-1/2})$ |

## 实验关键数据

作者明确声明：实验**只用于佐证"偏差集中"趋势，不是对有限样本界的直接检验**，因为 InterSort 评分的精确最优解无多项式算法，仿真只能用可微松弛 **DiffIntersort**（基于 Sinkhorn 优化，可扩展到 2000 变量）近似，会引入偏差。

### 主实验
仿真在三类随机 DAG 上进行，图规模 $d$ 从 30 到 2000，每个配置重复 10 次，报告归一化拓扑误差 $g(I,E)$ 经验均值的典型偏差宽度（用四分位距 IQR 度量）。

| 图族 | 密度/异质参数 | 干预覆盖 $p_{int}$ | 观测到的 IQR 随 $d$ 变化 |
|------|---------------|--------------------|--------------------------|
| Erdős–Rényi | $p_e\in\{0.2,0.4,0.6\}$ | $\{0.25,0.5,0.75\}$ | 随 $d$ 增大而消失，最清晰地确认理论 |
| 稀疏/scale-free ER | 平均度 $c\in\{2,3,5\}$ | $\{0.25,0.5,0.75\}$ | 随 $d$ 增大而消失 |
| scale-free BA | $\kappa\in\{1.0,3.0,9.0\}$，$m=3$ | $\{0.25,0.5,0.75\}$ | $\kappa=9,3$ 收缩；$\kappa=1$ 失稳、波动更大 |

### 关键发现
- **BA 图最能区分理论的细腻预测**：$\kappa=9.0$（$\gamma$ 大、$\beta<1/2$）偏差应消失，实验确实收缩；$\kappa=3.0$ 理论上应趋于常数，实验也基本稳定收缩；$\kappa=1.0$ 对应 $\gamma=7/3<3$ 的重尾失控区，理论预测偏差以 $O(d^{1/4})$ 增长，实验里它确实更不稳定、抖动更强——与该 regime 较弱的理论控制一致。
- **稀疏 ER 验证最干净**：偏差随维度单调收窄，印证"高维让 FNR 更集中"这一核心主张。
- **偏差界不依赖 $p_{int}$**：这是 McDiarmid 类不等式的固有特性（论文坦承为局限），实验中不同 $p_{int}$ 的曲线形状一致、主要影响均值而非集中速率。

## 亮点与洞察
- **"高维 + 重尾 = 障碍"这一直觉被推翻**：论文证明 hub 主导的异质度数结构会抑制定向误差的方差，让大规模因果发现**更稳**而非更难——这对系统生物学/神经科学里动辄上千变量的网络是个令人安心的结论。
- **首个因果定向误差的有限维度偏差不等式**：此前文献要么给最坏情况界、要么给渐近一致性，本文填了"典型实例波动有多大"这一空白，且结论以"偏差宽度的阶"形式给出，可直接指导随机干预实验的预算-误差权衡。
- **可迁移的证明范式**：把"误差函数对单坐标的敏感度 = 图的局部度数"这一翻译，配上有界差分/Warnke 不等式，是一套可复用到其他图度量（如未定向边数、I-MEC 大小）的模板。
- **结论与具体算法解耦**：偏差界只依赖图结构与 $\epsilon$-faithfulness，任何用单节点干预的方法都受其约束，因此它既是"乐观假设下的性能上界参照"，也是"合成 benchmark 应预期多大波动"的标尺。

## 局限与展望
- **作者承认的局限**：$\epsilon$-干预 faithfulness 仍要求干预造成可测分布漂移，在高噪声/弱效应系统可能失效；随机图模型是风格化抽象，不含模块化、层级等领域结构；假设可获得足够多次干预（在高通量扰动筛/大规模神经刺激中渐成常态，但不总现实）；偏差界不随 $p_{int}$ 变化；假设系统无环，排除了反馈回路。
- **实验侧局限**：仿真用 DiffIntersort 作代理搜索，而理论针对评分的精确最优解，二者之间有近似 gap，因此实验只能定性佐证趋势、不能定量验证常数。
- **自己发现的局限**：三种图族的偏差宽度结论横向比较时要小心——它们对应不同的均值阶（密集 ER 均值消失、稀疏 ER 均值常数、BA 均值常数），"偏差更小"不等于"误差更小"，需结合均值一起读。
- **改进思路**：让偏差界刻画对 $p_{int}$ 的依赖（需超越 McDiarmid 的证明工具）；扩展到自适应干预策略、其他误差度量与更一般图分布；以及为大规模设定开发更接近精确最优的实用搜索算法。

## 相关工作与启发
- **vs Katz et al. (2019)**：同样在 ER 图上分析干预 Markov 等价类的期望规模，但他们依赖因果充分性 + 完整 faithfulness + 最优干预选择，只给均值的极限与上界；本文放宽到 $\epsilon$-faithfulness + 随机干预，并首次给**偏差界**而非均值。
- **vs Chevalley et al. (2025c)**：本文的直接前作，提供了 InterSort 评分与误定向计数 $D_{top}$ 的期望上界；本文换用归一化 FNR、补上偏差集中、并把全部结论扩展到广义 BA 图。
- **vs 最坏情况干预复杂度（Eberhardt 2005/2006、Shanmugam 2015、Kocaoglu 2017）**：他们关心"恢复整图最少需多少干预"的对抗式下界；本文是**平均情况 + 有限维度集中**视角，回答"典型随机实例下误差波动多大、如何随维度衰减"，二者互补。
- **启发**：把"算法的统计难度"直接绑定到"图的度数分布"，提示在做大规模干预实验设计时，可先在中等规模系统估出 FNR 波动，再借集中速率外推到目标规模，从而以更少实验达到目标可靠度。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个因果定向误差的有限维度偏差集中界，且揭示重尾结构正则化效应，视角新颖
- 实验充分度: ⭐⭐⭐ 仿真覆盖三图族、规模到 2000 变量，但只能定性佐证趋势、不直接检验界，且受 DiffIntersort 近似限制
- 写作质量: ⭐⭐⭐⭐ 假设、引理到定理的逻辑链清晰，结论汇总表很有帮助
- 价值: ⭐⭐⭐⭐ 为大规模干预实验设计提供了原理性的预算-误差权衡参照，对系统生物学等高维场景有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ICLR 2026\] Learning Dynamic Causal Graphs Under Parametric Uncertainty via Polynomial Chaos Expansions](learning_dynamic_causal_graphs_under_parametric_uncertainty_via_polynomial_chaos.md)
- [\[ICLR 2026\] Causal Discovery in the Wild: A Voting-Theoretic Ensemble Approach](causal_discovery_in_the_wild_a_voting-theoretic_ensemble_approach.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)

</div>

<!-- RELATED:END -->
