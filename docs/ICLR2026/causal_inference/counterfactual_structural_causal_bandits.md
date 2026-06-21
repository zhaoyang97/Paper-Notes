---
title: >-
  [论文解读] Counterfactual Structural Causal Bandits
description: >-
  [ICLR2026][因果推理][结构因果模型] 本文把结构因果 bandit（SCB）从 Pearl 因果层级的 L1/L2（观测+干预）层提升到 L3（反事实）层，提出 CTF-SCB 框架，把"如果某下游变量当初听到的是 $x'$ 而非 $x$ 会怎样"这类可实现的反事实动作纳入臂空间，并用一套图论刻画（CTF-MIS / CTF-POMIS / 反事实 regime 图）把指数级的臂空间剪到只剩"可能最优"的代表性子集，配标准 bandit 求解器即可拿到更低的累积 regret。
tags:
  - "ICLR2026"
  - "因果推理"
  - "结构因果模型"
  - "反事实"
  - "Pearl 因果层级"
  - "多臂老虎机"
  - "动作空间剪枝"
---

# Counterfactual Structural Causal Bandits

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=gjvTNxVd2f](https://openreview.net/forum?id=gjvTNxVd2f)  
**代码**: 待确认  
**领域**: 因果推断 / 因果 bandit / 在线学习理论  
**关键词**: 结构因果模型, 反事实, Pearl 因果层级, 多臂老虎机, 动作空间剪枝

## 一句话总结
本文把结构因果 bandit（SCB）从 Pearl 因果层级的 L1/L2（观测+干预）层提升到 L3（反事实）层，提出 CTF-SCB 框架，把"如果某下游变量当初听到的是 $x'$ 而非 $x$ 会怎样"这类可实现的反事实动作纳入臂空间，并用一套图论刻画（CTF-MIS / CTF-POMIS / 反事实 regime 图）把指数级的臂空间剪到只剩"可能最优"的代表性子集，配标准 bandit 求解器即可拿到更低的累积 regret。

## 研究背景与动机

**领域现状**：把因果模型嫁接到多臂老虎机（MAB）上，是近年决策理论的一条主线。结构因果 bandit（SCB，Lee & Bareinboim 2018/2019）把每个臂看成对结构因果模型（SCM）中某个变量的一次干预 $do(X=x)$，于是"该拉哪个臂"变成"该干预哪个变量、设成什么值"。这条线已经证明：天真地把所有干预都当独立臂喂给标准 bandit 算法会白白产生巨大 regret，应该先用因果图把动作空间剪一遍再学。

**现有痛点**：但所有这些 SCB 工作都被关在 Pearl 因果层级（PCH）的 L1（观测 $P(Y\mid x)$）和 L2（干预 $P(Y\mid do(x))$）这两层里，把每个"物理上能做的干预"当成一个臂，**反事实（L3）量被整体排除在外**。原因是大家默认反事实"取不到样本"——一旦某个单元自然选了 $X=x'$，它在反事实干预 $do(x)$ 下的产出 $Y_x$ 就没法同时观测到。

**核心矛盾**：可是反事实其实**比 L2 更细粒度、信息更多**。一次干预 $do(X=x)$ 等价于让 $X$ 的所有孩子都收到同一个 $x$；但反事实允许让 $X$ 的不同孩子收到不同的"假想输入"——比如让医生评估收到病人真实主诉 $x$、却让 AI 工具收到一个重整过的报告 $x'$。这种"同一个原因变量在不同下游通道里取不同值"的世界，标准干预表达不了，却恰恰可能带来更高的回报。把这层信息白白扔掉，意味着 SCB 的最优臂可能根本不在它考虑的动作空间里。

**切入角度**：作者抓住一个关键事实——并非所有 L3 量都不可采样。Bareinboim 等人提出的**反事实随机化**（counterfactual randomization）以及 Raghavan & Bareinboim (2025) 对"可实现分布"的刻画表明，像 $P(Y_x, x')$ 这类带反事实中介的量是可以通过一套物理操作采到样本的。既然这些反事实动作**真能执行、真能拿到 reward**，就没有理由把它们排除在 bandit 的臂之外。

**核心 idea**：把 SCB 的动作空间从"L2 干预"扩张到"可实现的反事实 regime"，得到 CTF-SCB；再发展一整套图论工具，在不依赖参数假设的前提下，把这个指数膨胀的反事实臂空间剪到一个**保证不丢最优臂**的最小代表性子集，从而让标准 bandit 求解器直接享受更低 regret。

## 方法详解

### 整体框架

CTF-SCB 把环境建模为一个 SCM $M=\langle U,V,F,P(U)\rangle$，其中含一个 reward 变量 $Y$。智能体**知道因果图 $G$ 和 $Y$，但不知道机制 $F$ 与外生分布 $P(U)$**（结构已知、参数未知）。和传统 SCB 优化 $\mathbb{E}Y_x=\mathbb{E}[Y\mid do(X=x)]$ 不同，这里把臂推广成一个**反事实事件** $X_*=\{X_{1[w_1]},X_{2[w_2]},\dots\}$：每个 $X_i$ 不再是被固定成常数，而是"表现得像在子模型 $M_{w_i}$ 里那样"，再用这个值替换掉自然机制 $f_{X_i}$。每一轮 $t$，智能体拉一个臂 $X_*$，观测到对应的反事实 reward $Y_{X_*}$，目标是最小化累积 regret $R_T^{\mathcal{A}}=T\mu_{X_*^\star}-\sum_{t}\mu_{X_*^{(t)}}$。

直接在所有形如 $\{X_{i[w_i]}\}$ 的反事实事件上搜，组合数随节点数**超指数**爆炸，绝大多数还是冗余或可证次优的。所以全文的主线是一条"层层收缩"的剪枝管线（论文 Fig. 2）：

$$
\underbrace{\mathcal{A}}_{\text{全部可实现反事实臂}}\;\xrightarrow{\text{Sec. 3.1}}\;\underbrace{\text{CTF-MIS}}_{\text{去掉对 reward 无影响的冗余}}\;\xrightarrow{\text{Sec. 3.2}}\;\underbrace{\text{CTF-POMIS}}_{\text{去掉可证次优}}\;\xrightarrow{\text{Sec. 3.2.2}}\;\underbrace{\text{代表性 CTF-POMIS}}_{\mathcal{A}^\star}
$$

整条链的硬约束是 $\mathbb{E}R_T^{\mathcal{A}^\star}\le\mathbb{E}R_T^{\mathcal{A}}$：剪枝**只删不该探索的臂，绝不删掉可能最优的臂**。收缩到 $\mathcal{A}^\star$ 后，直接套用有有限时间 regret 保证的标准求解器（KL-UCB、Thompson Sampling）即可。下面四块设计依次对应：什么样的反事实臂算"合法且不冗余"（合法性 + CTF-MIS）、哪些臂"可能最优"（CTF-POMIS 的图论判据 + 反事实 regime 图）、以及怎么高效把它们全枚举出来（Algorithm 1）。

### 关键设计

**1. 反事实动作的合法性判据：用祖先冲突检测可实现性**

并不是任意写出来的反事实事件都能在系统里采到样本。痛点是：如果你想同时要求 $X_*=\{W_x, Z_{x'}\}$（$x\ne x'$），那么为了让 $Y$ 听到 $W_x$，机制 $f_Z$ 得收到 $x$；但为了让 $Y$ 同时听到 $Z_{x'}$，同一个 $f_Z$ 又得收到 $x'$——同一个机制被要求收两个互斥的输入，这个臂物理上不可实现。作者用**祖先关系上的冲突检测**给出可实现性的充要条件（Prop. 1）：定义反事实变量 $X_w$ 的祖先集 $\mathrm{An}(X_w)$，则 $X_*$ 是合法动作当且仅当 $\mathrm{An}(Y_x, X_*)$ 里**不存在同一个变量 $X$ 在两个不同 regime $w\ne t$ 下的副本** $X_w, X_t$。其理论依据是反事实去嵌套定理（CUT）：$P(y_{X_*})=\sum_x P(y_x, X_*=x)$，把反事实 reward 拆成可采样的形式，再用 Raghavan & Bareinboim (2025) 的可实现性刻画封口。这一步把"哪些反事实臂根本不存在"用纯图操作判掉，是后面所有剪枝的地基。

**2. CTF-MIS：剔除对 reward 无影响的冗余反事实变量**

合法的臂里仍有大量冗余——某个 $X_w\in X_*$ 可能对 reward 毫无贡献，即 $\mu_{X_*}=\mu_{X_*\setminus\{X_w\}}$。作者借助 CTF-calculus（L3 版的 do-calculus）和**干预最小化**（Lemma 1：$\lVert X_w\rVert\triangleq X_{w\cap\mathrm{An}(X)_{G_{\overline{W}}}}$，把订阅里用不上的部分砍掉）来识别这些等价关系。由此定义**反事实最小干预集 CTF-MIS**（Def. 3）：满足 $X_*=\lVert X_*\rVert$ 且不存在更小的 $Z_*\subsetneq X_*$ 使 $\mu_{X_*}=\mu_{Z_*}$ 对所有兼容 $G$ 的 SCM 都成立。它的图论充要刻画（Thm. 1）很直观：(i) 每个 $X_i$ 都有一条到 $Y$ 的有向因果路径（$X\subseteq\mathrm{An}(Y)_{G_{\overline{X}}}$）；(ii) 每个 $X_{i[w_i]}$ 的订阅 $W_i$ 要真正"管到" $X_i$ 的祖先（$W_i\cap\mathrm{An}(X_i)_{G_{\overline{X}\setminus\{X_i\}}}\ne\emptyset$）。值得注意的是论文给出的一个反直觉点（Remark 1）：在普通 SCB 里不同 MIS 之间没有等价关系，但在 CTF-SCB 里**不同 CTF-MIS 可以彼此等价**——例如 $\{W_x,Z_{x'}\}$ 和 $\{T_x,Z_{x'}\}$ 因为把 $x$、$x'$ 沿相同方式传播到 $Y$ 而等价，这为下一步"只留一个代表"埋了伏笔。

**3. CTF-POMIS + 反事实 regime 图：用干预边界判定"可能最优"**

CTF-MIS 之间还存在**偏序**：某些反事实干预无论模型参数怎么取，都至少不差于另一些。**反事实可能最优最小干预集 CTF-POMIS**（Def. 4）定义为：存在某个兼容 $G$ 的 SCM，使该臂严格优于所有不等价的 CTF-MIS。问题是怎么不靠枚举参数就判出它。作者的关键构造是**反事实 regime 图** $H_{X_*}=\langle V^\dagger, E^\dagger\rangle$，其中 $V^\dagger=V(\mathrm{An}(Y_x,X_*))$：它和普通因果图的区别在于**保留了反事实变量之间的真实因果关系**——基于一致性 $X_*=x\Rightarrow Y_x=Y_{X_*}$，把每个 $X_i$ 连向它在 $G$ 里的孩子（如 $W\to T$ 表示 $W_x\to T_{W_x}$），而这恰是 AMWN、counterfactual graph 等旧表示丢掉的信息。有了这张图，CTF-POMIS 得到一个干净的图论充要条件（Thm. 2）：$X_*$ 是 CTF-POMIS 当且仅当 $\mathrm{IB}(H_{X_*}, Y)=\emptyset$，其中 IB 是 Lee & Bareinboim (2018) 的**干预边界**（最小 UC-territory 之外、直接影响它的那圈节点）。一个重要推论（Prop. 2 / Cor. 2）：若 $Y$ 不与其祖先被未观测混杂耦合（如 Markovian 图），则唯一的 CTF-POMIS 就是 $\mathrm{Pa}(Y)_G$——也就是说**在无混杂场景下根本不需要反事实动作，直接干预 $Y$ 的父节点就够了**；反事实的价值只在 $Y$ 被混杂时才真正显现。

**4. Algorithm 1：边选择枚举所有代表性 CTF-POMIS**

逐个验证每个反事实事件是不是 CTF-MIS、再是不是 CTF-POMIS，复杂度随节点超指数增长，不可行。作者的破解办法是只盯形如 $\{X_{i[\mathrm{Pa}(X_i)_G]}\}$ 的反事实——每个这样的"代表"能代表几十个等价的 CTF-(PO)MIS，且它们合起来恰好覆盖 $\mathcal{A}^\star$（Prop. 3）。算法（Algo. 1，复杂度 $O(n^2\cdot 2^{|E|})$）走一条**边选择**路线：对每种"孩子选择"组合，从图里删掉对应的边得到子图 $H'$，若 $\mathrm{IB}(H',Y)=\emptyset$ 就据此构造一个反事实事件 $X_*$，再调用子程序 `CTF-MISIFY`（按 Thm. 1 的两条件把不满足的 $X_{j[w_j]}$ 删掉）规整成合法 CTF-MIS，最后用 Prop. 1 检验无冲突后输出。Thm. 3 保证它**不重不漏**地返回全部代表性 CTF-POMIS。拿到这个 $\mathcal{A}^\star$ 后，标准 bandit 求解器在其上享有 $O\!\big(\sum_{X_*\in\mathcal{A}^\star:\Delta_{X_*}>0}\frac{\log T}{\Delta_{X_*}}\big)$ 的有限时间 regret 界。

### 损失函数 / 训练策略
这是理论 + 仿真工作，没有可训练参数。在线学习目标即最小化累积 regret（式 1），由 KL-UCB 与 Thompson Sampling 两类标准求解器在剪枝后的动作空间 $\mathcal{A}^\star$ 上实现；regret 上界 $O(\sum_{X_*\in\mathcal{A}^\star:\Delta_{X_*}>0}\frac{\log T}{\Delta_{X_*}})$ 表明把臂数压小就能直接降低 regret 常数。

## 实验关键数据

实验用累积 regret（CR）对比三种动作空间选择策略——**CTF-POMIS**（本文剪枝后）、**CTF-MIS**（只去冗余、未去次优）、**POMIS**（停在 L≤2 的旧 SCB）——每种各配 Thompson Sampling（TS）和 KL-UCB 两个求解器。前两个任务跑 10k 轮、第三个跑 100k 轮，每组重复 1,000 次取均值与标准差。

### 主实验（三个因果图任务的累积 regret）

| 任务 | 求解器 | CTF-POMIS 的 CR | 占 CTF-MIS 的比例 | 结论 |
|------|--------|-----------------|-------------------|------|
| Task 1（Fig. 5a） | TS | 66.69 | ≈38.92% | CTF-POMIS 最低 |
| Task 1 | KL-UCB | 148.19 | ≈38.96% | CTF-POMIS 最低 |
| Task 2（Fig. 3b） | TS | 387.01 | 57.46% | CTF-(PO)MIS 持续优于 POMIS |
| Task 2 | KL-UCB | 831.24 | 70.17% | 同上 |
| Task 3（Fig. 9） | TS / KL-UCB | 见 Appendix 表 1–2 | — | 最优臂不在 POMIS 时 POMIS 吃线性 regret |

### 关键发现
- **CTF-POMIS 把 regret 砍到 CTF-MIS 的约 39%（Task 1）**：去掉"可证次优"的反事实臂，比只去冗余（CTF-MIS）再省一大截探索成本，说明 POMIS 这层剪枝是 regret 下降的主力。
- **当最优臂落在 L≤2 之外时，旧 POMIS 吃线性 regret**：Task 3 的左图里存在严格正的间隙 $\Delta_{L\le2}\triangleq\mu_{X_*^\star}-\mu_{x^\star}>0$，POMIS 因为根本够不到反事实最优臂，regret 随轮次线性增长——这是"忽略 L3 会系统性次优"最直接的证据。
- **当最优臂恰好落在 L≤2 时，POMIS 反而收敛更快**：Task 3 右图里 $\Delta_{L\le2}=0$，此时 POMIS 臂数更少、探索更省，会暂时领先；但智能体在交互前无法预知最优臂在哪层，所以这并不削弱"默认要考虑反事实"的结论——POMIS 早期的小 CR 是用"赌对了层"换来的。

## 亮点与洞察
- **把 Pearl 因果层级的 L3 真正"接进"了序贯决策**：以往 causal bandit 只敢用 L1/L2，本文用可实现性理论说明一类反事实动作其实能采样、能执行，从而把臂空间扩到更细粒度的世界——这是概念上的破壁。
- **反事实 regime 图是点睛之笔**：它保留了"$X_i$ 连向它真实孩子"的因果边（$W_x\to T_{W_x}$），正是这条信息让干预边界 $\mathrm{IB}=\emptyset$ 这个**已有的 L2 工具**能原封不动迁移来刻画 L3 的 CTF-POMIS——用旧锤子敲新钉子，优雅且省事。
- **"Markovian 下不需要反事实"是个有用的负结论**：Prop. 2/Cor. 2 明确告诉实践者，只有当 reward 被未观测混杂污染时反事实动作才有增益，否则干预 $\mathrm{Pa}(Y)$ 即最优——避免在用不上的场景里白搭复杂度。
- **代表性 CTF-POMIS + 边选择枚举**把超指数搜索压成 $O(n^2\cdot 2^{|E|})$ 的可算流程，且不重不漏，使整套理论真的能落到一个可运行的剪枝算法上。

## 局限与展望
- **SCM 建模假设**：要求变量集明确、因果结构固定，难以刻画动态、高维或部分可观测的系统。
- **反事实动作的可执行性**：框架假设智能体能执行任意可实现反事实 $P(Y_{X_*})$，这隐含需要识别合适的"反事实中介"；现实中受伦理、安全、技术约束，未必构造得出。
- **已知因果图假设**：默认智能体拿到真实因果图。作者指出可用因果发现缓解，且**保守地多假设一条混杂边（super-model）不会破坏 soundness**——例如把 $G$ 扩成 $G'=\langle\{X,Y\},\{X\to Y, X\leftrightarrow Y\}\rangle$ 仍能覆盖真实 CTF-POMIS，只是推断更不精确；反方向（漏边）则会破坏正确性，这是一个**保守安全、漏边危险的不对称性**。
- 可改进方向（笔者）：把因果发现的不确定性显式纳入 regret 分析、或在图部分错指定下给出鲁棒性保证，会让框架更贴近实战。

## 相关工作与启发
- **vs 传统 SCB（Lee & Bareinboim 2018/2019）**：他们在 L≤2 上定义 MIS/POMIS，把每个物理干预当臂；本文把同一套"最小干预集 / 可能最优集"的思想抬到 L3 反事实层，得到 CTF-MIS/CTF-POMIS，并指出 SCB 里"不同 MIS 互不等价"在 CTF 里失效，需要新的代表性枚举。
- **vs 可实现性刻画（Raghavan & Bareinboim 2025；Bareinboim et al. 2015；Forney et al. 2017）**：那条线回答"哪些反事实分布能采样"，是本文 Prop. 1 合法性判据的理论支柱；本文则进一步回答"这些可采样的反事实里，哪些值得当 bandit 臂"。
- **vs 标准 MAB 求解器（KL-UCB / Thompson Sampling）**：本文不改求解器本身，而是改它们作用的动作空间——先用因果图把臂剪到 $\mathcal{A}^\star$，再让现成的有限时间 regret 保证直接生效，体现"因果剪枝 + 通用学习器"的解耦思路。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 Pearl 层级 L3 反事实纳入结构因果 bandit 的臂空间，并给出完整图论刻画。
- 实验充分度: ⭐⭐⭐⭐ 三组因果图仿真充分验证 regret 优势，但均为合成 SCM，无真实系统验证。
- 写作质量: ⭐⭐⭐⭐ 理论链条（合法性→MIS→POMIS→算法）层层递进且自洽，但符号密度高、对非因果背景读者门槛偏陡。
- 价值: ⭐⭐⭐⭐⭐ 在因果推断、决策理论与在线学习交叉处开了一条新方向，理论基础扎实。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Self-Supervised Learning from Structural Invariance](self-supervised_learning_from_structural_invariance.md)
- [\[NeurIPS 2025\] Root Cause Analysis of Outliers with Missing Structural Knowledge](../../NeurIPS2025/causal_inference/root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[ICLR 2026\] Multiverse Mechanica: A Testbed for Learning Game Mechanics via Counterfactual Worlds](multiverse_mechanica_a_testbed_for_learning_game_mechanics_via_counterfactual_wo.md)

</div>

<!-- RELATED:END -->
