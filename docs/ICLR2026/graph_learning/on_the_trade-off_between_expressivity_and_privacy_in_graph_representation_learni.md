---
title: >-
  [论文解读] On the Trade-off Between Expressivity and Privacy in Graph Representation Learning
description: >-
  [ICLR 2026][图学习][同态密度] 本文首次在理论上刻画「图表示的表达力」与「边级差分隐私」之间的根本张力，提出用**带噪同态密度向量**作为图嵌入：它在期望意义下保持完整的判别能力，同时按每个密度的平滑敏感度注入校准噪声以满足形式化的差分隐私保证，并证明越强表达力的模式类需要越多噪声这一显式权衡。
tags:
  - "ICLR 2026"
  - "图学习"
  - "同态密度"
  - "差分隐私"
  - "图表达力"
  - "平滑敏感度"
  - "隐私-表达力权衡"
---

# On the Trade-off Between Expressivity and Privacy in Graph Representation Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=XXLDvwMwbe](https://openreview.net/forum?id=XXLDvwMwbe)  
**代码**: [https://github.com/tamaramagdr/private-homcounts](https://github.com/tamaramagdr/private-homcounts)  
**领域**: 图表示学习 / 差分隐私 / 表达力理论  
**关键词**: 同态密度, 差分隐私, 图表达力, 平滑敏感度, 隐私-表达力权衡

## 一句话总结
本文首次在理论上刻画「图表示的表达力」与「边级差分隐私」之间的根本张力，提出用**带噪同态密度向量**作为图嵌入：它在期望意义下保持完整的判别能力，同时按每个密度的平滑敏感度注入校准噪声以满足形式化的差分隐私保证，并证明越强表达力的模式类需要越多噪声这一显式权衡。

## 研究背景与动机
**领域现状**：图表示学习里有两条几乎正交的研究线。一条是**表达力分析**，关心一个算法能不能区分两个非同构的图——主流用 $k$-WL 层级、或用同态计数（homomorphism counts）来刻画 GNN 的判别上界。另一条是**图隐私**，关心算法是否泄露敏感的结构信息（哪条边存在），主流做法是给输出加噪声满足差分隐私（DP）。

**现有痛点**：这两个目标天然冲突。考虑两个只差一条边的非同构图 $G_1, G_2$：一个有表达力的算法**必须**给出不同的嵌入 $\varphi(G_1)\ne\varphi(G_2)$ 才能捕捉这条边带来的结构差异；而一个边级差分隐私的算法**必须**让这两个嵌入难以区分 $\varphi(G_1)\approx\varphi(G_2)$，否则攻击者就能确认这条边的有无。已有的差分隐私图学习方法只追求隐私约束下的最优 utility，**完全不提供表达力保证**；而表达力分析的工作又从不考虑隐私。

**核心矛盾**：表达力要求「邻接图被区分开」，隐私要求「邻接图被混在一起」——两者在邻接图（差一条边）这个最小扰动尺度上直接对立。此前几乎没有工作正式研究三者（隐私、表达力、utility）之间的权衡。

**本文目标**：在图级任务上、给出边级隐私保证的前提下，研究「同时具备可证表达力与可证隐私」的嵌入到底能做到什么程度，并把这个权衡量化出来。

**切入角度**：作者抓住一个朴素但关键的观察——**DP 噪声均值为零**。如果嵌入本身是某种「期望意义下完整」的表示，那么加零均值噪声不会破坏它的期望，从而可以「在期望上保留表达力」。同态计数恰好是这样一个工具：Lovász 的经典结果表明同态计数能区分任意一对非同构图，且归一化后的同态密度可以构造期望完整（expectation-complete）的嵌入。

**核心 idea**：用「带噪同态密度向量」当图嵌入——按每个密度对单边扰动的敏感度校准高斯噪声，让差一条边的图分布充分重叠（满足 DP），同时因为噪声零均值，嵌入在期望上仍等于无噪密度（保留表达力）。

## 方法详解

### 整体框架
方法是一个「采样模式 → 算同态密度 → 按平滑敏感度校准噪声 → 发布私有嵌入」的一次性预处理流水线。输入是一个边级敏感的图 $G$，输出是一个满足 tCDP 的实值向量嵌入 $\tilde t(\mathcal F, G)$，之后可以喂给**任意**下游学习器（k-NN、SVM、随机森林等）做图分类或回归，且由于 DP 的后处理性质，下游不再产生额外隐私开销。

具体地：先从一个对目标图类 $\mathcal F$ 有全支撑（full support）的分布 $\mathcal D$ 中采样一组模式 $\mathbf F=(F_1,\dots,F_d)$，模式类的选择决定了嵌入能达到的表达力等级；对每个模式算归一化同态密度 $t(F_i,G)$ 得到密度向量；再用「平滑敏感度」上界算出每个图局部所需的噪声尺度，注入校准好的零均值高斯噪声得到私有嵌入；最后把节点数 $|V(G)|$ 拼接到嵌入上，解决同态密度无法区分图与其 blowup 的退化情形。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：边级敏感图 G"] --> B["从分布 D 采样模式向量<br/>F = (F1...Fd)，模式类决定表达力"]
    B --> C["带噪同态密度嵌入<br/>算 t(F,G) 再加零均值噪声<br/>期望上保留判别力"]
    C --> D["平滑敏感度校准<br/>按局部敏感度定噪声尺度<br/>给出 tCDP 保证"]
    D -->|拼接节点数 V(G) 破 blowup 退化| E["私有嵌入 t̃(F,G)"]
    E -->|DP 后处理性质，下游零额外开销| F["下游：k-NN / SVM / RF<br/>图分类或回归"]
    B -.->|模式类越强表达力越高<br/>需注入更多噪声| D
```

### 关键设计

**1. 带噪同态密度嵌入：用零均值噪声换隐私，又靠期望守住表达力**

针对「表达力与隐私在邻接图尺度上直接对立」这个核心矛盾，作者的解法是把判别力从「逐次实现」搬到「期望」上。同态密度定义为归一化的同态计数 $t(F,G)=\dfrac{\hom(F,G)}{|V(G)|^{|V(F)|}}$，带噪嵌入就是 $\tilde t(F,G)=t(F,G)+N$，其中 $N$ 是任意零均值噪声（DP 噪声满足此条件）。关键在于：虽然加噪后嵌入不再是置换不变的（DP 本就要求随机化机制），但因为 $\mathbb E[N]=0$，有 $\mathbb E[\tilde t(F,G)]=t(F,G)$，于是嵌入在期望上等于无噪密度。

作者证明（Theorem 4.1）只要模式 $F\sim\mathcal D$ 且 $\mathcal D$ 对 $\mathcal F$ 有全支撑，单个带噪密度就是 $\mathcal F$-期望表达的（$\mathcal F$-expectation-expressive）；当 $\mathcal F=\mathcal G$（全体图）时甚至是期望完整的。进一步（Theorem 4.2）当采样模式数 $d$ 足够大时，带噪密度向量以至少 $1-\theta$ 的概率是真正 $\mathcal F$-表达的——也就是说，加噪不仅在期望上、而且高概率地保住了完整判别力。一个技术细节是同态密度天生无法区分图 $G$ 和它的 $p$-blowup（每个节点复制 $p$ 份），作者把节点数 $|V(G)|$ 直接拼到嵌入上即可破除这种退化，而由于邻接图节点数相同，这一步「零隐私成本」。

**2. 平滑敏感度校准：用局部敏感度而非全局敏感度，把噪声压到可用**

针对「直接按最坏情况加噪会让嵌入噪声大到不可用」的实际问题，作者用平滑敏感度框架精细化噪声尺度。先由 counting lemma 给出全局敏感度上界：对差一条边的邻接图 $G\sim_1 G'$，$GS_{t,F}=|t(F,G)-t(F,G')|\le e(F)\,d_\square(G,G')=2e(F)/n^2$。但全局敏感度刻画的是任意图周围的最坏行为，按它加噪性能很差。局部敏感度 $LS_{t,F}(G)=\max_{G':d_{\text{edge}}(G,G')\le 1}|t(F,G)-t(F,G')|$ 通常小得多，但直接按它加噪不保证 DP。于是作者采用 Nissim 等人的**平滑敏感度**——给局部敏感度一个平滑上界：

$$S_{t,F}(G)=\max_{G'\in\mathcal G}\Big(e^{-\beta\,d_{\text{edge}}(G,G')}\cdot LS_{t,F}(G')\Big).$$

对密度向量，整体平滑敏感度可由各分量的 $\ell_2$ 范数 $S_{t,*}(G)=\|S_{t,F_1}(G),\dots,S_{t,F_d}(G)\|_2$ 上界（Proposition 4.4）。当图的最大度 $\Delta_{\max}$ 有界时（很多领域有这种先验），作者还给出更紧的敏感度界（Theorem 4.5）：对含 $m>1$ 个节点的模式 $F$，

$$|t(F,G)-t(F,G')|\le\frac{2e(F)}{n^2}\Big(\frac{\Delta_{\max}}{n}\Big)^{m-2}.$$

由于大图大模式下 $(\Delta_{\max}/n)^{m-2}\ll 1$，这个界往往比 counting lemma 紧得多；若没有可用的公开度界，可私有估计 $\Delta_{\max}$ 或令 $\Delta_{\max}=n$ 退回 counting lemma。最终主结果（Theorem 4.6）给出私有机制：加高斯噪声 $\tilde t(\mathbf F,G)=t(\mathbf F,G)+N\!\big(0,\frac{[S_{t,*}(G)]^2}{2\rho'}I_d\big)$，它满足 $\big(2\rho'+d\cdot 4\beta^2,\ \frac{1}{4\beta}\big)$-tCDP。因为平滑敏感度被全局敏感度上界，这个机制的隐私-utility 权衡显著优于标准高斯机制（附录 D.3 给出实证）。作者选 tCDP（截断集中差分隐私）正是因为它允许配合局部敏感度的高斯噪声，且可转换为更易解释的 $(\epsilon,\delta)$-DP。

**3. 模式类即权衡旋钮：表达力越强，所需噪声越多**

这是本文最核心的理论洞察，把抽象的「表达力 vs 隐私」量化成一个可操作的旋钮。作者证明（Proposition 4.9）：在给定隐私参数 $\rho'$ 下，为达到 Theorem 4.8 的隐私保证所需高斯噪声的方差为

$$\sigma^2=O\!\Big(\big(\max_{F\in\mathcal F}e(F)\big)^2/n^4\Big),$$

即噪声量由模式类中模式的最大边数 $\max_{F}e(F)$ 决定。而更强表达力的图类，其模式的边数界通常更大——作者借助「同态可区分闭包图类」（homomorphism-distinguishing closed graph class）把这一点和具体 GNN 架构挂钩：MPNN（1-WL）对应树（$e(F)\le m-1$），$k$-FGNN（$k$-WL）对应树宽 $\le k$ 的图（$e(F)\le km-\binom{k+1}{2}$），subgraph $k$-GNN、谱不变 GNN 等各有对应图类（Table 1）。于是结论很干脆：要让嵌入在期望上匹配某个 GNN 的表达力，就从对应图类采样模式；但**越强的图类边数越大、所需噪声越多、utility 越可能下降**——这就是被显式量化出来的隐私-表达力权衡。Theorem 4.8 把前两个设计与本设计合在一起，保证最终嵌入同时是 $\mathcal F$-期望表达的与 tCDP 的，让用户能按需在表达力和隐私之间取舍。

### 损失函数 / 训练策略
本文不训练嵌入本身——同态密度是确定性计算 + 加噪，属一次性预处理。计算上同态计数一般 NP 难，但对有界树宽模式有 Díaz 等人的 $O(|V(F)|\,|V(G)|^{\text{tw}(F)+1})$ 多项式算法，循环模式可用邻接矩阵幂高效算；作者沿用 Welke 等人的采样策略让运行时在期望上多项式。下游用基础学习器（k-NN/SVM/RF），不针对嵌入再做端到端训练。

## 实验关键数据

实验目标不是刷 SOTA，而是验证三个问题：Q1 固定隐私预算下更强表达力的模式类是否带来更好性能；Q2 固定模式类时加强隐私要求性能如何退化；Q3 私有嵌入是否实际有用。数据集用 4 个 OGBG 分子数据集（MOLHIV/MOLBACE/MOLBBBP/MOLLIPO）、3 个社交网络数据集（REDDIT 系列、GitHub STARGAZERS）与合成 SBM；模式数 $d=50$，隐私参数 $\rho'\in[10^{-8},1]$，并转成 $(\epsilon,\delta)$-DP 解释（$\epsilon\in(0,10]$ 被认为有意义）。

### 主实验（OGBG 数据集，utility 与攻击成功率）

| 设置 | MOLHIV (AUC↑) | MOLBBBP (AUC↑) | MOLBACE (AUC↑) | MOLLIPO (RMSE↓) |
|------|------|------|------|------|
| 本文私有 ($\epsilon=1$) utility | **0.692** | **0.602** | **0.652** | **1.086** |
| 本文私有 ($\epsilon=1$) 攻击成功率 | 0.003 | 0.025 | 0.027 | 0.011 |
| 本文非私有 ($\epsilon=\infty$) utility | 0.745 | 0.644 | 0.752 | 1.055 |
| 本文非私有 ($\epsilon=\infty$) 攻击成功率 | 0.955 | 1.000 | 0.990 | 0.992 |
| RR baseline 私有 ($\epsilon=1$) | 0.488 | 0.440 | 0.457 | 1.568 |
| DPRR baseline 私有 ($\epsilon=1$) | 0.595 | 0.539 | 0.648 | 1.499 |

在 $\epsilon=1$ 下，本文私有嵌入在所有 OGBG 数据集上都优于 RR/DPRR 两个 GNN baseline；MOLHIV/MOLBBBP/MOLLIPO 上 utility 仍保持在非私有版本的 90% 以上。同时攻击成功率从非私有的 0.95~1.0 暴跌到 0.003~0.027，说明隐私保护实际有效。

### 表达力-隐私权衡验证（Q1/Q2）

| 实验 | 设置 | 现象 | 结论 |
|------|------|------|------|
| SBM (Q1) | cycle 模式 vs tree 模式，cycle 可证更强 | cycle 密度性能远好于 tree，且 $\epsilon=1$ 附近仍佳 | 更强表达力的模式类带来更好性能 |
| MOLHIV (Q2) | 最大树宽 tw $\in\{1,2,3\}$ 递增 | AUC 随 tw 增大而下降 | 此数据集 tw=1 已够，更强模式只换来更多噪声、utility 反降 |

### 关键发现
- **零均值噪声是整个方法的支点**：表达力被搬到期望上，使加噪与判别力不再硬冲突——这是论文能同时拿到两个保证的根本原因。
- **平滑/有界度敏感度不是锦上添花而是必需**：附录 D.3 显示直接用全局敏感度实现 DP 会得到「不可用的噪声嵌入」，正是更精细的敏感度界才让私有嵌入有实用 utility。
- **更强表达力不总是更好**：是否值得换强模式类取决于任务——SBM 上 cycle 大幅领先，MOLHIV 上 tw=1 即够，盲目加表达力只会因噪声变多而掉点。
- **私有嵌入本身高度信息化**：仅用 k-NN/SVM/RF 这类基础学习器就能在网络数据集上逼近 Hidano & Murakami 的结果，说明嵌入承载了足够结构信息。

## 亮点与洞察
- **把「DP 噪声均值为零」用成表达力的护身符**：大多数人把 DP 噪声当作纯粹的 utility 损失，本文反过来论证只要表示是「期望完整」的，零均值噪声就不破坏期望表达力——这是一个很巧的视角转换。
- **用同态密度的最大边数 $e(F)$ 一个量同时刻画表达力和隐私代价**：把一个看似定性的张力压缩成 $\sigma^2=O((\max e(F))^2/n^4)$ 这样一个可计算的标度，并接上现成的 GNN-同态对应表，让「选多强的模型」变成「选多大边数的模式类」这一可操作决策。
- **可迁移思路**：「先找一个期望意义下无偏/完整的表示，再叠零均值 DP 噪声」这套配方可推广到其他需要兼顾判别力与隐私的表示学习场景；平滑/有界度敏感度的精细化也可借鉴到别的结构敏感统计量的私有发布。

## 局限与展望
- **继承同态计数的固有缺陷**：同态计数计算可能昂贵；更关键的是仅靠结构同态密度、忽略节点/边特征及其拓扑排布，在特征主导的任务上性能可能不足（作者明确承认）。
- **只覆盖边级隐私、图级任务**：方法专为「保护单边有无 + 图级预测」设计，节点级隐私或节点级任务不在保证范围内；邻接图被定义为节点数相同、差一条边，限制了适用扰动模型。
- **有界度先验依赖**：更紧的 Theorem 4.5 依赖可用的最大度上界，没有公开度界时要么私有估计（额外隐私预算）要么退回较松的 counting lemma。
- **改进方向**：作者提出更精细地分析特定图类的敏感度以减小噪声、以及私有编码边特征来进一步改善隐私-utility 权衡。

## 相关工作与启发
- **vs 差分隐私图学习（如 Sajadmanesh、Hidano & Murakami 的 DPRR）**：他们在隐私约束下追 utility 但不给表达力保证；本文同时给出可证表达力与可证隐私，且实验中以更简单的分类器逼近甚至超过这些 baseline。
- **vs 同态计数表达力工作（Nguyen & Maehara、Welke 等）**：他们用同态密度构造期望完整的高表达嵌入但不考虑隐私；本文在其期望完整性之上叠加 DP 噪声并证明加噪后仍保留期望表达力。
- **vs $k$-WL / GNN 表达力分析（Xu、Morris 等）**：传统分析只识别表达力的理论上界；本文借「同态可区分闭包图类」把表达力等级翻译成模式类，进而第一次把表达力和隐私代价绑到同一个量 $e(F)$ 上。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次形式化研究图表示学习中表达力与边级隐私的权衡，并给出可计算的量化刻画。
- 实验充分度: ⭐⭐⭐⭐ 紧凑但针对性强，覆盖合成+分子+网络数据集并验证 Q1-Q3；非 SOTA 导向，规模有限。
- 写作质量: ⭐⭐⭐⭐⭐ 定义、定理、权衡推导层层递进，理论叙述清晰。
- 价值: ⭐⭐⭐⭐ 为隐私保护图学习提供了带形式化保证的新基线与可迁移的「期望表达力 + 零均值 DP 噪声」配方。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses](robustness_in_text-attributed_graph_learning_insights_trade-offs_and_new_defense.md)
- [\[ICLR 2026\] Graph Representational Learning: When Does More Expressivity Hurt Generalization?](graph_representational_learning_when_does_more_expressivity_hurt_generalization.md)
- [\[ICLR 2026\] Topology Matters in RTL Circuit Representation Learning](topology_matters_in_rtl_circuit_representation_learning.md)
- [\[ICLR 2026\] Temporal Graph Thumbnail: Robust Representation Learning with Global Evolutionary Skeleton](temporal_graph_thumbnail_robust_representation_learning_with_global_evolutionary.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](../../ICML2026/graph_learning/t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)

</div>

<!-- RELATED:END -->
