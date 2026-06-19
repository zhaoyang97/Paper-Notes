---
title: >-
  [论文解读] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁
description: >-
  [CVPR 2026][F-score] 本文从排名理论角度系统研究了 $F_\beta$ 分数族作为 Precision 与 Recall 排名折中的性质，证明 $F_\beta$ 诱导的排名构成 Precision 和 Recall 排名之间的测地线（最短路径），进而提出闭式公式来找到最优的 $\beta$ 值，并证明常用的 $F_1$ 和 skew-insensitive $F_1$ 在大多数情况下都不是最优排名折中。
tags:
  - "CVPR 2026"
  - "F-score"
  - "排名优化"
  - "Kendall距离"
  - "Precision-Recall折中"
  - "性能评估理论"
---

# What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁

**会议**: CVPR 2026  
**arXiv**: [2511.22442](https://arxiv.org/abs/2511.22442)  
**代码**: [https://github.com/pierard/cvpr-2026-optimal-tradeoff-precision-recall](https://github.com/pierard/cvpr-2026-optimal-tradeoff-precision-recall)  
**领域**: 其他  
**关键词**: F-score、排名优化、Kendall距离、Precision-Recall折中、性能评估理论

## 一句话总结

本文从排名理论角度系统研究了 $F_\beta$ 分数族作为 Precision 与 Recall 排名折中的性质，证明 $F_\beta$ 诱导的排名构成 Precision 和 Recall 排名之间的测地线（最短路径），进而提出闭式公式来找到最优的 $\beta$ 值，并证明常用的 $F_1$ 和 skew-insensitive $F_1$ 在大多数情况下都不是最优排名折中。

## 研究背景与动机

1. **领域现状**：Precision 和 Recall 是分类任务最基本的评价指标，但单独使用任何一个都不够全面，因此常取加权调和平均得到 $F_\beta$ 分数。实践中最常用的是 $F_1$（等权），Google Scholar 显示约 31.5 万篇论文使用 F-measure，约 10% 的 CVPR 2025 论文使用 $F_1$。
2. **现有痛点**：虽然 $F_\beta$ 在数值上确实是 Precision 和 Recall 的折中，但从排名角度看，$F_1$ 诱导的排名是否真的是 Precision 排名和 Recall 排名之间的最优折中？这个问题从未被严格研究过。特别地，当类先验 $\pi_+ \to 0$ 时，所有 $F_\beta$ 都退化为模仿 Precision 的排名而忽略 Recall。
3. **核心矛盾**：$F_1$ 被默认为"平衡"的折中，但 $\beta=1$ 的选择没有排名理论基础——同一个 $\beta$ 无法在所有类先验下提供最优排名。
4. **本文目标** (a) $F_\beta$ 诱导的排名是否构成有意义的路径？(b) $F_1$ 是否是最优排名折中？(c) 如何找到最优的 $\beta$？
5. **切入角度**：用 Kendall 秩相关/距离的理论框架来度量排名之间的距离，将寻找最优折中形式化为 Fréchet 方差最小化问题。
6. **核心 idea**：$F_\beta$ 诱导的排名流形是 Precision 与 Recall 排名之间的测地线，最优 $\beta^2$ 等于所有性能对之间 $\vartheta$ 值的中位数。

## 方法详解

### 整体框架

本文不训练任何模型，而是把"$F_1$ 是不是 Precision 与 Recall 之间最好的折中"这个问题搬到**排名空间**里来回答。给定一组分类器的性能集合 $\Pi = \{P_1, \ldots, P_n\}$，每个 $P_i$ 对应一个混淆矩阵；任何评分函数 $X$（Precision、Recall、$F_\beta$ 等）都会把这 $n$ 个分类器排成一个序，记作排名 $\mathbf{x}$。两个排名之间的差异用 Kendall 距离 $d_\tau$ 度量——它数的是两个排名里"谁排谁前面"意见相反的分类器对数。于是"找最优折中"就变成：在 $F_\beta$ 这一族排名里，找一个到 Precision 排名和 Recall 排名"两边等远"的 $\beta$。整条推理分三步走：先证明 $F_\beta$ 族确实落在两端之间的最短路径上（值得搜），再给出最优 $\beta$ 的闭式解（怎么搜到），最后给一个可报告的指标量化任意 $\beta$ 离最优有多远（怎么评）。

### 关键设计

**1. $F_\beta$ 排名的测地线性质：先证明这族分数值得搜**

在 $F_\beta$ 里挑 $\beta$ 之前，得先确认这族分数没"绕路"——否则在里面找等距点毫无意义。关键观察是 $F_\beta$ 本质上是 Precision 与 Recall 的加权 $f$-均值（取 $f(x)=x^{-1}$ 即调和均值）。这种均值有个好性质：对任意两个性能 $P_A, P_B$，只要 Precision 和 Recall **都**认为 $A$ 该排在 $B$ 前面，那么所有 $F_\beta$ 也都这么认为。把这个性质累加到整个性能集合上，就得到核心等式

$$d_\tau(Pr; Re) = d_\tau(Pr; F_\beta) + d_\tau(F_\beta; Re)$$

即从 Precision 排名走到 Recall 排名，途经任意 $F_\beta$ 排名的"路程"恰好相加不亏——$F_\beta$ 排名严格落在两端之间的**测地线**（最短路径）上。这就保证了在 $F_\beta$ 族内搜索折中点是有意义的几何问题。值得强调的是这不是平凡的：算术均值和几何均值都**不**满足性能排名公理，它们可能诱导出违反直觉的排名，因此并非任何"介于两者之间"的分数都能当折中，唯独 $F_\beta$ 族可以。

**2. 最优折中的闭式解：把"等距点"直接算出来，不用数值优化**

既然 $F_\beta$ 排名在测地线上随 $\beta$ 移动，最优折中 $F_*$ 自然定义为到两端等距的那个点，即最小化 Fréchet 方差 $\sigma^2(\beta) = d_\tau^2(Pr; F_\beta) + d_\tau^2(F_\beta; Re)$ 的 Karcher 均值，等价于满足等距条件 $d_\tau(Pr; F_*) = d_\tau(F_*; Re)$。难点在于 $\beta$ 连续而排名离散：随 $\beta$ 从小增大，$F_\beta$ 排名并非平滑变化，而是在某些临界点上发生**离散跳变**——某两个性能 $P_1, P_2$ 恰好被 $F_\beta$ 打成平手然后交换位序。每个这样的跳变点对应一个临界值

$$\beta^2 = \vartheta(P_1, P_2) = -\frac{PTP(P_1)\cdot PFP(P_2) - PTP(P_2)\cdot PFP(P_1)}{PTP(P_1)\cdot PFN(P_2) - PTP(P_2)\cdot PFN(P_1)}$$

其中 $PTP/PFP/PFN$ 是归一化的真正例/假正例/假负例概率。每个跳变让排名向 Recall 一侧挪一格，那么"两边等距"就等价于让跳变点对半分——**最优 $\beta^2$ 正是所有正 $\vartheta$ 值的中位数**。整个解完全解析，只需枚举性能对、算 $\vartheta$、取中位数，不需要任何迭代优化。

**3. 最优度 $\mathcal{O}$：给任意 $\beta$ 一个"离最优多远"的分数**

有了 $F_*$，使用者自然想知道自己手头用的 $\beta$（比如默认的 $F_1$）排得有多好。本文把所有分类器对分成三类：Precision 与 Recall 本就一致的（不用选，谁来都对）、需要做折中选择且选对的、需要选择却选错的。最优度定义为

$$\mathcal{O} = 1 - \frac{d_\tau(F_\beta; F_*)}{d_\tau(Pr; Re)}$$

分子是当前 $\beta$ 的排名与最优排名 $F_*$ 的 Kendall 距离，分母是两端总距离作归一化；当且仅当 $\beta$ 取到最优时 $\mathcal{O}=1$，偏离越远 $\mathcal{O}$ 越小。这是一个可以直接写进论文的单一数字，让任何使用 $F_\beta$ 的人都能报告自己的 $\beta$ 选择离最优排名差多少。

> ⚠️ $\mathcal{O}$ 的精确归一化形式以原文为准。

### 损失函数 / 训练策略

本文为纯理论工作，无训练过程。落地只需一行：$\beta^2 = \text{median}(\{\vartheta(P_i, P_j) \mid i \neq j \wedge \vartheta(P_i, P_j) \geq 0\})$，对所有正的成对临界值取中位数即得最优 $\beta$。

## 实验关键数据

### 主实验

**六个案例研究的最优 $\beta^2$ 值：**

| 案例 | $\tau(Pr; Re)$ | $F_1$ 最优度 | SIVF 最优度 | 最优 $\beta^2$ |
|------|---------------|-------------|------------|---------------|
| 所有性能上的均匀分布 | 1/3 | 100% (最优) | 无意义排名 | 1.0 (恰好是 $F_1$) |
| 固定 TN 概率的均匀分布 | 1/3 | 100% (最优) | 无意义排名 | 1.0 |
| beta 分布 | 变化 | 远非最优 | 远非最优 | 需计算 |
| CADA-RRE 医学挑战赛 (16 个模型) | - | 非最优 | 非最优 | 由闭式公式确定 |
| VOC 2012 分割 (47 类) | - | 非最优 | 非最优 | 逐类不同 |
| 自定义场景 | - | 非最优 | 非最优 | 依赖数据 |

### 消融实验

| 配置 | 说明 |
|------|------|
| $F_1$ ($\beta^2=1$) | 仅在均匀分布等高度对称情况下恰好最优 |
| SIVF ($\beta^2=\pi_-/\pi_+$) | 满足公理（固定先验下），但在真实场景中通常非最优 |
| 启发式 $\beta^2 = E[PFP]/E[PFN]$ | 在多个分布下恰好最优或接近最优 |
| 闭式中位数公式 | 始终最优，$\mathcal{O}=100\%$ |

### 关键发现
- **$F_1$ 极少是最优的**：只有在性能分布具有特定对称性（如均匀分布）时 $\beta=1$ 才恰好是最优折中。在绝大多数实际场景中，$F_1$ 给出偏向 Precision 或偏向 Recall 的排名
- **$F_\beta$ 族是唯一正确的搜索空间**：算术均值和几何均值不满足性能排名公理（它们可以诱导无意义的排名），而所有 $F_\beta$ 都满足
- 启发式规则 $\beta^2 = E[PFP]/E[PFN]$ 在多个分布和一些案例下恰好给出最优结果，可作为简单近似
- 最优 $\beta$ 是数据依赖的——不同分类挑战赛、不同语义类别、不同性能分布需要不同的 $\beta$

## 亮点与洞察
- **理论贡献极为扎实**: 从排名公理出发，经过测地线证明、Karcher 均值推导，最终给出闭式解，整个推导链完整优雅。将一个被用了 50 多年的指标的根本性问题第一次严格回答了。
- **实际意义深远**: CVPR 约 10% 的论文使用 $F_1$，本文指出在大多数情况下 $F_1$ 不是最优排名。虽然不会颠覆现有结论（因为偏离通常不大），但为未来的严谨评估提供了理论武器。
- **闭式公式极简实用**: 只需计算所有性能对的 $\vartheta$ 值取中位数即可，无需复杂优化。
- 作者提出可以在论文中同时报告 $\tau(Pr; F_\beta)$ 和 $\tau(F_\beta; Re)$ 来评估所选 $\beta$ 的最优度，这是一个很好的实践建议。

## 局限与展望
- 本文仅研究二分类的 Precision-Recall 折中，多类分类需要扩展到 macro/micro 平均场景
- $\vartheta$ 计算需要所有分类器的完整混淆矩阵，在只有排行榜分数的场景下无法直接使用
- 论文集中于 $F_\beta$ 族，但是否存在更优的非调和均值形式的排名折中尚未讨论
- 案例研究主要针对分类任务，检测/分割中常用的 AP 指标（整合了多个阈值）不在讨论范围内

## 相关工作与启发
- **vs 传统 $F_1$ 使用**: 大量论文默认 $\beta=1$，本文证明这几乎从不是排名最优的——具体性能集合决定最优 $\beta$
- **vs skew-insensitive $F_1$ (SIVF)**: Flach & Kull 提出的 SIVF 在固定先验下等价于某个 $\beta^2 = \pi_-/\pi_+$ 的 $F_\beta$，能保证有意义排名但不保证最优
- **vs Ferri et al. / Liu et al.**: 之前的分数间相关性研究使用 Pearson/Spearman 相关，本文首次使用 Kendall 秩相关来分析排名折中

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 对一个被使用了 50 年的根本问题给出了第一个理论完备的回答
- 实验充分度: ⭐⭐⭐⭐ 提供了六个案例研究覆盖理论分布和真实挑战赛，但可以加入更多大规模实际竞赛数据
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨清晰，但符号较多，对非理论背景读者有一定门槛
- 价值: ⭐⭐⭐⭐ 为评估实践提供了严格理论基础，但对现有结论的实际影响可能不大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] What Does It Take to Build a Performant Selective Classifier?](../../NeurIPS2025/others/what_does_it_take_to_build_a_performant_selective_classifier.md)
- [\[AAAI 2026\] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](../../AAAI2026/others/how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](../../AAAI2026/others/how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[CVPR 2026\] Region-Wise Correspondence Prediction between Manga Line Art Images](region-wise_correspondence_prediction_between_manga_line_art_images.md)
- [\[CVPR 2026\] Learning What Helps: Task-Aligned Context Selection for Vision Tasks](learning_what_helps_task-aligned_context_selection_for_vision_tasks.md)

</div>

<!-- RELATED:END -->
