---
title: >-
  [论文解读] Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation
description: >-
  [ICML2025][bipartite ranking] 本文从理论上分析了多标签二部排序（bipartite ranking）中两种聚合策略——损失聚合（loss aggregation）与标签聚合（label aggregation）——的Bayes最优解，揭示了损失聚合会产生"标签独裁"（label dictatorship）现象（某一标签因边际偏斜度而主导排序），而标签聚合能更均衡地对待所有标签。
tags:
  - "ICML2025"
  - "bipartite ranking"
  - "AUC"
  - "多标签排序"
  - "loss aggregation"
  - "label aggregation"
  - "Pareto最优"
  - "标签独裁"
---

# Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation

**会议**: ICML2025  
**arXiv**: [2504.11284](https://arxiv.org/abs/2504.11284)  
**代码**: 无  
**领域**: 其他/学习理论  
**关键词**: bipartite ranking, AUC, 多标签排序, loss aggregation, label aggregation, Pareto最优, 标签独裁

## 一句话总结

本文从理论上分析了多标签二部排序（bipartite ranking）中两种聚合策略——损失聚合（loss aggregation）与标签聚合（label aggregation）——的Bayes最优解，揭示了损失聚合会产生"标签独裁"（label dictatorship）现象（某一标签因边际偏斜度而主导排序），而标签聚合能更均衡地对待所有标签。

## 研究背景与动机

- **二部排序（Bipartite Ranking）** 是监督学习的基础问题，目标是学习一个评分函数 $f: \mathcal{X} \to \mathbb{R}$，使正例排在负例之上，以最大化 AUC（ROC曲线下面积）。
- 传统设定假设只有**单个二值标签**，但实际中常存在**多个二值标签**（如信息检索中的相关性标注与点击率、医学诊断中多位专家的意见、推荐系统中点击概率与多样性）。
- 面对多标签，如何产生单一的、合理的排序？文献中主流有两种做法：
    - **损失聚合（Loss Aggregation, LoA）**：将各标签的 AUC 目标加权求和并优化。
    - **标签聚合（Label Aggregation, LaA）**：先将多个标签合并成一个聚合标签（如求和），再在聚合标签上优化多部 AUC。
- 两者虽已在实践中被广泛使用，但缺乏理论分析指导应该如何选择。本文填补了这一空白。

## 方法详解

### 问题形式化

给定 $K$ 个二值标签 $(\mathsf{Y}^{(1)}, \ldots, \mathsf{Y}^{(K)}) \in \{0,1\}^K$，联合分布 $D^{\text{jnt}}$，每个标签的条件概率为 $\eta^{(k)}(x) = P(\mathsf{Y}^{(k)}=1 \mid \mathsf{X}=x)$，边际先验为 $\pi^{(k)} = P(\mathsf{Y}^{(k)}=1)$。

### 损失聚合（Loss Aggregation）

$$\text{AUC}_{\text{LoA}}(f) = \sum_{k \in [K]} a_k \cdot \text{AUC}(f; D^{(k)})$$

**Bayes最优解**（Proposition 5.2）：最优评分函数等价于按以下加权排序：

$$\gamma(x) = \frac{1}{K} \sum_{k \in [K]} \frac{a_k}{\pi^{(k)} \cdot (1-\pi^{(k)})} \cdot \eta^{(k)}(x)$$

关键观察：即使权重均匀（$a_k = 1$），最优解仍依赖于**边际类先验** $\pi^{(k)}$，偏斜越严重的标签权重越大（$1/[\pi^{(k)}(1-\pi^{(k)})]$ 在 $\pi^{(k)}$ 远离 0.5 时急剧增大）。

### 标签聚合（Label Aggregation）

先聚合标签 $\bar{\mathsf{Y}} = \sum_k \mathsf{Y}^{(k)}$，再优化多部AUC：

$$\text{AUC}_{\text{LaA}}(f) = \mathbb{E}[c_{\bar{Y}\bar{Y}'} \cdot H(f(X)-f(X')) \mid \bar{Y} > \bar{Y}']$$

当代价设为 $c_{\bar{y}\bar{y}'} = |\bar{y} - \bar{y}'| \cdot \mathbf{1}(\bar{y} > \bar{y}')$ 时，**Bayes最优解**（Proposition 5.4）：

$$\gamma(x) = \sum_{k \in [K]} \eta^{(k)}(x)$$

关键优势：最优解是各标签条件概率的**等权求和**，不依赖 $\pi^{(k)}$，天然避免标签独裁。

### "标签独裁"现象（Label Dictatorship）

对确定性标签情形（$\eta^{(k)}(x) \in \{0,1\}$），Proposition 6.1 揭示：

- 若 $\alpha^{(1)} = a_1/[\pi^{(1)}(1-\pi^{(1)})] > \alpha^{(2)}$，则排序**完全由标签1决定**；
- 若反之则完全由标签2决定。

这意味着边际分布偏斜的标签会"劫持"整个排序——即使两个标签的条件质量相同。而标签聚合对 $(1,0)$ 和 $(0,1)$ 不施加偏序，避免了此问题。

## 实验关键数据

### 合成数据实验

| 设置 | 方法 | $\Delta_{\text{AUC}}$（越低越好） |
|------|------|----------------------------------|
| $\pi^{(2)}$ 偏斜增大 | Loss Aggregation | 随偏斜度显著增大 |
| $\pi^{(2)}$ 偏斜增大 | Label Aggregation | 始终保持较低 |

- 随着标签2的偏斜度增加，损失聚合的两个标签AUC差距急剧增大（独裁现象），标签聚合则保持稳定均衡。

### Banking 数据集（UCI）

| 目标函数 | AUC mortgage↑ | AUC loan↑ | Diff AUC↓ | Min AUC↑ |
|----------|---------------|-----------|-----------|----------|
| AUC(mortgage) | 0.637 | 0.523 | 0.113 | 0.523 |
| AUC(loan) | 0.550 | 0.573 | 0.023 | 0.550 |
| **AUC_LaA** | **0.616** | **0.562** | **0.054** | **0.562** |
| AUC_LoA(1,1) | 0.626 | 0.555 | 0.071 | 0.555 |

- 标签聚合在 Min AUC 上最优（0.562），差异值最低（0.054），更均衡。

### MSLR Web30k 数据集

| 目标函数 | AUC Click↑ | AUC Rel↑ | Diff AUC↓ | Min AUC↑ |
|----------|-----------|----------|-----------|----------|
| AUC(Click) | 0.74 | 0.61 | 0.13 | 0.61 |
| AUC(Rel) | 0.70 | 0.67 | 0.03 | 0.67 |
| **AUC_LaA** | **0.73** | **0.68** | **0.05** | **0.68** |
| AUC_LoA | 0.73 | 0.67 | 0.06 | 0.67 |

- 标签聚合 Pareto 支配损失聚合（等权），Min AUC 0.68 > 0.67。

### HelpSteer 数据集

- 固定第一标签为 coherence（最均衡），变化第二标签（不同偏斜度）。
- 损失聚合的每标签AUC差异在所有设置下均高于标签聚合。

## 亮点与洞察

- **理论贡献扎实**：首次完整刻画了两种聚合方式的 Bayes 最优解形式，清晰揭示隐藏在损失聚合中的 $\pi^{(k)}$ 依赖。
- **"标签独裁"概念精准**：直观地用偏序图展示了损失聚合如何在 $(1,0)$ 与 $(0,1)$ 之间强加全局偏好（红色箭头 vs 标签聚合的无序关系）。
- **实用指导价值高**：对于有多个标签源的排序场景（信息检索、推荐系统、医学诊断），建议采用标签聚合 + 线性代价 $c = |y - y'|$。
- **代价函数的选择至关重要**：均匀代价 $c=1$ 的标签聚合不保证 Pareto 最优（Proposition 5.3），而线性代价 $c=|y-y'|$ 则可（Proposition 5.4）。

## 局限与展望

- **仅处理二值标签**：虽然聚合后的标签是多值的，但起始标签限制为 $\{0,1\}$，未扩展到连续评级。
- **未提出新算法**：偏理论分析，实验直接用已有的 AUC 优化方法（逻辑/hinge 代理损失）。
- **未探索直接最大化 Min AUC**：作者自己提到这是有趣的未来方向。
- **标签间的权重控制有限**：标签聚合的 $\sum \eta^{(k)}$ 默认等权，若需非均匀权重则需额外设计代价函数。
- **可扩展性讨论不足**：当 $K$ 较大时，聚合标签的取值范围 $\{0,\ldots,K\}$ 增大，实际效果未充分验证。

## 相关工作与启发

- **多目标排序**：Svore et al. (2011) 开创多标签排序范式；Carmel et al. (2020) 提出标签聚合在搜索中的应用。
- **多部AUC理论**：Uematsu & Lee (2015) 给出多部 AUC 的 Bayes 最优解条件（scale condition），本文直接利用其结果。
- **Pareto最优与线性标量化**：Ruchte & Grabocka (2021) 证明线性标量化的最优解是 Pareto 最优的，但本文揭示 Pareto 最优不意味着实用。
- **公平性关联**：Min AUC 指标与公平学习中的最坏类表现（Sagawa et al., 2020）有思想类比。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统分析两种聚合策略的理论差异，标签独裁概念新颖
- 实验充分度: ⭐⭐⭐⭐ — 合成+3个真实数据集，覆盖不同偏斜度
- 写作质量: ⭐⭐⭐⭐⭐ — 定义-命题-证明-实验一气呵成，图表直观
- 价值: ⭐⭐⭐⭐ — 对多标签排序的实践选择提供了理论指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICML 2025\] Fixed-Confidence Multiple Change Point Identification under Bandit Feedback](fixed-confidence_multiple_change_point_identification_under_bandit_feedback.md)
- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](online_sparsification_of_bipartite-like_clusters_in_graphs.md)
- [\[ICML 2025\] LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection](lapsum_--_one_method_to_differentiate_them_all_ranking_sorting_and_top-k_selecti.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

</div>

<!-- RELATED:END -->
