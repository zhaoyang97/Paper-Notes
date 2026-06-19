---
title: >-
  [论文解读] Near Optimal Best Arm Identification for Clustered Bandits
description: >-
  [ICML2025][best arm identification] 在多智能体聚类多臂赌博机设置下，提出 Cl-BAI 和 BAI-Cl 两种算法，利用聚类结构大幅降低最优臂识别的样本复杂度，并证明 BAI-Cl++ 在 $M$ 为常数时达到 minimax 最优。 联邦多臂赌博机 (Federated MAB) 近年来…
tags:
  - "ICML2025"
  - "best arm identification"
  - "clustered bandits"
  - "federated MAB"
  - "successive elimination"
  - "sample complexity"
---

# Near Optimal Best Arm Identification for Clustered Bandits

**会议**: ICML2025  
**arXiv**: [2505.10147](https://arxiv.org/abs/2505.10147)  
**代码**: 无  
**领域**: 其他  
**关键词**: best arm identification, clustered bandits, federated MAB, successive elimination, sample complexity

## 一句话总结

在多智能体聚类多臂赌博机设置下，提出 Cl-BAI 和 BAI-Cl 两种算法，利用聚类结构大幅降低最优臂识别的样本复杂度，并证明 BAI-Cl++ 在 $M$ 为常数时达到 minimax 最优。

## 研究背景与动机

联邦多臂赌博机 (Federated MAB) 近年来受到广泛关注：$N$ 个智能体通过中央学习器协作，各自面对随机 bandit 问题。现实场景（电影推荐、广告投放、Yelp 餐厅推荐）中，不同用户的偏好天然存在**异质性**，朴素地让每个智能体独立做 BAI 会浪费大量样本。

本文用**聚类**建模异质性：$N$ 个智能体被分为 $M$ 个未知簇，同一簇内的智能体共享同一个 bandit 实例（即同一组均值向量和最优臂），不同簇的最优臂不同。目标是在 $\delta$-PC 框架下，为所有 $N$ 个智能体识别各自的最优臂，同时最小化样本复杂度和通信开销。

核心挑战在于：

- 智能体到簇的映射 $\mathcal{M}: [N] \to [M]$ 事先未知
- 需要同时完成聚类和最优臂识别两个子任务
- 联邦场景下通信代价昂贵

## 方法详解

### 问题形式化

实例 $\mathcal{I} = ([N], [M], [K], \mathcal{M}, \Pi)$，其中 $K$ 个臂对所有智能体通用，bandit $m$ 的臂 $k$ 对应 1-subGaussian 奖励分布 $\Pi_{m,k}$，均值 $\mu_{m,k}$。最优臂 $k_m^* = \arg\max_k \mu_{m,k}$，gap 定义为 $\Delta_{m,j} = \mu_{m,k_m^*} - \mu_{m,j}$。

**可分离性假设 (Assumption 2.1)**：存在 $\eta > 0$，使得对任意两个不同的 bandit $a, b$：

$$\mu_{b, k_b^*} - \mu_{b, k_a^*} \geq \eta, \quad \forall a \neq b$$

即每个 bandit 的最优臂在其他 bandit 上表现至少差 $\eta$，这保证了不同 bandit 可通过采样区分。

### 算法 I：Cl-BAI（先聚类，再 BAI）

**两阶段流程**：

1. **聚类阶段**：每个智能体 $i$ 独立运行 Successive Elimination (SE)，参数 $\gamma = (\delta / 12NK)^{4/3}$，$R = \log(17/\eta)$，得到存活臂集 $S_i$ 和经验均值 $\hat{\mu}^i$。利用关键性质——同簇智能体的经验均值在 $\eta/2$ 内一致，不同簇智能体必存在差异——构建图 $\mathcal{G}$，连通分量即为聚类结果。
2. **BAI 阶段**：每个簇选一个代表智能体，在存活臂集 $S_i$ 上再次运行 SE（$R = \infty$），找出最优臂并广播给同簇所有智能体。

### 算法 II：BAI-Cl（先 BAI，再聚类）

**两阶段流程（顺序相反）**：

1. **BAI 阶段**：随机采样智能体，对每个新智能体运行 SE 找最优臂；若该最优臂已在集合 $S$ 中则直接匹配，否则加入 $S$。当 $|S| = M$ 时第一阶段结束。采样智能体数量由 Coupon Collector 问题决定，期望为 $O(M \log M)$。
2. **聚类阶段**：对剩余 $N - O(M\log M)$ 个智能体，只在 $M$ 个候选最优臂上运行 SE，大幅减少每个智能体的采样量。

### 算法 III：BAI-Cl++（改进版）

在 BAI-Cl 基础上引入额外假设 (Assumption 6.1)：最优臂 $k_i^*$ 在不同 bandit 下的均值差异至少为 $\eta_1$，即 $|\mu_{i,k_i^*} - \mu_{j,k_i^*}| \geq \eta_1$。第二阶段用改进的 $\widehat{SE}$ 过程（Algorithm 4），利用第一阶段估计的均值 $\bar{\mu}_S$ 做快速验证，减少 $\log(1/\delta)$ 和 $\log N$ 项的乘子。

## 理论结果

### 样本复杂度对比

| 算法 | 样本复杂度（主项，忽略 log 因子） | 通信代价 |
|------|------|------|
| Naive（独立 BAI） | $N \cdot K \cdot \bar{\Delta}^{-2}$ | $O(N \log K \cdot c_b)$ |
| Cl-BAI | $N \cdot K \cdot \max\{\bar{\Delta}, \eta\}^{-2} + M \cdot K \cdot \bar{\Delta}^{-2}$ | $O(N \cdot K \cdot c_r)$ |
| BAI-Cl | $M \cdot K \cdot \bar{\Delta}^{-2} + N \cdot M \cdot \eta^{-2}$ | $O(N \cdot M \log K \cdot c_b)$ |
| BAI-Cl++ | $M \cdot K \cdot \bar{\Delta}^{-2} + N \cdot M \cdot \bar{\Delta}^{-2} \cdot (\log M) + N \cdot \eta_1^{-2}$ | $O(N \cdot M \log K \cdot c_b)$ |

其中 $\bar{\Delta}^{-2} = \frac{1}{MK} \sum_{m,k} \Delta_{m,k}^{-2}$ 反映平均问题难度。

### Minimax 下界

对满足假设的实例类，任何 $\delta$-PC 算法的期望样本复杂度满足：

$$\mathbb{E}[T_\delta^\nu(\mathcal{A})] \gtrsim \max\{M(K-M),\; N\} \cdot \frac{\log(1/\delta)}{\Delta^2}$$

下界是两项的 max：第一项 $M(K-M)$ 对应识别 $M$ 个 bandit 的最优臂集合，第二项 $N$ 对应确定每个智能体所属的 bandit。当 $M$ 为常数时，BAI-Cl++ 达到 order-wise minimax 最优。

### 实验验证

- **MovieLens-1M**：100 用户分为 6 个年龄组，BAI-Cl++ 相比 Naive 方案减少 **72%** 样本复杂度
- **Yelp**：类似设置，BAI-Cl++ 减少 **65%** 样本复杂度
- 在合成数据上，$M \ll N$ 时优势更加显著

## 亮点与洞察

1. **两种互补的算法设计思路**：先聚类再 BAI vs 先 BAI 再聚类，分别适用于不同场景。Cl-BAI 第一阶段可并行，BAI-Cl 在 $M \ll N,K$ 时样本效率更高
2. **Coupon Collector 的巧妙运用**：BAI-Cl 利用随机采样 + coupon collector 分析，仅需 $O(M\log M)$ 个代表即可覆盖所有簇
3. **完整的理论保证**：$\delta$-PC 正确性、样本复杂度上界、minimax 下界，且当 $M$ 为常数时证明达到最优
4. **通信-样本复杂度权衡的清晰分析**：体现了联邦学习中利用协作以采样换通信的核心思想

## 局限与展望

1. **可分离性假设较强**：Assumption 2.1 要求不同 bandit 的最优臂不同且差距至少为 $\eta$，$\eta$ 过小时聚类阶段采样量 $\sim \eta^{-2}$ 可能主导总复杂度
2. **簇大小需均匀**：BAI-Cl 的分析假设智能体均匀分配到 $M$ 个簇，非均匀情况虽可推广但未深入分析
3. **$M$ 需已知**：算法要求预先知道簇数量 $M$（或其上界），实际场景中 $M$ 可能未知
4. **仅限 fixed-confidence 设置**：未讨论 fixed-budget 变体
5. **BAI-Cl++ 需额外假设**：Assumption 6.1 需要 $\eta_1$ 的先验知识，限制了适用范围
6. **仅用 Successive Elimination**：虽然作者提到可替换为 Track-and-Stop 等方法，但未给出对应分析

## 相关工作与启发

- **(Chawla et al., 2023)**：最相关，研究类似聚类结构但目标为 group regret 且允许 gossip 通信
- **(Kota et al., 2023)**：非聚类联邦 BAI，同时找全局和局部最优臂
- **(Pal et al., 2023)**：聚类联邦 bandit 的 regret minimization，用在线矩阵补全
- 启发：可以考虑将聚类结构推广到 contextual bandit 或 linear bandit 设置下的 BAI 问题

## 评分

- 新颖性: ⭐⭐⭐⭐ — 先聚类/先BAI两种互补范式+minimax最优性证明
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实数据（MovieLens, Yelp），验证理论预测
- 写作质量: ⭐⭐⭐⭐ — 理论分析清晰，算法描述规范
- 价值: ⭐⭐⭐⭐ — 为聚类联邦BAI提供了近最优的算法框架和完整理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Near-Optimal Best-of-Both-Worlds Algorithm for Federated Bandits](../../ICLR2026/learning_theory/a_near-optimal_best-of-both-worlds_algorithm_for_federated_bandits.md)
- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [\[ICML 2025\] Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](../../ICML2026/learning_theory/optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)

</div>

<!-- RELATED:END -->
