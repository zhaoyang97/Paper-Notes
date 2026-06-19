---
title: >-
  [论文解读] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator
description: >-
  [AAAI 2026][优化/理论][凸聚类] 本文将 Median of Means (MoM) 估计器融入凸聚类框架，提出 COMET 算法，通过随机分箱与中位数聚合实现对噪声和离群点的鲁棒性，同时无需预知簇数 $k$，理论上证明了弱一致性，实验在多个真实数据集上显著超越 k-means、MoM k-means、凸聚类等六种基线方法。
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "凸聚类"
  - "Median of Means"
  - "鲁棒聚类"
  - "离群点检测"
  - "Adam优化"
---

# Convex Clustering Redefined: Robust Learning with the Median of Means Estimator

**会议**: AAAI 2026  
**arXiv**: [2511.14784](https://arxiv.org/abs/2511.14784)  
**代码**: [https://tinyurl.com/2v3dx75x](https://tinyurl.com/2v3dx75x)  
**领域**: 聚类 / 鲁棒优化  
**关键词**: 凸聚类, Median of Means, 鲁棒聚类, 离群点检测, Adam优化  

## 一句话总结
本文将 Median of Means (MoM) 估计器融入凸聚类框架，提出 COMET 算法，通过随机分箱与中位数聚合实现对噪声和离群点的鲁棒性，同时无需预知簇数 $k$，理论上证明了弱一致性，实验在多个真实数据集上显著超越 k-means、MoM k-means、凸聚类等六种基线方法。

## 研究背景与动机

### 凸聚类的优势与不足
传统聚类方法（如 k-means）将聚类建模为非凸优化问题，存在三大固有缺陷：(1) 需要预先指定簇数 $k$；(2) 对初始化敏感；(3) 在高维和噪声数据中性能退化。**凸聚类**（sum-of-norms clustering）将问题重新表达为凸优化，保证全局唯一解，从根本上避免了初始化敏感问题。其目标函数为：

$$\min_{\mathbf{u}} \frac{1}{2} \left[ \|\mathbf{x}_i - \mathbf{u}_i\|_2^2 + \gamma \sum_{i,j} \theta_{ij} \|\mathbf{u}_i - \mathbf{u}_j\|_p^2 \right]$$

其中第一项保持每个点靠近其质心，第二项通过调参 $\gamma$ 促进质心融合，自动确定簇数。然而，当融合正则化过强时，凸聚类容易将离群点与真实簇错误合并，在高维含噪数据中表现不佳。

### Median of Means 的鲁棒性保证
MoM 估计器的核心思想是：将数据随机划分为 $L$ 个不相交子集 $B_1, \ldots, B_L$，对每个子集独立计算统计量，然后取中位数作为最终估计。由于离群点通常只污染少数子集，中位数操作能有效抑制其影响。这一特性在对抗性条件下依然成立，具有强鲁棒性和集中不等式保证。

### 本文动机
如何将 MoM 的鲁棒性与凸聚类的稳定性相结合？现有方法（如 Robust Convex Clustering、Robust Continuous Clustering）虽然考虑了鲁棒性，但在噪声水平较高时仍然表现不佳。本文旨在构建一个**无需预知 $k$、对噪声和离群点天然鲁棒**的聚类框架。

## 方法详解

### 整体框架：COMET 算法
COMET（**Co**nvex clustering with **Me**dian of means es**T**imator）的核心设计包含三个关键组件：

#### 1. 随机分箱（Random Binning）
将数据索引集 $\{1,2,\ldots,n\}$ 随机划分为 $l = \mathcal{O}(n)$ 个子集 $B = \{B_i\}_{i=1}^l$，每个子集包含 $b = \lfloor n/l \rfloor$ 个样本。这一策略源自随机特征方法，但本文简化用于数据分箱而非核近似。

#### 2. MoM 代价函数
定义点 $\mathbf{x}_r$ 在凸型代价函数中的"贡献"为：

$$f_U(\mathbf{x}_r) = \frac{1}{2}\|\mathbf{x}_r - \mathbf{u}_r\|_2^2 + \frac{\gamma}{2}\sum_{i,j} w_{ij}\|\mathbf{u}_i - \mathbf{u}_j\|_2^2$$

不直接最小化全局均值 $\frac{1}{n}\sum_r f_U(\mathbf{x}_r)$，而是最小化 MoM 目标：

$$C(\mathbf{U}) = \text{Median}\left(\left\{\frac{1}{b}\sum_{r \in B_j} f_U(\mathbf{x}_r)\right\}_{j=1}^l\right)$$

通过取中位数，离群点对代价函数的影响被有效抑制。

#### 3. 距离截断机制
引入超参数 $\mu$ 对成对距离进行截断：将 $\|\mathbf{u}_i - \mathbf{u}_j\|_2^2$ 替换为 $\min(\mu, \|\mathbf{u}_i - \mathbf{u}_j\|_2^2)$。当任意一对质心的距离超过阈值 $\mu$ 时，对应的融合边被切断，从而防止离群点与真实簇发生虚假合并。最终代价函数为：

$$C(\mathbf{U}) = \text{MoM}_B(\mathbf{U}) + \frac{\gamma}{2}\sum_{i,j} w_{ij} \min\{\mu, \|\mathbf{u}_i - \mathbf{u}_j\|_2^2\}$$

#### 4. Adam 梯度下降优化
由于引入截断后目标函数变为非凸，使用 Adam 优化器来最小化代价函数。梯度为：

$$g_i = \frac{1}{b}(\mathbf{u}_i - \mathbf{x}_i)\mathbb{1}(i \in B_{l_t}) + \gamma \sum_j w_{ij}(\mathbf{u}_i - \mathbf{u}_j)\mathbb{1}(\|\mathbf{u}_i - \mathbf{u}_j\|_2^2 < \mu)$$

#### 5. 后处理聚类分配
优化完成后，以 $\{\mathbf{u}_i\}$ 为顶点构造图，若 $\|\mathbf{u}_i - \mathbf{u}_j\| < \eta_1$ 则连边。每个连通分量视为一个簇；将小于平均簇大小一半的簇合并并标记为噪声。

### 权重设计
采用 k-近邻加高斯核的经典方案：$w_{ij} = \mathbb{1}_{ij,k} \cdot e^{-\phi\|\mathbf{x}_i - \mathbf{x}_j\|_2^2}$，其中 $\phi$ 为带宽参数。

### 理论保证
- **Theorem 1**：在有界噪声 $|\epsilon_i| \leq M$ 的假设下，建立了有限样本误差界，给出 $\|\hat{\mathbf{u}} - \mathbf{u}\|$ 的概率上界
- **Corollary 1.1**：当 $d = o(n)$ 时，质心估计弱一致，即 $\frac{1}{2ndb}\|\hat{\mathbf{u}} - \mathbf{u}\|^2 \xrightarrow{p} 0$
- **Corollary 1.2**：收敛速率为 $O(1/\sqrt{n})$
- **计算复杂度**：$\mathcal{O}(Nnkd)$，与 Robust Convex Clustering 持平，优于标准凸聚类的 $\mathcal{O}(N(n^2d + d\epsilon))$

## 实验

### 实验设置
- **基线方法**：k-means (KM), MoM k-means (MKM), Convex Clustering (CC), Robust Continuous Clustering (RCC), Robust Convex Clustering (RConv), Robust Bregman k-means (RBKM)
- **评估指标**：Adjusted Rand Index (ARI)、Adjusted Mutual Information (AMI)
- **噪声注入**：在数据最小包围超立方体内均匀采样 $p\%$ 个噪声点加入数据
- 为公平比较，对需要 $k$ 的方法使用 GapStat 自动估计簇数

### 表1：真实数据集上的性能（10%噪声）

| 数据集 | 指标 | KM | MKM | CC | RCC | RConv | RBKM | **COMET** |
|--------|------|-----|------|------|------|-------|------|-----------|
| Newthyroid (k=3) | ARI | 0.34±0.21 | 0.40±0.26 | 0.69±0.04 | 0.00±0.00 | 0.81±0.21 | 0.11±0.03 | **0.97±0.01** |
| Wine (k=3) | ARI | 0.66±0.31 | 0.59±0.29 | 0.59±0.15 | 0.00±0.00 | 0.22±0.28 | 0.01±0.02 | **0.79±0.15** |
| Dermatology (k=6) | ARI | 0.61±0.17 | 0.56±0.17 | 0.21±0.00 | 0.00±0.00 | 0.66±0.01 | 0.004±0.02 | **0.81±0.06** |
| Lung-Discrete (k=7) | ARI | 0.44±0.09 | 0.50±0.10 | 0.07±0.03 | 0.41±0.12 | 0.39±0.05 | 0.01±0.01 | **0.71±0.02** |
| ORLRaws10p (k=10) | ARI | 0.33±0.11 | 0.33±0.10 | 0.53±0.00 | 0.00±0.00 | 0.54±0.00 | 0.02±0.01 | **0.73±0.00** |

COMET 在所有数据集上 ARI 均为最高，且标准差显著更低，说明算法稳定性极佳。Wilcoxon 秩和检验证实 COMET 与基线的差异具有统计显著性（†标记）。

### 表2：Brain 数据集不同噪声水平下的性能

| 噪声(%) | 指标 | KM | MKM | CC | RCC | RConv | RBKM | **COMET** |
|---------|------|-----|------|------|------|-------|------|-----------|
| 0% | ARI | 0.28±0.10 | 0.23±0.11 | 0.64±0.00 | 0.00±0.00 | 0.56±0.00 | 0.01±0.01 | **0.65±0.00** |
| 5% | ARI | 0.31±0.13 | 0.31±0.13 | 0.64±0.00 | 0.00±0.00 | 0.56±0.01 | 0.01±0.01 | **0.66±0.00** |
| 10% | ARI | 0.26±0.10 | 0.26±0.10 | 0.64±0.02 | 0.00±0.00 | 0.56±0.06 | 0.016±0.02 | **0.66±0.03** |
| 15% | ARI | 0.22±0.09 | 0.10±0.08 | 0.63±0.02 | 0.00±0.00 | 0.55±0.06 | 0.02±0.02 | **0.66±0.03** |
| 20% | ARI | 0.19±0.11 | 0.08±0.07 | 0.63±0.04 | 0.00±0.00 | 0.63±0.03 | 0.02±0.02 | **0.65±0.02** |

Brain 数据集包含 42 个脑肿瘤样本、5597 维特征、5 个类别。COMET 在所有噪声水平下 ARI 均保持 0.65 以上，而 k-means/MKM 随噪声增加显著退化（20%噪声时降至 0.08-0.19）。RCC 和 RBKM 在所有条件下几乎完全失效。

## 亮点

1. **MoM + 凸聚类的巧妙结合**：通过随机分箱和中位数聚合，将 MoM 的鲁棒统计理论嫁接到凸聚类中，同时保留了无需预知 $k$ 的优势
2. **距离截断的边切断机制**：超参数 $\mu$ 控制的截断函数 $\min(\mu, \|\cdot\|^2)$ 提供了第二层鲁棒性——距离过远的点对不再相互融合
3. **理论支撑完备**：利用 Hanson-Wright 不等式给出了有限样本误差界和弱一致性证明，收敛速率为 $O(1/\sqrt{n})$
4. **计算效率好**：复杂度 $\mathcal{O}(Nnkd)$，与现有鲁棒方法持平，但优于经典凸聚类
5. **实验全面**：涵盖合成数据和 6 个不同规模/维度的真实数据集，5 种噪声水平，6 个基线，并配有 Wilcoxon 统计检验

## 局限性

1. **非凸优化**：引入截断后目标函数变为非凸，Adam 只能找到局部最优，丧失了凸聚类全局唯一解的核心优势
2. **超参数多**：需要调节 $N, k(\text{kNN}), \phi, \gamma, \mu, \eta_1$ 共 6 个超参数，实际使用门槛较高
3. **高维限制**：理论一致性要求 $d = o(n)$，对现代高维小样本场景（如基因组学中 $d \gg n$）可能不适用
4. **噪声分布假设**：仅测试了均匀分布噪声，对结构化噪声（如子空间噪声、长尾分布）的表现未知
5. **估计簇数偏差**：在部分数据集（如 ORLRaws10p、Wisconsin）上估计的簇数与真实值有偏差（14 vs 10, 3 vs 2）
6. **可扩展性未验证**：虽然理论复杂度合理，但实际大规模数据（$n > 10^5$）上的性能未被报告

## 相关工作

- **凸聚类**：[Pelckmans & De Moor 2005] 首次提出凸聚类，[Hocking et al. 2011] 发展正则化路径算法，[Chi & Lange 2015] 提出 ADMM/AMA 优化方法
- **鲁棒聚类**：[Wang et al. 2016] 提出 Robust Convex Clustering，通过特征选择提升鲁棒性；[Shah & Koltun 2017] 提出 Robust Continuous Clustering，处理非凸连续优化
- **MoM 聚类**：[Brunet et al. 2022] 将 bootstrap MoM 用于 k-means；[Paul et al. 2021] 在一般 Bregman 散度下统一了鲁棒中心聚类
- **SDP 松弛**：[Mixon et al. 2017] 证明半定规划松弛在随机单位球模型下可精确恢复

## 评分

- **新颖性**: 3/5 — MoM 和凸聚类均为成熟技术，组合创新但各组件均非全新
- **技术深度**: 4/5 — 理论分析（有限样本界、一致性）扎实，利用 Hanson-Wright 不等式框架较专业
- **实验充分性**: 4/5 — 多数据集、多基线、多噪声水平、统计检验，但缺少大规模实验
- **清晰度**: 3/5 — 公式推导较密集，超参数间的交互关系解释不足
- **影响力**: 3/5 — 鲁棒聚类有实用价值，但高维限制和超参数问题可能限制实际采用
- **总评**: 3.5/5 — 理论与实验俱佳的扎实工作，MoM 缓解离群点影响的思路有价值，但组合创新有限且丧失了凸聚类的凸性保证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World](../../CVPR2025/optimization/convex_relaxation_for_robust_vanishing_point_estimation_in_manhattan_world.md)
- [\[ICML 2026\] CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings](../../ICML2026/optimization/clove_personalized_federated_learning_through_clustering_of_loss_vector_embeddin.md)
- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](../../ICLR2026/optimization/convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](../../ICML2026/optimization/convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)

</div>

<!-- RELATED:END -->
