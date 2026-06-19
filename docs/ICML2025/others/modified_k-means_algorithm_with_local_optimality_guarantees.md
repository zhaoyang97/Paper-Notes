---
title: >-
  [论文解读] Modified K-means Algorithm with Local Optimality Guarantees
description: >-
  [ICML2025][K-means] 首次指出经典K-means算法并不总是收敛到局部最优解这一长期误解，并提出LO-K-means修改方案，在不增加单步计算复杂度的前提下保证收敛到连续或离散意义下的局部最优解。 K-means（Lloyd算法）是最广泛使用的聚类算法之一。长期以来，学术界普遍认为K-means虽然无法达到…
tags:
  - "ICML2025"
  - "K-means"
  - "局部最优性"
  - "Bregman散度"
  - "聚类算法"
  - "组合优化"
---

# Modified K-means Algorithm with Local Optimality Guarantees

**会议**: ICML2025  
**arXiv**: [2506.06990](https://arxiv.org/abs/2506.06990)  
**代码**: [GitHub](https://github.com/lmingyi/LO-K-means)  
**领域**: 其他/聚类  
**关键词**: K-means, 局部最优性, Bregman散度, 聚类算法, 组合优化

## 一句话总结

首次指出经典K-means算法并不总是收敛到局部最优解这一长期误解，并提出LO-K-means修改方案，在不增加单步计算复杂度的前提下保证收敛到连续或离散意义下的局部最优解。

## 研究背景与动机

K-means（Lloyd算法）是最广泛使用的聚类算法之一。长期以来，学术界普遍认为K-means虽然无法达到全局最优，但至少能收敛到**局部最优解**。scikit-learn文档、多篇经典论文（Grunau & Rozhoň 2022, Balcan et al. 2018）均持此观点。这一"共识"追溯到Selim & Ismail (1984)的证明，但实际上那篇工作只证明了K-means在有限步内收敛，并未严格证明局部最优性。

作者发现Selim & Ismail的Lemma 7存在关键错误：该引理使用的局部最优性条件仅在目标函数 $F$ 为**凸**时成立，而 $F(P)$ 实际上是关于 $P$ 的**凹函数**。这意味着K-means收敛到的解可能**不是**局部最优的。

核心问题：能否在保持相同计算复杂度的前提下，修改K-means算法使其保证局部最优性？

## 方法详解

### 问题形式化

考虑加权K-means问题，给定 $N$ 个点 $X=\{x_i\}_{i=1}^N \subset \mathbb{R}^d$，权重 $W=\{w_i\}$，$K$ 个聚类，使用Bregman散度 $D_\phi$ 作为度量：

$$\min_{P,C} f(P,C) = \sum_{k=1}^{K}\sum_{n=1}^{N} p_{k,n} w_n D(x_n, c_k)$$

其中 $P \in S_1$（离散赋值矩阵），$c_k$ 为聚类中心。Bregman散度定义为：

$$D_\phi(x,y) = \phi(x) - \phi(y) - \langle x-y, \nabla\phi(y) \rangle$$

当 $\phi(x)=\|x\|_2^2$ 时退化为平方欧氏距离。

### 两种局部最优性定义

- **C-local（连续局部最优）**：在 $P$ 的连续松弛域 $S_2$ 的邻域内最优
- **D-local（离散局部最优）**：在 $P$ 的所有相邻离散点 $T(P)$（$|T(P)|=N(K-1)$）中最优。D-local强于C-local

### 反例构造

作者给出一个一维5点2聚类的反例：$x_1=-4, x_2=-2, x_3=0, x_4=1.5, x_5=2.5$，初始中心 $c_1=0, c_2=2.5$。K-means收敛到 $c_1^*=-2, c_2^*=2$，但将 $x_3$ 的赋值从聚类1连续移向聚类2（$\alpha$ 从0到1），聚类损失的导数

$$\frac{d}{d\alpha}f = \frac{-20\alpha(\alpha+12)}{(\alpha-3)^2(\alpha+2)^2} < 0$$

在 $(0,1]$ 上恒为负，说明该收敛解**既不是CJ-local、C-local，也不是D-local**。

### LO-K-means算法

**核心思路**：在K-means收敛后，额外检查并修正局部最优性。

**Function 1 (C-LO)**：保证C-local收敛。当K-means收敛后，检查是否存在距离某点等距的多个聚类中心（$|A(C_P^*)|>1$）。若存在，将该点重新分配到索引更大的聚类（打破对称性），由Theorem 4.3保证严格降低损失。

**Function 2 (D-LO)**：保证D-local收敛。遍历所有点 $n$ 及其非当前聚类 $k_2$，利用Lemma 4.1计算重新分配的损失变化量：

$$\Delta_1(g,a,b) = w_g\big(D(x_g, c_b^*) - D(x_g, c_a^*)\big) - \big((s_a - w_g)D(c_a^{new}, c_a^*) + (s_b + w_g)D(c_b^{new}, c_b^*)\big)$$

若 $\Delta_1 < 0$，则执行该重新分配。**Min-D-LO** 变体选择使 $\Delta_1$ 最小的 $(n, k_2)$ 对。

### 复杂度

- 单步时间复杂度：$O(NK\Gamma_\phi(d))$，与原始K-means**完全相同**
- 空间复杂度：$O((N+K)d)$，与原始K-means**完全相同**

## 实验关键数据

### 合成数据（Figure 1-2）

在不同 $N, K, d$ 的1000次实验中，C-LO在大量设置下能改善K-means结果。改善比例随 $K$ 增大和维度降低而升高，表明K-means不收敛到C-local的情况相当普遍。

### 真实数据集主表（Table 1，欧氏距离）

| 数据集 | K | 算法 | 平均损失 | 最小损失 | 时间(s) |
|--------|---|------|----------|----------|---------|
| Iris (N=150,d=4) | 10 | K-means | 31.55 | 26.78 | <0.001 |
| | | D-LO | **30.53** | **26.18** | <0.001 |
| | | Min-D-LO | 30.55 | **26.18** | <0.001 |
| Wine Quality (N=6497,d=11) | 10 | K-means | 1,378,087 | 1,367,222 | 0.01 |
| | | D-LO | **1,377,711** | **1,367,222** | 0.02 |
| News20 (N=2000,d=1089) | 10 | K-means | 919,058 | 734,571 | 0.71 |
| | | D-LO | **637,004** | **625,281** | 19.52 |
| | | Min-D-LO | 642,980 | 637,400 | 6.65 |

### 消融与关键发现

| 对比维度 | 发现 |
|----------|------|
| C-LO改善频率 | K越大、d越低，K-means不收敛C-local的概率越高 |
| D-LO vs K-means | 所有数据集均取得更低或相同的平均/最小损失 |
| Min-D-LO vs D-LO | 准确性接近，但速度显著更快（News20数据集：6.65s vs 19.52s） |
| K-means++ 初始化 | 搭配D-LO++效果更优，互为补充 |
| 其他Bregman散度 | KL散度、Itakura-Saito散度下同样有效 |
| 与Peng&Xia(2005)对比 | Min-D-LO在高维数据上更快且同样保证D-local |

## 亮点与洞察

1. **破除经典误解**：首次严格证明K-means不一定收敛到局部最优解，颠覆了40年来的共识
2. **极简修改、即插即用**：仅在K-means收敛后加一个检查/修正步骤，可直接集成到现有代码库
3. **零额外复杂度**：单步时间和空间复杂度均不变，理论保证严谨
4. **广泛适用性**：适用于所有Bregman散度（平方欧氏、KL、Itakura-Saito、Mahalanobis等），不限于欧氏距离
5. **与K-means++互补**：可叠加使用，进一步降低聚类损失

## 局限与展望

1. D-LO在高维大规模数据上额外迭代次数可能较多（News20需要870次迭代 vs K-means的35次），虽然Min-D-LO缓解了这一问题
2. 仅考虑单点重新分配（相邻顶点），未考虑多点同时交换的更强局部最优定义
3. 空聚类处理策略（从最大聚类拆分）较为启发式，可能影响最终解的质量
4. 未与更复杂的聚类方法（如谱聚类、深度聚类）比较
5. 对全局最优性的差距（optimality gap）缺乏理论上界

## 相关工作与启发

- **Selim & Ismail (1984)**：最早尝试证明K-means局部最优性（本文指出其错误）
- **Peng & Xia (2005)**：提出D-local最优算法，但本文方法更易集成且高维更快
- **Arthur & Vassilvitskii (2006)**：K-means++初始化，与LO-K-means互补
- **Banerjee et al. (2005)**：Bregman散度下的K-means，声称局部最优但证明不完整
- **启发**：该修改思路可推广到K-means的其他鲁棒变体（如K-means–去除离群点）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 破除持续40年的经典误解，首创性极强
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实数据+多种散度，但缺乏与深度聚类等方法的对比
- 写作质量: ⭐⭐⭐⭐⭐ — 数学严谨，反例清晰，结构完整
- 价值: ⭐⭐⭐⭐ — 对聚类基础算法有重要理论贡献，实用性强，但影响面受限于传统聚类场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](../../AAAI2026/others/approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[ICML 2025\] OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable?](ood-chameleon_is_algorithm_selection_for_ood_generalization_learnable.md)
- [\[ACL 2025\] Theoretical Guarantees for Minimum Bayes Risk Decoding](../../ACL2025/others/theoretical_guarantees_for_minimum_bayes_risk_decoding.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](../../NeurIPS2025/others/sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [\[NeurIPS 2025\] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](../../NeurIPS2025/others/fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)

</div>

<!-- RELATED:END -->
