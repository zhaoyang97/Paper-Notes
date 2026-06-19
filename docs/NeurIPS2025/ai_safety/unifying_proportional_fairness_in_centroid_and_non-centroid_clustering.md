---
title: >-
  [论文解读] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering
description: >-
  [NeurIPS 2025][AI安全][公平聚类] 将质心聚类(centroid)和非质心聚类(non-centroid)的比例公平性研究统一到"半质心聚类"框架中，证明了两者不可同时实现的不可能性定理，并设计了新算法在双度量损失下实现常数倍近似的核(core)保证。 公平聚类要求对每一个可能的群体：提供与其规模成比例的公…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "公平聚类"
  - "比例公平"
  - "核稳定性"
  - "半质心聚类"
  - "近似算法"
---

# Unifying Proportional Fairness in Centroid and Non-Centroid Clustering

**会议**: NeurIPS 2025  
**arXiv**: [2601.00447](https://arxiv.org/abs/2601.00447)  
**代码**: 无  
**领域**: AI安全 / 公平性  
**关键词**: 公平聚类, 比例公平, 核稳定性, 半质心聚类, 近似算法

## 一句话总结
将质心聚类(centroid)和非质心聚类(non-centroid)的比例公平性研究统一到"半质心聚类"框架中，证明了两者不可同时实现的不可能性定理，并设计了新算法在双度量损失下实现常数倍近似的核(core)保证。

## 研究背景与动机

公平聚类要求对**每一个可能的群体**提供与其规模成比例的公平保证，而不仅仅关注预定义的受保护属性。先前工作在两个独立范式中研究了比例公平性：

**质心聚类** (Chen et al., 2019)：每个数据点的损失由其到所在簇的代表点（质心）的距离决定。典型应用：设施选址——居民希望分配的设施离自己近。

**非质心聚类** (Caragiannis et al., 2024)：每个数据点的损失由其到同簇中最远数据点的距离决定。典型应用：联邦学习——参与者希望同簇伙伴的数据分布与自己相近。

**核心矛盾**：在许多真实应用中，用户同时关心这两种损失。例如设施选址中，居民既希望设施离家近(质心损失)，又希望使用同一设施的人来自自己的社区(非质心损失)。再如联邦学习中，参与者既关心学到的模型好不好(质心)，又关心同伴数据分布是否相似(非质心)。更进一步，两种损失的度量空间可能完全不同——老师分组时，学生对项目主题的偏好(质心)和对同组同学的偏好(非质心)可能基于完全不同的相似性度量。

**不可能性结果**：论文首先证明了一个重要的不可能性定理(Theorem 1)——没有任何聚类方案能同时在质心和非质心损失上保持有限近似的核稳定性。这意味着直接组合两种损失的已有算法是不可行的，必须设计全新的联合算法。

## 方法详解

### 整体框架

提出**半质心聚类(semi-centroid clustering)**模型，每个agent $i$ 在簇 $(C_t, x_t)$ 中的损失 $\ell_i(C_t, x_t)$ 同时依赖于：(1) 簇的成员集合 $C_t$（谁和我在一起）和 (2) 簇的质心 $x_t$（质心在哪里）。

两种结构化损失函数族：
- **双度量损失(Dual Metric Loss)**：$\ell_i(\mathcal{X}) = \max_{j \in C} d^m(i,j) + d^c(i, x_t)$，使用两个不同的度量空间
- **加权单度量损失(Weighted Single Metric Loss)**：$\ell_i(\mathcal{X}) = \lambda \cdot \max_{j \in C} d(i,j) + (1-\lambda) \cdot d(i, x_t)$，使用同一度量空间的加权组合

### 关键设计

1. **不可能性定理(Theorem 1)**：

    - 构造了一个6个agent、3个簇的精巧实例：$\{a,b,c,d,e,f\}$
    - 证明非质心核要求 $\{a,b\}$、$\{e,f\}$、$\{c,d\}$ 必须各自成簇
    - 证明质心核对该固定分区无法满足（无论怎么选质心，都存在偏离联盟）
    - 意义：排除了直接复用已有Greedy Capture算法变体的可能性

2. **双度量算法(Algorithm 3)**：

    - 两阶段设计：
        - **Phase 1**：迭代找最紧凑簇(Most Cohesive Cluster, MCC)——找使最大成员损失最小的簇-质心对。每次找到后移除这些agent
        - **Phase 2**：选择性允许agent切换簇，关键约束：
       - 切换不能过度损害新簇成员：切换后新簇成员的损失 $\leq c \cdot r_{t'}$
       - 切换后自己的损失有上界保证
       - 每个agent的决策独立于其他agent的切换决策
    - 增长因子 $c$ 是关键参数，控制切换的宽松程度：$c = \frac{3\alpha + \sqrt{\alpha(\alpha+8)}}{4\alpha}$
    - 设计动机：在质心GC（允许自由切换）和非质心GC（禁止切换）之间取折中

3. **核心理论结果(Theorems 2-3)**：

    - 存在性：双度量损失下总存在3-core的聚类(使用精确MCC子程序)
    - 多项式算法：使用4-近似MCC子程序，可在多项式时间内找到$(3+2\sqrt{3})$-core的聚类
    - FJR(Fully Justified Representation)的更强保证：存在1-FJR（精确FJR），多项式时间4-FJR

### 损失函数 / 训练策略
- 本文是理论工作，不涉及训练
- 算法运行时间取决于MCC子程序的效率
- 对加权单度量损失，当 $\lambda$ 变化时提供了参数化的近似因子 $\min\{2/\lambda, f_\lambda\}$

## 实验关键数据

### 主实验

**真实数据集上的核近似倍数**

| 数据集 | $\lambda$ | Algorithm 3 核近似 | k-means++ 核近似 | 说明 |
|------|------|------|----------|------|
| Occupancy | 0.5 | ~1.1 | ~2.5 | 远优于理论上界 |
| Iris | 0.5 | ~1.0 | ~3.0 | 近乎完美 |
| Wine | 0.5 | ~1.1 | ~2.8 | 一致优势 |

### 消融实验

| 配置 | 核近似(中位数) | 经典聚类目标损失 | 说明 |
|------|---------|------|------|
| Algorithm 3 | ~1.05 | 仅轻微增加 | 公平性与效率兼顾 |
| k-means++ | ~2.5 | 最优 | 不关注公平性 |
| 纯理论下界 | ≥2 | — | 双度量 |
| 纯理论下界 | ≥1 | — | FJR |

### 关键发现
- 实际性能远优于理论最坏情况：算法通常实现接近1的核近似倍数
- 公平性的代价很小：经典聚类目标仅有轻微增加
- 与k-means++相比，公平聚类在核近似上一致且显著更优
- 结果推广了Chen et al.和Caragiannis et al.在各自范式中的类似观察
- 不可能性定理意味着需要将 $\lambda$ 作为输入（不能仅使用度量 $d$）

## 亮点与洞察
- 不可能性定理(Theorem 1)的构造非常精巧：仅用6个点和3个簇就证明了根本性的不可能结果
- Phase 2的选择性切换是算法核心创新：巧妙地在"允许自由切换"和"禁止切换"之间找到平衡
- 理论结果非常完整：存在性上界、算法上界、下界三者形成了清晰的landscape
- 半质心聚类模型的提出具有前瞻性：统一了两个独立的研究方向，且对应很自然的真实场景

## 局限与展望
- $(3+2\sqrt{3}) \approx 6.46$ 的多项式时间核近似因子仍有改进空间，理论下界仅为2
- 非质心损失仅考虑了最大距离(max distance)，平均距离或其他聚合方式未涵盖
- 实验仅使用小规模真实数据集验证，大规模设置的可扩展性未经测试
- 加权参数 $\lambda$ 需要预先指定，未讨论如何从数据中学习最优 $\lambda$
- FJR的多项式时间近似(4-FJR)与存在性(1-FJR)之间有较大gap

## 相关工作与启发
- Chen et al. (2019)的Greedy Capture算法是质心聚类公平性的开创性工作，达到$(1+\sqrt{2})$-core
- Caragiannis et al. (2024)将比例公平扩展到非质心聚类，动机来自联邦学习
- Most Cohesive Cluster问题的求解是算法效率的瓶颈
- 本文统一框架为未来研究提供了一个更一般的平台——可以在半质心设置下同时改进两个方向
- 从排序困(sortition)和策略博弈角度研究聚类公平性值得进一步探索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 统一两个独立范式、不可能性定理和新算法都是重要的理论贡献
- 实验充分度: ⭐⭐⭐ 理论为主，实验验证充分但规模有限
- 写作质量: ⭐⭐⭐⭐ 定理陈述清晰，证明结构严谨，但对非理论读者有一定阅读门槛
- 价值: ⭐⭐⭐⭐ 推进了公平聚类理论，统一框架具有长期影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)
- [\[NeurIPS 2025\] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act](its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[NeurIPS 2025\] OmniFC: Rethinking Federated Clustering via Lossless and Secure Distance Reconstruction](omnifc_rethinking_federated_clustering_via_lossless_and_secure_distance_reconstr.md)

</div>

<!-- RELATED:END -->
