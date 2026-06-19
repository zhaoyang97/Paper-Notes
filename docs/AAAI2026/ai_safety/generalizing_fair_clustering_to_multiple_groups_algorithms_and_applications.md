---
title: >-
  [论文解读] Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications
description: >-
  [AAAI 2026 Oral][AI安全][公平聚类] 将最近公平聚类（Closest Fair Clustering）问题从仅两个群体推广到任意多群体，证明三群体以上等比例情形已为NP-hard，提出近线性时间近似算法（等比例 $O(|\chi|^{1.6}\log^{2.81}|\chi|)$、任意比例 $O(|\chi|^{3.81})$），并将结果推广至公平相关聚类和公平共识聚类问题。
tags:
  - "AAAI 2026 Oral"
  - "AI安全"
  - "公平聚类"
  - "多群体公平"
  - "近似算法"
  - "相关聚类"
  - "共识聚类"
  - "NP-hard"
---

# Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.11539](https://arxiv.org/abs/2511.11539)  
**代码**: 待确认  
**领域**: AI安全/公平性  
**关键词**: 公平聚类, 多群体公平, 近似算法, 相关聚类, 共识聚类, NP-hard

## 一句话总结

将最近公平聚类（Closest Fair Clustering）问题从仅两个群体推广到任意多群体，证明三群体以上等比例情形已为NP-hard，提出近线性时间近似算法（等比例 $O(|\chi|^{1.6}\log^{2.81}|\chi|)$、任意比例 $O(|\chi|^{3.81})$），并将结果推广至公平相关聚类和公平共识聚类问题。

## 研究背景与动机

- **领域现状**：聚类是机器学习中的基础无监督学习任务，但传统算法在涉及受保护属性（性别、种族、年龄等）时容易产生不公平的结果。自 Chierichetti et al. (2017) 提出公平聚类以来，社区在 k-center/median/means、相关聚类等多种变体上不断推进公平约束。
- **核心痛点**：Chakraborty et al. [COLT'25] 首次研究了"最近公平聚类"问题——给定已有聚类，以最少修改得到公平聚类。但该工作仅处理**两个群体**的情形，而实际数据往往有年龄、种族、性别等多个受保护属性对应多个群体。
- **核心矛盾**：两群体等比例情形存在精确算法（近线性时间）；本文证明多群体（$\geq 3$）即使等比例也是NP-hard，揭示了从两群体到多群体的**计算复杂度断崖**。
- **切入角度**：通过分层合并（hierarchical block merging）策略设计近似算法，回答了 Chakraborty et al. 提出的开放问题。

## 方法详解

### 整体框架

采用分层递归策略，将多群体公平聚类分解为多轮两两合并操作。核心包括三个算法：(1) **fairpower-of-two**——处理颜色数为2的幂的等比例情形；(2) **make-pdc-fair**——处理任意比例的p-divisible聚类转公平聚类；(3) **create-pdc**——将任意聚类转化为p-divisible聚类。三者组合实现从任意聚类到多群体公平聚类的近似转换。

### 关键设计

1. **算法 fairpower-of-two（等比例 + 颜色数为2的幂）**

    - 功能：将任意聚类 $\mathcal{D}$ 转化为每个簇中各颜色人数相等的公平聚类
    - 核心思路：进行 $\log|\chi|$ 轮迭代，每轮将相邻颜色块两两合并。在第 $i$ 轮，颜色集被分为 $|\chi|/2^i$ 个大小为 $2^i$ 的块，维护不变量——每个簇内同块颜色数相等
    - 每轮操作：对每对相邻块，计算每个簇中两块之间的"盈余（surplus）"，将盈余从簇中移除，收集到集合 $S_j, S_{j+1}$ 中，调用 **multi-GM** 子程序将盈余重新配对形成新的公平子集
    - 近似比：每轮引入因子2，$\log|\chi|$ 轮后通过三角不等式递归得到 $3^{\log|\chi|} = O(|\chi|^{1.6})$

2. **算法 make-pdc-fair（任意比例 p-divisible → 公平聚类）**

    - 功能：给定比例 $p_1:p_2:\cdots:p_r$ 和 p-divisible 聚类，输出满足全局比例的公平聚类
    - 核心思路：类似 fairpower-of-two 的分层策略，但需处理不等比例。进行 $\lceil\log_2 r\rceil$ 轮，每轮合并相邻颜色块，通过"缩放因子（scaling factor）"平衡两块间的不匹配
    - 平衡规则：对子块 $A$ 和 $B$，计算缩放因子 $x$ 和 $y$。若 $x > y$，向簇中合入 $B$ 类颜色的点；若 $x < y$，从簇中裁剪 $B$ 类颜色的点
    - 近似比：每轮引入因子6，总计 $7^{\log r} = O(r^{2.81})$

3. **算法 create-pdc（任意聚类 → p-divisible 聚类）**

    - 功能：对每种颜色 $c_j$，使每个簇中该颜色点数为 $p_j$ 的倍数
    - 核心思路：将簇分为 CUT 集（盈余 $\leq p_j/2$，裁剪盈余）和 MERGE 集（盈余 $> p_j/2$，接收来自其他簇的点以填补赤字）。创建辅助簇容纳多余盈余
    - 近似比：$O(|\chi|)$-close p-divisible

### 损失函数 / 目标函数

本文为组合优化问题，优化目标为最小化 $\text{dist}(\mathcal{D}, \mathcal{F})$——原始聚类与输出公平聚类之间的距离（不一致点对数）。理论分析通过三角不等式链组合各步骤近似比：

$$\text{dist}(\mathcal{D}, \mathcal{F}) \leq O(|\chi|^{3.81}) \cdot \text{dist}(\mathcal{D}, \mathcal{F}^*)$$

## 实验

本文为**理论工作**，不含实证实验，但给出了完整的近似比和计算复杂度分析。

### 主要理论结果

| 问题设定 | 近似比 | 时间复杂度 | 备注 |
|:--|:--|:--|:--|
| 等比例 + $|\chi|$ 为2的幂 | $O(|\chi|^{1.6})$ | $O(|V|\log|V|)$ | fairpower-of-two |
| 等比例 + 任意 $|\chi|$ | $O(|\chi|^{1.6}\log^{2.81}|\chi|)$ | $O(|V|\log|V|)$ | fair-equi |
| 任意比例 | $O(|\chi|^{3.81})$ | $O(|V|\log|V|)$ | fair-general |
| 公平相关聚类（等比例） | $O(|\chi|^{1.6}\log^{2.81}|\chi|)$ | — | 结合 $O(1)$ 相关聚类 |
| 公平相关聚类（任意比例） | $O(|\chi|^{3.81})$ | — | 消除对群体比例 $q$ 的依赖 |
| 公平共识聚类 | 同上 | $O(m^2|V|^2)$ | 首次多群体结果 |

### NP-hard 结果

| 问题 | 硬度结论 | 归约来源 |
|:--|:--|:--|
| $k$-Closest EquiFair ($k \geq 3$) | NP-hard | 3-Partition（强NP完全） |
| 任意比例 Closest Fair Clustering | NP-hard | 同上（扩展归约） |

### 关键发现

- **两群体 vs 多群体的计算断崖**：两群体等比例存在精确的近线性时间算法，但三群体等比例已 NP-hard，是第一个严格证明这一 gap 的工作
- **消除对群体比例的依赖**：公平相关聚类的先前最优近似为 $O(q^2|\chi|^2)$（$q = \max(p_j)/\min(p_j)$ 可达 $\text{poly}(|V|)$），本文的 $O(|\chi|^{3.81})$ 与比例无关
- **首个多群体公平共识聚类算法**：此前仅有两群体常数近似结果

## 亮点

- ⭐ **严格的理论贡献**：证明了多群体公平聚类的NP-hard性，揭示两群体到多群体的根本计算复杂度跳变
- ⭐ **分层合并策略**的通用性：fairpower-of-two + make-pdc-fair 的分层递归框架可自然推广到相关聚类和共识聚类
- ⭐ **近线性时间**算法：$O(|V|\log|V|)$ 的时间复杂度使算法具有良好的实际可伸缩性
- ⭐ 回答了 COLT'25 提出的两个开放问题

## 局限性

- **近似比随颜色数增长**：$O(|\chi|^{3.81})$ 在群体数量较多时近似质量可能不理想，改进近似因子是重要的未来方向
- **纯理论工作**：缺乏实证实验验证算法在真实数据集上的实际效果和运行效率
- **公平性定义较严格**：要求每个簇的群体比例精确匹配全局比例，没有考虑允许一定容差的松弛公平定义
- **距离度量单一**：仅考虑基于不一致点对数的距离，未涉及其他度量空间下的公平聚类

## 相关工作

- **公平聚类开创性工作**：Chierichetti et al. (2017) 首次研究公平聚类，处理两群体; Rosner & Schmidt (2018) 推广到多群体比例约束
- **Closest Fair Clustering**：Chakraborty et al. [COLT'25] 提出最近公平聚类问题，给出两群体精确和常数因子近似算法
- **公平相关聚类**：Ahmadian et al. [AISTATS'20]、Ahmadi et al. (2020) 给出 $O(q^2|\chi|^2)$ 近似，本文改进为与 $q$ 无关的近似
- **相关聚类**：Bansal et al. (2004) 综合研究; 最优 1.438-近似 (Cao et al. 2024); APX-hard
- **公平共识聚类**：Chakraborty et al. [COLT'25] 提出并解决两群体情形

## 评分

⭐⭐⭐⭐ — 理论贡献扎实，解决了重要的开放问题，但缺少实验验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Fair Model-Based Clustering](fair_model-based_clustering.md)
- [\[ICLR 2026\] Fair Conformal Classification via Learning Representation-Based Groups](../../ICLR2026/ai_safety/fair_conformal_classification_via_learning_representation-based_groups.md)
- [\[ICML 2025\] Relative Error Fair Clustering in the Weak-Strong Oracle Model](../../ICML2025/ai_safety/relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)
- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](../../NeurIPS2025/ai_safety/unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [\[CVPR 2026\] PoInit-of-View: Poisoning Initialization of Views Transfers Across Multiple 3D Reconstruction Systems](../../CVPR2026/ai_safety/poinit-of-view_poisoning_initialization_of_views_transfers_across_multiple_3d_re.md)

</div>

<!-- RELATED:END -->
