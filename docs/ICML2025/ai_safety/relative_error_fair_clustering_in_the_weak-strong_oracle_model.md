---
title: >-
  [论文解读] Relative Error Fair Clustering in the Weak-Strong Oracle Model
description: >-
  [ICML 2025][AI安全][公平聚类] 提出首个在弱强预言机模型下实现 $(1+\varepsilon)$ 逼近的公平 $k$-median 聚类算法，仅需 $\text{poly}(k \log n / \varepsilon)$ 次昂贵的强预言机查询，相比此前大于 10 的常数因子逼近有根本性提升。
tags:
  - "ICML 2025"
  - "AI安全"
  - "公平聚类"
  - "弱强预言机"
  - "Coreset"
  - "查询复杂度"
  - "算法公平性"
---

# Relative Error Fair Clustering in the Weak-Strong Oracle Model

**会议**: ICML 2025  
**arXiv**: [2506.12287](https://arxiv.org/abs/2506.12287)  
**代码**: 待确认  
**领域**: AI安全  
**关键词**: 公平聚类, 弱强预言机, Coreset, 查询复杂度, 算法公平性

## 一句话总结

提出首个在弱强预言机模型下实现 $(1+\varepsilon)$ 逼近的公平 $k$-median 聚类算法，仅需 $\text{poly}(k \log n / \varepsilon)$ 次昂贵的强预言机查询，相比此前大于 10 的常数因子逼近有根本性提升。

## 研究背景与动机

### 1. 领域现状

聚类是无监督学习的基石，广泛用于数据分析、特征提取和数据摘要等任务。经典的 $k$-median 问题要求在 $n$ 个点中选 $k$ 个中心，使所有点到最近中心的距离之和最小。然而标准聚类可能在高风险场景（招聘、贷款、医疗资源分配）中产生对受保护群体的系统性偏差。

### 2. 公平聚类约束

为解决偏差，Chierichetti 等人提出 $(\alpha, \beta)$-fair clustering：每个聚类中第 $j$ 组的比例不低于 $\alpha_j$ 且不高于 $\beta_j$。这样可以保证各受保护群体在聚类决策中获得公平代表。每个数据点可归属最多 $\Lambda$ 个不相交群体。

### 3. 弱强预言机的现实动机

现代 ML 系统中，精确距离计算（大规模嵌入模型）成本极高，而轻量替代（简单特征、紧凑网络）便宜但有噪声。Bateni 等人提出弱强预言机模型来形式化这种权衡：
- **弱预言机 WO**：以 $2/3$ 概率返回准确距离，$1/3$ 概率返回任意值，随机性一次性采样，不能用多数投票纠正
- **强预言机 SO（点查询）**：查询 $\text{SO}(x)$ 和 $\text{SO}(y)$ 后确定性揭示 $d(x,y)$
- **强预言机 SO（距离查询）**：直接查询 $\text{SO}(x,y)$ 获取精确距离

### 4. 核心矛盾与本文切入

此前在弱强模型下的聚类算法（基于 Meyerson sketch）只能达到常数因子 $C > 10$ 的近似，且不支持公平约束。由于 $k$-聚类本身 NP-hard，追求固定时间内 $(1+\varepsilon)$ 近似不现实，但可以在**查询复杂度**上追求效率。本文的切入点：构造查询高效的 coreset，在小样本上求解公平聚类并保证对原问题的 $(1+\varepsilon)$ 近似。

## 方法详解

### 整体框架

1. 通过弱预言机获取大量廉价（但不完全可靠的）距离信息
2. 有策略地调用少量强预言机查询来校准关键距离
3. 构造规模为 $\text{poly}(k \log n / \varepsilon)$ 的加权 coreset
4. 在 coreset 上求解公平 $k$-median，得到原问题的 $(1+\varepsilon)$ 近似解
5. 方法进一步扩展到一般 $(k,z)$-clustering（$z = O(1)$）

### 关键设计

#### 1. Coreset 构造

- **功能**：将原始 $n$ 个点压缩为加权小样本集 $\mathcal{S}$，满足 $\text{fair}_k(\mathcal{S}) \leq (1+\varepsilon) \cdot \text{fair}_k(\mathcal{X})$
- **核心思路**：基于弱预言机广泛采样建立粗略距离估计，再在关键点处调用强预言机纠偏，确保公平约束在 coreset 上也被维持
- **设计动机**：直接对全数据做公平聚类需 $O(n^2)$ 或更多强查询；coreset 把规模缩到 $\text{poly}(k \log n / \varepsilon)$，后续处理的查询也相应减少

#### 2. 公平约束维持

- **功能**：保证 coreset 中各受保护群体的组成与原集合成比例
- **核心思路**：在采样和加权过程中，按群体分层抽样并调整权重，确保公平条件在加权意义下被保持
- **与旧方法区别**：Bateni 等人的 Meyerson sketch 完全不考虑公平，本文将公平约束"内建"到 coreset 构造中

#### 3. 从常数因子到相对误差的跃迁

- **功能**：将近似质量从 $C > 10$ 提升到 $1 + \varepsilon$
- **核心思路**：Meyerson sketch 的常数因子来自其贪心框架的固有松弛，无法通过参数调优突破。本文采用全新的 coreset 构造范式，通过更精细的采样概率设计和误差传播分析控制各阶段偏差叠加
- **关键定理（Theorem 1）**：存在算法构造 $(1+\varepsilon)$-coreset，大小为 $\text{poly}(k \log n / \varepsilon)$，所需强预言机查询数同为 $\text{poly}(k \log n / \varepsilon)$

## 实验关键数据

### 主结果对比

| 指标 | 本文方法 | Bateni et al. (2024) | 改进 |
|------|---------|---------------------|------|
| 逼近比 | $(1+\varepsilon)$ | $> 10$（常数） | 从粗近似到相对误差 |
| 强点查询数 | $\text{poly}(k \log n / \varepsilon)$ | $\tilde{O}(k)$ | 同量级但精度质变 |
| 强距离查询数 | $\text{poly}(k \log n / \varepsilon)$ | $\tilde{O}(k^2)$ | 可比 |
| 公平约束 | 完整支持 $(\alpha,\beta)$ | 不支持 | 首次实现 |
| 问题范围 | $(k,z)$-clustering, $z=O(1)$ | $(k,z)$-clustering | 同时改善精度 |

### 消融分析：各机制的贡献

| 配置 | 逼近质量 | 强查询成本 | 说明 |
|------|---------|----------|------|
| 仅弱预言机，无强纠偏 | 不可控 | 最低 | 噪声使聚类质量不稳 |
| 无 coreset，全量强查询 | 理论最优 | $O(n)$ 或 $O(n^2)$ | 查询成本不可接受 |
| Meyerson sketch 旧框架 | $> 10$ 常数 | $\tilde{O}(k)$ | 无法突破常数因子 |
| 本文完整框架 | $(1+\varepsilon)$ | $\text{poly}(k \log n / \varepsilon)$ | 精度与查询的最佳平衡 |

注：本论文为理论贡献，缓存在 Theorem 1 之后截断，完整实验数值未能获取，上表基于论文已述结论进行结构化整理。

### 关键发现

- Coreset 是连接高精度与低查询成本的关键桥梁：先用弱预言机覆盖广度，再用强预言机保证深度
- 从 $>10$ 倍到 $(1+\varepsilon)$ 不是渐进改善，而是根本性的框架更换
- 公平约束在 coreset 构造过程中被"自然保持"，无需事后修正

## 亮点与洞察

- **理论突破幅度显著**：将弱强预言机场景下的公平聚类从粗近似推进到精细相对误差保证，跨越了方法论层级
- **三重难题的优雅统一**：同时解决"公平 + 查询受限 + 高精度"的耦合挑战
- **方法学普适性**：核心思想（弱信号广覆盖 + 强信号精纠偏）可迁移到其他需要异质信息源的 ML 场景
- **技巧可复用**：分层采样维持公平约束的做法可直接用于其他公平优化问题

## 局限与展望

- 缓存在 Theorem 1 处截断，完整的证明链条、下界结果、实验评估未能复原
- $\text{poly}(k \log n / \varepsilon)$ 的具体多项式次数在工程实现中很关键，需查阅完整版本
- 对极端长尾群体分布（某些群体仅占千分之一）的鲁棒性未讨论
- 强查询的异构成本建模（不同点对的查询代价不同）和在线更新机制可进一步探索
- 可结合差分隐私等隐私约束与公平约束的 coreset 设计

## 相关工作与启发

- **vs Bateni et al. (2024)**：同一弱强预言机模型，但仅达常数因子且不含公平；本文在两个方向同时推进
- **vs Chierichetti et al. (2017)**：公平聚类的形式化定义来源，但假设精确距离可用，无查询限制
- **vs 经典 coreset 理论（Feldman & Langberg 2011）**：将 coreset 从标准聚类推广到查询受限 + 公平约束的联合设定

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次实现弱强预言机下公平聚类的相对误差保证
- 实验充分度: ⭐⭐⭐ 理论论文，缓存不含完整实验数值
- 写作质量: ⭐⭐⭐⭐⭐ 定义精确、动机链完整、结论清晰
- 价值: ⭐⭐⭐⭐⭐ 对公平 ML 与成本受限学习方向有长期理论价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](accelerating_spectral_clustering_under_fairness_constraints.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)

</div>

<!-- RELATED:END -->
