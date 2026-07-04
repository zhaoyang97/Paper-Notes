---
title: >-
  [论文解读] Improved and Oracle-Efficient Online $\ell_1$-Multicalibration
description: >-
  [ICML2025][公平性/在线学习][multicalibration] 提出将在线 $\ell_1$-multicalibration 归约为新定义的在线线性乘积优化 (OLPO) 问题，分别达到 $\widetilde{O}(T^{-1/3})$（改进速率）和 $\widetilde{O}(T^{-1/4})$（oracle 高效速率）的多校准误差上界。
tags:
  - "ICML2025"
  - "公平性/在线学习"
  - "multicalibration"
  - "在线学习"
  - "公平性"
  - "oracle-efficient"
  - "校准"
  - "omniprediction"
---

# Improved and Oracle-Efficient Online $\ell_1$-Multicalibration

**会议**: ICML2025  
**arXiv**: [2505.17365](https://arxiv.org/abs/2505.17365)  
**代码**: 待确认  
**领域**: 公平性/在线学习  
**关键词**: multicalibration, 在线学习, 公平性, oracle-efficient, 校准, omniprediction

## 一句话总结
提出将在线 $\ell_1$-multicalibration 归约为新定义的在线线性乘积优化 (OLPO) 问题，分别达到 $\widetilde{O}(T^{-1/3})$（改进速率）和 $\widetilde{O}(T^{-1/4})$（oracle 高效速率）的多校准误差上界。

## 研究背景与动机

### 校准与多校准
- **校准 (Calibration)** 是衡量概率预测质量的经典指标：对于每个预测值 $p$，实际出现该预测的样本中正例比例应趋近 $p$。Foster & Vohra (1998) 证明在线校准在 $\ell_1$ 度量下可达 $O(T^{-1/3})$。
- **多校准 (Multicalibration)** 由 Hébert-Johnson et al. (2018) 提出，要求预测不仅在整体上校准，还需在由假设类 $\mathcal{H}$ 定义的多个子群体上都保持校准。这直接关系到算法公平性——避免对特定性别、种族、年龄等亚群体产生歧视性预测。

### 现有方法的问题

**间接方法导致速率损失**：先前工作 (Gupta et al. 2022; Lee et al. 2022) 先在 $\ell_\infty$ 或 $\ell_2$ 范数下获得多校准保证，再转换到 $\ell_1$，导致额外损失。最好的间接速率仅为 $\widetilde{O}(T^{-1/4})$。

**计算复杂度高**：大多数算法运行时间与 $|\mathcal{H}|$ 成线性，当假设类为指数大小（如 $d$ 维线性函数）时不可行。

**Oracle 高效方法速率差**：Garg et al. (2024) 的 oracle 高效算法速率仅为 $\widetilde{O}(T^{-1/8})$，且需要更强的在线回归 oracle。

### 核心问题
> 能否设计直接在 $\ell_1$ 度量下工作、达到 $O(T^{-1/3})$ 速率、且 oracle 高效的在线多校准算法？

## 方法详解

### 核心归约：多校准 → OLPO

论文的关键洞察是定义了一个新的在线学习问题——**在线线性乘积优化 (Online Linear-Product Optimization, OLPO)**：

- 每轮 $t$，学习者选择 $(h_t, \boldsymbol{\theta}_t) \in \mathcal{H} \times \mathbb{B}_\infty$
- 对手揭示上下文 $\mathbf{x}_t$ 和奖励向量 $\boldsymbol{f}_t \in \mathbb{R}^M$
- 学习者获得乘积形式奖励 $\langle \boldsymbol{\theta}_t, h_t(\mathbf{x}_t) \cdot \boldsymbol{f}_t \rangle$

**Theorem 3.1** 证明：任何 OLPO 后悔界为 $R_T(\mathcal{L}; \mathcal{H})$ 的算法，可被高效转化为多校准误差不超过
$$\frac{B}{m} + \frac{R_T(\mathcal{L}; \mathcal{H})}{T} + 4B\sqrt{\frac{m\log(6T|\mathcal{H}|)}{T}} + \frac{4mB}{T}$$
的在线多校准算法。归约依赖一个 **半空间 oracle (Halfspace Oracle)**，源自 Abernethy et al. (2011) 对校准与 Blackwell 可逼近性的联系。

### 改进速率方法：Lin-OLPO

为求解 OLPO（其奖励函数涉及变量乘积，非标准线性优化），论文引入**线性化版本 (Lin-OLPO)**：

1. **空间扩展**：将决策变量从 $\mathcal{H} \times \mathbb{B}_\infty$ 扩展到 $\mathbb{R}^{|\mathcal{H}| \times M}$ 空间中的混合范数球 $\mathbb{B}_{1,\infty}$
2. **混合范数**：定义 $\|\mathbb{z}\|_{1,\infty} = \sum_{h \in \mathcal{H}} \|\mathbb{z}(h)\|_\infty$，其对偶范数为 $\|\mathbb{z}\|_{\infty,1} = \max_{h} \|\mathbb{z}(h)\|_1$
3. **一致性保证**：通过概率采样 $h_t = h$ 以概率 $\gamma_t(h) = \|\widetilde{\boldsymbol{\theta}}_t(h)\|_\infty$ 并缩放 $\boldsymbol{\theta}_t$，确保 OLPO 和 Lin-OLPO 的最优行动一致

**求解 Lin-OLPO 的算法 (Algorithm 2)**：
- 运行 $|\mathcal{H}|$ 个并行的在线梯度下降 (OGD) 实例 $\mathcal{A}^h$，每个负责假设 $h$ 的校准优化
- 用乘性权重更新 (MWU) 作为专家算法 $\mathcal{E}$，选择"更难"的假设（即校准误差更大的 $h$）
- 最终获得 $\widetilde{O}(\sqrt{T \log |\mathcal{H}|})$ 后悔

**Theorem 1.1**：对有限假设类 $\mathcal{H}$，取 $m = T^{1/3}$，得到
$$\mathbb{E}[K(\pi_T, \mathcal{H})] \leq O(BT^{-1/3}\sqrt{\log(6T|\mathcal{H}|)})$$

### 推广到无穷假设类

**Theorem 1.2**：利用 $\ell_1$-多校准误差关于 $\mathcal{H}$ 的 1-Lipschitz 性质，通过构造 $\beta$-覆盖 $\mathcal{H}_\beta$，将误差界推广为
$$\mathbb{E}[K(\pi_T, \mathcal{H})] \leq O(BT^{-1/3}\sqrt{\log(6T|\mathcal{H}_\beta|)}) + \beta$$

**推论 (Corollary 1.3)**：对 $d$ 维有界线性函数类，速率为 $O(Bd^{1/2}T^{-1/3}\log(BT))$。

### Oracle 高效方法

Lin-OLPO 需枚举所有 $h \in \mathcal{H}$，对大假设类不可行。论文直接在 OLPO 上应用 oracle 高效框架：

- 采用 **Follow-the-Perturbed-Leader (FTPL)** 系列算法 (Dudík et al. 2020)
- 利用 OLPO 的特殊结构：决策可限制在布尔超立方体 $\{±1\}^M$ 上
- 定义离线 oracle：给定历史序列，一次性求解最优 $(h^*, \boldsymbol{\theta}^*)$——本质上等价于离线评估多校准误差
- 每轮仅调用一次离线 oracle

**Theorem 1.4**：在 transductive 或充分分离的上下文假设下，对二值假设类 $\mathcal{H}: \mathcal{X} \to \{0,1\}$，
$$\mathbb{E}[K(\pi_T, \mathcal{H})] \leq \widetilde{O}(T^{-1/4}\sqrt{\log T})$$

## 实验关键数据

本文为纯理论工作，无实验部分。主要理论结果对比：

| 方法 | $\ell_1$-多校准速率 | 计算复杂度 | Oracle 类型 |
|------|---------------------|-----------|-------------|
| Gupta et al. (2022) 经 $\ell_\infty$ 转换 | $\widetilde{O}(T^{-1/4})$ | $O(\|\mathcal{H}\|)$/轮 | 无 |
| Garg et al. (2024) | $\widetilde{O}(T^{-1/8})$ | Oracle 高效 | 在线回归 oracle |
| **本文 (改进速率)** | $\widetilde{O}(T^{-1/3})$ | $O(\|\mathcal{H}\|)$/轮 | 无 |
| **本文 (Oracle 高效)** | $\widetilde{O}(T^{-1/4})$ | 多项式/轮 | 离线 oracle |
| 在线校准下界参考 | $O(T^{-1/3})$ | — | — |

关键改进：
- 改进速率几乎匹配在线校准的最优 $O(T^{-1/3})$ 速率（在线校准是多校准的特例）
- Oracle 高效速率从 $T^{-1/8}$ 提升到 $T^{-1/4}$，且 oracle 从在线降为离线（更弱更易实现）

## 亮点与洞察

1. **OLPO 归约的优雅性**：将多校准这一复杂的公平性约束问题，归约为结构清晰的在线优化问题，既统一了改进速率和 oracle 高效两个目标，也具有独立的理论价值。
2. **线性化技巧**：通过混合范数空间的精巧设计，将涉及变量乘积的非线性奖励线性化，同时保持最优行动的一致性——这是技术上的核心创新。
3. **Lipschitz 性质的利用**：$\ell_1$-多校准误差关于 $\mathcal{H}$ 的 1-Lipschitz 性质是此前未被发掘的结构性质，使得从有限到无穷假设类的推广自然而然。
4. **离线 vs 在线 oracle**：将 oracle 需求从在线回归降到离线评估，是实用性的重要提升——离线 oracle 仅需一次性优化，不需要维护在线状态。

## 局限与展望

1. **Oracle 高效方法的假设限制**：Theorem 1.4 要求 (a) 二值假设类 $\mathcal{H}: \mathcal{X} \to \{0,1\}$，(b) transductive 或充分分离的上下文。能否放松这些假设是开放问题。
2. **速率差距**：改进速率 $T^{-1/3}$ 需要枚举 $\mathcal{H}$，oracle 高效版本降到 $T^{-1/4}$——能否在 oracle 高效下也达到 $T^{-1/3}$ 仍未解决。
3. **缺乏实验验证**：作为纯理论工作，未提供实际数据集上的验证。
4. **与 Noarov et al. (2025) 的关系**：同期工作通过不同框架得到了类似的有限假设类结果，但需要 small-loss regret bound，本文仅需 worst-case regret bound，算法更简单。
5. **更弱数据假设**：作者认为可将结果适配到 smoothed data 或 K-hint data 假设 (Haghtalab et al. 2022; Block et al. 2022)，但未在本文完成。

## 相关工作与启发

- **校准与在线学习的联系** (Abernethy et al. 2011)：本文的半空间 oracle 直接继承自该工作对校准与 Blackwell 可逼近性的联系。
- **多校准与全预测 (Omniprediction)** (Gopalan et al. 2022)：$\ell_1$-多校准的预测器自动成为全预测器，因此本文的改进直接提升了在线全预测的速率。
- **Oracle 高效在线学习** (Syrgkanis et al. 2016; Dudík et al. 2020)：本文的 FTPL 方法直接构建在 Dudík et al. 的广义 FTPL 框架之上。
- **在线多组学习** (Acharya et al. 2024; Deng et al. 2024)：相关但不同的问题，后者针对二值标签和损失函数设计，不直接适用于 OLPO。

## 评分
- 新颖性: ⭐⭐⭐⭐ (OLPO 归约和线性化技巧是新颖的贡献)
- 实验充分度: ⭐⭐ (纯理论工作，无实验)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰，归约层次分明)
- 价值: ⭐⭐⭐⭐ (显著改进了在线多校准的最优速率，对公平 ML 有直接意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)
- [\[ICML 2025\] Avoiding Catastrophe in Online Learning by Asking for Help](avoiding_catastrophe_in_online_learning_by_asking_for_help.md)
- [\[ICML 2025\] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)

</div>

<!-- RELATED:END -->
