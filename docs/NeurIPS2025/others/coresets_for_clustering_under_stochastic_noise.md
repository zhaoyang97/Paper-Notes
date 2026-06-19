---
title: >-
  [论文解读] Coresets for Clustering Under Stochastic Noise
description: >-
  [NeurIPS 2025][Coreset] 首次系统研究噪声数据下 $(k,z)$-聚类 coreset 构造问题，提出新的代理误差度量 $\mathsf{Err}_\alpha$ 替代传统 $\mathsf{Err}$，在温和数据假设下实现 coreset 大小缩减 $\text{poly}(k)$ 倍、质量保证收紧 $\text{poly}(k)$ 倍，并设计噪声感知的 cluster-wise 采样算法。
tags:
  - "NeurIPS 2025"
  - "Coreset"
  - "聚类"
  - "随机噪声"
  - "k-Means"
  - "代理误差度量"
---

# Coresets for Clustering Under Stochastic Noise

**会议**: NeurIPS 2025  
**arXiv**: [2510.23438](https://arxiv.org/abs/2510.23438)  
**领域**: 其他  
**关键词**: Coreset, 聚类, 随机噪声, k-Means, 代理误差度量

## 一句话总结
首次系统研究噪声数据下 $(k,z)$-聚类 coreset 构造问题，提出新的代理误差度量 $\mathsf{Err}_\alpha$ 替代传统 $\mathsf{Err}$，在温和数据假设下实现 coreset 大小缩减 $\text{poly}(k)$ 倍、质量保证收紧 $\text{poly}(k)$ 倍，并设计噪声感知的 cluster-wise 采样算法。

## 研究背景与动机

**领域现状**: Coreset 是聚类问题的核心数据压缩工具——加权子集 $S \subseteq P$ 满足 $\text{cost}_z(S, C) \in (1 \pm \varepsilon)\text{cost}_z(P, C)$ 对所有中心集 $C$ 成立。无噪声情形已有丰富的最优大小理论。

**现有痛点**: 实际数据几乎总是被噪声污染（传感器误差、隐私噪声、传输失真），但现有 coreset 构造**完全依赖无噪声假设**。从噪声数据 $\hat{P}$ 构造的 coreset 对真实数据 $P$ 的质量保证如何？这个根本问题几乎未被研究。

**核心挑战**: (a) 真实数据 $P$ 不可观测，无法直接评估 coreset 质量；(b) 噪声均匀膨胀聚类代价，使传统度量 $\mathsf{Err}$ 过度悲观；(c) 噪声可能改变点的簇分配。

**本文切入角度**: 引入 approximation-ratio 度量 $\mathsf{Err}_\alpha$，专注于最优中心集邻域内的相对性能比，对噪声的均匀代价膨胀天然免疫。

## 方法详解

### 噪声模型
**模型 I**: 每点以概率 $1-\theta$ 不变，概率 $\theta$ 在每维加独立噪声 $\xi_{p,j} \sim D_j$（零均值单位方差，满足 Bernstein 条件）。涵盖高斯机制（差分隐私）、传感器误差等场景。

**模型 II**: 每点每维均加噪声 $\xi_{p,j} \sim D_j$（方差 $\sigma^2$），更一般。

### 两种代理度量

**传统度量 $\mathsf{Err}$**:

$$\mathsf{Err}(S, P) = \sup_{C \in \mathcal{C}} \frac{|\text{cost}_z(S, C) - \text{cost}_z(P, C)|}{\text{cost}_z(S, C)}$$

**新度量 $\mathsf{Err}_\alpha$（核心创新）**:

$$\mathsf{Err}_\alpha(S, P) = \sup_{C \in \mathcal{C}_\alpha(S)} \frac{r_P(C)}{r_S(C)} - 1$$

其中 $\mathcal{C}_\alpha(S) = \{C: r_S(C) \leq \alpha\}$ 为 $S$ 上的 $\alpha$-近似中心集集合，$r_P(C) = \text{cost}_z(P, C) / \mathsf{OPT}_P$。

**关键区别**: $\mathsf{Err}$ 对所有中心集取 sup，对噪声的均匀代价膨胀极其敏感；$\mathsf{Err}_\alpha$ 仅在近似最优中心集上比较**相对排名**，噪声的均匀膨胀被分子分母抵消。

### 直观示例
1-Means 在 $\mathbb{R}$ 上，$n/2$ 个点在 $-1$，$n/2$ 个点在 $+1$，加 $\mathcal{N}(0,1)$ 噪声：
- $\mathsf{Err}(\hat{P}, P) \approx 2/3$ → $r_P(\hat{P}, 1) \leq 25/9$（过度悲观）
- $\mathsf{Err}_1(\hat{P}, P) \leq 1/n$ → $r_P(\hat{P}, 1) \leq 1 + 1/n$（接近完美）

### Theorem 3.1: 基于 $\mathsf{Err}$ 的 coreset
用任何保证 $\mathsf{Err}(S, \hat{P}) \leq \varepsilon$ 的标准算法，coreset 大小 $\tilde{O}(\min\{k^{1.5}\varepsilon^{-2}, k\varepsilon^{-4}\})$（与无噪声相同），但质量保证为：

$$r_P(S, \alpha) \leq \left(1 + \varepsilon + O\left(\frac{\theta nd}{\mathsf{OPT}_P} + \sqrt{\frac{\theta nd}{\mathsf{OPT}_P}}\right)\right)^2 \cdot \alpha$$

噪声引入的额外误差 $O(\theta nd / \mathsf{OPT}_P)$ 不可消除。

### Theorem 3.3: 基于 $\mathsf{Err}_\alpha$ 的 coreset（主要贡献）
**Assumption 3.2**: (1) $\gamma$-代价稳定性（$\mathsf{OPT}_P(k-1)/\mathsf{OPT}_P(k) \geq 1 + \gamma$）；(2) 无强异常值（$r_i \leq 8\bar{r}_i$）。

**Algorithm 1 (CN$_\alpha$)**: 
1. 将 $\hat{P}$ 按 $\hat{C}$ 分为 $k$ 簇 $\hat{P}_i$
2. 对每簇计算平均半径 $\hat{r}_i = \sqrt{\text{cost}(\hat{P}_i, \hat{c}_i) / |\hat{P}_i|}$
3. **去除极端噪声点**: $P_i' = \hat{P}_i \cap B(\hat{c}_i, R_i)$，$R_i = 3\hat{r}_i + O(\sqrt{d}\log\frac{1+\theta kd}{\sqrt{\alpha-1}})$
4. 从每簇均匀采样 $|S_i| = O\left(\frac{\log k}{\varepsilon - \Delta} + \frac{(\alpha-1)\log k}{(\varepsilon - \Delta)^2}\right)$，其中 $\Delta = \frac{\sqrt{\alpha-1}\theta nd}{\alpha\mathsf{OPT}_P}$

得到质量保证：

$$r_P(S, \alpha) \leq \left(1 + \varepsilon + O\left(\frac{\theta kd}{\mathsf{OPT}_P} + \frac{\sqrt{\alpha-1}}{\alpha} \cdot \frac{\sqrt{\theta kd \cdot \mathsf{OPT}_P} + \theta nd}{\mathsf{OPT}_P}\right)\right) \cdot \alpha$$

### 两种方法的对比

| 维度 | $\mathbf{CN}$ (Thm 3.1) | $\mathbf{CN}_\alpha$ (Thm 3.3) | 改善因子 |
|------|------------------------|-----------------------------|---------|
| Coreset 大小 | $\tilde{O}(\min\{k^{1.5}/\varepsilon^2, k/\varepsilon^4\})$ | $\tilde{O}(k/\varepsilon)$ | $\sqrt{k}/\varepsilon$ |
| $r_P$ 噪声误差 | $O(\theta nd / \mathsf{OPT}_P)$ | $O(\theta kd / \mathsf{OPT}_P)$ | $n/k$ |
| 假设需求 | 无 | Assumption 3.2 | — |

### 技术核心
$\mathsf{Err}_\alpha$ 的噪声容忍力来自"中心漂移 vs 代价膨胀"的分离：
- **中心漂移**: $\mathsf{C}(\hat{P}_i) = \mathsf{C}(P_i) + \frac{1}{|P_i|}\sum_p \xi_p$，由独立噪声集中不等式控制，贡献 $O(\theta d / \mathsf{OPT}_{P_i})$
- **代价膨胀**: $\text{cost}(P_i, \mathsf{C}(\hat{P}_i)) = \mathsf{OPT}_{P_i} + \frac{1}{|P_i|}\|\sum_p \xi_p\|_2^2$，同样可控

逐簇分析 + 聚合 → 总误差中 $n$ 被 $k$ 替代。

## 实验关键数据

### 真实数据集上的 coreset 质量（$k$-Means，对比 $\mathbf{CN}$ vs $\mathbf{CN}_\alpha$）

| 数据集 | $n$ | $d$ | $k$ | $\theta$ | $\mathbf{CN}$ 的 $r_P$ | $\mathbf{CN}_\alpha$ 的 $r_P$↓ |
|-------|-----|-----|-----|---------|---------------------|--------------------------|
| MNIST | 60000 | 784 | 10 | 0.1 | 1.32 | **1.08** |
| CIFAR-10 特征 | 50000 | 512 | 10 | 0.1 | 1.28 | **1.05** |

### 非 i.i.d. 噪声下的鲁棒性（Table 7）
当噪声协方差矩阵 $\Sigma \neq \sigma^2 I_d$ 时，$\mathbf{CN}_\alpha$ 仍显著优于 $\mathbf{CN}$，验证了方法在理论假设之外的实用性。

### 假设不满足时的表现
即使数据集违反 Assumption 3.2（如 $\gamma$ 较小的弱分离数据），$\mathbf{CN}_\alpha$ 在实验中仍表现良好（Table 2），显示理论假设是保守的。

### 关键发现
- $\mathsf{Err}_\alpha$ 在所有测试场景下给出比 $\mathsf{Err}$ 更紧的质量保证
- Coreset 大小在大 $k$ 时改善显著（$\sqrt{k}/\varepsilon$ 因子）
- 去除极端噪声点（Ball clipping, Line 3 of Algorithm 1）是关键创新——在不损失信息的前提下稳定簇结构

## 亮点与洞察
- **$\mathsf{Err}_\alpha$ 度量**是对 coreset 评价体系的根本性改进：在噪声场景下从 "对所有 $C$ 一致近似" 弱化为 "在近似最优 $C$ 上保持排名"，是自然且必要的
- **逐簇采样 + Ball clipping**的算法设计简洁有效：通过控制极端噪声点的影响，将分析从全局降维到逐簇
- **中心漂移 vs 代价膨胀的分离**是核心洞察：噪声对代价的**均匀**膨胀不改变最优中心的相对优势，$\mathsf{Err}_\alpha$ 恰好捕捉了这一点
- 理论与实验的一致性良好：Assumption 3.2 虽然是理论保证所需，但算法在不满足时依然有效

## 局限与展望
- Assumption 3.2 的 $\gamma$-代价稳定性在簇重叠严重时难以满足，且所需 $\gamma$ 随 $\alpha$ 和 $\theta$ 增大
- $\mathsf{OPT}_P$ 需要估计——虽然可用 $\text{cost}(\hat{P}, \hat{C})$ 作为代理，但在高噪声时精度下降
- 目前主要分析 $k$-Means ($z=2$)，$k$-Median ($z=1$) 等其他 $z$ 值的推广在附录中但不够深入
- $\mathsf{Err}$ 分析中 Cauchy-Schwarz 引入的 $\sqrt{\theta nd / \mathsf{OPT}_P}$ 项可能不紧——是否能消除尚不清楚
- 实验规模有限，未涉及百万级数据集或高维嵌入空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 新度量 + 新算法 + 首次系统研究噪声 coreset
- 理论深度: ⭐⭐⭐⭐ 两种度量的完整分析 + 明确的改善量化
- 实验充分度: ⭐⭐⭐ 多数据集验证，但实验不够大规模
- 写作质量: ⭐⭐⭐⭐ 问题建模清晰，直观示例有效，符号较重
- 综合: ⭐⭐⭐⭐ 填补了噪声 coreset 的理论空白，实际价值随噪声感知管线的普及而增长

## 相关工作与启发
- **vs 标准 coreset (Feldman-Langberg, Cohen-Addad et al.)**: 均假设无噪声数据，$\mathsf{Err}$ 度量直接适用。本文首次揭示该度量在噪声下的系统性过度悲观
- **vs Robust coreset [38,52,54]**: 将噪声视为可识别异常值并过滤。本文不假设噪声可分离，直接在噪声数据上构造 coreset
- **vs Ben-David & Haghtalab (2014)**: 研究噪声聚类算法鲁棒性（如 $k$-means 在高斯扰动下快速收敛），目标是求解聚类而非构造 coreset
- **与 Coreset for Robust Geometric Median (2510.24621) 互补**: 两篇从不同角度扩展 coreset 理论——前者处理确定性异常值的大小依赖，后者处理随机噪声的代价膨胀。技术工具不同但都在打破传统分析局限
- $\mathsf{Err}_\alpha$ 的设计范式——仅在近似最优解邻域比较相对性能——可迁移到噪声图划分 coreset、隐私保护 facility location 等问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](../../AAAI2026/others/enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)

</div>

<!-- RELATED:END -->
