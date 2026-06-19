---
title: >-
  [论文解读] Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes
description: >-
  [NeurIPS 2025][标记时间点过程] 本文首次揭示标记时间点过程(MTPP)中标记分布不平衡对预测性能的严重影响，提出先预测标记再预测时间的策略，设计阈值方法调节稀有标记的预测概率，并开发无积分近似的IFNMTPP模型高效支持标记概率估计和时间采样。 领域现状 领域现状：标记时间点过程(MTPP)建模事件流中每个事…
tags:
  - "NeurIPS 2025"
  - "标记时间点过程"
  - "标记不平衡"
  - "阈值方法"
  - "无积分近似"
  - "事件预测"
---

# Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes

**会议**: NeurIPS 2025  
**arXiv**: [2510.20414](https://arxiv.org/abs/2510.20414)  
**代码**: [GitHub](https://github.com/undes1red/IFNMTPP)  
**领域**: 其他  
**关键词**: 标记时间点过程, 标记不平衡, 阈值方法, 无积分近似, 事件预测

## 一句话总结

本文首次揭示标记时间点过程(MTPP)中标记分布不平衡对预测性能的严重影响，提出先预测标记再预测时间的策略，设计阈值方法调节稀有标记的预测概率，并开发无积分近似的IFNMTPP模型高效支持标记概率估计和时间采样。

## 研究背景与动机

### 领域现状

**领域现状**：标记时间点过程(MTPP)建模事件流中每个事件的类型(mark)和发生时间，在地震预测、社交媒体转发等场景广泛应用。现有MTPP模型忽略了一个关键问题：**标记分布高度不平衡**。例如7级地震(稀有但重要)比3级地震(频繁)少得多。

不平衡导致的问题：

### 现有痛点

**现有痛点**：频繁标记的条件概率 $p^*(m,t)$ 在大多数时间点上远高于稀有标记

### 核心矛盾

**核心矛盾**：模型几乎总是预测频繁标记，稀有标记的macro-F1极低（如Retweet数据集上稀有标记仅0.027 vs 频繁标记0.618）

现有方法通常"先预测时间 $t$，再预测在 $t$ 时的标记 $p^*(m|t)$"，但这使阈值方法难以应用（因为标记概率随时间变化，无法为所有时间点学习统一阈值）。

## 方法详解

### 整体框架

1. **反转预测顺序**：先预测标记 $p^*(m)$，再预测给定标记的时间 $p^*(t|m)$
2. **阈值方法**：对 $p^*(m)$ 做先验概率归一化后学习阈值，提升稀有标记预测
3. **IFNMTPP模型**：无需数值积分即可计算 $p^*(m)$ 和 $F^*(t|m)$

### 关键设计

1. **标记优先预测 + 阈值方法**:
    - 功能：先基于 $p^*(m) = \int_{t_l}^{+\infty} p^*(m,\tau)\,d\tau$ 预测标记，然后预测时间
    - 核心思路：计算比率 $r_m = p^*(m) / \bar{p}^*(m)$（概率/先验概率），学习阈值 $\epsilon_m$ 使 $m_p = \arg\max_m (r_m - \epsilon_m)$
    - 设计动机：$p^*(m)$ 与时间无关，可以用统一的阈值处理不平衡；稀有标记即使 $p^*(m)$ 低，$r_m$ 也可能高，表明该事件相对其自身基准概率更可能发生
    - 阈值学习：对每个标记m，通过最大化m vs 其他的F1分数确定最优 $\epsilon_m$

2. **无积分近似 (IFNMTPP)**:
    - 功能：避免计算两个代价高昂的反常积分来获取 $p^*(m)$ 和 $F^*(t|m)$
    - 核心思路：定义 $\Gamma^*(m,t) = \int_t^{+\infty} p^*(m,\tau)\,d\tau$，将两个积分统一为对 $\Gamma^*$ 的建模；用神经网络直接参数化 $\Gamma^*(m,t)$ 而非 $\lambda^*(m,t)$
    - 设计动机：$p^*(m) = \Gamma^*(m,t_l)$，$F^*(t|m) = 1 - \Gamma^*(m,t)/\Gamma^*(m,t_l)$，统一了标记概率和时间CDF的计算
    - 关键约束：$\Gamma^*(m,t)$ 必须关于t单调递减且趋于0，通过网络架构设计保证

### 损失函数 / 训练策略

- 负对数似然损失：$\mathcal{L} = -\sum_{i} \log p^*(m_i, t_i)$
- 通过IFNMTPP参数化直接计算 $p^*(m,t)$，无需数值积分
- 时间预测使用逆变换采样(ITS)：从 $F^*(t|m)$ 高效采样
- 阈值 $\epsilon_m$ 在训练集上单独优化，不参与梯度反传

## 实验关键数据

### 主实验（表格）

| 方法 | Retweet (macro-F1) | USearthquake (macro-F1) | StackOverflow (macro-F1) |
|------|-------------------|------------------------|-------------------------|
| SAHP | 0.236 | 0.045 | 0.141 |
| THP | 0.242 | 0.044 | 0.148 |
| IFNMTPP (ours, 无阈值) | 0.293 | 0.056 | 0.155 |
| **IFNMTPP + 阈值** | **0.368** | **0.103** | **0.213** |

阈值方法在所有数据集上显著提升macro-F1，尤其对稀有标记改善巨大。

### 消融实验

- 阈值方法适用于不同基础MTPP模型（SAHP、THP等），均有提升
- 先预测标记再预测时间 vs 先时间再标记：前者与阈值方法配合效果远优
- IFNMTPP的无积分近似精度与数值积分接近但速度快数倍
- 采样数N增大时时间预测精度提升，N=100即可达到良好平衡

### 关键发现

- 标记不平衡在真实数据集中普遍存在且严重影响预测性能
- 现有MTPP模型在稀有标记上的macro-F1极低（接近0）
- 阈值方法的提升主要来自稀有标记的大幅改善，频繁标记性能基本不受影响
- 预测顺序（先标记vs先时间）对不平衡处理方法的可行性有决定性影响

## 亮点与洞察

- **问题发掘重要**：首次系统揭示MTPP中不平衡问题的严重性，填补了重要空白
- **方法优雅**：通过反转预测顺序使阈值方法自然适用，统一积分简化模型设计
- **实用性强**：阈值方法可作为后处理应用于任何MTPP模型
- $\Gamma^*(m,t)$ 的统一建模是核心技术贡献，同时解决了标记概率和时间采样两个问题

## 局限与展望

- 仅处理分类标记，未扩展到连续标记空间
- 阈值方法假设先验概率在训练集和测试集间一致，分布漂移时可能失效
- IFNMTPP的表达能力受限于 $\Gamma^*$ 的单调性约束
- 未探索过采样/欠采样等其他不平衡处理方法与阈值方法的结合

## 相关工作与启发

- 受分类任务中不平衡处理方法（重采样、cost-sensitive、阈值调整）启发
- 与传统Hawkes过程和Neural TPP的区别：首次关注标记不平衡
- IFNMTPP的无积分设计可推广到其他需要避免数值积分的概率模型

## 评分

⭐⭐⭐⭐ — 问题重要且被忽视，方法设计精巧（统一积分+反转预测顺序），实验改善显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](../../ACL2025/others/a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [\[CVPR 2026\] Neural Mixture Density Processes](../../CVPR2026/others/neural_mixture_density_processes.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](fixed-point_rnns_interpolating_from_diagonal_to_dense.md)

</div>

<!-- RELATED:END -->
