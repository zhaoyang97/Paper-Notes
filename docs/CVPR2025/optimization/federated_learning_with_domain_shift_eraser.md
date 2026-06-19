---
title: >-
  [论文解读] Federated Learning with Domain Shift Eraser
description: >-
  [CVPR 2025][优化/理论][联邦学习] 提出FDSE方法，将每层网络分解为域无关特征提取器（DFE，全局聚合增强共识）和域特异偏移消除器（DSE，个性化聚合保留本地特性），结合BN一致性正则化，在DomainNet上达到76.77%（超Ditto 1.6%），在Office-Caltech10上达到91.58%（超FedBN 4.6%）。
tags:
  - "CVPR 2025"
  - "优化/理论"
  - "联邦学习"
  - "域偏移"
  - "层分解"
  - "一致性正则化"
  - "双聚合"
---

# Federated Learning with Domain Shift Eraser

**会议**: CVPR 2025  
**arXiv**: [2503.13063](https://arxiv.org/abs/2503.13063)  
**代码**: 无  
**领域**: 优化  
**关键词**: 联邦学习、域偏移、层分解、一致性正则化、双聚合

## 一句话总结
提出FDSE方法，将每层网络分解为域无关特征提取器（DFE，全局聚合增强共识）和域特异偏移消除器（DSE，个性化聚合保留本地特性），结合BN一致性正则化，在DomainNet上达到76.77%（超Ditto 1.6%），在Office-Caltech10上达到91.58%（超FedBN 4.6%）。

## 研究背景与动机

**领域现状**：联邦学习面临域偏移导致的特征空间不对齐——不同客户端的数据分布差异使得全局模型对某些客户端不适应。现有方法要么增强全局共识（FedAvg系列）、要么增强个性化（FedBN等），但很少同时兼顾。

**现有痛点**：FedAvg类方法通过权重平均促进共识但忽略个性化需求；FedBN类方法保留本地BN但全局共享层仍受域偏移干扰。两种策略的优势没有在细粒度层面被统一。

**核心矛盾**：联邦学习需要全局共识（跨客户端学习共享知识）和本地个性化（适应本地数据分布）的平衡，但现有方法在层级别上只能做单一选择。

**本文目标** 在每一层内部同时解耦和优化域无关特征（促进共识）和域特异特征（保留个性化）。

**切入角度**：将每层卷积分解为两个子模块——DFE提取域不变特征（通道数$\lceil T/G \rceil$，全局聚合），DSE用廉价1×1卷积捕获域特异偏移（扩展到$T$通道，个性化聚合）。

**核心 idea**：在每层内部将域无关和域特异特征解耦到两个子模块中，分别用全局共识聚合和相似性感知个性化聚合策略更新。

## 方法详解

### 整体框架
每层网络分为DFE（轻量主干，提取共享特征）和DSE（1×1卷积，消除域偏移）。训练时两者联合优化但分开聚合——DFE用FedAvg风格全局聚合，DSE用基于客户端间相似度的自注意力个性化聚合。BN一致性正则化将本地BN统计量拉向全局统计量。

### 关键设计

1. **层分解（DFE + DSE）**:

    - 功能：同层内分离域无关和域特异特征
    - 核心思路：DFE输出$\lceil T/G \rceil$个通道（G为分组数），提取域不变的基础特征；DSE用1×1卷积将通道扩展到$T$，捕获域特异的偏移模式并"消除"它。两者级联：输入→DFE→DSE→输出。DFE参数量占比由$G$控制，$G$越大DSE越轻量
    - 设计动机：1×1卷积参数极少（"cheap operation"），DSE用最少参数就能编码域特异信息，大部分参数留给可全局共享的DFE

2. **双聚合策略**:

    - 功能：DFE最大化全局共识，DSE保留本地个性化
    - 核心思路：DFE用FedAvg聚合（$L_2$范数最小化公平共识）；DSE用相似性感知自注意力聚合——计算客户端间DSE参数的余弦相似度，相似客户端的DSE互相学习更多。温度参数$\tau$控制个性化程度
    - 设计动机：域无关特征应该完全共享（FedAvg最优）；域特异特征不应简单平均（会丢失个性化），而应让相似域的客户端互相学习

3. **BN一致性正则化**:

    - 功能：减少本地BN与全局BN统计量的偏差
    - 核心思路：正则化损失$\lambda \sum_l (\|\mu_l^{local} - \mu_l^{global}\|^2 + \|\sigma_l^{local} - \sigma_l^{global}\|^2)$，权重按层指数衰减$\beta=0.001$。拉近本地BN统计量到全局，但不完全对齐（保留个性化空间）
    - 设计动机：BN统计量直接反映数据分布，轻微对齐可以减少域偏移但过度对齐会损害个性化

### 损失函数 / 训练策略
总损失 = 任务交叉熵 + $\lambda \cdot$ BN一致性正则化。500轮通信，学习率每轮衰减0.998。DomainNet/PACS用5个本地epoch，Office-Caltech10用1个。

## 实验关键数据

### 主实验

| 数据集 | FDSE (All/Avg) | Ditto | FedBN | FedAvg |
|--------|---------------|-------|-------|--------|
| DomainNet | 76.77/74.50 | 75.18/72.82 | 74.75/- | 69.17/- |
| Office-Caltech10 | 87.15/91.58 | -/- | 83.08/87.01 | -/- |
| PACS | 83.81/82.17 | 82.02/80.03 | -/- | -/- |

FDSE在所有数据集上一致最优，在DomainNet上超越20+方法。

### 关键发现
- T-SNE可视化确认FDSE的特征空间类别分离度更好、域对齐更好
- 在几乎所有客户端上都有提升（蜘蛛图），说明不是以牺牲某些客户端为代价
- 收敛速度虽不是最快，但最终精度最高

## 亮点与洞察
- **层内解耦而非层间分配**：之前方法在层级别决定"这层共享还是个性化"，FDSE在每层内部同时做两件事，粒度更细
- **1×1卷积作为域偏移消除器**：极其轻量的参数量就能捕获域特异信息，大部分参数保留给可共享的DFE

## 局限与展望
- 分组参数$G$需要手动设定
- BN一致性权重$\lambda$对性能敏感
- 仅在CV分类任务上验证

## 相关工作与启发
- **vs FedBN**: FedBN只本地化BN层；FDSE在每层都做解耦，更彻底
- **vs Ditto**: Ditto用全局模型正则化本地模型但不做层内分解；FDSE更精细

## 评分
- 新颖性: ⭐⭐⭐⭐ 层内解耦的想法新颖，DFE+DSE设计简洁
- 实验充分度: ⭐⭐⭐⭐ 20+基线、3个数据集、可视化分析
- 写作质量: ⭐⭐⭐⭐ 动机和方法阐述清晰
- 价值: ⭐⭐⭐⭐ 对联邦域泛化有实质推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](../../CVPR2026/optimization/fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)
- [\[CVPR 2025\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[CVPR 2025\] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)
- [\[CVPR 2026\] HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation](../../CVPR2026/optimization/hfedatm_hierarchical_federated_domain_generalization_via_optimal_transport_and_r.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](../../NeurIPS2025/optimization/streaming_federated_learning_with_markovian_data.md)

</div>

<!-- RELATED:END -->
