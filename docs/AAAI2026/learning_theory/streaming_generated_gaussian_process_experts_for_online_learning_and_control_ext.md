---
title: >-
  [论文解读] Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version
description: >-
  [AAAI 2026][机器学习 / 在线学习][高斯过程] 提出 SkyGP（Streaming Kernel-induced Progressively Generated Expert GP），通过**核距离驱动的渐进式专家生成**和**时间感知可配置聚合**处理流数据，继承精确 GP 的学习保证同时保持有界计算复杂度，在基准测试和实时控制实验中全面超越 SOTA。
tags:
  - "AAAI 2026"
  - "机器学习 / 在线学习"
  - "高斯过程"
  - "在线学习"
  - "流数据"
  - "混合专家"
  - "机器人控制"
---

# Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version

**会议**: AAAI 2026  
**arXiv**: [2508.03679](https://arxiv.org/abs/2508.03679)  
**代码**: [https://github.com/Zewen-Yang/SkyGP](https://github.com/Zewen-Yang/SkyGP)  
**领域**: 机器学习 / 在线学习  
**关键词**: 高斯过程, 在线学习, 流数据, 混合专家, 机器人控制

## 一句话总结

提出 SkyGP（Streaming Kernel-induced Progressively Generated Expert GP），通过**核距离驱动的渐进式专家生成**和**时间感知可配置聚合**处理流数据，继承精确 GP 的学习保证同时保持有界计算复杂度，在基准测试和实时控制实验中全面超越 SOTA。

## 研究背景与动机

1. **领域现状**：高斯过程（GP）作为非参数方法提供灵活建模和校准不确定性量化，支持通过多项式时间在线更新，非常适合安全关键系统。
2. **现有痛点**：精确 GP 处理流数据时推理更新为 $O(N^3)$ 时间和 $O(N^2)$ 内存，随数据增长不可扩展。现有解决方案包括稀疏 GP（需昂贵优化、丢失误差保证）和分布式 GP（如 LoG-GP 仅沿单一维度分割、不处理非平稳性）。
3. **核心矛盾**：在线学习需要快速适应和有界复杂度，但精确 GP 的预测性能保证需要完整数据矩阵。现有分布式方法要么忽略在线学习需求，要么分割策略缺乏空间/时间相关性利用。
4. **本文目标**：设计一个流 GP 框架，能动态管理有界数量的专家集合，同时继承精确 GP 的预测误差界。
5. **切入角度**：基于核函数距离决定新数据是加入现有专家还是初始化新专家，结合时间衰减因子管理专家的新旧程度。
6. **核心 idea**：核距离驱动的自适应专家分配 + 时间感知聚合 = 有界复杂度且保留精确 GP 性能保证的在线学习。

## 方法详解

### 整体框架

当新数据点 $(x^k, y^k)$ 到达时：(1) 通过核距离在自适应窗口内搜索最近专家；(2) 如果最近专家未满则加入（SkyGP-Fast: rank-1 Cholesky 更新），如果已满则根据变体选择替换或创建新专家（SkyGP-Dense: 数据替换）；(3) 预测时选择 $\bar{N}$ 个最近专家，用 MoE/PoE/BCM 聚合预测。

### 关键设计

1. **核距离驱动的专家定位与生成**

    - 功能：自适应地将流数据分配到最合适的专家，或在需要时创建新专家。
    - 核心思路：每个专家 $\mathcal{GP}_i$ 维护中心 $c_i$（增量更新 $c_i^k = (k-1)c_i^{k-1}/k + x^k/k$），核距离 $d_i^k = 1/\kappa(c_i^k, x^k)$。在自适应窗口 $W = \min(\bar{W}, \lfloor\exp(d_{temp}/\varrho)\rfloor)$ 内搜索最近专家。若最近专家已满且数据与之前被丢弃数据分布更匹配（$\Delta < 0$），则执行数据替换（SkyGP-Dense）；否则创建新专家。专家按中心位置维护有序列表，新专家插入最近专家的邻居位置。
    - 设计动机：传统方法将数据简单分配给第一个未满专家，不考虑数据分布特性。核距离确保专家局部性——相似上下文的数据被同一专家处理。

2. **时间感知聚合框架**

    - 功能：管理专家的新旧程度，确保非平稳环境下的预测质量。
    - 核心思路：每个专家维护时间感知因子 $\vartheta \in (0, 1]$，被查询时重置为 1，未被查询时以速率 $\rho$ 衰减。聚合时仅选择 $\vartheta > \bar{\vartheta}$ 的活跃专家。支持 MoE（$\omega_i = w_i$）、PoE（$\omega_i = w_i \sigma_i^2 / \varpi_i$）和 BCM（加入先验方差 $\sigma_*$）三种聚合策略。
    - 设计动机：老旧专家可能代表过时的数据分布，对非平稳环境的预测有害。时间衰减自然地淘汰过时专家。

3. **两种变体：SkyGP-Dense vs SkyGP-Fast**

    - SkyGP-Dense：已满专家通过数据替换策略保持局部代表性，需要完整 Cholesky 重计算（$O(\bar{N}^3)$ per 专家），适合内存受限场景。
    - SkyGP-Fast：不做替换，直接创建新专家，rank-1 Cholesky 更新（$O(\bar{N}^2)$），适合低延迟场景。

### 损失函数 / 训练策略

无传统训练。GP 通过 Cholesky 分解直接推理。控制任务中提供基于学习的控制策略，利用 GP 不确定性设计安全反馈控制器。有界预测误差 $|f(x) - \tilde{\mu}| \leq \beta\sigma(x) + \gamma(x)$ 的理论保证。

## 实验关键数据

### 主实验

在线回归基准（RMSE↓）和实时控制实验的结果：

| 方法 | 在线回归精度 | 推理时间 | 内存 | 控制误差 |
|------|------------|---------|------|---------|
| 精确 GP | 最优但不可扩展 | $O(N^3)$ | $O(N^2)$ | - |
| SSGP | 中等 | 高 | 中等 | - |
| LoG-GP | 中等 | 中等 | 有界 | 中等 |
| **SkyGP-Dense** | **接近精确 GP** | **有界** | **有界** | **最优** |
| **SkyGP-Fast** | 中等偏优 | **最快** | 有界 | 优 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| MoE vs PoE vs BCM 聚合 | BCM 通常最优 | 引入先验方差校准不确定性 |
| 时间感知 vs 无时间感知 | 时间感知更好 | 非平稳数据中老旧专家有害 |
| 事件触发替换 vs 总是替换 | 事件触发更好 | 减少不必要的 Cholesky 重计算 |
| 窗口大小 W | 自适应最优 | 固定窗口不适应数据密度变化 |

### 关键发现

- SkyGP-Dense 在预测精度上接近精确 GP 但复杂度有界——验证了"不牺牲性能的可扩展性"。
- SkyGP-Fast 的推理时间最少，适合实时控制场景。
- 时间感知因子对非平稳数据至关重要——移除后性能显著下降。
- 在机器人控制实验中，SkyGP 的不确定性估计直接驱动安全控制策略，实现了在线适应且安全的轨迹跟踪。
- 理论保证（Lemma 1）在实验中得到验证——预测误差确实落在 $\beta\sigma(x) + \gamma(x)$ 界内。

## 亮点与洞察

- **核距离驱动的智能数据分配**：不是简单的先来先服务或随机分割，而是基于核空间相似度自适应分配，确保每个专家的局部一致性。
- **理论保证的继承**：从精确 GP 的误差界推导到 SkyGP 的聚合误差界，提供了分布式 GP 少有的理论保障。
- **Dense/Fast 双模式设计**：一个框架两种部署模式，灵活适应不同场景的计算-内存权衡需求。

## 局限与展望

- 核函数超参数（lengthscale 等）假设预先确定或全局优化，未处理在线超参数自适应。
- 专家数量无上界限制——极端非平稳环境可能创建过多专家。
- 仅验证了单输出 GP，多输出 GP（如向量值函数）扩展未讨论。
- 实时控制实验仅在 Euler-Lagrange 系统上验证。

## 相关工作与启发

- **vs LoG-GP**：仅沿单一维度分割，高维空间中不够灵活；SkyGP 基于核距离在多维中自适应分配。
- **vs SSGP**：每步需昂贵优化且无误差保证；SkyGP 通过 rank-1 更新保持低延迟且有理论保证。
- **vs 稀疏 GP (FITC/VFE)**：需要全局训练且引导点选择非平凡；SkyGP 全在线、无需全局重训。
- 可推广到多机器人协同学习——每个机器人维护本地 GP 专家，通过通信网络共享。

## 评分

- 新颖性: ⭐⭐⭐⭐ 核距离驱动的渐进式专家生成+时间感知聚合是有创新的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 基准回归+实时控制+理论验证全面
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，算法描述清晰
- 价值: ⭐⭐⭐⭐ 对安全关键系统的在线学习有实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Online Learning with Recency: Algorithms for Sliding-window Streaming Multi-armed Bandits](../../ICML2026/learning_theory/online_learning_with_recency_algorithms_for_sliding-window_streaming_multi-armed.md)
- [\[ICLR 2026\] Transformers as Unsupervised Learning Algorithms: A study on Gaussian Mixtures](../../ICLR2026/learning_theory/transformers_as_unsupervised_learning_algorithms_a_study_on_gaussian_mixtures.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICML 2026\] Robustness of Mixtures of Experts to Feature Noise](../../ICML2026/learning_theory/robustness_of_mixtures_of_experts_to_feature_noise.md)

</div>

<!-- RELATED:END -->
