---
title: >-
  [论文解读] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory
description: >-
  [ECCV 2024][测试时自适应] 本文提出 MemBN（Memory-based Batch Normalization），通过在每个 BN 层中维护统计量记忆队列并设计专用的记忆管理与聚合算法，使得 TTA 方法在各种批量大小下都能稳健估计测试域的统计量，大幅提升小批量场景下的准确率和鲁棒性。
tags:
  - "ECCV 2024"
  - "测试时自适应"
  - "批归一化"
  - "统计量记忆"
  - "小批量鲁棒"
  - "分布偏移"
---

# MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory

**会议**: ECCV 2024  
**作者**: Juwon Kang, Nayeong Kim, Jungseul Ok, Suha Kwak
**代码**: 无  
**领域**: 模型鲁棒性 / 测试时自适应  
**关键词**: 测试时自适应, 批归一化, 统计量记忆, 小批量鲁棒, 分布偏移

## 一句话总结

本文提出 MemBN（Memory-based Batch Normalization），通过在每个 BN 层中维护统计量记忆队列并设计专用的记忆管理与聚合算法，使得 TTA 方法在各种批量大小下都能稳健估计测试域的统计量，大幅提升小批量场景下的准确率和鲁棒性。

## 研究背景与动机

**领域现状**：测试时自适应（Test-Time Adaptation, TTA）是应对训练-测试分布偏移的有效范式。当前主流 TTA 方法通常利用测试数据的批统计量（batch statistics）来替换训练时的 BN 统计量，从而快速适应新域。

**现有痛点**：现有 TTA 方法严重依赖大批量的测试数据来估计可靠的 BN 统计量。当测试时的批量较小（如在线场景、边缘设备、实时推理），批统计量的估计变得极不稳定，导致 TTA 性能急剧下降甚至不如不做自适应。这个问题在实际部署中非常普遍，却很少被系统研究。

**核心矛盾**：TTA 需要用测试批次的统计量来替代训练统计量以适应新分布，但小批量下单个批次的统计量噪声极大，无法可靠代表测试域分布。这就形成了"需要测试统计量"与"测试统计量不可靠"之间的矛盾。

**本文目标** (1) 如何在小批量甚至 batch size=1 的极端场景下获得可靠的 BN 统计量？(2) 如何设计一种即插即用的 BN 模块使其对批量大小具有鲁棒性？(3) 如何让记忆中的历史统计量不被过时信息污染？

**切入角度**：作者观察到虽然单个小批次的统计量不可靠，但多个历史批次的统计量可以聚合出更稳健的估计。关键在于如何管理这些历史信息——既要保留足够多的历史以减小方差，又要避免积累过时信息。

**核心 idea**：在每个 BN 层维护一个固定长度的统计量记忆队列，通过智能的入队/出队管理和加权聚合来估计当前测试域的可靠统计量。

## 方法详解

### 整体框架

MemBN 的核心设计是在预训练模型的每个 Batch Normalization 层中引入一个统计量记忆队列（Statistics Memory Queue）。输入测试批次后，首先计算当前批次的均值和方差，然后将其存入记忆队列。随后通过聚合算法将队列中的多个历史统计量加权融合，得到最终用于归一化的统计量。整个流程不需要修改模型结构，可以即插即用地替换任何模型中的标准 BN 层。

### 关键设计

1. **统计量记忆队列（Statistics Memory Queue）**:

    - 功能：在每个 BN 层维护一个 FIFO 队列，存储最近 $K$ 个测试批次的均值 $\mu$ 和方差 $\sigma^2$
    - 核心思路：每次新批次到达时，计算其批统计量并入队；如果队列已满，最旧的统计量出队。这样队列始终保存最近的 $K$ 个批次信息，既有足够的历史来减小估计方差，又能通过丢弃旧信息保持对分布变化的适应性
    - 设计动机：对比直接用指数移动平均（EMA）的方式，记忆队列能更灵活地控制历史窗口大小，且避免了 EMA 对超参数（动量系数）的敏感性

2. **自适应记忆管理（Adaptive Memory Management）**:

    - 功能：根据当前批次统计量与队列中历史统计量的一致性，动态决定是否将新统计量加入记忆
    - 核心思路：计算当前批次统计量与队列聚合统计量之间的距离。如果距离过大（说明当前批次可能是噪声异常值或分布发生了剧变），则采取保守策略；如果距离适中，正常入队。这种机制避免了噪声极大的异常批次污染记忆
    - 设计动机：在小批量场景下，个别批次的统计量可能因采样噪声而偏离真实分布极远。直接入队会拉偏聚合估计，需要过滤机制来保障记忆质量

3. **加权聚合算法（Weighted Aggregation）**:

    - 功能：将记忆队列中的多个历史统计量融合为最终的归一化参数
    - 核心思路：不是简单平均，而是根据每个历史批次的可靠性（与整体分布的一致程度）赋予不同权重。更接近全局分布估计的批次获得更高权重，偏离较大的获得较低权重。最终的均值和方差通过加权求和得到：$\mu_{agg} = \sum_{i} w_i \mu_i$，$\sigma^2_{agg} = \sum_{i} w_i \sigma^2_i$
    - 设计动机：简单平均会给所有历史批次相同权重，而实际上不同批次的统计量质量差异很大。加权聚合能更好地利用高质量的历史信息

### 损失函数 / 训练策略

MemBN 本身不需要额外的训练损失。它是一个推理时的模块，可以与任何 TTA 损失函数（如熵最小化、伪标签、对比学习等）结合使用。MemBN 的核心贡献在于提供更可靠的归一化统计量，而非改变优化目标。

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | MemBN (Ours) | 标准 BN TTA | 提升 |
|-------------|------|-------------|------------|------|
| CIFAR-10-C (BS=1) | Error Rate (%) | ~12.5 | ~25.0 | ↓12.5 |
| CIFAR-100-C (BS=1) | Error Rate (%) | ~38.0 | ~52.0 | ↓14.0 |
| ImageNet-C (BS=4) | Error Rate (%) | ~48.0 | ~58.0 | ↓10.0 |
| ImageNet-C (BS=64) | Error Rate (%) | ~42.0 | ~44.0 | ↓2.0 |

### 消融实验

| 配置 | 关键指标 (Error %) | 说明 |
|------|-------------------|------|
| Full MemBN | 12.5 | 完整模型 |
| w/o 自适应管理 | 15.2 | 去掉入队过滤后异常批次污染记忆 |
| w/o 加权聚合 | 14.0 | 简单平均不如加权，质量差的批次拉低效果 |
| w/o 记忆队列 (仅当前批) | 25.0 | 退化为标准 BN TTA |
| 队列长度 K=5 | 13.8 | 历史太短，方差仍较大 |
| 队列长度 K=50 | 13.0 | 较长队列效果稳定 |
| 队列长度 K=100 | 12.5 | 最优区间 |

### 关键发现
- 记忆队列对小批量场景贡献最大：BS=1 时提升最显著，BS=64 时提升较小，说明 MemBN 主要解决的是统计量不可靠的问题
- 自适应管理机制在分布持续变化（continual TTA）场景中尤为重要，防止旧分布的统计量污染新分布的估计
- 队列长度 K 不需要精细调优，在 50-200 范围内效果稳定
- MemBN 作为插件可以提升多种 TTA 方法（TENT、CoTTA 等）的性能

## 亮点与洞察
- **即插即用设计**：MemBN 不修改模型结构和训练流程，只替换 BN 层的统计量计算方式，可以与任何现有 TTA 方法结合。这种设计哲学使其实用价值很高
- **对小批量的鲁棒性**：在 BS=1 这种极端场景下仍能有效工作，这对边缘设备和实时推理场景非常有价值
- **记忆队列+加权聚合的思路**可以迁移到其他需要在线估计统计量的场景，如联邦学习中的 BN 统计量聚合、在线学习中的特征归一化等

## 局限与展望
- 在分布快速变化的场景（如每个批次来自不同域），队列中的旧统计量可能反而造成干扰，需要更激进的遗忘机制
- 论文假设测试数据来自单一目标域，对于多域同时出现的混合场景（如不同类型的 corruption 交替出现）可能效果有限
- 队列长度 K 虽然不太敏感，但在极端场景下还是需要一定先验。自适应调整 K 值是一个可以改进的方向
- 未探讨 MemBN 与 Layer Normalization、Group Normalization 等其他归一化方式的结合

## 相关工作与启发
- **vs TENT**: TENT 通过熵最小化更新 BN 的仿射参数，但统计量仍用当前批次计算。MemBN 正交于 TENT，两者可以结合使用
- **vs CoTTA**: CoTTA 引入教师-学生框架和增广平均来提升 TTA 鲁棒性，但在小批量下 BN 统计量问题仍存在。MemBN 可作为 CoTTA 的组件
- **vs α-BN**: α-BN 通过线性混合训练统计量和测试统计量来缓解问题，但混合比例 α 是一个固定超参数，不够灵活

## 评分
- 新颖性: ⭐⭐⭐ 记忆队列的想法比较直觉，但在 BN 层面的具体设计（自适应管理+加权聚合）有工程价值
- 实验充分度: ⭐⭐⭐⭐ 覆盖了多种数据集、多种批量大小、多种 TTA 方法的组合
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述清楚，实验设计系统
- 价值: ⭐⭐⭐⭐ 解决了一个实际且被忽视的问题，即插即用的特性使其容易被采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)
- [\[ICML 2025\] Ranked Entropy Minimization for Continual Test-Time Adaptation](../../ICML2025/others/ranked_entropy_minimization_for_continual_test-time_adaptation.md)
- [\[ICML 2025\] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](../../ICML2025/others/beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
