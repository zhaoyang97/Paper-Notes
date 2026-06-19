---
title: >-
  [论文解读] Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful
description: >-
  [NeurIPS 2025][优化/理论][小批量训练] 本文系统研究了小批量（甚至batch size=1）在语言模型预训练和微调中的表现，提出了基于"token半衰期"固定的Adam β₂缩放规则，发现小批量不仅训练稳定，还使vanilla SGD具备与自适应优化器相当的竞争力，并建议避免使用梯度累积。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "小批量训练"
  - "SGD"
  - "Adam"
  - "梯度累积"
  - "语言模型"
---

# Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful

**会议**: NeurIPS 2025  
**arXiv**: [2507.07101](https://arxiv.org/abs/2507.07101)  
**代码**: [https://github.com/martin-marek/batch-size](https://github.com/martin-marek/batch-size)  
**领域**: 优化  
**关键词**: 小批量训练, SGD, Adam, 梯度累积, 语言模型

## 一句话总结
本文系统研究了小批量（甚至batch size=1）在语言模型预训练和微调中的表现，提出了基于"token半衰期"固定的Adam β₂缩放规则，发现小批量不仅训练稳定，还使vanilla SGD具备与自适应优化器相当的竞争力，并建议避免使用梯度累积。

## 研究背景与动机
语言模型训练中，大批量被广泛认为是稳定训练的必要条件。当GPU显存不足以支撑大批量时，研究者通常采用梯度累积来模拟更大的batch size。然而，现有工作在减小批量时通常只调整学习率，而保持Adam中的动量衰减率β₁和β₂不变。本文指出，这种做法是小批量训练不稳定的根本原因。如果我们不是固定β₂本身，而是固定β₂的"半衰期"（以token数计量），小批量训练甚至可以到batch size=1的极端情况仍然稳定。核心insight：小批量使得优化器每步的预测距离更短，因此不需要复杂的优化器或精细的超参数调优。

## 方法详解

### 整体框架
本文的方法论核心并非一个新的优化器架构，而是一套关于如何正确缩放优化器超参数到小批量的理论和实践指南。主要包括：(1) 提出动量半衰期（moment half-life）的概念来替代直接使用β值；(2) 提出β₂的缩放启发式规则；(3) 系统性地验证小批量在多种优化器和模型规模下的优势。

### 关键设计

1. **动量半衰期（Moment Half-Life）**：不直接操作β₁和β₂，而是定义一个以token数量为单位的衰减半衰期t₁/₂。在Adam中，每次更新步骤中过去梯度的贡献按β衰减，半衰期表示梯度贡献衰减到一半所需的token数。这个量与batch size直接关联：β^(t₁/₂ / (B·T)) = 1/2。通过固定半衰期而非固定β值，可以在不同batch size之间自然地转换超参数。

2. **β₂缩放规则**：当batch size从B缩放到B*时，新的β₂*可以通过保持token半衰期t₂不变来计算：β₂* = β₂^(B*/B)。例如，从batch size 512的默认β₂=0.95缩放到batch size 1时，β₂应设为0.9999。这个简单的one-shot启发式规则无需额外调参即可在不同设置间迁移。

3. **小批量+简单优化器的组合策略**：在小批量下，所有优化器（SGD、Adam、Adafactor、Muon）表现相近，SGD甚至不需要动量也能竞争。这是因为小步长意味着优化器不需要对远处的损失面做预测，降低了对优化器复杂性的要求。

### 训练策略
- 学习率调度：前5%步线性warmup从0到峰值，随后cosine衰减到0
- β₁=0.9在所有batch size下均表现良好，无需缩放
- 小批量使用Adafactor替代Adam可大幅减少显存：Adafactor仅存储逐行和逐列的二阶矩，内存开销从O(d₁×d₂)降至O(d₁+d₂)
- 使用随机舍入（stochastic rounding）使bfloat16权重在小批量下也能正常工作

## 实验关键数据

### 主实验：不同优化器在不同batch size下的表现（30M模型，600M tokens）

| 优化器 | Batch=1最优loss | Batch=4096最优loss | 小批量是否优于大批量 |
|--------|----------------|-------------------|---------------------|
| SGD | ~3.95 | ~4.10 | ✅ 是 |
| Adafactor | ~3.95 | ~4.00 | ✅ 是 |
| Adam | ~3.95 | ~3.97 | ✅ 是 |
| Muon | ~3.95 | ~3.96 | ✅ 是 |

### 大规模验证：GPT-2 (124M) 和 GPT-3 (1.3B)

| 模型 | 优化器 | Batch Size | 验证Loss |
|------|--------|-----------|----------|
| GPT-3 (1.3B) | AdamW (baseline) | 512 | 基线 |
| GPT-3 (1.3B) | Adam (固定t₂) | 1 | 优于基线 |
| GPT-3 (1.3B) | Adafactor | 1 | 优于基线 |
| GPT-3 (1.3B) | Vanilla SGD | 1 | 与基线持平 |
| GPT-2 (124M) | AdamW (tuned) | 512 | 基线 |
| GPT-2 (124M) | Adam | 1 | 与基线持平 |
| GPT-2 (124M) | SGD | 1 | 略低于基线 |

### 微调实验：Gemma 3 (4B) 在MATH数据集

| 方法 | Batch Size | 内存/参数 | MATH准确率 |
|------|-----------|----------|-----------|
| Adam (full FT) | 16 (梯度累积) | 16 bytes | 基线 |
| LoRA + Adam | 16 | ~2 bytes | 低于full FT |
| Adafactor (full FT) | 1 | ~2 bytes | 与full FT持平 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 固定β₂ vs 固定t₂ | 验证loss | 固定t₂在所有batch size下表现一致，固定β₂在小批量时显著恶化 |
| 学习率√B缩放 vs 实际最优 | 最优学习率 | √B缩放过于激进，实际B从1到1024只需缩放约3倍而非32倍 |
| bfloat16 + 随机舍入 vs float32 | 验证loss | 随机舍入使bfloat16接近float32表现 |
| 有/无动量的SGD | 验证loss | 小批量下两者差距极小，大批量下动量必要 |

### 关键发现
- 所有优化器在batch size=1时表现最佳（per-FLOP效率）
- 小批量对超参数选择的鲁棒性远超大批量：batch=1在整个学习率和β₁搜索范围内几乎都是最优loss
- 固定β₂在小批量时导致训练不稳定，是之前文献中"小批量不work"的根本原因
- 在1.3B规模上，vanilla SGD（无动量、无weight decay）与AdamW基线持平
- 梯度累积不仅浪费计算步骤，还增加显存开销（需存储累积梯度）

## 亮点与洞察
- **小步长假说**：小批量+小学习率意味着优化器每步移动距离更短，不需要对远处损失面做预测，因此对优化器的复杂性要求更低
- **动量不必要之因**：大步长会在高曲率方向上overshoot导致振荡，需要动量来抑制；小步长不会overshoot，因此动量冗余
- **实际建议**：使用最小的、能最大化硬件throughput（tokens/秒）的batch size，而非尽可能大的batch size
- 该工作直接挑战了"大批量训练更好"和"SGD无法训练Transformer"的传统认知

## 局限与展望
- 小批量（batch=1或2）会导致MFU下降30-70%（计算利用率降低），实际应用中需要在稳定性和计算效率间平衡
- 学习率的缩放规则不如β₂清晰，学习率在不同batch size间的最优缩放比例没有找到简洁公式
- 未探索与batch size schedule的交互（训练中动态调整batch size）
- 未研究更低精度（如INT8/INT4）权重与小批量的组合效果
- 理论分析局限于compute-optimal前沿（Chinchilla范式），在convergent训练中是否也成立存疑

## 相关工作与启发
- 挑战了Xiao等人(2024)的结论：小批量Adam表现差是因为没调β₂，本文仅通过one-shot缩放就修复了
- 挑战了Zhao等人(2025)的结论：SGD远不如Adam，本文证明在小批量下SGD可以追平
- 对memory-efficient fine-tuning有启发：small batch + Adafactor可以替代LoRA，获得full FT性能但只用LoRA级别的显存
- 与data parallelism文献中的critical batch size概念呼应，但提供了更极端的实验验证

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](constrained_network_slice_assignment_via_llms.md)

</div>

<!-- RELATED:END -->
