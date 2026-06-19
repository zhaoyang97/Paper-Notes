---
title: >-
  [论文解读] Spikingformer: A Key Foundation Model for Spiking Neural Networks
description: >-
  [AAAI 2026 Oral][自监督学习][Spiking Neural Networks] 提出 Spikingformer，通过将 MS Residual 与 Self-Attention 以 spike-driven 方式结合，解决 Spikformer 中 SEW Residual 导致的非脉冲计算问题，同时保持全局建模能力。
tags:
  - "AAAI 2026 Oral"
  - "自监督学习"
  - "Spiking Neural Networks"
  - "Transformer"
  - "spike-driven"
  - "residual connection"
  - "energy-efficient AI"
---

# Spikingformer: A Key Foundation Model for Spiking Neural Networks

**会议**: AAAI 2026 Oral  
**arXiv**: [2304.11954](https://arxiv.org/abs/2304.11954)  
**代码**: [GitHub](https://github.com/TheBrainLab/Spikingformer)  
**领域**: 自监督  
**关键词**: Spiking Neural Networks, Transformer, spike-driven, residual connection, energy-efficient AI  

## 一句话总结

提出 Spikingformer，通过将 MS Residual 与 Self-Attention 以 spike-driven 方式结合，解决 Spikformer 中 SEW Residual 导致的非脉冲计算问题，同时保持全局建模能力。

## 背景与动机

### 领域现状

**领域现状**：SNN 的核心优势在于事件驱动的脉冲计算：用低功耗的累加 (AC, 0.9pJ) 替代高功耗的乘加 (MAC, 4.6pJ)。然而现有 SNN backbone 存在关键矛盾：

### 现有痛点

**现有痛点**：SEW ResNet / Spikformer** 使用 SEW Residual 连接，残差加法后输出范围为 $\{0,1,2,...,16\}$，下一层卷积变成整数-浮点乘法，破坏 spike-driven 特性

### 核心矛盾

**核心矛盾**：SD-Transformer** 使用 MS Residual + 线性注意力，保持 spike-driven 但失去全局建模能力

### 解决思路

**本文目标**：如何在保持**全局自注意力建模**能力的同时，确保整个网络的**纯 spike-driven**计算（仅累加操作）？

## 方法详解

### 整体框架

Spikingformer = Spiking Tokenizer + L 个 Spiking Transformer Block + Classification Head

### 关键设计

**MS Residual（核心改动）**：将 SN 层移到 ConvBN 之前：
$$O_l = \text{ConvBN}_l(\text{SN}_l(O_{l-1})) + O_{l-1}$$

对比 SEW Residual：$O_l = \text{SN}_l(\text{ConvBN}_l(O_{l-1})) + O_{l-1}$

MS Residual 中 SN 层确保输入 ConvBN 的数据为纯 spike（0/1），ConvBN 仅执行浮点加法；残差加法后的浮点值在下次进入 ConvBN 前会再经 SN 层转为 spike。

**Pre-activation SSA (PSSA)**：调整 Spikformer SSA 中 SN 位置，先 SN 再 ConvBN：
$$Q = \text{SN}_Q(\text{ConvBN}_Q(\text{SN}(X))), \quad K, V \text{ 同理}$$
$$\text{Attention}(Q,K,V) = \text{ConvBN}(\text{SN}(QK^TV \cdot s))$$

其中 $Q, K, V \in \{0,1\}^{T \times N \times D}$ 为纯 spike 数据，$s$ 为缩放因子。

**Spiking Tokenizer**：多级 ConvBN-SN 结构，支持可选 MaxPooling 下采样。

**Spikingformer†**：使用 CML (ConvBN-MaxPool-LIF) 下采样的增强变体，改善梯度反传。

### 能耗计算

$$E_{SNN} = E_{AC} \times \Big(\sum_{i=2}^N SOP_{Conv}^i + \sum_{j=1}^M SOP_{SSA}^j\Big) + E_{MAC} \times FLOP_{Conv}^1$$

其中 $SOP^l = fr \times T \times FLOPs^l$（fr 为发放率）。仅第一层编码非脉冲输入需 MAC。

## 实验关键数据


### 主实验

| 模型 | 参数量 | 时间步 | ImageNet Top-1 | 能耗 (mJ) |
|------|--------|--------|---------------|----------|
| Spikformer-8-768 | 66.34M | 4 | 74.81% | 32.07 |
| SD-Transformer-8-768 | 66.34M | 4 | 77.07% | 6.09 |
| Spikingformer-8-768 | 66.34M | 4 | 75.85% | 13.68 |
| **Spikingformer†-8-768** | 66.34M | 4 | **77.64%** | 16.30 |
| ANN Transformer-8-512 | 29.68M | 1 | 80.80% | 38.34 |

- CIFAR-10: 95.95% (Spikingformer†-9.32M, T=4)
- CIFAR-100: 80.37% (Spikingformer†-9.32M, T=4)
- DVS128 Gesture: 98.6% (T=16)
- 在 13 个数据集上全面验证

## 亮点与洞察

- 系统分析了 SEW vs MS Residual 的 spike-driven 行为，揭示 SEW Residual 的非脉冲范围随深度线性增长
- 同时实现 spike-driven + 全局注意力的唯一 SNN backbone
- 能耗仅为 ANN Transformer 的 ~42%（16.30 vs 38.34 mJ），且精度差距收窄到 3.16%
- CML 下采样变体 Spikingformer† 进一步提升性能

## 局限与展望

- 与 ANN Transformer 的精度差距仍有 ~3%（77.64 vs 80.80）
- 能耗计算基于 45nm 理论估算，实际神经形态硬件部署效果未验证
- 全局注意力的 $QK^TV$ 仍需浮点缩放因子 $s$，不完全 spike-driven
- 分类头的 AvgPooling-FC 仍含浮点运算

## 相关工作与启发

| 方法 | Spike-Driven | 全局注意力 | ImageNet Acc |
|------|-------------|-----------|-------------|
| SEW ResNet-152 | ✗ | ✗ | 69.26% |
| MS-ResNet-104 | ✓ | ✗ | 76.02% |
| Spikformer | ✗ | ✓ | 74.81% |
| SD-Transformer | ✓ | ✗ | 77.07% |
| **Spikingformer†** | **✓** | **✓** | **77.64%** |

## 启发

- "SN 层位置调整"这一简洁设计就解决了非脉冲计算问题，说明架构中算子顺序的重要性
- 为 SNN 社区提供了重要的实验 benchmark（13 个数据集），推动标准化评估
- spike-driven + 全局建模的结合为 SNN 在更多通用任务上的部署铺平道路

## 评分

⭐⭐⭐⭐ — 工程贡献扎实，设计简洁有效，实验全面，但核心创新（调整 SN 位置）相对直接

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Robust Spiking Neural Networks by Temporal Mutual Information](../../CVPR2026/self_supervised/robust_spiking_neural_networks_by_temporal_mutual_information.md)
- [\[CVPR 2026\] On the Role of Temporal Granularity in the Robustness of Spiking Neural Networks](../../CVPR2026/self_supervised/on_the_role_of_temporal_granularity_in_the_robustness_of_spiking_neural_networks.md)
- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](../../ICML2026/self_supervised/how_neural_is_a_neural_foundation_model.md)
- [\[ICLR 2026\] Maximizing Asynchronicity in Event-based Neural Networks](../../ICLR2026/self_supervised/maximizing_asynchronicity_in_event-based_neural_networks.md)
- [\[AAAI 2026\] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)

</div>

<!-- RELATED:END -->
