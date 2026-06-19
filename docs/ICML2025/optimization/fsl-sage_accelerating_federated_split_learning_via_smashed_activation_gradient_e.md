---
title: >-
  [论文解读] FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation
description: >-
  [ICML 2025][优化/理论][联邦学习] 本文提出 FSL-SAGE，一种联邦分裂学习算法，通过辅助模型估计服务端梯度反馈，在保持与 FedAvg 相当的 $O(1/\sqrt{T})$ 收敛速率的同时，大幅降低通信开销和客户端内存需求。 领域现状： 联邦学习（FL）和分裂学习（SL）是两种主流的分布式隐私保护训练范…
tags:
  - "ICML 2025"
  - "优化/理论"
  - "联邦学习"
  - "split learning"
  - "gradient estimation"
  - "auxiliary model"
  - "communication efficiency"
---

# FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation

**会议**: ICML 2025  
**arXiv**: [2505.23182](https://arxiv.org/abs/2505.23182)  
**代码**: 无  
**领域**: 优化  
**关键词**: federated learning, split learning, gradient estimation, auxiliary model, communication efficiency

## 一句话总结
本文提出 FSL-SAGE，一种联邦分裂学习算法，通过辅助模型估计服务端梯度反馈，在保持与 FedAvg 相当的 $O(1/\sqrt{T})$ 收敛速率的同时，大幅降低通信开销和客户端内存需求。

## 研究背景与动机

**领域现状**: 联邦学习（FL）和分裂学习（SL）是两种主流的分布式隐私保护训练范式。FL（如 FedAvg）要求客户端训练完整模型，SL 将模型拆分为客户端和服务端两部分以减少客户端计算负担。

**现有痛点**: FL 要求客户端有足够内存存储和训练整个模型，对于大模型（如 LLM）不可行。SL 虽然减轻了客户端负担，但由于每个客户端必须按顺序与服务端交互（前向-反向传播），通信延迟随客户端数量线性增长。现有的联邦分裂学习（FSL）方法试图结合两者优点，但要么缺乏服务端反馈（用本地损失替代）导致精度下降，要么仍然存在通信瓶颈。

**核心矛盾**: 如何在保持低客户端内存需求的同时获得高质量的梯度信号？使用本地损失虽然避免了通信瓶颈，但梯度质量差；而等待服务端反馈虽然梯度准确，但通信效率低。

**本文目标**: 设计一种 FSL 算法，能够并行训练多个客户端，同时提供高质量的梯度估计。

**切入角度**: 在客户端侧引入轻量级辅助模型来模拟服务端模型的行为，定期从服务端同步辅助模型。

**核心 idea**: 用客户端本地的辅助模型估计服务端梯度反馈（smashed activation gradient），辅助模型通过定期同步来跟踪服务端模型的变化。

## 方法详解

### 整体框架
模型被拆分为两部分：客户端模型 $f_c$ 和服务端模型 $f_s$。正常的 SL 中，客户端前向传播得到 smashed activation $a = f_c(x)$，发送给服务端，服务端完成前向+反向传播并返回梯度 $\nabla_a \ell$。FSL-SAGE 用辅助模型 $\tilde{f}_s$ 替代这一交互，在客户端本地完成梯度估计。

输入：分布在 $K$ 个客户端上的本地数据 → 客户端前向传播 → 辅助模型估计梯度 → 客户端更新 → 定期同步辅助模型和聚合

### 关键设计

1. **辅助模型梯度估计（Auxiliary Model Gradient Estimation）**:

    - 功能：在客户端本地用辅助模型 $\tilde{f}_s$ 替代服务端模型 $f_s$，估计 smashed activation 的梯度
    - 核心思路：$\hat{g} = \nabla_a \ell(\tilde{f}_s(a), y)$ 作为真实梯度 $\nabla_a \ell(f_s(a), y)$ 的估计
    - 设计动机：避免客户端-服务端的逐一通信瓶颈。辅助模型只是服务端模型的近似副本，开销小但能提供有意义的梯度方向

2. **周期性辅助模型同步（Periodic Auxiliary Model Adaptation）**:

    - 功能：每隔 $\tau$ 轮通信，服务端将最新模型参数下发给所有客户端，更新辅助模型
    - 核心思路：同步周期 $\tau$ 控制了辅助模型与服务端模型的偏差——$\tau$ 越小，梯度估计越准但通信开销越大
    - 设计动机：辅助模型会随训练过程逐渐过时（stale），定期同步是控制估计偏差的关键。通过理论分析确定合理的同步频率

3. **并行客户端训练（Parallel Client Training）**:

    - 功能：所有客户端可以同时进行本地训练，无需等待服务端反馈
    - 核心思路：由于梯度估计完全在本地完成，客户端之间无依赖关系
    - 设计动机：这是 FSL-SAGE 相比传统 SL 在通信效率上的核心优势来源

### 损失函数 / 训练策略
标准的交叉熵损失。训练策略为：客户端本地进行多步梯度下降（使用辅助模型估计的梯度），然后将客户端模型参数发送到服务端进行聚合（类似 FedAvg），同时定期同步辅助模型。

## 实验关键数据

### 主实验

| 数据集/模型 | 指标 (Top-1 Acc%) | FSL-SAGE | SplitFed | FedAvg | LocalLoss-FSL |
|------------|-------------------|----------|----------|--------|--------------|
| CIFAR-10 / ResNet-18 | 测试准确率 | **92.4%** | 90.1% | 93.1% | 88.7% |
| CIFAR-100 / ResNet-34 | 测试准确率 | **71.8%** | 67.3% | 73.2% | 63.5% |
| Tiny-ImageNet / VGG-16 | 测试准确率 | **58.6%** | 53.9% | 60.1% | 49.8% |

### 消融实验

| 配置 | CIFAR-100 准确率 | 通信量 (相对 FedAvg) | 说明 |
|------|----------------|---------------------|------|
| FSL-SAGE ($\tau$=5) | **71.8%** | 0.3x | 最佳平衡点 |
| FSL-SAGE ($\tau$=1) | 72.5% | 0.8x | 频繁同步，准确率略高但通信增加 |
| FSL-SAGE ($\tau$=20) | 68.2% | 0.15x | 同步过稀，辅助模型过时 |
| 无辅助模型 (本地损失) | 63.5% | 0.1x | 梯度质量差 |

### 关键发现
- FSL-SAGE 达到 $O(1/\sqrt{T})$ 收敛速率，与 FedAvg 一致
- 在通信量仅为 FedAvg 30% 的情况下，准确率差距控制在 1-2% 以内
- 辅助模型的同步频率 $\tau$ 是关键超参数，5-10 轮同步一次是合理的选择
- 相比基于本地损失的方法，FSL-SAGE 的准确率提升 5-8 个百分点

## 亮点与洞察
- **理论与实践结合良好**: 不仅有 $O(1/\sqrt{T})$ 的收敛证明，实验也验证了理论预测
- **实用价值高**: 对于资源受限设备（如移动设备、IoT）上的大模型分布式训练有实际意义
- **辅助模型思路**可推广到其他需要近似反馈的分布式优化场景

## 局限与展望
- 辅助模型本身也需要内存开销，对于超大模型可能仍然是瓶颈
- 非 IID 数据分布下辅助模型的估计偏差可能更大
- 目前仅在 CV 任务上验证，NLP/LLM 场景有待探索
- 辅助模型同步频率的自适应调节策略值得研究

## 相关工作与启发
- FedAvg（McMahan et al., 2017）: 联邦学习基线
- SplitFed（Thapa et al., 2022）: 联邦分裂学习
- 辅助模型估计梯度的思路类似于知识蒸馏的反向应用

## 个人思考
- 辅助模型的思路本质上是用本地近似替代远程精确计算，这种 trade-off 在边缘计算中普遍存在
- 同步频率 $\tau$ 与估计质量之间的关系可以通过 bias-variance 分解更精确地刻画
- 可以考虑自适应同步策略——当辅助模型与服务端偏差较大时触发同步
- FSL-SAGE 与模型蒸馏的结合也是一个有趣的方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 辅助模型梯度估计是有意义的创新
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多基线比较，消融充分
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 对资源受限的联邦学习场景有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[ICML 2025\] A Unified View on Learning Unnormalized Distributions via Noise-Contrastive Estimation](a_unified_view_on_learning_unnormalized_distributions_via_noise-contrastive_esti.md)
- [\[ICML 2025\] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](the_panaceas_for_improving_low-rank_decomposition_in_communication-efficient_fed.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](../../NeurIPS2025/optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)

</div>

<!-- RELATED:END -->
