---
title: "Condensing Action Segmentation Datasets via Generative Network Inversion"
description: "提出通过生成网络反演进行时序动作分割数据集蒸馏，使用TCA生成模型和潜在码优化实现极致压缩比"
tags:
  - CVPR2025
  - Segmentation
  - Dataset Condensation
  - Action Segmentation
  - Network Inversion
  - 图像分割
---

# Condensing Action Segmentation Datasets via Generative Network Inversion

**会议**: CVPR 2025  
**机构**: National University of Singapore (NUS)  
**主题**: 数据集蒸馏 / 时序动作分割  

## 研究背景与动机

### 领域现状

**领域现状**：时序动作分割（Temporal Action Segmentation）旨在将长视频序列中的每一帧分配到对应的动作类别，是视频理解中的重要任务。然而，训练高性能的动作分割模型需要大量的长视频数据和密集的逐帧标注，数据的存储和传输成本巨大。

**核心痛点：**

**数据规模庞大**：动作分割数据集通常包含长时间的连续视频。例如Breakfast数据集包含约28GB的特征数据，50Salads数据集也有约4.5GB

**标注成本极高**：需要对视频中每一帧进行动作类别标注，这比图像分类的标注要昂贵得多

**隐私与分发问题**：在某些领域（如医疗、工业），原始视频数据可能涉及敏感信息，无法自由分发

**现有蒸馏方法的局限**：图像级别的数据集蒸馏方法（如Dataset Distillation）难以直接应用于时序数据，因为动作分割涉及长序列的时间结构和动作转换模式

**时序一致性**：蒸馏后的数据需要保持动作的时序结构，简单的帧级采样会破坏动作的连续性和转换模式

本文的关键洞察是：**可以利用生成模型来学习数据的分布，然后通过网络反演（Network Inversion）找到能够代表整个数据集的紧凑潜在表示**。

## 方法详解

### 整体框架

本文提出的方法包含两个核心阶段：（1）训练TCA生成模型学习数据分布；（2）通过网络反演优化潜在码，找到能够替代原始数据集的紧凑表示。

### TCA生成模型

TCA（Temporally Coherent Action）是专门为动作分割数据设计的生成模型，能够生成时序一致的动作序列。

**模型结构：**

生成器 $G$ 以潜在码 $z$、动作标签序列 $a$ 和条件信息 $c$ 为输入，生成对应的特征序列：

$$x_{\text{gen}} = G(z, a, c)$$

**时序一致性设计：**

TCA通过以下机制确保生成的动作序列在时间上是连贯的：

- 动作标签序列 $a$ 提供了帧级别的动作引导
- 条件信息 $c$ 编码了全局的动作转换模式
- 生成器内部使用时序卷积保持局部的时间连续性

### 网络反演与潜在码优化

核心步骤是通过优化潜在码来找到能够重建原始数据的表示：

$$z^* = \arg\min_{z} \| D(z, a, c) - x \|^2$$

其中 $D$ 是解码器，$x$ 是原始数据。优化后的潜在码 $z^*$ 比原始数据紧凑得多，存储每个样本只需要一个低维向量。

### 多样性序列采样

为了确保蒸馏后的数据集能够覆盖原始数据的多样性，本文设计了基于编辑距离和最远点采样的多样性选择策略：

| 策略 | 目标 | 方法 |
|------|------|------|
| 编辑距离 | 衡量动作序列之间的结构差异 | 计算动作标签序列间的编辑距离 |
| 最远点采样 | 在潜在空间中选择最分散的样本 | 贪心地选择与已选样本距离最远的点 |
| 类别平衡 | 确保每个动作类别都有足够的覆盖 | 按动作类别比例分配采样配额 |

### 蒸馏后的训练

使用蒸馏后的紧凑数据集训练动作分割模型：

1. 从存储的潜在码中随机采样
2. 通过冻结的生成器解码为特征序列
3. 使用解码的特征和对应的标签训练分割模型

## 实验结果

### 压缩比

| 数据集 | 原始大小 | 蒸馏大小 | 压缩比 |
|--------|----------|----------|--------|
| Breakfast | 28GB | 44MB | 636× |
| 50Salads | 4.5%GB | 3.9%MB | ~1150× |

### 分割性能

| 数据集 | 方法 | MS-TCN准确率 |
|--------|------|-------------|
| Breakfast | 蒸馏数据 | 55.5% |
| Breakfast | 完整数据 | 67.2% |
| 50Salads | 蒸馏数据 | 74.4% |
| 50Salads | 完整数据 | 80.6% |

虽然蒸馏后的数据在性能上有所损失，但考虑到数百倍的压缩比，这个性能保留率是令人印象深刻的。

### 消融实验

- TCA生成模型的时序一致性设计是关键，使用普通生成模型会导致生成的序列在动作转换处出现不自然的跳变
- 多样性采样策略显著优于随机采样，验证了覆盖数据分布多样性的重要性
- 潜在码的维度在精度和压缩比之间存在权衡

## 总结与展望

本文首次将数据集蒸馏技术应用于时序动作分割任务，通过TCA生成模型和网络反演实现了极致的数据压缩。在Breakfast和50Salads数据集上分别实现了636倍和约1150倍的压缩比，同时保留了可观的分割性能。该方法为数据高效的视频理解研究开辟了新方向，未来可以扩展到其他时序任务和更大规模的数据集。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](../../ICML2025/segmentation/actionpiece_contextually_tokenizing_action_sequences_generative_recommendation.md)
- [\[CVPR 2025\] Scale Efficient Training for Large Datasets](scale_efficient_training_for_large_datasets.md)
- [\[CVPR 2025\] Golden Cudgel Network for Real-Time Semantic Segmentation](golden_cudgel_network_for_real-time_semantic_segmentation.md)
- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)

</div>

<!-- RELATED:END -->
