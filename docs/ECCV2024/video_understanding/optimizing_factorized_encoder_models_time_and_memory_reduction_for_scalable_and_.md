---
title: >-
  [论文解读] Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition
description: >-
  [ECCV 2024][视频理解][Transformer] 本文通过冻结 ViViT 因子化编码器中的空间 Transformer 并引入合理的时间 Transformer 初始化策略和紧凑的适配器模块，在保持甚至略微提升精度的同时大幅降低了训练成本和内存消耗，为资源受限的研究者提供了更高效的动作识别训练方案。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "Transformer"
  - "ViViT"
  - "因子化编码器"
  - "训练效率"
  - "动作识别"
---

# Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 视频理解 / 动作识别  
**关键词**: 视频Transformer, ViViT, 因子化编码器, 训练效率, 动作识别

## 一句话总结
本文通过冻结 ViViT 因子化编码器中的空间 Transformer 并引入合理的时间 Transformer 初始化策略和紧凑的适配器模块，在保持甚至略微提升精度的同时大幅降低了训练成本和内存消耗，为资源受限的研究者提供了更高效的动作识别训练方案。

## 研究背景与动机

**领域现状**：视频 Transformer（如 ViViT）在动作识别任务上表现优异。ViViT 的因子化编码器（Factorised Encoder）变体采用后融合（late-fusion）策略，先用空间 Transformer 处理每帧的空间特征，再用时间 Transformer 对帧间时序关系进行建模。这种策略被许多 SOTA 方法采用，因为它在速度和精度之间提供了良好的权衡。

**现有痛点**：尽管因子化编码器在 ViViT 的不同变体中已经是效率最优的选择，其训练时间和内存消耗仍然非常大。空间 Transformer 需要对每一帧独立处理，帧数越多、模型越大，训练开销呈线性或超线性增长。这对于 GPU 内存有限或计算预算有限的研究者来说是一个显著的准入障碍。

**核心矛盾**：空间 Transformer 通常基于强大的图像预训练模型（如 ViT），其空间特征提取能力已经很强。在视频训练过程中继续更新空间 Transformer 的参数，不仅计算代价大，而且收益有限——视频理解的核心挑战更多在时序建模而非空间表示。然而，简单冻结空间 Transformer 会导致精度显著下降。

**本文目标** 如何在冻结空间 Transformer（大幅节省训练成本）的同时，不牺牲动作识别精度？

**切入角度**：作者观察到简单冻结失败的原因在于：（1）时间 Transformer 的初始化不合理，无法有效处理冻结空间特征；（2）冻结的空间表示缺乏面向时序任务的适配能力。只要解决这两个问题，冻结空间 Transformer 就是可行的。

**核心 idea**：通过合理的时间 Transformer 初始化和紧凑的空间-时序适配器，使冻结空间 Transformer 的因子化编码器在训练效率和精度上都优于传统全参数训练。

## 方法详解

### 整体框架
方法基于 ViViT 因子化编码器架构。输入视频被分成多帧，每帧经过冻结的空间 Transformer 提取空间 token 特征。这些空间特征通过一个轻量级适配器模块进行适配，然后输入到可训练的时间 Transformer 中进行时序建模，最终输出动作分类结果。训练时只更新适配器和时间 Transformer 的参数。

### 关键设计

1. **空间 Transformer 冻结策略**:

    - 功能：通过冻结空间 Transformer 的全部参数来大幅减少训练时间和内存
    - 核心思路：空间 Transformer 使用 ImageNet 预训练的 ViT 权重，在视频训练过程中完全冻结。由于不需要计算空间 Transformer 的梯度，反向传播的计算量和内存占用大幅减少。对于每帧独立处理的空间编码器，冻结后不仅省去了梯度计算，还可以将空间特征缓存复用
    - 设计动机：空间 Transformer 的参数量通常占模型总量的 50% 以上，冻结它可以直接减少一半以上的训练计算量。同时预训练的空间表示已经足够强，继续训练的边际收益很小

2. **时间 Transformer 初始化策略**:

    - 功能：为时间 Transformer 提供更好的初始化，使其能有效处理冻结空间特征
    - 核心思路：传统做法是随机初始化时间 Transformer，但这在空间 Transformer 被冻结时效果很差。本文探索了多种初始化方案，包括从空间 Transformer 权重中迁移初始化、使用 ImageNet 预训练权重直接初始化等。关键发现是用图像预训练权重初始化时间 Transformer，即使它要处理的是时序信息而非空间信息，也比随机初始化好得多，因为注意力机制的一般能力是可迁移的
    - 设计动机：冻结空间 Transformer 后，时间 Transformer 成为唯一可训练的主干，它需要在有限的训练周期内快速收敛。好的初始化可以加速收敛并提高最终性能

3. **紧凑适配器模块（Compact Adapter）**:

    - 功能：在冻结的空间表示和可训练的时间 Transformer 之间建立桥梁
    - 核心思路：在空间 Transformer 输出的 token 序列和时间 Transformer 输入之间插入一个轻量级适配器。适配器包含少量可训练参数（如线性投影 + LayerNorm），选择性地关注输入帧的不同空间区域，对冻结的空间特征进行任务适配性调整，使其更适合时序建模
    - 设计动机：冻结的空间特征是通用的图像表示，不一定最适合特定的动作识别任务。适配器以极小的参数开销（相对于全调空间 Transformer）提供了任务特定的表示调整能力

### 损失函数 / 训练策略
使用标准的交叉熵损失进行动作分类训练。训练时仅更新适配器和时间 Transformer 的参数，空间 Transformer 完全冻结。由于内存节省，可以在相同硬件条件下使用更大的空间 Transformer 模型或处理更多视频帧，进一步提升性能。

## 实验关键数据

### 主实验
在 6 个动作识别基准上进行评估：

| 数据集 | 指标 | 本文方法 | ViViT baseline | 提升 |
|--------|------|----------|---------------|------|
| Kinetics-400 | Top-1 Acc | +0.5~1.79% | baseline | 精度持平或提升 |
| Something-SomethingV2 | Top-1 Acc | 持平 | baseline | 保持精度 |
| Epic-Kitchens | Acc | 持平/略升 | baseline | 保持精度 |
| Moments in Time | Top-1 Acc | 持平 | baseline | 保持精度 |
| 训练时间 | GPU hours | 减少 ~43% | baseline | 训练时间大幅下降 |
| 内存消耗 | GB | 显著减少 | baseline | 可用更大模型/更多帧 |

### 消融实验

| 配置 | Top-1 Acc | 训练时间 | 说明 |
|------|-----------|----------|------|
| 全参数训练 (baseline) | 基准 | 100% | 标准ViViT训练 |
| 朴素冻结（随机初始化） | 明显下降 | ~57% | 精度损失过大，不可行 |
| 冻结 + 合理初始化 | 接近基准 | ~57% | 初始化策略关键 |
| 冻结 + 初始化 + 适配器 | 持平或超越 | ~57% | 适配器补足最后差距 |
| 使用更大空间模型 | 超越基准 | ~57-65% | 省出的内存换更大模型 |

### 关键发现
- 时间 Transformer 的初始化策略是冻结方案成功的关键因素——随机初始化导致精度下降超过 3%，而合理初始化几乎消除了差距
- 适配器虽然参数量很小，但提供了 0.5-1% 的精度提升，在冻结场景下是不可或缺的
- 冻结策略可以推广到其他因子化编码器模型，不仅限于 ViViT
- 省下的内存可以用来放更大的空间模型或处理更多帧，间接带来精度提升

## 亮点与洞察
- **冻结+初始化+适配器三板斧**：看似简单，但组合起来效果出色。这种"先冻结大组件，再用小适配器弥补"的思路在 NLP 的 LoRA/Adapter 中已被广泛验证，本文将其成功应用到视频理解领域
- **把省出的资源换性能**：冻结空间 Transformer 不仅仅是为了"省钱"，更重要的是释放的内存可以用来扩大模型或增加帧数，形成新的性能提升来源。这种资源再分配的思路值得借鉴
- **跨模态初始化的有效性**：用处理空间信息的预训练权重初始化一个要处理时间信息的 Transformer，居然有效——说明 Transformer 学习到的注意力计算能力具有很强的通用性

## 局限与展望
- 方法与因子化编码器架构绑定，对于非因子化的视频 Transformer（如联合时空注意力）不直接适用
- 适配器的设计较为简单（线性投影），探索更复杂的适配器结构（如 cross-attention adapter）可能进一步提升性能
- 实验未涉及视频生成、视频检索等其他视频任务，方法的泛化性有待验证
- 冻结空间编码器后，模型完全依赖预训练的空间表示，如果目标任务的视觉分布与 ImageNet 差异很大（如医学视频），效果可能受限

## 相关工作与启发
- **vs 全参数微调 ViViT**: 全参数微调可以获得略好的精度，但训练成本高一倍。本文方法是成本-效果的更优折中
- **vs TimeSFormer**: TimeSFormer 采用分离的时空注意力，也有类似的分解结构，但没有探索冻结空间部分的可能性
- **vs Adapter-based 方法（如 AIM, ST-Adapter）**: NLP/多模态领域的 adapter 思路类似，本文的贡献在于系统验证了这一思路在视频因子化编码器中的有效性，并额外发现了初始化策略的关键作用

## 评分
- 新颖性: ⭐⭐⭐ 方法组件（冻结、适配器、初始化）都不算新，但在视频因子化编码器中的系统组合是新的
- 实验充分度: ⭐⭐⭐⭐ 6个基准、详细消融、泛化到其他因子化模型
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验设计合理，结论直接
- 价值: ⭐⭐⭐⭐ 对资源有限的视频理解研究者有直接帮助，降低了训练门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] On the Utility of 3D Hand Poses for Action Recognition](on_the_utility_of_3d_hand_poses_for_action_recognition.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)

</div>

<!-- RELATED:END -->
