---
title: >-
  [论文解读] Revisiting Supervision for Continual Representation Learning
description: >-
  [ECCV 2024][自监督学习][continual learning] 挑战了"自监督学习在持续表征学习中优于监督学习"的普遍观点，发现**监督学习加上 MLP 投影头**即可在持续学习场景下构建出比 SSL 更强的表征——关键不在于有无标签，而在于 MLP projector 对特征可迁移性的提升作用。
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "continual learning"
  - "Representation Learning"
  - "MLP Projector"
  - "Feature Transferability"
---

# Revisiting Supervision for Continual Representation Learning

**会议**: ECCV 2024  
**arXiv**: [2311.13321](https://arxiv.org/abs/2311.13321)  
**代码**: [GitHub](https://github.com/danielm1405/sl-vs-ssl-cl)  
**领域**: 自监督学习 / 持续学习  
**关键词**: continual learning, Representation Learning, Self-Supervised Learning, MLP Projector, Feature Transferability

## 一句话总结

挑战了"自监督学习在持续表征学习中优于监督学习"的普遍观点，发现**监督学习加上 MLP 投影头**即可在持续学习场景下构建出比 SSL 更强的表征——关键不在于有无标签，而在于 MLP projector 对特征可迁移性的提升作用。

## 研究背景与动机

持续学习（Continual Learning）要求模型在任务序列上不断学习而不遗忘旧知识。近年研究普遍认为自监督学习（SSL）在持续表征学习中优于监督学习（SL），因为 SSL 产生的表征更鲁棒、遗忘更少。这种观点被广泛接受但令人困惑——**额外的标注信息（监督信号）为什么会导致更差的表征？**

作者注意到 SSL 方法（SimCLR、BarlowTwins、BYOL 等）普遍使用 **MLP projector**，而标准 SL 直接接线性分类头。最近的迁移学习研究（Wang et al. 2021, Sariyildiz et al. 2023）已经发现 MLP projector 是提升监督模型特征可迁移性的关键组件，可以缩小 SL 与 SSL 的迁移性差距。

本文将这一洞察引入持续学习：在持续微调场景下，给 SL 加上 MLP projector 是否足以超越 SSL？

核心论点：SSL 在持续学习中的优势并非来自"无标签"训练本身，而是来自 MLP projector 带来的特征可迁移性提升。SL 只要加上 MLP projector（训练时使用、测试时丢弃），就能在持续学习中构建更高质量的表征。

## 方法详解

### 整体框架

方法极其简单——在标准监督学习的 backbone 和分类头之间插入一个 MLP projector：

- **SL**：Backbone → Linear Head → Cross-Entropy Loss
- **SL+MLP**：Backbone → MLP Projector → Linear Head → Cross-Entropy Loss

MLP projector 仅在训练时存在，测试时丢弃。评估使用 backbone 输出的表征做 k-NN 分类。

### 关键设计

#### 1. MLP Projector 结构

**功能**：在 backbone 和分类头之间添加非线性投影层，提升特征可迁移性。

**核心结构**：每个 block 包含 Linear → BatchNorm → ReLU，后接输出线性层。采用 MLPP（Wang et al. 2021）架构，隐含维度 $d_h=4096$，输出维度 $d_o=512$。

**设计动机**：MLP projector 将分类任务的特化信息"隔离"在投影层中，使 backbone 产生更通用、可迁移的表征。丢弃 projector 后，backbone 特征对未见任务保持更强的适应性。消融实验表明 **BatchNorm 是最关键的组件**，但完整的 Linear+BN+ReLU 组合效果最好。

#### 2. 与持续学习策略的协同

**功能**：验证 SL+MLP 与现有 CL 方法的兼容性。

**实验组合**：

- SL+MLP + LwF（知识蒸馏）
- SL+MLP + PFR（投影正则化特征蒸馏）
- SupCon + CaSSLe / PFR

**发现**：SL+MLP 与 PFR 组合效果最佳，因为 PFR 使用可学习投影来增强特征蒸馏，与 MLP projector 的设计理念一致。

#### 3. 表征质量分析

作者从多个角度分析了 SL+MLP 表征质量优越的原因：

- **遗忘更低**：SL+MLP 的表征遗忘率最低（C10→C100: 4.5%），远低于 SL（12.4%）和 SSL（9.7%）
- **任务特异性特征保留（EXC）**：SL+MLP 的 EXC 分数最高（4.3），说明能保留旧任务特征。SSL 竟然为负（-1.8），意味着在旧任务上训练后微调到新任务，旧任务表征反而变差
- **特征多样性递增**：奇异值谱分析表明 SL+MLP 是唯一在持续学习过程中特征多样性持续增加的方法，SL 表现出 neural collapse，SSL 多样性基本不变

### 损失函数 / 训练策略

- 使用标准 Cross-Entropy Loss
- SL 训练 100 epochs/task，SSL 训练 500 epochs/task
- SGD + cosine schedule
- SL+MLP 每个新任务**随机重初始化 projector**（消融发现 SL 模式下重初始化优于继承权重）

## 实验关键数据

### 主实验

持续微调场景下 k-NN 精度（%，ResNet-18 backbone）：

| 方法 | C10/5 | C100/5 | C100/20 | IN100/5 |
|------|:---:|:---:|:---:|:---:|
| SL（无 projector） | 59.8 | 45.3 | 23.1 | 40.4 |
| **SL+MLP** | **65.9** | **61.9** | **47.1** | **62.4** |
| t-ReX | 69.3 | 59.2 | 50.8 | 59.2 |
| SupCon | 60.4 | 49.4 | 30.0 | 57.6 |
| BarlowTwins (SSL) | 76.2 | 54.1 | 40.0 | 57.0 |
| SimCLR (SSL) | 72.4 | 48.9 | 33.4 | 54.7 |

SL+MLP 在 C100/5、C100/20、IN100/5 上**全面超越 SSL**。C10/5 上 SSL 领先，但结合 CL 方法后差距被消除。

### 消融实验

MLP Projector 组件消融（CIFAR100/5, k-NN 精度 %）：

| Linear | ReLU | BatchNorm | Acc (%) |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 46.74 |
| ✗ | ✓ | ✗ | 49.06 |
| ✓ | ✗ | ✗ | 47.59 |
| ✓ | ✓ | ✗ | 53.87 |
| ✗ | ✗ | ✓ | 57.33 |
| ✗ | ✓ | ✓ | 59.15 |
| ✓ | ✗ | ✓ | 58.47 |
| ✓ | ✓ | ✓ | **61.46** |

BatchNorm 贡献最大（+10.6 vs 不使用任何组件），完整 block 效果最优。Projector 深度消融发现：**一个 block 就够了**，更多层几乎无额外增益。

与 CL 策略结合效果（k-NN 精度 %）：

| 方法 | CL 策略 | C100/5 | IN100/5 |
|------|---------|:---:|:---:|
| SL+MLP | Finetune | 61.9 | 62.4 |
| SL+MLP | PFR | **63.6** | **65.2** |
| BarlowTwins | Finetune | 54.1 | 57.0 |
| BarlowTwins | CaSSLe | 58.6 | 64.9 |

### 关键发现

1. **所有使用 MLP projector 的监督方法**（SL+MLP、t-ReX、SupCon）都大幅超越不使用 projector 的 SL，且使用不同损失函数（CE、cosine softmax CE、contrastive loss），说明**关键是 projector 而非损失函数**
2. SL+MLP 仅用 **30% 数据**即可超越使用全部数据训练的 BarlowTwins
3. SL+MLP 在 **40% 标签噪声**下仍优于 SSL，表明对噪声标注有良好鲁棒性
4. 迁移学习评估（8 个下游数据集平均）：SL+MLP 41.9% vs SSL 36.1% vs SL 23.3%

## 亮点与洞察

- **反直觉的结论**：标注信号不应导致更差的持续学习表征，MLP projector 才是 SSL 优势的真正来源
- **极其简单的方法**：只需在 backbone 和分类头之间加一个标准 MLP，无需改变训练流程
- **SL+MLP 是唯一能在序列任务中持续积累知识的方法**：其他方法的任务无关精度要么下降（SL）要么停滞（SSL），SL+MLP 稳步上升
- **SSL 的 EXC 为负**：说明 SSL 在旧任务上的预训练反而比从零开始训练更差，这挑战了"SSL 天然抗遗忘"的观点

## 局限与展望

1. **标注成本**：SL+MLP 仍需要标注数据。虽然 30% 数据即可超越 SSL，但标注本身可能比获取更多无标注数据代价更高
2. **仅评估了表征质量**：未直接解决 class-incremental learning 问题，即分类头的持续更新
3. **训练时间不对等**：SL 100 epochs vs SSL 500 epochs。虽然作者解释了原因，但公平比较存在争议
4. **仅使用 ResNet-18**：更大的 backbone 或 ViT 架构上的结论是否一致尚不清楚

## 相关工作与启发

- **Wang et al. (CVPR 2021)**：发现 MLP projector 可提升监督模型的迁移学习能力，本文将此推广到持续学习
- **CaSSLe (CVPR 2022)**、**PFR (CVPR Workshop 2021)**：SSL 持续学习方法，使用可学习投影做特征蒸馏
- **Madaan et al. (ICLR 2022)**：主张 SSL 在持续表征学习中优于 SL，本文直接反驳

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 方法极简但洞察深刻，"加个 MLP 就能翻转 SL vs SSL 的结论"具有较强颠覆性
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 4 数据集 × 多种 CL 和学习方法 × 8 个下游任务 × 丰富的消融和分析
- **写作质量**: ⭐⭐⭐⭐ — 条理清晰，Figure 1 的直觉展示很好，分析部分深入
- **价值**: ⭐⭐⭐⭐ — 对持续学习社区有重要的认知纠偏价值，方法简单可复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)
- [\[AAAI 2026\] Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](../../AAAI2026/self_supervised/improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)
- [\[ECCV 2024\] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)
- [\[ICLR 2026\] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](../../ICLR2026/self_supervised/fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)
- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/self_supervised/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)

</div>

<!-- RELATED:END -->
