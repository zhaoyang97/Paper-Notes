---
title: >-
  [论文解读] Scale Efficient Training for Large Datasets
description: >-
  [CVPR 2025][语义分割][高效训练] 提出 SeTa（Scale Efficient Training），一种基于 loss 的动态样本剪枝框架，通过随机采样去冗余、loss 聚类分难度、滑动窗口渐进式课程学习三步策略，在 11 个数据集 10 类任务 14 种模型上实现最高 50% 训练成本削减且性能无损。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "高效训练"
  - "动态样本剪枝"
  - "课程学习"
  - "数据高效"
  - "滑动窗口策略"
---

# Scale Efficient Training for Large Datasets

**会议**: CVPR 2025  
**arXiv**: [2503.13385](https://arxiv.org/abs/2503.13385)  
**代码**: [GitHub](https://github.com/mrazhou/SeTa)  
**领域**: 分割/通用训练加速  
**关键词**: 高效训练, 动态样本剪枝, 课程学习, 数据高效, 滑动窗口策略

## 一句话总结

提出 SeTa（Scale Efficient Training），一种基于 loss 的动态样本剪枝框架，通过随机采样去冗余、loss 聚类分难度、滑动窗口渐进式课程学习三步策略，在 11 个数据集 10 类任务 14 种模型上实现最高 50% 训练成本削减且性能无损。

## 研究背景与动机

大规模数据集是深度学习的基石，但随着数据量增长，训练效率与数据规模之间的矛盾日益突出。大量低价值样本的存在造成了训练计算的浪费，这些低价值样本包括三类：(1) 冗余重复样本——提供的信息边际递减；(2) 过难样本——消耗大量计算但对模型优化贡献甚微；(3) 过易样本——已被充分学习不再产生有效梯度。

现有动态剪枝方法（如 InfoBatch）仅基于 loss 均值剔除已学好的样本，忽略了冗余样本和过难样本。静态核心集选择方法需要昂贵的预处理且跨架构泛化性差。

SeTa 的设计哲学是：用 loss 作为计算成本几乎为零的难度代理指标，通过随机采样+loss 聚类+滑动窗口课程学习同时消除三类低价值样本。特别值得注意的是，如图 1 所示，当 ToCa 数据量超过 3M 后继续增加数据对性能提升已趋近饱和，说明冗余是大规模数据集的固有问题。

## 方法详解

### 整体框架

SeTa 是一个即插即用的训练加速框架，仅需修改 3 行代码即可集成。流程分三步：(1) 随机采样以比例 $r$ 去除冗余样本；(2) 对剩余样本按 loss 进行 K-means 聚类得到 $k$ 个难度组；(3) 用滑动窗口从易到难渐进选择样本组进行训练，末期加退火阶段。

### 关键设计1：Loss 引导的样本聚类

**功能**：将训练样本按学习难度分层组织。

**核心思路**：先以比例 $r$ 均匀下采样去除冗余，然后对下采样后的样本按其当前 loss 值 $l_i^t$ 进行 K-means 聚类：$\mathcal{C}^* = \arg\min \sum_{j=1}^{k} \sum_{x_i \in \mathcal{G}_j} \|l_i^t - c_j\|^2$。聚类得到按难度递增排列的 $k$ 个样本组 $\{\mathcal{G}_1, ..., \mathcal{G}_k\}$。

**设计动机**：loss 是深度学习中通用的指标，计算开销为零（训练过程已计算）。相比于 InfoBatch 的二分法（高于/低于均值），K-means 提供更细粒度的难度分层，为后续课程学习提供基础。随机采样在聚类前执行，避免了信息量评估的额外成本。

### 关键设计2：滑动窗口课程学习

**功能**：渐进式地从简单到困难暴露训练样本，同时排除过易和过难样本。

**核心思路**：定义窗口大小 $w = \lceil \alpha k \rceil$（$\alpha \in (0,1]$ 控制每次选择的比例）。窗口位置 $s_t = n \mod (k - w + 1)$ 循环递增，选择第 $s_t$ 到 $s_t + w - 1$ 个难度组。窗口滑动是周期性的——到达最难组后重置回简单组，实现多轮从易到难的循环课程。

**设计动机**：仅训练简单样本会导致欠拟合，仅训练困难样本会导致优化不稳定。滑动窗口的关键优势在于同时排除了"过易组"和"过难组"——任何时刻只选择处于模型当前学习前沿的中间难度组。循环机制让模型反复巩固简单知识同时逐步适应更难样本。

### 关键设计3：部分退火策略

**功能**：减少局部样本选择引入的优化偏差，确保稳定收敛。

**核心思路**：在训练末期阶段，不再使用滑动窗口，而是从全部样本中以概率 $r$ 随机采样：$\mathcal{S}_t^{anneal} = \{x_i | x_i \in \mathcal{G}, u_i < r\}$，$u_i \sim \text{Uniform}(0,1)$。

**设计动机**：滑动窗口虽然高效但可能引入分布偏差（始终缺少极易和极难样本）。部分退火在最终阶段恢复对全分布的暴露，但仍通过采样率 $r$ 维持效率。相比 InfoBatch 的全数据集退火，部分退火更高效。

### 损失函数

SeTa 不修改任何任务特定的损失函数，仅改变每个 epoch 参与训练的样本子集 $S_t$。其训练节省等价于剪枝比例 $\rho_O \approx \bar{\rho}$（因为数据选择开销 $O_d \ll O_m$）。

## 实验关键数据

### 主实验：大规模合成数据集

| 数据集 | 任务 | Baseline | SeTa剪枝率 | SeTa性能 |
|--------|------|----------|-----------|---------|
| ToCa (3M) | 图像描述 COCO CIDEr | 112.7 | 50% | **114.3** (+1.6) |
| SS1M (3M) | 零样本描述 CIDEr | 91.2 | 50% | **92.1** (+0.9) |
| ST+MJ (15M) | 场景文字识别 Avg Acc | 96.3 | 50% | **96.5** (+0.2) |

### ImageNet 分类实验

| 方法 | 骨干 | 剪枝率 | Top-1 Acc |
|------|------|--------|-----------|
| Baseline | ResNet50 | 0% | 76.4 |
| InfoBatch | ResNet50 | 40% | 76.3 (-0.1) |
| **SeTa** | ResNet50 | 40% | **76.5** (+0.1) |
| Baseline | ViT-S | 0% | 79.9 |
| **SeTa** | ViT-S | 50% | **80.0** (+0.1) |
| Baseline | Vim-S | 0% | 80.5 |
| **SeTa** | Vim-S | 40% | **80.7** (+0.2) |

### CIFAR-100 对比

| 方法 | 30% 剪枝 | 50% 剪枝 | 70% 剪枝 |
|------|---------|---------|---------|
| Static Random | 73.8 (-4.4) | 72.1 (-6.1) | 69.7 (-8.5) |
| InfoBatch | 77.5 (-0.7) | 76.2 (-2.0) | - |
| **SeTa** | **78.7** (+0.5) | **78.0** (-0.2) | **76.3** (-1.9) |

### 关键发现

- 在 3M+ 规模的合成数据集上，SeTa 在 50% 剪枝率下不仅无性能损失，反而性能提升（+0.2~+1.6），说明剔除低价值样本有正向效果。
- 跨架构泛化性极强：在 CNN（ResNet）、Transformer（ViT, Swin）、Mamba（Vim）上均有效。
- 跨任务通用性突出：覆盖分类、描述、分割、检索、立体匹配、地理定位等 10 类任务。
- 即使在 70% 极端剪枝率下，性能退化也很小（CIFAR-100 仅 -1.9）。
- LLM 指令微调（LLaMA-7B on Alpaca 52K）中 50% 剪枝仍可保持 MT-bench 分数，说明对小数据集也有效。

## 亮点与洞察

1. **极简设计**：仅需 3 行代码修改即可集成的即插即用框架，这种工程实用性极高。
2. **三类低价值样本的统一处理**：随机采样去冗余+滑动窗口排除过易过难，比 InfoBatch 只处理过易样本更全面。
3. **超越无损**：多个场景下 50% 剪枝反而性能提升，说明过多低质量样本实际上阻碍了模型训练。

## 局限与展望

- Loss 作为难度代理虽然通用但可能不够精细——高 loss 可能来自标注噪声而非真正困难。
- K-means 聚类假设 loss 分布是单峰或可分的，对多模式分布可能效果次优。
- 滑动窗口的超参数（$k$, $\alpha$, $r$, 退火 epoch 数）需要调整，虽然论文声称使用默认设置。
- 对于极小数据集（如 CIFAR-10，50K），随机采样去冗余的必要性较弱。

## 相关工作与启发

- **InfoBatch (ICLR'24)**：基于 loss 均值的动态剪枝先驱，SeTa 通过 K-means 聚类和滑动窗口进一步识别并排除冗余和过难样本。
- **EfficientTrain++**：样本级方法（频率裁剪+课程学习），跨领域通用性不如 SeTa。
- **课程学习**：SeTa 的滑动窗口是一种高效的课程学习实现，循环式从易到难避免了传统课程学习的单调递进限制。

## 评分

⭐⭐⭐⭐ — 方法极简但覆盖面极广（11 数据集 × 10 任务 × 14 模型），实用价值高。核心洞察——同时去除三类低价值样本——比现有方法更全面。3 行代码集成的工程友好度是真正的加分项。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping](../../ICCV2025/segmentation/ragnet_large-scale_reasoning-based_affordance_segmentation_benchmark_towards_gen.md)
- [\[CVPR 2025\] F-LMM: Grounding Frozen Large Multimodal Models](f-lmm_grounding_frozen_large_multimodal_models.md)
- [\[CVPR 2026\] Making Training-Free Diffusion Segmentors Scale with the Generative Power](../../CVPR2026/segmentation/making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)
- [\[CVPR 2025\] ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect](shiftwiseconv_small_convolutional_kernel_with_large_kernel_effect.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)

</div>

<!-- RELATED:END -->
