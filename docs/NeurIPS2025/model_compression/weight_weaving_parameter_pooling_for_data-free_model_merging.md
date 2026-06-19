---
title: >-
  [论文解读] Weight Weaving: Parameter Pooling for Data-Free Model Merging
description: >-
  [NeurIPS 2025][模型压缩][模型合并] 本文提出Weight Weaving，一种即插即用的无数据模型合并增强方法，通过在缩放因子搜索空间上对模型参数进行池化操作（如平均、随机选择），消除了对评估数据的依赖，在多任务学习、持续学习和域泛化三个场景中平均准确率最高提升15.9个百分点。 模型合并（Model Me…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "模型合并"
  - "无数据"
  - "缩放因子"
  - "参数池化"
  - "任务向量"
---

# Weight Weaving: Parameter Pooling for Data-Free Model Merging

**会议**: NeurIPS 2025  
**arXiv**: [2510.13921](https://arxiv.org/abs/2510.13921)  
**代码**: [https://github.com/VirtualSpaceman/weight_weaving](https://github.com/VirtualSpaceman/weight_weaving)  
**领域**: 其他  
**关键词**: 模型合并, 无数据, 缩放因子, 参数池化, 任务向量

## 一句话总结
本文提出Weight Weaving，一种即插即用的无数据模型合并增强方法，通过在缩放因子搜索空间上对模型参数进行池化操作（如平均、随机选择），消除了对评估数据的依赖，在多任务学习、持续学习和域泛化三个场景中平均准确率最高提升15.9个百分点。

## 研究背景与动机
模型合并（Model Merging）通过参数级别的操作将多个专家模型整合为一个统一模型，无需重训练。Task Arithmetic等经典方法通过缩放因子λ加权任务向量（fine-tuned与pre-trained权重的差值）并加回预训练模型。然而，λ的选择对性能影响极大：

**核心矛盾**：几乎所有现有方法都依赖缩放因子λ，而正确设置λ通常需要访问评估集数据（"特权数据"），这在实际部署中往往不可得。研究者常常错误地在评估集上调λ，这在真实场景中不可行。仅有的无数据方案（如MetaGPT）仅限于Task Arithmetic且不可推广。

**本文的insight**：与其寻找一个最优λ，不如在λ的搜索空间上对所有候选参数做池化（类似集成的思想）。这样既不需要数据来选择λ，又能从多个λ值中聚合信息。

## 方法详解

### 整体框架
Weight Weaving接收三个用户定义的输入：(1) 基础合并函数f_merge（如TIES、PCB等现有方法）；(2) 缩放因子搜索空间λ_search；(3) 池化函数f_pooling（如平均、随机选择或其他合并方法）。算法流程分三步：
1. 计算delta weights：Δw = {θ_t - θ_pre}
2. 对搜索空间中每个λ_i，用f_merge生成一组增强权重（augmented weights）
3. 将delta weights和augmented weights合并为集合A*，用f_pooling池化后加回预训练模型

### 关键设计

1. **参数级池化而非模型级选择**：传统方法选择一个λ产生一个合并模型。Weight Weaving对搜索空间中所有λ值产生的参数做逐元素池化。直觉是：不同任务的最优λ不同，通过池化可以"边际化"掉对λ的依赖。

2. **协作变体（Collaborative Variant）**：池化不仅在augmented weights上操作，还加入原始delta weights形成更丰富的参数集A* = Δw ∪ A。实验发现纳入更广泛的参数来源有助于提升最终性能。

3. **与现有方法正交**：Weight Weaving作为外层包裹，可以组合任何依赖λ的合并方法。它不修改f_merge的内部逻辑，只在其外部对不同λ的输出做聚合。搜索空间也不限于标量λ，可以是类别变量、概率分布甚至函数。

### 池化函数选项
- **平均池化（Average）**：逐参数算术平均
- **随机均匀选择（Random Uniform）**：每个参数位置独立地从N个候选值中等概率选一个
- **MagMax池化**：每个参数位置选绝对值最大的值

## 实验关键数据

### 主实验：Data-free设置下Weight Weaving的增强效果（平均准确率）

| 基础方法 | 原始（无数据） | +Weight Weaving | 提升 |
|---------|---------------|----------------|------|
| Breadcrumbs | 52.17 | **68.11** | +15.94 |
| MagMax | 60.14 | **69.77** | +9.63 |
| TIES | 68.39 | **71.21** | +2.82 |
| PCB | 71.41 | **72.10** | +0.69 |
| TSV | 73.11 | **74.01** | +0.90 |
| ISO-C | 72.38 | **73.78** | +1.40 |

### 分场景详细结果（ViT-B-32/B-16/L-14三模型均值）

| 方法 | 多任务学习 | 持续学习 | 域泛化 |
|------|-----------|---------|--------|
| TIES | 78.18 | 72.95 | 54.04 |
| TIES+Ours | 78.62 | 74.79 | 60.21 |
| PCB | 81.21 | 74.39 | 58.63 |
| PCB+Ours | 80.92 | 74.86 | 60.53 |
| TSV | 87.10 | 75.32 | 56.91 |
| TSV+Ours | 85.65 | 75.48 | 60.89 |

### 池化函数消融

| 池化函数 | Breadcrumbs | TIES | PCB | TSV | ISO-C |
|---------|------------|------|-----|-----|-------|
| Average | 68.11 | 71.21 | 72.10 | **74.01** | **73.78** |
| Random | 68.11 | 71.21 | 72.08 | 73.64 | 73.61 |
| MagMax | 51.93 | 54.56 | 50.36 | 55.72 | 64.34 |

### 最优λ分布分析

| 场景 | λ分布特点 | Weight Weaving效果 |
|------|----------|-------------------|
| 多任务学习 | 集中在单一值（如ISO-C集中在1.0） | 提升有限或略降 |
| 持续学习 | 广泛分散在整个搜索空间 | 显著提升 |
| 域泛化 | 广泛分散 | 显著提升 |

### 关键发现
- Weight Weaving在持续学习和域泛化场景中提升最大，这恰好是最优λ跨任务分散最广的场景
- 当最优λ集中在单一值时（如多任务学习中的ISO-C），池化效果有限甚至略降
- Average和Random池化效果相近，但MagMax池化效果极差——因为它倾向于选择最大λ对应的参数
- 持续学习中的顺序微调引入了任务权重之间的相关性（与多任务学习中的近正交性形成对比），这是一个独特挑战
- 原始性能较弱的方法（如Breadcrumbs）获益最大（+15.94），已经很强的方法（如TSV）获益较小（+0.90）

## 亮点与洞察
- **极度简单但有效**：Weight Weaving的核心思想就是"试多个λ然后做平均"，概念上非常直观，实现简单
- **真正的无数据方案**：不需要验证集、评估集、测试数据或任何特权信息，在实际部署中切实可行
- **即插即用的模块化设计**：作为外层包裹，可以增强任何依赖λ的合并方法，无需修改原方法
- **对持续学习的观察有启发性**：发现顺序微调导致任务向量之间高度相关，这为设计持续学习专用的合并方法提供了方向
- **最优λ分布分析**：揭示了一个有用的诊断工具——如果最优λ在任务间高度分散，则Weight Weaving最有效

## 局限与展望
- 当最优λ集中在单一值时，池化可能引入次优参数导致性能略降（如ISO-C在多任务学习中）
- 如何在不使用特权数据的情况下过滤掉搜索空间中的次优λ值仍是开放问题
- 计算开销与搜索空间大小和f_merge复杂度成正比，大规模模型（如billions参数）可能需要并行计算
- 实验仅在视觉任务（ViT）上验证，缺乏NLP等其他模态的验证
- Average池化虽简单有效，但可能不是最优选择，自适应加权池化是一个自然的改进方向
- 搜索空间的设计（范围和步长）目前依赖人工经验，自动确定搜索空间值得研究

## 相关工作与启发
- Task Arithmetic(Ilharco et al. 2023)提出任务向量概念，Weight Weaving在其基础上解决了λ选择的核心瓶颈
- TIES、PCB、MagMax等方法关注参数冲突（task conflicts），Weight Weaving从另一个角度（λ鲁棒性）来改善合并质量
- MetaGPT提出闭式解找λ，但仅限Task Arithmetic；Weight Weaving适用于所有合并方法
- 对实际应用有直接价值：在部署时无法获得评估数据的场景（如边缘设备、隐私敏感场景），Weight Weaving是目前最实用的方案

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[CVPR 2025\] Task Singular Vectors: Reducing Task Interference in Model Merging](../../CVPR2025/model_compression/task_singular_vectors_reducing_task_interference_in_model_merging.md)
- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](../../ICLR2026/model_compression/rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)

</div>

<!-- RELATED:END -->
