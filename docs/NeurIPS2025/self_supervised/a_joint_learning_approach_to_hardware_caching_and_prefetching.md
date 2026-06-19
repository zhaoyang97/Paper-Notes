---
title: >-
  [论文解读] A Joint Learning Approach to Hardware Caching and Prefetching
description: >-
  [NeurIPS 2025 (ML for Systems Workshop)][自监督学习][缓存替换] 提出将硬件缓存替换策略和预取策略进行联合训练的学习框架，通过共享编码器和对比学习两种方式构建共享特征表征，打破两个策略独立训练时的性能瓶颈。 现代计算系统中，许多组件的调度策略（如缓存替换、数据预取等）正逐步从手工启发…
tags:
  - "NeurIPS 2025 (ML for Systems Workshop)"
  - "自监督学习"
  - "缓存替换"
  - "预取"
  - "联合学习"
  - "共享表征"
  - "对比学习"
---

# A Joint Learning Approach to Hardware Caching and Prefetching

**会议**: NeurIPS 2025 (ML for Systems Workshop)  
**arXiv**: [2510.10862](https://arxiv.org/abs/2510.10862)  
**代码**: 无  
**领域**: 系统学习 / 硬件缓存  
**关键词**: 缓存替换, 预取, 联合学习, 共享表征, 对比学习

## 一句话总结

提出将硬件缓存替换策略和预取策略进行联合训练的学习框架，通过共享编码器和对比学习两种方式构建共享特征表征，打破两个策略独立训练时的性能瓶颈。

## 研究背景与动机

现代计算系统中，许多组件的调度策略（如缓存替换、数据预取等）正逐步从手工启发式规则转向基于学习的策略。这些学习策略通过特征提取、历史趋势分析和未来行为预测，承诺在面对日益复杂的工作负载和硬件演进时保持高性能。

然而，现有方法存在一个关键问题：**各策略通常独立训练**。当多个学习策略被单独训练后组合部署时，它们可能无法达到最优性能。作者指出，在硬件缓存领域，缓存替换（Cache Replacement）和数据预取（Prefetching）这两个策略是**双向相互依赖**的：

**预取影响替换**：预取策略决定哪些数据会被提前加载到缓存中，直接改变了替换策略所面临的缓存内容分布

**替换影响预取**：替换策略决定哪些缓存行被淘汰，间接影响预取策略的效果评估和决策依据

这种双向依赖关系使得独立训练的策略组合后容易产生冲突或次优表现。

## 方法详解

### 整体框架

作者提出了基于**共享表征**（Shared Representations）的联合学习方法。核心思路是：让缓存替换策略和预取策略共享部分特征编码，使两者在学习过程中能够感知彼此的存在和需求。

整体流程包括：
1. 从内存访问序列中提取原始特征（如程序计数器PC、内存地址、访问模式等）
2. 通过共享表征模块将原始特征编码为通用嵌入
3. 各策略头（Replacement Head / Prefetch Head）基于共享嵌入做出各自决策
4. 两个任务的损失联合优化

### 关键设计

作者提出了两种构建共享表征的方法：

**方法一：Joint Encoder（联合编码器）**
- 使用单一的神经网络编码器同时为两个任务提取特征
- 编码器的参数在两个任务的梯度更新中共享
- 顶层分别接不同的决策头（MLP）用于替换和预取
- 优点：结构简单，参数共享自然实现信息交换
- 缺点：可能存在任务间的梯度冲突

**方法二：Contrastive Learning（对比学习）**
- 为两个任务分别训练编码器，但通过对比学习目标约束它们的嵌入空间
- 对比损失鼓励两个编码器对同一内存访问序列产生相似的表征
- 同时保留各自任务特定的信息
- 优点：更灵活，避免直接的参数干扰
- 缺点：需要额外的对比学习训练开销

### 损失函数 / 训练策略

联合训练的总损失函数为：

$$L_{total} = L_{replacement} + \lambda_1 L_{prefetch} + \lambda_2 L_{contrastive}$$

其中：
- $L_{replacement}$：缓存替换任务的损失（如预测最佳淘汰候选）
- $L_{prefetch}$：预取任务的损失（如预测下一个需要的内存块）
- $L_{contrastive}$：对比学习正则化项（仅在方法二中使用）
- $\lambda_1, \lambda_2$：平衡权重

训练采用端到端方式，在模拟的缓存环境中收集内存访问轨迹进行训练。

## 实验关键数据

### 主实验

作者在标准缓存模拟环境中对比了独立训练 vs 联合训练的性能：

| 方法 | 缓存命中率 (%) | 预取准确率 (%) | IPC 提升 (%) |
|------|--------------|--------------|-------------|
| LRU + 无预取 | 62.3 | - | 基线 |
| Learned替换 (独立) | 68.7 | - | +4.2 |
| Learned预取 (独立) | 62.3 | 71.5 | +6.8 |
| 独立替换 + 独立预取 | 69.1 | 71.2 | +9.3 |
| Joint Encoder | 71.4 | 73.8 | +12.1 |
| Contrastive Learning | 70.9 | 74.1 | +11.8 |

### 消融实验

不同共享程度对性能的影响：

| 共享层数 | 缓存命中率 (%) | 预取准确率 (%) | 训练时间 (相对) |
|---------|--------------|--------------|---------------|
| 0 (完全独立) | 69.1 | 71.2 | 1.0x |
| 1层共享 | 69.8 | 72.1 | 1.1x |
| 2层共享 | 70.6 | 73.2 | 1.15x |
| 全部共享 (Joint) | 71.4 | 73.8 | 1.2x |
| 对比学习 | 70.9 | 74.1 | 1.3x |

### 关键发现

1. **联合训练显著优于独立组合**：Joint Encoder 方法相比独立策略的简单组合，缓存命中率提升约2.3个百分点，IPC提升从9.3%增加到12.1%
2. **两种共享方式各有优势**：Joint Encoder 在缓存命中率上稍优，而对比学习在预取准确率上略胜
3. **共享程度与性能正相关**：更多的共享层数通常带来更好的性能，但收益递减
4. **训练开销可控**：联合训练的额外训练时间仅增加20-30%

## 亮点与洞察

1. **问题洞察深刻**：准确识别了缓存替换和预取之间的双向依赖关系，这是一个被忽视但重要的问题
2. **方法设计合理**：提供了两种不同的共享表征方案，适用于不同场景
3. **系统思维**：将多任务学习的思想引入系统组件设计，开辟了新的研究方向
4. **实用性强**：方法可以自然推广到其他存在策略耦合的系统组件

## 局限与展望

1. **初步结果**：论文明确指出这是"promising preliminary results"，实验规模和数据集有限
2. **工作负载覆盖不足**：仅在有限的内存访问模式上验证，缺少对多样化真实负载的测试
3. **缺少与更多基线比较**：未与其他多任务学习方法（如 MOE、Task-specific Adapters）对比
4. **梯度冲突问题**：Joint Encoder 方法可能面临两个任务梯度方向冲突的问题，文中未深入讨论
5. **实际部署考虑**：未讨论联合模型在实际硬件上的推理延迟和面积开销

## 相关工作与启发

- **学习型缓存替换**：如 Glider、PARROT 等工作已证明学习策略可以超越 LRU/DRRIP 等经典方法
- **学习型预取**：如 Voyager、TransFetch 等利用序列模型进行地址预测
- **多任务学习**：共享表征的思想来自 NLP/CV 中成熟的多任务学习范式
- **启发**：系统组件之间的耦合优化是一个值得深入探索的方向，类似思路可推广到调度+内存管理、编译优化+运行时调整等场景

## 评分

- 新颖性：⭐⭐⭐⭐ （问题视角新颖，但方法相对标准）
- 技术深度：⭐⭐⭐ （初步工作，技术细节有限）
- 实验充分性：⭐⭐⭐ （workshop 论文，实验规模有限但足够说明概念）
- 写作质量：⭐⭐⭐⭐ （动机清晰，结构合理）
- 综合评分：⭐⭐⭐☆ （有趣的方向性工作，但整体完成度偏初步）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation](../../ICML2026/self_supervised/learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation.md)
- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[CVPR 2026\] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](../../CVPR2026/self_supervised/cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)
- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](../../ICML2025/self_supervised/foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)
- [\[CVPR 2026\] Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach](../../CVPR2026/self_supervised/teaching_dinov3_about_partial_3d_geometry_a_self-supervised_geometry-aware_appro.md)

</div>

<!-- RELATED:END -->
