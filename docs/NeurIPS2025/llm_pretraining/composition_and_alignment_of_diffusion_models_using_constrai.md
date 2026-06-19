---
title: >-
  [论文解读] Composition and Alignment of Diffusion Models using Constrained Learning
description: >-
  [NeurIPS 2025][预训练][扩散模型] 提出统一的约束优化框架，将扩散模型的奖励对齐和多模型组合问题形式化为约束优化，通过拉格朗日对偶方法自动确定最优权重，避免手动超参数搜索。 1. 领域现状： 扩散模型的对齐（adjustment to rewards/preferences）和组合（combining mul…
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "扩散模型"
  - "约束优化"
  - "模型对齐"
  - "模型组合"
  - "拉格朗日对偶"
---

# Composition and Alignment of Diffusion Models using Constrained Learning

**会议**: NeurIPS 2025  
**arXiv**: [2508.19104](https://arxiv.org/abs/2508.19104)  
**代码**: [GitHub](https://github.com/shervinkhalafi/constrained_comp_align)  
**领域**: 图像生成  
**关键词**: 扩散模型, 约束优化, 模型对齐, 模型组合, 拉格朗日对偶

## 一句话总结

提出统一的约束优化框架，将扩散模型的奖励对齐和多模型组合问题形式化为约束优化，通过拉格朗日对偶方法自动确定最优权重，避免手动超参数搜索。

## 研究背景与动机

1. **领域现状**: 扩散模型的对齐（adjustment to rewards/preferences）和组合（combining multiple pretrained models）是两种常用的适配方法，但面临多目标冲突的权衡。

2. **现有痛点**: 对齐中使用KL散度和奖励的加权平均需要手动调权，权重不当会导致过拟合某个奖励或偏离预训练模型过远。组合中等权重混合可能偏向某些相似模型而忽略其他。

3. **核心矛盾**: 加权方法的搜索空间不直观——权重本身没有语义含义，而约束形式（如"奖励至少达到b"）更自然可解释。

4. **本文目标**: 提供一种自动平衡多目标的统一框架，消除手动调权需求。

5. **切入角度**: 用约束优化替代加权优化：对齐问题改为"在满足奖励约束下最小化与预训练模型的KL散度"；组合问题改为"最小化到所有模型的最大KL散度"。

6. **核心 idea**: 约束优化框架统一了对齐和组合，通过拉格朗日对偶自动学习最优权重，使约束阈值成为唯一需要设置的直观超参数。

## 方法详解

### 整体框架

两个核心问题：(1) **约束对齐(UR-A)**：$\min_p D_{KL}(p\|q)$ s.t. $\mathbb{E}_{x\sim p}[r_i(x)] \geq b_i$；(2) **约束组合(UR-C)**：$\min_{p,u} u$ s.t. $D_{KL}(p\|q^i) \leq u$。两者都通过拉格朗日对偶转化为可训练的原始-对偶算法。

### 关键设计

**1. 约束对齐的理论解（Theorem 1）**

- **功能**: 提供约束对齐问题的闭式最优分布
- **核心思路**: 最优解为奖励加权分布 $q_{rw}^{(\lambda^*)}(\cdot) = \frac{1}{Z}q(\cdot)e^{\lambda^{*\top}r(\cdot)}$，其中最优对偶变量 $\lambda^*$ 通过对偶上升自动确定
- **设计动机**: 定理证明了约束优化等价于对预训练分布的指数奖励倾斜，且最优权重由约束自动决定

**2. 约束组合的倾斜积分布（Theorem 3）**

- **功能**: 自动确定多模型组合的最优权重
- **核心思路**: 最优解为倾斜积分布 $q_{AND}^{(\lambda)}(\cdot) \propto \prod_{i=1}^m (q^i(\cdot))^{\lambda_i/\mathbf{1}^\top\lambda}$。通过KL约束确保组合分布与每个预训练模型等距离偏离
- **设计动机**: 等权重组合会偏向最相似的模型（如两个相近的高斯分布主导第三个），约束方法自动平衡

**3. 路径KL与点KL散度的区分**

- **功能**: 为不同任务选择合适的KL度量
- **核心思路**: 路径KL（path-wise）度量整个扩散轨迹的差异，适用于对齐；点KL（point-wise）度量最终分布差异，更适用于组合。通过Lemma 2提出新方法计算点KL散度
- **设计动机**: 两种KL度量性质不同，正确选择影响优化的理论保证和实际效果

### 损失函数 / 训练策略

原始-对偶交替优化：原始步最小化拉格朗日函数（SGD），对偶步通过约束违反程度的次梯度上升更新乘子。使用LoRA微调Stable Diffusion v1.5。

## 实验关键数据

### 主实验

多奖励约束对齐（Stable Diffusion + MPS/饱和度/局部对比度）：

| 方法 | MPS奖励 | 饱和度约束 | 对比度约束 | 与预训练KL |
|------|---------|-----------|-----------|-----------|
| 等权加权 | 下降 | 部分满足 | 部分满足 | 较大 |
| **约束对齐** | **提升50%** | **满足** | **满足** | **更小** |

多模型组合（每个模型用不同奖励微调）：

| 方法 | 各奖励维持率 | 最差奖励 |
|------|------------|---------|
| 等权组合 | 某些奖励下降30% | 明显退化 |
| **约束组合** | 全部维持或提升 | 无明显退化 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 约束对齐 vs 加权对齐 | 约束对齐KL更小且全部奖励满足 | 加权方法易过拟合某些奖励 |
| 约束组合 vs 等权组合 | 约束组合的CLIP/BLIP最低分更高 | 等权方法偏向相似模型 |

### 关键发现

- 加权方法在多奖励对齐中容易过拟合容易优化的奖励而忽略困难的（如HPS）
- 约束方法同时改善了所有奖励且与预训练模型的KL散度更小
- 概念组合中，约束方法比等权提高了CLIP和BLIP的最低分

## 亮点与洞察

- **理论扎实**: 从分布空间到扩散模型空间的完整理论链条，建立强对偶性
- **实用性**: 约束阈值比权重更直观——"奖励至少提升50%"比"权重设为0.3"更容易理解
- **统一性**: 同一框架解决对齐和组合两个看似不同的问题

## 局限与展望

- MCMC采样在高维（如图像生成）中代价高
- 点KL散度的计算依赖于分数函数的质量
- 目前仅在Stable Diffusion v1.5上验证，更大模型的scaling需研究
- 可扩展到混合组合（OR模式）和条件约束

## 相关工作与启发

- AlignProp框架被扩展为多奖励约束版本
- 与DPO等对齐方法相比，约束方法不需要手动调正则化权重
- 启发: 约束优化提供了比加权优化更可解释、更易调参的替代方案

## 评分

- 新颖性: ⭐⭐⭐⭐ 约束视角统一对齐和组合的思路新颖
- 实验充分度: ⭐⭐⭐ 实验规模适中，但验证了核心claims
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 为扩散模型适配提供了原则性框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[NeurIPS 2025\] A Practical Guide for Incorporating Symmetry in Diffusion Policy](a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)
- [\[ICML 2025\] Provable Maximum Entropy Manifold Exploration via Diffusion Models](../../ICML2025/llm_pretraining/provable_maximum_entropy_manifold_exploration_via_diffusion_models.md)
- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](ricl_temporal_credit.md)

</div>

<!-- RELATED:END -->
