---
title: >-
  [论文解读] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning
description: >-
  [ICML 2025][强化学习][网络稀疏化] 本文发现简单的一次性随机剪枝就能解锁深度 RL 的扩展潜力——稀疏网络比配备 SOTA 架构的稠密网络实现更高的参数效率、更强的可塑性保持和更少的梯度干扰。 领域现状 领域现状：领域现状: 深度 RL 中扩大网络规模 (scaling up) 一直困难重重…
tags:
  - "ICML 2025"
  - "强化学习"
  - "网络稀疏化"
  - "深度RL扩展"
  - "随机剪枝"
  - "可塑性损失"
  - "梯度干扰"
---

# Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning

**会议**: ICML 2025  
**arXiv**: [2506.17204](https://arxiv.org/abs/2506.17204)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 网络稀疏化, 深度RL扩展, 随机剪枝, 可塑性损失, 梯度干扰

## 一句话总结
本文发现简单的一次性随机剪枝就能解锁深度 RL 的扩展潜力——稀疏网络比配备 SOTA 架构的稠密网络实现更高的参数效率、更强的可塑性保持和更少的梯度干扰。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**: 深度 RL 中扩大网络规模 (scaling up) 一直困难重重，与监督学习的 scaling law 形成鲜明对比。

**现有痛点**: 天真地增大网络参数量在 RL 中往往不带来性能提升，反而因为可塑性损失 (plasticity loss)、梯度干扰等训练病理现象导致性能下降。现有解决方案包括周期性重置 (periodic reset)、Layer Normalization 等，都是针对性修补。

**核心矛盾**: 更大的网络 ≠ 更好的 RL 性能——训练动态中的病理现象随网络增大而加剧。

**本文切入**: 不追求更复杂的架构修改，而是研究最简单的干预——静态网络稀疏化。

**核心 idea**: 在训练前一次性随机移除一定比例的权重，该稀疏网络在同等参数量下比稠密网络拥有更好的扩展性。

### 解决思路

**本文目标**：### 整体框架
输入：标准 DRL 网络 (如 MLP/CNN) → 训练前一次性随机剪枝 (random pruning，移除预定比例权重) → 正常 RL 训练 → 输出稀疏策略网络。


## 方法详解

### 整体框架
输入：标准 DRL 网络 (如 MLP/CNN) → 训练前一次性随机剪枝 (random pruning，移除预定比例权重) → 正常 RL 训练 → 输出稀疏策略网络。

### 关键设计

1. **One-shot Random Pruning**:

    - 在初始化后、训练前，随机移除预定义比例 (如 50%, 80%, 90%) 的权重连接
    - 剪枝后固定稀疏结构，不再恢复或改变
    - 设计动机：最简单的稀疏化策略，排除了结构搜索的干扰

2. **参数效率分析**:

    - 稀疏网络用更少的有效参数达到更高的网络表达能力
    - 因为稀疏结构减少了参数冗余，每个参数都更"有用"
    - 设计动机：理解为什么稀疏有效——不只是正则化效果

3. **训练病理抵抗力分析**:

    - 可塑性损失 (plasticity loss)：稀疏网络更好地保持学习新信息的能力
    - 梯度干扰 (gradient interference)：稠密网络中不同目标的梯度更容易冲突
    - 设计动机：从优化动力学角度解释稀疏的好处

### 损失函数 / 训练策略
- 使用标准 RL 算法 (PPO, SAC 等)，不修改训练流程
- 唯一修改：训练前做一次随机剪枝

## 实验关键数据

### 主实验

| 环境类型 | 指标 | 稀疏网络 | 稠密网络 (SOTA arch) | 结果 |
|---------|------|---------|-------------------|------|
| DMControl | 回报 | 更高 | 基线 | 稀疏优于稠密 |
| Atari (视觉RL) | 分数 | 更高 | 基线 | 一致提升 |
| Streaming RL | 性能 | 更稳定 | 退化 | 稀疏性缓解遗忘 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同稀疏率 | 性能曲线 | 中等稀疏率 (50-80%) 通常最优 |
| 随机 vs 结构化剪枝 | 性能 | 随机剪枝已足够有效 |
| 有/无 LayerNorm | 性能 | 稀疏+LN 进一步提升 |
| 可塑性指标 | 有效秩 | 稀疏网络保持更高有效秩 |

### 关键发现
- 随机剪枝这一最简单操作就能让 DRL 网络在扩展时保持性能增益
- 稀疏网络展现出更强的梯度一致性（减少干扰）
- 效果在视觉 RL 和 streaming RL 中也一致存在
- 稀疏性与现有的 LayerNorm、periodic reset 等方法互补

## 亮点与洞察
- **反直觉结论**：不需要复杂方法，简单的稀疏化就能解决 DRL scaling 难题
- **统一解释**：参数效率、可塑性保持、梯度干扰减少三个视角统一解释了稀疏的好处
- **实用性极强**：一行代码的修改就能生效

## 局限与展望
- 固定稀疏结构可能不是最优的——动态稀疏化是否能进一步提升？
- 极高稀疏率下性能可能下降，最优稀疏率依赖任务
- 理论解释仍以经验分析为主，缺乏严格证明

## 相关工作与启发
- Lottery Ticket Hypothesis (Frankle & Carlin 2019) 发现稀疏子网络
- Dormant Neuron (Sokar et al. 2023) 分析 RL 中的可塑性损失
- 启发：RL 的 scaling 困难可能不在于网络太小，而在于参数过多导致优化问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 发现简单而深刻：随机剪枝解锁 DRL 扩展
- 实验充分度: ⭐⭐⭐⭐ 多环境类型验证 + 机制分析
- 写作质量: ⭐⭐⭐⭐ 分析清晰，论证有力
- 价值: ⭐⭐⭐⭐⭐ 为 DRL scaling 提供了最简洁的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [\[ICML 2025\] Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC](beyond_the_rainbow_high_performance_deep_reinforcement_learning_on_a_desktop_pc.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
