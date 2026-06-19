---
title: >-
  [论文解读] TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception
description: >-
  [CVPR 2025][协作感知] 提出 TraF-Align 框架，通过在特征级别预测目标运动轨迹来学习特征的时空流动路径，沿轨迹生成时序有序的采样点将当前时刻 query 引导至相关历史特征，实现异步多智能体感知中的精确特征对齐，在 V2V4Real 和 DAIR-V2X-Seq 两个真实数据集上刷新SOTA。
tags:
  - "CVPR 2025"
  - "协作感知"
  - "异步融合"
  - "特征对齐"
  - "轨迹预测"
  - "V2V通信"
---

# TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception

**会议**: CVPR 2025  
**arXiv**: [2503.19391](https://arxiv.org/abs/2503.19391)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 协作感知, 异步融合, 特征对齐, 轨迹预测, V2V通信

## 一句话总结

提出 TraF-Align 框架，通过在特征级别预测目标运动轨迹来学习特征的时空流动路径，沿轨迹生成时序有序的采样点将当前时刻 query 引导至相关历史特征，实现异步多智能体感知中的精确特征对齐，在 V2V4Real 和 DAIR-V2X-Seq 两个真实数据集上刷新SOTA。

## 研究背景与动机

**领域现状**：协作感知（cooperative perception）通过多辆车共享传感器数据来增强单车的感知范围。然而实际通信中不可避免存在延迟，导致接收到的特征与自车当前观测在时间上不同步。

**现有痛点**：延迟造成两种失配——(1) **空间失配**：物体在延迟期间已移动，接收特征中的物体位置与实际不符；(2) **语义失配**：物体的外观、遮挡状态等在不同时刻变化，直接融合会引入不一致信息。现有方法仅用简单的运动补偿（如 warp），无法处理语义层面的不一致。

**核心矛盾**：需要同时修正空间位移和保证语义一致，但空间补偿是几何问题而语义对齐是高层特征匹配问题，二者的联合优化困难。

**本文目标**：设计一个统一框架同时解决异步融合中的空间和语义失配。

**切入角度**：如果能预测特征级别的目标轨迹（即特征随时间的运动路径），就能沿轨迹从历史特征中精确聚合信息到当前时刻。

**核心 idea**：从历史观测中预测物体在特征空间的轨迹，沿轨迹采样点引导注意力从当前 query "追溯"到各时刻的相关特征位置。

## 方法详解

### 整体框架

TraF-Align 三阶段流程：(1) 各智能体提取 BEV 特征并带时间戳共享；(2) 利用历史帧预测每个 query 位置的目标运动轨迹；(3) 沿预测轨迹生成时序采样点，通过可变形注意力从多帧历史特征中聚合信息到当前时刻特征。

### 关键设计

1. **特征级轨迹预测**:

    - 功能：预测每个空间位置上目标的运动轨迹
    - 核心思路：给定当前时刻 query 位置和多帧历史 BEV 特征，通过轻量级 MLP 预测目标从过去到现在的运动轨迹（一系列 2D 偏移量）。轨迹从最早的历史帧开始，终止于当前时刻。这些轨迹为后续的注意力采样提供了精确的空间参考线。
    - 设计动机：简单的 warp 假设匀速运动，无法处理加速、转弯等复杂运动。轨迹预测提供了更精确的空间对应关系。

2. **轨迹引导的可变形注意力**:

    - 功能：沿预测轨迹从历史特征中聚合信息
    - 核心思路：沿每条预测轨迹均匀采样时序点，这些点指示了目标在各历史时刻的特征位置。以当前时刻 query 为中心，用可变形注意力机制在这些采样点处提取历史特征，通过注意力权重自适应地融合多帧信息。采样点的时序有序性保证了时空一致性。
    - 设计动机：可变形注意力允许在精确位置采样而非全局扫描，计算高效且精度高。轨迹提供的时序采样点确保注意力关注到"对的地方"。

3. **跨帧语义交互**:

    - 功能：促进多帧特征间的语义一致性
    - 核心思路：在轨迹引导的注意力中，不同时刻的特征通过共享的 query 实现隐式交互，帮助模型学习时间维度上的语义演变（如物体从完全可见到部分遮挡）。这种跨帧交互自然地处理了语义失配问题。
    - 设计动机：单纯的空间对齐无法解决语义变化，跨帧特征交互让模型理解"同一物体在不同时刻的不同样子"。

### 损失函数 / 训练策略

端到端训练，包含 3D 检测损失和轨迹预测辅助损失。

## 实验关键数据

### 主实验

| 方法 | V2V4Real AP@0.5↑ | DAIR-V2X AP@0.5↑ |
|------|-------------------|-------------------|
| SyncNet | 基线 | 基线 |
| CoBEVFlow | +改善 | +改善 |
| **TraF-Align** | **SOTA** | **SOTA** |

*在两个真实世界异步协作感知数据集上均达到最优*

### 关键发现
- 轨迹预测模块显著优于简单的线性运动假设
- 在大延迟（>300ms）场景下优势更明显
- 时序有序采样比随机采样在语义一致性上更好

## 亮点与洞察
- **轨迹作为注意力的空间导航**：将目标运动的先验知识编码为注意力采样的参考线，兼顾了物理合理性和学习灵活性
- **统一处理空间和语义失配**：用同一个注意力机制同时解决两种不同性质的失配
- **在真实数据集上验证**：不同于多数工作仅在仿真器上测试

## 局限与展望
- 轨迹预测在遮挡物体或新出现的物体上可能不准确
- 当前假设所有智能体的延迟已知，未处理延迟估计不确定性
- 仅在 3D 检测任务上验证，协作分割等任务需扩展

## 相关工作与启发
- **vs CoBEVFlow**: CoBEVFlow 用光流补偿空间失配，TraF-Align 用轨迹预测，后者对非刚性运动更鲁棒
- **vs SyncNet**: SyncNet 用简单 warp，TraF-Align 用注意力机制，更灵活
- 轨迹引导注意力的思路可迁移到视频理解中的跨帧特征对齐

## 评分
- 新颖性: ⭐⭐⭐⭐ 轨迹引导注意力的设计针对性强
- 实验充分度: ⭐⭐⭐⭐ 两个真实数据集，消融充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 对异步协作感知有重要推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception](../../ECCV2024/others/align_before_collaborate_mitigating_feature_misalignment_for_robust_multi-agent_.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [\[CVPR 2025\] EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching](edm_equirectangular_projection-oriented_dense_kernelized_feature_matching.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)

</div>

<!-- RELATED:END -->
