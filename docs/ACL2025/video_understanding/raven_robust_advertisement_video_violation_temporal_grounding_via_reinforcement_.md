---
title: >-
  [论文解读] RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning
description: >-
  [ACL 2025][视频理解][广告违规检测] 本文提出RAVEN框架，将课程强化学习与多模态LLM结合，通过分层奖励机制和渐进式训练策略，实现广告视频违规内容的精确时序定位和类别预测，无需显式推理标注数据即可激发涌现推理能力。 领域现状： 广告视频违规检测是平台合规的关键环节。早期方法依赖小模型（ViT、ResNet等）…
tags:
  - "ACL 2025"
  - "视频理解"
  - "广告违规检测"
  - "时序定位"
  - "课程强化学习"
  - "GRPO"
  - "多模态大语言模型"
---

# RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning

**会议**: ACL 2025  
**arXiv**: [2510.16455](https://arxiv.org/abs/2510.16455)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 广告违规检测, 时序定位, 课程强化学习, GRPO, 多模态大语言模型

## 一句话总结

本文提出RAVEN框架，将课程强化学习与多模态LLM结合，通过分层奖励机制和渐进式训练策略，实现广告视频违规内容的精确时序定位和类别预测，无需显式推理标注数据即可激发涌现推理能力。

## 研究背景与动机

**领域现状**: 广告视频违规检测是平台合规的关键环节。早期方法依赖小模型（ViT、ResNet等），泛化能力有限。随着LLM的发展，多模态大语言模型（MLLM）被越来越多地用于内容审核。

**现有痛点**: 现有方法面临三大挑战——（1）视频中不仅需要预测违规类别，还需精确定位对应的时间片段，同一视频可能包含多种违规；（2）标注数据中时间区间标注不精确（噪声标注），SFT直接拟合会导致错误学习；（3）SFT在域外数据上泛化能力差，且引发灾难性遗忘。

**核心矛盾**: 精确时间标注成本极高，而粗粒度标注又会误导SFT训练；同时需要模型具备推理能力来处理复杂场景，但不可能为每个样本标注推理过程。

**本文目标**: 设计一个既能利用大规模粗粒度标注数据、又能实现精确时序定位的框架，同时保持MLLM的泛化能力。

**切入角度**: 受DeepSeek-R1启发，用纯强化学习（而非SFT）激发推理能力，结合课程学习策略处理不同质量的标注数据。

**核心 idea**: 通过课程强化学习+分层奖励机制，让MLLM在无推理标注的条件下自发产生结构化推理能力，实现对噪声标注数据的鲁棒学习。

## 方法详解

### 整体框架

RAVEN基于Qwen2.5-VL作为推理模型，整体流程为：输入视频 $V$、违规标签列表 $T$ 和提示 $P$，输出违规类别集合 $\mathcal{C} = \{c_1, c_2, \dots, c_n\}$ 和对应时间区间 $\mathcal{X}_c = (t_c^l, t_c^r)$。模型在 `<think>` 标签内生成推理过程，在 `<answer>` 标签内输出结构化结果。

### 关键设计

#### 1. 数据构建与课程策略

- **功能**: 按标注质量将数据分为精确标注子集和粗粒度标注大规模数据
- **核心思路**: 精确标注用于早期课程学习建立基础，粗粒度标注用于后期大规模训练。不需要任何离线推理数据
- **设计动机**: 回避SFT直接拟合噪声标注的问题，利用RL对噪声的天然鲁棒性

#### 2. 分层奖励函数设计

RAVEN设计了五类奖励函数：

**（a）Thinking Format Reward**: 确保模型在 `<think></think>` 标签内输出推理、在 `<answer></answer>` 标签内输出结果

**（b）Grounding Format Reward**: 分soft和strict两级——soft仅要求包含时间坐标，strict要求使用"temporal start"/"temporal end"等关键词

**（c）Temporal IoU Reward（主奖励）**: 评估预测区间与标注区间的重叠度，使用二值化阈值保持鲁棒性：

$$R_{\text{IoU}} = \begin{cases} 1 & \text{if IoU}(\mathcal{X}_c, \mathcal{Y}_c) > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**（d）Temporal Boundary Alignment Reward（辅助奖励）**: 鼓励边界精确对齐，连续值：

$$R_{\text{Boundary}} = \exp\left(-\sigma^2\left[(t_c^l - y_c^l)^2 + (t_c^r - y_c^r)^2\right]\right)$$

**（e）Violation Category Consistency Reward**: 确保预测类别匹配标注类别，二值化

#### 3. 三阶段课程强化训练

- **Stage 1（精确标注数据）**: $R_{\text{Total}} = R_{\text{IoU}} + \alpha_1 \cdot R_{\text{Boundary}} + R_{\text{Category}}$，建立区间预测和类别识别的基础
- **Stage 2（粗粒度标注数据）**: $R_{\text{Total}} = R_{\text{IoU}} + \alpha_2 \cdot R_{\text{Boundary}}$，去掉类别一致性奖励（因粗标注不可靠），学习近似正确的区间
- **Stage 3（全数据集微调）**: $R_{\text{Total}} = \alpha_3 \cdot R_{\text{IoU}} + \alpha_4 \cdot R_{\text{Boundary}} + \alpha_5 \cdot R_{\text{Category}}$，平衡所有目标

### 损失函数/训练策略

使用GRPO（Group Relative Policy Optimization）算法进行强化学习，不需要冷启动推理训练，直接从Qwen2.5-VL预训练模型开始训练。

## 实验关键数据

### 主实验

**工业数据集上的违规检测性能**（38K训练视频，5K测试视频，6类违规）：

| 方法 | 平均类别精确率/召回率 | 平均时序定位(mIoU) |
|------|----------------------|-------------------|
| Small Models | 0.697/0.657 | - |
| LLaVA-v1.5-SFT | 0.782/0.758 | 0.436 |
| Qwen2.5-VL-7B-SFT | 0.805/0.774 | 0.456 |
| **RAVEN** | **0.826/0.788** | **0.555** |

**公开数据集MultiHateClip上的表现**：

| 方法 | 类别精确率/召回率 | 时序定位(mIoU) |
|------|-----------------|---------------|
| LLaVA-v1.5-SFT | 0.509/0.501 | 0.370 |
| Qwen2.5-VL-7B-SFT | 0.537/0.517 | 0.384 |
| **RAVEN** | **0.551/0.530** | **0.435** |

**在线A/B测试**（20%流量，一整天）：

| 方法 | 类别精确率/召回率 | 时序定位(mIoU) |
|------|-----------------|---------------|
| Small Models | 0.711/0.668 | - |
| Qwen2.5-VL-7B-SFT | 0.800/0.787 | 0.478 |
| **RAVEN** | **0.821/0.803** | **0.563** |

### 消融实验

**结构化思维的作用**：

| 方法 | 类别精确率/召回率 | 时序定位(mIoU) |
|------|-----------------|---------------|
| Qwen2.5-VL-7B-SFT | 0.805/0.774 | 0.456 |
| RAVEN (w/o Structured Thinking) | 0.810/0.779 | 0.537 |
| RAVEN (w/ Structured Thinking) | 0.826/0.788 | 0.555 |

**泛化能力（3类域内训练→2类域外测试）**：

| 方法 | 域内mIoU | 域外mIoU |
|------|---------|---------|
| Qwen2.5-VL-7B-SFT | 0.433 | 0.246 |
| RAVEN | 0.546 | 0.408 |

**奖励函数消融**：

| Boundary Reward | Format Reward | 课程学习 | mIoU |
|----------------|---------------|---------|------|
| ✓ | strict | ✓ | **0.555** |
| ✓ | soft | ✓ | 0.547 |
| ✓ | strict | ✗ | 0.508 |
| ✗ | strict | ✓ | 0.540 |

### 关键发现

1. RAVEN在时序定位上超越SFT基线约10个百分点（mIoU 0.555 vs 0.456）
2. 在线A/B测试验证了实际效果，比SFT模型时序精度高8.5%
3. RL训练的泛化能力远超SFT：域外mIoU 0.408 vs 0.246，提升65.9%
4. 结构化思维对时序定位贡献显著：mIoU从0.537提升到0.555
5. 课程学习策略至关重要：移除后mIoU下降4.7个百分点

## 亮点与洞察

1. **无冷启动推理**: 不需要标注推理过程数据，RL自发激发推理能力，大幅降低数据成本
2. **噪声标注鲁棒性**: 巧妙利用课程学习分离精确/粗粒度数据，配合IoU二值化阈值处理标注噪声
3. **工业级验证**: 不仅有离线实验，还有线上A/B测试验证，证明方案的实际可部署性
4. **缓解灾难性遗忘**: RL相比SFT显著提升了域外泛化能力，这对实际应用至关重要
5. **分层奖励设计精巧**: 格式→位置→边界→类别的层次化设计，每项都有明确的优化目标

## 局限与展望

1. RAVEN基于7B模型，更大模型可能带来进一步提升
2. 论文未公开工业数据集，可复现性受限
3. 违规类别较粗（6大类），更细粒度的子类别检测有待探索
4. 视频采样策略（帧率、关键帧选择）对性能的影响未充分讨论
5. 可探索将RAVEN框架推广到其他时序定位任务（如视频摘要、事件检测）

## 相关工作与启发

- **DeepSeek-R1 (Guo et al., 2025)**: RL激发推理能力的思路直接启发了RAVEN
- **GRPO (Shao et al., 2024)**: Group Relative Policy Optimization算法，RAVEN的训练核心
- **VSLNet / 2D-TAN**: 传统时序定位方法，缺乏推理能力
- **启发**: RL+课程学习+分层奖励的组合范式可推广到其他需要噪声标注鲁棒训练的场景

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[CVPR 2026\] SARL-STG: A Spatially Aware Reinforcement Learning Framework for Refining MLLMs in Spatio-Temporal Video Grounding](../../CVPR2026/video_understanding/sarl-stg_a_spatially_aware_reinforcement_learning_framework_for_refining_mllms_i.md)
- [\[CVPR 2026\] Learning to Refuse: Refusal-Aware Reinforcement Fine-Tuning for Hard-Irrelevant Queries in Video Temporal Grounding](../../CVPR2026/video_understanding/learning_to_refuse_refusal-aware_reinforcement_fine-tuning_for_hard-irrelevant_q.md)
- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](../../NeurIPS2025/video_understanding/tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)

</div>

<!-- RELATED:END -->
