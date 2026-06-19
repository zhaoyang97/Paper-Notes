---
title: >-
  [论文解读] Moment Quantization for Video Temporal Grounding
description: >-
  [ICCV 2025][视频理解][视频时序定位] 提出 MQVTG，首次将向量量化引入视频时序定位任务，通过时刻码本和软量化将视频片段映射为离散向量，增强前景/背景的区分度，在 6 个基准上取得 SOTA。 视频时序定位（VTG）旨在根据自然语言描述定位视频中的相关时刻，核心挑战在于区分相关和不相关的时刻…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "视频时序定位"
  - "向量量化"
  - "时刻码本"
  - "高光检测"
  - "离散表示学习"
---

# Moment Quantization for Video Temporal Grounding

**会议**: ICCV 2025  
**arXiv**: [2504.02286](https://arxiv.org/abs/2504.02286)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频时序定位, 向量量化, 时刻码本, 高光检测, 离散表示学习

## 一句话总结

提出 MQVTG，首次将向量量化引入视频时序定位任务，通过时刻码本和软量化将视频片段映射为离散向量，增强前景/背景的区分度，在 6 个基准上取得 SOTA。

## 研究背景与动机

视频时序定位（VTG）旨在根据自然语言描述定位视频中的相关时刻，核心挑战在于区分相关和不相关的时刻。

现有方法的不足：

**连续特征区分度弱**：以 TR-DETR 为例，前景和相似背景特征在特征空间中距离很近，难以区分

**前景特征聚合不足**：不同前景区域的特征在空间中分散，未能有效聚合

**视频的冗余信息**：视频包含大量冗余信息，而 VTG 的核心是提取判别性信息区分前景/背景

关键洞察：前景可以用简洁的离散语言描述（如"用勺子搅拌咖喱"），那么能否用离散向量描述连续视频时刻来增强区分度？向量量化的聚类过程天然符合 VTG 对前景/背景分离的需求。

## 方法详解

### 整体框架

MQVTG 从简单的 Clip Quantization 进阶到 Moment Quantization，核心改进三点：(1) 量化置于时序建模之后以适应跨片段特性；(2) 使用软量化保留视觉多样性；(3) 设计带先验初始化和联合投影的时刻码本。

### 关键设计

1. **Clip Quantization → Moment Quantization 的演进**：

    - Clip Quantization 类似图像量化，直接量化单独片段，忽略了视频时刻的两个特性：**跨片段性**（一个动作跨越多个片段）和**视觉多样性**（同一描述有多种视觉表现）
    - Moment Quantization 在时序编码器 $E_t$ 之后执行量化，让量化操作作用于已建模时序关系的特征 $z_t = E_t(z_s)$

2. **软量化（Soft Quantization）**：不直接替换连续特征为离散码字（硬量化），而是将量化过程作为聚类正则化。通过码本损失 $\mathcal{L}_{cb} = \|C(z_t) - \text{sg}(E_t(z_s))\|_2^2$ 和承诺损失 $\mathcal{L}_{cmt} = \|\text{sg}(C(z_t)) - E_t(z_s)\|_2^2$ 驱动特征-码字聚类，但下游定位模块仍使用连续特征 $z_t$。这避免了有限容量码本导致的信息丢失。

3. **时刻码本（Moment Codebook）**：

    - **先验初始化**：用预训练 CLIP 提取所有训练视频片段的特征，k-means 聚类后以聚类中心初始化码本，确保码字从一开始就有效
    - **联合投影**：引入可训练投影层 $C' = P(C)$（线性层）替代直接优化码本向量，探索不同码字间的时序语义关联

4. **即插即用特性**：量化模块可集成到 encoder-only 和 encoder-decoder (DETR) 架构中，训练时只增加码本参数，推理时零额外成本。

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{overall} = \mathcal{L}_{mr} + \lambda_{hd}\mathcal{L}_{hd} + \lambda_{mq}\mathcal{L}_{mq} + \lambda_{align}\mathcal{L}_{align}$

- $\mathcal{L}_{mr}$：时刻检索损失（L1 + Focal）
- $\mathcal{L}_{hd}$：高光检测损失（视频内对比学习）
- $\mathcal{L}_{mq} = \mathcal{L}_{cb} + \lambda_{cmt}\mathcal{L}_{cmt}$：量化监督
- $\mathcal{L}_{align}$：InfoNCE 视频-文本对齐损失

## 实验关键数据

### 主实验

QVHighlights 验证集（MR + HD）：

| 方法 | R1@0.5 | R1@0.7 | mAP@0.5 | mAP Avg. | HD mAP | HD HIT@1 |
|------|--------|--------|---------|----------|--------|----------|
| TR-DETR | 67.10 | 51.48 | 66.27 | 45.09 | 40.55 | 64.77 |
| CG-DETR | 67.35 | 52.06 | 65.57 | 44.93 | **40.79** | **66.71** |
| R²-Tuning | **68.71** | 52.06 | - | 47.59 | 40.59 | 64.32 |
| **MQVTG** | 67.94 | **53.03** | **68.54** | **48.81** | 40.23 | 65.29 |

Charades-STA / TACoS / Ego4D-NLQ 时刻检索：

| 方法 | Charades R1@0.7 | TACoS R1@0.7 | Ego4D mIoU |
|------|-----------------|--------------|------------|
| R²-Tuning | 37.02 | 25.12 | 4.94 |
| **MQVTG** | **38.84** | **25.82** | **5.08** |

### 消融实验

核心组件消融（QVHighlights val）：

| 配置 | R1@0.5 | R1@0.7 | mAP@0.5 | mAP Avg. |
|------|--------|--------|---------|----------|
| Baseline（无量化） | 65.35 | 49.42 | 66.99 | 45.63 |
| + 时序后量化(QATM) | 66.37 | 51.11 | 67.43 | 47.02 |
| + QATM + 软量化(SQ) | 66.52 | 51.23 | 68.18 | 47.54 |
| + QATM + SQ + 时刻码本(MC) | **67.94** | **53.03** | **68.54** | **48.81** |

量化方式对比：

| 量化方式 | R1@0.7 | mAP Avg. | 说明 |
|---------|--------|----------|------|
| Image Quantization | 51.03 | 46.55 | 图像级量化 |
| Clip Quantization | 51.61 | 46.93 | 片段级量化 |
| **Moment Quantization** | **53.03** | **48.81** | 时刻级量化 |
| Hard Quantization | 50.90 | 47.46 | 直接替换为离散向量 |

### 关键发现

- 即插即用验证：集成到 QD-DETR、TR-DETR、TaskWeave 等 DETR 模型均有一致提升
- YouTube HL 和 TVSum 高光检测分别超 SOTA 2.1% 和 1.4%
- 码本利用率低（<10%）是当前瓶颈，限制细粒度场景的性能提升

## 亮点与洞察

- 首次将向量量化从图像/音频领域迁移到视频时序定位，填补了该方向的空白
- 软量化策略巧妙：量化过程作为正则化驱动特征聚类，但不直接使用离散码字，兼顾区分度和信息完整性
- k-means 先验初始化简单有效，解决了码本训练的冷启动问题

## 局限与展望

- 码本利用率低（<10%），大量码字未被激活，限制了细粒度场景的表现
- 高光检测的提升不如时刻检索明显（全局 vs 局部信息需求冲突）
- 未来可探索动态码本大小或层次化码本结构

## 相关工作与启发

- 量化作为正则化手段（而非压缩/重建目的）的思路可推广到其他需要前景/背景分离的视频任务
- 时刻码本的联合投影策略启发了如何在码本向量间建立关联
- 即插即用特性使其可与未来更强的基线模型组合

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](../../NeurIPS2025/video_understanding/when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)

</div>

<!-- RELATED:END -->
