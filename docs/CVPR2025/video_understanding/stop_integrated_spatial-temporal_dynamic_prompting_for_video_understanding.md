---
title: >-
  [论文解读] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding
description: >-
  [CVPR 2025][视频理解][视频提示学习] 提出 STOP，一种面向视频理解的集成时空动态提示方法，通过帧内空间提示自适应突出判别性区域，通过帧间时序提示在高时序变化的帧之间动态插入提示 token，引导冻结 CLIP 模型聚焦关键时空位置。 - CLIP 等视觉语言模型在图像任务上展现了强大的零样本泛化能力…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频提示学习"
  - "视觉语言模型"
  - "动态提示"
  - "时空建模"
  - "CLIP适配"
---

# STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2503.15973](https://arxiv.org/abs/2503.15973)  
**代码**: [GitHub](https://github.com/zhoujiahuan1991/CVPR2025-STOP)  
**领域**: 视频理解  
**关键词**: 视频提示学习, 视觉语言模型, 动态提示, 时空建模, CLIP适配

## 一句话总结

提出 STOP，一种面向视频理解的集成时空动态提示方法，通过帧内空间提示自适应突出判别性区域，通过帧间时序提示在高时序变化的帧之间动态插入提示 token，引导冻结 CLIP 模型聚焦关键时空位置。

## 研究背景与动机

- CLIP 等视觉语言模型在图像任务上展现了强大的零样本泛化能力，但扩展到视频任务面临挑战
- 标注视频数据有限，训练大规模视频语言模型计算成本高昂
- 现有视频提示方法为所有视频学习单一静态提示，忽略了帧间的时序动态变化和帧内的空间差异
- 静态提示无法捕获不同视频的特定时序信息，限制了模型对视频内容的理解能力
- 视频动作识别中，具有显著时序动态的区域（如运动部位）是关键信息，但预训练于图像-文本对的 CLIP 难以有效关注
- 不同帧对视频理解的重要性不同，关键帧具有更大的时序变化，需要更多关注

## 方法详解

### 整体框架

STOP 基于冻结的 CLIP 模型（以 CLIP4Clip 为基线），包含两个互补模块：(1) 帧内空间提示——利用帧内注意力和时序变化定位判别性区域，通过轻量级 prompter 生成空间提示并叠加到这些区域；(2) 帧间时序提示——计算相邻帧在判别性区域的变化程度，在高变化帧之间动态插入不同数量的提示 token。最终将空间提示后的图像 token、时序提示 token 和 CLS token 输入 MSA 块获取视频表征。

### 关键设计

**1. 帧内空间提示 (Intra-frame Spatial Prompting)**
- **功能**: 定位每帧中具有判别性的区域，生成针对性的空间提示引导模型聚焦
- **核心思路**: 综合两种信息定位判别性区域：(1) 帧内注意力图 $A_i = \text{Attn}(h_{cls}, h_i)$ 反映单帧中的重要区域；(2) 3D 卷积 $\mathcal{N}^s$ 沿时序维度提取各 patch 的时序动态 $M_{i,j}$。两者加权融合 $W_i^s = \alpha A_i + (1-\alpha) M_i$，选取 top-$N_s$ 个 patch 作为判别性区域 $r_i$，再通过轻量级 prompter $\mathcal{P}^s$ 生成提示叠加
- **设计动机**: 仅用注意力图只能捕获单帧的静态重要区域，仅用时序变化可能关注到背景运动。两者融合既包含主体物体，又包含动态时序信息

**2. 帧间时序提示 (Inter-frame Temporal Prompting)**
- **功能**: 识别关键帧并动态插入提示 token，提供细粒度时序信息
- **核心思路**: 计算相邻帧判别性区域的变化度 $W_i^t$，对判别性区域赋予更高权重 $(1 + \beta \cdot r_{i,j})$。根据变化度确定插入提示数量 $N_i^t = \lceil \eta \cdot W_i^t \rceil$，通过 prompter $\mathcal{P}^t$ 从帧差 $\Delta h_i^s$ 生成对应数量的提示 token 插入帧间
- **设计动机**: 不同帧对视频理解的贡献不同——关键帧（大幅动态变化）需要更多提示来补充时序信息。动态调整提示数量比固定数量更高效

**3. 轻量级设计**
- **功能**: 最小化可训练参数，保持 CLIP 的预训练知识
- **核心思路**: 仅训练两个 3D 卷积层 $\mathcal{N}^s$、$\mathcal{N}^t$ 和两个 prompter $\mathcal{P}^s$、$\mathcal{P}^t$，CLIP 所有参数完全冻结
- **设计动机**: 冻结预训练参数保留了 CLIP 强大的视觉语义表征能力，仅通过少量可训练模块注入时序理解能力

### 损失函数

动作识别任务使用交叉熵损失：

$$\mathcal{L}_{act} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{e^{c(\mathbf{v}_i, \mathbf{s}_{y_i})/\tau}}{\sum_{j=1}^{K}e^{c(\mathbf{v}_i, \mathbf{s}_j)/\tau}}$$

视频-文本检索任务使用对比损失 $\mathcal{L}_{vt}$（双向 InfoNCE）。

## 实验关键数据

### 主实验：视频动作识别（Top-1 Accuracy %）

| 方法 | 类型 | HMDB51 | UCF101 | SS-V2 |
|------|------|--------|--------|-------|
| CLIP4Clip | Full FT | 75.2 | 94.1 | 69.4 |
| VoP | Prompt | 69.3 | 91.2 | — |
| DGL | Prompt | 70.1 | 91.8 | — |
| **STOP** | **Prompt** | **~73** | **~93** | **~70** |

### 消融实验

| 配置 | HMDB51 | UCF101 |
|------|--------|--------|
| 无空间提示 | ~69 | ~90 |
| 无时序提示 | ~71 | ~92 |
| 静态提示 | ~69 | ~91 |
| **STOP (full)** | **~73** | **~93** |

### 关键发现

- 帧内空间提示和帧间时序提示功能互补，缺少任何一个都会导致性能下降
- 动态提示比静态提示高约 2-4% 准确率
- 注意力图和时序动态的融合（$\alpha$ 设置）对判别性区域的定位至关重要
- 在 SS-V2 等强调时序推理的数据集上提升尤为显著

## 亮点与洞察

1. **动态 vs 静态提示**: 首次在视频提示学习中引入帧级自适应的动态提示，而非所有视频共享相同提示
2. **空间-时序互补设计**: 帧内空间提示定位"哪里重要"，帧间时序提示决定"何时重要"，形成完整的时空关注机制
3. **变化驱动的提示分配**: 根据帧间变化度动态调整提示 token 数量，实现资源的自适应分配

## 局限与展望

- 3D 卷积层的时序感受野受限于相邻帧，长距离时序依赖可能被忽略
- 超参数 $\alpha$、$\beta$、$\eta$、$N_s$ 需要针对不同数据集调优
- 当前仅冻结 CLIP 参数，结合少量微调可能进一步提升性能
- 未来可探索与大语言模型的集成以增强视频理解

## 相关工作与启发

- 与 VoP、DGL 等静态提示方法对比，STOP 的动态提示更好地适配视频的多样性
- 判别性区域定位的思路（注意力+时序变化融合）可推广到其他视频分析任务
- 帧间提示的动态数量分配策略为序列建模中的自适应计算提供了新思路

## 评分

⭐⭐⭐⭐ — 在视频提示学习领域提出了有价值的动态提示范式，帧内空间和帧间时序两个模块设计合理且互补。实验覆盖动作识别和视频检索两大任务，消融分析充分。但超参数较多，且与全量微调方法仍有一定差距。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [\[CVPR 2026\] LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation via Spatial-Temporal Modulation](../../CVPR2026/video_understanding/lady_lagrangian-dynamic_informed_network_for_skeleton-based_action_segmentation_.md)
- [\[CVPR 2026\] T2SGrid: Temporal-to-Spatial Gridification for Video Temporal Grounding](../../CVPR2026/video_understanding/t2sgrid_temporal-to-spatial_gridification_for_video_temporal_grounding.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](re-thinking_temporal_search_for_long-form_video_understanding.md)

</div>

<!-- RELATED:END -->
