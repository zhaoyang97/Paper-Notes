---
title: >-
  [论文解读] Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models
description: >-
  [CVPR 2025][多模态VLM][多模态语言模型] 本文提出Coarse Correspondences，一种轻量级的training-free视觉提示方法，通过在图像帧上叠加目标跟踪得到的粗粒度实例对应关系标记，显著增强MLLM的空间时序推理能力，在ScanQA上提升+20.5%、OpenEQA上+9.7%、EgoSchema上+6.0%和R2R导航上+11%。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "多模态语言模型"
  - "空间时序推理"
  - "视觉提示"
  - "目标跟踪"
  - "3D场景理解"
---

# Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models

**会议**: CVPR 2025  
**arXiv**: [2408.00754](https://arxiv.org/abs/2408.00754)  
**代码**: [GitHub](https://coarse-correspondence.github.io)  
**领域**: 视频理解  
**关键词**: 多模态语言模型, 空间时序推理, 视觉提示, 目标跟踪, 3D场景理解

## 一句话总结
本文提出Coarse Correspondences，一种轻量级的training-free视觉提示方法，通过在图像帧上叠加目标跟踪得到的粗粒度实例对应关系标记，显著增强MLLM的空间时序推理能力，在ScanQA上提升+20.5%、OpenEQA上+9.7%、EgoSchema上+6.0%和R2R导航上+11%。

## 研究背景与动机
多模态大语言模型（MLLM）在视觉语言任务上表现出色，但在涉及3D空间理解和长视频时序推理方面仍然表现不佳，经常只比纯文本基线稍好。现有解决方案通常需要专门的3D架构设计、特定任务微调或额外的3D数据输入（如点云），代价高昂且通用性差。

核心矛盾在于：通用MLLM本身具备一定的空间时序推理潜力，但缺乏跨帧的物体对应信息——当输入多帧图像时，模型难以判断哪些区域在不同帧中对应同一物体。

本文的核心idea：借助现有的轻量级视频跟踪模型提取物体级别的跨帧对应关系，然后通过简单的视觉标记（如编号圆点）将这些对应关系直接可视化在图像上，让MLLM"看到"跨帧的物体关联，从而在不修改模型架构、不需要任务特定微调的前提下大幅提升空间时序推理。

## 方法详解

### 整体框架
Coarse Correspondences由四个步骤组成：(1) 跟踪对应关系 → (2) 稀疏化帧采样 → (3) 选择显著对应实例 → (4) 可视化对应关系。处理后的标记图像直接输入通用MLLM进行推理。

### 关键设计
1. **跟踪对应关系（Tracking Correspondences）**:

    - 功能：使用现有的视频目标跟踪模型（如Tracking Anything, SAMv2）从高帧率输入图像序列中提取类别无关的实例分割mask和跨帧ID
    - 核心思路：对于每帧 $I_i$，得到实例分割 $M_i$（H×W矩阵），每个像素标注其所属实例的ID，同一物体在不同帧中共享同一ID
    - 设计动机：视频跟踪是相对轻量的操作，远比在MLLM中处理大量帧要高效

2. **稀疏化帧与选择显著实例（Sparsify Frames & Select Correspondences）**:

    - 功能：从高帧率跟踪结果中均匀采样少量帧 $m \ll n$，然后选择Top-K最显著的物体实例
    - 核心思路：显著性通过两个标准排序——(1) 跨帧出现频率 $\mathcal{F}req(\text{ID}) = \sum_{i=s_1}^{s_m} \mathbf{1}_{\{\text{ID} \in M_i\}}$；(2) 总面积 $\mathcal{A}rea(\text{ID})$。优先选择出现最频繁的物体
    - 设计动机：消融实验发现标注所有对应关系反而降低性能（信息过载），仅保留少数显著实例效果最佳

3. **可视化对应关系（Visualizing Correspondences）**:

    - 功能：在每帧图像上为被选中的实例叠加固定大小的编号标记（圆点+数字），标记位置为实例mask的质心 $(\\bar{x}_{ij}, \\bar{y}_{ij})$
    - 核心思路：质心位置通过mask像素坐标加权平均计算，同一物体在不同帧中使用相同的编号标记，让MLLM能直觉地理解"哪些区域是同一个东西"
    - 设计动机：视觉标记是一种MLLM原生支持的信息传递方式，无需修改任何模型组件

### 损失函数 / 训练策略
对于闭源模型（GPT-4V/O），完全不需要训练，直接在推理时使用Coarse Correspondences处理后的图像。对于开源模型（如LLaVA），可在instruction tuning中也使用标记图像训练，推理时同样使用标记图像，且提升效果可迁移到未见过的数据集。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(GPT-4O+CC) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| ScanQA | CIDEr | 87.0 | 72.2 (GPT-4O) | +14.8 |
| ScanQA | BLEU-2 | 25.5 | 19.8 (GPT-4O) | +5.7 |
| OpenEQA (EM-EQA) | Accuracy | 59.1 | 49.4 (GPT-4O) | +9.7 |
| EgoSchema | Accuracy | 73.2 | 67.2 (GPT-4O) | +6.0 |
| R2R Navigation | Success Rate | 23.0 | 12.0 (GPT-4O) | +11.0 |

| 数据集 | 指标 | LLaVA+CC | LLaVA(Fine-tuned) | 提升 |
|--------|------|------|----------|------|
| ScanQA | CIDEr | 74.2 | 67.3 | +6.9 |
| SQA3D (unseen) | - | +3.1 | - | 泛化能力验证 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不可视化对应 | baseline | 使用原始图像 |
| 标记所有实例 | 低于baseline | 信息过载伤害推理 |
| 仅用分割轮廓 | 略低于CC | mask边界太密增加视觉噪声 |
| 仅用编号标记 | 最优 | 简洁的视觉标记效果最好 |
| K=3-5（标记数量） | 最优范围 | 过多或过少都不理想 |

### 关键发现
- 通用MLLM（GPT-4O）仅用8帧+CC标记即零样本超越了许多使用3D点云或大量帧的专用模型
- 在EgoSchema上用8帧超越了使用256帧的LongViViT、128+帧的MC-ViT-L等方法
- 开源模型（LLaVA）训练时加入CC可获得+6.9%提升，且泛化到未见数据集SQA3D（+3.1%）
- 导航任务中成功率翻倍（12% → 23%），验证了空间时序推理对实体任务的价值

## 亮点与洞察
- **极简而有效**：无需训练、无需修改架构、无需3D数据，仅用2D图像上的视觉标记就大幅提升3D和时序推理
- **少帧高效**：通过先用大量帧做跟踪再稀疏采样的策略，用极少帧保持甚至超越多帧方法的效果
- **跨任务通用**：同一套方法在3D QA、视频QA、导航三类完全不同的任务上都有显著提升
- **视觉提示的范式洞察**：对应关系本质上是帮MLLM建立跨帧的"指代"能力，这一思路可迁移到其他需要跨图像/跨视角理解的场景
- **与视觉对应在3D重建中的经典角色呼应**：论文强调视觉对应关系在深度学习时代被低估了在语义任务中的价值

## 局限与展望
- 依赖跟踪模型质量：如果跟踪器在快速运动或遮挡严重的场景中失败，CC效果会打折扣
- 仅使用实例级别的粗对应，未利用更细粒度（像素级、语义部件级）的对应信息
- 标记可视化的设计（大小、颜色、形状）可能需要针对不同模型调优
- 未探索与模型内部注意力机制的交互——如果对应关系能直接注入模型内部而非视觉叠加，效果可能更好
- 导航实验仅在100个样本上评估，由于计算成本限制

## 相机运动不变性分析

这是论文中一个非常有意思的诊断实验。作者构建了一组专用测试：10个场景从左到右拍摄，每景5个空间关系问题，每问测20次，共1000次试验。然后将帧序反转（模拟从右到左拍摄）再测1000次。

关键发现：不使用 CC 时，左到右序列准确率 58%，但反转后骤降至 50.4%，说明模型高度依赖帧序来"猜"空间关系。使用 CC 后，正向和反向准确率都达到 71.2%，谐波均值从 53.9% 提升到 71.2%（+17.3%）。这表明对应标记让模型真正建立了空间理解，而非记忆帧序模式。

## 相关工作与启发
- **vs 3D-LLM/ScanRefer+MCAN**: 这些方法需要专门的3D架构和数据，CC作为即插即用的方案超越了它们
- **vs LLoVi/LangRepo (Socratic方法)**: 这些方法将视频转为文本再用LLM推理，丢失了视觉细节；CC直接在视觉层面增强MLLM
- **vs VideoAgent**: Agent方法需要多步推理和多次调用模型，CC一次前向传递即可
- **vs Set-of-Mark Prompting**: SoM在单图上标注，CC将其扩展到跨帧对应的维度
- 启发：当我们觉得模型"理解"了空间时，它可能只是在利用帧序等捷径，CC 提供了验证手段

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 思路极其简洁但洞察深刻，将跟踪-标注-MLLM推理优雅串联，首次证明粗粒度对应关系足以大幅提升MLLM的空间时序推理
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖3D QA、视频QA、导航、开源/闭源模型、训练/推理等多维度，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，方法简单易懂，实验展示有力
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高，可即插即用增强任何支持多图输入的MLLM，且开辟了视觉提示增强空间时序推理的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)
- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](../../ACL2025/multimodal_vlm/spatialmqa_mllm_spatial_relations.md)
- [\[AAAI 2026\] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](../../AAAI2026/multimodal_vlm/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)
- [\[CVPR 2025\] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)

</div>

<!-- RELATED:END -->
