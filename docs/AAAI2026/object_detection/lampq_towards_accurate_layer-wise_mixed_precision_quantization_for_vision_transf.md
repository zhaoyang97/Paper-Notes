---
title: >-
  [论文解读] LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers
description: >-
  [AAAI 2026][目标检测][混合精度量化] 本文提出 LampQ，一种基于度量（metric-based）的逐层混合精度量化方法，通过类型感知的 Fisher 信息度量衡量 ViT 各层对量化的敏感度，结合整数线性规划优化比特宽度分配并迭代更新，在图像分类、目标检测和零样本量化等多个任务上取得 SOTA 性能。
tags:
  - "AAAI 2026"
  - "目标检测"
  - "混合精度量化"
  - "Transformer"
  - "层自适应"
  - "Fisher信息"
  - "整数线性规划"
---

# LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers

**会议**: AAAI 2026  
**arXiv**: [2511.10004](https://arxiv.org/abs/2511.10004)  
**代码**: 无  
**领域**: 目标检测 / 模型压缩  
**关键词**: 混合精度量化, Vision Transformer, 层自适应, Fisher信息, 整数线性规划

## 一句话总结

本文提出 LampQ，一种基于度量（metric-based）的逐层混合精度量化方法，通过类型感知的 Fisher 信息度量衡量 ViT 各层对量化的敏感度，结合整数线性规划优化比特宽度分配并迭代更新，在图像分类、目标检测和零样本量化等多个任务上取得 SOTA 性能。

## 研究背景与动机

**领域现状**：Vision Transformer（ViT）在视觉任务上取得了卓越性能，但其巨大的参数量和计算需求限制了边缘部署。量化是主要的压缩手段之一，将浮点权重和激活映射到低比特整数表示。

**现有痛点**：现有量化方法多采用统一精度（如所有层都量化到 4-bit），忽略了 ViT 不同组件（注意力层、FFN 层、嵌入层等）对量化的敏感度差异巨大。混合精度量化（MPQ）是一种解决方案，但针对 ViT 的 MPQ 方法存在三个主要限制：（1）量化粒度过粗（如按模块而非按层分配比特）；（2）不同类型组件之间的敏感度度量尺度不可比（attention 和 FFN 的 Fisher 信息量级差异很大）；（3）比特分配不考虑量化后的实际误差。

**核心矛盾**：ViT 的不同组件对量化的敏感度差异巨大，但现有方法无法精确度量这种差异，导致比特分配次优。

**本文目标**：（1）实现逐层的精细粒度混合精度量化；（2）设计可跨组件类型比较的敏感度度量；（3）优化比特分配以最小化整体量化误差。

**切入角度**：使用类型感知的 Fisher 信息矩阵来归一化不同类型组件的敏感度，使其可以在统一尺度上比较。

**核心 idea**：类型感知 Fisher 度量 + 整数线性规划最优比特分配 + 迭代细化 = 精确的逐层混合精度量化。

## 方法详解

### 整体框架

LampQ 的 pipeline：（1）用类型感知 Fisher 度量计算每层的量化敏感度；（2）用整数线性规划在比特预算约束下求解最优比特分配；（3）迭代更新比特分配以进一步精细化。最终输出混合精度量化的 ViT 模型。

### 关键设计

1. **类型感知 Fisher 度量（Type-Aware Fisher Metric）**:

    - 功能：跨组件类型的可比较量化敏感度衡量。
    - 核心思路：对每层计算 Fisher 信息矩阵的迹 $\text{tr}(F_l)$ 作为敏感度基础指标。但不同类型组件（attention Q/K/V、FFN up/down、嵌入层等）的 Fisher 量级差异大，直接比较会偏向某些类型。LampQ 引入类型感知归一化——对同类型的层进行组内归一化，使所有层的敏感度在统一尺度上可比。
    - 设计动机：这是 ViT MPQ 区别于 CNN MPQ 的关键问题——CNN 的层类型相对均匀，而 ViT 包含多种截然不同的组件类型。

2. **整数线性规划比特分配（ILP Bit Allocation）**:

    - 功能：在总比特预算约束下找到最优的逐层比特分配。
    - 核心思路：将比特分配问题建模为整数线性规划：目标函数最小化加权量化误差（由 Fisher 度量给出），约束包括总比特预算和每层的比特范围（如 2-8 bit）。ILP 求解器可以在合理时间内找到精确最优解（因为变量数等于层数，规模不大）。
    - 设计动机：启发式比特分配（如贪心、Top-k）可能陷入局部最优。ILP 提供了全局最优保证。

3. **迭代比特分配更新**:

    - 功能：补偿初始 Fisher 度量与实际量化误差的偏差。
    - 核心思路：初始 Fisher 度量是在全精度模型上计算的，但量化后模型的统计特性会变化。LampQ 在首次比特分配后，用量化模型重新估计 Fisher 度量，再次执行 ILP 求解，迭代更新比特分配。通常 2-3 轮迭代即可收敛。
    - 设计动机：一次性分配可能因 Fisher 估计不准而次优；迭代更新使度量更准确地反映量化后的实际情况。

### 损失函数 / 训练策略

LampQ 是后训练量化（PTQ）方法，不需要重新训练。量化过程中使用少量校准数据计算 Fisher 信息和量化参数。

## 实验关键数据

### 主实验

| 任务 | 模型 | 指标 | LampQ | 统一精度量化 | 提升 |
|------|------|------|-------|-------------|------|
| 图像分类 | ViT/DeiT | Top-1 Acc | SOTA | 基线 | 显著 |
| 目标检测 | ViT-based Det | mAP | SOTA | 基线 | 显著 |
| 零样本量化 | Various | Acc | SOTA | 基线 | 显著 |

### 消融实验

| 配置 | 性能 | 说明 |
|------|------|------|
| LampQ (Full) | 最佳 | 类型感知Fisher + ILP + 迭代 |
| w/o 类型感知归一化 | 下降 | 不同类型层的敏感度不可比 |
| 贪心替代ILP | 次优 | 局部最优vs全局最优 |
| 无迭代更新 | 略差 | 首轮度量不够准确 |

### 关键发现

- 类型感知归一化是关键创新——没有它，MPQ 会严重偏向某些类型的层。
- ILP 比贪心分配一致更好，且求解时间可接受（层数通常在数十到数百量级）。
- 在低比特（如平均 4-bit）下，LampQ 的优势更加明显——此时比特分配的优化空间更大。
- 方法在三种不同任务上一致有效，泛化性好。

## 亮点与洞察

- **类型感知归一化**解决了 ViT MPQ 的根本问题——使不同类型组件的敏感度可以在同一标尺下比较，这对实际应用至关重要。
- **ILP 的使用**在量化领域少见但非常有效——利用了比特分配问题规模适中（层数有限）的特点。
- 后训练量化+混合精度的组合使得方法可以即时应用于已有模型。

## 局限与展望

- 逐层粒度可能不够精细——同一层内不同通道的敏感度也可能不同。
- Fisher 信息的计算需要校准数据，零数据场景下不可用。
- 未考虑硬件对混合精度的支持限制——某些硬件可能只支持特定的比特宽度组合。
- 可以探索与知识蒸馏结合的量化感知训练版本。

## 相关工作与启发

- **vs 统一精度量化（如 PTQ4ViT）**: 统一精度方法简单但次优，LampQ 通过混合精度获得更好的精度-压缩平衡。
- **vs 基于搜索的MPQ（如 NAS-based）**: NAS 方法搜索空间大计算昂贵，LampQ 的 Fisher+ILP 方法更高效。
- **vs CNN 的 MPQ 方法**: ViT 的模块异质性远大于 CNN，需要类型感知的处理。

## 评分

- 新颖性: ⭐⭐⭐⭐ 类型感知Fisher + ILP的组合方案新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三种任务+多种模型+完整消融
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，方法描述清晰
- 价值: ⭐⭐⭐⭐ 对ViT量化部署有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Tri-Modal Fusion Transformers for UAV-based Object Detection](../../CVPR2026/object_detection/tri-modal_fusion_transformers_for_uav-based_object_detection.md)
- [\[AAAI 2026\] Harnessing Vision-Language Models for Time Series Anomaly Detection](harnessing_vision-language_models_for_time_series_anomaly_detection.md)
- [\[ECCV 2024\] TAPTR: Tracking Any Point with Transformers as Detection](../../ECCV2024/object_detection/taptr_tracking_any_point_with_transformers_as_detection.md)
- [\[ECCV 2024\] GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](../../ECCV2024/object_detection/gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](../../ICCV2025/object_detection/adversarial_attention_perturbations_for_large_object_detection_transformers.md)

</div>

<!-- RELATED:END -->
