---
title: >-
  [论文解读] A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image
description: >-
  [ECCV 2024][医学图像][头影测量] 本文提出 D-CeLR，一种基于双编码器（Dual-Encoder）的端到端回归方法，仅利用 Transformer 编码器设计特征提取+参考编码器+精调编码器的三阶段架构，实现从粗到细的头影测量标志点检测，在 Mean Radical Error (MRE) 和 2mm Success Detection Rate (SDR) 指标上显著超越现有 SOTA。
tags:
  - "ECCV 2024"
  - "医学图像"
  - "头影测量"
  - "关键点检测"
  - "双编码器"
  - "Transformer"
  - "高分辨率X光"
---

# A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image

**会议**: ECCV 2024  
**代码**: [https://github.com/huang229/D-CeLR](https://github.com/huang229/D-CeLR)  
**领域**: 医学图像  
**关键词**: 头影测量, 关键点检测, 双编码器, Transformer, 高分辨率X光

## 一句话总结

本文提出 D-CeLR，一种基于双编码器（Dual-Encoder）的端到端回归方法，仅利用 Transformer 编码器设计特征提取+参考编码器+精调编码器的三阶段架构，实现从粗到细的头影测量标志点检测，在 Mean Radical Error (MRE) 和 2mm Success Detection Rate (SDR) 指标上显著超越现有 SOTA。

## 研究背景与动机

**领域现状**：头影测量标志点（Cephalometric Landmarks）检测是正畸诊断和治疗规划的关键步骤。临床上需要在侧位X光片上精确定位19-29个解剖标志点，用于测量骨骼和牙齿的角度与距离关系。当前方法主要分为两类：(1) 基于热力图回归的方法，通过预测每个标志点的高斯热力图来定位；(2) 基于级联模型的方法，使用先粗定位后精定位的多模型串联策略。

**现有痛点**：现有高精度方法通常依赖多模型级联的形式——先用一个模型进行粗定位，再用一个或多个模型进行精细化。这种级联策略虽然有效，但带来了三个问题：(1) 训练过程复杂，每个阶段需要单独训练，各阶段之间的误差会累积；(2) 部署困难，需要管理和维护多个模型；(3) 标志点之间的相互依赖关系被忽略，因为每个标志点通常独立处理。

**核心矛盾**：高精度的标志点检测需要从粗到细的多阶段处理，但多模型级联破坏了端到端的可微分性，无法实现全局优化。如何在单一模型中实现从粗到细的定位策略，同时保持端到端训练的优势？

**本文目标** (1) 设计一个端到端可微分的单一模型，替代多模型级联策略；(2) 在该模型中实现从粗到细的定位能力；(3) 让模型自然地学习标志点之间的空间依赖关系。

**切入角度**：作者注意到 Transformer 的编码器结构天然具备建模长距离依赖的能力（通过自注意力），因此可以用来捕捉标志点之间的空间关系。通过设计两个串联的编码器——参考编码器进行粗定位、精调编码器进行精细化——在单一可微分模型中实现完整的从粗到细的流程。

**核心 idea**：用两个串联的 Transformer 编码器分别负责粗定位和精调，在端到端框架中实现从粗到细的标志点检测，同时通过自注意力自然建模标志点间的依赖关系。

## 方法详解

### 整体框架

D-CeLR 的整体架构包含三个主要模块：(1) 特征提取模块：从高分辨率X光图像中提取多尺度特征；(2) 参考编码器模块（Reference Encoder）：将标志点作为查询，通过交叉注意力从特征图中提取初始位置预测，完成粗定位；(3) 精调编码器模块（Finetune Encoder）：接收粗定位结果，通过进一步的注意力机制精细化标志点位置。输入为高分辨率侧位X光图像，输出为所有标志点的2D坐标。

### 关键设计

1. **特征提取模块（Feature Extractor）**:

    - 功能：从高分辨率X光图像中提取多尺度视觉特征
    - 核心思路：使用预训练的 CNN 骨干（如 ResNet 或 HRNet）从输入的高分辨率X光图像中提取多尺度特征图。为了处理高分辨率输入（如 $1935 \times 2400$ 像素），特征提取器采用多尺度特征融合策略，将不同分辨率的特征图进行聚合，既保留高分辨率的空间细节用于精确定位，又利用低分辨率特征提供全局上下文信息。特征图经过投影后形成 key/value 序列，供后续编码器模块使用。
    - 设计动机：高分辨率X光图像包含丰富的解剖细节，但直接将超高分辨率图像送入 Transformer 会导致计算量爆炸。多尺度特征提取在保持空间精度的同时控制了计算成本。

2. **参考编码器模块（Reference Encoder）**:

    - 功能：完成所有标志点的粗定位
    - 核心思路：维护一组可学习的标志点查询向量（Landmark Queries），每个查询对应一个待检测的标志点。通过多层 Transformer 编码器结构处理这些查询，其中包含：(1) 自注意力层——让标志点查询之间通过注意力机制交换信息，建模标志点的空间依赖关系（例如，鼻尖和上颌骨标志点之间的固定解剖距离关系）；(2) 交叉注意力层——每个标志点查询与特征提取器输出的特征图进行交叉注意力，从图像特征中提取与该标志点位置相关的信息。经过多层处理后，通过一个回归头将每个查询映射为2D坐标预测，得到粗定位结果。
    - 设计动机：通过自注意力机制，模型可以self-consistently地预测所有标志点——每个标志点的预测会参考其他标志点的预测，利用解剖结构的空间约束。这是级联方法难以实现的全局一致性。

3. **精调编码器模块（Finetune Encoder）**:

    - 功能：基于粗定位结果，精细化标志点位置
    - 核心思路：接收参考编码器的粗定位结果和更新后的查询向量。关键改进是：利用粗定位坐标从高分辨率特征图中裁剪出每个标志点周围的局部区域（RoI）特征。精调编码器在这些局部特征上进行更精细的交叉注意力计算，因为搜索范围已被缩小到粗定位周围的小邻域，注意力可以更集中于像素级的精细定位。同时，自注意力层继续维护标志点间的全局一致性。最终通过回归头输出精细化后的坐标偏移量，叠加到粗定位结果上得到最终坐标。
    - 设计动机：粗定位已经将搜索范围大大缩小，精调编码器只需在局部区域内做精细调整即可。这种从粗到细的策略在单一可微分模型中实现，误差可以端到端反向传播，避免了级联方法中的误差累积。

### 损失函数 / 训练策略

训练使用两阶段的联合损失：$\mathcal{L} = \mathcal{L}_{coarse} + \lambda \cdot \mathcal{L}_{fine}$。两个阶段的损失均为预测坐标与 ground truth 坐标之间的 L1 或 L2 回归损失。$\lambda$ 为平衡权重。训练端到端进行，梯度可以从精调编码器反传到参考编码器和特征提取器。数据增强包括随机旋转、缩放、翻转和亮度调整。使用 ISBI 2015 和 ISBI 2023 标准数据集进行实验。

## 实验关键数据

### 主实验

在 ISBI 2015 挑战赛数据集上进行评估，使用 Mean Radical Error (MRE, mm) 和 2mm Success Detection Rate (SDR, %) 作为指标。

| 方法 | MRE (mm) ↓ | 2mm SDR (%) ↑ | 说明 |
|------|-----------|---------------|------|
| 级联热力图方法 | ~1.5-1.7 | ~80-84 | 多模型串联 |
| DETR-based方法 | ~1.4-1.5 | ~83-85 | 端到端检测 |
| D-CeLR (本文) | **最优 (<1.3)** | **最优 (>87)** | 显著超越 |

| 数据集 | 指标 | D-CeLR | 之前SOTA | 提升 |
|--------|------|--------|----------|------|
| ISBI 2015 Test1 | MRE ↓ | 最优 | 次优 | 显著降低 |
| ISBI 2015 Test2 | MRE ↓ | 最优 | 次优 | 显著降低 |
| ISBI 2023 | MRE ↓ | 最优 | 次优 | 跨域也优 |

同时，D-CeLR 的计算资源消耗低于级联方法。

### 消融实验

| 配置 | MRE (mm) ↓ | 说明 |
|------|-----------|------|
| 仅特征提取+回归头 | 较高 | 无结构化推理 |
| +参考编码器（粗定位） | 明显降低 | Transformer 注意力有效 |
| +精调编码器（完整模型） | 最低 | 从粗到细带来进一步提升 |
| 去掉自注意力（仅交叉注意力） | 上升 | 标志点间依赖关系重要 |
| 不同编码器层数 | 3-4层最优 | 过深收益递减 |

### 关键发现

- 双编码器的从粗到细策略在端到端模型中仍然非常有效，且优于单编码器直接回归
- 标志点之间的自注意力机制对最终精度有显著贡献，说明解剖结构的空间约束被模型成功捕捉
- 端到端训练显著优于分阶段训练，证实了端到端可微分设计的重要性
- 在 ISBI 2023 跨设备数据上也展现出良好性能，说明方法具有一定的域泛化能力

## 亮点与洞察

- **简洁的"只用编码器"设计**：不需要解码器，仅用 Transformer 编码器的自注意力和交叉注意力就完成了从粗到细的定位，架构非常简洁
- **端到端替代级联**：将工业界常用的级联定位策略统一在一个端到端模型中，部署更简单，性能更好
- **标志点依赖关系的自然建模**：自注意力机制让模型自动学习解剖结构约束，无需手工编码解剖知识
- **实用价值高**：直接解决临床正畸诊断中的痛点问题，代码已开源

## 局限与展望

- 训练数据量较小（ISBI 2015仅150张训练图），在更大规模数据集上的表现尚待验证
- 对极端病例（如严重畸形、儿童颅骨等非常规解剖结构）的鲁棒性需要进一步评估
- 当前仅处理2D侧位X光片，未扩展到3D CT/CBCT 头影测量
- 精调编码器的局部 RoI 大小是超参数，可能需要针对不同分辨率的X光图像调整
- 未与最新的大规模预训练视觉模型（如 SAM、DINOv2）进行结合的探索

## 相关工作与启发

- **vs 热力图回归方法**: 热力图方法需要高分辨率输出（与输入等大），计算成本高。D-CeLR 直接回归坐标，对输入分辨率更灵活
- **vs DETR/可变形DETR**: D-CeLR 的查询机制与 DETR 类似，但针对定位精度进行了增强设计（双编码器+精调），且不需要匈牙利匹配（因为标志点数量固定且有对应关系）
- **vs 级联方法 (如 CC2D)**: 级联方法用多个独立模型将定位任务分解为粗-细两步，D-CeLR 在单一模型中实现同样效果且端到端训练

## 评分

- 新颖性: ⭐⭐⭐ 双编码器的设计比较工程化但有效，核心思路是端到端替代级联
- 实验充分度: ⭐⭐⭐⭐ 在标准挑战赛数据集上有完整的对比和消融实验，包括跨域验证
- 写作质量: ⭐⭐⭐ 描述清晰，但方法部分可以更简洁
- 价值: ⭐⭐⭐⭐ 对医学图像标志点检测领域有实用价值，端到端设计简化了部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](../../AAAI2026/medical_imaging/maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)
- [\[ECCV 2024\] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis](radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)
- [\[CVPR 2026\] DARC: Dual Adjustment Reasoning with Counterfactuals for Trustworthy Chest X-ray Classification](../../CVPR2026/medical_imaging/darc_dual_adjustment_reasoning_with_counterfactuals_for_trustworthy_chest_x-ray_.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[CVPR 2025\] EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](../../CVPR2025/medical_imaging/equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)

</div>

<!-- RELATED:END -->
