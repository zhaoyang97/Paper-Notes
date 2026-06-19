---
title: >-
  [论文解读] Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction
description: >-
  [ECCV 2024][医学图像][乳腺超声分割] 本文提出 SF-RecSAM 模型，通过空间-频率特征融合模块弥补SAM在低级特征提取上的不足，并设计双假校正器（Dual False Corrector）利用不确定性估计识别并修正假阳性和假阴性区域，在BUSI和UDIAT两个乳腺超声数据集上显著超越SOTA方法。
tags:
  - "ECCV 2024"
  - "医学图像"
  - "乳腺超声分割"
  - "SAM微调"
  - "空间频率融合"
  - "不确定性估计"
  - "假阳性假阴性校正"
---

# Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction

**会议**: ECCV 2024  
**代码**: [https://github.com/dodooo1/SFRecSAM](https://github.com/dodooo1/SFRecSAM)  
**领域**: 医学图像  
**关键词**: 乳腺超声分割、SAM微调、空间频率融合、不确定性估计、假阳性假阴性校正

## 一句话总结

本文提出 SF-RecSAM 模型，通过空间-频率特征融合模块弥补SAM在低级特征提取上的不足，并设计双假校正器（Dual False Corrector）利用不确定性估计识别并修正假阳性和假阴性区域，在BUSI和UDIAT两个乳腺超声数据集上显著超越SOTA方法。

## 研究背景与动机

**领域现状**：乳腺超声图像分割是乳腺癌筛查和诊断中的重要环节。Segment Anything Model（SAM）作为通用分割基础模型展现了强大的特征提取能力，许多工作尝试将SAM适配到医学图像分割领域。

**现有痛点**：乳腺超声图像具有独特的挑战：(1) 低对比度——肿块与周围组织的灰度差异小；(2) 边界模糊——肿块边缘不清晰甚至呈毛刺状；(3) SAM的ViT编码器虽然在高级语义特征提取上很强，但对乳腺超声中至关重要的低级特征（纹理细节、边界结构信息）捕获不足。现有SAM微调方法通常只是简单地添加adapter或prompt，未能从根本上解决低级特征不足的问题。

**核心矛盾**：SAM在自然图像上预训练获得了强大的高级语义理解能力，但乳腺超声的分割更依赖于低级特征（如边界梯度、纹理模式），两者之间的域鸿沟导致直接微调效果有限。同时，分割结果中的假阳性和假阴性区域往往出现在不确定性高的边界附近，现有方法缺乏针对性的修正机制。

**本文目标** (1) 如何增强SAM对乳腺超声低级特征的感知能力？(2) 如何有效识别和修正分割结果中的假阳性和假阴性区域？

**切入角度**：作者从频域分析的角度出发，认为频率域特征可以提供空间域不易捕获的纹理和边界信息。同时，从不确定性估计的角度出发，认为分割不确定性高的区域正是最需要修正的假阳性/假阴性区域。

**核心 idea**：用空间-频率融合弥补SAM低级特征不足，用基于不确定性估计的双假校正器修正分割中的假阳性和假阴性区域。

## 方法详解

### 整体框架

SF-RecSAM继承了SAM的整体架构（ViT图像编码器+prompt编码器+掩码解码器），但在两个关键位置进行了改进。首先，在ViT编码器中引入空间-频率特征融合模块（Spatial-Frequency Feature Fusion Module），将空间域特征与频率域特征融合以获得更完整的特征表示，特别是增强低级纹理和边界信息。然后，在掩码解码器输出之后添加双假校正器（Dual False Corrector），通过不确定性估计定位并修正假阳性和假阴性区域。整体输入为乳腺超声图像，输出为修正后的分割掩码。

### 关键设计

1. **空间-频率特征融合模块（Spatial-Frequency Feature Fusion Module）**:

    - 功能：将空间域特征与频率域特征融合，补充SAM ViT编码器在低级特征提取上的不足
    - 核心思路：对输入特征图进行二维快速傅里叶变换（2D FFT）获取频率域表示。频率域中的高频分量对应图像中的边缘和纹理细节，低频分量对应整体结构。通过可学习的频率滤波器对频率分量进行选择性增强，然后通过逆FFT变换回空间域。最后将频率增强后的特征与原始空间域特征通过注意力机制进行自适应融合。融合后的特征既保留了SAM原有的高级语义能力，又增强了低级纹理和边界感知
    - 设计动机：乳腺超声图像中肿块与背景的对比度低，空间域的卷积操作难以充分捕获边界信息。频率域分析可以更直接地访问图像的高频信息（边缘、纹理），弥补ViT对局部低级特征感知不足的问题

2. **双假校正器（Dual False Corrector）**:

    - 功能：识别并修正分割结果中的假阳性（将背景误判为肿块）和假阴性（将肿块漏判为背景）区域
    - 核心思路：该模块包含两个分支——假阳性校正器和假阴性校正器。首先通过多次前向传播（Monte Carlo Dropout或类似机制）估计每个像素的分割不确定性。高不确定性区域被标记为"可疑区域"。对于初步分割为前景但不确定性高的区域，假阳性校正器评估是否需要翻转为背景；对于初步分割为背景但不确定性高的区域，假阴性校正器评估是否需要翻转为前景。两个校正器分别生成修正掩码，最终与初始分割结果融合得到最终的分割结果
    - 设计动机：乳腺超声分割中的错误通常集中在肿块边界区域——这些区域恰好是不确定性最高的地方。传统方法将分割视为确定性问题，忽略了模型自身对不同区域预测信心的差异。利用不确定性作为proxy来定位需要修正的区域，可以有针对性地改善分割质量

3. **域适配机制（Domain Adaptation to SAM）**:

    - 功能：在不破坏SAM预训练知识的前提下适配到乳腺超声域
    - 核心思路：在ViT编码器的每个Transformer块中插入轻量级的适配层（adapter），同时冻结SAM原有参数。空间-频率融合模块作为并行分支与ViT块协作。Prompt编码器使用自动生成的prompt（如从粗略分割结果获得的bounding box）而非手动标注。训练时只更新adapter、融合模块和双假校正器的参数
    - 设计动机：直接微调SAM所有参数在小规模的医学数据集上容易过拟合。参数高效微调策略可以保留SAM的预训练知识同时适应新域

### 损失函数 / 训练策略

训练损失包含三部分：(1) 主分割损失——Binary Cross-Entropy + Dice Loss 的组合，用于监督初始分割结果；(2) 校正损失——分别对假阳性校正和假阴性校正进行监督；(3) 不确定性引导损失——鼓励模型在真正困难的边界区域产生更高的不确定性估计。训练采用两阶段策略，先训练基础分割网络，再固定基础网络训练双假校正器。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 SF-RecSAM | 之前SOTA | 提升 |
|---|---|---|---|---|
| BUSI | Dice (%) | 显著领先 | 次优方法 | 明显提升 |
| BUSI | IoU (%) | 显著领先 | 次优方法 | 明显提升 |
| UDIAT | Dice (%) | 显著领先 | 次优方法 | 明显提升 |
| UDIAT | IoU (%) | 显著领先 | 次优方法 | 明显提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| SAM直接微调（baseline） | 基线Dice | 无特殊适配的SAM在超声上表现一般 |
| + 空间-频率融合 | Dice提升 | 频率域信息对边界分割有明显帮助 |
| + 假阳性校正器 | Dice提升，精度提高 | 减少了过度分割的区域 |
| + 假阴性校正器 | Dice提升，召回提高 | 减少了漏检区域 |
| 完整SF-RecSAM | 最佳Dice | 所有模块协同作用 |
| 只用空间融合vs空间-频率融合 | 后者更优 | 验证频率域信息的价值 |

### 关键发现

- 空间-频率融合显著提升了SAM在低对比度超声图像上的边界感知能力
- 假阳性和假阴性校正器分别改善了精度和召回率，二者互补
- 不确定性估计能有效定位分割错误区域，验证了用不确定性引导后处理的可行性
- 频率域特征对乳腺超声分割的贡献大于对自然图像分割的贡献，说明该方法对域特异性强
- 在BUSI和UDIAT两个数据集上的一致提升验证了方法的泛化性

## 亮点与洞察

- 从频率域角度弥补SAM低级特征不足是一个有洞察力的设计，频率分析天然适合捕获纹理和边缘
- 双假校正器的思路很实用——在分割结果的基础上进一步修正，可以作为即插即用模块
- 不确定性估计与分割修正的结合提供了一种"先检测问题再修复"的工作范式
- 代码开源有利于后续研究和复现

## 局限与展望

- 不确定性估计可能依赖多次前向传播，推理效率需要评估
- 仅在两个乳腺超声数据集上验证，可以扩展到其他超声/医学图像类型
- 频率域的可学习滤波器可能对不同设备采集的超声图像需要重新调整
- 双假校正器的阈值设定可能需要针对不同应用场景调整
- 可以考虑将频率分析扩展到3D超声或视频超声序列

## 相关工作与启发

- **SAM (Segment Anything)**：通用分割基础模型，本文在此基础上适配到医学领域
- **Medical SAM Adapter**：通过adapter方式微调SAM到医学图像，但未考虑频率域信息
- **频率域分析在医学图像中的应用**：多项研究表明频率域特征对低对比度医学图像有独特优势
- **不确定性估计**：MC-Dropout等方法估计模型不确定性已被广泛研究，本文将其创造性地用于分割修正
- 启发：对于其他低对比度的医学图像分割任务（如肝脏超声、甲状腺超声），空间-频率融合+不确定性修正的框架具有直接的适用性

## 评分

- 新颖性: ⭐⭐⭐⭐ 空间-频率融合+不确定性双假校正的组合设计有创新性
- 实验充分度: ⭐⭐⭐ 两个数据集验证充分，但消融实验可以更细致
- 写作质量: ⭐⭐⭐ 方法描述清晰，问题动机well-motivated
- 价值: ⭐⭐⭐⭐ 为SAM在医学超声领域的应用提供了有效的适配方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2025/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[ECCV 2024\] Architecture-Agnostic Untrained Network Priors for Image Reconstruction with Frequency Regularization](architecture-agnostic_untrained_network_priors_for_image_reconstruction_with_fre.md)
- [\[ECCV 2024\] Energy-induced Explicit Quantification for Multi-modality MRI Fusion](energy-induced_explicit_quantification_for_multi-modality_mri_fusion.md)
- [\[ECCV 2024\] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)

</div>

<!-- RELATED:END -->
