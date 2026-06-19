---
title: >-
  [论文解读] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis
description: >-
  [NeurIPS 2025][医学图像][类风湿关节炎] 首个公开的多任务腕骨常规X光数据集RAM-W600，包含1048张影像，支持腕骨实例分割和SvdH骨侵蚀评分两大任务，并提供全面的基准测试。 类风湿关节炎(RA)是一种常见的自身免疫疾病，腕关节是RA诊断的核心区域。临床上广泛使用常规X光(CR)进行病程筛查和评估…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "类风湿关节炎"
  - "腕骨分割"
  - "骨侵蚀评分"
  - "数据集"
  - "实例分割"
---

# RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis

**会议**: NeurIPS 2025  
**arXiv**: [2507.05193](https://arxiv.org/abs/2507.05193)  
**代码**: [GitHub](https://github.com/YSongxiao/RAM-W600)  
**领域**: 医学图像  
**关键词**: 类风湿关节炎, 腕骨分割, 骨侵蚀评分, 数据集, 实例分割

## 一句话总结

首个公开的多任务腕骨常规X光数据集RAM-W600，包含1048张影像，支持腕骨实例分割和SvdH骨侵蚀评分两大任务，并提供全面的基准测试。

## 研究背景与动机

类风湿关节炎(RA)是一种常见的自身免疫疾病，腕关节是RA诊断的核心区域。临床上广泛使用常规X光(CR)进行病程筛查和评估，因为CR成本低且可及性强。然而，**计算机辅助诊断(CAD)在腕骨领域的研究严重受限**，核心瓶颈在于高质量实例级标注的获取极其困难：

**解剖复杂性**：腕部由多块小骨组成，关节间隙窄、结构复杂、骨骼频繁重叠，准确标注需要深厚的解剖学知识

**病理干扰**：RA进展导致骨赘、骨侵蚀(BE)甚至骨性融合，改变骨骼形态，使标注难度进一步升高

**现有数据集缺陷**：公开的手部X光数据集要么缺少腕部像素级分割标注，要么BE评分不完整，无法满足RA研究需求

现有工作主要聚焦CT/MRI模态的腕骨分割，且数据集规模小（多为私有数据，几十到几百张）。对于CR的腕骨分割研究极少，尤其缺少面向复杂病理条件的公开数据集。

本文的切入角度是：**构建一个多任务、多中心的腕骨CR基准数据集**，同时覆盖实例分割和SvdH BE评分两个临床关键任务，降低RA腕部研究的门槛。

## 方法详解

### 整体框架

RAM-W600并非提出新模型，而是一个**数据集+基准测试**的贡献。数据集设计涵盖：数据采集→影像预处理→专家标注→数据划分→基准评估的完整流程。

### 关键设计

1. **数据集构成与标注体系**

   数据集包含来自6个医疗中心的388名患者的1048张腕部CR影像。其中618张提供了像素级实例分割标注（覆盖14种腕骨），800张提供了SvdH BE评分（涵盖6个关节面共4800个得分）。

   标注体系包括三个层次：
    - **解剖结构标注**：对14种腕骨（MC1-5、Tr、Tz、Sca、Lu、Cap、Ham、Tri、Radius、Ulna）进行精确轮廓勾画，采用多标签策略独立标注每个骨结构
    - **骨位置标注**：对SvdH系统关注的6个关节区域进行ROI标注
    - **SvdH BE评分标注**：对6个关键关节面进行BE严重程度评分

2. **数据多样性与质量控制**

   数据来源包括3家日本医疗机构（HMCRD、SCGH、HU）和3个公开数据集（DHA、BTXRD、FA），覆盖207名RA患者和181名非RA患者。影像参数通过DICOM标准管理，分辨率为0.15 mm/pixel（内部队列）。

   标注流程由高级放射科医师指导，采用严格的审核程序确保质量。数据集设计注重BE和非BE病例的分层划分，确保训练/测试集的代表性。

3. **基准评估方案**

    - **分割任务**：评估13种监督模型（Unet、DeepLabV3+、TransUNet、SwinUMamba等）和3种基础模型（SAM、MedSAM），使用DSC、NSD、VOE、MSD、RAVD指标
    - **BE分类任务**：评估7种分类模型（MobileViT、ResNet、MedMamba等），使用BACC、F1、DOR、ACC、SEN、SPC、PRE指标

### 损失函数 / 训练策略

基准实验统一使用AdamW优化器（weight decay=1e-2），余弦退火学习率调度。分割任务初始学习率1e-4，100 epochs，batch size 8；分类任务初始学习率1e-6，100 epochs，batch size 16。所有实验在RTX 4090上重复5次（5个固定种子）。

## 实验关键数据

### 主实验——分割

| 模型 | DSC(%) ↑ | NSD(%) ↑ | VOE(%) ↓ | MSD(pix) ↓ | 参数量 |
|------|---------|---------|---------|-----------|-------|
| SwinUMamba | **97.75** | **90.71** | **4.35** | **1.06** | 59.89M |
| TransUNet | 97.62 | 89.48 | 4.60 | 1.05 | 105.91M |
| UMambaEnc | 97.56 | 89.10 | 4.71 | 1.11 | 4.58M |
| Unet++ | 97.33 | 86.99 | 5.15 | 1.36 | 2.41M |
| SAM (box) | 88.74 | 64.40 | 18.45 | 4.25 | 641.09M |
| MedSAM (box) | 85.07 | 38.81 | 25.15 | 5.97 | 93.74M |

### 主实验——BE分类

| 模型 | BACC(%) | F1(%) | DOR | SEN(%) | SPC(%) |
|------|---------|-------|-----|--------|--------|
| MobileViT | **52.64** | 11.85 | 1.82 | 21.06 | 84.23 |
| EfficientFormer | 50.63 | **12.40** | 1.06 | **27.90** | 73.37 |
| MedMamba | 50.83 | 6.91 | **5.89** | 8.94 | 92.73 |
| ConvKAN | 49.26 | 3.49 | 0.44 | 3.82 | **94.70** |

### 关键发现

1. **分割任务**：主流模型在DSC上表现优异（最高97.75%），但NSD指标偏低（最高90.71%），说明**骨边界精确分割仍是瓶颈**。BE组和非BE组的DSC存在显著差异（p<0.05-0.001），证实骨侵蚀对分割性能有负面影响
2. **分类任务**：所有模型BACC和F1均很低（最高仅52.64%和12.40%），反映任务极具挑战性。模型普遍对非BE类偏向严重(高特异性/低敏感性)，**极端类不平衡是核心难点**
3. **基础模型差距**：SAM和MedSAM在腕骨分割上远不如监督模型，通用基础模型在细粒度医学分割上仍有较大提升空间

## 亮点与洞察

- 首个公开的腕骨实例分割数据集，且覆盖分割和BE评分两大临床任务，填补了RA腕部CAD研究的数据空白
- 多中心数据来源（6个机构）增强了数据多样性和模型泛化评估的可靠性
- 基准结果清晰地揭示了当前模型在骨重叠、边界模糊、侵蚀形变等方面的不足，为后续研究指明了具体方向
- Mamba架构模型（SwinUMamba、UMambaEnc）在性能和参数效率上展现出良好平衡

## 局限与展望

- RA案例主要来自日本单一地区，人口学同质性较高，可能限制模型在不同种族/地区的泛化能力
- SvdH BE评分分布严重不均衡，高分病例极度稀缺（3分和5分几乎没有），影响细粒度评分模型的训练和评估
- 仅评估了二分类（有无BE），未探索多等级回归评分
- 未结合临床流程的下游任务（如JSN进展量化、纵向监测）

## 相关工作与启发

- 该数据集可启发将腕骨分割与BE检测结合的多任务学习框架
- 类似数据集构建思路可推广到其他关节（手指、足部）的RA评估
- BE分类任务的极端不平衡可作为医学影像中少样本学习/不平衡学习的良好测试场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个同类公开数据集，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 基准覆盖广泛，但缺少多等级评分和下游任务验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，统计分析详尽
- 价值: ⭐⭐⭐⭐⭐ 对RA领域CAD研究有很高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[ICCV 2025\] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users](../../ICCV2025/medical_imaging/progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)
- [\[NeurIPS 2025\] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders](dermacon-in_a_multi-concept_annotated_dermatological_image_dataset_of_indian_ski.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[CVPR 2025\] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](../../CVPR2025/medical_imaging/cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)

</div>

<!-- RELATED:END -->
