---
title: >-
  [论文解读] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks
description: >-
  [NeurIPS 2025][语义分割][火星科学] 本文提出 Mars-Bench——首个面向火星科学任务的综合基准，涵盖20个数据集（分类/分割/目标检测三大任务类型），系统评估了 ImageNet 预训练模型、地球观测基础模型和视觉语言模型在火星数据上的表现，发现当前通用模型在火星领域仍有明显不足，呼吁开发火星专用基础模型。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "火星科学"
  - "基准测试"
  - "Foundation Model"
  - "遥感分割"
  - "行星科学"
---

# Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks

**会议**: NeurIPS 2025  
**arXiv**: [2510.24010](https://arxiv.org/abs/2510.24010)  
**代码**: [有 (GitHub)](https://github.com/kerner-lab/Mars-Bench)  
**领域**: 图像分割  
**关键词**: 火星科学、基准测试、Foundation Model、遥感分割、行星科学

## 一句话总结
本文提出 Mars-Bench——首个面向火星科学任务的综合基准，涵盖20个数据集（分类/分割/目标检测三大任务类型），系统评估了 ImageNet 预训练模型、地球观测基础模型和视觉语言模型在火星数据上的表现，发现当前通用模型在火星领域仍有明显不足，呼吁开发火星专用基础模型。

## 研究背景与动机
**领域现状**: Foundation Model 已在医学影像、地球观测(EO)、法律、天文等专业领域取得重大进展。EO 领域有 Geo-Bench、PANGAEA 等标准化基准推动了快速发展。然而火星科学领域缺乏类似的标准化评估体系。

**现有痛点**: 火星相关的 ML 研究（陨石坑检测、地标分类、锥形体分割等）存在数据格式不统一、数据集互操作性差、缺乏标准化评估等问题。现有火星基础模型研究仅在1-2个下游任务上评估，无法全面评估泛化能力。

**核心矛盾**: 火星拥有极为丰富的轨道和表面图像数据（MRO、好奇号/毅力号等采集了PB级图像），但缺乏标准化 ML-ready 格式和统一评估基准，导致这些数据的价值未被充分挖掘。

**本文目标**: 构建首个覆盖多任务、多传感器、多地质特征的火星科学标准化基准，提供统一的评估框架和基线模型。

**切入角度**: 借鉴 EO 领域 Geo-Bench 的成功范式，对现有火星数据集进行质量检查、纠正和格式统一化，配合全面的模型评估。

**核心 idea**: 通过标准化20个火星数据集并系统评估多种模型，填补火星科学领域基准测试的空白。

## 方法详解

### 整体框架
Mars-Bench 是一个包含 **20个数据集** 的综合基准，覆盖三大任务类型：
- **分类（9个数据集）**: 二分类、多类别、多标签设置
- **分割（8个数据集）**: 二值分割和多类别分割
- **目标检测（3个数据集）**: 地物检测

数据来源包括 **2个火星轨道器**（MRO、Mars Odyssey）、**3个火星车**（好奇号、机遇号、勇气号）和 **6种成像传感器**（HiRISE、CTX、THEMIS、Mastcam、Navcam、Pancam 等）。

### 关键设计
1. **易用性设计**: 所有数据集统一为 ML-ready 格式，提供标准化数据加载代码；目标检测同时提供 COCO、Pascal VOC、YOLO 三种标注格式。
2. **专家验证与纠正**: 与行星科学家共同开发，所有分割数据集经专家验证，部分分类数据集与原作者通信后进行了修订和纠正。
3. **标准化数据划分**: 所有数据集提供统一的 train/val/test 划分，保证评估一致性和可复现性。
4. **跨域数据分区**: 按传感器类型、数据模态、任务类别或任务来源进行分区，支持跨传感器/跨任务域迁移实验。
5. **科学覆盖面**: 涵盖陨石坑、锥形体、巨石、滑坡、尘暴、霜冻、大气尘土等火星关键地质特征。

### 评估策略
- **三种训练设置**: 从零训练（随机初始化）、冻结骨干网络做特征提取、全参数微调
- **多模型系统评估**: 分类用 ResNet101/SqueezeNet1.1/InceptionV3/SwinV2-B/ViT-L/16；分割用 U-Net/DeepLabV3+/SegFormer/DPT；检测用 YOLO11/SSD/RetinaNet/Faster R-CNN
- **超参搜索 + 7种子**：对每种模型-数据集-训练策略组合做网格搜索，选最优后用7个随机种子重训，报告 IQM + bootstrap 置信区间

## 实验关键数据

### 主实验

| 任务 | 最佳模型 | 关键发现 |
|------|----------|----------|
| 分类（特征提取） | ViT-L/16, SwinV2-B | Transformer 架构表现最佳，SqueezeNet 持续垫底 |
| 分割（特征提取） | U-Net | 尽管简单，U-Net 在几乎所有数据集上优于 Transformer 模型 |
| 目标检测（特征提取） | YOLO11 | 三个数据集均最优，但 boulder 和 dust devil 检测表现仍较弱 |

**VLM 评估结果（Gemini 2.0 Flash vs GPT-4o Mini）**:

| 数据集 | Gemini F1 | GPT F1 |
|--------|-----------|--------|
| mb-domars16k | 0.32 | 0.30 |
| mb-surface_cls | 0.44 | 0.41 |
| mb-frost_cls | 0.55 | 0.54 |
| mb-atmospheric_dust_cls_edr | 0.50 | 0.56 |
| mb-crater_multi_seg | 0.41 | 0.51 |
| mb-mars_seg_msl | 0.84 | 0.70 |

### 消融实验

| 实验 | 关键结论 |
|------|----------|
| 训练集大小影响 | 数据量增加普遍提升性能，但不同数据集的提升速率和鲁棒性差异显著 |
| EO 模型 vs ImageNet 预训练 | ImageNet 预训练 ViT 反而优于 SatMAE/CROMA/Prithvi 等 EO 基础模型 |
| 小型 VLM（CLIP/SigLIP/SmolVLM） | 表现趋势与 Gemini/GPT 一致，均在需要专业知识的任务上表现不佳 |

### 关键发现
1. **U-Net 仍是火星分割任务的强基线**：尽管架构简单，但稳定优于 SegFormer、DPT 等 Transformer 模型。DPT 方差极大，可靠性低。
2. **EO 预训练模型未能胜过 ImageNet 模型**：可能原因是 ViT 在 1400 万张 ImageNet 图像上预训练，数据多样性远超 EO 模型（≤100万张）。地球和火星轨道图像存在显著域差距（火星无植被/水体/人造结构）。
3. **VLM 在需要专业领域知识的任务上表现差**：在通用类别（沙/岩/天空）上尚可，在精细地质结构（陨石坑类型、火星地标）上显著下降。
4. **目标检测整体困难**：数据集小、每图目标数少、灰度图像对比度低是主要瓶颈。

## 亮点与洞察
- **填补重要空白**: 首个火星科学标准化 ML 基准，类比 EO 领域的 Geo-Bench
- **全面系统评估**: 从传统 CNN 到 Transformer、EO 基础模型、闭源 VLM，评估覆盖面极广
- **实用性强**: 开源全部代码、数据集、基线模型，支持 Hugging Face + Zenodo 两个平台
- **科学驱动**: 与行星科学家共同开发，确保任务的科学相关性
- **重要洞察**: 明确指出需要火星专用基础模型，通用模型和 EO 模型均有局限

## 局限与展望
1. **缺乏地理参考**: 大多数数据集不含空间元数据（经纬度），无法进行空间分布分析或区域泛化研究
2. **数据集规模偏小**: 火星数据标注需要行星科学专家，耗时数月至数年，导致部分数据集很小
3. **缺少自监督/基础模型预训练评估**: 未评估在火星数据本身上做自监督预训练的效果
4. **THEMIS 数据版本老旧**: 陨石坑分割数据集基于 2010 年旧版 THEMIS，2017 年有更新版本可用
5. **未来方向**: 建立火星专用基础模型、探索跨域迁移学习方法、利用大量未标注火星图像做自监督预训练

## 相关工作与启发
- **Geo-Bench / PANGAEA**: EO 领域标准化基准的成功范式，Mars-Bench 直接借鉴了其设计哲学
- **WILDS**: 跨域分布偏移评估方向与 Mars-Bench 的跨传感器/跨任务评估理念一致
- **SatMAE / CROMA / Prithvi**: 地球观测基础模型在火星数据上的迁移效果验证了域差距的存在
- 本文对 VLM 在专业科学领域的局限性分析具有参考价值，不仅限于火星科学

## 评分 ⭐4
首个火星科学标准化基准，填补重要空白，评估全面且实用价值高，但本身无模型创新。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[CVPR 2026\] Kαlos finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks](../../CVPR2026/segmentation/kαlos_finds_consensus_a_meta-algorithm_for_evaluating_inter-annotator_agreement_.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](../../ICCV2025/segmentation/tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[CVPR 2025\] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](../../CVPR2025/segmentation/uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)

</div>

<!-- RELATED:END -->
