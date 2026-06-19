---
title: >-
  [论文解读] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users
description: >-
  [ICCV 2025][医学图像][步态分析] 提出ProGait——首个面向大腿截肢假肢用户的多用途视频数据集，支持视频目标分割、2D人体姿态估计和步态分析三项任务，并提供基线模型证明数据集对改善假肢检测的有效性。 核心矛盾 核心矛盾：领域现状：假肢腿在临床康复中至关重要，步态分析是优化假肢设计和对齐的基础…
tags:
  - "ICCV 2025"
  - "医学图像"
  - "步态分析"
  - "假肢检测"
  - "视频数据集"
  - "人体姿态估计"
  - "视频目标分割"
---

# ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users

**会议**: ICCV 2025  
**arXiv**: [2507.10223](https://arxiv.org/abs/2507.10223)  
**代码**: [https://github.com/pittisl/ProGait](https://github.com/pittisl/ProGait)  
**领域**: 医学图像  
**关键词**: 步态分析, 假肢检测, 视频数据集, 人体姿态估计, 视频目标分割

## 一句话总结

提出ProGait——首个面向大腿截肢假肢用户的多用途视频数据集，支持视频目标分割、2D人体姿态估计和步态分析三项任务，并提供基线模型证明数据集对改善假肢检测的有效性。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：假肢腿在临床康复中至关重要，步态分析是优化假肢设计和对齐的基础。传统步态分析依赖专业运动捕捉系统或穿戴式传感器，设备昂贵、侵入性强、受限于实验室环境。基于视觉的机器学习方法提供了可扩展、非侵入的替代方案，但现有视觉模型在检测和分析假肢用户时表现不佳——原因是训练数据几乎全部来自健全人群，无法处理假肢的独特外观和运动模式。

本文的核心动机是：**填补假肢用户视觉分析的数据空白，为视觉模型提供专门的训练和评估资源**。

## 方法详解

### 整体框架

ProGait是一个多用途视频数据集，支持三项核心任务：
1. **视频目标分割（VOS）**: 检测和追踪假肢用户的完整身体
2. **2D人体姿态估计（HPE）**: 检测包括假肢部分在内的23个关键点
3. **步态分析（GA）**: 将步态模式分类为9个类别

### 关键设计

1. **数据采集**: 来自4位大腿截肢者的412个视频片段，每人测试多种新配假肢。涵盖两种场景（平行杠内独立行走 + 杠外辅助行走），每次试验同时采集正面和矢状面视角。1920×1080分辨率，30fps。假肢类型多样（机械膝关节/液压/电脑控制），确保步态模式的多样性。

2. **标注流水线**: 采用Human-in-the-Loop半自动标注方案

    - VOS: GroundingDINO + SAM2生成初始分割，人工修正追踪失败帧
    - HPE: 先在约100帧上人工修正→微调RTMW→在1000帧上推理→二次微调→全数据标注，仅25%视频需手动修正
    - GA: 康复科学研究者提供文本描述，包含步态类别、偏差、假肢调整建议及原因

3. **隐私保护**: 使用GroundingDINO+SAM2检测敏感元素（人脸、可识别标识），高斯模糊处理

### 损失函数 / 训练策略

- VOS基线: YOLO11微调，分别学习离散身体部位再合并mask
- HPE基线: RTMPose微调，两阶段迭代微调策略
- GA基线: 128维LSTM分类器，输入12个下肢关键点的(x,y)坐标时间序列
- 数据划分: ~70%训练/~20%验证/~10%测试，测试集受试者不出现在训练集

## 实验关键数据

### 主实验 (三项任务基准结果)

**视频目标分割 (mIoU)**:

| 方法 | 整体mIoU | 杠内 | 杠外 |
|------|---------|------|------|
| YOLO11 (pretrained) | 0.784 | 0.831 | 0.774 |
| Grounded SAM2 ("a person") | 0.358 | 0.643 | 0.559 |
| Grounded SAM2 ("amputee") | 0.905 | 0.900 | 0.907 |
| **YOLO11-ProGait** | **0.847** | **0.815** | **0.866** |

**2D人体姿态估计 (AP@[.5,.95])**:

| 方法 | 整体AP | 杠内 | 杠外 |
|------|-------|------|------|
| HRNet | 0.750 | 0.825 | 0.733 |
| ViTPose | 0.830 | 0.845 | 0.822 |
| RTMPose | 0.855 | 0.876 | 0.850 |
| **RTMPose-ProGait** | **0.947** | **0.968** | **0.942** |

### 消融实验 (步态分类)

| 输入配置 | Top-1 Acc | Balanced Acc |
|---------|----------|-------------|
| 正面视角 | 0.510 | 0.545 |
| **矢状面视角** | **0.826** | **0.790** |
| 杠内 | 0.364 | 0.437 |
| 杠外 | 0.486 | 0.320 |
| 全部23关键点 | 0.372 | 0.403 |
| 仅12下肢关键点 | 0.384 | 0.413 |

矢状面视角效果远优于正面，但混合两种视角反而下降。下肢关键点足以完成步态分类任务。

### 关键发现

- 微调后的RTMPose-ProGait在下肢关键点AP上提升约29.3%（0.625→0.918 vs HRNet），证明假肢专用数据集对改善检测的显著效果
- Grounded SAM2高度依赖prompt文本：用"a person"只有0.358 mIoU，用正确prompt可达0.905，但仍会偶尔丢失假肢追踪
- 简单LSTM分类器在矢状面步态分类上达82.6%准确率，与GaitGraph2等专门方法相比也具竞争力
- 多种步态识别方法（GaitGraph2/GPGait/GaitBase）通过简单微调即可适配ProGait数据集

## 亮点与洞察

- **填补数据空白**: 首个面向假肢用户的多用途视觉数据集，具有明确的临床价值
- **半自动标注流水线高效**: 两阶段迭代微调策略将HPE人工标注量降至25%以下
- **临床洞察**: 矢状面优于正面、下肢关键点足够等发现与临床步态分析实践一致
- **IRB审批+隐私保护**: 完整的伦理流程使数据集具有更高的学术可信度

## 局限与展望

- 仅4位受试者，样本量受限（高度专业化的弱势群体，招募和测试成本高）
- 步态类别分布不均衡（最多41样本，最少4样本），影响分类泛化性
- 未支持3D姿态估计任务
- 视频长度差异大（2-40秒），可能影响模型一致性
- 步态分类仅考虑主要偏差，临床中常存在多种偏差共现

## 相关工作与启发

- GAVD、Health&Gait等现有步态数据集缺少假肢用户表征
- SAM2的零样本追踪能力impressive但对prompt敏感，专用微调仍有价值
- 未来方向：结合LLM利用文本描述实现自动步态评估和假肢调整建议

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个假肢步态视频数据集，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 三项任务基准完整，多种对比方法
- 写作质量: ⭐⭐⭐⭐ 结构清晰，临床背景介绍到位
- 价值: ⭐⭐⭐⭐ 对假肢用户辅助技术研究有直接推动作用，但数据规模有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](../../NeurIPS2025/medical_imaging/ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](../../CVPR2025/medical_imaging/interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](../../NeurIPS2025/medical_imaging/care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[CVPR 2025\] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](../../CVPR2025/medical_imaging/cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)

</div>

<!-- RELATED:END -->
