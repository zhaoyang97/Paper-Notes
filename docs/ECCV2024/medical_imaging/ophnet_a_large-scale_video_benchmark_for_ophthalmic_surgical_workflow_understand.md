---
title: >-
  [论文解读] OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding
description: >-
  [ECCV 2024][医学图像][手术工作流理解] 构建了OphNet——目前最大规模的眼科手术视频基准数据集（2278个视频、285小时、66种手术类型、102种手术阶段、150种精细操作），支持手术类型识别、阶段识别、时序定位和阶段预测四大任务，其规模约为现有最大手术工作流分析基准的20倍。 手术场景的视频理解对于推进…
tags:
  - "ECCV 2024"
  - "医学图像"
  - "手术工作流理解"
  - "眼科手术"
  - "视频基准"
  - "动作识别"
  - "时序定位"
---

# OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding

**会议**: ECCV 2024  
**arXiv**: [2406.07471](https://arxiv.org/abs/2406.07471)  
**代码**: [https://minghu0830.github.io/OphNet-benchmark/](https://minghu0830.github.io/OphNet-benchmark/)  
**领域**: 医学图像  
**关键词**: 手术工作流理解, 眼科手术, 视频基准, 动作识别, 时序定位

## 一句话总结
构建了OphNet——目前最大规模的眼科手术视频基准数据集（2278个视频、285小时、66种手术类型、102种手术阶段、150种精细操作），支持手术类型识别、阶段识别、时序定位和阶段预测四大任务，其规模约为现有最大手术工作流分析基准的20倍。

## 研究背景与动机
手术场景的视频理解对于推进机器人手术、远程手术和AI辅助手术至关重要，特别是在眼科领域。然而，现有数据集存在五个关键缺陷：(1) **规模小**：大多数不超过100个视频（如CATARACTS仅50个、CatRelDet仅21个）；(2) **手术和阶段类别有限**：几乎所有眼科数据集仅包含白内障手术，阶段类别少（CatRelDet仅4个阶段）；(3) **标注粒度粗**：粗粒度定义导致同一操作在不同阶段被归入不同类别，产生标注偏差；(4) **缺乏层次化标注**：忽视手术-阶段-操作的层次关系和连续性；(5) **域单一**：精心采集的视频风格统一，不利于测试域泛化能力。

核心矛盾：深度学习技术对大规模数据的需求与手术视频数据集的稀缺性之间的巨大鸿沟。研究表明，I3D模型需要100个以上视频训练才能达到80%准确率，超过700个视频后性能才持续提升。

本文的切入角度：**利用YouTube作为数据源规避隐私问题，构建覆盖白内障、青光眼和角膜手术三大类66种手术的大规模、多层次标注基准**。核心idea是通过专业眼科医生团队进行精细的层次化标注（手术→阶段→操作），同时提供分类和时序定位两类任务。

## 方法详解

### 整体框架
OphNet的构建分为三个阶段：(1) 数据收集与预处理——从YouTube搜索和筛选眼科手术视频；(2) 数据标注——分类标注和层次化定位标注；(3) 评估基准——在四个任务上建立baseline。整体pipeline围绕数据集构建而非新方法提出。

### 关键设计
1. **数据收集与质量控制**: 

    - 利用文本搜索算法在YouTube上检索手术关键词，并扩展同义词和缩写（如PHACO=超声乳化术，IOL=人工晶体植入术，ECCE=囊外摘除术）
    - 五名专业人员初筛+六名主治眼科医生复审
    - 过滤标准：排除低分辨率、黑白、动画演示、非人眼（猪眼/兔眼/假眼）视频
    - 设计动机：YouTube视频规避了医疗数据隐私问题，同时保证了视频风格的多样性

2. **多层次分类与定位标注**: 

    - **分类标注**：不同于自然视频的单标签分类，眼科手术常一台手术同时处理多种眼病（如白内障+青光眼），因此支持多标签
    - 三层标注结构：主要手术类别（仅一种）→ 次要手术类型（可多种）→ 阶段→ 操作
    - **定位标注**：523个视频被选出进行精细时序标注，每个视频平均22个操作标注
    - 采用完全链接算法(complete linkage)将多名标注者的时序边界聚合为稳定一致的边界
    - 设计动机：眼睛作为复杂器官，多种疾病常共存，需要多标签和层次化标注来准确反映真实临床场景

3. **四大评估任务设计**: 

    - **手术类型识别**：未裁剪视频中弱监督识别主要手术类型（66类）
    - **阶段/操作识别**：裁剪片段中分类识别各阶段（102类）和操作（150类）
    - **阶段定位**：在未裁剪视频中定位各阶段的起止时间
    - **阶段预测**：在仅观察部分视频的情况下预测当前阶段
    - 设计动机：覆盖手术工作流理解的全方位需求，从粗到细、从识别到预测

### 损失函数 / 训练策略
作为基准数据集论文，主要评估现有模型：
- 分类任务：使用I3D、SlowFast、X3D、MViT V2（随机初始化和K400预训练两种）；引入X-CLIP和ViFi-CLIP等CLIP-based模型
- 定位任务：使用ActionFormer和TriDet，backbone为CSN/SwinViviT/SlowFast
- 预测任务：不同观察比例下的Top-1准确率评估
- 数据划分：70%训练/10%验证/20%测试

## 实验关键数据

### 主实验
手术类型识别（All类 Top-1/Top-5 准确率%）：

| 方法 | Backbone | Top-1 | Top-5 |
|------|----------|-------|-------|
| I3D | - | 29.8 | 53.2 |
| SlowFast | - | 27.2 | 54.4 |
| X3D | - | 28.5 | 62.7 |
| MViT V2 | - | 29.1 | 60.1 |
| X-CLIP₃₂ | ViT-B/16 | **58.9** | 81.0 |
| ViFi-CLIP₁₆ | ViT-B/16 | 58.9 | **79.8** |

阶段定位(mAP %)：

| 方法 | Backbone | IoU=0.1 | IoU=0.3 | IoU=0.5 | IoU=0.7 | Avg. |
|------|----------|---------|---------|---------|---------|------|
| ActionFormer | SwinViviT | 59.3 | 54.7 | 43.3 | 26.3 | 46.4 |
| ActionFormer | SlowFast | 60.0 | 55.9 | 45.1 | 26.0 | 47.5 |
| TriDet | SwinViviT | **61.0** | **57.1** | **47.1** | **33.1** | **50.4** |
| TriDet | SlowFast | 61.3 | 56.0 | 45.6 | 30.4 | 48.6 |

### 消融实验
阶段/操作分类中不同输入帧数和预训练的影响（ViFi-CLIP模型）：

| 模型配置 | 阶段Top-1 (All) | 操作Top-1 (All) |
|---------|----------------|----------------|
| ViFi-CLIP₁₆ | 66.1 | 65.0 |
| ViFi-CLIP₃₂ | **68.4** | 64.8 |
| X-CLIP₁₆ | 63.4 | 62.5 |
| X-CLIP₃₂ | 62.7 | 62.0 |

阶段预测（不同观察比例下Top-1准确率%）：

| 方法 | ratio=0.1 | ratio=0.3 | ratio=0.5 | ratio=0.7 | Avg. |
|------|-----------|-----------|-----------|-----------|------|
| MViT V2 | 25.6 | 43.7 | 49.3 | 52.3 | 47.5 |
| MViT V2* (K400) | **27.8** | **43.8** | **50.5** | 51.7 | **48.2** |
| SlowFast* | 27.5 | 43.2 | 49.9 | **52.3** | 47.8 |

### 关键发现
- **CLIP-based模型显著优于传统视频模型**：X-CLIP的58.9% vs I3D的29.8%，几乎翻倍，说明语言-视觉预训练对手术视频理解有巨大帮助
- **ViFi-CLIP在阶段和操作分类中表现最佳**，尤其在白内障手术中Top-1达75.9%
- **Kinetics 400预训练普遍有帮助**，但不如CLIP预训练
- TriDet+SwinViviT在定位任务中表现最优，在高IoU阈值下优势更明显
- 更多输入帧数通常有正面影响，但增益不一致
- 整体准确率仍然较低（手术识别Top-1仅~59%），说明数据集确实challenging
- 注意力可视化显示模型正确聚焦于手术器械和眼部区域

## 亮点与洞察
- **规模与多样性上的巨大突破**: 2278个视频、285小时、66种手术、150种操作——远超此前所有手术工作流数据集
- **层次化标注设计合理**: 手术→阶段→操作的三层标注符合真实临床场景需求，一个视频可能包含多种手术
- **利用YouTube规避隐私**: 既解决了医疗数据隐私问题，又带来了风格多样性以测试域泛化
- **15名专业眼科医生参与标注**: 保证了标注的专业性和准确性
- **四个任务全面覆盖**: 从识别到定位到预测，为手术工作流理解提供完整的评估体系
- 完全链接算法聚合多人标注边界是一个实用的标注一致性解决方案

## 局限与展望
- YouTube来源视频质量参差不齐，部分视频可能存在标注噪声
- 手术识别Top-1准确率仅59%，说明需要更强的模型或更好的表征学习
- 定位标注仅覆盖523个视频（约23%），大部分视频仅有分类标签
- 未提供器械和解剖结构的像素级分割标注，限制了多任务学习的可能
- 标签类别严重不平衡——某些罕见手术类型样本极少
- 未来可探索弱监督/半监督方法来利用大量未标注视频

## 相关工作与启发
- 与Cholec80/CholecT50等内窥镜手术数据集互补，扩展了手术视频理解到眼科微镜领域
- CLIP-based模型在手术视频分类中的优势，暗示医学视频领域可以更多借鉴视觉-语言预训练
- 层次化标注设计可推广到其他复杂多步骤流程的视频理解任务
- 对手术教育培训和AI辅助手术系统的开发有直接推动作用
- 大规模标注需要专家团队的投入，数据集价值体现在其不可替代的专业性

## 评分
- 新颖性: ⭐⭐⭐ (数据集贡献为主，方法论创新有限)
- 实验充分度: ⭐⭐⭐⭐⭐ (四个任务全面评估，多种baseline对比)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，统计详实)
- 价值: ⭐⭐⭐⭐⭐ (填补了眼科手术视频理解的重要数据空白，规模领先)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[CVPR 2026\] X-PCR: A Benchmark for Cross-modality Progressive Clinical Reasoning in Ophthalmic Diagnosis](../../CVPR2026/medical_imaging/x-pcr_a_benchmark_for_cross-modality_progressive_clinical_reasoning_in_ophthalmi.md)
- [\[NeurIPS 2025\] THUNDER: Tile-level Histopathology image UNDERstanding benchmark](../../NeurIPS2025/medical_imaging/thunder_tile-level_histopathology_image_understanding_benchmark.md)
- [\[CVPR 2026\] Efficient Unrolled Networks for Large-Scale 3D Inverse Problems](../../CVPR2026/medical_imaging/efficient_unrolled_networks_for_large-scale_3d_inverse_problems.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)

</div>

<!-- RELATED:END -->
